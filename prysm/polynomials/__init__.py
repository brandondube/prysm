"""Various polynomials of optics."""
import warnings

from prysm.mathops import np

from .jacobi import (  # NOQA
    jacobi,
    jacobi_seq,
    jacobi_der,
    jacobi_der_seq,
    jacobi_sum_clenshaw,
    jacobi_sum_clenshaw_der
)
from .cheby import (  # NOQA
    cheby1,
    cheby1_seq,
    cheby1_der,
    cheby1_der_seq,
    cheby2,
    cheby2_seq,
    cheby2_der,
    cheby2_der_seq,
    cheby3,
    cheby3_seq,
    cheby3_der,
    cheby3_der_seq,
    cheby4,
    cheby4_seq,
    cheby4_der,
    cheby4_der_seq,
)
from .legendre import (  # NOQA
    legendre,
    legendre_seq,
    legendre_der,
    legendre_der_seq,
)  # NOQA
from .hermite import (  # NOQA
    hermite_He,
    hermite_He_seq,
    hermite_He_der,
    hermite_He_der_seq,
    hermite_H,
    hermite_H_seq,
    hermite_H_der,
    hermite_H_der_seq,
)
from .qpoly import (  # NOQA
    Qbfs,
    Qbfs_seq,
    Qcon,
    Qcon_seq,
    Q2d,
    Q2d_seq,
)
from .dickson import (  # NOQA
    dickson1,
    dickson1_seq,
    dickson2,
    dickson2_seq
)
from .xy import (  # NOQA
    j_to_mn,
    xy,
    xy_seq,
)

from .zernike import (  # NOQA
    zernike_norm,
    zernike_nm,
    zernike_nm_seq,
    zernike_nm_der,
    zernike_nm_der_seq,
    zernikes_to_magnitude_angle,
    zernikes_to_magnitude_angle_nmkey,
    zero_separation as zernike_zero_separation,
    ansi_j_to_nm,
    nm_to_ansi_j,
    nm_to_fringe,
    nm_to_name,
    noll_to_nm,
    fringe_to_nm,
    barplot as zernike_barplot,
    barplot_magnitudes as zernike_barplot_magnitudes,
    top_n,
)


def sum_of_2d_modes(modes, weights):
    """Compute a sum of 2D modes.

    Parameters
    ----------
    modes : iterable
        seq of ndarray of shape (k, m, n);
        a list of length k with elements of shape (m,n) works
    weights : ndarray
        weight of each mode

    Returns
    -------
    numpy.ndarry
        ndarray of shape (m, n) that is the sum of modes as given

    """
    if isinstance(modes, (list, tuple)):
        warnings.warn('sum_of_2d_modes: modes is a list or tuple: for optimal performance, pre convert to array of shape (k, m, n)')
        modes = np.asarray(modes)

    if isinstance(weights, (list, tuple)):
        warnings.warn('sum_of_2d_modes weights is a list or tuple: for optimal performance, pre convert to array of shape (k,)')
        weights = np.asarray(weights)

    if weights.dtype != modes.dtype:
        warnings.warn("sum_of_2d_modes weights dtype mismatched to modes dtype, converting weights to modes' dtype: use same dtype for optimal speed")
        weights = weights.astype(modes.dtype)

    # dot product of the 0th dim of modes and weights => weighted sum
    return np.tensordot(modes, weights, axes=(0, 0))


def sum_of_2d_modes_backprop(modes, databar):
    """Gradient backpropagation through sum_of_2d_modes.

    Parameters
    ----------
    modes : iterable
        seq of ndarray of shape (k, m, n);
        a list of length k with elements of shape (m,n) works
    databar : ndarray
        partial gradient backpropated up to the return of sum_of_2d_modes

    Returns
    -------
    numpy.ndarry
        cumulative gradient through to the weights vector given to sum_of_2d_modes

    """
    return np.tensordot(modes, databar)


def hopkins(a, b, c, r, t, H):
    """Hopkins' aberration expansion.

    This function uses the "W020" or "W131" like notation, with Wabc separating
    into the a, b, c arguments.  To produce a sine term instead of cosine,
    make a the negative of the order.  In other words, for W222S you would use
    hopkins(2, 2, 2, ...) and for W222T you would use
    hopkins(-2, 2, 2, ...).

    Parameters
    ----------
    a : int
        azimuthal order
    b : int
        radial order
    c : int
        order in field ("H-order")
    r : ndarray
        radial pupil coordinate
    t : ndarray
        azimuthal pupil coordinate
    H : ndarray
        field coordinate

    Returns
    -------
    ndarray
        polynomial evaluated at this point

    """
    # c = "component"
    if a < 0:
        c1 = np.sin(abs(a)*t)
    else:
        c1 = np.cos(a*t)

    c2 = r ** b

    c3 = H ** c

    return c1 * c2 * c3


def lstsq(modes, data):
    """Least-Squares fit of modes to data.

    Parameters
    ----------
    modes : iterable
        modes to fit; seq of ndarray of shape (m, n);
        array of shape (k, m, n), k=num modes, (m,n) = spatial domain is best
    data : ndarray
        data to fit, of shape (m, n)
        place NaN values in data for points to ignore

    Returns
    -------
    ndarray
        fit coefficients

    """
    mask = np.isfinite(data)
    data = data[mask]
    modes = np.asarray(modes)
    modes = modes.reshape((modes.shape[0], -1))  # flatten second dim
    # transpose moves modes to columns, as needed for least squares fit
    modes = modes[:, mask.ravel()].T
    c, *_ = np.linalg.lstsq(modes, data, rcond=None)
    return c


def normalize_modes(modes, mask, to='std', remove_piston=True):
    """Scale modes such that they have unit RMS.

    Parameters
    ----------
    modes : ndarray
        mode shape (m, n) or modes shape (k, m, n) to scale
    mask : ndarray
        2D boolean array, True in the interior of the appropriate domain
    to : str
        what to normalize modes by, use std for "RMS" or ptp for PV

    Returns
    -------
    ndarray
        scaled modes

    """
    # first sweep, take out piston
    func = getattr(np, to)
    if modes.ndim == 2:
        mode = modes[:, mask]
        if remove_piston:
            mode = mode - np.mean(mode, axis=1)[:, np.newaxis]

        norm = func(mode)

        # loophole for piston
        if norm < 1e-9:
            norm = 1.

        return modes * (1/norm)

    modes_masked = modes[:, mask]
    if remove_piston:
        mean = np.mean(modes_masked, axis=1)
        modes_masked = modes_masked - mean[:, np.newaxis]

    norms = func(modes_masked, axis=1)
    norms[norms < 1e-9] = 1.  # loophole for piston
    # newaxes for correct numpy broadcast semantics
    out = np.zeros_like(modes)
    out[:, mask] = modes_masked * (1/norms[:, np.newaxis])
    return out


def orthogonalize_modes(modes, mask):
    """Use a Gram-Schmidt like process to orthogonalize modes over mask.

    Parameters
    ----------
    modes : ndarray
        array of shape (k, m, n) to scale
    mask : ndarray
        2D boolean array, True in the interior of the appropriate domain

    Returns
    -------
    ndarray
        orthogonal modes

    """
    basis = modes[:, mask]
    Q, R = np.linalg.qr(basis.T)
    sgn = np.sign(np.diag(R))  # modes can flip, undo that
    Qmod = Q*sgn
    out = np.zeros_like(modes)
    out[:, mask] = Qmod.T
    return out
