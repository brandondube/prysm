"""Various polynomials of optics."""
import warnings

from prysm.mathops import np

from .jacobi import (  # NOQA
    jacobi,
    jacobi_sequence,
    jacobi_der,
    jacobi_der_sequence,
    jacobi_sum_clenshaw,
    jacobi_sum_clenshaw_der
)
from .cheby import (  # NOQA
    cheby1,
    cheby1_sequence,
    cheby1_der,
    cheby1_der_sequence,
    cheby2,
    cheby2_sequence,
    cheby2_der,
    cheby2_der_sequence,
    cheby3,
    cheby3_sequence,
    cheby3_der,
    cheby3_der_sequence,
    cheby4,
    cheby4_sequence,
    cheby4_der,
    cheby4_der_sequence,
)
from .legendre import (  # NOQA
    legendre,
    legendre_sequence,
    legendre_der,
    legendre_der_sequence,
)  # NOQA
from .hermite import (  # NOQA
    hermite_He,
    hermite_He_sequence,
    hermite_He_der,
    hermite_He_der_sequence,
    hermite_H,
    hermite_H_sequence,
    hermite_H_der,
    hermite_H_der_sequence,
)
from .qpoly import (  # NOQA
    Qbfs,
    Qbfs_sequence,
    Qcon,
    Qcon_sequence,
    Q2d,
    Q2d_sequence,
)
from .dickson import (  # NOQA
    dickson1,
    dickson1_sequence,
    dickson2,
    dickson2_sequence
)
from .xy import (  # NOQA
    j_to_xy,
    xy_polynomial,
    xy_polynomial_sequence,
)

from .zernike import (  # NOQA
    zernike_norm,
    zernike_nm,
    zernike_nm_sequence,
    zernike_nm_der,
    zernike_nm_der_sequence,
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
        sequence of ndarray of shape (k, m, n);
        a list of length k with elements of shape (m,n) works
    weights : numpy.ndarray
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
        sequence of ndarray of shape (k, m, n);
        a list of length k with elements of shape (m,n) works
    databar : numpy.ndarray
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
    r : numpy.ndarray
        radial pupil coordinate
    t : numpy.ndarray
        azimuthal pupil coordinate
    H : numpy.ndarray
        field coordinate

    Returns
    -------
    numpy.ndarray
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
        modes to fit; sequence of ndarray of shape (m, n);
        array of shape (k, m, n), k=num modes, (m,n) = spatial domain is best
    data : numpy.ndarray
        data to fit, of shape (m, n)
        place NaN values in data for points to ignore

    Returns
    -------
    numpy.ndarray
        fit coefficients

    """
    mask = np.isfinite(data)
    data = data[mask]
    modes = np.asarray(modes)
    modes = modes.reshape((modes.shape[0], -1))  # flatten second dim
    modes = modes[:, mask.ravel()].T  # transpose moves modes to columns, as needed for least squares fit
    c, *_ = np.linalg.lstsq(modes, data, rcond=None)
    return c


def orthonormalize(modes, mask):
    """Orthonormalize modes over the domain of mask.

    Parameters
    ----------
    modes : iterable
        """
    pass
