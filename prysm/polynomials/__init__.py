"""Various polynomials of optics."""

from prysm.mathops import np
from prysm.coordinates import optimize_xy_separable

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


def separable_2d_sequence(ns, ms, x, y, fx, fy=None, greedy=True):
    """Sequence of separable (x,y) orthogonal polynomials.

    Parameters
    ----------
    ns : Iterable of int
        sequence of orders to evaluate in the X dimension
    ms : Iterable of int
        sequence of orders to evaluate in the Y dimension
    x : numpy.ndarray
        array of shape (m, n) or (n,) containing the X points
    y : numpy.ndarray
        array of shape (m, n) or (m,) containing the Y points
    fx : callable
        function which returns a generator or other sequence
        of modes, given args (ns, x)
    fy : callable, optional
        function which returns a generator or other sequence
        of modes, given args (ns, x);
        y equivalent of fx, fx is used if None
    greedy : bool, optional
        if True, consumes any generators returned by fx or fy and
        returns lists.

    Returns
    -------
    Iterable, Iterable
        sequence of x modes (1D) and y modes (1D)

    """
    if fy is None:
        fy = fx

    x, y = optimize_xy_separable(x, y)
    modes_x = fx(ns, x)
    modes_y = fy(ms, y)
    if greedy:
        modes_x = list(modes_x)
        modes_y = list(modes_y)

    return modes_x, modes_y


def mode_1d_to_2d(mode, x, y, which='x'):
    """Expand a 1D representation of a mode to 2D.

    Notes
    -----
    You likely only want to use this function for plotting or similar, it is
    much faster to use sum_of_xy_modes to produce 2D surfaces described by
    a sum of modes which are separable in x and y.

    Parameters
    ----------
    mode : numpy.ndarray
        mode, representing a separable mode in X, Y along {which} axis
    x : numpy.ndarray
        x dimension, either 1D or 2D
    y : numpy.ndarray
        y dimension, either 1D or 2D
    which : str, {'x', 'y'}
        which dimension the mode is produced along

    Returns
    -------
    numpy.ndarray
        2D version of the mode

    """
    x, y = optimize_xy_separable(x, y)
    out = np.broadcast_to(mode, (y.size, x.size))
    return out


def sum_of_xy_modes(modesx, modesy, x, y, weightsx=None, weightsy=None):
    """Weighted sum of separable x and y modes projected over the 2D aperture.

    Parameters
    ----------
    modesx : iterable
        sequence of x modes
    modesy : iterable
        sequence of y modes
    x : numpy.ndarray
        x points
    y : numpy.ndarray
        y points
    weightsx : iterable, optional
        weights to apply to modesx.  If None, [1]*len(modesx)
    weightsy : iterable, optional
        weights to apply to modesy.  If None, [1]*len(modesy)

    Returns
    -------
    numpy.ndarray
        modes summed over the 2D aperture

    """
    x, y = optimize_xy_separable(x, y)

    if weightsx is None:
        weightsx = [1]*len(modesx)
    if weightsy is None:
        weightsy = [1]*len(modesy)

    # apply the weights to the modes
    modesx = [m*w for m, w in zip(modesx, weightsx)]
    modesy = [m*w for m, w in zip(modesy, weightsy)]

    # sum the separable bases in 1D
    sum_x = np.zeros_like(x)
    sum_y = np.zeros_like(y)
    for m in modesx:
        sum_x += m
    for m in modesy:
        sum_y += m

    # broadcast to 2D and return
    shape = (y.size, x.size)
    sum_x = np.broadcast_to(sum_x, shape)
    sum_y = np.broadcast_to(sum_y, shape)
    return sum_x + sum_y


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
    modes = np.asarray(modes)
    weights = np.asarray(weights).astype(modes.dtype)

    # dot product of the 0th dim of modes and weights => weighted sum
    return np.tensordot(modes, weights, axes=(0, 0))


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
        modes to fit; sequence of ndarray of shape (m, n)
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
