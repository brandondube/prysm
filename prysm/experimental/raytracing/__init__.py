"""Library routines for raytracing (for now)."""
from prysm import polynomials
from prysm.mathops import np


def find_zero_indices_2d(x, y, tol=1e-8):
    """Find the (y,x) indices into x and y where x==y==0."""
    # assuming we're FFT-centered, we will never do the ifs
    # this probably blows up if zero is not in the array
    lookup = tuple(s//2 for s in x.shape)
    x0 = x[lookup]
    if x0 > tol:
        lookup2 = (lookup[0], lookup[1]+1)
        x1 = x[lookup2]
        dx = x1-x0
        shift_samples = (x0 / dx)
        lookup = (lookup[0], lookup[1]+shift_samples)
    y0 = y[lookup]
    if y0 > tol:
        lookup2 = (lookup[0]+1, lookup[1])
        y1 = x[lookup2]
        dy = y1-y0
        shift_samples = (y0 / dy)
        lookup = (lookup[0]+shift_samples, lookup[1])

    return lookup


def fix_zero_singularity(arr, x, y, fill='xypoly', order=2):
    """Fix a singularity at the origin of arr by polynomial interpolation.

    Parameters
    ----------
    arr : numpy.ndarray
        array of dimension 2 to modify at the origin (x==y==0)
    x : numpy.ndarray
        array of dimension 2 of X coordinates
    y : numpy.ndarray
        array of dimension 2 of Y coordinates
    fill : str, optional, {'xypoly'}
        how to fill
    order : int
        polynomial order to fit

    Returns
    -------
    numpy.ndarray
        arr (modified in-place)

    """
    zloc = find_zero_indices_2d(x, y)
    min_y = zloc[0]-order
    max_y = zloc[0]+order+1
    min_x = zloc[1]-order
    max_x = zloc[1]+order+1
    # newaxis schenanigans to get broadcasting right without
    # meshgrid
    ypts = np.arange(min_y, max_y)[:, np.newaxis]
    xpts = np.arange(min_x, max_x)[np.newaxis, :]
    window = arr[ypts, xpts].copy()
    c = [s//2 for s in window.shape]
    window[c] = np.nan
    # no longer need xpts, ypts
    # really don't care about fp64 vs fp32 (very small arrays)
    xpts = xpts.astype(float)
    ypts = ypts.astype(float)
    # use Hermite polynomials as
    # XY polynomial-like basis orthogonal
    # over the infinite plane
    # H0 = 0
    # H1 = x
    # H2 = x^2 - 1, and so on
    ns = np.arange(order+1)
    xbasis = polynomials.hermite_He_sequence(ns, xpts)
    ybasis = polynomials.hermite_He_sequence(ns, ypts)
    xbasis = [polynomials.mode_1d_to_2d(mode, xpts, ypts, 'x') for mode in xbasis]
    ybasis = [polynomials.mode_1d_to_2d(mode, xpts, ypts, 'y') for mode in ybasis]
    basis_set = np.asarray([*xbasis, *ybasis])
    coefs = polynomials.lstsq(basis_set, window)
    projected = np.dot(basis_set[:, c[0], c[1]], coefs)
    arr[zloc] = projected
    return arr
