"""Functions to compute Zernike polynomials."""

from collections import defaultdict

from prysm.mathops import np, kronecker
from prysm.jacobi import jacobi, jacobi_sequence

# the functions in this module that compute Zernike polynomials (zernike_nm, _sequence) use the relation between the
# Zernike and Jacobi polynomials to accelerate the computation and stabilize it to high order, removing catastrophic
# roundoff error


def zernike_norm(n, m):
    """Norm of a Zernike polynomial with n, m indexing."""
    return np.sqrt((2 * (n + 1)) / (1 + kronecker(m, 0)))


def zero_separation(n):
    """Zero separation in normalized r based on radial order n."""
    return 1 / n ** 2


def zernike_nm(n, m, r, t, norm=True):
    """Zernike polynomial of radial order n, azimuthal order m at point r, t.

    Parameters
    ----------
    n : `int`
        radial order
    m : `int`
        azimuthal order
    r : `numpy.ndarray`
        radial coordinates
    t : `numpy.ndarray`
        azimuthal coordinates
    norm : `bool`, optional
        if True, orthonormalize the result (unit RMS)
        else leave orthogonal (zero-to-peak = 1)

    """
    x = 2 * r ** 2 - 1
    am = abs(m)
    n_j = (n - am) // 2
    out = jacobi(n_j, 0, am, x)
    if m != 0:
        if m < 0:
            out *= (r ** am * np.sin(m*t))
        else:
            out *= (r ** am * np.cos(m*t))

    if norm:
        out *= zernike_norm(n, m)

    return out


def zernike_nm_sequence(nms, r, t, norm=True):
    """Zernike polynomial of radial order n, azimuthal order m at point r, t.

    Parameters
    ----------
    nms : iterable of tuple of int,
        sequence of (n, m); looks like [(1,1), (3,1), ...]
    r : `numpy.ndarray`
        radial coordinates
    t : `numpy.ndarray`
        azimuthal coordinates
    norm : `bool`, optional
        if True, orthonormalize the result (unit RMS)
        else leave orthogonal (zero-to-peak = 1)

    """
    # this function deduplicates all possible work.  It uses a connection
    # to the jacobi polynomials to efficiently compute a series of zernike
    # polynomials
    # it follows this basic algorithm:
    # for each (n, m) compute the appropriate Jacobi polynomial order
    # collate the unique values of that for each |m|
    # compute a set of jacobi polynomials for each |m|
    # compute r^|m| , sin(|m|*t), and cos(|m|*t for each |m|
    #
    # benchmarked at 12.26 ns/element (256x256), 4.6GHz CPU = 56 clocks per element
    # ~36% faster than previous impl (12ms => 8.84 ms)
    x = 2 * r ** 2 - 1
    ms = list(e[1] for e in nms)
    am = np.abs(ms)
    amu = np.unique(am)

    def factory():
        return 0

    jacobi_sequences_mjn = defaultdict(factory)
    # jacobi_sequences_mjn is a lookup table from |m| to all orders < max(n_j)
    # for each |m|, i.e. 0 .. n_j_max
    for nm, am_ in zip(nms, am):
        n = nm[0]
        nj = (n-am_) // 2
        if nj > jacobi_sequences_mjn[am_]:
            jacobi_sequences_mjn[am_] = nj

    for k in jacobi_sequences_mjn:
        nj = jacobi_sequences_mjn[k]
        jacobi_sequences_mjn[k] = np.arange(nj+1)

    jacobi_sequences = {}

    jacobi_sequences_mjn = dict(jacobi_sequences_mjn)
    for k in jacobi_sequences_mjn:
        n_jac = jacobi_sequences_mjn[k]
        jacobi_sequences[k] = list(jacobi_sequence(n_jac, 0, k, x))

    powers_of_m = {}
    sines = {}
    cosines = {}
    for m in amu:
        powers_of_m[m] = r ** m
        sines[m] = np.sin(m*t)
        cosines[m] = np.cos(m*t)

    for n, m in nms:
        absm = abs(m)
        nj = (n-absm) // 2
        jac = jacobi_sequences[absm][nj]
        if norm:
            jac = jac * zernike_norm(n, m)

        if m == 0:
            # rotationally symmetric Zernikes are jacobi
            yield jac
        else:
            if m < 0:
                azpiece = sines[absm]
            else:
                azpiece = cosines[absm]

            radialpiece = powers_of_m[absm]
            out = jac * azpiece * radialpiece  # jac already contains the norm
            yield out


def zernike_fit(data, x=None, y=None,
               rho=None, phi=None, terms=16,
               norm=False, residual=False,
               round_at=6, map_='Fringe'):
    """Fits a number of Zernike coefficients to provided data.

    Works by minimizing the mean square error  between each coefficient and the
    given data.  The data should be uniformly sampled in an x,y grid.

    Parameters
    ----------
    data : `numpy.ndarray`
        data to fit to.
    x : `numpy.ndarray`, optional
        x coordinates, same shape as data
    y : `numpy.ndarray`, optional
        y coordinates, same shape as data
    rho : `numpy.ndarray`, optional
        radial coordinates, same shape as data
    phi : `numpy.ndarray`, optional
        azimuthal coordinates, same shape as data
    terms : `int` or iterable, optional
        if an int, number of terms to fit,
        otherwise, specific terms to fit.
        If an iterable of ints, members of the single index set map_,
        else interpreted as (n,m) terms, in which case both m+ and m- must be given.
    norm : `bool`, optional
        if True, normalize coefficients to unit RMS value
    residual : `bool`, optional
        if True, return a tuple of (coefficients, residual)
    round_at : `int`, optional
        decimal place to round values at.
    map_ : `str`, optional, {'Fringe', 'Noll', 'ANSI'}
        which ordering of Zernikes to use

    Returns
    -------
    coefficients : `numpy.ndarray`
        an array of coefficients matching the input data.
    residual : `float`
        RMS error between the input data and the fit.

    Raises
    ------
    ValueError
        too many terms requested.

    """
    data = data.T  # transpose to mimic transpose of zernikes

    # precompute the valid indexes in the original data
    pts = np.isfinite(data)

    # set up an x/y rho/phi grid to evaluate Zernikes on
    if x is None and rho is None:
        rho, phi = make_rho_phi_grid(*reversed(data.shape))
        rho = rho[pts].flatten()
        phi = phi[pts].flatten()
    elif rho is None:
        rho, phi = cart_to_polar(x, y)
        rho, phi = rho[pts].flatten(), phi[pts].flatten()

    # convert indices to (n,m)
    if isinstance(terms, int):
        # case 1, number of terms
        nms = [nm_funcs[map_](i+1) for i in range(terms)]
    elif isinstance(terms[0], int):
        nms = [nm_funcs[map_](i) for i in terms]
    else:
        nms = terms

    # compute each Zernike term
    zerns_raw = []
    for (n, m) in nms:
        zern = zcachemn.grid_bypass(n, m, norm, rho, phi)
        zerns_raw.append(zern)

    zcachemn.grid_bypass_cleanup(rho, phi)
    zerns = np.asarray(zerns_raw).T

    # use least squares to compute the coefficients
    meas_pts = data[pts].flatten()
    coefs = np.linalg.lstsq(zerns, meas_pts, rcond=None)[0]
    if round_at is not None:
        coefs = coefs.round(round_at)

    if residual is True:
        components = []
        for zern, coef in zip(zerns_raw, coefs):
            components.append(coef * zern)

        _fit = np.asarray(components)
        _fit = _fit.sum(axis=0)
        rmserr = rms(data[pts].flatten() - _fit)
        return coefs, rmserr
    else:
        return coefs
