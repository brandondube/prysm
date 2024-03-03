"""Routines for working with optical fibers."""
import numpy as truenp

from scipy.optimize import brentq


from prysm.mathops import np, special


cutoff = np.pi


def critical_angle(n_core, n_clad, deg=True):
    """Angle at which TIR happens in a step index fiber.

    Parameters
    ----------
    n_core : float
        core index
    n_clad : float
        cladding index
    deg : bool, optional
        if True, return is in degrees
        else radians

    Returns
    -------
    float
        TIR angle

    """
    ang = np.arcsin(n_clad / n_core)
    if deg:
        return np.degrees(ang)
    return ang


def numerical_aperture(n_core, n_clad):
    """NA of a step-index fiber.

    Parameters
    ----------
    n_core : float
        core index
    n_clad : float
        cladding index

    Returns
    -------
    float
        numerical aperture

    """
    return np.sqrt(n_core*n_core - n_clad*n_clad)


def V(radius, NA, wavelength):
    """Compute the V number of a fiber.

    Parameters
    ----------
    radius : float
        core radius, microns
    NA : float
        numerical aperture
    wavelength : float
        vacuum wavelength, microns

    Returns
    -------
    float
        V-number

    Notes
    -----
    if V is less than ~2.4048, a fiber behaves as a single mode fiber.
    V is a "normalized frequency"
    For multi-mode fibers, the number of guided modes is M ~= V^2/2

    """
    # k * r * NA
    # k = wavenumber
    return 2 * np.pi / wavelength * radius * NA


def _ghatak_eq_8_40(b, V, l):  # NOQA
    """Ghatak's Eq. 8.40.

    Returns left hand side minus right hand side.  This function is a boundary
    value problem; when LHS=RHS, the mode in the cladding and the mode in the
    core are of equal power.  Maxwell's equations require this for a mode to
    propagate.

    Also 4.41 for l=0

    Parameters
    ----------
    b : float
        normalized propagation constant, 0 < b < 1
    V : float
        V number (see the V function)
    l : int
        fiber mode index, l=0 are the radially symmetric modes,
        l=1..N are the asymmetric modes

    Returns
    -------
    float
        the difference between the core field at the boundary and the cladding
        field at the boundary.  A mode only propagates when this difference is
        zero, i.e. at the roots of this function

    """
    jn = special.jn
    kn = special.kn
    j0 = special.j0
    j1 = special.j1
    k0 = special.k0
    k1 = special.k1

    U = V * np.sqrt(1-b)
    W = V * np.sqrt(b)
    if l >= 1:  # noqa
        # right looks like it may be a typo in Ghatak?  -W in 8.40, not in 8.41
        # however, fig 8.1 only replicates for -W, and the same for fig 8.4
        left = U * jn(l-1, U) / jn(l, U)
        right = -W * kn(l-1, W) / kn(l, W)
    else:
        # left = U * jn(l-1, U) / jn(l, U)
        # right = -W * kn(l-1, W) / kn(l, W)
        left = U * j1(U) / j0(U)
        right = W * k1(W) / k0(W)
    return left-right


def find_all_roots(f, args=(), kwargs=None, interval=(0, 1), npts_signsearch=1000, maxiter=50):
    """Find all roots of f on interval.

    This routine is customized for finding fiber modes.
    The logic with fl and fr (f at left bound, f at right bound)
    and discarding roots is a heuristic for this specific class of problem.

    Remove this logic to generalize this routine.

    Parameters
    ----------
    f: callable
        function which may or may not have one or more roots on interval
        must take a single argument
    args : iterable
        additional positional arguments to pass to f
    kwargs : dict
        additional keyword arguments to pass to f
    interval: length 2 tuple
        (lower, upper) bound on which to search for roots
    npts_signsearch: int
        number of points used in a coarse search for sign changes in f
    maxiter : int
        maximum number of iterations to use when searching for a root on each
        segment

    Returns
    -------
    roots: np.ndarray
        Array containing all unique roots that were found in `bracket`.
        if there are no roots, empty ndarray
        if there is one root, length 1 ndarray
        if there are multiple roots, array in ascending x

    """
    if kwargs is None:
        kwargs = {}

    def curried_f(x):
        return f(x, *args, **kwargs)

    x = np.linspace(*interval, npts_signsearch)
    y = curried_f(x)

    sgn = np.sign(y)
    sign_changes = (sgn[:-1] != sgn[1:]).nonzero()[0]  # nonzero returns array inside a tuple
    # != makes a logical array, nonzero returns the indices of the non-zero
    # (False) elements
    roots = []
    for j in sign_changes:
        left = x[j]
        right = x[j+1]
        fl = curried_f(left)
        fr = curried_f(right)

        if (fl < -cutoff and fr > cutoff) or (fl > cutoff and fr < -cutoff):
            # this is a region where either the left or right hand side of
            # Ghatak Eq. 8.40 is an infinity or zero; discard this root
            continue

        _, root = brentq(curried_f, a=left, b=right, maxiter=maxiter, full_output=True)
        if not root.converged:
            raise ValueError(f'root search on interval{x[j]}-{x[j+1]} failed')
        roots.append(root.root)

    # it should not happen, but two brackets may find the same root if f is
    # extremely locally nonconvex.
    roots2 = truenp.unique(roots)
    if len(roots) != len(roots2):
        raise ValueError(f'root search found duplicate roots, all roots were {roots}')

    return roots


def find_all_modes(V):
    """Identify the modes of a step-index fiber.

    Parameters
    ----------
    V : float
        V-number (see the V function)

    Returns
    -------
    dict
        keys of l, values of b for each m [0, 1, ...]
        for example
        {
            0: (0.9, 0.6, 0.3)
        }
        would be a three-mode fiber, with no azimuthally variant modes

    """
    # heuristic: need more than say 50 points to find all zero crossings
    # if not a single-mode fiber.  _ghatak_eq_8_40 runs quickly, so never try
    # less than ~ 50 pts
    # as V increases, number of guided modes increases, so use ~V^1.5
    # as additional number of points

    # LP are "Linearly Polarized" modes, ghatak below eq. 8.12, pg 134
    npts = int(50 + V**2)
    kwargs = dict(V=V, l=0)
    eps = 1e-14  # brentq will find the NaNs at b=0 and b=1 and mistake them for roots
    interval = (0+eps, 1-eps)
    # ::-1 -- reverse the order to be in descending b
    l0_bs = find_all_roots(_ghatak_eq_8_40, kwargs=kwargs, npts_signsearch=npts, interval=interval)[::-1]
    out = {0: l0_bs}

    bs = l0_bs
    ell = 0
    while len(bs) > 0:
        ell += 1
        kwargs['l'] = ell
        bs = find_all_roots(_ghatak_eq_8_40, kwargs=kwargs, npts_signsearch=npts, interval=interval)[::-1]
        if len(bs) > 0:
            out[ell] = bs
            out[-ell] = bs

    return out


def compute_LP_modes(V, mode_dict, a, r, t):
    """Numerically compute Linearly Polarized mode for a step-index cylindrical fiber.

    Parameters
    ----------
    V : float
        V-number (see the V function)
    mode_dict : dict
        the dictionary returned by find_all_modes
    a : float
        fiber's core radius, microns
    r : ndarray
        radial coordinates, microns
    t : ndarray
        azimuthal coordinates, radians

    Returns
    -------
    dict
        a dict of the same "structure" as the one returned by find_all_modes,
        but instead of values of b, the values are ndarrays containing the
        spatial modes

    """
    jn = special.jn
    kn = special.kn
    j0 = special.j0
    j1 = special.j1
    k0 = special.k0
    k1 = special.k1

    # the boundary condition for a mode to be propagating requires that the
    # field at the edge of the cladding be equal to the edge of the core,
    # so there is no "third region" that is neither within core nor clad and
    # it is arbitrary which we take on the boundary
    rnorm = r/a
    within_core = r <= a
    within_clad = ~within_core

    max_l = max(mode_dict.keys())
    sines = {}
    cosines = {}
    for l in range(1, max_l+1):  # NOQA
        sines[l] = np.sin(l*t)
        cosines[l] = np.cos(l*t)

    out = {}

    for l in mode_dict.keys():  # NOQA
        bs = mode_dict[l][::-1]
        modes_l = []
        for b in bs:
            U = V * np.sqrt(1-b)
            W = V * np.sqrt(b)
            tmp = np.zeros_like(r)
            if l == 0:  # noqa
                num_core = j0(U*rnorm[within_core])
                den_core = j0(U)
                num_clad = k0(W*rnorm[within_clad])
                den_clad = k0(W)
            elif l == 1:  # noqa
                num_core = j1(U*rnorm[within_core])
                den_core = j1(U)
                num_clad = k1(W*rnorm[within_clad])
                den_clad = k1(W)
            else:
                num_core = jn(l, U*rnorm[within_core])
                den_core = jn(l, U)
                num_clad = kn(l, W*rnorm[within_clad])
                den_clad = kn(l, W)

            tmp[within_core] = num_core/den_core
            tmp[within_clad] = num_clad/den_clad

            if l != 0:  # noqa
                if l < 0:
                    tmp *= sines[-l]
                else:
                    tmp *= cosines[l]

            modes_l.append(tmp)

        out[l] = modes_l

    return out


def marcuse_mfr_from_V(V):
    """Marcuse' estimate for the mode field radius based on the V-number."""
    # D. Marcuse, “Loss analysis of single-mode fiber splices”, Bell Syst. Tech. J. 56, 703 (1977)
    # https://doi.org/10.1002/j.1538-7305.1977.tb00534.x

    return 0.65 + 1.619 * V ** -1.5 + 2.879 * V ** -6


def petermann_mfr_from_V(V):
    """Petermann's estimate for the mode field radius based on the V-number.

    More accurate than Marcuse

    """
    # TODO: cite
    # accurate to within ~1% from V=1.5 to 2.5
    # see also https://www.rp-photonics.com/mode_radius.html
    return marcuse_mfr_from_V(V) - 0.016 - 1.567 * V ** -7


def mode_overlap_integral(E1, E2, E2conj=None, I1sum=None, I2sum=None):
    r"""Compute the mode overlap integral.

    ..math::
        \eta = \frac{\left| \int{}E_1^* E_2 \right|^2}{\int I_1 \int I_2}


    When repeatedly computing coupling of varying fields into a consistent mode,
    consider precomputing E2conj and I2sum and passing them as arguments to
    accelerate computation.

    Parameters
    ----------
    E1 : array
        complex field of mode 1
    E2 : array
        complex field of mode 2
    E2conj : array
        E2.conj()
    I1sum : array, optional
        sum of the intensity of mode 1; I1 = abs(E1)**2; I1sum = I1.sum()
    I2sum : array, optional
        sum of the intensity of mode 2; I2 = abs(E2)**2; I2sum = I2.sum()

    Returns
    -------
    float
        eta, coupling efficiency into the mode; bounded between [0,1]

    """
    if I1sum is None:
        I1 = abs(E1)
        I1 *= I1
        I1sum = I1.sum()
    if I2sum is None:
        I2 = abs(E2)
        I2 *= I2
        I2sum = I2.sum()
    if E2conj is None:
        E2conj = E2.conj()

    cross_intensity = E1 * E2conj
    num = abs(cross_intensity.sum())**2
    den = I1sum*I2sum
    return num/den
