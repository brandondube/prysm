"""Routines for working with optical fibers."""
# scipy.special is used directly here because fiber LP solving depends on
# host-side Bessel ufuncs plus Bessel-zero table generators.
from scipy import special

from prysm.conf import config
from prysm.mathops import np


_BESSELJ_ZERO_CACHE = {}


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
    U = V * np.sqrt(1-b)
    W = V * np.sqrt(b)
    with np.errstate(divide='ignore', invalid='ignore'):
        if l >= 1:  # noqa
            # right looks like it may be a typo in Ghatak?  -W in 8.40, not in 8.41
            # however, fig 8.1 only replicates for -W, and the same for fig 8.4
            left = U * special.jv(l - 1, U) / special.jv(l, U)
            right = -W * special.kve(l - 1, W) / special.kve(l, W)
        else:
            # left = U * J_{l-1}(U) / J_l(U)
            # right = -W * K_{l-1}(W) / K_l(W)
            left = U * special.j1(U) / special.j0(U)
            right = W * special.k1(W) / special.k0(W)
    return left-right


def _besselj_positive_zeros(l, x_max):
    """All positive zeros of J_l strictly below x_max, ascending.

    SciPy's jn_zeros gives the exact cutoff table needed for LP mode counting
    and bracketing, so this helper only has to request enough zeros to cover
    x_max and cache the result.
    """
    x_max = float(x_max)
    dtype = np.dtype(config.precision)
    cache_key = (int(l), dtype.str)
    cached = _BESSELJ_ZERO_CACHE.get(cache_key)
    if cached is not None:
        cached_x_max, cached_zeros = cached
        if cached_x_max >= x_max:
            return cached_zeros[cached_zeros < x_max].copy()

    nt = max(8, int(x_max / np.pi) + 8)
    while True:
        zeros = np.asarray(special.jn_zeros(l, nt), dtype=config.precision)
        if zeros[-1] >= x_max:
            zeros = zeros[zeros < x_max]
            _BESSELJ_ZERO_CACHE[cache_key] = (x_max, zeros)
            return zeros.copy()
        nt *= 2


def _ghatak_u_with_derivative(U, V, ell):
    """Dispersion equation f(U) and its derivative df/dU.

    Uses U as the natural coordinate (W = sqrt(V^2 - U^2)), avoiding the extra
    sqrt(1-b) needed in the b-parameterization.  Derivation uses the standard
    Bessel recurrences to express J_{ell-2}/J_ell and K_{ell-2}/K_ell in terms
    of the ratios r_J, r_K, yielding closed forms with no higher-order Bessels.
    """
    W = np.sqrt(V * V - U * U)
    with np.errstate(divide='ignore', invalid='ignore'):
        if ell == 0:
            r_J = special.j1(U) / special.j0(U)
            r_K = special.k1(W) / special.k0(W)
            f = U * r_J - W * r_K
            df = U * (r_J * r_J + r_K * r_K)
        else:
            r_J = special.jv(ell - 1, U) / special.jv(ell, U)
            r_K = special.kve(ell - 1, W) / special.kve(ell, W)
            f = U * r_J + W * r_K
            df = 2 * ell * (r_J - U * r_K / W) - U * (r_J * r_J + r_K * r_K)
    return f, df


def _vectorized_safeguarded_newton_u(V, ell, lower, upper, max_iter=28, atol=1e-12):
    """Newton-Raphson on f(U)=0 for many roots at once, with bisection fallback.

    lower, upper are arrays of U-bracket endpoints (one bracket per root).
    Each iteration evaluates (f, df) in one batched Bessel call across all
    roots; if a Newton step would exit its bracket the corresponding root
    falls back to the bracket midpoint (bisection step).  Brackets shrink
    monotonically using the sign of f at the new point.

    Near-cutoff modes can have very steep residuals, so bisection remains part
    of the iteration even though Newton usually lands quickly.
    """
    a = np.asarray(lower).copy()
    b = np.asarray(upper).copy()
    fa, _ = _ghatak_u_with_derivative(a, V, ell)
    x = 0.5 * (a + b)
    fx, dfx = _ghatak_u_with_derivative(x, V, ell)

    for _ in range(max_iter):
        converged = np.abs(fx) < atol
        step = np.where(dfx != 0, -fx / dfx, 0.0)
        x_newton = x + step
        in_bracket = (x_newton > a) & (x_newton < b)
        x_new = np.where(in_bracket, x_newton, 0.5 * (a + b))
        x_new = np.where(converged, x, x_new)

        f_new, df_new = _ghatak_u_with_derivative(x_new, V, ell)

        update = ~converged
        same_sign_as_a = (np.sign(f_new) == np.sign(fa))
        a = np.where(update & same_sign_as_a, x_new, a)
        fa = np.where(update & same_sign_as_a, f_new, fa)
        b = np.where(update & ~same_sign_as_a, x_new, b)

        x = x_new
        fx = f_new
        dfx = df_new
        if bool(np.all(np.abs(fx) < atol)):
            break
    return x


def _mode_u_brackets(V, cutoffs, poles):
    """Return U-domain (lower, upper) brackets implied by LP cutoff and pole theory.

    For ell >= 1, LP_{ell,m} cuts off at a zero of J_{ell-1} and the
    corresponding dispersion-equation pole is the same-index zero of J_ell.
    For ell == 0, the caller supplies the special cutoffs [0, zeros(J_1)] and
    poles zeros(J_0).  If the next pole sits above V, the upper U endpoint is
    simply V, i.e. b -> 0.
    """
    if len(cutoffs) == 0:
        return np.empty(0, dtype=config.precision), np.empty(0, dtype=config.precision)

    V = float(V)
    tiny_u = np.sqrt(np.finfo(config.precision).eps) * max(V, 1.0)
    lower = []
    upper = []

    for idx, cutoff_u in enumerate(cutoffs):
        cutoff_u = float(cutoff_u)
        pole_u = float(poles[idx]) if idx < len(poles) else V
        upper_u = min(pole_u, V)
        span = upper_u - cutoff_u
        if span <= 0:
            continue
        du = min(tiny_u, 1e-3 * span)
        left_u = cutoff_u + du if cutoff_u > 0 else du
        right_u = upper_u - du
        if right_u <= left_u:
            continue
        lower.append(left_u)
        upper.append(right_u)

    return (
        np.asarray(lower, dtype=config.precision),
        np.asarray(upper, dtype=config.precision),
    )


def _families(V):
    """Yield (ell, cutoffs, poles) per LP family present at this V.

    The ell=0 family carries a synthetic 0 prepended to its cutoff list, since
    LP_{0,1} has no real cutoff; subsequent families use real zeros of J_{ell-1}.
    Iteration terminates when the next family's cutoff list (zeros of
    J_{ell-1} below V) is empty.  A module-local cache reuses zero arrays
    across the (ell-1, ell) pairs.
    """
    zero_cache = {}

    def zeros(order):
        if order not in zero_cache:
            zero_cache[order] = _besselj_positive_zeros(order, V)
        return zero_cache[order]

    yield (
        0,
        np.concatenate((np.asarray([0], dtype=config.precision), zeros(1))),
        zeros(0),
    )
    ell = 1
    while True:
        cutoffs = zeros(ell - 1)
        if len(cutoffs) == 0:
            return
        yield ell, cutoffs, zeros(ell)
        ell += 1


def find_all_modes(V, count_only=False):
    """Identify the modes of a step-index fiber.

    Parameters
    ----------
    V : float
        V-number (see the V function)
    count_only : bool, optional
        If True, return per-l mode counts from cutoff theory without solving
        the dispersion equation.

    Returns
    -------
    dict
        keys of l, values of b for each m [0, 1, ...], or integer counts when
        count_only is True
        for example::

            {
                0: (0.9, 0.6, 0.3)
            }

        would be a three-mode fiber, with no azimuthally variant modes

    """
    # LP are "Linearly Polarized" modes, Ghatak below eq. 8.12, pg. 134.
    # The mode intervals are known from cutoff theory, so a coarse b-grid is
    # unnecessary: LP_0,1 starts at U=0, LP_0,m>=2 cuts off at zeros of J_1,
    # and LP_l,m for l>=1 cuts off at zeros of J_{l-1}.  Each root is bracketed
    # above by the same-index pole of the J_l denominator, or by U=V when the
    # pole is above the normalized frequency.
    out = {}
    for ell, cutoffs, poles in _families(V):
        if count_only:
            n = len(cutoffs)
            out[ell] = n
            if ell > 0:
                out[-ell] = n
            continue

        lower, upper = _mode_u_brackets(V, cutoffs, poles)
        if len(lower) == 0:
            continue
        roots_u = _vectorized_safeguarded_newton_u(V, ell, lower, upper)
        # b decreases as U increases; brackets are ordered by ascending U,
        # so flip to get descending b (matches the original convention).
        roots_b = (1 - (roots_u / V) ** 2)[::-1]
        out[ell] = roots_b
        if ell > 0:
            out[-ell] = roots_b

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
                num_core = special.j0(U*rnorm[within_core])
                den_core = special.j0(U)
                num_clad = special.k0(W*rnorm[within_clad])
                den_clad = special.k0(W)
            elif l == 1:  # noqa
                num_core = special.j1(U*rnorm[within_core])
                den_core = special.j1(U)
                num_clad = special.k1(W*rnorm[within_clad])
                den_clad = special.k1(W)
            else:
                num_core = special.jv(l, U*rnorm[within_core])
                den_core = special.jv(l, U)
                num_clad = special.kv(l, W*rnorm[within_clad])
                den_clad = special.kv(l, W)

            with np.errstate(divide='ignore', invalid='ignore'):
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


def smf_mode_field(V, a, b, r):
    """Mode field of a single mode fiber.

    Parameters
    ----------
    V : float
        V-number (see the V function)
    a : float
        fiber's core radius, microns
    b : float
        propagation constant for the mode
    r : ndarray
        radial coordinates, microns

    Returns
    -------
    float
        the single mode of the fiber

    """
    U = V * np.sqrt(1-b)
    W = V * np.sqrt(b)
    # inside core
    rnorm = r*(1/a)  # faster to divide on scalar, mul on vector
    rinterior = rnorm < 1
    out = np.empty_like(r)
    with np.errstate(divide='ignore', invalid='ignore'):
        num = special.j0(U*rnorm[rinterior])
        den = special.j1(U)
        out[rinterior] = num*(1/den)

        rexterior = ~rinterior
        num = special.k0(W*rnorm[rexterior])
        den = special.k1(W)
        out[rexterior] = num*(1/den)
    return out


def marcuse_mfr_from_V(V):
    """Marcuse' estimate for the mode field radius based on the V-number.

    Parameters
    ----------
    V : float
        V-number (see the V function)

    Returns
    -------
    float
        w/a, the ratio of mode field radius to core radius

    """
    # D. Marcuse, “Loss analysis of single-mode fiber splices”, Bell Syst. Tech. J. 56, 703 (1977)
    # https://doi.org/10.1002/j.1538-7305.1977.tb00534.x

    return 0.65 + 1.619 * V ** -1.5 + 2.879 * V ** -6


def petermann_mfr_from_V(V):
    """Petermann's estimate for the mode field radius based on the V-number.

    More accurate than Marcuse

    Parameters
    ----------
    V : float
        V-number (see the V function)

    Returns
    -------
    float
        w/a, the ratio of mode field radius to core radius


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


def multimode_coupling(E_in, mode_fields):
    """Per-LP-mode coupling efficiency from an incident field.

    Parameters
    ----------
    E_in : ndarray
        complex incident field at the fiber face, same grid as the mode fields
    mode_fields : dict
        dict returned by compute_LP_modes; keys are azimuthal indices l (with
        negative l for the sine-azimuthal partners), values are lists of 2D
        arrays for each radial index m

    Returns
    -------
    dict
        same key structure as mode_fields, values are lists of float coupling
        efficiencies eta_(l,m), each in [0, 1].  Summing across all returned
        values approximates the total guided-mode coupling.

    Notes
    -----
    LP modes are approximately orthonormal on a grid spanning several core
    radii; the (l, -l) pairs share radial shape and only differ in their
    cos(l*t) / sin(l*t) azimuthal factor, so each captures part of the
    angular content of the input.

    """
    I_in = abs(E_in)
    I_in = I_in * I_in
    I_in_sum = I_in.sum()
    E_in_conj = E_in.conj()
    out = {}
    for l, modes in mode_fields.items():  # NOQA
        out[l] = [
            mode_overlap_integral(mode, E_in, E2conj=E_in_conj, I2sum=I_in_sum)
            for mode in modes
        ]
    return out
