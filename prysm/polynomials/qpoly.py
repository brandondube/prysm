"""Tools for working with Q (Forbes) polynomials."""
# not special engine, only concerns scalars here
from collections import defaultdict
from functools import lru_cache

from scipy import special

from .jacobi import jacobi, jacobi_sequence, jacobi_sum_clenshaw_der

from prysm.mathops import np, kronecker, gamma, sign
from prysm.conf import config


@lru_cache(1000)
def g_qbfs(n_minus_1):
    """g(m-1) from oe-18-19-19700 eq. (A.15)."""
    if n_minus_1 == 0:
        return - 1 / 2
    else:
        n_minus_2 = n_minus_1 - 1
        return - (1 + g_qbfs(n_minus_2) * h_qbfs(n_minus_2)) / f_qbfs(n_minus_1)


@lru_cache(1000)
def h_qbfs(n_minus_2):
    """h(m-2) from oe-18-19-19700 eq. (A.14)."""
    n = n_minus_2 + 2
    return -n * (n - 1) / (2 * f_qbfs(n_minus_2))


@lru_cache(1000)
def f_qbfs(n):
    """f(m) from oe-18-19-19700 eq. (A.16)."""
    if n == 0:
        return 2
    elif n == 1:
        return np.sqrt(19) / 2
    else:
        term1 = n * (n + 1) + 3
        term2 = g_qbfs(n - 1) ** 2
        term3 = h_qbfs(n - 2) ** 2
        return np.sqrt(term1 - term2 - term3)


def Qbfs(n, x):
    """Qbfs polynomial of order n at point(s) x.

    Parameters
    ----------
    n : int
        polynomial order
    x : numpy.array
        point(s) at which to evaluate

    Returns
    -------
    numpy.ndarray
        Qbfs_n(x)

    """
    # to compute the Qbfs polynomials, compute the auxiliary polynomial P_n
    # recursively.  Simultaneously use the recurrence relation for Q_n
    # to compute the intermediary Q polynomials.
    # for input x, transform r = x ^ 2
    # then compute P(r) and consequently Q(r)
    # and scale outputs by Qbfs = r*(1-r) * Q
    # the auxiliary polynomials are the jacobi polynomials with
    # alpha,beta = (-1/2,+1/2),
    # also known as the chebyshev polynomials of the third kind, V(x)

    # the first two Qbfs polynomials are
    # Q_bfs0 = x^2 - x^4
    # Q_bfs1 = 1/19^.5 * (13 - 16 * x^2) * (x^2 - x^4)
    rho = x ** 2
    # c_Q is the leading term used to convert Qm to Qbfs
    c_Q = rho * (1 - rho)
    if n == 0:
        return c_Q  # == x^2 - x^4

    if n == 1:
        return 1 / np.sqrt(19) * (13 - 16 * rho) * c_Q

    # c is the leading term of the recurrence relation for P
    c = 2 - 4 * rho
    # P0, P1 are the first two terms of the recurrence relation for auxiliary
    # polynomial P_n
    P0 = np.ones_like(x) * 2
    P1 = 6 - 8 * rho
    Pnm2 = P0
    Pnm1 = P1

    # Q0, Q1 are the first two terms of the recurrence relation for Qm
    Q0 = np.ones_like(x)
    Q1 = 1 / np.sqrt(19) * (13 - 16 * rho)
    Qnm2 = Q0
    Qnm1 = Q1
    for nn in range(2, n+1):
        Pn = c * Pnm1 - Pnm2
        Pnm2 = Pnm1
        Pnm1 = Pn
        g = g_qbfs(nn - 1)
        h = h_qbfs(nn - 2)
        f = f_qbfs(nn)
        Qn = (Pn - g * Qnm1 - h * Qnm2) * (1/f)  # small optimization; mul by 1/f instead of div by f
        Qnm2 = Qnm1
        Qnm1 = Qn

    # Qn is certainly defined (flake8 can't tell the previous ifs bound the loop
    # to always happen once)
    return Qn * c_Q  # NOQA

# to do Qn derivative, Qn = [Pn - g Qnm1 - h Qnm2]/f
# then,                Qn'= [Pn' - g Qnm1' - hQnm2']/f
# ... this process would be miserable, so we use the change of basis instead
# Forbes2010 Qbfs Eq. 3.2 to 3.5
# a_m = Qbfs coefficients
# b_m = Cheby third kind coefficients
# b_M = a_M / f_M
# B_M-1 = (a_M-1 - g_M-1 bM) / f_M-1
# B_m = (a_m - g_m b_m+1 - h_m b_m+2) / f_m
# so, general proces... for Qbfs, don't provide derivatives, but provide a way
# to change basis to cheby third kind, which can then be differentiated.


def change_basis_Qbfs_to_Pn(cs):
    """Perform the change of basis from Qbfs to the auxiliary polynomial Pn.

    The auxiliary polynomial is defined in A.4 of oe-18-19-19700 and is the
    shifted Chebyshev polynomials of the third kind.

    Qbfs polynomials u^2(1-u^2)Qbfs_n(u^2) can be expressed as u^2(1-u^2)Pn(u^2)
    u in Forbes' parlance is the normalized radial coordinate, so given points r
    in the range [0,1], use this function and then polynomials.cheby3(n, r*r).
    The u^2 (1 - u^2) is baked into the Qbfs function and will need to be applied
    by the caller for Cheby3.

    Parameters
    ----------
    cs : iterable
        sequence of polynomial coefficients, from order n=0..len(cs)-1

    Returns
    -------
    numpy.ndarray
        array of same type as cs holding the coefficients that represent the
        same surface as a sum of shifted Chebyshev polynomials of the third kind


    """
    if hasattr(cs, 'dtype'):
        # array, initialize as array
        bs = np.empty_like(cs)
    else:
        # iterable input
        bs = np.empty(len(cs), dtype=config.precision)

    M = len(bs)-1
    fM = f_qbfs(M)
    bs[M] = cs[M]/fM
    g = g_qbfs(M-1)
    f = f_qbfs(M-1)
    bs[M-1] = (cs[M-1] - g * bs[M])/f
    for i in range(M-2, -1, -1):
        g = g_qbfs(i)
        h = h_qbfs(i)
        f = f_qbfs(i)
        bs[i] = (cs[i] - g * bs[i+1] - h*bs[i+2])/f

    return bs


def clenshaw_qbfs(cs, u, alphas=None):
    """Use Clenshaw's method to compute a Qbfs surface from its coefficients.

    Parameters
    ----------
    cs : iterable of float
        coefficients for a Qbfs surface, from order 0..len(cs)-1
    u : numpy.ndarray
        radial coordinate(s) to evaluate, notionally in the range [0,1]
        the variable u from oe-18-19-19700
    alphas : numpy.ndarray, optional
        array to store the alpha sums in,
        the surface is u^2(1-u^2) * (2 * (alphas[0]+alphas[1])
        if not None, alphas should be of shape (len(s), *x.shape)
        see _initialize_alphas if you desire more information

    Returns
    -------
    numpy.ndarray
        Qbfs surface, the quantity u^2(1-u^2) S(u^2) from Eq. (3.13)
        note: excludes the division by phi, since c and rho are unknown

    """
    x = u * u
    bs = change_basis_Qbfs_to_Pn(cs)
    # alphas = np.zeros((len(cs), len(u)), dtype=u.dtype)
    alphas = _initialize_alphas(cs, u, alphas, j=0)
    M = len(bs)-1
    prefix = 2 - 4 * x
    alphas[M] = bs[M]
    alphas[M-1] = bs[M-1] + prefix * alphas[M]
    for i in range(M-2, -1, -1):
        alphas[i] = bs[i] + prefix * alphas[i+1] - alphas[i+2]

    S = 2 * (alphas[0] + alphas[1])
    return (x * (1 - x)) * S


def _initialize_alphas(cs, x, alphas, j=0):
    # j = derivative order
    if alphas is None:
        if hasattr(x, 'dtype'):
            dtype = x.dtype
        else:
            dtype = config.precision
        if hasattr(x, 'shape'):
            shape = (len(cs), *x.shape)
        elif hasattr(x, '__len__'):
            shape = (len(cs), len(x))
        else:
            shape = (len(cs),)

        if j != 0:
            shape = (j+1, *shape)

        alphas = np.zeros(shape, dtype=dtype)
    return alphas


def clenshaw_qbfs_der(cs, u, j=1, alphas=None):
    """Use Clenshaw's method to compute Nth order derivatives of a Qbfs surface.

    This function does not really compute the surface derivative.  It computes
    S^j from Eq. 3.12.  Because the product rule must be applied, some
    combination of S^j, S^j-1, .. S^0 are needed to really compute the surface
    derivative.

    For the first two derivatives, the necessary calculation is:
    rho = un-normalized radial coordinate
    u = rho/rho_max (rho_max being the normalization radius)
    phi = sqrt(1 - c^2 rho^2)
    if z(rho) is the surface, then
    z(rho) = [c rho^2 / (1 + phi)] + u^2(1 - u^2)/phi * S(u^2)
    z'(rho) = [c rho / phi] + (u[1+phi^2 - u^2(1+3phi^2)])/(rho_max phi^3) S(u^2) + 2u^3(1-u^2)/(rho_max phi) S'(u^2)
    z''(rho) = ... very long expression I am not willing to type out, see Eq. 3.15
    Note: S^j is simply 2 * (alphas[j][0] + alphas[j][1]) for any j (including 0)

    See compute_zprime_Qbfs for this calculation integrated

    Parameters
    ----------
    cs : iterable of float
        coefficients for a Qbfs surface, from order 0..len(cs)-1
    u : numpy.ndarray
        radial coordinate(s) to evaluate, notionally in the range [0,1]
        the variable u from oe-18-19-19700
    j : int
        derivative order
    alphas : numpy.ndarray, optional
        array to store the alpha sums in,
        if x = u * u, then
        S   = (x * (1 - x)) * 2 * (alphas[0][0] + alphas[0][1])
        S'  = ... .. the same, but alphas[1][0] and alphas[1][1]
        S'' = ... ... ... ... ... ... [2][0] ... ... ..[1][1]
        etc

        if not None, alphas should be of shape (j+1, len(cs), *x.shape)
        see _initialize_alphas if you desire more information

    Returns
    -------
    numpy.ndarray
        the alphas array

    """
    # not-so-TODO: a little optimization room here, since we compute u*u multiple
    # times and what-not.  The flags to disable that would be really confusing
    # to users, though.
    x = u * u
    M = len(cs) - 1
    prefix = 2 - 4 * x
    alphas = _initialize_alphas(cs, u, alphas, j=j)
    # seed with j=0 (S, not its derivative)
    clenshaw_qbfs(cs, u, alphas[0])
    for jj in range(1, j+1):
        alphas[jj][M-j] = -4 * jj * alphas[jj-1][M-jj+1]
        for n in range(M-2, -1, -1):
            # this is hideous, and just expresses:
            # for the jth derivative, alpha_n is 2 - 4x * a_n+1 - a_n+2 - 4 j a_n+1^j-1
            alphas[jj][n] = prefix * alphas[jj][n+1] - alphas[jj][n+2] - 4 * jj * alphas[jj-1][n+1]

    return alphas


def compute_z_zprime_Qbfs(coefs, rho, rho_max, u, c):
    """Compute the surface sag and first radial derivative of a Qbfs surface.

    Requires composition with raytracing.sphere_sag for actual surface sag,
    this is only the aspheric cap.

    from Eq. 3.13 and 3.14 of oe-18-19-19700.

    Parameters
    ----------
    coefs : iterable
        surface coefficients for Q0..QN, N=len(coefs)-1
    rho : numpy.ndarray
        unnormalized radial coordinates
    rho_max : numpy.ndarray
        normalization radius
    u : numpy.ndarray
        normalized radial coordinates (rho/rho_max)
    c : float
        best fit sphere curvature
        use c=0 for a flat base surface

    Returns
    -------
    numpy.ndarray, numpy.ndarray
        surface sag, surface sag derivative

    """
    # clenshaw does its own u^2
    alphas = clenshaw_qbfs_der(coefs, u, j=1)
    S = 2 * (alphas[0][0] + alphas[0][1])
    Sprime = 2 * (alphas[1][0] + alphas[1][1])
    rhosq = rho ** 2
    usq = u ** 2
    phi = np.sqrt(1 - c ** 2 * rhosq)

    num = usq * (1 - usq)
    den = phi
    z = num / den * S

    phisq = phi * phi
    phicub = phi * phi * phi
    num = u * (1 + phisq - usq * (1 + 3 * phisq))
    den = rho_max * phicub
    term2 = num / den * S

    #         u^3
    num = 2 * usq * u * (1 - usq)
    den = rho_max * phi
    term3 = num / den * Sprime

    zprime = term2 + term3
    return z, zprime


def compute_z_zprime_Qcon(coefs, rho_max, u):
    """Compute the surface sag and first radial derivative of a Qcon surface.

    Requires composition with raytracing.sphere_sag for actual surface sag,
    this is only the aspheric cap.

    from Eq. 5.3 and 5.3 of oe-18-13-13851.

    Parameters
    ----------
    coefs : iterable
        surface coefficients for Q0..QN, N=len(coefs)-1
    rho_max : float
        maximum radial coordinate
        use rho_max=1 if not interested in a "real surface" of a given diameter
        and only interested in the normalized world.
    u : numpy.ndarray
        normalized radial coordinates (rho/rho_max)

    Returns
    -------
    numpy.ndarray, numpy.ndarray
        surface sag, surface sag derivative

    """
    usq = u * u
    u3 = usq * u
    u4 = usq * usq
    u5 = usq * u3
    x = 2 * usq - 1
    alphas = jacobi_sum_clenshaw_der(coefs, 0, 4, x=x, j=1)
    S = alphas[0][0]
    Sprime = alphas[1][0]

    z = u4 * S

    # Forbes' paper is wrong, or I am just "cheating" with Jacobis in a way
    # that is slightly different.
    # Forbes has u^4 S(u^2), and says use a connection to Z_m=4 to do it.
    # I don't do that, and go straight to the Jacobis
    # but the change of variables is different.
    # u is the normalized radial variable,
    # u^2 is the argument, but the change-of-variables for the Jacobi polynomials
    # is 2x-1, so the chain rule works out differently.
    term1 = 4 * u3 * S
    term2 = 4 * u5 * Sprime
    zprime = term1 + term2
    return z, zprime


def Qbfs_sequence(ns, x):
    """Qbfs polynomials of orders ns at point(s) x.

    Parameters
    ----------
    ns : Iterable of int
        polynomial orders
    x : numpy.array
        point(s) at which to evaluate

    Returns
    -------
    generator of numpy.ndarray
        yielding one order of ns at a time

    """
    # see the leading comment of Qbfs for some explanation of this code
    # and prysm:jacobi.py#jacobi_sequence the "_sequence" portion

    ns = list(ns)
    min_i = 0

    rho = x ** 2
    # c_Q is the leading term used to convert Qm to Qbfs
    c_Q = rho * (1 - rho)
    if ns[min_i] == 0:
        yield np.ones_like(x) * c_Q
        min_i += 1

    if ns[min_i] == 1:
        yield 1 / np.sqrt(19) * (13 - 16 * rho) * c_Q
        min_i += 1

    # c is the leading term of the recurrence relation for P
    c = 2 - 4 * rho
    # P0, P1 are the first two terms of the recurrence relation for auxiliary
    # polynomial P_n
    P0 = np.ones_like(x) * 2
    P1 = 6 - 8 * rho
    Pnm2 = P0
    Pnm1 = P1

    # Q0, Q1 are the first two terms of the recurrence relation for Qbfs_n
    Q0 = np.ones_like(x)
    Q1 = 1 / np.sqrt(19) * (13 - 16 * rho)
    Qnm2 = Q0
    Qnm1 = Q1
    for nn in range(2, ns[-1]+1):
        Pn = c * Pnm1 - Pnm2
        Pnm2 = Pnm1
        Pnm1 = Pn
        g = g_qbfs(nn - 1)
        h = h_qbfs(nn - 2)
        f = f_qbfs(nn)
        Qn = (Pn - g * Qnm1 - h * Qnm2) * (1/f)  # small optimization; mul by 1/f instead of div by f
        Qnm2 = Qnm1
        Qnm1 = Qn
        if ns[min_i] == nn:
            yield Qn * c_Q
            min_i += 1


def Qcon(n, x):
    """Qcon polynomial of order n at point(s) x.

    Parameters
    ----------
    n : int
        polynomial order
    x : numpy.array
        point(s) at which to evaluate

    Returns
    -------
    numpy.ndarray
        Qcon_n(x)

    Notes
    -----
    The argument x is notionally uniformly spaced 0..1.
    The Qcon polynomials are obtained by computing c = x^4.
    A transformation is then made, x => 2x^2 - 1
    and the Qcon polynomials are defined as the jacobi polynomials with
    alpha=0, beta=4, the same order n, and the transformed x.
    The result of that is multiplied by c to yield a Qcon polynomial.
    Sums can more quickly be calculated by deferring the multiplication by
    c.

    """
    xx = x ** 2
    xx = 2 * xx - 1
    Pn = jacobi(n, 0, 4, xx)
    return Pn * x ** 4


def Qcon_sequence(ns, x):
    """Qcon polynomials of orders ns at point(s) x.

    Parameters
    ----------
    ns : Iterable of int
        polynomial orders
    x : numpy.array
        point(s) at which to evaluate

    Returns
    -------
    generator of numpy.ndarray
        yielding one order of ns at a time

    """
    xx = x ** 2
    xx = 2 * xx - 1
    x4 = x ** 4
    Pns = jacobi_sequence(ns, 0, 4, xx)
    for Pn in Pns:
        yield Pn * x4


@lru_cache(4000)
def abc_q2d(n, m):
    """A, B, C terms for 2D-Q polynomials.  oe-20-3-2483 Eq. (A.3).

    Parameters
    ----------
    n : int
        radial order
    m : int
        azimuthal order

    Returns
    -------
    float, float, float
        A, B, C

    """
    # D is used everywhere
    D = (4 * n ** 2 - 1) * (m + n - 2) * (m + 2 * n - 3)

    # A
    term1 = (2 * n - 1) * (m + 2 * n - 2)
    term2 = (4 * n * (m + n - 2) + (m - 3) * (2 * m - 1))
    A = (term1 * term2) / D

    # B
    num = -2 * (2 * n - 1) * (m + 2 * n - 3) * (m + 2 * n - 2) * (m + 2 * n - 1)
    B = num / D

    # C
    num = n * (2 * n - 3) * (m + 2 * n - 1) * (2 * m + 2 * n - 3)
    C = num / D

    return A, B, C


@lru_cache(4000)
def G_q2d(n, m):
    """G term for 2D-Q polynomials.  oe-20-3-2483 Eq. (A.15).

    Parameters
    ----------
    n : int
        radial order
    m : int
        azimuthal order

    Returns
    -------
    float
        G

    """
    if n == 0:
        num = special.factorial2(2 * m - 1)
        den = 2 ** (m + 1) * special.factorial(m - 1)
        return num / den
    elif n > 0 and m == 1:
        t1num = (2 * n ** 2 - 1) * (n ** 2 - 1)
        t1den = 8 * (4 * n ** 2 - 1)
        term1 = -t1num / t1den
        term2 = 1 / 24 * kronecker(n, 1)
        return term1 - term2
    else:
        # nt1 = numerator term 1, d = denominator...
        nt1 = 2 * n * (m + n - 1) - m
        nt2 = (n + 1) * (2 * m + 2 * n - 1)
        num = nt1 * nt2
        dt1 = (m + 2 * n - 2) * (m + 2 * n - 1)
        dt2 = (m + 2 * n) * (2 * n + 1)
        den = dt1 * dt2

        term1 = -num / den
        return term1 * gamma(n, m)


@lru_cache(4000)
def F_q2d(n, m):
    """F term for 2D-Q polynomials.  oe-20-3-2483 Eq. (A.13).

    Parameters
    ----------
    n : int
        radial order
    m : int
        azimuthal order

    Returns
    -------
    float
        F

    """
    if n == 0:
        num = m ** 2 * special.factorial2(2 * m - 3)
        den = 2 ** (m + 1) * special.factorial(m - 1)
        return num / den
    elif n > 0 and m == 1:
        t1num = 4 * (n - 1) ** 2 * n ** 2 + 1
        t1den = 8 * (2 * n - 1) ** 2
        term1 = t1num / t1den
        term2 = 11 / 32 * kronecker(n, 1)
        return term1 + term2
    else:
        Chi = m + n - 2
        nt1 = 2 * n * Chi * (3 - 5 * m + 4 * n * Chi)
        nt2 = m ** 2 * (3 - m + 4 * n * Chi)
        num = nt1 + nt2

        dt1 = (m + 2 * n - 3) * (m + 2 * n - 2)
        dt2 = (m + 2 * n - 1) * (2 * n - 1)
        den = dt1 * dt2

        term1 = num / den
        return term1 * gamma(n, m)


@lru_cache(4000)
def g_q2d(n, m):
    """Lowercase g term for 2D-Q polynomials.  oe-20-3-2483 Eq. (A.18a).

    Parameters
    ----------
    n : int
        radial order less one (n - 1)
    m : int
        azimuthal order

    Returns
    -------
    float
        g

    """
    return G_q2d(n, m) / f_q2d(n, m)


@lru_cache(4000)
def f_q2d(n, m):
    """Lowercase f term for 2D-Q polynomials.  oe-20-3-2483 Eq. (A.18b).

    Parameters
    ----------
    n : int
        radial order
    m : int
        azimuthal order

    Returns
    -------
    float
        f

    """
    if n == 0:
        return np.sqrt(F_q2d(n=0, m=m))
    else:
        return np.sqrt(F_q2d(n, m) - g_q2d(n-1, m) ** 2)


def Q2d(n, m, r, t):
    """2D Q polynomial, aka the Forbes polynomials.

    Parameters
    ----------
    n : int
        radial polynomial order
    m : int
        azimuthal polynomial order
    r : numpy.ndarray
        radial coordinate, slope orthogonal in [0,1]
    t : numpy.ndarray
        azimuthal coordinate, radians

    Returns
    -------
    numpy.ndarray
        array containing Q2d_n^m(r,t)
        the leading coefficient u^m or u^2 (1 - u^2) and sines/cosines
        are included in the return

    """
    # Q polynomials have auxiliary polynomials "P"
    # which are scaled jacobi polynomials under the change of variables
    # x => 2x - 1 with alpha = -3/2, beta = m-3/2
    # the scaling prefix may be found in A.4 of oe-20-3-2483

    # impl notes:
    # Pn is computed using a recurrence over order n.  The recurrence is for
    # a single value of m, and the 'seed' depends on both m and n.
    #
    # in general, Q_n^m = [P_n^m(x) - g_n-1^m Q_n-1^m] / f_n^m

    # for the sake of consistency, this function takes args of (r,t)
    # but the papers define an argument of u (really, u^2...)
    # which is what I call rho (or r).
    # for the sake of consistency of impl, I alias r=>u
    # and compute x = u**2 to match the papers
    u = r
    x = u ** 2
    if m == 0:
        return Qbfs(n, r)

    # m == 0 already was short circuited, so we only
    # need to consider the m =/= 0 case for azimuthal terms
    if sign(m) == -1:
        m = abs(m)
        prefix = u ** m * np.sin(m*t)
    else:
        prefix = u ** m * np.cos(m*t)
        m = abs(m)

    P0 = 1/2
    if m == 1 and n == 1:
        P1 = 1 - x/2
    else:
        P1 = (m - .5) + (1 - m) * x

    f0 = f_q2d(0, m)
    Q0 = 1 / (2 * f0)
    if n == 0:
        return Q0 * prefix

    g0 = g_q2d(0, m)
    f1 = f_q2d(1, m)
    Q1 = (P1 - g0 * Q0) * (1/f1)
    if n == 1:
        return Q1 * prefix
    # everything above here works, or at least everything in the returns works
    if m == 1:
        P2 = (3 - x * (12 - 8 * x)) / 6
        P3 = (5 - x * (60 - x * (120 - 64 * x))) / 10

        g1 = g_q2d(1, m)
        f2 = f_q2d(2, m)
        Q2 = (P2 - g1 * Q1) * (1/f2)

        g2 = g_q2d(2, m)
        f3 = f_q2d(3, m)
        Q3 = (P3 - g2 * Q2) * (1/f3)
        # Q2, Q3 correct
        if n == 2:
            return Q2 * prefix
        elif n == 3:
            return Q3 * prefix

        Pnm2, Pnm1 = P2, P3
        Qnm1 = Q3
        min_n = 4
    else:
        Pnm2, Pnm1 = P0, P1
        Qnm1 = Q1
        min_n = 2

    for nn in range(min_n, n+1):
        A, B, C = abc_q2d(nn-1, m)
        Pn = (A + B * x) * Pnm1 - C * Pnm2

        gnm1 = g_q2d(nn-1, m)
        fn = f_q2d(nn, m)
        Qn = (Pn - gnm1 * Qnm1) * (1/fn)

        Pnm2, Pnm1 = Pnm1, Pn
        Qnm1 = Qn

    # flake8 can't prove that the branches above the loop guarantee that we
    # enter the loop and Qn is defined
    return Qn * prefix  # NOQA


def Q2d_sequence(nms, r, t):
    """Sequence of 2D-Q polynomials.

    Parameters
    ----------
    nms : iterable of tuple
        (n,m) for each desired term
    r : numpy.ndarray
        radial coordinates
    t : numpy.ndarray
        azimuthal coordinates

    Returns
    -------
    generator
        yields one term for each element of nms

    """
    # see Q2d for general sense of this algorithm.
    # the way this one works is to compute the maximum N for each |m|, and then
    # compute the recurrence for each of those sequences and storing it.  A loop
    # is then iterated over the input nms, and selected value with appropriate
    # prefixes / other terms yielded.

    u = r
    x = u ** 2

    def factory():
        return 0

    # maps |m| => N
    m_has_pos = set()
    m_has_neg = set()
    max_ns = defaultdict(factory)
    for n, m in nms:
        m_ = abs(m)
        if max_ns[m_] < n:
            max_ns[m_] = n
        if m > 0:
            m_has_pos.add(m_)
        else:
            m_has_neg.add(m_)

    # precompute these reusable pieces of data
    u_scales = {}
    sin_scales = {}
    cos_scales = {}

    for absm in max_ns.keys():
        u_scales[absm] = u ** absm
        if absm in m_has_neg:
            sin_scales[absm] = np.sin(absm * t)
        if absm in m_has_pos:
            cos_scales[absm] = np.cos(absm * t)

    sequences = {}
    for m, N in max_ns.items():
        if m == 0:
            sequences[m] = list(Qbfs_sequence(range(N+1), r))
        else:
            sequences[m] = []
            P0 = 1/2
            if m == 1:
                P1 = 1 - x/2
            else:
                P1 = (m - .5) + (1 - m) * x

            f0 = f_q2d(0, m)
            Q0 = 1 / (2 * f0)
            sequences[m].append(Q0)
            if N == 0:
                continue

            g0 = g_q2d(0, m)
            f1 = f_q2d(1, m)
            Q1 = (P1 - g0 * Q0) * (1/f1)
            sequences[m].append(Q1)
            if N == 1:
                continue
            # everything above here works, or at least everything in the returns works
            if m == 1:
                P2 = (3 - x * (12 - 8 * x)) / 6
                P3 = (5 - x * (60 - x * (120 - 64 * x))) / 10

                g1 = g_q2d(1, m)
                f2 = f_q2d(2, m)
                Q2 = (P2 - g1 * Q1) * (1/f2)

                g2 = g_q2d(2, m)
                f3 = f_q2d(3, m)
                Q3 = (P3 - g2 * Q2) * (1/f3)
                sequences[m].append(Q2)
                sequences[m].append(Q3)
                # Q2, Q3 correct
                if N <= 3:
                    continue
                Pnm2, Pnm1 = P2, P3
                Qnm1 = Q3
                min_n = 4
            else:
                Pnm2, Pnm1 = P0, P1
                Qnm1 = Q1
                min_n = 2

            for nn in range(min_n, N+1):
                A, B, C = abc_q2d(nn-1, m)
                Pn = (A + B * x) * Pnm1 - C * Pnm2

                gnm1 = g_q2d(nn-1, m)
                fn = f_q2d(nn, m)
                Qn = (Pn - gnm1 * Qnm1) * (1/fn)
                sequences[m].append(Qn)

                Pnm2, Pnm1 = Pnm1, Pn
                Qnm1 = Qn

    for n, m in nms:
        if m != 0:
            if m < 0:
                # m < 0, double neg = pos
                prefix = sin_scales[-m] * u_scales[-m]
            else:
                prefix = cos_scales[m] * u_scales[m]

            yield sequences[abs(m)][n] * prefix
        else:
            yield sequences[0][n]
