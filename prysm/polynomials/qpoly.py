"""Tools for working with Q (Forbes) polynomials."""
from collections import defaultdict
from functools import lru_cache

from scipy.special import factorial, factorial2

from .jacobi import (
    jacobi,
    jacobi_with_der,
    jacobi_seq,
    jacobi_seq_with_der,
    jacobi_sum_clenshaw_der,
)
from ._clenshaw import _initialize_alphas, _clenshaw_sum, _clenshaw_sum_der

from prysm.mathops import np, kronecker, gamma, sign
from prysm.conf import config


_INV_SQRT19 = 1.0 / np.sqrt(19)


def _trim_trailing_zeros(coefs):
    """Drop trailing exact-zero coefficients from a dense coefficient vector."""
    if coefs is None:
        return []
    if not hasattr(coefs, '__len__'):
        coefs = list(coefs)

    n = len(coefs)
    while n > 0 and coefs[n - 1] == 0:
        n -= 1

    if n == 0:
        return []
    return coefs[:n]


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
    ndarray
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
    rho = x * x
    # c_Q is the leading term used to convert Qm to Qbfs
    c_Q = rho * (1 - rho)
    if n == 0:
        return c_Q  # == x^2 - x^4

    if n == 1:
        return _INV_SQRT19 * (13 - 16 * rho) * c_Q

    # c is the leading term of the recurrence relation for P
    c = 2 - 4 * rho
    # P0, P1 are the first two terms of the recurrence relation for auxiliary
    # polynomial P_n
    P0 = 2
    P1 = 6 - 8 * rho
    Pnm2 = P0
    Pnm1 = P1

    # Q0, Q1 are the first two terms of the recurrence relation for Qm
    Q0 = 1
    Q1 = _INV_SQRT19 * (13 - 16 * rho)
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
# so, general process... for Qbfs, don't provide derivatives, but provide a way
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
        seq of polynomial coefficients, from order n=0..len(cs)-1

    Returns
    -------
    ndarray
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
    if M == 0:
        return bs

    g = g_qbfs(M-1)
    f = f_qbfs(M-1)
    bs[M-1] = (cs[M-1] - g * bs[M])/f
    for i in range(M-2, -1, -1):
        g = g_qbfs(i)
        h = h_qbfs(i)
        f = f_qbfs(i)
        bs[i] = (cs[i] - g * bs[i+1] - h*bs[i+2])/f

    return bs


def clenshaw_qbfs(cs, usq, alphas=None):
    """Use Clenshaw's method to compute a Qbfs surface from its coefficients.

    Parameters
    ----------
    cs : iterable of float
        coefficients for a Qbfs surface, from order 0..len(cs)-1
    usq : ndarray
        radial coordinate(s) to evaluate, squared, notionally in the range [0,1]
        the variable u^2 from oe-18-19-19700
    alphas : ndarray, optional
        array to store the alpha sums in,
        the surface is u^2(1-u^2) times 2 times (alphas[0]+alphas[1])
        if not None, alphas should be of shape (len(s), x.shape)
        see _initialize_alphas if you desire more information

    Returns
    -------
    ndarray
        the alphas array; the surface sag, the quantity u^2(1-u^2) S(u^2)
        from Eq. (3.13), is u^2(1-u^2) times 2 times (alphas[0]+alphas[1]),
        see compute_z_Qbfs for that assembly
        note: excludes the division by phi, since c and rho are unknown

    """
    x = usq
    cs = _trim_trailing_zeros(cs)
    if len(cs) == 0:
        alphas = _initialize_alphas([0], x, alphas, j=0)
        alphas[...] = 0
        return alphas

    bs = change_basis_Qbfs_to_Pn(cs)
    alphas = _initialize_alphas(cs, x, alphas, j=0)
    # Qbfs recurrence: P_n = (2 - 4x) P_{n-1} - P_{n-2}; lin is n-independent,
    # c is 1.
    prefix = 2 - 4 * x
    _clenshaw_sum(bs, lambda n: prefix, lambda n: 1, alphas)
    return alphas


def clenshaw_qbfs_der(cs, usq, j=1, alphas=None):
    """Use Clenshaw's method to compute Nth order derivatives of a sum of Qbfs polynomials.

    Excludes base sphere and u^2(1-u^2) prefix

    As an end-user, you are likely more interested in compute_zprime_Qbfs.

    Parameters
    ----------
    cs : iterable of float
        coefficients for a Qbfs surface, from order 0..len(cs)-1
    usq : ndarray
        radial coordinate(s) to evaluate, squared, notionally in the range [0,1]
        the variable u^2 from oe-18-19-19700
    j : int
        derivative order
    alphas : ndarray, optional
        array to store the alpha sums in,
        if x = u * u, then
        S   = (x times (1 - x)) times 2 times (alphas[0][0] + alphas[0][1])
        S'  = ... .. the same, but alphas[1][0] and alphas[1][1]
        S'' = ... ... ... ... ... ... [2][0] ... ... ..[1][1]
        etc

        if not None, alphas should have shape (j+1, len(cs)) followed by x.shape
        see _initialize_alphas if you desire more information

    Returns
    -------
    ndarray
        the alphas array

    """
    x = usq
    cs = _trim_trailing_zeros(cs)
    if len(cs) == 0:
        alphas = _initialize_alphas([0], x, alphas, j=j)
        alphas[...] = 0
        return alphas

    bs = change_basis_Qbfs_to_Pn(cs)
    alphas = _initialize_alphas(cs, x, alphas, j=j)
    prefix = 2 - 4 * x
    _clenshaw_sum_der(
        bs,
        lambda n: prefix,        # lin_n(x)        = 2 - 4x (n-independent)
        lambda n: -4,            # coefficient of x in lin_n
        lambda n: 1,             # c_n
        alphas,
        j,
    )
    return alphas


def product_rule(u, v, du, dv):
    """The product rule of calculus, d/dx uv = u dv v du."""
    return u * dv + v * du


def compute_z_zprime_Qbfs(coefs, u, usq):
    """Compute the surface sag and first radial derivative of a Qbfs surface.

    Excludes base sphere.

    from Eq. 3.13 and 3.14 of oe-18-19-19700.

    Parameters
    ----------
    coefs : iterable
        surface coefficients for Q0..QN, N=len(coefs)-1
    u : ndarray
        normalized radial coordinates (rho/rho_max)
    usq : ndarray
        u^2
    c : float
        best fit sphere curvature
        use c=0 for a flat base surface

    Returns
    -------
    ndarray, ndarray
        S, Sprime in Forbes' parlance

    """
    coefs = _trim_trailing_zeros(coefs)
    if len(coefs) == 0:
        return np.zeros_like(u), np.zeros_like(u)

    # clenshaw does its own u^2; _initialize_alphas pads the mode axis to
    # at least 2, so alphas[*][1] is always readable (zero for len==1).
    alphas = clenshaw_qbfs_der(coefs, usq, j=1)
    S = 2 * (alphas[0][0] + alphas[0][1])
    # Sprime should be two times the alphas, just like S, but as a performance
    # optimization, S = sum cn Qn u^2
    # we're doing d/du, so a prefix of 2u comes in front
    # and 2*u * (2 * alphas)
    # = 4*u*alphas
    # = do two in-place muls on Sprime for speed
    Sprime = alphas[1][0] + alphas[1][1]
    Sprime *= 4
    Sprime *= u

    prefix = usq * (1 - usq)
    #                        u3
    dprefix = 2 * u - 4 * (usq * u)
    u = prefix
    du = dprefix
    v = S
    dv = Sprime
    Sprime = product_rule(u, v, du, dv)
    S *= prefix
    return S, Sprime


def compute_z_Qbfs(coefs, u, usq):
    """Sag-only sibling of compute_z_zprime_Qbfs.

    Assembles the surface u^2 (1 - u^2) S(u^2) from the alpha sums returned
    by clenshaw_qbfs, mirroring compute_z_zprime_Qbfs's signature.

    """
    alphas = clenshaw_qbfs(coefs, usq)
    # alphas[1] is zero for the len-1 case (mode axis is padded to >=2 by
    # _initialize_alphas), so the formula collapses correctly.
    return (usq * (1 - usq)) * (2 * (alphas[0] + alphas[1]))


def compute_z_zprime_Qcon(coefs, u, usq):
    """Compute the surface sag and first radial derivative of a Qcon surface.

    Excludes base sphere.

    from Eq. 5.3 and 5.3 of oe-18-13-13851.

    Parameters
    ----------
    coefs : iterable
        surface coefficients for Q0..QN, N=len(coefs)-1
    u : ndarray
        normalized radial coordinates (rho/rho_max)
    usq : ndarray
        u^2

    Returns
    -------
    ndarray, ndarray
        S, Sprime in Forbes' parlance

    """
    coefs = _trim_trailing_zeros(coefs)
    if len(coefs) == 0:
        return np.zeros_like(u), np.zeros_like(u)

    x = 2 * usq - 1
    alphas = jacobi_sum_clenshaw_der(coefs, 0, 4, x=x, j=1)
    S = alphas[0][0]
    Sprime = alphas[1][0]
    Sprime *= 4  # this 4 u is not the same 4u as Qbfs, 4u in Qbfs is a
    Sprime *= u  # composition of 2*alphas and 2u, this is just der of x=2usq - 1

    # u^4
    prefix = usq * usq
    # 4u^3
    dprefix = 4 * (usq * u)
    u = prefix
    du = dprefix
    v = S
    dv = Sprime
    Sprime = product_rule(u, v, du, dv)
    S *= prefix
    return S, Sprime


def Qbfs_seq(ns, x):
    """Qbfs polynomials of orders ns at point(s) x.

    Parameters
    ----------
    ns : Iterable of int
        polynomial orders
    x : numpy.array
        point(s) at which to evaluate

    Returns
    -------
    ndarray
        has shape (len(ns),) followed by x.shape
        e.g., for 5 modes and x of dimension 100x100,
        return has shape (5, 100, 100)

    """
    # see the leading comment of Qbfs for some explanation of this code
    # and prysm:jacobi.py#jacobi_seq the "_seq" portion

    if not hasattr(ns, '__len__'):
        ns = list(ns)
    if len(ns) == 0:
        return np.empty((0, *x.shape), dtype=x.dtype)
    min_i = 0
    out = np.empty((len(ns), *x.shape), dtype=x.dtype)

    rho = x * x
    # c_Q is the leading term used to convert Qm to Qbfs
    c_Q = rho * (1 - rho)
    if ns[min_i] == 0:
        out[min_i] = c_Q
        min_i += 1

    if min_i == len(ns):
        return out

    if ns[min_i] == 1:
        out[min_i] = _INV_SQRT19 * (13 - 16 * rho) * c_Q
        min_i += 1

    if min_i == len(ns):
        return out

    # c is the leading term of the recurrence relation for P
    c = 2 - 4 * rho
    # P0, P1 are the first two terms of the recurrence relation for auxiliary
    # polynomial P_n
    P0 = 2
    P1 = 6 - 8 * rho
    Pnm2 = P0
    Pnm1 = P1

    # Q0, Q1 are the first two terms of the recurrence relation for Qbfs_n
    Q0 = 1
    Q1 = _INV_SQRT19 * (13 - 16 * rho)
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
            out[min_i] = Qn * c_Q
            min_i += 1

    return out


def Qbfs_der(n, x):
    """Partial derivative w.r.t. x of the Qbfs polynomial of order n.

    Uses the parallel auxiliary recurrence on y = x^2 to evaluate both
    Q_n(y) and Q'_n(y) = dQ_n/dy, then chains through the leading
    x^2(1 - x^2) envelope:

        dQbfs_n/dx = (2x - 4x^3) Q_n(x^2) + (2x^3 - 2x^5) Q'_n(x^2)

    Parameters
    ----------
    n : int
        polynomial order
    x : ndarray
        point(s) at which to evaluate, notionally in [0, 1]

    Returns
    -------
    ndarray
        d/dx Qbfs_n(x)

    """
    rho = x * x
    # leading-envelope derivative pieces, reused at the bottom
    env = rho * (1 - rho)
    denv_dx = 2*x - 4*x*rho  # 2x - 4x^3

    Q_list, dQ_list = _qbfs_aux_recurrence(n, rho)
    return denv_dx * Q_list[n] + env * (2 * x) * dQ_list[n]


def Qbfs_der_seq(ns, x):
    """Partial derivative w.r.t. x of Qbfs polynomials of orders ns.

    Companion to Qbfs_seq; see Qbfs_der for the derivation.

    Parameters
    ----------
    ns : Iterable of int
        polynomial orders (assumed sorted ascending)
    x : ndarray
        point(s) at which to evaluate

    Returns
    -------
    ndarray
        has shape (len(ns),) followed by x.shape; the i-th plane is d/dx Qbfs_{ns[i]}(x)

    """
    if not hasattr(ns, '__len__'):
        ns = list(ns)
    if len(ns) == 0:
        return np.empty((0, *x.shape), dtype=x.dtype)

    rho = x * x
    env = rho * (1 - rho)
    denv_dx = 2*x - 4*x*rho
    two_x = 2 * x

    Q_list, dQ_list = _qbfs_aux_recurrence(ns[-1], rho)
    out = np.empty((len(ns), *x.shape), dtype=x.dtype)
    for i, n in enumerate(ns):
        out[i] = denv_dx * Q_list[n] + env * two_x * dQ_list[n]
    return out


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
    ndarray
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
    x2 = x * x
    xx = 2 * x2 - 1
    Pn = jacobi(n, 0, 4, xx)
    return Pn * x2 * x2


def Qcon_seq(ns, x):
    """Qcon polynomials of orders ns at point(s) x.

    Parameters
    ----------
    ns : Iterable of int
        polynomial orders
    x : numpy.array
        point(s) at which to evaluate

    Returns
    -------
    ndarray
        has shape (len(ns),) followed by x.shape
        e.g., for 5 modes and x of dimension 100x100,
        return has shape (5, 100, 100)

    """
    x2 = x * x
    xx = 2 * x2 - 1
    x4 = x2 * x2
    Pns = jacobi_seq(ns, 0, 4, xx)
    return Pns * x4


def Qcon_der(n, x):
    """Partial derivative w.r.t. x of the Qcon polynomial of order n.

    Qcon_n(x) = x^4 * P_n(2x^2 - 1) where P_n is the Jacobi polynomial with
    alpha=0, beta=4.  Differentiating::

        dQcon_n/dx = 4 x^3 P_n(2x^2 - 1) + 4 x^5 P'_n(2x^2 - 1)

    Parameters
    ----------
    n : int
        polynomial order
    x : ndarray
        point(s) at which to evaluate, notionally in [0, 1]

    Returns
    -------
    ndarray
        d/dx Qcon_n(x)

    """
    xx = 2 * x * x - 1
    x3 = x * x * x
    Pn, dPn = jacobi_with_der(n, 0, 4, xx)
    return 4 * x3 * Pn + 4 * x3 * (x * x) * dPn


def Qcon_der_seq(ns, x):
    """Partial derivative w.r.t. x of Qcon polynomials of orders ns.

    Companion to Qcon_seq; see Qcon_der for the derivation.

    Parameters
    ----------
    ns : Iterable of int
        polynomial orders (assumed sorted ascending)
    x : ndarray
        point(s) at which to evaluate

    Returns
    -------
    ndarray
        has shape (len(ns),) followed by x.shape; the i-th plane is d/dx Qcon_{ns[i]}(x)

    """
    xx = 2 * x * x - 1
    x3 = x * x * x
    x5 = x3 * x * x
    Pns, dPns = jacobi_seq_with_der(ns, 0, 4, xx)
    return 4 * x3 * Pns + 4 * x5 * dPns


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
        num = factorial2(2 * m - 1)
        den = 2 ** (m + 1) * factorial(m - 1)
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
    if n == 0 and m == 1:
        return 0.25
    if n == 0:
        num = m ** 2 * factorial2(2 * m - 3)
        den = 2 ** (m + 1) * factorial(m - 1)
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
    r : ndarray
        radial coordinate, slope orthogonal in [0,1]
    t : ndarray
        azimuthal coordinate, radians

    Returns
    -------
    ndarray
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
    x = u * u
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
    if m == 1:
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


def Q2d_seq(nms, r, t):
    """Seq of 2D-Q polynomials.

    Parameters
    ----------
    nms : iterable of tuple
        (n,m) for each desired term
    r : ndarray
        radial coordinates
    t : ndarray
        azimuthal coordinates

    Returns
    -------
    ndarray
        has shape (len(ns),) followed by x.shape
        e.g., for 5 modes and x of dimension 100x100,
        return has shape (5, 100, 100)

    """
    # see Q2d for general sense of this algorithm.
    # the way this one works is to compute the maximum N for each |m|, and then
    # compute the recurrence for each of those seqs and storing it.  A loop
    # is then iterated over the input nms, and selected value with appropriate
    # prefixes / other terms yielded.

    u = r
    x = u * u

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
        elif m < 0:
            m_has_neg.add(m_)

    # precompute these reusable pieces of data
    u_scales = {}
    sin_scales = {}
    cos_scales = {}

    for absm in max_ns.keys():
        if absm == 0:
            continue
        u_scales[absm] = u ** absm
        if absm in m_has_neg:
            sin_scales[absm] = np.sin(absm * t)
        if absm in m_has_pos:
            cos_scales[absm] = np.cos(absm * t)

    seqs = {}
    for m, N in max_ns.items():
        if m == 0:
            seqs[m] = Qbfs_seq(range(N+1), r)
        else:
            seqs[m] = []
            P0 = 1/2
            if m == 1:
                P1 = 1 - x/2
            else:
                P1 = (m - .5) + (1 - m) * x

            f0 = f_q2d(0, m)
            Q0 = 1 / (2 * f0)
            seqs[m].append(Q0)
            if N == 0:
                continue

            g0 = g_q2d(0, m)
            f1 = f_q2d(1, m)
            Q1 = (P1 - g0 * Q0) * (1/f1)
            seqs[m].append(Q1)
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
                seqs[m].append(Q2)
                seqs[m].append(Q3)
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
                seqs[m].append(Qn)

                Pnm2, Pnm1 = Pnm1, Pn
                Qnm1 = Qn

    j = 0
    out = np.empty((len(nms), *x.shape), dtype=x.dtype)
    for n, m in nms:
        if m != 0:
            if m < 0:
                # m < 0, double neg = pos
                prefix = sin_scales[-m] * u_scales[-m]
            else:
                prefix = cos_scales[m] * u_scales[m]

            out[j] = seqs[abs(m)][n] * prefix
            j += 1
        else:
            out[j] = seqs[0][n]
            j += 1

    return out


def _qbfs_aux_recurrence(Nmax, u):
    """Tables of the auxiliary Qbfs polynomial Q_n(u) and dQ_n/du.

    These are the unprefixed Qbfs polynomials; i.e., Qbfs_n(x) =
    x^2(1-x^2) times Q_n(x^2). Used by Q2d Cartesian derivatives when m=0.

    Returns
    -------
    list, list
        Q[0..Nmax] and dQ[0..Nmax]; each entry is an ndarray broadcast to
        the shape of u.

    """
    Q0 = np.ones_like(u)
    dQ0 = np.zeros_like(u)
    Q_list = [Q0]
    dQ_list = [dQ0]
    if Nmax == 0:
        return Q_list, dQ_list

    Q1 = _INV_SQRT19 * (13 - 16 * u)
    dQ1 = -16 * _INV_SQRT19
    Q_list.append(Q1)
    dQ_list.append(dQ1)
    if Nmax == 1:
        return Q_list, dQ_list

    # parallel P / P' / Q / Q' recurrence on u
    P_prev = 2.0
    P_curr = 6 - 8 * u
    dP_prev = 0.0
    dP_curr = -8.0
    Q_prev = Q0
    Q_curr = Q1
    dQ_prev = dQ0
    dQ_curr = dQ1
    prefix = 2 - 4 * u
    for nn in range(2, Nmax + 1):
        Pn = prefix * P_curr - P_prev
        dPn = -4 * P_curr + prefix * dP_curr - dP_prev
        g = g_qbfs(nn - 1)
        h = h_qbfs(nn - 2)
        inv_f = 1 / f_qbfs(nn)
        Qn = (Pn - g * Q_curr - h * Q_prev) * inv_f
        dQn = (dPn - g * dQ_curr - h * dQ_prev) * inv_f
        P_prev, P_curr = P_curr, Pn
        dP_prev, dP_curr = dP_curr, dPn
        Q_prev, Q_curr = Q_curr, Qn
        dQ_prev, dQ_curr = dQ_curr, dQn
        Q_list.append(Qn)
        dQ_list.append(dQn)

    return Q_list, dQ_list


def _q2d_radial_recurrence(Nmax, m, u):
    """Tables of Q_n^m(u) and dQ_n^m/du for n=0..Nmax, m >= 1.

    Where u = rho^2 in Forbes' parlance. This computes the radial part
    without the u^m or trig prefix.

    Returns
    -------
    list, list
        Q[0..Nmax] and dQ[0..Nmax]; each entry is an ndarray broadcast to
        the shape of u.

    """
    if m < 1:
        raise ValueError(f'_q2d_radial_recurrence requires m >= 1, got {m}')

    f0 = f_q2d(0, m)
    Q_prev = np.full_like(u, 1 / (2 * f0))
    dQ_prev = np.zeros_like(u)
    Q_list = [Q_prev]
    dQ_list = [dQ_prev]
    if Nmax == 0:
        return Q_list, dQ_list

    P_prev = np.full_like(u, 0.5)         # P_0 = 1/2
    dP_prev = np.zeros_like(u)
    if m == 1:
        P_curr = 1 - u / 2
        dP_curr = np.full_like(u, -0.5)
    else:
        P_curr = (m - 0.5) + (1 - m) * u
        dP_curr = np.full_like(u, 1 - m)

    g0 = g_q2d(0, m)
    f1 = f_q2d(1, m)
    inv_f1 = 1 / f1
    Q_curr = (P_curr - g0 * Q_prev) * inv_f1
    dQ_curr = (dP_curr - g0 * dQ_prev) * inv_f1
    Q_list.append(Q_curr)
    dQ_list.append(dQ_curr)
    if Nmax == 1:
        return Q_list, dQ_list

    if m == 1:
        # hardcoded P_2, P_3 mirror Q2d
        P2 = (3 - u * (12 - 8 * u)) / 6
        dP2 = (-12 + 16 * u) / 6
        g1 = g_q2d(1, 1)
        f2 = f_q2d(2, 1)
        inv_f2 = 1 / f2
        Q2 = (P2 - g1 * Q_curr) * inv_f2
        dQ2 = (dP2 - g1 * dQ_curr) * inv_f2
        Q_list.append(Q2)
        dQ_list.append(dQ2)
        if Nmax == 2:
            return Q_list, dQ_list

        P3 = (5 - u * (60 - u * (120 - 64 * u))) / 10
        dP3 = (-60 + u * (240 - 192 * u)) / 10
        g2 = g_q2d(2, 1)
        f3 = f_q2d(3, 1)
        inv_f3 = 1 / f3
        Q3 = (P3 - g2 * Q2) * inv_f3
        dQ3 = (dP3 - g2 * dQ2) * inv_f3
        Q_list.append(Q3)
        dQ_list.append(dQ3)
        if Nmax == 3:
            return Q_list, dQ_list

        P_prev, P_curr = P2, P3
        dP_prev, dP_curr = dP2, dP3
        Q_curr = Q3
        dQ_curr = dQ3
        start_n = 4
    else:
        start_n = 2

    for nn in range(start_n, Nmax + 1):
        A, B, C = abc_q2d(nn - 1, m)
        Pn = (A + B * u) * P_curr - C * P_prev
        dPn = B * P_curr + (A + B * u) * dP_curr - C * dP_prev
        gnm1 = g_q2d(nn - 1, m)
        fn = f_q2d(nn, m)
        inv_fn = 1 / fn
        Qn = (Pn - gnm1 * Q_curr) * inv_fn
        dQn = (dPn - gnm1 * dQ_curr) * inv_fn
        P_prev, P_curr = P_curr, Pn
        dP_prev, dP_curr = dP_curr, dPn
        Q_curr = Qn
        dQ_curr = dQn
        Q_list.append(Qn)
        dQ_list.append(dQn)

    return Q_list, dQ_list


def _harmonic_powers(am, x, y):
    """Real and imaginary parts of (x + i y)^k for k=0..am.

    Returns
    -------
    list of (C_k, S_k)
        out[k] = (Re((x+iy)^k), Im((x+iy)^k)) for k in 0..am.

    """
    C0 = np.ones_like(x)
    S0 = np.zeros_like(x)
    out = [(C0, S0)]
    C_prev, S_prev = C0, S0
    for _ in range(am):
        C_new = x * C_prev - y * S_prev
        S_new = x * S_prev + y * C_prev
        out.append((C_new, S_new))
        C_prev, S_prev = C_new, S_new
    return out


def Q2d_der(n, m, r, t):
    """Polar partial derivatives of the 2D Q polynomial Q2d_n^m.

    Parameters
    ----------
    n : int
        radial order
    m : int
        azimuthal order (sign controls cosine vs sine prefix; m=0 is the
        purely-radial Qbfs case)
    r : ndarray
        radial coordinate in [0, 1]
    t : ndarray
        azimuthal coordinate, radians

    Returns
    -------
    ndarray, ndarray
        d/dr Q2d_n^m, d/dt Q2d_n^m

    """
    if m == 0:
        # Q2d for m=0 reduces to Qbfs(n, r) which has no t-dependence
        return Qbfs_der(n, r), np.zeros_like(r * t)

    u = r * r
    am = abs(m)
    Q_list, dQ_list = _q2d_radial_recurrence(n, am, u)
    Q = Q_list[n]
    dQdu = dQ_list[n]

    if m > 0:
        trig = np.cos(am * t)
        trig_der = -am * np.sin(am * t)
    else:
        trig = np.sin(am * t)
        trig_der = am * np.cos(am * t)

    # F(r) = r^am * Q(r^2);  F'(r) = am r^(am-1) Q + 2 r^(am+1) Q'(r^2)
    if am == 1:
        r_am_minus_1 = np.ones_like(r)
        r_am = r
    else:
        r_am_minus_1 = r ** (am - 1)
        r_am = r_am_minus_1 * r

    F = r_am * Q
    Fp = am * r_am_minus_1 * Q + 2 * r_am * r * dQdu

    return trig * Fp, trig_der * F


def Q2d_der_xy(n, m, x, y):
    """Cartesian partial derivatives of the 2D Q polynomial Q2d_n^m.

    Computed directly in (x, y) via the harmonic decomposition
    r^abs(m) cos(m t) = Re((x + i y)^abs(m)) (and Im for m<0), so the result is
    smooth at the origin with no 1/r singularity.

    Parameters
    ----------
    n : int
        radial order
    m : int
        azimuthal order
    x : ndarray
        Cartesian x coordinate (same normalization as r in Q2d)
    y : ndarray
        Cartesian y coordinate

    Returns
    -------
    ndarray, ndarray
        dQ/dx, dQ/dy

    """
    rho_sq = x * x + y * y
    am = abs(m)

    if m == 0:
        Q_list, dQ_list = _qbfs_aux_recurrence(n, rho_sq)
        Q = Q_list[n]
        dQdu = dQ_list[n]
        u = rho_sq
        env = u * (1 - u)
        denv_du = 1 - 2 * u
        common = denv_du * Q + env * dQdu  # d(sag)/du
        return 2 * x * common, 2 * y * common

    Q_list, dQ_list = _q2d_radial_recurrence(n, am, rho_sq)
    J = Q_list[n]
    Jp = dQ_list[n]

    harm = _harmonic_powers(am, x, y)
    C_am, S_am = harm[am]
    C_amm1, S_amm1 = harm[am - 1]
    if m > 0:
        H = C_am
        dHdx = am * C_amm1
        dHdy = -am * S_amm1
    else:
        H = S_am
        dHdx = am * S_amm1
        dHdy = am * C_amm1

    # d/dx J(rho^2) = Jp * 2x; similarly for y
    return 2 * x * Jp * H + J * dHdx, 2 * y * Jp * H + J * dHdy


def Q2d_der_seq(nms, r, t):
    """Polar partial derivatives of a sequence of Q2d polynomials.

    Companion to Q2d_seq; per-m recurrence is shared across all radial
    orders for that m.

    Parameters
    ----------
    nms : iterable of tuple
        (n, m) for each desired term
    r : ndarray
        radial coordinates
    t : ndarray
        azimuthal coordinates

    Returns
    -------
    ndarray, ndarray
        arrays of shape (len(nms),) followed by r.shape; the first is d/dr, the second
        is d/dt, in the same order as nms

    """
    u = r * r

    # Plan per-m work: max radial order for each |m|, and which signs occur
    m_has_pos = set()
    m_has_neg = set()
    max_ns = defaultdict(int)
    for n, m in nms:
        am = abs(m)
        if max_ns[am] < n:
            max_ns[am] = n
        if m > 0:
            m_has_pos.add(am)
        elif m < 0:
            m_has_neg.add(am)

    # Precompute trig and radial-prefix tables per |m|
    cos_table = {}
    sin_table = {}
    cos_der_table = {}
    sin_der_table = {}
    r_am_table = {}        # r^am
    r_am_minus_1_table = {}  # r^(am-1)
    for am in max_ns:
        if am == 0:
            continue
        if am in m_has_pos:
            cos_table[am] = np.cos(am * t)
            sin_der_table[am] = -am * np.sin(am * t)
        if am in m_has_neg:
            sin_table[am] = np.sin(am * t)
            cos_der_table[am] = am * np.cos(am * t)
        if am == 1:
            r_am_minus_1_table[am] = np.ones_like(r)
            r_am_table[am] = r
        else:
            r_am_minus_1_table[am] = r ** (am - 1)
            r_am_table[am] = r_am_minus_1_table[am] * r

    # Per-|m| recurrence
    Q_tables = {}
    dQ_tables = {}
    qbfs_der_table = None
    for am, Nmax in max_ns.items():
        if am == 0:
            qbfs_der_table = Qbfs_der_seq(range(Nmax + 1), r)
        else:
            Q_tables[am], dQ_tables[am] = _q2d_radial_recurrence(Nmax, am, u)

    out_dr = np.empty((len(nms), *r.shape), dtype=r.dtype)
    out_dt = np.empty((len(nms), *r.shape), dtype=r.dtype)
    for j, (n, m) in enumerate(nms):
        if m == 0:
            out_dr[j] = qbfs_der_table[n]
            out_dt[j] = 0
            continue
        am = abs(m)
        Q = Q_tables[am][n]
        dQdu = dQ_tables[am][n]
        r_am = r_am_table[am]
        r_am_minus_1 = r_am_minus_1_table[am]
        F = r_am * Q
        Fp = am * r_am_minus_1 * Q + 2 * r_am * r * dQdu
        if m > 0:
            out_dr[j] = cos_table[am] * Fp
            out_dt[j] = sin_der_table[am] * F
        else:
            out_dr[j] = sin_table[am] * Fp
            out_dt[j] = cos_der_table[am] * F

    return out_dr, out_dt


def Q2d_der_xy_seq(nms, x, y):
    """Cartesian partial derivatives of a sequence of Q2d polynomials.

    Companion to Q2d_der_xy; per-m recurrence and per-am harmonic powers
    are shared across all radial orders for that m.

    Parameters
    ----------
    nms : iterable of tuple
        (n, m) for each desired term
    x : ndarray
        Cartesian x coordinate
    y : ndarray
        Cartesian y coordinate

    Returns
    -------
    ndarray, ndarray
        arrays of shape (len(nms),) followed by x.shape; the first is d/dx, the second
        is d/dy, in the same order as nms

    """
    rho_sq = x * x + y * y

    max_ns = defaultdict(int)
    for n, m in nms:
        am = abs(m)
        if max_ns[am] < n:
            max_ns[am] = n

    # Per-|m| recurrence
    Q_tables = {}
    dQ_tables = {}
    for am, Nmax in max_ns.items():
        if am == 0:
            Q_tables[0], dQ_tables[0] = _qbfs_aux_recurrence(Nmax, rho_sq)
        else:
            Q_tables[am], dQ_tables[am] = _q2d_radial_recurrence(Nmax, am, rho_sq)

    # Harmonic powers up to the largest am
    am_max = max(max_ns) if max_ns else 0
    harm = _harmonic_powers(am_max, x, y) if am_max > 0 else None

    # m=0 envelope pieces (reused per (n, 0))
    if 0 in max_ns:
        u = rho_sq
        env = u * (1 - u)
        denv_du = 1 - 2 * u

    out_dx = np.empty((len(nms), *x.shape), dtype=x.dtype)
    out_dy = np.empty((len(nms), *x.shape), dtype=x.dtype)
    for j, (n, m) in enumerate(nms):
        am = abs(m)
        if m == 0:
            Q = Q_tables[0][n]
            dQdu = dQ_tables[0][n]
            common = denv_du * Q + env * dQdu
            out_dx[j] = 2 * x * common
            out_dy[j] = 2 * y * common
            continue
        J = Q_tables[am][n]
        Jp = dQ_tables[am][n]
        C_am, S_am = harm[am]  # NOQA - guarded by if m == 0 above
        C_amm1, S_amm1 = harm[am - 1]  # NOQA - guarded by if m == 0 above
        if m > 0:
            H = C_am
            dHdx = am * C_amm1
            dHdy = -am * S_amm1
        else:
            H = S_am
            dHdx = am * S_amm1
            dHdy = am * C_amm1
        out_dx[j] = 2 * x * Jp * H + J * dHdx
        out_dy[j] = 2 * y * Jp * H + J * dHdy

    return out_dx, out_dy


def change_of_basis_Q2d_to_Pnm(cns, m):
    """Perform the change of basis from Q_n^m to the auxiliary polynomial P_n^m.

    The auxiliary polynomial is defined in A.1 of oe-20-3-2483 and is the
    an unconventional variant of Jacobi polynomials.

    For terms where m=0, see change_basis_Qbfs_to_Pn.  This function only concerns
    those terms within the sum u^m a_n^m cos(mt) + b_n^m sin(mt) Q_n^m(u^2) sum

    Parameters
    ----------
    cns : iterable
        seq of polynomial coefficients, from order n=0..len(cs)-1 and a given
        m (not absolute m, but m, i.e. either "-2" or "+2" but not both)
    m : int
        azimuthal order

    Returns
    -------
    ndarray
        array of same type as cs holding the coefficients that represent the
        same surface as a sum of shifted Chebyshev polynomials of the third kind

    """
    if m < 0:
        m = -m

    cs = cns
    if hasattr(cs, 'dtype'):
        # array, initialize as array
        ds = np.empty_like(cs)
    else:
        # iterable input
        ds = np.empty(len(cs), dtype=config.precision)

    N = len(cs) - 1
    ds[N] = cs[N] / f_q2d(N, m)
    for n in range(N-1, -1, -1):
        ds[n] = (cs[n] - g_q2d(n, m) * ds[n+1]) / f_q2d(n, m)

    return ds


@lru_cache(4000)
def abc_q2d_clenshaw(n, m):
    """Special twist on A.3 for B.7."""
    # rewrite: 5 unique patches, easier to write each one as an if
    # had bugs trying to be more clever
    if m == 1:
        # left column
        if n == 0:
            return 2, -1, 0
        if n == 1:
            return -4/3, -8/3, -11/3
        if n == 2:
            return 9/5, -24/5, 0

    if m == 2 and n == 0:
        return 3, -2, 0

    if m == 3 and n == 0:
        return 5, -4, 0

    return abc_q2d(n, m)


def clenshaw_q2d(cns, m, usq, alphas=None):
    """Use Clenshaw's method to compute the alpha sums for a piece of a Q2D surface.

    Parameters
    ----------
    cns : iterable of float
        coefficients for a Qbfs surface, from order 0..len(cs)-1
    m : int
        azimuthal order for the cns
    usq : ndarray
        radial coordinate(s) to evaluate, squared, notionally in the range [0,1]
        the variable u^2 from oe-18-19-19700
    alphas : ndarray, optional
        array to store the alpha sums in,
        the surface is u^2(1-u^2) times 2 times (alphas[0]+alphas[1])
        if not None, alphas should have shape (len(s),) followed by x.shape
        see _initialize_alphas if you desire more information

    Returns
    -------
    alphas
        array containing components to compute the surface sag
        sum(cn Qn) is .5 alphas[0] - 2/5 alphas[3] if m=1 and N>2,
        and .5 alphas[0] otherwise.

    """
    x = usq
    cns = _trim_trailing_zeros(cns)
    if len(cns) == 0:
        alphas = _initialize_alphas([0], x, alphas, j=0)
        alphas[...] = 0
        return alphas

    ds = change_of_basis_Q2d_to_Pnm(cns, m)
    alphas = _initialize_alphas(ds, x, alphas, j=0)
    # Q2d recurrence: P_n = (A_n + B_n x) P_{n-1} - C_n P_{n-2} (Forbes notation;
    # do not swap A, B vs the paper — kept consistent with Forbes previously).

    def lin(n):
        A, B, _ = abc_q2d_clenshaw(n, m)
        return A + B * x

    def lin_x(n):
        return abc_q2d_clenshaw(n, m)[1]

    def c_fn(n):
        return abc_q2d_clenshaw(n, m)[2]

    _clenshaw_sum(ds, lin, c_fn, alphas)
    return alphas


def clenshaw_q2d_der(cns, m, usq, j=1, alphas=None):
    """Use Clenshaw's method to compute Nth order derivatives of a Q2D surface.

    This function is to be consumed by the other parts of prysm, and simply
    does the "alphas" computations (B.10) and adjacent Eqns

    See compute_zprime_Q2D for this calculation integrated

    Parameters
    ----------
    cns : iterable of float
        coefficients for a Qbfs surface, from order 0..len(cs)-1
    m : int
        azimuthal order
    usq : ndarray
        radial coordinate(s) to evaluate, squared, notionally in the range [0,1]
        the variable u from oe-18-19-19700
    j : int
        derivative order
    alphas : ndarray, optional
        array to store the alpha sums in,
        if not None, alphas should have shape (j+1, len(cs)) followed by x.shape
        see _initialize_alphas if you desire more information

    Returns
    -------
    ndarray
        the alphas array

    """
    x = usq
    cns = _trim_trailing_zeros(cns)
    if len(cns) == 0:
        alphas = _initialize_alphas([0], x, alphas, j=j)
        alphas[...] = 0
        return alphas

    ds = change_of_basis_Q2d_to_Pnm(cns, m)
    alphas = _initialize_alphas(cns, x, alphas, j=j)

    def lin(n):
        A, B, _ = abc_q2d_clenshaw(n, m)
        return A + B * x

    def lin_x(n):
        return abc_q2d_clenshaw(n, m)[1]

    def c_fn(n):
        return abc_q2d_clenshaw(n, m)[2]

    _clenshaw_sum_der(ds, lin, lin_x, c_fn, alphas, j)
    return alphas


def compute_z_zprime_Q2d(cm0, ams, bms, u, t):
    """Compute the surface sag and first radial and azimuthal derivative of a Q2D surface.

    Excludes base sphere.

    from Eq. 2.2 and Appendix B of oe-20-3-2483.

    Parameters
    ----------
    cm0 : iterable
        surface coefficients when m=0 (inside curly brace, top line, Eq. B.1)
        span n=0 .. len(cms)-1 and mus tbe fully dense
    ams : iterable of iterables
        ams[0] are the coefficients for the m=1 cosine terms,
        ams[1] for the m=2 cosines, and so on.  Same order n rules as cm0
    bms : iterable of iterables
        same as ams, but for the sine terms
        ams and bms must be the same length - that is, if an azimuthal order m
        is presnet in ams, it must be present in bms.  The azimuthal orders
        need not have equal radial expansions.

        For example, if ams extends to m=3, then bms must reach m=3
        but, if the ams for m=3 span n=0..5, it is OK for the bms to span n=0..3,
        or any other value, even just [0].
    u : ndarray
        normalized radial coordinates (rho/rho_max)
    t : ndarray
        azimuthal coordinate, in the range [0, 2pi]

    Returns
    -------
    ndarray, ndarray, ndarray
        surface sag, radial derivative of sag, azimuthal derivative of sag

    """
    usq = u * u
    z = np.zeros_like(u)
    dr = np.zeros_like(u)
    dt = np.zeros_like(u)

    cm0 = _trim_trailing_zeros(cm0)
    if len(cm0) > 0:
        zm0, zprimem0 = compute_z_zprime_Qbfs(cm0, u, usq)
        z += zm0
        dr += zprimem0

    # B.1
    # cos(mt)[sum a^m Q^m(u^2)] + sin(mt)[sum b^m Q^m(u^2)]
    #        ~~~~~~~~~~~~~~~~~~          ~~~~~~~~~~~~~~~~~~
    # variables:     Sa                           Sb
    # => because of am/bm going into Clenshaw's method, cannot
    # simplify, need to do the recurrence twice
    # u^m is outside the entire expression, think about that later
    m = 0
    # initialize to zero and incr at the front of the loop
    # to avoid putting an m += 1 at the bottom (too far from init)
    for a_coef, b_coef in zip(ams, bms):
        m += 1
        a_coef = _trim_trailing_zeros(a_coef)
        b_coef = _trim_trailing_zeros(b_coef)
        if len(a_coef) == 0 and len(b_coef) == 0:
            continue

        # can't use "as" => as keyword
        Na = len(a_coef) - 1
        Nb = len(b_coef) - 1
        Sa = 0
        Sb = 0
        Sprimea = 0
        Sprimeb = 0
        if len(a_coef) > 0:
            alphas_a = clenshaw_q2d_der(a_coef, m, usq)
            Sa = 0.5 * alphas_a[0][0]
            Sprimea = 0.5 * alphas_a[1][0]
        if len(b_coef) > 0:
            alphas_b = clenshaw_q2d_der(b_coef, m, usq)
            Sb = 0.5 * alphas_b[0][0]
            Sprimeb = 0.5 * alphas_b[1][0]
        if m == 1 and Na > 2:
            Sa -= 2/5 * alphas_a[0][3]
            # derivative is same, but instead of 0 index, index=j==1
            Sprimea -= 2/5 * alphas_a[1][3]
        if m == 1 and Nb > 2:
            Sb -= 2/5 * alphas_b[0][3]
            Sprimeb -= 2/5 * alphas_b[1][3]

        um = u ** m
        cost = np.cos(m*t)
        sint = np.sin(m*t)

        kernel = cost * Sa + sint * Sb
        total_sum = um * kernel

        z += total_sum

        # for the derivatives, we have two cases of the product rule:
        # between "cost" and Sa, and between "sint" and "Sb"
        # within each of those is a chain rule, just as for Zernike
        # then there is a final product rule for the outer term
        # differentiating in this way is just like for the classical asphere
        # equation; differentiate each power separately
        # if F(x) = S(x^2), then
        # d/dx(cos(m * t) * Fx) = 2x F'(x^2) cos(mt)
        # with u^m in front, taken to its conclusion
        # F = Sa, G = Sb
        # d/dx(x^m (cos(m y) F(x^2) + sin(m y) G(x^2))) =
        # x^(m - 1) (2 x^2 (F'(x^2) cos(m y) + G'(x^2) sin(m y)) + m F(x^2) cos(m y) + m G(x^2) sin(m y))
        #                                                          ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #                                                                         m x "kernel" above
        # d/dy(x^m (cos(m y) F(x^2) + sin(m y) G(x^2))) = m x^m (G(x^2) cos(m y) - F(x^2) sin(m y))
        umm1 = u ** (m-1)
        twousq = 2 * usq
        aterm = cost * (twousq * Sprimea + m * Sa)
        bterm = sint * (twousq * Sprimeb + m * Sb)
        dr += umm1 * (aterm + bterm)
        dt += m * um * (-Sa * sint + Sb * cost)

    return z, dr, dt


def compute_z_Q2d(cm0, ams, bms, u, t):
    """Sag-only sibling of compute_z_zprime_Q2d."""
    usq = u * u
    z = np.zeros_like(u)

    cm0 = _trim_trailing_zeros(cm0)
    if len(cm0) > 0:
        z += compute_z_Qbfs(cm0, u, usq)

    m = 0
    for a_coef, b_coef in zip(ams, bms):
        m += 1
        a_coef = _trim_trailing_zeros(a_coef)
        b_coef = _trim_trailing_zeros(b_coef)
        if len(a_coef) == 0 and len(b_coef) == 0:
            continue

        Na = len(a_coef) - 1
        Nb = len(b_coef) - 1
        Sa = 0
        Sb = 0
        if len(a_coef) > 0:
            alphas_a = clenshaw_q2d(a_coef, m, usq)
            Sa = 0.5 * alphas_a[0]
        if len(b_coef) > 0:
            alphas_b = clenshaw_q2d(b_coef, m, usq)
            Sb = 0.5 * alphas_b[0]
        if m == 1 and Na > 2:
            Sa -= 2/5 * alphas_a[3]
        if m == 1 and Nb > 2:
            Sb -= 2/5 * alphas_b[3]

        um = u ** m
        z += um * (np.cos(m*t) * Sa + np.sin(m*t) * Sb)

    return z


def Q2d_nm_c_to_a_b(nms, coefs):
    """Re-structure Q2D coefficients to the form needed by compute_z_zprime_Q2d.

    Parameters
    ----------
    nms : iterable
        seq of [(n1, m1), (n2, m2), ...]
        negative m encodes "sine term" while positive m encodes "cosine term"
    coefs : iterable
        same length as nms, coefficients for mode n_m

    Returns
    -------
    list, list, list
        list 1 is cms, the "Qbfs" coefficients (m=0)
        list 2 is the "a" coefficients (cosine terms)
        list 3 is the "b" coefficients (sine terms)

        lists 2 and 3 are lists-of-lists and begin from m=1 to m=M, containing
        an empty list if that order was not present in the input

    """
    def factory():
        return []

    def expand_and_copy(cs, N):
        cs2 = [None] * (N+1)
        for i, cc in enumerate(cs):
            cs2[i] = cc

        return cs2

    cms = []
    ac = defaultdict(factory)  # start with dicts, will go to lists later
    bc = defaultdict(factory)
    # given arbitrary n, m, c which may be sparse
    # => go to dense, ordered arrays

    for (n, m), c in zip(nms, coefs):
        if c == 0:
            continue
        if m == 0:
            if len(cms) < n+1:
                cms = expand_and_copy(cms, n)

            cms[n] = c
        elif m > 0:
            if len(ac[m]) < n+1:
                ac[m] = expand_and_copy(ac[m], n)

            ac[m][n] = c
        else:
            m = -m
            if len(bc[m]) < n+1:
                bc[m] = expand_and_copy(bc[m], n)

            bc[m][n] = c

    for i, c in enumerate(cms):
        if c is None:
            cms[i] = 0

    for k in ac:
        for i, c in enumerate(ac[k]):
            if ac[k][i] is None:
                ac[k][i] = 0

    for k in bc:
        for i, c in enumerate(bc[k]):
            if bc[k][i] is None:
                bc[k][i] = 0

    cms = list(_trim_trailing_zeros(cms))
    for k in list(ac.keys()):
        ac[k] = list(_trim_trailing_zeros(ac[k]))
        if len(ac[k]) == 0:
            del ac[k]
    for k in list(bc.keys()):
        bc[k] = list(_trim_trailing_zeros(bc[k]))
        if len(bc[k]) == 0:
            del bc[k]

    max_m_a = max(ac.keys(), default=0)
    max_m_b = max(bc.keys(), default=0)
    max_m = max(max_m_a, max_m_b)
    ac_ret = []
    bc_ret = []
    for i in range(1, max_m+1):
        ac_ret.append(ac[i])
        bc_ret.append(bc[i])
    return cms, ac_ret, bc_ret
