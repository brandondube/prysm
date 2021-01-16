"""Tools for working with Q (Forbes) polynomials."""
# not special engine, only concerns scalars here
from scipy import special

from .mathops import engine as np, kronecker, gamma, sign
from .jacobi import jacobi, jacobi_sequence

MAX_ELEMENTS_IN_CACHE = 1024  # surely no one wants > 1000 terms...


def g_qbfs(n_minus_1):
    """g(m-1) from oe-18-19-19700 eq. (A.15)."""
    if n_minus_1 == 0:
        return - 1 / 2
    else:
        n_minus_2 = n_minus_1 - 1
        return - (1 + g_qbfs(n_minus_2) * h_qbfs(n_minus_2)) / f_qbfs(n_minus_1)


def h_qbfs(n_minus_2):
    """h(m-2) from oe-18-19-19700 eq. (A.14)."""
    n = n_minus_2 + 2
    return -n * (n - 1) / (2 * f_qbfs(n_minus_2))


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
    x : `numpy.array`
        point(s) at which to evaluate

    Returns
    -------
    `numpy.ndarray`
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
    # also known as the asymmetric chebyshev polynomials

    rho = x ** 2
    # c_Q is the leading term used to convert Qm to Qbfs
    c_Q = rho * (1 - rho)
    if n == 0:
        return np.ones_like(x) * c_Q

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


def Qbfs_sequence(ns, x):
    """Qbfs polynomials of orders ns at point(s) x.

    Parameters
    ----------
    ns : `Iterable` of int
        polynomial orders
    x : `numpy.array`
        point(s) at which to evaluate

    Returns
    -------
    generator of `numpy.ndarray`
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
    x : `numpy.array`
        point(s) at which to evaluate

    Returns
    -------
    `numpy.ndarray`
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
    ns : `Iterable` of int
        polynomial orders
    x : `numpy.array`
        point(s) at which to evaluate

    Returns
    -------
    generator of `numpy.ndarray`
        yielding one order of ns at a time

    """
    xx = x ** 2
    xx = 2 * xx - 1
    x4 = x ** 4
    Pns = jacobi_sequence(ns, 0, 4, xx)
    for Pn in Pns:
        yield Pn * x4


def abc_q2d(n, m):
    """A, B, C terms for 2D-Q polynomials.  oe-20-3-2483 Eq. (A.3).

    Parameters
    ----------
    n : `int`
        radial order
    m : `int`
        azimuthal order

    Returns
    -------
    `float`, `float`, `float`
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


def G_q2d(n, m):
    """G term for 2D-Q polynomials.  oe-20-3-2483 Eq. (A.15).

    Parameters
    ----------
    n : `int`
        radial order
    m : `int`
        azimuthal order

    Returns
    -------
    `float`
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


def F_q2d(n, m):
    """F term for 2D-Q polynomials.  oe-20-3-2483 Eq. (A.13).

    Parameters
    ----------
    n : `int`
        radial order
    m : `int`
        azimuthal order

    Returns
    -------
    `float`
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


def g_q2d(n, m):
    """Lowercase g term for 2D-Q polynomials.  oe-20-3-2483 Eq. (A.18a).

    Parameters
    ----------
    n : `int`
        radial order less one (n - 1)
    m : `int`
        azimuthal order

    Returns
    -------
    `float`
        g

    """
    return G_q2d(n, m) / f_q2d(n, m)


def f_q2d(n, m):
    """Lowercase f term for 2D-Q polynomials.  oe-20-3-2483 Eq. (A.18b).

    Parameters
    ----------
    n : `int`
        radial order
    m : `int`
        azimuthal order

    Returns
    -------
    `float`
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
    n : `int`
        radial polynomial order
    m : `int`
        azimuthal polynomial order
    r : `numpy.ndarray`
        radial coordinate, slope orthogonal in [0,1]
    t : `numpy.ndarray`
        azimuthal coordinate, radians

    Returns
    -------
    `numpy.ndarray`
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
        prefix = u ** abs(m) * np.sin(m*t)
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
