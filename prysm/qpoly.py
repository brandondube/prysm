"""Tools for working with Q (Forbes) polynomials."""
from functools import lru_cache

from .conf import config
from .mathops import engine as np, special_engine as special, kronecker, gamma
from .jacobi import jacobi, jacobi_sequence

MAX_ELEMENTS_IN_CACHE = 1024  # surely no one wants > 1000 terms...


@lru_cache(MAX_ELEMENTS_IN_CACHE)
def g_qbfs(n_minus_1):
    """g(m-1) from oe-18-19-19700 eq. (A.15)."""
    if n_minus_1 == 0:
        return - 1 / 2
    else:
        n_minus_2 = n_minus_1 - 1
        return - (1 + g_qbfs(n_minus_2) * h_qbfs(n_minus_2)) / f_qbfs(n_minus_1)


@lru_cache(MAX_ELEMENTS_IN_CACHE)
def h_qbfs(n_minus_2):
    """h(m-2) from oe-18-19-19700 eq. (A.14)."""
    n = n_minus_2 + 2
    return -n * (n - 1) / (2 * f_qbfs(n_minus_2))


@lru_cache(MAX_ELEMENTS_IN_CACHE)
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
        return term1 + term2  # this is minus in the paper
    else:
        # nt1 = numerator term 1, d = denominator...
        nt1 = 2 * n * (m + n - 1) - m
        nt2 = (n + 1) * (2 * m + 2 * n - 1)
        num = nt1 * nt2
        dt1 = (m + 2 * n - 2) * (m + 2 * n - 1)
        dt2 = (m + 2 * n) * (2 * n + 1)
        den = dt1 * dt2

        term1 = num / den  # there is a leading negative in the paper
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


def g_q2d(nm1, m):
    """Lowercase g term for 2D-Q polynomials.  oe-20-3-2483 Eq. (A.18a).

    Parameters
    ----------
    nm1 : `int`
        radial order less one (n - 1)
    m : `int`
        azimuthal order

    Returns
    -------
    `float`
        g

    """
    return G_q2d(nm1, m) / f_q2d(nm1, m)


def f_q2d(n, m):
    """Lowercase f term for 2D-Q polynomials.  oe-20-3-2483 Eq. (A.18b).

    Parameters
    ----------
    nm1 : `int`
        radial order
    m : `int`
        azimuthal order

    Returns
    -------
    `float`
        f

    """
    if n == 0:
        return e.sqrt(F_q2d(n=0, m=m))
    else:
        return e.sqrt(F_q2d(n, m) - g_q2d(n-1, m) ** 2)


def q2d_recurrence_P(n, m, x, Pnm1=None, Pnm2=None):
    """Auxiliary polynomial P to the 2DQ polynomials (Q).  oe-20-3-2483 Eq. (A.17).

    Parameters
    ----------
    n : `int`
        radial order
    m : `int`
        azimuthal order
    x : `numpy.ndarray`
        spatial coordinates, x = r^2
    Pnm1 : `numpy.ndarray`
        value of this function for argument n - 1
    Pnm2 : `numpy.ndarray`
        value of this function for argument n - 2

    Returns
    -------
    `numpy.ndarray`
        P polynomial evaluated over x

    """
    if m == 0:
        return qbfs_recurrence_P(n=n, x=x, Pnm1=Pnm1, Pnm2=Pnm2)
    if n == 0:
        return 1 / 2
    if n == 1:
        if m == 1:
            return 1 - x / 2
        elif m < 1:
            raise ValueError('2D-Q auxiliary polynomial is undefined for n=1, m < 1')
        else:
            return m - (1 / 2) - (m - 1) * x
    if m == 1 and (n == 2 or n == 3):
        if n == 2:
            num = 3 - x * (12 - 8 * x)
            den = 6
            return num / den
        if n == 3:
            numt1 = 5 - x
            numt2 = 60 - x * (120 - 64 * x)
            num = numt1 * numt2
            den = 10
            return num / den
    else:
        if Pnm2 is None:
            Pnm2 = q2d_recurrence_P(n=n-2, m=m, x=x)
        if Pnm1 is None:
            Pnm1 = q2d_recurrence_P(n=n-1, m=m, x=x, Pnm1=Pnm2)

        Anm, Bnm, Cnm = abc_q2d(n, m)
        term1 = Anm + Bnm * x
        term2 = Pnm1
        term3 = Cnm * Pnm2
        return term1 * term2 - term3


def q2d_recurrence_Q(n, m, x, Pn=None, Qnm1=None, Pnm1=None, Pnm2=None):
    """2DQ polynomials (Q).  oe-20-3-2483 Eq. (A.22).

    Parameters
    ----------
    n : `int`
        radial order
    m : `int`
        azimuthal order
    x : `numpy.ndarray`
        spatial coordinates, x = r^2
    Pn : `numpy.ndarray`
        value of this function for same order n
    Qnm1 : `numpy.ndarray`
        value of this function for argument n - 1
    Pnm1 : `numpy.ndarray`
        value of the paired P function for n - 1
    Pnm2 : `numpy.ndarray`
        value of the paired P function for n - 2

    Returns
    -------
    `numpy.ndarray`
        P polynomial evaluated over x

    """
    if n == 0:
        return 1 / (2 * f_q2d(0, m))
    elif m == 0:
        return qbfs_recurrence_Q(n=n, x=x, Pn=Pn, Pnm1=Pnm1, Pnm2=Pnm2, Qnm1=Qnm1)

    # manual startup, do not try to recurse for n <= 2
    if n == 1:
        Pn = q2d_recurrence_P(n=n, m=m, x=x, Pnm1=Pnm1)
        Qnm1 = 1 / (2 * f_q2d(0, m))  # same as L2 of this function, n=0
        g = g_q2d(0, m)
        f = f_q2d(n, m)
        return (Pn - g * Qnm1) / f
    if n == 2:
        Pn = q2d_recurrence_P(n=n, m=m, x=x, Pnm1=Pnm1, Pnm2=Pnm2)
        Qnm1 = q2d_recurrence_Q(n=n-1, m=m, x=x, Pnm1=Pnm2, Qnm1=1 / (2 * f_q2d(0, m)))
        g = g_q2d(1, m)
        f = f_q2d(n, m)
        return (Pn - g * Qnm1) / f

    if Pnm2 is None:
        Pnm2 = q2d_recurrence_P(n=n-2, m=m, x=x)
    if Pnm1 is None:
        Pnm1 = q2d_recurrence_P(n=n-1, m=m, x=x, Pnm1=Pnm2)
    if Pn is None:
        if n == 0:
            Pn = q2d_recurrence_P(n=n, m=m, x=x, Pnm1=Pnm1, Pnm2=Pnm2)

    if Qnm1 is None:
        Qnm1 = q2d_recurrence_Q(n=n-1, m=m, x=x, Pnm=Pnm1, Pnm1=Pnm2)

    return (Pn - g_q2d(n-1, m) * Qnm1) / f_q2d(n, m)
