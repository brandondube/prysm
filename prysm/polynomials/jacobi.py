"""High performance / recursive jacobi polynomial calculation."""
from prysm.mathops import engine as np


def weight(alpha, beta, x):
    """The weight function of the jacobi polynomials for a given alpha, beta value."""
    one_minus_x = 1 - x
    return (one_minus_x ** alpha) * (one_minus_x ** beta)


def recurrence_ac_startb(n, alpha, beta):
    """a and c terms of the recurrence relation from Wikipedia, * P_n^(a,b).

    Also computes partial b term; all components without x
    """
    a = (2 * n) * (n + alpha + beta) * (2 * n + alpha + beta - 2)
    c = 2 * (n + alpha - 1) * (n + beta - 1) * (2 * n + alpha + beta)
    b1 = (2 * n + alpha + beta - 1)
    b2 = (2 * n + alpha + beta)
    b2 = b2 * (b2 - 2)
    b3 = alpha ** 2 - beta ** 2
    return a, c, b1, b2, b3


def jacobi(n, alpha, beta, x):
    """Jacobi polynomial of order n with weight parameters alpha and beta.

    Notes
    -----
    This function is faster than scipy.special.jacobi when Pnm1 and Pnm2 are
    supplied and is stable to high order.  Performance benefit ranges from 2-5x.

    Parameters
    ----------
    n : `int`
        polynomial order
    alpha : `float`
        first weight parameter
    beta : `float`
        second weight parameter
    x : `numpy.ndarray`
        x coordinates to evaluate at

    Returns
    -------
    `numpy.ndarray`
        jacobi polynomial evaluated at the given points

    """
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        term1 = alpha + 1
        term2 = alpha + beta + 2
        term3 = (x - 1) / 2
        return term1 + term2 * term3

    Pnm1 = alpha + 1 + (alpha + beta + 2) * ((x - 1) / 2)
    a, c, b1, b2, b3 = recurrence_ac_startb(2, alpha, beta)
    inva = 1 / a
    Pn = (b1 * (b2 * x + b3) * Pnm1 - c) * inva  # no Pnm2 because Pnm2 == ones, c*Pnm2 is a noop
    if n == 2:
        return Pn

    for i in range(3, n+1):
        Pnm2, Pnm1 = Pnm1, Pn
        a, c, b1, b2, b3 = recurrence_ac_startb(i, alpha, beta)
        inva = 1 / a
        Pn = (b1 * (b2 * x + b3) * Pnm1 - c * Pnm2) * inva

    return Pn


def jacobi_sequence(ns, alpha, beta, x):
    """Jacobi polynomials of order 0..n_max with weight parameters alpha and beta.

    Parameters
    ----------
    ns : iterable
        sorted polynomial orders to return, e.g. [1, 3, 5, 7, ...]
    alpha : `float`
        first weight parameter
    beta : `float`
        second weight parameter
    x : `numpy.ndarray`
        x coordinates to evaluate at

    Returns
    -------
    `numpy.ndarray`
        array of shape (n_max+1, len(x))

    """
    # three key flavors: return list, return array, or return generator
    # return generator has most pleasant interface, benchmarked at 68 ns
    # per yield (315 clocks).  With 32 clocks per element of x, 1% of the
    # time is spent on yield when x has 1000 elements, or 32x32
    # => use generator
    # benchmarked at 4.6 ns/element (256x256), 4.6GHz CPU = 21 clocks
    # ~4x faster than previous impl (118 ms => 29.8)
    ns = list(ns)
    min_i = 0
    Pn = np.ones_like(x)
    if ns[min_i] == 0:
        yield Pn
        min_i += 1

    if min_i == len(ns):
        return

    Pn = alpha + 1 + (alpha + beta + 2) * ((x - 1) / 2)
    if ns[min_i] == 1:
        yield Pn
        min_i += 1

    if min_i == len(ns):
        return

    Pnm1 = Pn
    a, c, b1, b2, b3 = recurrence_ac_startb(2, alpha, beta)
    inva = 1 / a
    Pn = (b1 * (b2 * x + b3) * Pnm1 - c) * inva  # no Pnm2 because Pnm2 == ones, c*Pnm2 is a noop
    if ns[min_i] == 2:
        yield Pn
        min_i += 1

    if min_i == len(ns):
        return

    max_n = ns[-1]
    for i in range(3, max_n+1):
        Pnm2, Pnm1 = Pnm1, Pn
        a, c, b1, b2, b3 = recurrence_ac_startb(i, alpha, beta)
        inva = 1 / a
        Pn = (b1 * (b2 * x + b3) * Pnm1 - c * Pnm2) * inva
        if ns[min_i] == i:
            yield Pn
            min_i += 1
