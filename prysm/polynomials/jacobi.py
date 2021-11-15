"""High performance / recursive jacobi polynomial calculation."""
from prysm.mathops import np


def weight(alpha, beta, x):
    """The weight function of the jacobi polynomials for a given alpha, beta value."""
    return (1 - x) ** alpha * (1 + x) ** beta


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

    Parameters
    ----------
    n : int
        polynomial order
    alpha : float
        first weight parameter
    beta : float
        second weight parameter
    x : numpy.ndarray
        x coordinates to evaluate at

    Returns
    -------
    numpy.ndarray
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
    alpha : float
        first weight parameter
    beta : float
        second weight parameter
    x : numpy.ndarray
        x coordinates to evaluate at

    Returns
    -------
    generator
        equivalent to array of shape (len(ns), len(x))

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


def jacobi_der(n, alpha, beta, x):
    """First derivative of Pn with respect to x, at points x.

    Parameters
    ----------
    n : int
        polynomial order
    alpha : float
        first weight parameter
    beta : float
        second weight parameter
    x : numpy.ndarray
        x coordinates to evaluate at

    Returns
    -------
    numpy.ndarray
        jacobi polynomial evaluated at the given points

    """
    # see https://dlmf.nist.gov/18.9
    # dPn = (1/2) (n + a + b + 1)P_{n-1}^{a+1,b+1}
    # first two terms are specialized for speed
    if n == 0:
        return np.zeros_like(x)
    if n == 1:
        return np.ones_like(x) * (0.5 * (n + alpha + beta + 1))

    Pn = jacobi(n-1, alpha+1, beta+1, x)
    coef = 0.5 * (n + alpha + beta + 1)
    return coef * Pn


def jacobi_der_sequence(ns, alpha, beta, x):
    """First partial derivative of Pn w.r.t. x for order ns, i.e. P_n'.

    Parameters
    ----------
    ns : iterable
        sorted orders to return, e.g. [1, 2, 3, 10] returns P1', P2', P3', P10'
    alpha : float
        first weight parameter
    beta : float
        second weight parameter
    x : numpy.ndarray
        x coordinates to evaluate at

    Returns
    -------
    generator
        equivalent to array of shape (len(ns), len(x))

    """
    # the body of this function is very similar to that of jacobi_sequence,
    # except note that der is related to jacobi n-1,
    # and the actual jacobi polynomial has a different alpha and beta

    # special note: P0 is invariant of alpha, beta
    # and within this function alphap1 and betap1 are "a+1" and "b+1"
    alphap1 = alpha + 1
    betap1 = beta + 1
    # except when it comes time to yield terms, we yield the modification
    # per A&S / the NIST link
    # and we modify the arguments to
    ns = list(ns)
    min_i = 0
    if ns[min_i] == 0:
        # n=0 is piston, der==0
        yield np.zeros_like(x)
        min_i += 1

    if min_i == len(ns):
        return

    if ns[min_i] == 1:
        yield np.ones_like(x) * (0.5 * (1 + alpha + beta + 1))
        min_i += 1

    if min_i == len(ns):
        return

    # min_n is at least two, which means min n-1 is 1
    # from here below, Pn is P of order i to keep the reader sane, but Pnm1
    # is all that is needed;
    # therefor, Pn is computed only after testing if we are done and can return
    # to avoid a waste computation at the end of the loop
    # note that we can hardcode / unroll the loop up to n=3, one further than
    # in jacobi, because we use Pnm1
    P1 = alphap1 + 1 + (alphap1 + betap1 + 2) * ((x - 1) / 2)
    if ns[min_i] == 2:
        yield P1 * (0.5 * (2 + alpha + beta + 1))
        min_i += 1

    if min_i == len(ns):
        return

    a, c, b1, b2, b3 = recurrence_ac_startb(2, alphap1, betap1)
    inva = 1 / a
    P2 = (b1 * (b2 * x + b3) * P1 - c) * inva  # no Pnm2 because Pnm2 == ones, c*Pnm2 is a noop
    if ns[min_i] == 3:
        yield P2 * (0.5 * (3 + alpha + beta + 1))
        min_i += 1

    if min_i == len(ns):
        return

    # weird look just above P2, need to prepare for lower loop
    # by setting Pnm2 = P1, Pnm1 = P2
    Pnm2 = P1
    Pnm1 = P2
    a, c, b1, b2, b3 = recurrence_ac_startb(3, alphap1, betap1)
    inva = 1 / a
    P3 = (b1 * (b2 * x + b3) * Pnm1 - c * Pnm2) * inva
    Pn = P3

    max_n = ns[-1]
    for i in range(4, max_n+1):
        Pnm2, Pnm1 = Pnm1, Pn
        if ns[min_i] == i:
            coef = 0.5 * (i + alpha + beta + 1)
            yield Pnm1 * coef
            min_i += 1

        if min_i == len(ns):
            return

        a, c, b1, b2, b3 = recurrence_ac_startb(i, alphap1, betap1)
        inva = 1 / a
        Pn = (b1 * (b2 * x + b3) * Pnm1 - c * Pnm2) * inva
