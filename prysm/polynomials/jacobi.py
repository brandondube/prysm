"""High performance / recursive jacobi polynomial calculation."""
from prysm.mathops import np
from prysm.conf import config

from functools import lru_cache


def weight(alpha, beta, x):
    """The weight function of the jacobi polynomials for a given alpha, beta value."""
    return (1 - x) ** alpha * (1 + x) ** beta


@lru_cache(512)
def recurrence_abc(n, alpha, beta):
    """See A&S online - https://dlmf.nist.gov/18.9 .

    Pn = (an-1 x + bn-1) Pn-1 - cn-1 * Pn-2

    This function makes a, b, c for the given n,
    i.e. to get a(n-1), do recurrence_abc(n-1)

    """
    aplusb = alpha+beta
    if n == 0 and (aplusb == 0 or aplusb == -1):
        A = 1/2 * (alpha + beta) + 1
        B = 1/2 * (alpha - beta)
        C = 1
    else:
        Anum = (2 * n + alpha + beta + 1) * (2 * n + alpha + beta + 2)
        Aden = 2 * (n + 1) * (n + alpha + beta + 1)
        A = Anum/Aden

        Bnum = (alpha**2 - beta**2) * (2 * n + alpha + beta + 1)
        Bden = 2 * (n+1) * (n + alpha + beta + 1) * (2 * n + alpha + beta)
        B = Bnum / Bden

        Cnum = (n + alpha) * (n + beta) * (2 * n + alpha + beta + 2)
        Cden = (n + 1) * (n + alpha + beta + 1) * (2 * n + alpha + beta)
        C = Cnum / Cden

    return A, B, C


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
    A, B, C = recurrence_abc(1, alpha, beta)
    Pn = (A * x + B) * Pnm1 - C  # no C * Pnm2 =because Pnm2 = 1
    if n == 2:
        return Pn

    for i in range(3, n+1):
        Pnm2, Pnm1 = Pnm1, Pn
        A, B, C = recurrence_abc(i-1, alpha, beta)
        Pn = (A * x + B) * Pnm1 - C * Pnm2

    return Pn


def jacobi_sequence(ns, alpha, beta, x):
    """Jacobi polynomials of orders ns with weight parameters alpha and beta.

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
    A, B, C = recurrence_abc(1, alpha, beta)
    Pn = (A * x + B) * Pnm1 - C  # no C * Pnm2 =because Pnm2 = 1
    if ns[min_i] == 2:
        yield Pn
        min_i += 1

    if min_i == len(ns):
        return

    max_n = ns[-1]
    for i in range(3, max_n+1):
        Pnm2, Pnm1 = Pnm1, Pn
        A, B, C = recurrence_abc(i-1, alpha, beta)
        Pn = (A * x + B) * Pnm1 - C * Pnm2
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

    A, B, C = recurrence_abc(1, alphap1, betap1)
    P2 = (A * x + B) * P1 - C  # no C * Pnm2 =because Pnm2 = 1
    if ns[min_i] == 3:
        yield P2 * (0.5 * (3 + alpha + beta + 1))
        min_i += 1

    if min_i == len(ns):
        return

    # weird look just above P2, need to prepare for lower loop
    # by setting Pnm2 = P1, Pnm1 = P2
    Pnm2 = P1
    Pnm1 = P1
    Pn = P2
    # A, B, C = recurrence_abc(2, alpha, beta)
    # P3 = (A * x + B) * P2 - C * P1
    # Pn = P3

    max_n = ns[-1]
    for i in range(3, max_n+1):
        Pnm2, Pnm1 = Pnm1, Pn
        if ns[min_i] == i:
            coef = 0.5 * (i + alpha + beta + 1)
            yield Pnm1 * coef
            min_i += 1

        if min_i == len(ns):
            return

        A, B, C = recurrence_abc(i-1, alphap1, betap1)
        Pn = (A * x + B) * Pnm1 - C * Pnm2


def _initialize_alphas(s, x, alphas, j=0):
    # j = derivative order
    if alphas is None:
        if hasattr(x, 'dtype'):
            dtype = x.dtype
        else:
            dtype = config.precision
        if hasattr(x, 'shape'):
            shape = (len(s), *x.shape)
        elif hasattr(x, '__len__'):
            shape = (len(s), len(x))
        else:
            shape = (len(s),)

        if j != 0:
            shape = (j+1, *shape)

        alphas = np.zeros(shape, dtype=dtype)
    return alphas


def jacobi_sum_clenshaw(s, alpha, beta, x, alphas=None):
    """Compute a weighted sum of Jacobi polynomials using Clenshaw's method.

    Parameters
    ----------
    s : iterable
        weights in ascending order, beginning with P0, then P1, etc.
        must be fully dense when iterated
    alpha : float
        first Jacobi shape parameter
    beta : float
        second Jacobi shape parameter
    x : numpy.ndarray or float_like
        coordinates to evaluate the sum at,
        orthogonal over [-1,1]
    alphas : numpy.ndarray, optional
        array to store the alpha sums in, alphas[0] contains the sum and is returned
        if not None, alphas should be of shape (len(s), *x.shape)
        see _initialize_alphas if you desire more information

    Returns
    -------
    numpy.ndarray
        weighted sum of Jacobi polynomials

    """
    # this doesn't match Forbes,
    # because Forbes uses Pn = (a + b x)Pn-1 - cPn-2
    # and I use the A&S notation Pn = (a x + b)Pn-1 - cPn-2
    # so the "a" and "b" below are swapped here
    # checked to be correct, though...
    alphas = _initialize_alphas(s, x, alphas)
    M = len(s) - 1
    alphas[M] = s[M]
    a, b, c = recurrence_abc(M-1, alpha, beta)
    alphas[M-1] = s[M-1] + (a * x + b) * s[M]
    for n in range(M-2, -1, -1):
        a, b, _ = recurrence_abc(n, alpha, beta)
        _, _, c = recurrence_abc(n+1, alpha, beta)
        alphas[n] = s[n] + (a * x + b) * alphas[n+1] - c * alphas[n+2]

    return alphas[0]


def jacobi_sum_clenshaw_der(s, alpha, beta, x, j=1, alphas=None):
    """Compute a weighted sum of partial derivatives w.r.t. x of Jacobi polynomials using Clenshaw's method.

    Notes
    -----
    If the polynomial values and their derivatives are desired, pass
    alphas instead of leaving it None.  alphas[0,0] will contain the
    sum of the polynomials, alphas[1,0] the sum of the first derivative,
    and so on.

    Parameters
    ----------
    s : iterable
        weights in ascending order, beginning with P0, then P1, etc.
        must be fully dense when iterated
    alpha : float
        first Jacobi shape parameter
    beta : float
        second Jacobi shape parameter
    x : numpy.ndarray or float_like
        coordinates to evaluate the sum at,
        orthogonal over [-1,1]
    j : int
        derivative order to compute
    alphas : numpy.ndarray, optional
        array to store the alpha sums in,
        alphas[n] is the nth order derivative alpha terms
        with n=0 being the non-derivative terms.

        for a given n, the value of alphas[0] is the nth derivative of the surface sum
        if not None, alphas should be of shape (j+1, len(s), *x.shape)
        see _initialize_alphas if you desire more information

    Returns
    -------
    numpy.ndarray
        alphas array, see alphas parameter documentation for meaning

    """
    # alphas is dual indexed by alphas[j][n]
    # j = derivative
    # n = order
    # inner loop over n, outer loop over j
    alphas = _initialize_alphas(s, x, None, j=j)
    M = len(s) - 1
    # seed the first sweep of alpha, for j=0, by side effect
    jacobi_sum_clenshaw(s, alpha, beta, x, alphas=alphas[0])
    # now loop over increasing j
    for jj in range(1, j+1):
        # more twisted notation - follow Forbes' paper, but our
        # idea of b and a are swapped
        a, *_ = recurrence_abc(M-jj, alpha, beta)
        alphas[jj][M-jj] = j * a * alphas[jj-1][M-jj+1]
        for n in range(M-jj-1, -1, -1):
            a, b, _ = recurrence_abc(n, alpha, beta)
            _, _, c = recurrence_abc(n+1, alpha, beta)
            alphas[jj][n] = jj * a * alphas[jj-1][n+1] + (a * x + b) * alphas[jj][n+1] - c * alphas[jj][n+2]

    return alphas
