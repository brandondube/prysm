"""Laguerre polynomials."""

from prysm.mathops import np
from prysm.conf import config


def laguerre(n, alpha, x):
    """Generalized Laguerre polynomial of order n.

    Parameters
    ----------
    n : int
        polynomial order
    alpha : float
        shaping parameter
    x : numpy.ndarray
        coordinates to evaluate at; the laguerre polynomials are orthogonal on
        the interval [0,inf)

    Returns
    -------
    numpy.ndarray
        generalized laguerre polynomial evaluated at the given points

    """
    if n == 0:
        return np.ones_like(x)

    if n == 1:
        return (alpha+1) - x

    # see https://math.colgate.edu/~integers/v95/v95.pdf
    # recurrence relation stated as
    # (n+1) L(n+1) = (a + 2n + 1 - x) Ln - (alpha + n) L(n-1)

    L0 = 1
    L1 = (alpha+1) - x

    # n-1 and n
    Lnm1 = L0
    Ln = L1

    # 3 = 2n+1, n=1
    A = (alpha + 3 - x)
    B = alpha + 1
    Lnp1 = 0.5 * (A*Ln - B*Lnm1)  # written differently; ordinarily / n
    if n == 2:
        return Lnp1

    Ln, Lnm1 = Lnp1, Ln

    # np1 = n+1, because recurrence is written to compute the future
    # n+1 in range is because range is end-exclusive
    for np1 in range(3, n+1):
        n = np1 - 1
        A = (alpha + 2*n + 1 - x)
        B = alpha + n
        Lnp1 = 1/(n+1) * (A*Ln - B*Lnm1)
        Ln, Lnm1 = Lnp1, Ln

    return Lnp1


def laguerre_seq(ns, alpha, x):
    """Generalized Laguerre polynomial of orders ns.

    Parameters
    ----------
    ns : sequence
        polynomial orders, ascending order
    alpha : float
        shaping parameter
    x : numpy.ndarray
        coordinates to evaluate at; the laguerre polynomials are orthogonal on
        the interval [0,inf)

    Returns
    -------
    numpy.ndarray
        shape (k, len(x))
        generalized laguerre polynomials evaluated at the given points

    """
    ns = list(ns)
    min_i = 0
    out = np.empty((len(ns), *x.shape), dtype=x.dtype)

    if ns[min_i] == 0:
        out[min_i] = 1
        min_i += 1

    if min_i == len(ns):
        return out

    if ns[min_i] == 1:
        out[min_i] = (alpha+1) - x
        min_i += 1

    if min_i == len(ns):
        return out

    L0 = 1
    L1 = (alpha+1) - x

    # n-1 and n
    Lnm1 = L0
    Ln = L1

    # 3 = 2n+1, n=1
    A = (alpha + 3 - x)
    B = alpha + 1
    Lnp1 = 0.5 * (A*Ln - B*Lnm1)  # written differently; ordinarily / n

    if ns[min_i] == 2:
        out[min_i] = Lnp1
        min_i += 1

    Ln, Lnm1 = Lnp1, Ln

    if min_i == len(ns):
        return out

    max_n = ns[-1]
    for np1 in range(3, max_n+1):
        n = np1 - 1
        A = (alpha + 2*n + 1 - x)
        B = alpha + n
        Lnp1 = 1/(n+1) * (A*Ln - B*Lnm1)
        Ln, Lnm1 = Lnp1, Ln

        if ns[min_i] == np1:
            out[min_i] = Lnp1
            min_i += 1

    return out


def laguerre_der(n, alpha, x):
    """d/dx of Laguerre polynomial of order n.

    Parameters
    ----------
    n : int
        polynomial order
    alpha : float
        shaping parameter
    x : numpy.ndarray
        coordinates to evaluate at; the laguerre polynomials are orthogonal on
        the interval [0,inf)

    Returns
    -------
    numpy.ndarray
        d/dx of generalized laguerre polynomial evaluated at the given points

    """
    # see wiki
    # d^k/dx^k L_n^alpha = (-1)^k L_(n-k)^(alpha+k)
    k = 1
    return laguerre(n-k, alpha+k, x)


def laguerre_der_seq(ns, alpha, x):
    """d/dx of Generalized Laguerre polynomial of orders ns.

    Parameters
    ----------
    ns : sequence
        polynomial orders, ascending order
    alpha : float
        shaping parameter
    x : numpy.ndarray
        coordinates to evaluate at; the laguerre polynomials are orthogonal on
        the interval [0,inf)

    Returns
    -------
    numpy.ndarray
        shape (k, len(x))
        d/dx of generalized laguerre polynomials evaluated at the given points

    """
    k = 1
    ns = [n-k for n in ns]
    return laguerre_seq(ns, alpha+k, x)
