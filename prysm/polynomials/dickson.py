"""Dickson Polynomials."""

from prysm.mathops import np


def dickson1(n, alpha, x):
    """Dickson Polynomial of the first kind of order n.

    Parameters
    ----------
    n : int
        polynomial order
    alpha : float
        shape parameter
        if alpha = -1, the dickson polynomials are Fibonacci Polynomials
        if alpha = 0, the dickson polynomials are the monomials x^n
        if alpha = 1, the dickson polynomials and cheby1 polynomials are
        related by D_n(2x) = 2T_n(x)
    x : numpy.ndarray
        coordinates to evaluate the polynomial at

    Returns
    -------
    numpy.ndarray
        D_n(x)

    """
    if n == 0:
        return np.ones_like(x) * 2
    if n == 1:
        return x

    # general recursive polynomials:
    # P0, P1 are the n=0,1 seed terms
    # Pnm1 = P_{n-1}, Pnm2 = P_{n-2}
    P0 = np.ones_like(x) * 2
    P1 = x
    Pnm2 = P0
    Pnm1 = P1
    for _ in range(2, n+1):
        Pn = x * Pnm1 - alpha * Pnm2
        Pnm1, Pnm2 = Pn, Pnm1

    return Pn


def dickson2(n, alpha, x):
    """Dickson Polynomial of the second kind of order n.

    Parameters
    ----------
    n : int
        polynomial order
    alpha : float
        shape parameter
        if alpha = -1, the dickson polynomials are Lucas Polynomials
    x : numpy.ndarray
        coordinates to evaluate the polynomial at

    Returns
    -------
    numpy.ndarray
        E_n(x)

    """
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return x

    # general recursive polynomials:
    # P0, P1 are the n=0,1 seed terms
    # Pnm1 = P_{n-1}, Pnm2 = P_{n-2}
    P0 = np.ones_like(x)
    P1 = x
    Pnm2 = P0
    Pnm1 = P1
    for _ in range(2, n+1):
        Pn = x * Pnm1 - alpha * Pnm2
        Pnm1, Pnm2 = Pn, Pnm1

    return Pn


def dickson1_sequence(ns, alpha, x):
    """Sequence of Dickson Polynomial of the first kind of orders ns.

    Parameters
    ----------
    ns : iterable of int
        rising polynomial orders, assumed to be sorted
    alpha : float
        shape parameter
        if alpha = -1, the dickson polynomials are Fibonacci Polynomials
        if alpha = 0, the dickson polynomials are the monomials x^n
        if alpha = 1, the dickson polynomials and cheby1 polynomials are
        related by D_n(2x) = 2T_n(x)
    x : numpy.ndarray
        coordinates to evaluate the polynomial at

    Returns
    -------
    ndarray
        has shape (len(ns), *x.shape)
        e.g., for 5 modes and x of dimension 100x100,
        return has shape (5, 100, 100)

    """
    ns = list(ns)
    min_i = 0
    j = 0
    out = np.empty((len(ns), *x.shape), dtype=x.dtype)
    P0 = 2
    if ns[min_i] == 0:
        out[j] = 2
        min_i += 1
        j += 1

    if min_i == len(ns):
        return

    P1 = x
    if ns[min_i] == 1:
        out[j] = x
        min_i += 1
        j += 1

    if min_i == len(ns):
        return

    Pnm2 = P0
    Pnm1 = P1
    for i in range(2, ns[-1]+1):
        Pn = x * Pnm1 - alpha * Pnm2
        Pnm1, Pnm2 = Pn, Pnm1
        if ns[min_i] == i:
            out[j] = Pn
            min_i += 1
            j += 1

    return out


def dickson2_sequence(ns, alpha, x):
    """Sequence of Dickson Polynomial of the second kind of orders ns.

    Parameters
    ----------
    ns : iterable of int
        rising polynomial orders, assumed to be sorted
    alpha : float
        shape parameter
        if alpha = -1, the dickson polynomials are Lucas Polynomials
    x : numpy.ndarray
        coordinates to evaluate the polynomial at

    Returns
    -------
    ndarray
        has shape (len(ns), *x.shape)
        e.g., for 5 modes and x of dimension 100x100,
        return has shape (5, 100, 100)

    """
    ns = list(ns)
    min_i = 0
    j = 0
    out = np.empty((len(ns), *x.shape), dtype=x.dtype)
    P0 = 1
    if ns[min_i] == 0:
        out[j] = 1
        min_i += 1
        j += 1

    if min_i == len(ns):
        return

    P1 = x
    if ns[min_i] == 1:
        out[j] = x
        min_i += 1
        j += 1

    if min_i == len(ns):
        return

    Pnm2 = P0
    Pnm1 = P1
    for i in range(2, ns[-1]+1):
        Pn = x * Pnm1 - alpha * Pnm2
        Pnm1, Pnm2 = Pn, Pnm1
        if ns[min_i] == i:
            out[j] = Pn
            min_i += 1
            j += 1

    return out
