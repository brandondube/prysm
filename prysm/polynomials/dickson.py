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
    x : ndarray
        coordinates to evaluate the polynomial at

    Returns
    -------
    ndarray
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
    x : ndarray
        coordinates to evaluate the polynomial at

    Returns
    -------
    ndarray
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


def dickson1_seq(ns, alpha, x):
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
    x : ndarray
        coordinates to evaluate the polynomial at

    Returns
    -------
    ndarray
        has shape (len(ns),) followed by x.shape
        e.g., for 5 modes and x of dimension 100x100,
        return has shape (5, 100, 100)

    """
    if not hasattr(ns, '__len__'):
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
        return out

    P1 = x
    if ns[min_i] == 1:
        out[j] = x
        min_i += 1
        j += 1

    if min_i == len(ns):
        return out

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


def dickson1_der(n, alpha, x):
    """Partial derivative w.r.t. x of the Dickson polynomial of the first kind of order n.

    Uses the differentiated recurrence D'_n = D_{n-1} + x D'_{n-1} - alpha D'_{n-2}
    with D'_0 = 0, D'_1 = 1.

    Parameters
    ----------
    n : int
        polynomial order
    alpha : float
        shape parameter (see dickson1)
    x : ndarray
        coordinates to evaluate the derivative at

    Returns
    -------
    ndarray
        d/dx D_n(x)

    """
    if n == 0:
        return np.zeros_like(x)
    if n == 1:
        return np.ones_like(x)

    # carry along values D_k AND derivatives D'_k via the parallel recurrence
    Pnm2 = np.ones_like(x) * 2  # D_0
    Pnm1 = x                    # D_1
    Dnm2 = np.zeros_like(x)     # D'_0
    Dnm1 = np.ones_like(x)      # D'_1
    for _ in range(2, n+1):
        Pn = x * Pnm1 - alpha * Pnm2
        Dn = Pnm1 + x * Dnm1 - alpha * Dnm2
        Pnm2, Pnm1 = Pnm1, Pn
        Dnm2, Dnm1 = Dnm1, Dn

    return Dn


def dickson1_der_seq(ns, alpha, x):
    """Partial derivative w.r.t. x of Dickson Polynomials of the first kind for orders ns.

    Parameters
    ----------
    ns : iterable of int
        rising polynomial orders, assumed to be sorted
    alpha : float
        shape parameter (see dickson1)
    x : ndarray
        coordinates to evaluate the derivative at

    Returns
    -------
    ndarray
        has shape (len(ns),) followed by x.shape; the i-th plane is d/dx D_{ns[i]}(x)

    """
    if not hasattr(ns, '__len__'):
        ns = list(ns)
    min_i = 0
    j = 0
    out = np.empty((len(ns), *x.shape), dtype=x.dtype)
    if ns[min_i] == 0:
        out[j] = 0
        min_i += 1
        j += 1

    if min_i == len(ns):
        return out

    if ns[min_i] == 1:
        out[j] = 1
        min_i += 1
        j += 1

    if min_i == len(ns):
        return out

    Pnm2 = np.ones_like(x) * 2
    Pnm1 = x
    Dnm2 = np.zeros_like(x)
    Dnm1 = np.ones_like(x)
    for i in range(2, ns[-1]+1):
        Pn = x * Pnm1 - alpha * Pnm2
        Dn = Pnm1 + x * Dnm1 - alpha * Dnm2
        Pnm2, Pnm1 = Pnm1, Pn
        Dnm2, Dnm1 = Dnm1, Dn
        if ns[min_i] == i:
            out[j] = Dn
            min_i += 1
            j += 1

    return out


def dickson2_der(n, alpha, x):
    """Partial derivative w.r.t. x of the Dickson polynomial of the second kind of order n.

    The recurrence for E_n and its derivative share the same coefficients as
    for D_n; only the n=0 seed differs (E_0 = 1 vs D_0 = 2), but E'_0 = D'_0 = 0.

    Parameters
    ----------
    n : int
        polynomial order
    alpha : float
        shape parameter (see dickson2)
    x : ndarray
        coordinates to evaluate the derivative at

    Returns
    -------
    ndarray
        d/dx E_n(x)

    """
    if n == 0:
        return np.zeros_like(x)
    if n == 1:
        return np.ones_like(x)

    Pnm2 = np.ones_like(x)
    Pnm1 = x
    Dnm2 = np.zeros_like(x)
    Dnm1 = np.ones_like(x)
    for _ in range(2, n+1):
        Pn = x * Pnm1 - alpha * Pnm2
        Dn = Pnm1 + x * Dnm1 - alpha * Dnm2
        Pnm2, Pnm1 = Pnm1, Pn
        Dnm2, Dnm1 = Dnm1, Dn

    return Dn


def dickson2_der_seq(ns, alpha, x):
    """Partial derivative w.r.t. x of Dickson Polynomials of the second kind for orders ns.

    Parameters
    ----------
    ns : iterable of int
        rising polynomial orders, assumed to be sorted
    alpha : float
        shape parameter (see dickson2)
    x : ndarray
        coordinates to evaluate the derivative at

    Returns
    -------
    ndarray
        has shape (len(ns),) followed by x.shape; the i-th plane is d/dx E_{ns[i]}(x)

    """
    if not hasattr(ns, '__len__'):
        ns = list(ns)
    min_i = 0
    j = 0
    out = np.empty((len(ns), *x.shape), dtype=x.dtype)
    if ns[min_i] == 0:
        out[j] = 0
        min_i += 1
        j += 1

    if min_i == len(ns):
        return out

    if ns[min_i] == 1:
        out[j] = 1
        min_i += 1
        j += 1

    if min_i == len(ns):
        return out

    Pnm2 = np.ones_like(x)
    Pnm1 = x
    Dnm2 = np.zeros_like(x)
    Dnm1 = np.ones_like(x)
    for i in range(2, ns[-1]+1):
        Pn = x * Pnm1 - alpha * Pnm2
        Dn = Pnm1 + x * Dnm1 - alpha * Dnm2
        Pnm2, Pnm1 = Pnm1, Pn
        Dnm2, Dnm1 = Dnm1, Dn
        if ns[min_i] == i:
            out[j] = Dn
            min_i += 1
            j += 1

    return out


def dickson2_seq(ns, alpha, x):
    """Sequence of Dickson Polynomial of the second kind of orders ns.

    Parameters
    ----------
    ns : iterable of int
        rising polynomial orders, assumed to be sorted
    alpha : float
        shape parameter
        if alpha = -1, the dickson polynomials are Lucas Polynomials
    x : ndarray
        coordinates to evaluate the polynomial at

    Returns
    -------
    ndarray
        has shape (len(ns),) followed by x.shape
        e.g., for 5 modes and x of dimension 100x100,
        return has shape (5, 100, 100)

    """
    if not hasattr(ns, '__len__'):
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
        return out

    P1 = x
    if ns[min_i] == 1:
        out[j] = x
        min_i += 1
        j += 1

    if min_i == len(ns):
        return out

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
