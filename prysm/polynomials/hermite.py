"""Hermite Polynomials."""

from prysm.mathops import np


def hermite_He(n, x):
    """Probabilist's Hermite polynomial He of order n at points x.

    Parameters
    ----------
    n : int
        polynomial order
    x : ndarray
        point(s) to evaluate at.  Scalars and arrays both work.

    Returns
    -------
    ndarray
        He_n(x)

    """
    # note: A, B, C = 1, 0, n
    # avoid a "recurrence_abc_He" function call on each loop to do these inline,
    # and avoiding adding zero to an array (waste work)
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return x

    # if n not in (0,1), then n >= 2
    # standard three-term recurrence relation
    # Pn+1 = (An x + Bn) Pn - Cn Pn-1
    # notation here does Pn = (An-1 ...)
    # Pnm2 = Pn-2
    # Pnm1 = Pn-1
    # i.e., starting from P2 = (A1 x + B1) P1 - C1 P0
    #
    # given that, we optimize P2 significantly by seeing that:
    # P0 = 1
    # A, B, C = 1, 0, n
    # P2 = x P1 - 1
    # -> P2 == x^2 - 1
    P2 = x * x - 1
    if n == 2:
        return P2

    Pnm2 = x
    Pnm1 = P2
    for nn in range(3, n+1):
        # it would look like this without optimization
        # Pn = (A * x + B) * Pnm1 - C * Pnm2
        # A, B, C = 1, 0, n
        Pn = x * Pnm1 - (nn-1) * Pnm2
        Pnm2, Pnm1 = Pnm1, Pn

    return Pn


def hermite_He_seq(ns, x):
    """Probabilist's Hermite polynomials He of orders ns at points x.

    Parameters
    ----------
    ns : iterable of int
        rising polynomial orders, assumed to be sorted
    x : ndarray
        point(s) to evaluate at.  Scalars and arrays both work.

    Returns
    -------
    ndarray
        has shape (len(ns), *x.shape)
        e.g., for 5 modes and x of dimension 100x100,
        return has shape (5, 100, 100)

    """
    # this function includes all the optimizations in the hermite_He func,
    # but excludes the note comments.  Read that first if you're looking for
    # clarity

    # see also: prysm.polynomials.jacobi.jacobi_seq for the meta machinery
    # in use here
    ns = list(ns)
    min_i = 0
    out = np.empty((len(ns), *x.shape), dtype=x.dtype)
    if ns[min_i] == 0:
        out[min_i] = 1
        min_i += 1

    if min_i == len(ns):
        return out

    if ns[min_i] == 1:
        out[min_i] = x
        min_i += 1

    if min_i == len(ns):
        return out

    P1 = x
    P2 = x * x - 1
    if ns[min_i] == 2:
        out[min_i] = P2
        min_i += 1

    if min_i == len(ns):
        return out

    Pnm2, Pnm1 = P1, P2
    max_n = ns[-1]
    for nn in range(3, max_n+1):
        Pn = x * Pnm1 - (nn-1) * Pnm2
        Pnm2, Pnm1 = Pnm1, Pn
        if ns[min_i] == nn:
            out[min_i] = Pn
            min_i += 1

    return out


def hermite_He_der(n, x):
    """First derivative of He_n with respect to x, at points x.

    Parameters
    ----------
    n : int
        polynomial order
    x : ndarray
        point(s) to evaluate at.  Scalars and arrays both work.

    Returns
    -------
    ndarray
        d/dx[He_n(x)]

    """
    if n == 0:
        return np.zeros_like(x)
    return n * hermite_He(n-1, x)


def hermite_He_der_seq(ns, x):
    """First derivative of He_[ns] with respect to x, at points x.

    Parameters
    ----------
    ns : iterable of int
        rising polynomial orders, assumed to be sorted
    x : ndarray
        point(s) to evaluate at.  Scalars and arrays both work.

    Returns
    -------
    ndarray
        has shape (len(ns), *x.shape)
        e.g., for 5 modes and x of dimension 100x100,
        return has shape (5, 100, 100)

    """
    # this function includes all the optimizations in the hermite_He func,
    # but excludes the note comments.  Read that first if you're looking for
    # clarity

    # see also: prysm.polynomials.jacobi.jacobi_seq for the meta machinery
    # in use here
    ns = list(ns)
    min_i = 0
    out = np.empty((len(ns), *x.shape), dtype=x.dtype)
    if ns[min_i] == 0:
        out[min_i] = 0
        min_i += 1

    if min_i == len(ns):
        return out

    if ns[min_i] == 1:
        out[min_i] = 1
        min_i += 1

    if min_i == len(ns):
        return out

    P1 = x
    P2 = x * x - 1
    if ns[min_i] == 2:
        out[min_i] = 2 * x
        min_i += 1

    if min_i == len(ns):
        return out

    Pnm2, Pnm1 = P1, P2
    max_n = ns[-1]
    for nn in range(3, max_n+1):
        Pn = x * Pnm1 - (nn-1) * Pnm2
        if ns[min_i] == nn:
            out[min_i] = nn * Pnm1
            min_i += 1

        Pnm2, Pnm1 = Pnm1, Pn

    return out


def hermite_H(n, x):
    """Physicist's Hermite polynomial H of order n at points x.

    Parameters
    ----------
    n : int
        polynomial order
    x : ndarray
        point(s) to evaluate at.  Scalars and arrays both work.

    Returns
    -------
    ndarray
        H_n(x)

    """
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 2 * x

    # if n not in (0,1), then n >= 2
    # standard three-term recurrence relation
    # Pn+1 = (An x + Bn) Pn - Cn Pn-1
    # notation here does Pn = (An-1 ...)
    # Pnm2 = Pn-2
    # Pnm1 = Pn-1
    # i.e., starting from P2 = (A1 x + B1) P1 - C1 P0
    #
    # given that, we optimize P2 significantly by seeing that:
    # P0 = 1
    # A, B, C = 2, 0, 2n
    # P2 = x P1 - 1
    # -> P2 == 4^2 - 2

    # another optimization: Ax == 2x == P1, so we compute that just once
    x2 = 2 * x
    P2 = 4 * (x * x) - 2
    if n == 2:
        return P2

    Pnm2 = x2
    Pnm1 = P2
    for nn in range(3, n+1):
        # it would look like this without optimization
        # Pn = (A * x + B) * Pnm1 - C * Pnm2
        # A, B, C = 2, 0, 2n
        Pn = x2 * Pnm1 - (2 * (nn-1)) * Pnm2
        Pnm2, Pnm1 = Pnm1, Pn

    return Pn


def hermite_H_seq(ns, x):
    """Physicist's Hermite polynomials H of orders ns at points x.

    Parameters
    ----------
    ns : iterable of int
        rising polynomial orders, assumed to be sorted
    x : ndarray
        point(s) to evaluate at.  Scalars and arrays both work.

    Returns
    -------
    ndarray
        has shape (len(ns), *x.shape)
        e.g., for 5 modes and x of dimension 100x100,
        return has shape (5, 100, 100)

    """
    # this function includes all the optimizations in the hermite_He func,
    # but excludes the note comments.  Read that first if you're looking for
    # clarity

    # see also: prysm.polynomials.jacobi.jacobi_seq for the meta machinery
    # in use here
    ns = list(ns)
    min_i = 0
    out = np.empty((len(ns), *x.shape), dtype=x.dtype)
    if ns[min_i] == 0:
        out[min_i] = 1
        min_i += 1

    if min_i == len(ns):
        return out

    x2 = 2 * x
    if ns[min_i] == 1:
        out[min_i] = x2
        min_i += 1

    if min_i == len(ns):
        return out

    P1 = x2
    P2 = 4 * (x * x) - 2
    if ns[min_i] == 2:
        out[min_i] = P2
        min_i += 1

    if min_i == len(ns):
        return out

    Pnm2, Pnm1 = P1, P2
    max_n = ns[-1]
    for nn in range(3, max_n+1):
        Pn = x2 * Pnm1 - (2*(nn-1)) * Pnm2
        Pnm2, Pnm1 = Pnm1, Pn
        if ns[min_i] == nn:
            out[min_i] = Pn
            min_i += 1

    return out


def hermite_H_der(n, x):
    """First derivative of H_n with respect to x, at points x.

    Parameters
    ----------
    n : int
        polynomial order
    x : ndarray
        point(s) to evaluate at.  Scalars and arrays both work.

    Returns
    -------
    ndarray
        d/dx[H_n(x)]

    """
    if n == 0:
        return np.zeros_like(x)
    return 2 * n * hermite_H(n-1, x)


def hermite_H_der_seq(ns, x):
    """First derivative of He_[ns] with respect to x, at points x.

    Parameters
    ----------
    ns : iterable of int
        rising polynomial orders, assumed to be sorted
    x : ndarray
        point(s) to evaluate at.  Scalars and arrays both work.

    Returns
    -------
    ndarray
        has shape (len(ns), *x.shape)
        e.g., for 5 modes and x of dimension 100x100,
        return has shape (5, 100, 100)

    """
    # this function includes all the optimizations in the hermite_He func,
    # but excludes the note comments.  Read that first if you're looking for
    # clarity

    # see also: prysm.polynomials.jacobi.jacobi_seq for the meta machinery
    # in use here
    ns = list(ns)
    min_i = 0
    out = np.empty((len(ns), *x.shape), dtype=x.dtype)
    if ns[min_i] == 0:
        out[min_i] = 0
        min_i += 1

    if min_i == len(ns):
        return out

    if ns[min_i] == 1:
        out[min_i] = 2
        min_i += 1

    if min_i == len(ns):
        return out

    x2 = 2 * x
    P1 = x2
    P2 = 4 * (x * x) - 2
    if ns[min_i] == 2:
        out[min_i] = 4 * P1
        min_i += 1

    if min_i == len(ns):
        return out

    Pnm2, Pnm1 = P1, P2
    max_n = ns[-1]
    for nn in range(3, max_n+1):
        Pn = x2 * Pnm1 - (2*(nn-1)) * Pnm2
        if ns[min_i] == nn:
            out[min_i] = 2 * nn * Pnm1
            min_i += 1

        Pnm2, Pnm1 = Pnm1, Pn

    return out
