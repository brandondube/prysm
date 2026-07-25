"""Legendre polynomials."""
from prysm.mathops import np

from ._recurrence import _seq_by_recurrence, _seq_by_recurrence_with_der


def legendre(n, x):
    """Legendre polynomial of order n.

    Parameters
    ----------
    n : int
        order to evaluate
    x : ndarray
        point(s) at which to evaluate, orthogonal over [-1,1]

    Returns
    -------
    ndarray
        legendre polynomial evaluated at the given points

    """
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return x

    Pnm2, Pnm1 = np.ones_like(x), x
    for k in range(2, n + 1):
        Pnm2, Pnm1 = Pnm1, ((2*k-1)*x*Pnm1 - (k-1)*Pnm2) / k
    return Pnm1


def legendre_seq(ns, x):
    """Legendre polynomials of orders ns.

    Faster than legendre in a loop.

    Parameters
    ----------
    ns : int
        orders to evaluate
    x : ndarray
        point(s) at which to evaluate, orthogonal over [-1,1]

    Returns
    -------
    ndarray
        has shape (len(ns),) followed by x.shape
        e.g., for 5 modes and x of dimension 100x100,
        return has shape (5, 100, 100)

    """
    def step(k, Pnm1, Pnm2):
        return ((2*k-1)*x*Pnm1 - (k-1)*Pnm2) / k

    return _seq_by_recurrence(ns, x, 1, x, step)


def legendre_der(n, x):
    """Partial derivative w.r.t. x of Legendre polynomial of order n.

    Parameters
    ----------
    n : int
        order to evaluate
    x : ndarray
        point(s) at which to evaluate, orthogonal over [-1,1]

    Returns
    -------
    ndarray
        d/dx of legendre polynomial evaluated at the given points

    """
    if n == 0:
        return np.zeros_like(x)
    if n == 1:
        return np.ones_like(x)

    Pnm2, Pnm1 = np.ones_like(x), x
    Dnm2, Dnm1 = np.zeros_like(x), np.ones_like(x)
    for k in range(2, n + 1):
        Pn = ((2*k-1)*x*Pnm1 - (k-1)*Pnm2) / k
        Dn = ((2*k-1)*(Pnm1 + x*Dnm1) - (k-1)*Dnm2) / k
        Pnm2, Pnm1 = Pnm1, Pn
        Dnm2, Dnm1 = Dnm1, Dn
    return Dnm1


def legendre_der_seq(ns, x):
    """Partial derivative w.r.t. x of Legendre polynomials of orders ns.

    Faster than legendre_der in a loop.

    Parameters
    ----------
    ns : int
        orders to evaluate
    x : ndarray
        point(s) at which to evaluate, orthogonal over [-1,1]

    Returns
    -------
    ndarray
        has shape (len(ns),) followed by x.shape
        e.g., for 5 modes and x of dimension 100x100,
        return has shape (5, 100, 100)

    """
    def step(k, Pnm1, Pnm2, Dnm1, Dnm2):
        Pn = ((2*k-1)*x*Pnm1 - (k-1)*Pnm2) / k
        Dn = ((2*k-1)*(Pnm1 + x*Dnm1) - (k-1)*Dnm2) / k
        return Pn, Dn

    _, dout = _seq_by_recurrence_with_der(ns, x, 1, x, 0, 1, step)
    return dout
