"""Legendre polynomials."""

from .jacobi import (
    jacobi,
    jacobi_seq,
    jacobi_der,
    jacobi_der_seq,
)


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
    return jacobi(n, 0, 0, x)


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
        has shape (len(ns), *x.shape)
        e.g., for 5 modes and x of dimension 100x100,
        return has shape (5, 100, 100)

    """
    return jacobi_seq(ns, 0, 0, x)


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
    return jacobi_der(n, 0, 0, x)


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
        has shape (len(ns), *x.shape)
        e.g., for 5 modes and x of dimension 100x100,
        return has shape (5, 100, 100)

    """
    return jacobi_der_seq(ns, 0, 0, x)
