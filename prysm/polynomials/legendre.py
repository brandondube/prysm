"""Legendre polynomials."""

from .jacobi import (
    jacobi,
    jacobi_sequence,
    jacobi_der,
    jacobi_der_sequence,
)


def legendre(n, x):
    """Legendre polynomial of order n.

    Parameters
    ----------
    n : int
        order to evaluate
    x : numpy.ndarray
        point(s) at which to evaluate, orthogonal over [-1,1]

    """
    return jacobi(n, 0, 0, x)


def legendre_sequence(ns, x):
    """Legendre polynomials of orders ns.

    Faster than legendre in a loop.

    Parameters
    ----------
    ns : int
        orders to evaluate
    x : numpy.ndarray
        point(s) at which to evaluate, orthogonal over [-1,1]

    """
    return jacobi_sequence(ns, 0, 0, x)


def legendre_der(n, x):
    """Partial derivative w.r.t. x of Legendre polynomial of order n.

    Parameters
    ----------
    n : int
        order to evaluate
    x : numpy.ndarray
        point(s) at which to evaluate, orthogonal over [-1,1]

    """
    return jacobi_der(n, 0, 0, x)


def legendre_der_sequence(ns, x):
    """Partial derivative w.r.t. x of Legendre polynomials of orders ns.

    Faster than legendre_der in a loop.

    Parameters
    ----------
    ns : int
        orders to evaluate
    x : numpy.ndarray
        point(s) at which to evaluate, orthogonal over [-1,1]

    """
    return jacobi_der_sequence(ns, 0, 0, x)
