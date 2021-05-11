"""Legendre polynomials."""

from .jacobi import jacobi, jacobi_sequence


def legendre(n, x):
    """Legendre polynomial of order n.

    Parameters
    ----------
    n : `int`
        order to evaluate
    x : `numpy.ndarray`
        point(s) at which to evaluate, orthogonal over [-1,1]

    """
    return jacobi(n, 0, 0, x)


def legendre_sequence(ns, x):
    """Legendre polynomials of orders ns.

    Faster than chevy1 in a loop.

    Parameters
    ----------
    ns : `int`
        orders to evaluate
    x : `numpy.ndarray`
        point(s) at which to evaluate, orthogonal over [-1,1]

    """
    return jacobi_sequence(ns, 0, 0, x)
