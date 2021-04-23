"""Chebyshev polynomials."""

from .jacobi import jacobi, jacobi_sequence


def cheby1(n, x):
    """Chebyshev polynomial of the first kind of order n.

    Parameters
    ----------
    n : `int`
        order to evaluate
    x : `numpy.ndarray`
        point(s) at which to evaluate, orthogonal over [-1,1]

    """
    c = 1 / jacobi(n, -.5, -.5, 1)  # single div, many mul
    return jacobi(n, -.5, -.5, x) * c


def cheby1_sequence(ns, x):
    """Chebyshev polynomials of the first kind of orders ns.

    Faster than chevy1 in a loop.

    Parameters
    ----------
    ns : `Iterable` of `int`
        orders to evaluate
    x : `numpy.ndarray`
        point(s) at which to evaluate, orthogonal over [-1,1]

    """
    ns = list(ns)
    cs = [1/jacobi(n, -.5, -.5, 1) for n in ns]
    seq = jacobi_sequence(ns, -.5, -.5, x)
    cntr = 0
    for elem in seq:
        yield elem * cs[cntr]
        cntr += 1


def cheby2(n, x):
    """Chebyshev polynomial of the second kind of order n.

    Parameters
    ----------
    n : `int`
        order to evaluate
    x : `numpy.ndarray`
        point(s) at which to evaluate, orthogonal over [-1,1]

    """
    c = (n+1) / jacobi(n, .5, .5, 1)  # single div, many mul
    return jacobi(n, .5, .5, x) * c


def cheby2_sequence(ns, x):
    """Chebyshev polynomials of the second kind of orders ns.

    Faster than chevy1 in a loop.

    Parameters
    ----------
    ns : `Iterable` of `int`
        orders to evaluate
    x : `numpy.ndarray`
        point(s) at which to evaluate, orthogonal over [-1,1]

    """
    ns = list(ns)
    cs = [(n+1)/jacobi(n, .5, .5, 1) for n in ns]
    seq = jacobi_sequence(ns, .5, .5, x)
    cntr = 0
    for elem in seq:
        yield elem * cs[cntr]
        cntr += 1
