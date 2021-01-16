"""Chebyshev polynomials."""

from .jacobi import jacobi, jacobi_sequence

from prysm.coordinates import optimize_xy_separable


def cheby1(n, x):
    """Chebyshev polynomial of the first kind of order n.

    Parameters
    ----------
    n : `int`
        order to evaluate
    x : `numpy.ndarray`
        point(s) at which to evaluate, orthogonal over [-1,1]

    """
    return jacobi(n, -.5, -.5, x)


def cheby1_sequence(ns, x):
    """Chebyshev polynomials of the first kind of orders ns.

    Faster than chevy1 in a loop.

    Parameters
    ----------
    ns : `int`
        orders to evaluate
    x : `numpy.ndarray`
        point(s) at which to evaluate, orthogonal over [-1,1]

    """
    return jacobi_sequence(ns, -.5, -.5, x)


def cheby2(n, x):
    """Chebyshev polynomial of the second kind of order n.

    Parameters
    ----------
    n : `int`
        order to evaluate
    x : `numpy.ndarray`
        point(s) at which to evaluate, orthogonal over [-1,1]

    """
    return jacobi(n, .5, .5, x)


def cheby2_sequence(ns, x):
    """Chebyshev polynomials of the second kind of orders ns.

    Faster than chevy1 in a loop.

    Parameters
    ----------
    ns : `int`
        orders to evaluate
    x : `numpy.ndarray`
        point(s) at which to evaluate, orthogonal over [-1,1]

    """
    return jacobi_sequence(ns, .5, .5, x)


def cheby1_2d_sequence(ns, ms, x, y):
    """Chebyshev polynomials of the first kind in both X and Y (as for a rectangular aperture).

    Parameters
    ----------
    ns : iterable of `int`
        orders n for the x axis
    ms : iterable of `int`
        orders m for the y axis
    x : `numpy.ndarray`
        x coordinates, 1D or 2D
    y : `numpy.ndarray`
        y coordinates, 1D or 2D

    Returns
    -------
    `list`, `list` [x, y] modes, with each of 'x' and 'y' in the return being
        a list of its own containing 1D modes

    """
    x, y = optimize_xy_separable(x, y)
    xs = list(jacobi_sequence(ns, -.5, -.5, x))
    ys = list(jacobi_sequence(ms, -.5, -.5, y))
    return xs, ys


def cheby2_2d_sequence(ns, ms, x, y):
    """Chebyshev polynomials of the second kind in both X and Y (as for a rectangular aperture).

    Parameters
    ----------
    ns : iterable of `int`
        orders n for the x axis
    ms : iterable of `int`
        orders m for the y axis
    x : `numpy.ndarray`
        x coordinates, 1D or 2D
    y : `numpy.ndarray`
        y coordinates, 1D or 2D

    Returns
    -------
    `list`, `list` [x, y] modes, with each of 'x' and 'y' in the return being
        a list of its own containing 1D modes

    """
    x, y = optimize_xy_separable(x, y)
    xs = list(jacobi_sequence(ns, .5, .5, x))
    ys = list(jacobi_sequence(ms, .5, .5, y))
    return xs, ys
