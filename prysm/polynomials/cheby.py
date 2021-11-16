"""Chebyshev polynomials."""

from .jacobi import (
    jacobi,
    jacobi_der,
    jacobi_sequence,
    jacobi_der_sequence,
)


def cheby1(n, x):
    """Chebyshev polynomial of the first kind of order n.

    Parameters
    ----------
    n : int
        order to evaluate
    x : numpy.ndarray
        point(s) at which to evaluate, orthogonal over [-1,1]

    """
    c = 1 / jacobi(n, -.5, -.5, 1)  # single div, many mul
    return jacobi(n, -.5, -.5, x) * c


def cheby1_sequence(ns, x):
    """Chebyshev polynomials of the first kind of orders ns.

    Faster than chevy1 in a loop.

    Parameters
    ----------
    ns : Iterable of int
        orders to evaluate
    x : numpy.ndarray
        point(s) at which to evaluate, orthogonal over [-1,1]

    """
    ns = list(ns)
    cs = [1/jacobi(n, -.5, -.5, 1) for n in ns]
    seq = jacobi_sequence(ns, -.5, -.5, x)
    cntr = 0
    for elem in seq:
        yield elem * cs[cntr]
        cntr += 1


def cheby1_der(n, x):
    """Partial derivative w.r.t. x of Chebyshev polynomial of the first kind of order n.

    Parameters
    ----------
    n : int
        order to evaluate
    x : numpy.ndarray
        point(s) at which to evaluate, orthogonal over [-1,1]

    """
    c = 1 / jacobi(n, -.5, -.5, 1)  # single div, many mul
    return jacobi_der(n, -0.5, -0.5, x) * c


def cheby1_der_sequence(ns, x):
    """Partial derivative w.r.t. x of Chebyshev polynomials of the first kind of orders ns.

    Faster than chevy1_der in a loop.

    Parameters
    ----------
    ns : Iterable of int
        orders to evaluate
    x : numpy.ndarray
        point(s) at which to evaluate, orthogonal over [-1,1]

    """
    ns = list(ns)
    cs = [1/jacobi(n, -.5, -.5, 1) for n in ns]
    seq = jacobi_der_sequence(ns, -.5, -.5, x)
    cntr = 0
    for elem in seq:
        yield elem * cs[cntr]
        cntr += 1


def cheby2(n, x):
    """Chebyshev polynomial of the second kind of order n.

    Parameters
    ----------
    n : int
        order to evaluate
    x : numpy.ndarray
        point(s) at which to evaluate, orthogonal over [-1,1]

    """
    c = (n+1) / jacobi(n, .5, .5, 1)  # single div, many mul
    return jacobi(n, .5, .5, x) * c


def cheby2_sequence(ns, x):
    """Chebyshev polynomials of the second kind of orders ns.

    Faster than chevy1 in a loop.

    Parameters
    ----------
    ns : Iterable of int
        orders to evaluate
    x : numpy.ndarray
        point(s) at which to evaluate, orthogonal over [-1,1]

    """
    ns = list(ns)
    cs = [(n+1)/jacobi(n, .5, .5, 1) for n in ns]
    seq = jacobi_sequence(ns, .5, .5, x)
    cntr = 0
    for elem in seq:
        yield elem * cs[cntr]
        cntr += 1



def cheby2_der(n, x):
    """Partial derivative w.r.t. x of Chebyshev polynomial of the second kind of order n.

    Parameters
    ----------
    n : int
        order to evaluate
    x : numpy.ndarray
        point(s) at which to evaluate, orthogonal over [-1,1]

    """
    c = (n+1) / jacobi(n, .5, .5, 1)  # single div, many mul
    return jacobi_der(n, .5, .5, x) * c


def cheby2_der_sequence(ns, x):
    """Partial derivative w.r.t. x of Chebyshev polynomials of the second kind of orders ns.

    Faster than chevy2_der in a loop.

    Parameters
    ----------
    ns : Iterable of int
        orders to evaluate
    x : numpy.ndarray
        point(s) at which to evaluate, orthogonal over [-1,1]

    """
    ns = list(ns)
    cs = [(n+1) / jacobi(n, .5, .5, 1) for n in ns]
    seq = jacobi_der_sequence(ns, .5, .5, x)
    cntr = 0
    for elem in seq:
        yield elem * cs[cntr]
        cntr += 1


def cheby3(n, x):
    """Chebyshev polynomial of the third kind of order n.

    Parameters
    ----------
    n : int
        order to evaluate
    x : numpy.ndarray
        point(s) at which to evaluate, orthogonal over [-1,1]

    """
    c = 1 / jacobi(n, -.5, .5, 1)  # single div, many mul
    return jacobi(n, -.5, .5, x) * c


def cheby3_sequence(ns, x):
    """Chebyshev polynomials of the third kind of orders ns.

    Faster than chevy1 in a loop.

    Parameters
    ----------
    ns : Iterable of int
        orders to evaluate
    x : numpy.ndarray
        point(s) at which to evaluate, orthogonal over [-1,1]

    """
    ns = list(ns)
    cs = [1/jacobi(n, -.5, .5, 1) for n in ns]
    seq = jacobi_sequence(ns, -.5, .5, x)
    cntr = 0
    for elem in seq:
        yield elem * cs[cntr]
        cntr += 1


def cheby3_der(n, x):
    """Partial derivative w.r.t. x of Chebyshev polynomial of the third kind of order n.

    Parameters
    ----------
    n : int
        order to evaluate
    x : numpy.ndarray
        point(s) at which to evaluate, orthogonal over [-1,1]

    """
    c = 1 / jacobi(n, -.5, .5, 1)  # single div, many mul
    return jacobi_der(n, -0.5, 0.5, x) * c


def cheby3_der_sequence(ns, x):
    """Partial derivative w.r.t. x of Chebyshev polynomials of the third kind of orders ns.

    Faster than chevy1_der in a loop.

    Parameters
    ----------
    ns : Iterable of int
        orders to evaluate
    x : numpy.ndarray
        point(s) at which to evaluate, orthogonal over [-1,1]

    """
    ns = list(ns)
    cs = [1/jacobi(n, -.5, .5, 1) for n in ns]
    seq = jacobi_der_sequence(ns, -.5, .5, x)
    cntr = 0
    for elem in seq:
        yield elem * cs[cntr]
        cntr += 1


def cheby4(n, x):
    """Chebyshev polynomial of the fourth kind of order n.

    Parameters
    ----------
    n : int
        order to evaluate
    x : numpy.ndarray
        point(s) at which to evaluate, orthogonal over [-1,1]

    """
    c = (2 * n + 1) / jacobi(n, .5, -.5, 1)  # single div, many mul
    return jacobi(n, .5, -.5, x) * c


def cheby4_sequence(ns, x):
    """Chebyshev polynomials of the fourth kind of orders ns.

    Faster than chevy1 in a loop.

    Parameters
    ----------
    ns : Iterable of int
        orders to evaluate
    x : numpy.ndarray
        point(s) at which to evaluate, orthogonal over [-1,1]

    """
    ns = list(ns)
    cs = [(2 * n + 1) / jacobi(n, .5, -.5, 1) for n in ns]
    seq = jacobi_sequence(ns, .5, -.5, x)
    cntr = 0
    for elem in seq:
        yield elem * cs[cntr]
        cntr += 1


def cheby4_der(n, x):
    """Partial derivative w.r.t. x of Chebyshev polynomial of the fourth kind of order n.

    Parameters
    ----------
    n : int
        order to evaluate
    x : numpy.ndarray
        point(s) at which to evaluate, orthogonal over [-1,1]

    """
    c = (2 * n + 1) / jacobi(n, .5, -.5, 1)  # single div, many mul
    return jacobi_der(n, 0.5, -0.5, x) * c


def cheby4_der_sequence(ns, x):
    """Partial derivative w.r.t. x of Chebyshev polynomials of the fourth kind of orders ns.

    Faster than chevy1_der in a loop.

    Parameters
    ----------
    ns : Iterable of int
        orders to evaluate
    x : numpy.ndarray
        point(s) at which to evaluate, orthogonal over [-1,1]

    """
    ns = list(ns)
    cs = [(2 * n + 1) / jacobi(n, .5, -.5, 1) for n in ns]
    seq = jacobi_der_sequence(ns, .5, -.5, x)
    cntr = 0
    for elem in seq:
        yield elem * cs[cntr]
        cntr += 1
