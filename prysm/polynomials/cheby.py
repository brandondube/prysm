"""Chebyshev polynomials."""
from prysm.mathops import np

from .jacobi import (
    jacobi,
    jacobi_der,
    jacobi_seq,
    jacobi_der_seq,
)


def cheby1(n, x):
    """Chebyshev polynomial of the first kind of order n.

    Parameters
    ----------
    n : int
        order to evaluate
    x : ndarray
        point(s) at which to evaluate, orthogonal over [-1,1]

    """
    c = 1 / jacobi(n, -.5, -.5, 1)  # single div, many mul
    return jacobi(n, -.5, -.5, x) * c


def cheby1_seq(ns, x):
    """Chebyshev polynomials of the first kind of orders ns.

    Faster than chevy1 in a loop.

    Parameters
    ----------
    ns : Iterable of int
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
    ns = list(ns)
    cs = 1/jacobi_seq(ns, -.5, -.5, np.ones(1, dtype=x.dtype))
    seq = jacobi_seq(ns, -.5, -.5, x)
    # cs is (N, 1) from a 1-D jacobi argument; broadcast against seq's
    # (N, *x.shape) for any x.ndim by reshaping to (N,) + (1,) * x.ndim.
    return seq * cs.reshape((-1,) + (1,) * x.ndim)


def cheby1_der(n, x):
    """Partial derivative w.r.t. x of Chebyshev polynomial of the first kind of order n.

    Parameters
    ----------
    n : int
        order to evaluate
    x : ndarray
        point(s) at which to evaluate, orthogonal over [-1,1]

    """
    c = 1 / jacobi(n, -.5, -.5, 1)  # single div, many mul
    return jacobi_der(n, -0.5, -0.5, x) * c


def cheby1_der_seq(ns, x):
    """Partial derivative w.r.t. x of Chebyshev polynomials of the first kind of orders ns.

    Faster than chevy1_der in a loop.

    Parameters
    ----------
    ns : Iterable of int
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
    ns = list(ns)
    cs = 1/jacobi_seq(ns, -.5, -.5, np.ones(1, dtype=x.dtype))
    seq = jacobi_der_seq(ns, -.5, -.5, x)
    return seq * cs.reshape((-1,) + (1,) * x.ndim)


def cheby2(n, x):
    """Chebyshev polynomial of the second kind of order n.

    Parameters
    ----------
    n : int
        order to evaluate
    x : ndarray
        point(s) at which to evaluate, orthogonal over [-1,1]

    """
    c = (n+1) / jacobi(n, .5, .5, 1)  # single div, many mul
    return jacobi(n, .5, .5, x) * c


def cheby2_seq(ns, x):
    """Chebyshev polynomials of the second kind of orders ns.

    Faster than chevy1 in a loop.

    Parameters
    ----------
    ns : Iterable of int
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
    ns = np.asarray(ns)
    cs = (ns+1)/np.squeeze(jacobi_seq(ns, .5, .5, np.ones(1, dtype=x.dtype)))
    seq = jacobi_seq(ns, .5, .5, x)
    # cs is 1-D after squeeze; broadcast against seq's (N, *x.shape) for any
    # x.ndim by reshaping to (N,) + (1,) * x.ndim.
    return seq * cs.reshape((-1,) + (1,) * x.ndim)


def cheby2_der(n, x):
    """Partial derivative w.r.t. x of Chebyshev polynomial of the second kind of order n.

    Parameters
    ----------
    n : int
        order to evaluate
    x : ndarray
        point(s) at which to evaluate, orthogonal over [-1,1]

    """
    c = (n+1) / jacobi(n, .5, .5, 1)  # single div, many mul
    return jacobi_der(n, .5, .5, x) * c


def cheby2_der_seq(ns, x):
    """Partial derivative w.r.t. x of Chebyshev polynomials of the second kind of orders ns.

    Faster than chevy2_der in a loop.

    Parameters
    ----------
    ns : Iterable of int
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
    ns = np.asarray(ns)
    cs = (ns + 1)/np.squeeze(jacobi_seq(ns, .5, .5, np.ones(1, dtype=x.dtype)))
    seq = jacobi_der_seq(ns, .5, .5, x)
    return seq * cs.reshape((-1,) + (1,) * x.ndim)


def cheby3(n, x):
    """Chebyshev polynomial of the third kind of order n.

    Parameters
    ----------
    n : int
        order to evaluate
    x : ndarray
        point(s) at which to evaluate, orthogonal over [-1,1]

    """
    c = 1 / jacobi(n, -.5, .5, 1)  # single div, many mul
    return jacobi(n, -.5, .5, x) * c


def cheby3_seq(ns, x):
    """Chebyshev polynomials of the third kind of orders ns.

    Faster than chevy1 in a loop.

    Parameters
    ----------
    ns : Iterable of int
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
    ns = list(ns)
    cs = 1/jacobi_seq(ns, -.5, .5, np.ones(1, dtype=x.dtype))
    seq = jacobi_seq(ns, -.5, .5, x)
    return seq * cs.reshape((-1,) + (1,) * x.ndim)


def cheby3_der(n, x):
    """Partial derivative w.r.t. x of Chebyshev polynomial of the third kind of order n.

    Parameters
    ----------
    n : int
        order to evaluate
    x : ndarray
        point(s) at which to evaluate, orthogonal over [-1,1]

    """
    c = 1 / jacobi(n, -.5, .5, 1)  # single div, many mul
    return jacobi_der(n, -0.5, 0.5, x) * c


def cheby3_der_seq(ns, x):
    """Partial derivative w.r.t. x of Chebyshev polynomials of the third kind of orders ns.

    Faster than chevy1_der in a loop.

    Parameters
    ----------
    ns : Iterable of int
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
    ns = list(ns)
    cs = 1/jacobi_seq(ns, -.5, .5, np.ones(1, dtype=x.dtype))
    seq = jacobi_der_seq(ns, -.5, .5, x)
    return seq * cs.reshape((-1,) + (1,) * x.ndim)


def cheby4(n, x):
    """Chebyshev polynomial of the fourth kind of order n.

    Parameters
    ----------
    n : int
        order to evaluate
    x : ndarray
        point(s) at which to evaluate, orthogonal over [-1,1]

    """
    c = (2 * n + 1) / jacobi(n, .5, -.5, 1)  # single div, many mul
    return jacobi(n, .5, -.5, x) * c


def cheby4_seq(ns, x):
    """Chebyshev polynomials of the fourth kind of orders ns.

    Faster than chevy1 in a loop.

    Parameters
    ----------
    ns : Iterable of int
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
    ns = np.asarray(ns)
    cs = (2*ns+1)/np.squeeze(jacobi_seq(ns, .5, -.5, np.ones(1, dtype=x.dtype)))
    seq = jacobi_seq(ns, .5, -.5, x)
    return seq * cs.reshape((-1,) + (1,) * x.ndim)


def cheby4_der(n, x):
    """Partial derivative w.r.t. x of Chebyshev polynomial of the fourth kind of order n.

    Parameters
    ----------
    n : int
        order to evaluate
    x : ndarray
        point(s) at which to evaluate, orthogonal over [-1,1]

    """
    c = (2 * n + 1) / jacobi(n, .5, -.5, 1)  # single div, many mul
    return jacobi_der(n, 0.5, -0.5, x) * c


def cheby4_der_seq(ns, x):
    """Partial derivative w.r.t. x of Chebyshev polynomials of the fourth kind of orders ns.

    Faster than chevy1_der in a loop.

    Parameters
    ----------
    ns : Iterable of int
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
    ns = np.asarray(ns)
    cs = (2*ns+1)/np.squeeze(jacobi_seq(ns, .5, -.5, np.ones(1, dtype=x.dtype)))
    seq = jacobi_der_seq(ns, .5, -.5, x)
    return seq * cs.reshape((-1,) + (1,) * x.ndim)
