"""Chebyshev polynomials."""
from prysm.mathops import np

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

    Returns
    -------
    ndarray
        has shape (len(ns), *x.shape)
        e.g., for 5 modes and x of dimension 100x100,
        return has shape (5, 100, 100)

    """
    ns = list(ns)
    cs = 1/jacobi_sequence(ns, -.5, -.5, np.ones(1, dtype=x.dtype))
    seq = jacobi_sequence(ns, -.5, -.5, x)
    return seq*cs


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

    Returns
    -------
    ndarray
        has shape (len(ns), *x.shape)
        e.g., for 5 modes and x of dimension 100x100,
        return has shape (5, 100, 100)

    """
    ns = list(ns)
    cs = 1/jacobi_sequence(ns, -.5, -.5, np.ones(1, dtype=x.dtype))
    seq = jacobi_der_sequence(ns, -.5, -.5, x)
    return seq*cs


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

    Returns
    -------
    ndarray
        has shape (len(ns), *x.shape)
        e.g., for 5 modes and x of dimension 100x100,
        return has shape (5, 100, 100)

    """
    # gross squeeze -> new axis dance;
    # seq is (N,M)
    # cs is (N,)
    # return of jacobi_sequence is (N,1)
    # drop the 1 to avoid broadcast to (N,N)
    # then put back 1 for compatibility on the multiply
    ns = np.asarray(ns)
    cs = (ns+1)/np.squeeze(jacobi_sequence(ns, .5, .5, np.ones(1, dtype=x.dtype)))
    seq = jacobi_sequence(ns, .5, .5, x)
    return seq*cs[:, np.newaxis]


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

    Returns
    -------
    ndarray
        has shape (len(ns), *x.shape)
        e.g., for 5 modes and x of dimension 100x100,
        return has shape (5, 100, 100)

    """
    ns = np.asarray(ns)
    cs = (ns + 1)/np.squeeze(jacobi_sequence(ns, .5, .5, np.ones(1, dtype=x.dtype)))
    seq = jacobi_der_sequence(ns, .5, .5, x)
    return seq*cs[:, np.newaxis]


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

    Returns
    -------
    ndarray
        has shape (len(ns), *x.shape)
        e.g., for 5 modes and x of dimension 100x100,
        return has shape (5, 100, 100)

    """
    ns = list(ns)
    cs = 1/jacobi_sequence(ns, -.5, .5, np.ones(1, dtype=x.dtype))
    seq = jacobi_sequence(ns, -.5, .5, x)
    return seq*cs


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

    Returns
    -------
    ndarray
        has shape (len(ns), *x.shape)
        e.g., for 5 modes and x of dimension 100x100,
        return has shape (5, 100, 100)

    """
    ns = list(ns)
    cs = 1/jacobi_sequence(ns, -.5, .5, np.ones(1, dtype=x.dtype))
    seq = jacobi_der_sequence(ns, -.5, .5, x)
    return seq*cs


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

    Returns
    -------
    ndarray
        has shape (len(ns), *x.shape)
        e.g., for 5 modes and x of dimension 100x100,
        return has shape (5, 100, 100)

    """
    ns = np.asarray(ns)
    cs = (2*ns+1)/np.squeeze(jacobi_sequence(ns, .5, -.5, np.ones(1, dtype=x.dtype)))
    seq = jacobi_sequence(ns, .5, -.5, x)
    return seq*cs[:, np.newaxis]


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

    Returns
    -------
    ndarray
        has shape (len(ns), *x.shape)
        e.g., for 5 modes and x of dimension 100x100,
        return has shape (5, 100, 100)

    """
    ns = np.asarray(ns)
    cs = (2*ns+1)/np.squeeze(jacobi_sequence(ns, .5, -.5, np.ones(1, dtype=x.dtype)))
    seq = jacobi_der_sequence(ns, .5, -.5, x)
    return seq*cs[:, np.newaxis]
