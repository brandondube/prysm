"""Chebyshev polynomials."""
from prysm.mathops import np

from .jacobi import (
    jacobi,
    jacobi_der,
    jacobi_seq,
    jacobi_der_seq,
)


def _cheby_inv_jacobi_at_one(ns, alpha, beta, dtype):
    """Return 1 / P_n^(alpha,beta)(1) for each n in ns, shape (len(ns),).

    The cheby families differ only in which (alpha, beta) they pull and what
    scalar coefficient they fold on top; this helper isolates the shared
    jacobi-at-1 normalization that every cheby*_seq / cheby*_der_seq performs.
    """
    pn_at_one = jacobi_seq(ns, alpha, beta, np.ones(1, dtype=dtype))[:, 0]
    return 1 / pn_at_one


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
    ns = np.asarray(ns)
    cs = _cheby_inv_jacobi_at_one(ns, -.5, -.5, x.dtype)
    seq = jacobi_seq(ns, -.5, -.5, x)
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
    ns = np.asarray(ns)
    cs = _cheby_inv_jacobi_at_one(ns, -.5, -.5, x.dtype)
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
    cs = (ns + 1) * _cheby_inv_jacobi_at_one(ns, .5, .5, x.dtype)
    seq = jacobi_seq(ns, .5, .5, x)
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
    cs = (ns + 1) * _cheby_inv_jacobi_at_one(ns, .5, .5, x.dtype)
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
    ns = np.asarray(ns)
    cs = _cheby_inv_jacobi_at_one(ns, -.5, .5, x.dtype)
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
    ns = np.asarray(ns)
    cs = _cheby_inv_jacobi_at_one(ns, -.5, .5, x.dtype)
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
    cs = (2 * ns + 1) * _cheby_inv_jacobi_at_one(ns, .5, -.5, x.dtype)
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
    cs = (2 * ns + 1) * _cheby_inv_jacobi_at_one(ns, .5, -.5, x.dtype)
    seq = jacobi_der_seq(ns, .5, -.5, x)
    return seq * cs.reshape((-1,) + (1,) * x.ndim)


def cheby1_2d_sum(coefs, mns, x, y):
    """Evaluate a weighted tensor-product Chebyshev-T sum."""
    mns = tuple(mns)
    if not mns:
        return np.zeros_like(x)
    max_m = max(m for m, _ in mns)
    max_n = max(n for _, n in mns)
    Tx = cheby1_seq(range(max_m + 1), x)
    Ty = cheby1_seq(range(max_n + 1), y)
    z = np.zeros_like(x)
    for c, (m, n) in zip(coefs, mns):
        if c == 0.0:
            continue
        z = z + c * Tx[m] * Ty[n]
    return z


def cheby1_2d_sum_der_xy(coefs, mns, x, y, x_norm=1.0, y_norm=1.0):
    """Evaluate a weighted Chebyshev-T sum and Cartesian derivatives."""
    mns = tuple(mns)
    if not mns:
        z = np.zeros_like(x)
        return z, z, np.zeros_like(y)
    max_m = max(m for m, _ in mns)
    max_n = max(n for _, n in mns)
    Tx = cheby1_seq(range(max_m + 1), x)
    Ty = cheby1_seq(range(max_n + 1), y)
    Tx_d = cheby1_der_seq(range(max_m + 1), x)
    Ty_d = cheby1_der_seq(range(max_n + 1), y)
    z = np.zeros_like(x)
    dzdx = np.zeros_like(x)
    dzdy = np.zeros_like(x)
    for c, (m, n) in zip(coefs, mns):
        if c == 0.0:
            continue
        z = z + c * Tx[m] * Ty[n]
        dzdx = dzdx + c * Tx_d[m] * Ty[n]
        dzdy = dzdy + c * Tx[m] * Ty_d[n]
    return z, dzdx / x_norm, dzdy / y_norm
