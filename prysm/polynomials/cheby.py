"""Chebyshev polynomials."""
from prysm.mathops import np

from ._recurrence import _seq_by_recurrence, _seq_by_recurrence_with_der


def _cheby_value(n, x, seed1):
    """Value of the shared T/U/V/W recurrence P_{k+1} = 2x P_k - P_{k-1} at order n."""
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return seed1

    Pnm2, Pnm1 = np.ones_like(x), seed1
    for _ in range(2, n + 1):
        Pnm2, Pnm1 = Pnm1, 2*x*Pnm1 - Pnm2
    return Pnm1


def _cheby_value_seq(ns, x, seed1):
    """Sequence form of _cheby_value."""
    def step(k, Pnm1, Pnm2):
        return 2*x*Pnm1 - Pnm2
    return _seq_by_recurrence(ns, x, 1, seed1, step)


def _cheby_pd_step(x):
    """Build the joint (value, derivative) step shared by all four kinds."""
    def step(k, Pnm1, Pnm2, Dnm1, Dnm2):
        Pn = 2*x*Pnm1 - Pnm2
        Dn = 2*Pnm1 + 2*x*Dnm1 - Dnm2
        return Pn, Dn
    return step


def _cheby_der(n, x, seed1, dseed1):
    """First derivative of the shared T/U/V/W recurrence at order n."""
    if n == 0:
        return np.zeros_like(x)
    if n == 1:
        return np.ones_like(x) * dseed1

    Pnm2, Pnm1 = np.ones_like(x), seed1
    Dnm2, Dnm1 = np.zeros_like(x), np.ones_like(x) * dseed1
    for _ in range(2, n + 1):
        Pn = 2*x*Pnm1 - Pnm2
        Dn = 2*Pnm1 + 2*x*Dnm1 - Dnm2
        Pnm2, Pnm1 = Pnm1, Pn
        Dnm2, Dnm1 = Dnm1, Dn
    return Dn


def _cheby_der_seq(ns, x, seed1, dseed1):
    """Sequence form of _cheby_der."""
    step = _cheby_pd_step(x)
    _, dout = _seq_by_recurrence_with_der(ns, x, 1, seed1, 0, dseed1, step)
    return dout


def cheby1(n, x):
    """Chebyshev polynomial of the first kind of order n.

    Parameters
    ----------
    n : int
        order to evaluate
    x : ndarray
        point(s) at which to evaluate, orthogonal over [-1,1]

    """
    return _cheby_value(n, x, x)


def cheby1_seq(ns, x):
    """Chebyshev polynomials of the first kind of orders ns.

    Parameters
    ----------
    ns : Iterable of int
        orders to evaluate
    x : ndarray
        point(s) at which to evaluate, orthogonal over [-1,1]

    Returns
    -------
    ndarray
        has shape (len(ns),) followed by x.shape

    """
    return _cheby_value_seq(ns, x, x)


def cheby1_der(n, x):
    """Partial derivative w.r.t. x of Chebyshev polynomial of the first kind of order n.

    Parameters
    ----------
    n : int
        order to evaluate
    x : ndarray
        point(s) at which to evaluate, orthogonal over [-1,1]

    """
    return _cheby_der(n, x, x, 1)


def cheby1_der_seq(ns, x):
    """Partial derivative w.r.t. x of Chebyshev polynomials of the first kind of orders ns.

    Parameters
    ----------
    ns : Iterable of int
        orders to evaluate
    x : ndarray
        point(s) at which to evaluate, orthogonal over [-1,1]

    Returns
    -------
    ndarray
        has shape (len(ns),) followed by x.shape

    """
    return _cheby_der_seq(ns, x, x, 1)


def cheby2(n, x):
    """Chebyshev polynomial of the second kind of order n.

    Parameters
    ----------
    n : int
        order to evaluate
    x : ndarray
        point(s) at which to evaluate, orthogonal over [-1,1]

    """
    return _cheby_value(n, x, 2*x)


def cheby2_seq(ns, x):
    """Chebyshev polynomials of the second kind of orders ns.

    Parameters
    ----------
    ns : Iterable of int
        orders to evaluate
    x : ndarray
        point(s) at which to evaluate, orthogonal over [-1,1]

    Returns
    -------
    ndarray
        has shape (len(ns),) followed by x.shape

    """
    return _cheby_value_seq(ns, x, 2*x)


def cheby2_der(n, x):
    """Partial derivative w.r.t. x of Chebyshev polynomial of the second kind of order n.

    Parameters
    ----------
    n : int
        order to evaluate
    x : ndarray
        point(s) at which to evaluate, orthogonal over [-1,1]

    """
    return _cheby_der(n, x, 2*x, 2)


def cheby2_der_seq(ns, x):
    """Partial derivative w.r.t. x of Chebyshev polynomials of the second kind of orders ns.

    Parameters
    ----------
    ns : Iterable of int
        orders to evaluate
    x : ndarray
        point(s) at which to evaluate, orthogonal over [-1,1]

    Returns
    -------
    ndarray
        has shape (len(ns),) followed by x.shape

    """
    return _cheby_der_seq(ns, x, 2*x, 2)


def cheby3(n, x):
    """Chebyshev polynomial of the third kind of order n.

    Parameters
    ----------
    n : int
        order to evaluate
    x : ndarray
        point(s) at which to evaluate, orthogonal over [-1,1]

    """
    return _cheby_value(n, x, 2*x - 1)


def cheby3_seq(ns, x):
    """Chebyshev polynomials of the third kind of orders ns.

    Parameters
    ----------
    ns : Iterable of int
        orders to evaluate
    x : ndarray
        point(s) at which to evaluate, orthogonal over [-1,1]

    Returns
    -------
    ndarray
        has shape (len(ns),) followed by x.shape

    """
    return _cheby_value_seq(ns, x, 2*x - 1)


def cheby3_der(n, x):
    """Partial derivative w.r.t. x of Chebyshev polynomial of the third kind of order n.

    Parameters
    ----------
    n : int
        order to evaluate
    x : ndarray
        point(s) at which to evaluate, orthogonal over [-1,1]

    """
    return _cheby_der(n, x, 2*x - 1, 2)


def cheby3_der_seq(ns, x):
    """Partial derivative w.r.t. x of Chebyshev polynomials of the third kind of orders ns.

    Parameters
    ----------
    ns : Iterable of int
        orders to evaluate
    x : ndarray
        point(s) at which to evaluate, orthogonal over [-1,1]

    Returns
    -------
    ndarray
        has shape (len(ns),) followed by x.shape

    """
    return _cheby_der_seq(ns, x, 2*x - 1, 2)


def cheby4(n, x):
    """Chebyshev polynomial of the fourth kind of order n.

    Parameters
    ----------
    n : int
        order to evaluate
    x : ndarray
        point(s) at which to evaluate, orthogonal over [-1,1]

    """
    return _cheby_value(n, x, 2*x + 1)


def cheby4_seq(ns, x):
    """Chebyshev polynomials of the fourth kind of orders ns.

    Parameters
    ----------
    ns : Iterable of int
        orders to evaluate
    x : ndarray
        point(s) at which to evaluate, orthogonal over [-1,1]

    Returns
    -------
    ndarray
        has shape (len(ns),) followed by x.shape

    """
    return _cheby_value_seq(ns, x, 2*x + 1)


def cheby4_der(n, x):
    """Partial derivative w.r.t. x of Chebyshev polynomial of the fourth kind of order n.

    Parameters
    ----------
    n : int
        order to evaluate
    x : ndarray
        point(s) at which to evaluate, orthogonal over [-1,1]

    """
    return _cheby_der(n, x, 2*x + 1, 2)


def cheby4_der_seq(ns, x):
    """Partial derivative w.r.t. x of Chebyshev polynomials of the fourth kind of orders ns.

    Parameters
    ----------
    ns : Iterable of int
        orders to evaluate
    x : ndarray
        point(s) at which to evaluate, orthogonal over [-1,1]

    Returns
    -------
    ndarray
        has shape (len(ns),) followed by x.shape

    """
    return _cheby_der_seq(ns, x, 2*x + 1, 2)


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
