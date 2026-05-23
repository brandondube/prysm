"""XY polynomials."""

# truenp: host-side j ↔ (m, n) index conversion and the small `mns2` array
#         used to sort coefficient tables; both are Python-loop bookkeeping.
import numpy as truenp

from prysm.mathops import np  # NOQA
from prysm.coordinates import optimize_xy_separable


def xy_j_to_mn(j):
    """Convert a mono-index j into the m and n powers.

    Does not precisely follow Code V; the j=1 term is piston, which does not
    exist in Code V.

    """
    if j < 1:
        raise ValueError('j must be >= 1')
    if j == 1:
        return 0, 0

    total_order = int(truenp.ceil((truenp.sqrt(8*j + 1) - 3) / 2))
    first_j = total_order * (total_order + 1) // 2 + 1
    y_order = j - first_j
    x_order = total_order - y_order
    return x_order, y_order


def xy(m, n, x, y, cartesian_grid=True):
    """Contemporary XY monomial for a given m, n.

    Parameters
    ----------
    m : int
        x order
    n : int
        y order
    x : ndarray
        x coordinates
    y : ndarray
        y coordinates
    cartesian_grid : bool, optional
        if True, the input grid is assumed to be cartesian, i.e., x and y
        axes are aligned to the array dimensions arr[y,x] to accelerate
        the computation

    Returns
    -------
    ndarray
        x^m times y^n evaluated on the input grid

    """
    if cartesian_grid:
        x, y = optimize_xy_separable(x, y)

    return x**m * y**n


def xy_der_x(m, n, x, y, cartesian_grid=True):
    """Partial derivative w.r.t. x of the XY monomial x^m times y^n.

    Returns m times x^(m-1) times y^n; zero everywhere when m == 0.

    Parameters
    ----------
    m : int
        x order
    n : int
        y order
    x : ndarray
        x coordinates
    y : ndarray
        y coordinates
    cartesian_grid : bool, optional
        if True, the input grid is assumed to be cartesian, i.e., x and y
        axes are aligned to the array dimensions arr[y,x] to accelerate
        the computation

    Returns
    -------
    ndarray
        d/dx of x^m times y^n evaluated on the input grid

    """
    if cartesian_grid:
        x, y = optimize_xy_separable(x, y)

    if m == 0:
        # broadcast zeros to the (y, x) grid using the input shapes
        return np.zeros_like(x * y)

    return m * x**(m-1) * y**n


def xy_der_y(m, n, x, y, cartesian_grid=True):
    """Partial derivative w.r.t. y of the XY monomial x^m * y^n.

    Returns n times x^m times y^(n-1); zero everywhere when n == 0.

    Parameters
    ----------
    m : int
        x order
    n : int
        y order
    x : ndarray
        x coordinates
    y : ndarray
        y coordinates
    cartesian_grid : bool, optional
        if True, the input grid is assumed to be cartesian, i.e., x and y
        axes are aligned to the array dimensions arr[y,x] to accelerate
        the computation

    Returns
    -------
    ndarray
        d/dy of x^m times y^n evaluated on the input grid

    """
    if cartesian_grid:
        x, y = optimize_xy_separable(x, y)

    if n == 0:
        return np.zeros_like(x * y)

    return n * x**m * y**(n-1)


def xy_der_xy(m, n, x, y, cartesian_grid=True):
    """Mixed partial derivative d^2/dxdy of the XY monomial x^m * y^n.

    Returns m times n times x^(m-1) times y^(n-1); zero everywhere when m == 0 or n == 0.

    Parameters
    ----------
    m : int
        x order
    n : int
        y order
    x : ndarray
        x coordinates
    y : ndarray
        y coordinates
    cartesian_grid : bool, optional
        if True, the input grid is assumed to be cartesian, i.e., x and y
        axes are aligned to the array dimensions arr[y,x] to accelerate
        the computation

    Returns
    -------
    ndarray
        d^2/dxdy of x^m times y^n evaluated on the input grid

    """
    if cartesian_grid:
        x, y = optimize_xy_separable(x, y)

    if m == 0 or n == 0:
        return np.zeros_like(x * y)

    return (m * n) * x**(m-1) * y**(n-1)


def _xy_seq_with(mns, x, y, cartesian_grid, x_powers_op, y_powers_op):
    """Internal shared engine for xy_seq / xy_der_*_seq.

    x_powers_op(maxm, x) returns the list of x-axis terms — either
    [1, x, x^2, ..., x^maxm] for plain monomials or [0, 1, 2x, ...,
    maxm*x^(maxm-1)] for the d/dx variant.  y_powers_op likewise for the
    y-axis.  The mixed factor m*n for the xy mixed partial falls out of
    multiplying the two derivative tables together.
    """
    mns2 = truenp.asarray(mns)
    maxm, maxn = mns2.max(axis=0)

    if cartesian_grid and x.ndim > 1:
        x, y = optimize_xy_separable(x, y)

    x_seq = x_powers_op(maxm, x)
    y_seq = y_powers_op(maxn, y)

    m0, n0 = mns2[0]
    first = x_seq[m0] * y_seq[n0]
    out = np.empty((len(mns2), *first.shape), dtype=first.dtype)
    out[0] = first
    for j, (m, n) in enumerate(mns2[1:], start=1):
        out[j] = x_seq[m] * y_seq[n]

    return out


def _monomial_seq(maxk, z):
    """List [z^0, z^1, ..., z^maxk]; cumulatively multiplied."""
    out = [np.ones_like(z)]
    current = None
    for _ in range(1, maxk + 1):
        current = z if current is None else current * z
        out.append(current)
    return out


def _monomial_der_seq(maxk, z):
    """List [d/dz z^0, d/dz z^1, ..., d/dz z^maxk] = [0, 1, 2z, 3z^2, ...]."""
    out = [np.zeros_like(z)]
    if maxk == 0:
        return out
    out.append(np.ones_like(z))
    current = None
    for k in range(2, maxk + 1):
        # k * z^(k-1)
        current = z if current is None else current * z
        out.append(k * current)
    return out


def xy_seq(mns, x, y, cartesian_grid=True):
    """Contemporary XY monomial seq.

    Parameters
    ----------
    mns : iterable of length 2 vectors
        seq [(m1, n1), (m2, n2), ...]
    x : ndarray
        x coordinates
    y : ndarray
        y coordinates
    cartesian_grid : bool, optional
        if True, the input grid is assumed to be cartesian, i.e., x and y
        axes are aligned to the array dimensions arr[y,x] to accelerate
        the computation

    Returns
    -------
    ndarray
        has shape (len(mns), broadcast(x, y).shape), in the same order as mns

    """
    return _xy_seq_with(
        mns, x, y, cartesian_grid,
        _monomial_seq, _monomial_seq,
    )


def xy_der_x_seq(mns, x, y, cartesian_grid=True):
    """Partial derivative w.r.t. x of the XY monomial seq.

    Parameters mirror xy_seq.  The (m, n) output is m times x^(m-1) times y^n;
    entries with m == 0 are zero.

    Returns
    -------
    ndarray
        has shape (len(mns), broadcast(x, y).shape); d/dx of x^m times y^n in
        the same order as mns

    """
    return _xy_seq_with(
        mns, x, y, cartesian_grid,
        _monomial_der_seq, _monomial_seq,
    )


def xy_der_y_seq(mns, x, y, cartesian_grid=True):
    """Partial derivative w.r.t. y of the XY monomial seq.

    Parameters mirror xy_seq.  The (m, n) output is n times x^m times y^(n-1);
    entries with n == 0 are zero.

    Returns
    -------
    ndarray
        has shape (len(mns), broadcast(x, y).shape); d/dy of x^m times y^n in
        the same order as mns

    """
    return _xy_seq_with(
        mns, x, y, cartesian_grid,
        _monomial_seq, _monomial_der_seq,
    )


def xy_der_xy_seq(mns, x, y, cartesian_grid=True):
    """Mixed partial derivative d^2/dxdy of the XY monomial seq.

    Parameters mirror xy_seq.  The (m, n) output is m*n times x^(m-1) times y^(n-1);
    entries where m == 0 or n == 0 are zero.

    Returns
    -------
    ndarray
        has shape (len(mns), broadcast(x, y).shape); d^2/dxdy of x^m times y^n
        in the same order as mns

    """
    return _xy_seq_with(
        mns, x, y, cartesian_grid,
        _monomial_der_seq, _monomial_der_seq,
    )


def xy_sum(coefs, mns, x, y, cartesian_grid=True):
    """Evaluate a weighted sum of XY monomials."""
    mns = tuple(mns)
    if not mns:
        return np.zeros_like(x)
    modes = xy_seq(mns, x, y, cartesian_grid=cartesian_grid)
    return np.tensordot(np.asarray(coefs, dtype=modes.dtype), modes, axes=1)


def xy_sum_der_xy(coefs, mns, x, y, cartesian_grid=True):
    """Evaluate a weighted XY sum and its Cartesian first derivatives."""
    mns = tuple(mns)
    if not mns:
        z = np.zeros_like(x)
        return z, z, np.zeros_like(y)
    coefs = np.asarray(coefs)
    modes = xy_seq(mns, x, y, cartesian_grid=cartesian_grid)
    dx_modes = xy_der_x_seq(mns, x, y, cartesian_grid=cartesian_grid)
    dy_modes = xy_der_y_seq(mns, x, y, cartesian_grid=cartesian_grid)
    if coefs.dtype != modes.dtype:
        coefs = coefs.astype(modes.dtype)
    z = np.tensordot(coefs, modes, axes=1)
    dzdx = np.tensordot(coefs, dx_modes, axes=1)
    dzdy = np.tensordot(coefs, dy_modes, axes=1)
    return z, dzdx, dzdy
