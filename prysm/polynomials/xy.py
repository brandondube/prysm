"""XY polynomials."""

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
    if j == 2:
        return 1, 0
    if j == 3:
        return 0, 1

    # exerpt from the Code V manual:
    # +-----+---------+----------+-----------+-----------+-----------+-----------+
    # |     | X0      | X1       | X2        | X3        | X4        | X5        |
    # +-----+---------+----------+-----------+-----------+-----------+-----------+
    # |     |         |          |           |           |           |           |
    # | Y0  |         | X C2     | X2 C4     | X3 C7     | X4 C11    | X5 C16    |
    # |     |         |          |           |           |           |           |
    # | Y1  | Y C3    | XY C5    | X2Y C8    | X3Y C12   | X4Y C17   | X5Y C23   |
    # |     |         |          |           |           |           |           |
    # | Y2  | Y2 C6   | XY2 C9   | X2Y2 C13  | X3Y2 C18  | X4Y2 C24  | X5Y2 C31  |
    # |     |         |          |           |           |           |           |
    # | Y3  | Y3 C10  | XY3 C14  | X2Y3 C19  | X3Y3 C25  | X4Y3 C32  | X5Y3 C40  |
    # |     |         |          |           |           |           |           |
    # | Y4  | Y4 C15  | XY4 C20  | X2Y4 C26  | X3Y4 C33  | X4Y4 C41  | X5Y3 C50  |
    # |     |         |          |           |           |           |           |
    # | Y5  | Y5 C21  | XY5 C27  | X2Y5 C34  | X3Y5 C42  | X4Y5 C51  | X5Y5 C61  |
    # |     |         |          |           |           |           |           |
    # +-----+---------+----------+-----------+-----------+-----------+-----------+

    # strategy: find the maximum dimension j would support,
    # then search efficiently in that matrix

    # number of elements in a triangular matrix of dimension k
    # without diagonal k(k-1)/2
    # with    diagonal k(k+1)/2

    # for j>3, dimension >= 3
    # TODO: this can be made more efficient with a heuristic
    # perhaps advance k by 2 and then seek one down if we miss
    k = 2
    max_j = k*(k+1)//2
    while max_j < j:
        max_j = k*(k+1)//2
        k += 1

    largest_pure_y_term = max_j
    largest_pure_x_term = max_j - k + 2

    diffy = abs(j-largest_pure_y_term)
    diffx = abs(j-largest_pure_x_term)

    if diffy < diffx:
        # iterate up and to the right
        x = 0
        y = k-2
        jj = largest_pure_y_term
        while jj != j:
            jj -= 1
            x += 1
            y -= 1
    else:
        # iterate down and to the left
        x = k-2
        y = 0
        jj = largest_pure_x_term
        while jj != j:
            jj += 1
            x -= 1
            y += 1

    return x, y


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
        x^m * y^n evaluated on the input grid

    """
    if cartesian_grid:
        x, y = optimize_xy_separable(x, y)

    return x**m * y**n


def xy_der_x(m, n, x, y, cartesian_grid=True):
    """Partial derivative w.r.t. x of the XY monomial x^m * y^n.

    Returns m * x^(m-1) * y^n; zero everywhere when m == 0.

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
        d/dx of x^m * y^n evaluated on the input grid

    """
    if cartesian_grid:
        x, y = optimize_xy_separable(x, y)

    if m == 0:
        # broadcast zeros to the (y, x) grid using the input shapes
        return np.zeros_like(x * y)

    return m * x**(m-1) * y**n


def xy_der_y(m, n, x, y, cartesian_grid=True):
    """Partial derivative w.r.t. y of the XY monomial x^m * y^n.

    Returns n * x^m * y^(n-1); zero everywhere when n == 0.

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
        d/dy of x^m * y^n evaluated on the input grid

    """
    if cartesian_grid:
        x, y = optimize_xy_separable(x, y)

    if n == 0:
        return np.zeros_like(x * y)

    return n * x**m * y**(n-1)


def xy_der_xy(m, n, x, y, cartesian_grid=True):
    """Mixed partial derivative d^2/dxdy of the XY monomial x^m * y^n.

    Returns m * n * x^(m-1) * y^(n-1); zero everywhere when m == 0 or n == 0.

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
        d^2/dxdy of x^m * y^n evaluated on the input grid

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

    return [x_seq[m] * y_seq[n] for m, n in mns]


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
    list
        list of modes, in the same order as mns

    """
    return _xy_seq_with(
        mns, x, y, cartesian_grid,
        _monomial_seq, _monomial_seq,
    )


def xy_der_x_seq(mns, x, y, cartesian_grid=True):
    """Partial derivative w.r.t. x of the XY monomial seq.

    Parameters mirror xy_seq.  The (m, n) output is m * x^(m-1) * y^n;
    entries with m == 0 are zero.

    Returns
    -------
    list
        list of d/dx of x^m * y^n, in the same order as mns

    """
    return _xy_seq_with(
        mns, x, y, cartesian_grid,
        _monomial_der_seq, _monomial_seq,
    )


def xy_der_y_seq(mns, x, y, cartesian_grid=True):
    """Partial derivative w.r.t. y of the XY monomial seq.

    Parameters mirror xy_seq.  The (m, n) output is n * x^m * y^(n-1);
    entries with n == 0 are zero.

    Returns
    -------
    list
        list of d/dy of x^m * y^n, in the same order as mns

    """
    return _xy_seq_with(
        mns, x, y, cartesian_grid,
        _monomial_seq, _monomial_der_seq,
    )


def xy_der_xy_seq(mns, x, y, cartesian_grid=True):
    """Mixed partial derivative d^2/dxdy of the XY monomial seq.

    Parameters mirror xy_seq.  The (m, n) output is m*n * x^(m-1) * y^(n-1);
    entries where m == 0 or n == 0 are zero.

    Returns
    -------
    list
        list of d^2/dxdy of x^m * y^n, in the same order as mns

    """
    return _xy_seq_with(
        mns, x, y, cartesian_grid,
        _monomial_der_seq, _monomial_der_seq,
    )
