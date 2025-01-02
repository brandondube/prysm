"""XY polynomials."""

import numpy as truenp

from prysm.mathops import np  # NOQA
from prysm.coordinates import optimize_xy_separable

from .dickson import dickson1_seq


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
    mns2 = truenp.asarray(mns)
    maxm, maxn = mns2.max(axis=0)

    if cartesian_grid and x.ndim > 1:
        x, y = optimize_xy_separable(x, y)

    ms = truenp.arange(0, maxm+1)
    ns = truenp.arange(0, maxn+1)
    # dicksons with alpha=0 are the monomials
    x_seq = list(dickson1_seq(ms, 0, x))
    y_seq = list(dickson1_seq(ns, 0, y))

    out = []
    for m, n in mns:
        xterm = x_seq[m]
        yterm = y_seq[n]
        out.append(xterm*yterm)

    return out


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
    list
        list of modes, in the same order as mns

    """
    if cartesian_grid:
        x, y = optimize_xy_separable(x, y)

    return x**m * y**n
