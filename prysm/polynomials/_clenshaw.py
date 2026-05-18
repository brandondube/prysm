"""Shared Clenshaw helpers for the polynomial families.

These primitives are not part of the public API.
"""

from prysm.mathops import np
from prysm.conf import config


def _initialize_alphas(s, x, alphas, j=0):
    """Allocate the alphas buffer used by Clenshaw recurrences.

    The trailing axis is broadcast over x.  The mode axis is sized
    max(len(s), 2) so callers that read alphas[..., 1] (e.g.
    compute_z_zprime_Qbfs) never need to guard the len(s) == 1 edge case —
    the unused slot is zero-padded.

    Parameters
    ----------
    s : sized iterable
        Clenshaw coefficient vector (used only for len(s))
    x : ndarray or scalar
        evaluation grid; supplies shape and dtype
    alphas : ndarray or None
        if not None, returned unchanged; otherwise a fresh zero array of the
        right shape is allocated
    j : int, optional
        derivative order.  If j == 0, alphas has shape (N, *x.shape);
        otherwise it is (j+1, N, *x.shape).

    """
    if alphas is not None:
        return alphas

    if hasattr(x, 'dtype'):
        dtype = x.dtype
    else:
        dtype = config.precision

    n_axis = max(len(s), 2)
    if hasattr(x, 'shape'):
        shape = (n_axis, *x.shape)
    elif hasattr(x, '__len__'):
        shape = (n_axis, len(x))
    else:
        shape = (n_axis,)

    if j != 0:
        shape = (j + 1, *shape)

    return np.zeros(shape, dtype=dtype)


def _clenshaw_sum(coefs, lin_fn, c_fn, alphas):
    """Run Clenshaw's recurrence and fill alphas in-place.

    The recurrence assumes orthogonal polynomials with three-term form

        P_n(x) = lin_n(x) * P_{n-1}(x) - c_n * P_{n-2}(x)

    where lin_fn(n) returns the (broadcast-compatible) ndarray for lin_n(x)
    (e.g., a_n * x + b_n for Jacobi or A_n + B_n * x for Q2D) and c_fn(n)
    returns the scalar c_n.

    The result alpha-table satisfies

        sum_{n=0..M} coefs[n] P_n(x) = P_0(x) * alphas[0] + remainder.

    Most callers can read alphas[0] directly (since P_0 = 1 for the families
    we support).  Qbfs is the exception: its envelope factor multiplies
    2 * (alphas[0] + alphas[1]) instead, which is also supported because
    _initialize_alphas pads the mode axis to at least 2.

    Parameters
    ----------
    coefs : sized iterable
        dense coefficient vector c_0 .. c_M
    lin_fn : callable
        lin_fn(n) -> ndarray giving the linear-in-x recurrence factor
    c_fn : callable
        c_fn(n) -> scalar giving the constant recurrence factor
    alphas : ndarray
        pre-allocated buffer of shape (>= max(len(coefs), 2), *x.shape)

    """
    M = len(coefs) - 1
    if M == 0:
        # only P_0 contributes; alphas[0] gets the sole weight, alphas[1+]
        # stay zero from the initial allocation.
        alphas[0] = coefs[0]
        alphas[1:] = 0
        return alphas
    alphas[M] = coefs[M]
    alphas[M - 1] = coefs[M - 1] + lin_fn(M - 1) * alphas[M]
    for n in range(M - 2, -1, -1):
        alphas[n] = coefs[n] + lin_fn(n) * alphas[n + 1] - c_fn(n + 1) * alphas[n + 2]
    return alphas


def _clenshaw_sum_der(coefs, lin_fn, lin_x_coef_fn, c_fn, alphas, j):
    """Clenshaw alpha-tables for the value and first j derivatives.

    alphas[0] is seeded by _clenshaw_sum.  For each higher derivative order
    jj in 1..j the recurrence is

        alphas[jj][n] = jj * b_n * alphas[jj-1][n+1]
                      + lin_n(x) * alphas[jj][n+1]
                      - c_{n+1} * alphas[jj][n+2]

    where lin_x_coef_fn(n) returns the coefficient of x in lin_n(x) (b_n).

    Parameters
    ----------
    coefs : sized iterable
        dense coefficient vector c_0 .. c_M
    lin_fn, c_fn : callables
        same meaning as in _clenshaw_sum
    lin_x_coef_fn : callable
        lin_x_coef_fn(n) -> scalar giving the coefficient of x inside lin_n(x)
    alphas : ndarray
        pre-allocated buffer of shape (j+1, >= max(len(coefs), 2), *x.shape)
    j : int
        max derivative order to compute

    """
    M = len(coefs) - 1
    _clenshaw_sum(coefs, lin_fn, c_fn, alphas[0])
    for jj in range(1, j + 1):
        if jj > M:
            # the (jj)-th derivative of a degree-M polynomial is identically zero
            alphas[jj] = 0
            continue
        b = lin_x_coef_fn(M - jj)
        alphas[jj][M - jj] = jj * b * alphas[jj - 1][M - jj + 1]
        for n in range(M - jj - 1, -1, -1):
            b = lin_x_coef_fn(n)
            alphas[jj][n] = (jj * b * alphas[jj - 1][n + 1]
                             + lin_fn(n) * alphas[jj][n + 1]
                             - c_fn(n + 1) * alphas[jj][n + 2])
    return alphas
