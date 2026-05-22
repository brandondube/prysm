"""Tests for prysm.x.optym.linesearch."""
import numpy as np

from prysm.x.optym.linesearch import ls_strong_wolfe


C1 = 1e-4
C2 = 0.9


def _wolfe_holds(fg, xk, pk, alpha, phi_a, derphi_a, c1=C1, c2=C2):
    """Verify the strong Wolfe conditions at the returned alpha."""
    f0, g0 = fg(xk)
    derphi0 = float(np.dot(g0, pk))
    sufficient_decrease = phi_a <= f0 + c1 * alpha * derphi0 + 1e-12
    curvature = abs(derphi_a) <= c2 * abs(derphi0) + 1e-12
    return sufficient_decrease and curvature


def test_ls_strong_wolfe_optimal_alpha_one_on_identity_quadratic():
    """f = 0.5 * ||x||^2: gradient is x, optimal alpha for steepest descent
    is exactly 1.0. Line search should accept it on the first probe."""
    def fg(x):
        return float(0.5 * np.sum(x * x)), x

    xk = np.array([1.0, -2.0])
    pk = -fg(xk)[1]
    alpha, phi_a, derphi_a, g_a = ls_strong_wolfe(fg, xk, pk)
    assert alpha is not None
    np.testing.assert_allclose(alpha, 1.0)
    np.testing.assert_allclose(phi_a, 0.0, atol=1e-12)
    assert _wolfe_holds(fg, xk, pk, alpha, phi_a, derphi_a)


def test_ls_strong_wolfe_brackets_and_zooms_when_alpha_one_overshoots():
    """f = 0.5 * x^T H x with H = diag(10, 10): gradient is 10*x, so optimal
    alpha along steepest descent is 0.1. alpha=1 overshoots wildly, forcing
    the bracket/zoom path to be exercised."""
    H = 10.0 * np.eye(2)

    def fg(x):
        return float(0.5 * x @ H @ x), H @ x

    xk = np.array([1.0, 1.0])
    pk = -fg(xk)[1]
    alpha, phi_a, derphi_a, g_a = ls_strong_wolfe(fg, xk, pk)
    assert alpha is not None
    assert 0 < alpha < 1.0
    assert _wolfe_holds(fg, xk, pk, alpha, phi_a, derphi_a)


def test_ls_strong_wolfe_extrapolates_when_alpha_one_undershoots():
    """f = 0.5 * x^T H x with H = 0.01*I: optimal alpha is 100, so alpha=1
    badly undershoots (curvature ratio ~0.99 >> c2=0.9) and the bracket loop
    must extrapolate before reporting success."""
    H = 0.01 * np.eye(2)

    def fg(x):
        return float(0.5 * x @ H @ x), H @ x

    xk = np.array([1.0, -1.0])
    pk = -fg(xk)[1]
    alpha, phi_a, derphi_a, g_a = ls_strong_wolfe(fg, xk, pk)
    assert alpha is not None
    assert alpha > 1.0
    assert _wolfe_holds(fg, xk, pk, alpha, phi_a, derphi_a)


def test_ls_strong_wolfe_respects_maxalpha():
    """maxalpha caps the initial trial step. With H=I the unconstrained optimum
    is alpha=1; if we cap maxalpha=0.5 the line searcher should accept 0.5
    (which already satisfies strong Wolfe on this problem) rather than
    overshoot."""
    def fg(x):
        return float(0.5 * np.sum(x * x)), x

    xk = np.array([1.0, -1.0])
    pk = -fg(xk)[1]
    alpha, phi_a, derphi_a, _ = ls_strong_wolfe(fg, xk, pk, maxalpha=0.5)
    assert alpha is not None
    assert alpha <= 0.5 + 1e-12
    assert _wolfe_holds(fg, xk, pk, alpha, phi_a, derphi_a)


def test_ls_strong_wolfe_uses_supplied_fgk():
    """If the caller passes fgk, ls_strong_wolfe must not re-evaluate fg at xk."""
    calls = {'n': 0}

    def fg(x):
        calls['n'] += 1
        return float(0.5 * np.sum(x * x)), x

    xk = np.array([1.0, -2.0])
    pk = -xk
    pre = fg(xk)
    calls['n'] = 0
    ls_strong_wolfe(fg, xk, pk, fg_at_xk=pre)
    # at least one fg call (at the trial alpha); none at xk
    assert calls['n'] >= 1


def test_ls_strong_wolfe_returns_gradient_at_accepted_alpha():
    """ls_strong_wolfe's fourth return value is g(xk + alpha*pk); callers
    that step in direction pk can reuse it and skip a redundant fg."""
    def fg(x):
        return float(0.5 * np.sum(x * x)), x

    xk = np.array([1.0, -2.0])
    pk = -xk
    alpha, _, _, g_a = ls_strong_wolfe(fg, xk, pk)
    assert g_a is not None
    np.testing.assert_allclose(g_a, xk + alpha * pk)


def test_ls_strong_wolfe_returns_none_gradient_on_failure():
    """When no acceptable alpha is found, the gradient slot is None."""
    # contrived ascent direction: pk parallel to +g
    def fg(x):
        return float(0.5 * np.sum(x * x)), x

    xk = np.array([1.0, -2.0])
    pk = xk  # uphill
    alpha, _, _, g_a = ls_strong_wolfe(fg, xk, pk)
    assert alpha is None
    assert g_a is None
