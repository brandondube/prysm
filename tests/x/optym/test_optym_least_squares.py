"""Tests for prysm.x.optym.least_squares."""

import numpy as np
import pytest

from prysm.x.optym import DampedLeastSquares, damped_least_squares, runN


class _VectorResidualProblem:
    def __init__(self, target):
        self.target = np.asarray(target, dtype=float)

    def residuals(self, x):
        return np.asarray(x, dtype=float) - self.target


class _ScaledResidualProblem:
    def residuals(self, x):
        x = np.asarray(x, dtype=float)
        return np.array([10 * x[0] - 1, x[1] - 1])


def test_dls_equality_lambda_constraint_from_optym():
    problem = _VectorResidualProblem([3.0, 4.0])
    result = damped_least_squares(
        problem,
        x0=np.array([0.0, 0.0]),
        equality_constraints=lambda x: x[0] + x[1] - 1.0,
        damping=0.0,
        maxiter=3,
    )
    assert result.success
    np.testing.assert_allclose(result.x, [0.0, 1.0], atol=1e-9)


def test_dls_active_inequality_lambda_constraint_from_optym():
    problem = _VectorResidualProblem([0.0, 0.0])
    result = damped_least_squares(
        problem,
        x0=np.array([4.0, 1.0]),
        inequality_constraints=lambda x: x[0] - 2.0,
        damping=0.0,
        maxiter=3,
    )
    assert result.success
    np.testing.assert_allclose(result.x, [2.0, 0.0], atol=1e-9)
    assert result.lambda_ineq[0] < 0.0


def test_dls_step_returns_old_x_convention_from_optym():
    problem = _VectorResidualProblem([1.0, 2.0])
    opt = DampedLeastSquares(
        problem,
        x0=np.array([0.0, 0.0]),
        damping=0.0,
        maxiter=5,
    )
    x_returned, f, g = opt.step()
    np.testing.assert_array_equal(x_returned, [0.0, 0.0])
    np.testing.assert_allclose(f, 2.5)
    np.testing.assert_allclose(g, [-1.0, -2.0])
    np.testing.assert_allclose(opt.x, [1.0, 2.0], atol=1e-9)
    assert opt.iter == 1


def test_dls_works_with_runN_from_optym():
    problem = _VectorResidualProblem([1.0, 2.0])
    opt = DampedLeastSquares(
        problem,
        x0=np.array([0.0, 0.0]),
        damping=0.0,
        maxiter=5,
    )
    x_returned, f, g = next(runN(opt, 1))
    np.testing.assert_array_equal(x_returned, [0.0, 0.0])
    np.testing.assert_allclose(f, 2.5)
    np.testing.assert_allclose(g, [-1.0, -2.0])
    np.testing.assert_allclose(opt.x, [1.0, 2.0], atol=1e-9)


def test_dls_sensitivity_damping_uses_current_column_sensitivity():
    opt = DampedLeastSquares(
        _ScaledResidualProblem(),
        x0=np.array([0.0, 0.0]),
        damping=0.5,
        damping_mode='sensitivity',
        damping_floor=0.0,
        maxiter=5,
    )
    opt.step()
    np.testing.assert_allclose(
        opt.last_step_metadata['damping_diagonal'],
        [50.0, 0.5],
        rtol=1e-5,
    )


def test_dls_trust_radii_scale_the_whole_step():
    opt = DampedLeastSquares(
        _VectorResidualProblem([10.0, 1.0]),
        x0=np.array([0.0, 0.0]),
        damping=0.0,
        trust_radii=np.array([0.5, np.inf]),
        maxiter=5,
    )
    opt.step()
    np.testing.assert_allclose(opt.x, [0.5, 0.05], atol=1e-12)
    assert opt.last_step_metadata['trust_scale'] == pytest.approx(0.05)


def test_dls_adaptive_damping_decreases_after_full_step():
    opt = DampedLeastSquares(
        _VectorResidualProblem([1.0]),
        x0=np.array([0.0]),
        damping=10.0,
        adaptive_damping=True,
        damping_decrease=0.5,
        maxiter=5,
    )
    opt.step()
    assert opt.last_step_metadata['damping'] == 10.0
    assert opt.damping == 5.0


class _AnalyticJacobianProblem:
    """Linear residuals with an analytic Jacobian and call counters."""

    def __init__(self, target, decline=False):
        self.target = np.asarray(target, dtype=float)
        self.decline = decline
        self.n_res = 0
        self.n_jac = 0

    def residuals(self, x):
        self.n_res += 1
        return np.asarray(x, dtype=float) - self.target

    def residual_jacobian(self, x):
        self.n_jac += 1
        if self.decline:
            return None
        return np.eye(self.target.size)


def test_dls_uses_problem_residual_jacobian():
    problem = _AnalyticJacobianProblem([3.0, 4.0])
    result = damped_least_squares(
        problem, x0=np.array([0.0, 0.0]), damping=0.0, maxiter=3,
    )
    assert result.success
    np.testing.assert_allclose(result.x, [3.0, 4.0], atol=1e-12)
    assert problem.n_jac >= 1
    # no FD bookkeeping: nfev counts only the residual evals of the line
    # search and acceptance tests, never 2 * n per linearization
    assert result.nfev < result.njev * 2 * 2 + result.njev + 2


def test_dls_falls_back_to_fd_when_jacobian_declines():
    declined = _AnalyticJacobianProblem([3.0, 4.0], decline=True)
    plain = _VectorResidualProblem([3.0, 4.0])
    r1 = damped_least_squares(declined, x0=np.array([0.0, 0.0]),
                              damping=0.0, maxiter=3)
    r2 = damped_least_squares(plain, x0=np.array([0.0, 0.0]),
                              damping=0.0, maxiter=3)
    assert r1.success
    np.testing.assert_allclose(r1.x, r2.x, atol=1e-12)
    # FD ran: per-iteration nfev includes the 2 * n stencil
    assert r1.nfev > r2.nit * 2  # at least the stencils
    assert r1.nfev == r2.nfev


def test_dls_analytic_jacobian_cuts_nfev_vs_fd():
    fast = _AnalyticJacobianProblem([3.0, 4.0])
    slow = _VectorResidualProblem([3.0, 4.0])
    rf = damped_least_squares(fast, x0=np.array([0.0, 0.0]),
                              damping=0.0, maxiter=3)
    rs = damped_least_squares(slow, x0=np.array([0.0, 0.0]),
                              damping=0.0, maxiter=3)
    np.testing.assert_allclose(rf.x, rs.x, atol=1e-12)
    assert rf.nfev < rs.nfev
