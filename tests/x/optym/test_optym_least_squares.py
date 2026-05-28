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
