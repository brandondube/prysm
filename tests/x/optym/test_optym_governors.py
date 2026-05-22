"""Tests for prysm.x.optym.governors."""

import numpy as np

from prysm.x.optym import (
    Adam,
    AnyGovernor,
    ConstraintTolerance,
    DampedLeastSquares,
    FunctionTolerance,
    GradientDescent,
    GradientTolerance,
    LBFGSB,
    MaxEvaluations,
    MaxIterations,
    StepRecord,
    StepTolerance,
    run_until,
)


def quadratic_fg(x):
    f = float(0.5 * np.sum(x * x))
    return f, x.copy()


class _VectorResidualProblem:
    def __init__(self, target):
        self.target = np.asarray(target, dtype=float)

    def residuals(self, x):
        return np.asarray(x, dtype=float) - self.target


def test_run_until_max_iterations_with_gradient_descent():
    opt = GradientDescent(
        quadratic_fg, np.array([1.0, -2.0]), alpha=0.1,
    )
    result = run_until(opt, MaxIterations(3))
    assert not result.success
    assert result.message == 'maximum iterations reached'
    assert result.nit == 3
    assert len(result.records) == 3
    np.testing.assert_allclose(result.x, [0.729, -1.458])
    np.testing.assert_array_equal(result.records[0].x, [1.0, -2.0])
    np.testing.assert_allclose(result.records[0].x_next, [0.9, -1.8])


def test_function_tolerance_stops_gradient_descent():
    opt = GradientDescent(quadratic_fg, np.array([1.0]), alpha=0.5)
    result = run_until(opt, FunctionTolerance(0.1, relative=False))
    assert result.success
    assert result.message == 'function tolerance reached'
    assert result.nit == 3


def test_gradient_tolerance_stops_gradient_descent():
    opt = GradientDescent(quadratic_fg, np.array([1.0]), alpha=0.5)
    result = run_until(opt, GradientTolerance(0.6))
    assert result.success
    assert result.message == 'gradient tolerance reached'
    assert result.nit == 2


def test_step_tolerance_stops_adam():
    opt = Adam(quadratic_fg, np.array([1.0]), alpha=0.1)
    result = run_until(opt, StepTolerance(0.2))
    assert result.success
    assert result.message == 'step tolerance reached'
    assert result.nit == 1


def test_run_until_max_iterations_with_lbfgsb():
    opt = LBFGSB(quadratic_fg, np.array([1.0, -1.0]))
    result = run_until(opt, MaxIterations(1))
    assert not result.success
    assert result.message == 'maximum iterations reached'
    assert result.nit == 1
    assert result.nfev >= 1
    assert 'task' in result.records[0].metadata


def test_dls_metadata_supports_generic_governors():
    problem = _VectorResidualProblem([1.0, 2.0])
    opt = DampedLeastSquares(
        problem,
        x0=np.array([0.0, 0.0]),
        damping=0.0,
        maxiter=5,
    )
    governor = AnyGovernor([
        FunctionTolerance(10.0, relative=False),
        ConstraintTolerance(1e-12),
    ])
    result = run_until(opt, governor)
    assert result.success
    assert result.message == 'function tolerance reached'
    assert result.nit == 1
    np.testing.assert_allclose(result.x, [1.0, 2.0], atol=1e-9)
    assert result.records[0].metadata['accepted']
    assert result.records[0].metadata['step_norm'] > 0


def test_max_evaluations_reads_optimizer_nfev():
    problem = _VectorResidualProblem([1.0, 2.0])
    opt = DampedLeastSquares(
        problem,
        x0=np.array([0.0, 0.0]),
        damping=0.0,
        maxiter=5,
    )
    result = run_until(opt, MaxEvaluations(1))
    assert not result.success
    assert result.message == 'maximum function evaluations reached'
    assert result.nfev >= 1


def test_step_record_aliases_inputs():
    """StepRecord stores its iterate / gradient inputs by reference; it
    does not take defensive copies.  Per-step copies on every governor
    observation are expensive at large n, and every optimizer in this
    module is responsible for returning step results that won't be
    mutated by a subsequent step.  Optimizers whose internal state IS a
    mutable buffer (e.g. scipy L-BFGS-B's Fortran driver) keep that
    buffer separate from self.x and snapshot at each accepted iteration.
    """
    x = np.array([1.0])
    g = np.array([2.0])
    x_next = np.array([3.0])
    record = StepRecord(None, 1, x, 4.0, g, x_next)

    assert record.x is x
    assert record.g is g
    assert record.x_next is x_next
