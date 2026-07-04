"""Behavioral tests for prysm.x.optym.optimizers."""
import warnings

import numpy as np
import pytest

from prysm.x.optym import (
    GradientDescent,
    AdaGrad,
    RMSProp,
    Adam,
    RAdam,
    AdaMomentum,
    Yogi,
    LBFGSB,
    MaxIterations,
    rosenbrock,
    run_until,
)
from prysm.x.optym._lbfgsb import CLBFGSB, F77LBFGSB, _scipy_has_c_lbfgsb


# Mildly non-symmetric quadratic — enough to exercise per-coordinate
# adaptive optimizers without being so ill-conditioned that small numerics
# changes cascade.
H_DIAG = np.array([1.0, 2.0, 0.5, 4.0])


def fg(x):
    f = float(0.5 * np.sum(H_DIAG * x * x))
    g = H_DIAG * x
    return f, g


X0 = np.array([1.0, -2.0, 3.0, -0.5])
NSTEPS = 30


def run_traj(opt):
    """Return the (x, f) trajectory for NSTEPS iterations."""
    xs = np.empty((NSTEPS, X0.size))
    fs = np.empty(NSTEPS)
    for i in range(NSTEPS):
        x, f, _ = opt.step()
        xs[i] = x
        fs[i] = f
    return xs, fs


FIRST_ORDER_OPTIMIZERS = [
    (GradientDescent, {'alpha': 0.1}),
    (AdaGrad, {'alpha': 0.1}),
    (RMSProp, {'alpha': 0.1}),
    (Adam, {'alpha': 0.1}),
    (RAdam, {'alpha': 0.1}),
    (AdaMomentum, {'alpha': 0.1}),
    (Yogi, {'alpha': 0.1}),
]


@pytest.fixture
def x0():
    # fresh copy each test — optimizers .copy() internally but be safe
    return X0.copy()


@pytest.mark.parametrize('cls, kwargs', FIRST_ORDER_OPTIMIZERS)
def test_first_order_optimizers_descend(cls, kwargs, x0):
    opt = cls(fg, x0, **kwargs)
    xs, fs = run_traj(opt)
    assert fs[-1] < fs[0]
    np.testing.assert_array_equal(xs[0], X0)


def test_lbfgsb_step_returns_coherent_triple(x0):
    """L-BFGS-B's step() returns (x, f, g) where (f, g) were evaluated at x.

    The driver's behavior past iteration 1 is not exercised here — with the
    aggressive factr=pgtol=0 settings the driver may signal convergence
    immediately, which is intentional and documented on the class.  What we
    are pinning is that the very first step's return triple is coherent
    with the input problem and that the driver was reached without crashing.
    """
    opt = LBFGSB(fg, x0)
    x, f, g = opt.step()
    np.testing.assert_array_equal(x, X0)
    # f at x must match the problem evaluated at x
    f_check, g_check = fg(x)
    np.testing.assert_allclose(f, f_check)
    np.testing.assert_allclose(g, g_check)


def test_lbfgsb_run_to_completes_quietly_on_quadratic(x0):
    """run_to() runs to completion without warning on a non-degenerate
    quadratic: the driver does not signal CONVERGENCE while there is
    still progress to make.
    """
    opt = LBFGSB(fg, x0)
    with warnings.catch_warnings():
        warnings.simplefilter('error')   # any warning fails the test
        traj = list(opt.run_to(10))
    assert len(traj) == 10


def test_lbfgsb_run_until_rosenbrock_does_not_stop_after_one_iteration():
    opt = LBFGSB(rosenbrock, np.array([-1.2, 1.0]))
    result = run_until(opt, MaxIterations(5), maxiter=5)

    assert result.nit == 5
    assert result.message == 'maximum iterations reached'
    assert not result.success


def test_lbfgsb_keeps_driver_state_private_and_returns_copies(x0):
    opt = LBFGSB(fg, x0)
    x_returned, _, g_returned = opt.step()
    x_after = opt.x

    x_returned[...] = 100
    g_returned[...] = 100
    x_after[...] = 100

    np.testing.assert_array_equal(opt.x, opt._x)
    assert not np.array_equal(opt.x, x_returned)
    assert not np.array_equal(opt.x, x_after)


def test_lbfgsb_run_to_warns_when_driver_signals_convergence():
    """If the underlying driver signals CONVERGENCE (e.g. when the
    initial point is already a stationary point), run_to swallows the
    StopIteration and emits a UserWarning instead of propagating.
    """
    x0 = np.zeros(4)           # gradient is exactly zero on `fg`
    opt = LBFGSB(fg, x0)
    with pytest.warns(UserWarning, match='L-BFGS-B'):
        list(opt.run_to(10))


def test_lbfgsb_alias_matches_scipy_version():
    """LBFGSB alias points to CLBFGSB on scipy>=1.15, F77 otherwise."""
    if _scipy_has_c_lbfgsb():
        assert LBFGSB is CLBFGSB
    else:
        assert LBFGSB is F77LBFGSB


@pytest.mark.skipif(not _scipy_has_c_lbfgsb(), reason='C driver only')
def test_c_lbfgsb_decodes_abnormal_task(x0):
    """SciPy's C driver can return status code 8 (ABNORMAL); it should be
    treated as a known failed termination, not as an unknown task.
    """
    opt = CLBFGSB(fg, x0)
    opt.task[0] = 8
    opt.task[1] = 0
    assert opt._decode_task() == 'ABNORMAL'
    assert opt._task_diagnostic() == 'ABNORMAL'


def test_all_optimizers_return_old_x_convention(x0):
    """Regression: every optimizer's step() returns the iterate at which
    (f, g) was evaluated — not the post-update iterate.  After step(),
    opt.x holds the new iterate; the returned x is the previous one.
    """
    for cls, kwargs in FIRST_ORDER_OPTIMIZERS:
        opt = cls(fg, x0.copy(), **kwargs)
        x_before = opt.x.copy()
        x_returned, f, g = opt.step()
        np.testing.assert_array_equal(
            x_returned, x_before,
            err_msg=f'{cls.__name__}.step() returned the wrong iterate',
        )
        # f was evaluated at x_before
        f_check, _ = fg(x_before)
        np.testing.assert_allclose(f, f_check)
        # opt.x is now the new iterate (not equal to the old one)
        assert not np.array_equal(opt.x, x_before)


@pytest.mark.parametrize('cls, kwargs', FIRST_ORDER_OPTIMIZERS)
def test_first_order_optimizers_default_bounds_are_unconstrained(cls, kwargs):
    x0 = np.array([1.0, -2.0], dtype=np.float32)
    opt = cls(fg, x0, **kwargs)
    assert opt.l.shape == x0.shape
    assert opt.u.shape == x0.shape
    assert opt.l.dtype == x0.dtype
    assert opt.u.dtype == x0.dtype
    assert not opt._has_bounds
    assert np.all(np.isneginf(opt.l))
    assert np.all(np.isposinf(opt.u))


@pytest.mark.parametrize('cls, kwargs', FIRST_ORDER_OPTIMIZERS)
def test_first_order_optimizers_project_x0_and_each_step(cls, kwargs):
    def outward_fg(x):
        return float(np.sum(x)), np.ones_like(x)

    x0 = np.array([-2.0, 2.0])
    lb = np.array([0.0, 0.0])
    ub = np.array([1.0, 1.0])
    opt = cls(outward_fg, x0, lower_bounds=lb, upper_bounds=ub, **kwargs)

    np.testing.assert_array_equal(opt.x, np.array([0.0, 1.0]))
    x_returned, _, _ = opt.step()

    np.testing.assert_array_equal(x_returned, np.array([0.0, 1.0]))
    assert np.all(opt.x >= lb)
    assert np.all(opt.x <= ub)
    assert opt.last_step_metadata['bounded_variables'] >= 1


@pytest.mark.parametrize('cls, kwargs', FIRST_ORDER_OPTIMIZERS)
def test_active_bound_masks_outward_gradient_but_allows_inward_motion(cls, kwargs):
    gradient = np.array([1.0])

    def fg_with_mutable_gradient(x):
        return float(gradient[0] * x[0]), gradient.copy()

    x0 = np.array([0.0])
    lb = np.array([0.0])
    ub = np.array([1.0])
    opt = cls(fg_with_mutable_gradient, x0,
              lower_bounds=lb, upper_bounds=ub, **kwargs)

    _, _, g = opt.step()
    np.testing.assert_array_equal(g, np.array([1.0]))
    np.testing.assert_array_equal(
        opt.last_step_metadata['projected_gradient'],
        np.array([0.0]),
    )
    np.testing.assert_array_equal(opt.x, lb)

    gradient[0] = -1.0
    opt.step()
    assert opt.x[0] > lb[0]


def test_adam_does_not_accumulate_outward_momentum_at_active_bound():
    def fg_lower_active(x):
        return float(x[0]), np.array([1.0])

    opt = Adam(
        fg_lower_active,
        np.array([0.0]),
        alpha=0.1,
        lower_bounds=np.array([0.0]),
        upper_bounds=np.array([1.0]),
    )
    opt.step()

    np.testing.assert_array_equal(opt.x, np.array([0.0]))
    np.testing.assert_array_equal(opt.m, np.array([0.0]))
    np.testing.assert_array_equal(opt.v, np.array([0.0]))


def test_adagrad_does_not_accumulate_outward_gradient_at_active_bound():
    def fg_lower_active(x):
        return float(x[0]), np.array([1.0])

    opt = AdaGrad(
        fg_lower_active,
        np.array([0.0]),
        alpha=0.1,
        lower_bounds=np.array([0.0]),
        upper_bounds=np.array([1.0]),
    )
    opt.step()

    np.testing.assert_array_equal(opt.x, np.array([0.0]))
    np.testing.assert_array_equal(opt.accumulator, np.array([0.0]))


def test_first_order_bounds_validate_shape_and_order():
    with pytest.raises(ValueError, match='same shape or size'):
        GradientDescent(
            fg,
            np.array([0.0, 1.0]),
            alpha=0.1,
            lower_bounds=np.array([0.0, 0.0, 0.0]),
        )

    with pytest.raises(ValueError, match='lower_bounds'):
        GradientDescent(
            fg,
            np.array([0.0, 1.0]),
            alpha=0.1,
            lower_bounds=np.array([1.0, 0.0]),
            upper_bounds=np.array([0.0, 1.0]),
        )
