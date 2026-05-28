"""Backend parity tests for PrysmLBFGSB.

The optimizer is supposed to run unchanged on cupy / pytorch backends
under prysm.mathops's BackendShim.  These tests run a small bounded and
unbounded quadratic under each alternate backend (skipping when the
backend isn't installed) and verify both that no host-only API leaks
in and that the converged answer matches the numpy reference run.

The fp32 path is exercised through conditioning tests in
test_optym_prysm_lbfgsb.py; this file additionally pins one round-trip
under config.precision = 32 to catch regressions in the precision
plumbing.
"""
import numpy as truenp

import pytest

from prysm.conf import config
from prysm.mathops import (
    np,
    set_backend_to_cupy,
    set_backend_to_defaults,
    set_backend_to_pytorch,
)
from prysm.x.optym import MaxIterations, PrysmLBFGSB, run_until


def _bounded_quadratic_problem(n, dtype):
    """f(x) = 0.5 (x - x_target)^T (x - x_target), bounded in [-1, 1]."""
    x_target = truenp.linspace(-2.0, 2.0, n).astype(dtype)
    x_target_backend = np.asarray(x_target)

    def fg(x):
        diff = x - x_target_backend
        f = 0.5 * float(np.sum(diff * diff))
        return f, diff

    lb = np.full(n, -1.0, dtype=dtype)
    ub = np.full(n, 1.0, dtype=dtype)
    # known minimum: clamp(x_target, -1, 1)
    x_star = truenp.clip(x_target, -1.0, 1.0)
    return fg, x_star, lb, ub


def _unbounded_quadratic_problem(n, dtype):
    """Unbounded variant of the same quadratic."""
    x_target = truenp.linspace(-2.0, 2.0, n).astype(dtype)
    x_target_backend = np.asarray(x_target)

    def fg(x):
        diff = x - x_target_backend
        f = 0.5 * float(np.sum(diff * diff))
        return f, diff

    return fg, x_target


def _to_host(x):
    """Coerce a backend array to a real numpy array for comparison."""
    if hasattr(x, 'get'):
        return x.get()
    if hasattr(x, 'cpu'):
        return x.cpu().numpy()
    return truenp.asarray(x)


# ----------------------- cupy -----------------------

def _cupy_available():
    try:
        import cupy  # NOQA
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _cupy_available(), reason='cupy not installed')
def test_cupy_unbounded_matches_numpy():
    """Unbounded quadratic converges to the same answer on cupy and numpy."""
    n = 16
    # numpy reference
    set_backend_to_defaults()
    fg_ref, x_star = _unbounded_quadratic_problem(n, truenp.float64)
    x0_ref = np.zeros(n, dtype=truenp.float64)
    opt_ref = PrysmLBFGSB(fg_ref, x0_ref)
    run_until(opt_ref, MaxIterations(50))
    x_ref = _to_host(opt_ref.x)

    set_backend_to_cupy()
    try:
        fg, _ = _unbounded_quadratic_problem(n, truenp.float64)
        x0 = np.zeros(n, dtype=truenp.float64)
        opt = PrysmLBFGSB(fg, x0)
        run_until(opt, MaxIterations(50))
        x_cu = _to_host(opt.x)
    finally:
        set_backend_to_defaults()

    truenp.testing.assert_allclose(x_cu, x_ref, atol=1e-10)
    truenp.testing.assert_allclose(x_cu, x_star, atol=1e-10)


@pytest.mark.skipif(not _cupy_available(), reason='cupy not installed')
def test_cupy_bounded_matches_numpy():
    """Bounded quadratic — exercises the Cauchy + subspace path on cupy."""
    n = 16
    set_backend_to_defaults()
    fg_ref, x_star, lb_ref, ub_ref = _bounded_quadratic_problem(n, truenp.float64)
    x0_ref = np.zeros(n, dtype=truenp.float64)
    opt_ref = PrysmLBFGSB(fg_ref, x0_ref, lower_bounds=lb_ref, upper_bounds=ub_ref)
    run_until(opt_ref, MaxIterations(80))
    x_ref = _to_host(opt_ref.x)

    set_backend_to_cupy()
    try:
        fg, _, lb, ub = _bounded_quadratic_problem(n, truenp.float64)
        x0 = np.zeros(n, dtype=truenp.float64)
        opt = PrysmLBFGSB(fg, x0, lower_bounds=lb, upper_bounds=ub)
        run_until(opt, MaxIterations(80))
        x_cu = _to_host(opt.x)
    finally:
        set_backend_to_defaults()

    truenp.testing.assert_allclose(x_cu, x_ref, atol=1e-9)
    truenp.testing.assert_allclose(x_cu, x_star, atol=1e-9)


# ----------------------- pytorch -----------------------
#
# set_backend_to_pytorch is documented as remapping only np and fft.
# Many LBFGSB ops (np.linalg.solve, fancy-bool indexing for free_mask
# writes, np.isfinite + np.where on inf-bound arrays) are not strictly
# guaranteed to match torch semantics.  We xfail-on-import: the test
# exists so that if torch is wired up later we get a hard signal, but
# doesn't block CI in the meantime.

def _torch_available():
    try:
        import torch  # NOQA
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _torch_available(), reason='torch not installed')
@pytest.mark.xfail(reason='torch backend shim is documented as partial '
                          '(no linalg.solve/where parity guarantees)',
                   strict=False)
def test_torch_unbounded_quadratic_smokes():
    """Smoke test: unbounded quadratic on torch backend.

    If this passes we've gained a backend; if it fails we get a clear
    signal of which API the shim is missing.
    """
    set_backend_to_pytorch()
    try:
        n = 8
        fg, x_star = _unbounded_quadratic_problem(n, truenp.float64)
        x0 = np.zeros(n, dtype=truenp.float64)
        opt = PrysmLBFGSB(fg, x0)
        run_until(opt, MaxIterations(50))
        truenp.testing.assert_allclose(_to_host(opt.x), x_star, atol=1e-8)
    finally:
        set_backend_to_defaults()


# ----------------------- config.precision = 32 -----------------------

def test_precision32_smoketest_unbounded():
    """Toggling config.precision flips the dtype of newly allocated arrays.

    x0 still drives PrysmLBFGSB's internal dtype directly (the optimizer
    promises to preserve x0.dtype), but caller-allocated x0/bounds should
    be free to use config.precision.  Verify the optimizer round-trips
    cleanly at fp32 and converges.
    """
    prev = config.precision
    try:
        config.precision = 32
        n = 8
        fg, x_star = _unbounded_quadratic_problem(n, truenp.float32)
        x0 = np.zeros(n, dtype=config.precision)
        assert x0.dtype == truenp.float32
        opt = PrysmLBFGSB(fg, x0)
        # All internal arrays inherit x0.dtype.
        assert opt.S.dtype == truenp.float32
        assert opt.Y.dtype == truenp.float32
        run_until(opt, MaxIterations(60))
        truenp.testing.assert_allclose(
            _to_host(opt.x), x_star, atol=5e-4,
        )
    finally:
        config.precision = prev


def test_precision32_smoketest_bounded():
    """Same, but with bounds active — exercises Cauchy + subspace at fp32."""
    prev = config.precision
    try:
        config.precision = 32
        n = 8
        fg, x_star, lb, ub = _bounded_quadratic_problem(n, truenp.float32)
        x0 = np.zeros(n, dtype=config.precision)
        opt = PrysmLBFGSB(fg, x0, lower_bounds=lb, upper_bounds=ub)
        run_until(opt, MaxIterations(80))
        truenp.testing.assert_allclose(
            _to_host(opt.x), x_star, atol=5e-4,
        )
    finally:
        config.precision = prev
