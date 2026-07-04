"""Tests for prysm.x.optym.cost."""
import numpy as np
import pytest

from prysm.x.optym.cost import (
    bias_and_gain_invariant_error,
    mean_square_error,
    negative_loglikelihood,
)


# -----------------------------------------------------------------------------
# mean_square_error
# -----------------------------------------------------------------------------

def test_mse_unmasked_cost_and_grad():
    M = np.array([1.0, 2.0, 3.0, 4.0])
    D = np.array([0.0, 0.0, 0.0, 0.0])
    cost, grad = mean_square_error(M, D)
    np.testing.assert_allclose(cost, (1 + 4 + 9 + 16) / 4)
    np.testing.assert_allclose(grad, 2 * M / M.size)


def test_mse_masked_scatters_grad_to_unmasked_zero():
    M = np.array([1.0, 2.0, 3.0, 4.0])
    D = np.array([0.0, 0.0, 0.0, 0.0])
    mask = np.array([True, False, True, False])
    cost, grad = mean_square_error(M, D, mask=mask)
    # cost only over the kept elements (1, 3)
    np.testing.assert_allclose(cost, (1 + 9) / 2)
    # grad zero where mask is False
    np.testing.assert_allclose(grad[~mask], 0.0)
    # nonzero where mask is True
    np.testing.assert_allclose(grad[mask], 2 * M[mask] / mask.sum())


def test_mse_grad_dtype_matches_input():
    M = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    D = np.zeros(3, dtype=np.float32)
    _, grad = mean_square_error(M, D)
    assert grad.dtype == np.float32


# -----------------------------------------------------------------------------
# bias_and_gain_invariant_error
# -----------------------------------------------------------------------------

def test_bgie_zero_on_exact_linear_transform():
    """Cost must be zero when I and D are related by exact gain and bias."""
    rng = np.random.default_rng(1)
    D = rng.standard_normal(64) + 2.0
    I = 2.5 * D + 7.0
    cost, grad = bias_and_gain_invariant_error(I, D)
    np.testing.assert_allclose(cost, 0.0, atol=1e-20)
    np.testing.assert_allclose(grad, 0.0, atol=1e-12)


def test_bgie_grad_matches_finite_difference():
    """The analytic gradient must match central differences; this exercises
    the envelope-theorem argument that alpha and beta need no chain terms."""
    rng = np.random.default_rng(2)
    D = rng.standard_normal(16) + 1.0
    I = rng.standard_normal(16) + 0.5
    _, grad = bias_and_gain_invariant_error(I, D)
    eps = 1e-6
    fd = np.zeros_like(I)
    for i in range(I.size):
        Ip = I.copy()
        Ip[i] += eps
        Im = I.copy()
        Im[i] -= eps
        fp, _ = bias_and_gain_invariant_error(Ip, D)
        fm, _ = bias_and_gain_invariant_error(Im, D)
        fd[i] = (fp - fm) / (2 * eps)
    np.testing.assert_allclose(grad, fd, rtol=1e-6, atol=1e-10)


def test_bgie_masked_matches_unmasked_on_masked_subset():
    """The decorator-applied mask must produce the same cost and same gradient
    (within mask) as calling the unmasked function on the manually-extracted
    subset.  This is the structural correctness guarantee of _masked_cost."""
    rng = np.random.default_rng(0)
    D = rng.standard_normal(32) + 1.0
    I = rng.standard_normal(32) + 0.5
    mask = np.zeros(32, dtype=bool)
    mask[:16] = True

    cost_m, grad_m = bias_and_gain_invariant_error(I, D, mask=mask)
    cost_u, grad_u = bias_and_gain_invariant_error(I[mask], D[mask])

    np.testing.assert_allclose(cost_m, cost_u)
    np.testing.assert_allclose(grad_m[mask], grad_u)
    np.testing.assert_allclose(grad_m[~mask], 0.0)


# -----------------------------------------------------------------------------
# negative_loglikelihood
# -----------------------------------------------------------------------------

def test_nll_array_yhat_unmasked():
    y = np.array([0.5, 0.7, 0.9])
    yhat = np.array([0.5, 0.7, 0.9])
    cost, grad = negative_loglikelihood(y, yhat)
    # at y == yhat, derivative of binary cross-entropy is 0
    np.testing.assert_allclose(grad, 0.0, atol=1e-12)
    # cost matches the closed form
    expected = -(yhat * np.log(y) + (1 - yhat) * np.log(1 - y)).mean()
    np.testing.assert_allclose(cost, expected)


def test_nll_scalar_yhat_skips_mask_indexing():
    """yhat may be a Python scalar — the decorator must not try to index it."""
    y = np.array([0.5, 0.7, 0.9])
    mask = np.array([True, False, True])
    cost, grad = negative_loglikelihood(y, 0.5, mask=mask)
    # grad zero outside mask
    np.testing.assert_allclose(grad[~mask], 0.0)
    # cost is finite and the math succeeded
    assert np.isfinite(cost)


# -----------------------------------------------------------------------------
# dtype mismatch guard
# -----------------------------------------------------------------------------

def test_dtype_mismatch_raises():
    M = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    D = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    with pytest.raises(TypeError, match='dtype mismatch'):
        mean_square_error(M, D)


def test_dtype_check_skipped_for_scalar_yhat():
    """A Python-scalar yhat has no .dtype — no spurious mismatch error."""
    y = np.array([0.5, 0.7], dtype=np.float64)
    # should not raise
    negative_loglikelihood(y, 0.5)
