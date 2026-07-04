"""Tests for the mathops submodule."""
import pytest

import numpy as np

from prysm import mathops

np.random.seed(1234)
TEST_ARR_SIZE = 32


@pytest.fixture
def sample_data_2d():
    return np.random.rand(TEST_ARR_SIZE, TEST_ARR_SIZE)


@pytest.mark.parametrize('num', [1, 3, 991, 100000000000001])
def test_is_odd_odd_numbers(num):
    assert mathops.is_odd(num)


@pytest.mark.parametrize('num', [0, 2, 1000, 100000000000000])
def test_is_odd_even_numbers(num):
    assert not mathops.is_odd(num)


@pytest.mark.parametrize('num', [2, 64, 8192])
def test_is_power_of_2_powers_of_2(num):
    assert mathops.is_power_of_2(num)


@pytest.mark.parametrize('num', [1, 3, 1000, -2])
def test_is_power_of_2_non_powers_of_2(num):
    assert not mathops.is_power_of_2(num)


@pytest.mark.parametrize('shim_name,probe_attr', [
    ('np', 'arange'),
    ('fft', 'fft'),
    ('ndimage', 'gaussian_filter'),
    ('interpolate', 'interp1d'),
    ('optimize', 'brentq'),
    ('signal', 'windows'),
])
def test_backend_shim_default_routes_to_scipy_numpy(shim_name, probe_attr):
    shim = getattr(mathops, shim_name)
    assert hasattr(shim, probe_attr)


def test_set_backend_to_defaults_restores_optimize_and_signal():
    # poke the shims, then verify defaults reseat them.
    sentinel = object()
    mathops.optimize._srcmodule = sentinel
    mathops.signal._srcmodule = sentinel
    mathops.set_backend_to_defaults()
    assert mathops.optimize._srcmodule is mathops._optimize
    assert mathops.signal._srcmodule is mathops._signal
    assert hasattr(mathops.optimize, 'brentq')
    assert hasattr(mathops.signal, 'windows')


@pytest.mark.parametrize('value', [1, 1.0, 1 + 2j, np.float64(1), np.bool_(True)])
def test_array_to_true_numpy_returns_scalars(value):
    assert mathops.array_to_true_numpy(value) == value


def test_array_to_true_numpy_returns_numpy_arrays_without_copy(sample_data_2d):
    assert mathops.array_to_true_numpy(sample_data_2d) is sample_data_2d


def test_array_to_true_numpy_prefers_cupy_get_over_numpy():
    class CupyLike:
        def get(self):
            return np.array([1, 2, 3])

        def numpy(self, force=True):
            return np.array([4, 5, 6])

    out = mathops.array_to_true_numpy(CupyLike())
    np.testing.assert_array_equal(out, np.array([1, 2, 3]))


def test_array_to_true_numpy_handles_torch_like_after_cupy():
    class TorchLike:
        def numpy(self, force=True):
            return np.array([1, 2, 3])

    out = mathops.array_to_true_numpy(TorchLike())
    np.testing.assert_array_equal(out, np.array([1, 2, 3]))


def test_array_to_true_numpy_handles_multiple_inputs(sample_data_2d):
    out = mathops.array_to_true_numpy(1, sample_data_2d)
    assert out[0] == 1
    assert out[1] is sample_data_2d
