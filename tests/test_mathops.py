"""Tests for the mathops submodule."""
import pytest

import numpy as np

from prysm import mathops

np.random.seed(1234)
TEST_ARR_SIZE = 32


@pytest.fixture
def sample_data_2d():
    return np.random.rand(TEST_ARR_SIZE, TEST_ARR_SIZE)


@pytest.mark.parametrize('num', [1, 3, 5, 7, 9, 11, 13, 15, 991, 100000000000001])
def test_is_odd_odd_numbers(num):
    assert mathops.is_odd(num)


@pytest.mark.parametrize('num', [0, 2, 4, 6, 8, 10, 12, 14, 1000, 100000000000000])
def test_is_odd_even_numbers(num):
    assert not mathops.is_odd(num)


@pytest.mark.parametrize('num', [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192])
def test_is_power_of_2_powers_of_2(num):
    assert mathops.is_power_of_2(num)


@pytest.mark.parametrize('num', [1, 3, 5, 7, 1000, -2])
def test_is_power_of_2_non_powers_of_2(num):
    assert not mathops.is_power_of_2(num)


@pytest.mark.parametrize('shim_name,probe_attr', [
    ('np', 'arange'),
    ('fft', 'fft'),
    ('ndimage', 'gaussian_filter'),
    ('interpolate', 'interp1d'),
    ('optimize', 'brentq'),
    ('signal', 'windows'),
    ('linalg', 'lu_factor'),
    ('linalg', 'lu_solve'),
])
def test_backend_shim_default_routes_to_scipy_numpy(shim_name, probe_attr):
    shim = getattr(mathops, shim_name)
    assert hasattr(shim, probe_attr)


def test_set_backend_to_defaults_restores_optimize_and_signal():
    # poke the shims, then verify defaults reseat them.
    sentinel = object()
    mathops.optimize._srcmodule = sentinel
    mathops.signal._srcmodule = sentinel
    mathops.linalg._srcmodule = sentinel
    mathops.set_backend_to_defaults()
    assert mathops.optimize._srcmodule is mathops._optimize
    assert mathops.signal._srcmodule is mathops._signal
    assert mathops.linalg._srcmodule is mathops._linalg
    assert hasattr(mathops.optimize, 'brentq')
    assert hasattr(mathops.signal, 'windows')
    assert hasattr(mathops.linalg, 'lu_factor')


def test_linalg_lu_factor_and_solve():
    """The default linalg shim wraps scipy.linalg's LU factor/solve."""
    M = np.array([[4.0, 3.0], [6.0, 3.0]])
    b = np.array([1.0, 1.0])
    lu = mathops.linalg.lu_factor(M)
    x = mathops.linalg.lu_solve(lu, b)
    np.testing.assert_allclose(M @ x, b)
