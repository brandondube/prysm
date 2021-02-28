"""Tests for the mathops submodule."""
import pytest

import numpy as np

from prysm import mathops

np.random.seed(1234)
TEST_ARR_SIZE = 32


@pytest.fixture
def sample_data_2d():
    return np.random.rand(TEST_ARR_SIZE, TEST_ARR_SIZE)


# below here, tests purely for function not accuracy
def test_fft2(sample_data_2d):
    result = mathops.engine.fft.fft2(sample_data_2d)
    assert type(result) is np.ndarray


def test_ifft2(sample_data_2d):
    result = mathops.engine.fft.ifft2(sample_data_2d)
    assert type(result) is np.ndarray


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

