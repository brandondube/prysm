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
