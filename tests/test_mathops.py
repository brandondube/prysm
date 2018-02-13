''' unit tests for the mathops submodule.
'''
import pytest

import numpy as np

from prysm import mathops

np.random.seed(1234)
TEST_ARR_SIZE = 32


@pytest.fixture
def sample_data_2d():
    return np.random.rand(TEST_ARR_SIZE, TEST_ARR_SIZE)


def test_mathops_handles_own_jit_and_vectorize_definitions():
    from importlib import reload
    from unittest import mock

    class FakeNumba():
        __version__ = '0.35.0'

    with mock.patch.dict('sys.modules', {'numba': FakeNumba()}):
        reload(mathops)  # may have side effect of disabling numba for downstream tests.

        def foo():
            pass

        foo_jit = mathops.jit(foo)
        foo_vec = mathops.vectorize(foo)

        assert foo_jit == foo
        assert foo_vec == foo


# below here, tests purely for function not accuracy
def test_fft2(sample_data_2d):
    result = mathops.fft2(sample_data_2d)
    assert type(result) is np.ndarray


def test_ifft2(sample_data_2d):
    result = mathops.ifft2(sample_data_2d)
    assert type(result) is np.ndarray
