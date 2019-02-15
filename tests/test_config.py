"""Tests verifying the functionality of the global prysm config."""
import pytest

import numpy as np

from prysm import config

PRECISIONS = {
    32: np.float32,
    64: np.float64,
}
PRECISIONS_COMPLEX = {
    32: np.complex64,
    64: np.complex128
}


@pytest.mark.parametrize('precision', [32, 64])
def test_set_precision(precision):
    config.precision = precision
    assert config.precision == PRECISIONS[precision]
    assert config.precision_complex == PRECISIONS_COMPLEX[precision]


def test_rejects_bad_precision():
    with pytest.raises(ValueError):
        config.precision = 1


# must make certain the backend is set to numpy last to avoid cuda errors for rest of test suite
@pytest.mark.parametrize('backend', ['np'])
def test_set_backend(backend):
    config.backend = backend
    assert config.backend == backend


def test_rejects_bad_backend():
    with pytest.raises(ValueError):
        config.backend = 'foo'


def test__force_testenv_backend_numpy():
    config.backend = 'np'
    assert config


@pytest.mark.parametrize('zbase', [0, 1])
def test_set_zernike_base(zbase):
    config.zernike_base = zbase
    assert config.zernike_base == zbase


def test_rejects_bad_zernike_base():
    with pytest.raises(ValueError):
        config.zernike_base = 2
