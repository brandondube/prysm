"""Tests verifying the functionality of the global prysm config."""
import pytest

import numpy as np

from prysm.conf import config

PRECISIONS = {
    16: np.float16,
    32: np.float32,
    64: np.float64,
}
PRECISIONS_COMPLEX = {
    16: np.complex64,
    32: np.complex64,
    64: np.complex128
}


@pytest.fixture(autouse=True)
def restore_precision():
    old = config.precision
    try:
        yield
    finally:
        config.precision = old


@pytest.mark.parametrize('precision', [16, np.int64(32), 64])
def test_set_precision_from_bit_depth(precision):
    config.precision = precision
    assert config.precision == PRECISIONS[int(precision)]
    assert config.precision_complex == PRECISIONS_COMPLEX[int(precision)]


@pytest.mark.parametrize('precision, expected, expected_complex', [
    (np.float16, np.float16, np.complex64),
    (np.dtype('float32'), np.float32, np.complex64),
    ('float64', np.float64, np.complex128),
    (float, np.float64, np.complex128),
])
def test_set_precision_from_dtype_like(precision, expected, expected_complex):
    config.precision = precision
    assert config.precision == expected
    assert config.precision_complex == expected_complex


def test_rejects_bad_precision():
    with pytest.raises(ValueError):
        config.precision = 1


@pytest.mark.parametrize('precision', [np.int32, 'int16', np.complex64])
def test_rejects_non_real_float_precision(precision):
    with pytest.raises(ValueError):
        config.precision = precision
