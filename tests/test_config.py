"""Tests verifying the functionality of the global prysm config."""
import pytest

import numpy as np

from prysm.conf import config

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
