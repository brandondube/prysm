"""Unit tests for bugs in utility functions."""

import numpy as np

from prysm import util


def test_smooth_returns_original_array_when_window_length_is_one():
    data = np.arange(10)
    smooth_data = util.smooth(data, 1)
    assert np.allclose(data, smooth_data)
