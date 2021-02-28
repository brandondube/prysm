"""Unit tests for utility functions."""
import pytest

import numpy as np
from matplotlib import pyplot as plt

from prysm import util

ARR_SIZE = 32


def test_rms_is_zero_for_single_value_array():
    arr = np.ones((ARR_SIZE, ARR_SIZE))
    assert util.rms(arr) == pytest.approx(1)


def test_ecdf_binary_distribution():
    x = np.asarray([0, 0, 0, 1, 1, 1])
    x, y = util.ecdf(x)
    assert np.allclose(np.unique(x), np.asarray([0, 1]))  # TODO: more rigorous tests.


def test_sort_xy():
    x = np.linspace(10, 0, 10)
    y = np.linspace(1, 10, 10)
    xx, yy = util.sort_xy(x, y)
    assert xx == tuple(reversed(x))
    assert yy == tuple(reversed(y))


def test_Sa_gives_correct_value():
    ary = np.array([1, 2, 3, 4, 5])
    assert util.Sa(ary) == 1.2
