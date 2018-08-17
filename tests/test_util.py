"""Unit tests for utility functions."""
import pytest

import numpy as np
from matplotlib import pyplot as plt

from prysm import util

ARR_SIZE = 32


@pytest.mark.parametrize('num', [1, 3, 5, 7, 9, 11, 13, 15, 991, 100000000000001])
def test_is_odd_odd_numbers(num):
    assert util.is_odd(num)


@pytest.mark.parametrize('num', [0, 2, 4, 6, 8, 10, 12, 14, 1000, 100000000000000])
def test_is_odd_even_numbers(num):
    assert not util.is_odd(num)


@pytest.mark.parametrize('num', [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192])
def test_is_power_of_2_powers_of_2(num):
    assert util.is_power_of_2(num)


@pytest.mark.parametrize('num', [1, 3, 5, 7, 1000, -2])
def test_is_power_of_2_non_powers_of_2(num):
    assert not util.is_power_of_2(num)


def test_rms_is_zero_for_single_value_array():
    arr = np.ones((ARR_SIZE, ARR_SIZE))
    assert util.rms(arr) == pytest.approx(1)


def test_ecdf_binary_distribution():
    x = np.asarray([0, 0, 0, 1, 1, 1])
    x, y = util.ecdf(x)
    assert np.allclose(np.unique(x), np.asarray([0, 1]))  # TODO: more rigorous tests.


def test_fold_array_function():
    arr = np.ones((ARR_SIZE, ARR_SIZE))
    assert util.fold_array(arr).all()
    assert util.fold_array(arr, axis=0).all()


def test_guarantee_array_functionality():
    a_float = 5.0
    an_int = 10
    a_str = 'foo'
    an_array = np.empty(1)
    assert util.guarantee_array(a_float)
    assert util.guarantee_array(an_int)
    assert util.guarantee_array(an_array)
    with pytest.raises(ValueError):
        util.guarantee_array(a_str)


def test_sort_xy():
    x = np.linspace(10, 0, 10)
    y = np.linspace(1, 10, 10)
    xx, yy = util.sort_xy(x, y)
    assert xx == tuple(reversed(x))
    assert yy == tuple(reversed(y))


def test_share_fig_ax_figure_number_remains_unchanged():
    fig, ax = plt.subplots()
    fig2, ax2 = util.share_fig_ax(fig, ax)
    assert fig.number == fig2.number


def test_share_fig_ax_produces_figure_and_axis():
    fig, ax = util.share_fig_ax()
    assert fig
    assert ax


def test_share_fig_ax_produces_an_axis():
    fig = plt.figure()
    fig, ax = util.share_fig_ax(fig)
    assert ax is not None
