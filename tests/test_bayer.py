"""Tests to verify proper bayer functionality."""
import pytest

import numpy as np

from prysm import bayer

TEST_CFAs = ['rggb', 'bggr']


N = 100


@pytest.mark.parametrize('cfa', TEST_CFAs)
def test_decomposite_recomposite_inverse(cfa):
    data = np.random.rand(N, N)
    fwd = bayer.decomposite_bayer(data, cfa)
    rev = bayer.recomposite_bayer(*fwd, cfa=cfa)
    assert (data == rev).all()


@pytest.mark.parametrize('cfa', TEST_CFAs)
def test_composite_does_nothing_if_all_same_data(cfa):
    data = np.random.rand(N, N)
    fwd = bayer.composite_bayer(data, data, data, data, cfa=cfa)
    assert (fwd == data).all()


@pytest.mark.parametrize('cfa', TEST_CFAs)
def test_demosaic_malvar_right_shape(cfa):
    data = np.random.rand(N, N)
    trichrom = bayer.demosaic_malvar(data, cfa)
    assert trichrom.shape == (N, N, 3)


def test_wb_prescale_applies_cfa_order_in_place():
    mosaic = np.ones((4, 4), dtype=float)

    bayer.wb_prescale(mosaic, 2, 3, 5, 7, cfa='rggb')

    np.testing.assert_array_equal(mosaic[bayer.top_left], 2)
    np.testing.assert_array_equal(mosaic[bayer.top_right], 3)
    np.testing.assert_array_equal(mosaic[bayer.bottom_left], 5)
    np.testing.assert_array_equal(mosaic[bayer.bottom_right], 7)


def test_wb_prescale_safe_desaturates_all_channels_by_largest_ratio():
    mosaic = np.ones((2, 2), dtype=float)
    mosaic[bayer.top_left] = 12

    bayer.wb_prescale(mosaic, 2, 4, 6, 8, cfa='rggb', safe=True, saturation=6)

    np.testing.assert_allclose(mosaic[bayer.top_left], 12)
    np.testing.assert_allclose(mosaic[bayer.top_right], 2)
    np.testing.assert_allclose(mosaic[bayer.bottom_left], 3)
    np.testing.assert_allclose(mosaic[bayer.bottom_right], 4)


def test_wb_postscale_applies_rgb_gains_in_place():
    rgb = np.ones((2, 2, 3), dtype=float)

    bayer.wb_postscale(rgb, 2, 3, 5)

    np.testing.assert_array_equal(rgb[..., 0], 2)
    np.testing.assert_array_equal(rgb[..., 1], 3)
    np.testing.assert_array_equal(rgb[..., 2], 5)
