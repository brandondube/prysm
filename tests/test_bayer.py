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


@pytest.mark.parametrize('cfa', TEST_CFAs)
def test_wb_prescale_functions(cfa):
    data = np.random.rand(N, N)
    bayer.wb_prescale(data, 1, 2, 3, 4, cfa)


def test_wb_scale_functions():
    data = np.random.rand(N, N, 3)
    bayer.wb_scale(data, 1, 2, 3)
