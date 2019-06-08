"""Optical Transfer Function (OTF) unit tests."""
import pytest

import numpy as np

from prysm import otf
from prysm.fttools import forward_ft_unit

import matplotlib
matplotlib.use('TkAgg')

SAMPLES = 32
LIM = 1e3


@pytest.fixture
def mtf():
    x, y = forward_ft_unit(1/1e3, 128), forward_ft_unit(1/1e3, 128)
    xx, yy = np.meshgrid(x, y)
    dat = np.sin(xx)
    return otf.MTF(data=dat, x=x)  # do not pass y, simultaneous test for y=None


def test_mtf_plot2d_functions(mtf):
    fig, ax = mtf.plot2d()
    assert fig
    assert ax


@pytest.mark.parametrize('azimuth', [None, 0, [0, 90, 90, 90]])
def test_mtf_exact_polar_functions(mtf, azimuth):
    freqs = [0, 1, 2, 3]
    mtf_ = mtf.exact_polar(freqs, azimuth)
    assert type(mtf_) is np.ndarray


@pytest.mark.parametrize('y', [None, 0, [0, 1, 2, 3]])
def test_mtf_exact_xy_functions(mtf, y):
    x = [0, 1, 2, 3]
    mtf_ = mtf.exact_xy(x, y)
    assert type(mtf_) is np.ndarray


def test_frompupil_functions():
    from prysm import Pupil
    pu = Pupil()
    mt = otf.MTF.from_pupil(pu, 2)
    assert mt


def test_azavg_plot_functions(mtf):
    fig, ax = mtf.plot_azimuthal_average()
    assert fig
    assert ax
