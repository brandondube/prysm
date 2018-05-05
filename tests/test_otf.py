''' Optical Transfer Function (OTF) unit tests.
'''
import pytest

import numpy as np

from prysm import otf


SAMPLES = 32
LIM = 1e3


@pytest.fixture
def mtf():
    x, y = np.linspace(-LIM, LIM, SAMPLES), np.linspace(-LIM, LIM, SAMPLES)
    xx, yy = np.meshgrid(x, y)
    dat = np.sin(xx)
    return otf.MTF(data=dat, unit_x=x)  # do not pass unit_y, simultaneous test for unit_y=None


def test_mtf_plot2d_functions(mtf):
    fig, ax = mtf.plot2d()
    assert fig
    assert ax


def test_mtf_plot_tan_sag_functions(mtf):
    fig, ax = mtf.plot_tan_sag()
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
