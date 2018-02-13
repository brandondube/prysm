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
    return otf.MTF(data=dat, unit_x=x, unit_y=y)


def test_mtf_plot2d_functions(mtf):
    fig, ax = mtf.plot2d()
    assert fig
    assert ax


def test_mtf_plot_tan_sag_functions(mtf):
    fig, ax = mtf.plot_tan_sag()
    assert fig
    assert ax


