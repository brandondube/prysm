''' Unit tests for PSFs.
'''
import pytest

import numpy as np

from prysm import psf
from prysm.coordinates import cart_to_polar

SAMPLES = 32
LIM = 100


@pytest.fixture
def tpsf():
    x, y = np.linspace(-LIM, LIM, SAMPLES), np.linspace(-LIM, LIM, SAMPLES)
    xx, yy = np.meshgrid(x, y)
    rho, phi = cart_to_polar(xx, yy)
    dat = psf.airydisk(rho, 10, 0.55)
    return psf.PSF(dat, x[1] - x[0])


def test_psf_plot2d_functions(tpsf):
    fig, ax = tpsf.plot2d()
    assert fig
    assert ax


def test_psf_plot_slice_xy_functions(tpsf):
    fig, ax = tpsf.plot_slice_xy()
    assert fig
    assert ax


def test_plot_encircled_energy_functions(tpsf):
    fig, ax = tpsf.plot_encircled_energy()
    assert fig
    assert ax
