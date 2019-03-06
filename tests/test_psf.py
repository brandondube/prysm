"""Tests for PSFs."""
import pytest

import numpy as np

from prysm import psf
from prysm.coordinates import cart_to_polar

SAMPLES = 32
LIM = 100


@pytest.fixture
def tpsf():
    x = y = np.linspace(-LIM, LIM, SAMPLES)
    xx, yy = np.meshgrid(x, y)
    rho, phi = cart_to_polar(xx, yy)
    dat = psf.airydisk(rho, 10, 0.55)
    return psf.PSF(dat, x, y)


@pytest.fixture
def tpsf_mutate():
    x = y = np.linspace(-LIM, LIM, SAMPLES)
    xx, yy = np.meshgrid(x, y)
    rho, phi = cart_to_polar(xx, yy)
    dat = psf.airydisk(rho, 10, 0.55)
    _psf = psf.PSF(dat, x, y)
    _psf.fno = 10
    _psf.wavelength = 0.55
    return _psf


def test_psf_plot2d_functions(tpsf):
    fig, ax = tpsf.plot2d()
    assert fig
    assert ax


def test_psf_plot_slice_xy_functions(tpsf):
    fig, ax = tpsf.plot_slice_xy()
    assert fig
    assert ax


def test_plot_encircled_energy_functions(tpsf):
    fig, ax = tpsf.plot_encircled_energy(axlim=10)
    assert fig
    assert ax


def test_renorm_functions(tpsf_mutate):
    mutated = tpsf_mutate._renorm()
    assert mutated.data.max() == 1


def test_polychromatic_functions():
    from prysm import Pupil
    p1 = Pupil(wavelength=0.55)
    p2 = Pupil(wavelength=0.5)
    p3 = Pupil(wavelength=0.6)

    psfs = [psf.PSF.from_pupil(p, 1) for p in [p1, p2, p3]]
    poly = psf.PSF.polychromatic(psfs)
    assert isinstance(poly, psf.PSF)


def test_airydisk_aft_origin():
    ad = psf.AiryDisk(0.5, 0.5)
    assert ad.analytic_ft(0, 0)[0, 0] == 1


def test_encircled_energy_radius_functions(tpsf_mutate):
    assert tpsf_mutate.ee_radius(0.9)


def test_encircled_energy_radius_diffraction_functions(tpsf_mutate):
    assert tpsf_mutate.ee_radius_diffraction(0.9)


def test_encircled_energy_radius_ratio_functions(tpsf_mutate):
    assert tpsf_mutate.ee_radius_ratio_to_diffraction(0.9) > 1
