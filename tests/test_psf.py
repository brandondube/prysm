"""Tests for PSFs."""
import pytest

import numpy as np

from prysm import psf
from prysm.coordinates import cart_to_polar, make_xy_grid

SAMPLES = 32
LIM = 100


@pytest.fixture
def tpsf():
    xx, yy = make_xy_grid(SAMPLES, diameter=LIM*2)
    rho, phi = cart_to_polar(xx, yy)
    dat = psf.airydisk(rho, 10, 0.55)
    return dat, xx[0, 1]-xx[0, 0]


@pytest.fixture
def tpsf_dense():
    xx, yy = make_xy_grid(SAMPLES*4, diameter=LIM/2)
    rho, phi = cart_to_polar(xx, yy)
    dat = psf.airydisk(rho, 10, 0.55)
    return dat, xx[0, 1]-xx[0, 0]


def test_airydisk_aft_origin():
    assert 1 == pytest.approx(psf.airydisk_ft(0, 3.14, 2.718))


def test_size_estimation_accurate(tpsf_dense):
    # tpsf is F/10 at lambda = 0.55 microns, so the size parameters are:
    # FWHM
    # 1.22 * .55 * 10 = 6.71 um
    # the 1/e^2 width is about the same as the airy radius
    tpsf, dx = tpsf_dense
    true_airy_radius = 1.22 * .55 * 10
    true_fwhm = 1.028 * .55 * 10
    fwhm = psf.fwhm(tpsf, dx)
    one_over_e = psf.one_over_e(tpsf, dx)
    one_over_esq = psf.one_over_e_sq(tpsf, dx)
    assert fwhm == pytest.approx(true_fwhm, abs=1)
    assert one_over_e == pytest.approx(true_airy_radius, abs=0.4)
    assert one_over_esq == pytest.approx(true_airy_radius*1.414, abs=.8)  # sqrt(2) is an empirical fudge factor.
    # TODO: find a better test for 1/e^2


def test_centroid_correct(tpsf_dense):
    tpsf, _ = tpsf_dense
    cy, cx = psf.centroid(tpsf, unit='pixels')
    ty, tx = (s/2 for s in tpsf.shape)
    assert cy == pytest.approx(ty, .1)
    assert cx == pytest.approx(tx, .1)


def test_autowindow_functions(tpsf):
    tpsf, _ = tpsf
    assert psf.autocrop(tpsf, 10).any
