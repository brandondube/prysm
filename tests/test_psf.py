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


def test_centered_odd_array_has_zero_spatial_centroid():
    data = np.zeros((5, 5))
    data[2, 2] = 1
    assert psf.centroid(data, dx=1) == pytest.approx((0, 0))


def test_estimate_size_accepts_numeric_metric_and_first_crossing():
    x, y = make_xy_grid(65, dx=0.1)
    data = np.exp(-(x*x + y*y))

    numeric = psf.estimate_size(data, 0.5, dx=0.1, criteria='first')
    named = psf.estimate_size(data, 'fwhm', dx=0.1, criteria='first')

    assert numeric == pytest.approx(named)


def test_autocrop_pads_near_array_boundary():
    data = np.zeros((5, 5))
    data[0, 0] = 1
    out = psf.autocrop(data, 4)
    assert out.shape == (4, 4)


def test_autocrop_returns_requested_centered_window(tpsf):
    tpsf, _ = tpsf

    cropped = psf.autocrop(tpsf, 10)
    cy, cx = (int(c) for c in psf.centroid(tpsf, unit='pixels'))
    expected = tpsf[cy - 5:cy + 5, cx - 5:cx + 5]
    assert cropped.shape == (10, 10)
    np.testing.assert_allclose(cropped, expected)
