"""Tests for detector modeling capabilities."""
import pytest

import numpy as np

from prysm import detector, coordinates

import matplotlib as mpl
mpl.use('Agg')

SAMPLES = 128

x, y = coordinates.make_xy_grid(SAMPLES, dx=1)
r, t = coordinates.cart_to_polar(x, y)


def test_pixel_shades_properly():
    px = detector.pixel(x, y, 10, 10)
    # 121 samples should be white, 5 row/col on each side of zero, plus zero,
    # = 11x11 = 121
    assert px.sum() == 121


def test_analytic_fts_match_closed_forms():
    fx = np.asarray([0, 0.25])
    fy = np.asarray([0, 0.5])

    olpf = detector.olpf_ft(fx, fy, 2, 3)
    pixel = detector.pixel_ft(fx, fy, 2, 3)

    np.testing.assert_allclose(olpf, np.cos(4 * fx) * np.cos(6 * fy))
    np.testing.assert_allclose(pixel, np.sinc(2 * fx) * np.sinc(3 * fy))


def test_detector_expose_zero_signal_is_bias_limited_dn():
    d = detector.Detector(0, 0, 10, 60_000, 2, 12, 1)
    field = np.zeros((3, 4))

    img = d.expose(field)

    assert img.dtype == np.uint16
    np.testing.assert_array_equal(img, 5)


def test_detector_expose_applies_lut_after_adc():
    lut = np.arange(16, dtype=np.uint8) + 10
    d = detector.Detector(0, 0, 3, 15, 1, 4, 1, lut=lut)

    img = d.expose(np.zeros((1, 2)))

    np.testing.assert_array_equal(img, 13)


def test_bindown_tile_reciprocate():
    d = np.random.rand(16, 16)
    binned = detector.bindown(d, 4, 'sum')
    tiled = detector.tile(binned, 4, 'sum')
    assert tiled.shape == d.shape
    assert tiled.sum() == pytest.approx(d.sum())  # energy conservation scaling
