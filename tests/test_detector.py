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


def test_analytic_fts_function():
    # these numbers have no meaning, and the sense of x and y is wrong.  Just
    # testing for crashes.
    # TODO: more thorough tests
    olpf_ft = detector.olpf_ft(x, y, 1.234, 4.567)
    assert olpf_ft.any()
    pixel_ft = detector.pixel_ft(x, y, 9.876, 5.4321)
    assert pixel_ft.any()


def test_detector_functions():
    d = detector.Detector(0.1, 8, 200, 60_000, .5, 14, 1)
    field = np.ones((128, 128))
    img = d.expose(field)
    assert img.any()
