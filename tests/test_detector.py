"""Tests for detector modeling capabilities."""
import pytest

import numpy as np

from prysm import detector, psf
from prysm.convolution import Convolvable

import matplotlib as mpl
mpl.use('Agg')


@pytest.fixture
def sample_psf():
    ps = psf.AiryDisk(4, .55, 20, 64)
    return Convolvable(x=ps.x, y=ps.y, data=ps.data, has_analytic_ft=False)


@pytest.fixture
def sample_detector():
    return detector.Detector(10)


def test_detector_can_sample_convolvable(sample_detector, sample_psf):
    assert sample_detector.capture(sample_psf)


def test_olpf_render_doesnt_crash():
    olpf = detector.OLPF(5, samples_x=32, sample_spacing=0.5)
    assert olpf


def test_olpf_ft_correct_at_origin():
    olpf = detector.OLPF(5)
    assert olpf.analytic_ft(0, 0) == 1
