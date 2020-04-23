"""Tests for detector modeling capabilities."""
import pytest

import numpy as np

from prysm import detector, psf, Convolvable

import matplotlib as mpl
mpl.use('TkAgg')


@pytest.fixture
def sample_psf():
    ps = psf.AiryDisk(4, .55, 20, 64)
    return Convolvable(x=ps.x, y=ps.y, data=ps.data, has_analytic_ft=False)


@pytest.fixture
def sample_detector():
    return detector.Detector(10)


@pytest.mark.dependency(name='sample_conv')
def test_detector_can_sample_convolvable(sample_detector, sample_psf):
    assert sample_detector.capture(sample_psf)


@pytest.mark.dependency(depends=['sample_conv'])
def test_detector_can_save_result(tmpdir, sample_detector, sample_psf):
    p = tmpdir.mkdir('detector_out').join('out.png')
    sample_detector.capture(sample_psf)
    sample_detector.save_image(str(p))


def test_detector_can_show(sample_detector, sample_psf):
    sample_detector.capture(sample_psf)
    fig, ax = sample_detector.show_image()
    assert fig
    assert ax


def test_detector_bindown_doesnt_fail(sample_detector):
    samples = 8
    x = np.arange(samples) * sample_detector.pitch / 2
    y = np.arange(samples) * sample_detector.pitch / 2
    z = np.ones((samples, samples))
    c = Convolvable(x=x, y=y, data=z)
    sample_detector.capture(c)
    assert sample_detector.last.sample_spacing == sample_detector.pitch


def test_olpf_render_doesnt_crash():
    olpf = detector.OLPF(5, samples_x=32, sample_spacing=0.5)
    assert olpf


def test_olpf_aft_correct_at_origin():
    olpf = detector.OLPF(5)
    assert olpf.analytic_ft(0, 0) == 1


def test_detector_properties(sample_detector):
    sd = sample_detector
    assert sd.pitch == 10
    assert sd.fill_factor == 1
    assert pytest.approx(sd.fs, 1 / sd.pitch * 1e3)
    assert pytest.approx(sd.nyquist, sd.fs / 2)


def test_detector_pitch_change_correctness():
    d = detector.Detector(5)
    d.pitch = 10
    assert d.pitch == 10
