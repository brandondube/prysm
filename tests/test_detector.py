"""Tests for detector modeling capabilities."""
import pytest

from prysm import detector, psf


@pytest.fixture
def sample_psf():
    return psf.AiryDisk(4, .55, 20, 64)


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


def test_olpf_render_doesnt_crash():
    olpf = detector.OLPF(5, samples_x=32, sample_spacing=0.5)
    assert olpf
