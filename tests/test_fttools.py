"""Unit tests specific to fourier transform tools."""

import pytest

import numpy as np

from prysm import fttools

ARRAY_SIZES = (8, 16, 32, 64, 128, 256, 512, 1024)

# one power of two, one odd number, one even non power of two
ARRAY_SIZES_FOR_PAD = (8, 9, 12)


@pytest.mark.parametrize('samples', ARRAY_SIZES)
def test_mtp_equivalent_to_fft(samples):
    inp = np.random.rand(samples, samples)
    fft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(inp)))
    mtp = fttools.mdft.dft2(inp, 1, samples)
    assert np.allclose(fft, mtp)


@pytest.mark.parametrize('samples', ARRAY_SIZES)
def test_mtp_reverses_self(samples):
    inp = np.random.rand(samples, samples)
    fwd = fttools.mdft.dft2(inp, 1, samples)
    back = fttools.mdft.idft2(fwd, 1, samples)
    assert np.allclose(inp, back)


def test_mtp_cache_empty_zeros_nbytes():
    fttools.mdft.clear()
    assert fttools.mdft.nbytes() == 0


@pytest.mark.parametrize('shape', ARRAY_SIZES_FOR_PAD)
def test_pad2d_cropcenter_adjoints(shape):
    inp = np.random.rand(shape, shape)
    intermediate = fttools.pad2d(inp, Q=2)
    out = fttools.crop_center(intermediate, inp.shape)
    assert np.allclose(inp, out)


@pytest.mark.parametrize('samples', ARRAY_SIZES)
def test_czt_equiv_to_fft(samples):
    inp = np.random.rand(samples, samples)
    fft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(inp)))
    czt = fttools.czt.czt2(inp, 1, samples)
    assert np.allclose(fft, czt)


@pytest.mark.parametrize('samples', ARRAY_SIZES)
def test_czt_reverses_self_(samples):
    inp = np.random.rand(samples, samples)
    fwd = fttools.czt.czt2(inp, 1, samples)
    back = fttools.czt.iczt2(fwd, 1, samples)
    assert np.allclose(inp, back)
