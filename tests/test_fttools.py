"""Unit tests specific to fourier transform tools."""

import pytest

import numpy as np

from prysm import fttools

ARRAY_SIZES = (8, 16, 32, 64, 128, 256, 512, 1024)


@pytest.mark.parametrize('samples', ARRAY_SIZES)
def test_mtp_equivalent_to_fft(samples):
    inp = np.random.rand(samples, samples)
    fft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(inp)))
    sf = fttools.mdft._norm(inp, 1, samples)
    mtp = fttools.mdft.dft2(inp, 1, samples) * sf
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
