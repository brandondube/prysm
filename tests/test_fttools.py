"""Unit tests specific to fourier transform tools."""

import pytest

import numpy as np

from prysm import fttools


@pytest.mark.parametrize('samples', [8, 16, 32, 64, 128, 256, 512, 1024])
def test_mtp_equivalent_to_fft(samples):
    inp = np.random.rand(samples, samples)
    fft = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(inp)))
    mtp = fttools.mdft.dft2(inp, 1, samples)
    assert np.allclose(fft, mtp)
