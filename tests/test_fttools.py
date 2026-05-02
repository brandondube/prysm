"""Unit tests specific to fourier transform tools."""

import pytest

import numpy as np

from prysm import fttools
from prysm.fttools import MDFT, CZT, fftrange

ARRAY_SIZES = (8, 16, 32, 64, 128, 256, 512, 1024)

# one power of two, one odd number, one even non power of two
ARRAY_SIZES_FOR_PAD = (8, 9, 12)


def _fft_equivalent_coords(samples):
    """(x, y, fx, fy) such that MDFT(...)(inp) * (1/N) equals
    fftshift(fft2(ifftshift(inp), norm='ortho')) for an N×N input."""
    x = fftrange(samples).astype(float)
    y = fftrange(samples).astype(float)
    fx = fftrange(samples).astype(float) / samples
    fy = fftrange(samples).astype(float) / samples
    return x, y, fx, fy


@pytest.mark.parametrize('samples', ARRAY_SIZES)
def test_mtp_equivalent_to_fft(samples):
    inp = np.random.rand(samples, samples)
    fft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(inp), norm='ortho'))
    x, y, fx, fy = _fft_equivalent_coords(samples)
    mtp = MDFT(x, y, fx, fy)(inp) / samples
    assert np.allclose(fft, mtp)


@pytest.mark.parametrize('samples', ARRAY_SIZES)
def test_mtp_reverses_self(samples):
    inp = np.random.rand(samples, samples)
    x, y, fx, fy = _fft_equivalent_coords(samples)
    op = MDFT(x, y, fx, fy)
    fwd = op(inp)
    # adjoint of a unitary-up-to-scale operator: (E·E†) = N·I per axis,
    # so adjoint(forward(inp)) = N² · inp for 2D
    back = op.adjoint(fwd) / (samples * samples)
    assert np.allclose(inp, back)


def test_mdft_nbytes_reports_basis_size():
    samples = 32
    x, y, fx, fy = _fft_equivalent_coords(samples)
    op = MDFT(x, y, fx, fy)
    # two complex 32×32 basis matrices, fp64 => 32*32*16 bytes each
    assert op.nbytes() == 2 * samples * samples * 16


@pytest.mark.parametrize('shape', ARRAY_SIZES_FOR_PAD)
def test_pad2d_cropcenter_adjoints(shape):
    inp = np.random.rand(shape, shape)
    intermediate = fttools.pad2d(inp, Q=2)
    out = fttools.crop_center(intermediate, inp.shape)
    assert np.allclose(inp, out)


@pytest.mark.parametrize('samples', ARRAY_SIZES)
def test_czt_equiv_to_fft(samples):
    inp = np.random.rand(samples, samples)
    fft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(inp), norm='ortho'))
    x, y, fx, fy = _fft_equivalent_coords(samples)
    czt = CZT(x, y, fx, fy)(inp) / samples
    assert np.allclose(fft, czt)


@pytest.mark.parametrize('samples', ARRAY_SIZES)
def test_czt_reverses_self_(samples):
    inp = np.random.rand(samples, samples)
    x, y, fx, fy = _fft_equivalent_coords(samples)
    fwd = CZT(x, y, fx, fy, sign=-1)(inp) / samples
    back = CZT(x, y, fx, fy, sign=+1)(fwd) / samples
    assert np.allclose(inp, back)


@pytest.mark.parametrize('samples', ARRAY_SIZES)
def test_czt_reverses_self_complex(samples):
    inp = np.random.rand(samples, samples) + 1.0j * np.random.rand(samples, samples)
    x, y, fx, fy = _fft_equivalent_coords(samples)
    fwd = CZT(x, y, fx, fy, sign=-1)(inp) / samples
    back = CZT(x, y, fx, fy, sign=+1)(fwd) / samples
    assert np.allclose(inp, back)
