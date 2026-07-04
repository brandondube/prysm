"""Unit tests specific to fourier transform tools."""

import pytest

import numpy as np

from prysm import fttools
from prysm.fttools import MDFT, CZT, fftrange

ARRAY_SIZES = (8, 64, 512)

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


@pytest.mark.parametrize(
    'input_shape,output_shape,forward_left_first,adjoint_left_first',
    [
        ((3, 9), (2, 9), True, False),
        ((9, 3), (9, 2), False, True),
    ],
)
def test_mdft_rectangular_multiply_order_matches_explicit_chain(
        input_shape, output_shape, forward_left_first, adjoint_left_first):
    rng = np.random.default_rng(123)
    ny, nx = input_shape
    my, mx = output_shape
    x = fftrange(nx).astype(float)
    y = fftrange(ny).astype(float)
    fx = fftrange(mx).astype(float) / nx
    fy = fftrange(my).astype(float) / ny
    op = MDFT(x, y, fx, fy, norm=0.25)

    assert op._forward_left_first is forward_left_first
    assert op._adjoint_left_first is adjoint_left_first

    inp = rng.normal(size=input_shape) + 1j * rng.normal(size=input_shape)
    grad = rng.normal(size=output_shape) + 1j * rng.normal(size=output_shape)

    fwd = op(inp)
    expected_fwd = (op.Ey @ inp @ op.Ex.T) * op.norm
    np.testing.assert_allclose(fwd, expected_fwd, atol=1e-12)

    adj = op.adjoint(grad)
    expected_adj = (op.Ey.conj().T @ grad @ op.Ex.conj()) * op.norm
    np.testing.assert_allclose(adj, expected_adj, atol=1e-12)


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
def test_czt_reverses_self_complex(samples):
    inp = np.random.rand(samples, samples) + 1.0j * np.random.rand(samples, samples)
    x, y, fx, fy = _fft_equivalent_coords(samples)
    fwd = CZT(x, y, fx, fy, sign=-1)(inp) / samples
    back = CZT(x, y, fx, fy, sign=+1)(fwd) / samples
    assert np.allclose(inp, back)


@pytest.mark.parametrize('sign', (-1, 1))
@pytest.mark.parametrize(
    'input_shape,output_shape',
    [
        ((7, 9), (5, 6)),
        ((5, 6), (7, 9)),
    ],
)
def test_czt_adjoint_is_adjoint(sign, input_shape, output_shape):
    rng = np.random.default_rng(123)
    ny, nx = input_shape
    my, mx = output_shape
    x = fftrange(nx).astype(float) * 0.2
    y = fftrange(ny).astype(float) * 0.17
    fx = (fftrange(mx).astype(float) + 0.25) * 0.13
    fy = (fftrange(my).astype(float) - 0.5) * 0.11
    op = CZT(x, y, fx, fy, sign=sign, norm=0.3)

    inp = rng.normal(size=input_shape) + 1j * rng.normal(size=input_shape)
    grad = rng.normal(size=output_shape) + 1j * rng.normal(size=output_shape)

    lhs = np.vdot(op(inp), grad)
    rhs = np.vdot(inp, op.adjoint(grad))
    np.testing.assert_allclose(lhs, rhs, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('sign', (-1, 1))
def test_czt_matches_mdft_for_shifted_uniform_grids(sign):
    rng = np.random.default_rng(123)
    input_shape = (7, 9)
    output_shape = (5, 6)
    ny, nx = input_shape
    my, mx = output_shape
    x = fftrange(nx).astype(float) * 0.2 + 0.33
    y = fftrange(ny).astype(float) * 0.17 - 0.41
    fx = (fftrange(mx).astype(float) + 0.25) * 0.13
    fy = (fftrange(my).astype(float) - 0.5) * 0.11
    inp = rng.normal(size=input_shape) + 1j * rng.normal(size=input_shape)

    mdft = MDFT(x, y, fx, fy, sign=sign)
    czt = CZT(x, y, fx, fy, sign=sign)
    np.testing.assert_allclose(czt(inp), mdft(inp), rtol=1e-12, atol=1e-12)
