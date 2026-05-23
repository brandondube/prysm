"""Tests for numerical propagation routines."""
import pytest

import numpy as np

from prysm import propagation, coordinates, geometry, polynomials
from prysm.wavelengths import HeNe


SAMPLES = 32


@pytest.mark.parametrize('dzeta', [1 / 128.0, 1 / 256.0, 11.123 / 128.0, 1e10 / 2048.0])
def test_psf_to_pupil_sample_inverts_pupil_to_psf_sample(dzeta):
    samples, wvl, efl = 128, 0.55, 10
    psf_sample = propagation.pupil_sample_to_psf_sample(dzeta, samples, wvl, efl)
    dzeta2 = propagation.psf_sample_to_pupil_sample(psf_sample, samples, wvl, efl)
    assert dzeta2 == dzeta


def test_obj_oriented_wavefront_focusing_reverses():
    z = np.random.rand(128, 128)
    dx = 1
    wf = propagation.Wavefront(dx=dx, cmplx_field=z, wavelength=HeNe)
    wf2 = wf.focus(1, 1).unfocus(1, 1)  # first is efl, meaningless.  second is Q, we neglect padding here
    assert np.allclose(wf.data, wf2.data)


@pytest.mark.parametrize('Q', [1, 1.5, 2])
def test_focus_backprop_is_adjoint(Q):
    rng = np.random.default_rng(789)
    x = rng.normal(size=(9, 12)) + 1j * rng.normal(size=(9, 12))
    y = rng.normal(size=propagation.focus(x, Q=Q).shape)
    y = y + 1j * rng.normal(size=y.shape)

    lhs = np.vdot(propagation.focus(x, Q=Q), y)
    rhs = np.vdot(x, propagation.focus_backprop(y, Q=Q))

    np.testing.assert_allclose(lhs, rhs, atol=1e-12)


@pytest.mark.parametrize('Q', [1, 1.5, 2])
def test_unfocus_backprop_is_adjoint(Q):
    rng = np.random.default_rng(987)
    x = rng.normal(size=(9, 12)) + 1j * rng.normal(size=(9, 12))
    y = rng.normal(size=propagation.unfocus(x, Q=Q).shape)
    y = y + 1j * rng.normal(size=y.shape)

    lhs = np.vdot(propagation.unfocus(x, Q=Q), y)
    rhs = np.vdot(x, propagation.unfocus_backprop(y, Q=Q))

    np.testing.assert_allclose(lhs, rhs, atol=1e-12)


def test_wavefront_focus_backprop_metadata_and_data():
    rng = np.random.default_rng(135)
    dx = 0.25
    efl = 10
    Q = 2
    data = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
    wf = propagation.Wavefront(dx=dx, cmplx_field=data, wavelength=HeNe, space='pupil')
    psf = wf.focus(efl=efl, Q=Q)
    grad_data = rng.normal(size=psf.data.shape) + 1j * rng.normal(size=psf.data.shape)
    grad = propagation.Wavefront(dx=psf.dx, cmplx_field=grad_data,
                                 wavelength=HeNe, space='psf')

    back = grad.focus_backprop(efl=efl, Q=Q)

    np.testing.assert_allclose(back.data, propagation.focus_backprop(grad_data, Q=Q))
    assert back.data.shape == wf.data.shape
    assert back.dx == pytest.approx(wf.dx)
    assert back.space == 'pupil'


def test_wavefront_unfocus_backprop_metadata_and_data():
    rng = np.random.default_rng(246)
    dx = 0.1
    efl = 10
    Q = 2
    data = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
    wf = propagation.Wavefront(dx=dx, cmplx_field=data, wavelength=HeNe, space='psf')
    pupil = wf.unfocus(efl=efl, Q=Q)
    grad_data = rng.normal(size=pupil.data.shape) + 1j * rng.normal(size=pupil.data.shape)
    grad = propagation.Wavefront(dx=pupil.dx, cmplx_field=grad_data,
                                 wavelength=HeNe, space='pupil')

    back = grad.unfocus_backprop(efl=efl, Q=Q)

    np.testing.assert_allclose(back.data, propagation.unfocus_backprop(grad_data, Q=Q))
    assert back.data.shape == wf.data.shape
    assert back.dx == pytest.approx(wf.dx)
    assert back.space == 'psf'


def test_unfocus_fft_mdft_equivalent_Wavefront():
    z = np.random.rand(128, 128)
    dx = 1
    wf = propagation.Wavefront(dx=dx, cmplx_field=z, wavelength=HeNe, space='psf')
    unfocus_fft = wf.unfocus(Q=2, efl=1)
    mdft = wf.prepare_executor(efl=1, dx=unfocus_fft.dx, samples=unfocus_fft.data.shape)
    unfocus_mdft = wf.unfocus_dft(mdft)

    assert np.allclose(unfocus_fft.data, unfocus_mdft.data)


def test_focus_fft_mdft_equivalent_Wavefront():
    dx = 1
    z = np.random.rand(SAMPLES, SAMPLES)
    wf = propagation.Wavefront(dx=dx, cmplx_field=z, wavelength=HeNe, space='pupil')
    focus_fft = wf.focus(Q=2, efl=1)
    mdft = wf.prepare_executor(efl=1, dx=focus_fft.dx, samples=focus_fft.data.shape)
    focus_mdft = wf.focus_dft(mdft)

    assert np.allclose(focus_fft.data, focus_mdft.data)


def test_free_space_zero_distance_is_identity():
    z = np.random.rand(SAMPLES, SAMPLES)
    wf = propagation.Wavefront(dx=1, cmplx_field=z, wavelength=HeNe, space='pupil')

    out = wf.free_space(0)

    np.testing.assert_allclose(out.data, wf.data, atol=1e-12)
    assert out.dx == wf.dx
    assert out.wavelength == wf.wavelength


def test_talbot_distance_correct():
    wvl = 123.456
    a = 987.654321
    truth = wvl / (1 - np.sqrt(1 - wvl**2/a**2))
    tal = propagation.talbot_distance(a, wvl)
    assert truth == pytest.approx(tal, abs=.1)


def test_fresnel_number_correct():
    wvl = 123.456
    a = 987.654321
    z = 5
    fres = propagation.fresnel_number(a, z, wvl)
    assert fres == (a**2 / (z * wvl))


def test_wavefront_multiply_and_divide_apply_to_data():
    data = np.arange(4, dtype=float).reshape(2, 2).astype(np.complex128)
    wf = propagation.Wavefront(cmplx_field=data, dx=1, wavelength=.6328)

    np.testing.assert_allclose((wf * 2).data, data * 2)
    np.testing.assert_allclose((wf / 2).data, data / 2)


def test_wavefront_scalar_arithmetic_operand_order():
    # non-commutative ops (sub, truediv) must compute self OP other, not other OP self
    data = (np.random.rand(2, 2) + 1).astype(np.complex128)
    wf = propagation.Wavefront(cmplx_field=data, dx=1, wavelength=.6328)
    assert np.allclose((wf - 1.0).data, data - 1.0)
    assert np.allclose((wf / 2.0).data, data / 2.0)


def test_to_fpm_and_back_backprop_accepts_wavefront_fpm():
    # regression: previously raised AttributeError on fpm.dtype / fpm.conj()
    dx = 1.0
    z = (np.random.rand(SAMPLES, SAMPLES) + 1j * np.random.rand(SAMPLES, SAMPLES))
    wf = propagation.Wavefront(cmplx_field=z, dx=dx, wavelength=HeNe, space='pupil')
    fpm_data = (np.random.rand(SAMPLES, SAMPLES)
                + 1j * np.random.rand(SAMPLES, SAMPLES)).astype(np.complex128)
    fpm = propagation.Wavefront(cmplx_field=fpm_data, dx=0.1, wavelength=HeNe, space='psf')
    mdft = wf.prepare_executor(efl=10.0, dx=fpm.dx, samples=fpm.data.shape)
    out = wf.to_fpm_and_back(fpm=fpm, executor=mdft)
    grad = out.to_fpm_and_back_backprop(fpm=fpm, executor=mdft)
    assert grad.data.shape == wf.data.shape


def _real_vdot(a, b):
    return np.real(np.vdot(a, b))


def test_to_fpm_and_back_backprop_returns_fpm_gradient():
    rng = np.random.default_rng(123)
    dx = 1.0
    z = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
    wf = propagation.Wavefront(cmplx_field=z, dx=dx, wavelength=HeNe, space='pupil')
    fpm_data = rng.normal(size=(8, 8))
    fpm = propagation.Wavefront(cmplx_field=fpm_data, dx=0.1, wavelength=HeNe, space='psf')
    mdft = wf.prepare_executor(efl=10.0, dx=fpm.dx, samples=fpm.data.shape)
    out, at_fpm, _ = wf.to_fpm_and_back(fpm=fpm, executor=mdft, return_more=True)
    outbar_data = rng.normal(size=out.data.shape) + 1j * rng.normal(size=out.data.shape)
    outbar = propagation.Wavefront(cmplx_field=outbar_data, dx=out.dx,
                                   wavelength=HeNe, space=out.space)

    _, fpm_bar = outbar.to_fpm_and_back_backprop(
        fpm=fpm, executor=mdft, return_fpm_grad=True, field_at_fpm=at_fpm,
    )

    y, x = 3, 4
    eps = 1e-6
    fpm_plus = fpm_data.copy()
    fpm_minus = fpm_data.copy()
    fpm_plus[y, x] += eps
    fpm_minus[y, x] -= eps
    j_plus = _real_vdot(outbar_data, wf.to_fpm_and_back(fpm=fpm_plus, executor=mdft).data)
    j_minus = _real_vdot(outbar_data, wf.to_fpm_and_back(fpm=fpm_minus, executor=mdft).data)
    fd = (j_plus - j_minus) / (2 * eps)
    assert fpm_bar.data[y, x] == pytest.approx(fd, rel=1e-6, abs=1e-8)


def test_babinet_backprop_returns_fpm_and_lyot_gradients():
    rng = np.random.default_rng(456)
    dx = 1.0
    z = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
    wf = propagation.Wavefront(cmplx_field=z, dx=dx, wavelength=HeNe, space='pupil')
    fpm_data = rng.normal(size=(8, 8))
    lyot_data = rng.normal(size=(8, 8))
    fpm = propagation.Wavefront(cmplx_field=fpm_data, dx=0.1, wavelength=HeNe, space='psf')
    lyot = propagation.Wavefront(cmplx_field=lyot_data, dx=dx, wavelength=HeNe, space='pupil')
    mdft = wf.prepare_executor(efl=10.0, dx=fpm.dx, samples=fpm.data.shape)
    out, at_fpm, _, at_lyot = wf.babinet(lyot=lyot, fpm=fpm, executor=mdft, return_more=True)
    outbar_data = rng.normal(size=out.data.shape) + 1j * rng.normal(size=out.data.shape)
    outbar = propagation.Wavefront(cmplx_field=outbar_data, dx=out.dx,
                                   wavelength=HeNe, space=out.space)

    _, fpm_bar, lyot_bar = outbar.babinet_backprop(
        lyot=lyot, fpm=fpm, executor=mdft,
        field_at_fpm=at_fpm, field_at_lyot=at_lyot,
        return_fpm_grad=True, return_lyot_grad=True,
    )

    eps = 1e-6
    fy, fx = 2, 5
    fpm_plus = fpm_data.copy()
    fpm_minus = fpm_data.copy()
    fpm_plus[fy, fx] += eps
    fpm_minus[fy, fx] -= eps
    fpm_plus = propagation.Wavefront(fpm_plus, HeNe, fpm.dx, 'psf')
    fpm_minus = propagation.Wavefront(fpm_minus, HeNe, fpm.dx, 'psf')
    j_plus = _real_vdot(outbar_data, wf.babinet(lyot=lyot, fpm=fpm_plus, executor=mdft).data)
    j_minus = _real_vdot(outbar_data, wf.babinet(lyot=lyot, fpm=fpm_minus, executor=mdft).data)
    fd_fpm = (j_plus - j_minus) / (2 * eps)

    ly, lx = 6, 1
    lyot_plus = lyot_data.copy()
    lyot_minus = lyot_data.copy()
    lyot_plus[ly, lx] += eps
    lyot_minus[ly, lx] -= eps
    lyot_plus = propagation.Wavefront(lyot_plus, HeNe, lyot.dx, 'pupil')
    lyot_minus = propagation.Wavefront(lyot_minus, HeNe, lyot.dx, 'pupil')
    j_plus = _real_vdot(outbar_data, wf.babinet(lyot=lyot_plus, fpm=fpm, executor=mdft).data)
    j_minus = _real_vdot(outbar_data, wf.babinet(lyot=lyot_minus, fpm=fpm, executor=mdft).data)
    fd_lyot = (j_plus - j_minus) / (2 * eps)

    assert fpm_bar.data[fy, fx] == pytest.approx(fd_fpm, rel=1e-6, abs=1e-8)
    assert lyot_bar.data[ly, lx] == pytest.approx(fd_lyot, rel=1e-6, abs=1e-8)


def test_precomputed_angular_spectrum_matches_direct_zero_distance():
    data = np.random.rand(4, 4)
    wf = propagation.Wavefront(cmplx_field=data, dx=1, wavelength=.6328)
    tf = propagation.angular_spectrum_transfer_function(wf.data.shape, wf.wavelength, wf.dx, z=0)

    out = wf.free_space(tf=tf)

    np.testing.assert_allclose(out.data, wf.data, atol=1e-12)


def test_thinlens_hopkins_agree():
    # F/10 beam
    x, y = coordinates.make_xy_grid(128, diameter=11)
    dx = x[0, 1] - x[0, 0]
    r = np.hypot(x, y)
    amp = geometry.circle(5, r)
    phs = polynomials.hopkins(0, 2, 0, r/5, 0, 1) * (1.975347661 * HeNe * 1000)  # 1000, nm to um
    wf = propagation.Wavefront.from_amp_and_phase(amp, phs, HeNe, dx)

    # easy case is to choose thin lens efl = 10,000
    # which will result in an overall focal length of 99.0 mm
    # solve defocus delta z relation, then 1000 = 8 * .6328 * 100 * x
    #                                  x = 1000 / 8 / .6328 / 100
    #                                    = 1.975347661
    psf = wf.focus(efl=100, Q=2).intensity

    no_phs_wf = propagation.Wavefront.from_amp_and_phase(amp, None, HeNe, dx)
    tl = propagation.Wavefront.thin_lens(10_000, HeNe, x, y)
    wf = no_phs_wf * tl
    psf2 = wf.focus(efl=100, Q=2).intensity
    assert np.allclose(psf.data, psf2.data, rtol=1e-5)
