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


def test_unfocus_fft_mdft_equivalent_Wavefront():
    z = np.random.rand(128, 128)
    dx = 1
    wf = propagation.Wavefront(dx=dx, cmplx_field=z, wavelength=HeNe, space='psf')
    unfocus_fft = wf.unfocus(Q=2, efl=1)
    unfocus_mdft = wf.unfocus_fixed_sampling(
        efl=1,
        dx=unfocus_fft.dx,
        samples=unfocus_fft.data.shape)

    assert np.allclose(unfocus_fft.data, unfocus_mdft.data)


def test_focus_fft_mdft_equivalent_Wavefront():
    dx = 1
    z = np.random.rand(SAMPLES, SAMPLES)
    wf = propagation.Wavefront(dx=dx, cmplx_field=z, wavelength=HeNe, space='pupil')
    unfocus_fft = wf.focus(Q=2, efl=1)
    unfocus_mdft = wf.focus_fixed_sampling(
        efl=1,
        dx=unfocus_fft.dx,
        samples=unfocus_fft.data.shape)

    assert np.allclose(unfocus_fft.data, unfocus_mdft.data)


def test_frespace_functions():
    dx = 1
    z = np.random.rand(SAMPLES, SAMPLES)
    wf = propagation.Wavefront(dx=dx, cmplx_field=z, wavelength=HeNe, space='pupil')
    wf = wf.free_space(1, 1)
    assert wf


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


def test_can_mul_wavefronts():
    data = np.random.rand(2, 2).astype(np.complex128)
    wf = propagation.Wavefront(cmplx_field=data, dx=1, wavelength=.6328)
    wf2 = wf * 2
    assert wf2


def test_can_div_wavefronts():
    data = np.random.rand(2, 2).astype(np.complex128)
    wf = propagation.Wavefront(cmplx_field=data, dx=1, wavelength=.6328)
    wf2 = wf / 2
    assert wf2


def test_wavefront_scalar_sub_and_div_use_wavefront_as_left_operand():
    data = np.asarray([[4, 8], [16, 32]], dtype=np.complex128)
    wf = propagation.Wavefront(cmplx_field=data, dx=1, wavelength=.6328)

    assert np.allclose((wf / 2).data, data / 2)
    assert np.allclose((wf - 2).data, data - 2)


def test_to_fpm_and_back_return_more_uses_wavefront_fpm_dx():
    wf = propagation.Wavefront(
        cmplx_field=np.ones((4, 4), dtype=np.complex128),
        dx=1,
        wavelength=.6328)
    fpm = propagation.Wavefront(
        cmplx_field=np.ones((4, 4), dtype=np.complex128),
        dx=.25,
        wavelength=.6328,
        space='psf')

    _, at_fpm, after_fpm = wf.to_fpm_and_back(
        efl=10,
        fpm=fpm,
        return_more=True)

    assert at_fpm.dx == fpm.dx
    assert after_fpm.dx == fpm.dx


def test_to_fpm_and_back_backprop_accepts_wavefront_fpm():
    fpm = propagation.Wavefront(
        cmplx_field=np.ones((4, 4), dtype=np.complex128),
        dx=.25,
        wavelength=.6328,
        space='psf')

    out = propagation.to_fpm_and_back_backprop(
        wavefunction=np.ones((4, 4), dtype=np.complex128),
        dx=1,
        wavelength=.6328,
        efl=10,
        fpm=fpm)

    assert out.shape == (4, 4)


def test_to_fpm_and_back_backprop_passes_shift_to_both_propagations(monkeypatch):
    shifts = []

    def fake_unfocus_backprop(wavefunction, input_dx, prop_dist, wavelength,
                              output_dx, output_samples, shift=(0, 0),
                              method='mdft'):
        shifts.append(('unfocus', shift))
        return np.ones(output_samples, dtype=np.complex128)

    def fake_focus_backprop(wavefunction, input_dx, prop_dist, wavelength,
                            output_dx, output_samples, shift=(0, 0),
                            method='mdft'):
        shifts.append(('focus', shift))
        return np.ones(output_samples, dtype=np.complex128)

    monkeypatch.setattr(
        propagation,
        'unfocus_fixed_sampling_backprop',
        fake_unfocus_backprop)
    monkeypatch.setattr(
        propagation,
        'focus_fixed_sampling_backprop',
        fake_focus_backprop)

    propagation.to_fpm_and_back_backprop(
        wavefunction=np.ones((4, 4), dtype=np.complex128),
        dx=1,
        wavelength=.6328,
        efl=10,
        fpm=np.ones((4, 4), dtype=np.complex128),
        fpm_dx=.25,
        shift=(1, -2))

    assert shifts == [('unfocus', (1, -2)), ('focus', (1, -2))]


@pytest.mark.parametrize('func,args', [
    (propagation.focus_fixed_sampling, (np.ones((4, 4)), 1, 10, .6328, .25, 4)),
    (propagation.focus_fixed_sampling_backprop, (np.ones((4, 4)), 1, 10, .6328, .25, 4)),
    (propagation.unfocus_fixed_sampling, (np.ones((4, 4)), .25, 10, .6328, 1, 4)),
    (propagation.unfocus_fixed_sampling_backprop, (np.ones((4, 4)), .25, 10, .6328, 1, 4)),
])
def test_fixed_sampling_rejects_unsupported_method(func, args):
    with pytest.raises(ValueError, match='unsupported propagation method'):
        func(*args, method='bogus')


def test_precomputed_angular_spectrum_functions():
    data = np.random.rand(2, 2)
    wf = propagation.Wavefront(cmplx_field=data, dx=1, wavelength=.6328)
    tf = propagation.angular_spectrum_transfer_function(2, wf.wavelength, wf.dx, 1)
    wf2 = wf.free_space(tf=tf)
    assert wf2


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
