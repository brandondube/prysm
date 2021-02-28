"""Tests for numerical propagation routines."""
import pytest

import numpy as np

from prysm import propagation, fttools
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
    # magic number 4 - a bit unclear, but accounts for non-energy
    # conserving fft; sf is to satisfy parseval's theorem
    sf = fttools.mdft._norm(wf.data, 2, unfocus_fft.samples_x) * 4
    unfocus_mdft = wf.unfocus_fixed_sampling(
        efl=1,
        sample_spacing=unfocus_fft.sample_spacing,
        samples=unfocus_fft.samples_x)

    assert np.allclose(unfocus_fft.data, unfocus_mdft.data/sf)


def test_focus_fft_mdft_equivalent_Wavefront():
    dx = 1
    z = np.random.rand(SAMPLES, SAMPLES)
    wf = propagation.Wavefront(dx=dx, cmplx_field=z, wavelength=HeNe, space='pupil')
    unfocus_fft = wf.focus(Q=2, efl=1)
    sf = fttools.mdft._norm(wf.data, 2, unfocus_fft.data.shape[1])
    unfocus_mdft = wf.focus_fixed_sampling(
        efl=1,
        dx=unfocus_fft.dx,
        samples=unfocus_fft.data.shape[1])

    assert np.allclose(unfocus_fft.data, unfocus_mdft.data*sf)


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
