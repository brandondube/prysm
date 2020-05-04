"""Tests for numerical propagation routines."""
import pytest

import numpy as np

from prysm import propagation
from prysm.wavelengths import HeNe


SAMPLES = 32


@pytest.mark.parametrize('dzeta', [1 / 128.0, 1 / 256.0, 11.123 / 128.0, 1e10 / 2048.0])
def test_psf_to_pupil_sample_inverts_pupil_to_psf_sample(dzeta):
    samples, wvl, efl = 128, 0.55, 10
    psf_sample = propagation.pupil_sample_to_psf_sample(dzeta, samples, wvl, efl)
    assert propagation.psf_sample_to_pupil_sample(psf_sample, samples, wvl, efl) == dzeta


def test_obj_oriented_wavefront_focusing_reverses():
    x = y = np.arange(128, dtype=np.float32)
    z = np.random.rand(128, 128)
    wf = propagation.Wavefront(x=x, y=y, fcn=z, wavelength=HeNe)
    wf2 = wf.focus(1, 1).unfocus(1, 1)  # first is efl, meaningless.  second is Q, we neglect padding at the moment
    assert np.allclose(wf.fcn, wf2.fcn)


def test_unfocus_fft_mdft_equivalent_Wavefront():
    x = y = np.linspace(-1, 1, SAMPLES)
    z = np.random.rand(SAMPLES, SAMPLES)
    wf = propagation.Wavefront(x=x, y=y, fcn=z, wavelength=HeNe, space='psf')
    unfocus_fft = wf.unfocus(Q=2, efl=1)
    unfocus_mdft = wf.unfocus_fixed_sampling(
        efl=1,
        sample_spacing=unfocus_fft.sample_spacing,
        samples=unfocus_fft.samples_x)

    assert np.allclose(unfocus_fft.data, unfocus_mdft.data)


def test_focus_fft_mdft_equivalent_Wavefront():
    x = y = np.linspace(-1, 1, SAMPLES)
    z = np.random.rand(SAMPLES, SAMPLES)
    wf = propagation.Wavefront(x=x, y=y, fcn=z, wavelength=HeNe, space='pupil')
    unfocus_fft = wf.focus(Q=2, efl=1)
    unfocus_mdft = wf.focus_fixed_sampling(
        efl=1,
        sample_spacing=unfocus_fft.sample_spacing,
        samples=unfocus_fft.samples_x)

    assert np.allclose(unfocus_fft.data, unfocus_mdft.data)
