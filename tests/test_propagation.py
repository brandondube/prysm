import pytest

from prysm import propagation

@pytest.mark.parametrize('dzeta', [1 / 128.0, 1 / 256.0, 11.123 / 128.0, 1e10 / 2048.0])
def test_psf_to_pupil_sample_inverts_pupil_to_psf_sample(dzeta):
    samples, wvl, efl = 128, 0.55, 10
    psf_sample = propagation.pupil_sample_to_psf_sample(dzeta, samples, wvl, efl)
    assert propagation.psf_sample_to_pupil_sample(psf_sample, samples, wvl, efl) == dzeta
