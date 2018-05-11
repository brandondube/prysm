"""Numerical optical propagation."""
from .fttools import pad2d
from .util import pupil_sample_to_psf_sample

from prysm import mathops as m


def prop_pupil_plane_to_psf_plane(wavefunction, input_sample_spacing, prop_dist, wavelength, Q):
    """Propagate a pupil plane to a PSF plane and compute the grid along which the PSF exists.

    Parameters
    ----------
    wavefunction : `numpy.ndarray`
        the pupil wavefunction
    input_sample_spacing : `float`
        spacing between samples in the pupil plane
    prop_dist : `float`
        propagation distance along the z distance
    wavelength : `float`
        wavelength of light
    Q : `float`
        oversampling / padding factor

    Returns
    -------
    psf : `numpy.ndarray`
        Description
    unit_x : `numpy.ndarray`
        x axis unit, 1D ndarray
    unit_y : `numpy.ndarray`
        y axis unit, 1D ndarray

    """
    padded_wavefront = pad2d(wavefunction, Q)
    impulse_response = m.ifftshift(m.fft2(m.fftshift(padded_wavefront)))
    psf = abs(impulse_response) ** 2

    s = wavefunction.shape
    samples_x, samples_y = s[0] * Q, s[1] * Q
    sample_spacing = pupil_sample_to_psf_sample(pupil_sample=input_sample_spacing,  # factor of
                                                num_samples=samples_x,              # 1e3 corrects
                                                wavelength=wavelength,              # for unit
                                                efl=prop_dist) / 1e3                # translation
    unit_x = m.arange(-1 * (samples_x // 2), samples_x // 2) * sample_spacing
    unit_y = m.arange(-1 * (samples_y // 2), samples_y // 2) * sample_spacing
    return psf, unit_x, unit_y
