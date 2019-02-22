"""Numerical optical propagation."""
from .fttools import pad2d

from prysm import mathops as m


def prop_pupil_plane_to_psf_plane(wavefunction, Q, incoherent=True, norm=None):
    """Propagate a pupil plane to a PSF plane and compute the grid along which the PSF exists.

    Parameters
    ----------
    wavefunction : `numpy.ndarray`
        the pupil wavefunction
    Q : `float`
        oversampling / padding factor
    incoherent : `bool`, optional
        whether to return the incoherent (real valued) PSF, or the
        coherent (complex-valued) PSF.  Incoherent = |coherent|^2
    norm : `str`, {None, 'ortho'}
        normalization parameter passed directly to numpy/cupy fft

    Returns
    -------
    psf : `numpy.ndarray`
        incoherent point spread function

    """
    padded_wavefront = pad2d(wavefunction, Q)
    impulse_response = m.ifftshift(m.fft2(m.fftshift(padded_wavefront), norm=norm))
    if incoherent:
        return abs(impulse_response) ** 2
    else:
        return impulse_response


def prop_pupil_plane_to_psf_plane_units(wavefunction, input_sample_spacing, prop_dist, wavelength, Q):
    """Compute the ordinate axes for a pupil plane to PSF plane propagation.

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
    unit_x : `numpy.ndarray`
        x axis unit, 1D ndarray
    unit_y : `numpy.ndarray`
        y axis unit, 1D ndarray

    """
    s = wavefunction.shape
    samples_x, samples_y = s[1] * Q, s[0] * Q
    sample_spacing_x = pupil_sample_to_psf_sample(pupil_sample=input_sample_spacing,  # factor of
                                                  samples=samples_x,              # 1e3 corrects
                                                  wavelength=wavelength,              # for unit
                                                  efl=prop_dist) / 1e3                # translation
    sample_spacing_y = pupil_sample_to_psf_sample(pupil_sample=input_sample_spacing,  # factor of
                                                  samples=samples_y,              # 1e3 corrects
                                                  wavelength=wavelength,              # for unit
                                                  efl=prop_dist) / 1e3                # translation
    unit_x = m.arange(-1 * int(m.ceil(samples_x / 2)), int(m.floor(samples_x / 2))) * sample_spacing_x
    unit_y = m.arange(-1 * int(m.ceil(samples_y / 2)), int(m.floor(samples_y / 2))) * sample_spacing_y
    return unit_x, unit_y


def pupil_sample_to_psf_sample(pupil_sample, samples, wavelength, efl):
    """Convert pupil sample spacing to PSF sample spacing.

    Parameters
    ----------
    pupil_sample : `float`
        sample spacing in the pupil plane
    samples : `int`
        number of samples present in both planes (must be equal)
    wavelength : `float`
        wavelength of light, in microns
    efl : `float`
        effective focal length of the optical system in mm

    Returns
    -------
    `float`
        the sample spacing in the PSF plane

    """
    return (wavelength * efl * 1e3) / (pupil_sample * samples)


def psf_sample_to_pupil_sample(psf_sample, samples, wavelength, efl):
    """Convert PSF sample spacing to pupil sample spacing.

    Parameters
    ----------
    psf_sample : `float`
        sample spacing in the PSF plane
    samples : `int`
        number of samples present in both planes (must be equal)
    wavelength : `float`
        wavelength of light, in microns
    efl : `float`
        effective focal length of the optical system in mm

    Returns
    -------
    `float`
        the sample spacing in the pupil plane

    """
    return (wavelength * efl * 1e3) / (psf_sample * samples)
