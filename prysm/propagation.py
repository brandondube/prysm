"""Numerical optical propagation."""
from .mathops import engine as e
from ._basicdata import BasicData
from .fttools import pad2d


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
    if Q != 1:
        padded_wavefront = pad2d(wavefunction, Q)
    else:
        padded_wavefront = wavefunction

    impulse_response = e.fft.ifftshift(e.fft.fft2(e.fft.fftshift(padded_wavefront), norm=norm))
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
    x : `numpy.ndarray`
        x axis unit, 1D ndarray
    y : `numpy.ndarray`
        y axis unit, 1D ndarray

    """
    s = wavefunction.shape
    samples_x, samples_y = s[1] * Q, s[0] * Q
    sample_spacing_x = pupil_sample_to_psf_sample(pupil_sample=input_sample_spacing,  # factor of
                                                  samples=samples_x,                  # 1e3 corrects
                                                  wavelength=wavelength,              # for unit
                                                  efl=prop_dist) / 1e3                # translation
    sample_spacing_y = pupil_sample_to_psf_sample(pupil_sample=input_sample_spacing,  # factor of
                                                  samples=samples_y,                  # 1e3 corrects
                                                  wavelength=wavelength,              # for unit
                                                  efl=prop_dist) / 1e3                # translation
    x = e.arange(-1 * int(e.ceil(samples_x / 2)), int(e.floor(samples_x / 2))) * sample_spacing_x
    y = e.arange(-1 * int(e.ceil(samples_y / 2)), int(e.floor(samples_y / 2))) * sample_spacing_y
    return x, y


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


class Wavefront(BasicData):
    """(Complex) representation of a wavefront."""
    _data_attr = 'fcn'

    def __init__(self, x, y, fcn, wavelength):
        """Create a new Wavefront instance.

        Parameters
        ----------
        x : `numpy.ndarray`
            x coordinates
        y : `numpy.ndarray`
            y coordinates
        fcn : `numpy.ndarray`
            complex-valued wavefront array
        wavelength : `float`
            wavelength of light, microns

        """
        super.__init__(x=x, y=y, data=fcn)
        self.wavelength = wavelength
