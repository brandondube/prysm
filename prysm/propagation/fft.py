"""FFT-based pupil <-> focal propagation and sample-spacing conversions.
"""
from ..mathops import fft
from ._kernels import _maybe_pad, _adjoint_pad2d


def focus(wavefunction, Q):
    """Propagate a pupil plane to a PSF plane.

    Parameters
    ----------
    wavefunction : ndarray
        the pupil wavefunction
    Q : float
        oversampling / padding factor

    Returns
    -------
    psf : ndarray
        point spread function

    """
    padded_wavefront = _maybe_pad(wavefunction, Q)
    impulse_response = fft.fftshift(fft.fft2(fft.ifftshift(padded_wavefront), norm='ortho'))
    return impulse_response


def focus_adjoint(wavefunction, Q):
    """Apply the adjoint of focus.

    Parameters
    ----------
    wavefunction : ndarray
        gradient at the PSF plane
    Q : float
        oversampling / padding factor used for the forward propagation

    Returns
    -------
    ndarray
        gradient at the pupil plane

    """
    padded_grad = fft.fftshift(fft.ifft2(fft.ifftshift(wavefunction), norm='ortho'))
    return _adjoint_pad2d(padded_grad, Q)


def unfocus(wavefunction, Q):
    """Propagate a PSF plane to a pupil plane.

    Parameters
    ----------
    wavefunction : ndarray
        the pupil wavefunction
    Q : float
        oversampling / padding factor

    Returns
    -------
    pupil : ndarray
        field in the pupil plane

    """
    padded_wavefront = _maybe_pad(wavefunction, Q)
    return fft.fftshift(fft.ifft2(fft.ifftshift(padded_wavefront), norm='ortho'))


def unfocus_adjoint(wavefunction, Q):
    """Apply the adjoint of unfocus.

    Parameters
    ----------
    wavefunction : ndarray
        gradient at the pupil plane
    Q : float
        oversampling / padding factor used for the forward propagation

    Returns
    -------
    ndarray
        gradient at the PSF plane

    """
    padded_grad = fft.fftshift(fft.fft2(fft.ifftshift(wavefunction), norm='ortho'))
    return _adjoint_pad2d(padded_grad, Q)


def Q_for_sampling(input_diameter, prop_dist, wavelength, output_dx):
    """Value of Q for a given output sampling, given input sampling.

    Parameters
    ----------
    input_diameter : float
        diameter of the input array in millimeters
    prop_dist : float
        propagation distance along the z distance, millimeters
    wavelength : float
        wavelength of light, microns
    output_dx : float
        sampling in the output plane, microns

    Returns
    -------
    float
        requesite Q

    """
    resolution_element = (wavelength * prop_dist) / (input_diameter)
    return resolution_element / output_dx


def pupil_sample_to_psf_sample(pupil_sample, samples, wavelength, efl):
    """Convert pupil sample spacing to PSF sample spacing.  fλ/D or Q.

    Parameters
    ----------
    pupil_sample : float
        sample spacing in the pupil plane
    samples : int
        number of samples present in both planes (must be equal)
    wavelength : float
        wavelength of light, in microns
    efl : float
        effective focal length of the optical system in mm

    Returns
    -------
    float
        the sample spacing in the PSF plane

    """
    return (efl * wavelength) / (pupil_sample * samples)


def psf_sample_to_pupil_sample(psf_sample, samples, wavelength, efl):
    """Convert PSF sample spacing to pupil sample spacing.

    Parameters
    ----------
    psf_sample : float
        sample spacing in the PSF plane
    samples : int
        number of samples present in both planes (must be equal)
    wavelength : float
        wavelength of light, in microns
    efl : float
        effective focal length of the optical system in mm

    Returns
    -------
    float
        the sample spacing in the pupil plane

    """
    return (efl * wavelength) / (psf_sample * samples)
