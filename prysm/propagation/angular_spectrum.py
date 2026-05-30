"""Angular-spectrum (plane-to-plane) free space propagation and related metrics.
"""
from ..conf import config
from ..mathops import np, fft
from ..fttools import pad2d, crop_center
from ._kernels import _shape_before_pad


def angular_spectrum(field, wvl, dx, z, Q=2, tf=None):
    """Propagate a field via the angular spectrum method.

    Parameters
    ----------
    field : ndarray
        2D array of complex electric field values
    wvl : float
        wavelength of light, microns
    z : float
        propagation distance, units of millimeters
    dx : float
        cartesian sample spacing, units of millimeters
    Q : float
        sampling factor used.  Q>=2 for Nyquist sampling of incoherent fields
    tf : ndarray
        if not None, clobbers all other arguments
        transfer function for the propagation

    Returns
    -------
    ndarray
        2D ndarray of the output field, complex

    """
    if tf is not None:
        return fft.ifft2(fft.fft2(field) * tf)

    if Q != 1:
        field = pad2d(field, Q=Q)

    transfer_function = angular_spectrum_transfer_function(field.shape, wvl, dx, z)
    forward = fft.fft2(field)
    return fft.ifft2(forward*transfer_function)


def angular_spectrum_adjoint(field, wvl, dx, z, Q=2, tf=None):
    """Apply the adjoint of angular_spectrum.

    Parameters
    ----------
    field : ndarray
        gradient at the output plane of the angular spectrum propagation
    wvl : float
        wavelength of light, microns
    z : float
        propagation distance used for the forward propagation, millimeters
    dx : float
        cartesian sample spacing, units of millimeters
    Q : float
        sampling factor used for the forward propagation
    tf : ndarray
        if not None, clobbers all other arguments
        transfer function used for the forward propagation

    Returns
    -------
    ndarray
        gradient at the input plane

    """
    if tf is None:
        tf = angular_spectrum_transfer_function(field.shape, wvl, dx, z)
        out_shape = _shape_before_pad(field.shape, Q)
    else:
        out_shape = field.shape

    out = fft.ifft2(fft.fft2(field) * np.conj(tf))
    if out_shape == field.shape:
        return out
    return crop_center(out, out_shape)


def angular_spectrum_transfer_function(samples, wvl, dx, z):
    """Precompute the transfer function of free space.

    Parameters
    ----------
    samples : int or tuple
        (y,x) or (r,c) samples in the output array
    wvl : float
        wavelength of light, microns
    dx : float
        intersample spacing, mm
    z : float
        propagation distance, mm

    Returns
    -------
    ndarray
        ndarray of shape samples containing the complex valued transfer function
        such that X = fft2(x); xhat = ifft2(X*tf) is signal x after free space propagation

    """
    if isinstance(samples, int):
        samples = (samples, samples)

    wvl = wvl / 1e3
    ky, kx = (fft.fftfreq(s, dx).astype(config.precision) for s in samples)
    kxx = kx * kx
    kyy = ky * ky

    prefix = -1j*np.pi*wvl*z
    tfx = np.exp(prefix*kxx)
    tfy = np.exp(prefix*kyy)
    return np.outer(tfy, tfx)


def fresnel_number(a, L, lambda_):
    """Compute the Fresnel number.

    Notes
    -----
    if the fresnel number is << 1, paraxial assumptions hold for propagation

    Parameters
    ----------
    a : float
        characteristic size ("radius") of an aperture
    L : float
        distance of observation
    lambda_ : float
        wavelength of light, same units as a

    Returns
    -------
    float
        the fresnel number for these parameters

    """
    return a**2 / (L * lambda_)


def talbot_distance(a, lambda_):
    """Compute the talbot distance.

    Parameters
    ----------
    a : float
        period of the grating, units of microns
    lambda_ : float
        wavelength of light, units of microns

    Returns
    -------
    float
        talbot distance, units of microns

    """
    num = lambda_
    den = 1 - np.sqrt(1 - lambda_**2/a**2)
    return num / den
