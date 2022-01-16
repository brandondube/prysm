"""Degredations in the image chain."""

from .mathops import np
from .coordinates import cart_to_polar, polar_to_cart


def smear_ft(fx, fy, width, angle):
    """Analytic Fourier Transform (OTF) of smear.

    Parameters
    ----------
    fx : numpy.ndarray
        X spatial frequencies, units of reciprocal width
    fy : numpy.ndarray
        Y spatial frequencies, units of reciprocal width
    width : float
        width of the smear, units of length (e.g. um)
    angle : float
        angle w.r.t the X axis of the smear, degrees

    Returns
    -------
    numpy.ndarray
        transfer function of the smear

    """
    # TODO: faster to do inline projection of fx, fy?
    if angle != 0:
        rho, phi = cart_to_polar(fx, fy)
        phi += np.radians(angle)
        x, y = polar_to_cart(rho, phi)

    return np.sinc(x * width)


def jitter_ft(fr, scale):
    """Analytic Fourier transform (OTF) of jitter.

    Parameters
    ----------
    fr : numpy.ndarray
        radial spatial frequency, units of reciprocal scale
    scale : float
        scale of the jitter

    Returns
    -------
    numpy.ndarray
        transfer function of the jitter

    """
    kernel = np.pi * scale / 2 * fr
    return np.exp(-2 * kernel**2)
