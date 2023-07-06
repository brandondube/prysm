"""Degredations in the image chain."""

from .mathops import np
from .conf import config
from .coordinates import cart_to_polar, polar_to_cart


def smear_ft(fx, fy, width, height):
    """Analytic Fourier Transform (OTF) of smear.

    Parameters
    ----------
    fx : numpy.ndarray
        X spatial frequencies, units of reciprocal width
    fy : numpy.ndarray
        Y spatial frequencies, units of reciprocal width
    width : float
        width of the smear, units of length (e.g. um)
    height : float
        height of the smear, units of length (e.g. um)

    Returns
    -------
    numpy.ndarray
        transfer function of the smear

    """
    assert width != 0 or height != 0, 'one of width or height must be nonzero'
    if width != 0:
        out1 = np.sinc(fx * width).astype(config.precision)
    else:
        out1 = 1

    if height != 0:
        out2 = np.sinc(fy * height).astype(config.precision)
    else:
        out2 = 1

    return out1*out2


def jitter_ft(fr, scale):
    """Analytic Fourier transform (OTF) of jitter.

    Parameters
    ----------
    fr : numpy.ndarray
        radial spatial frequency, units of reciprocal length, e.g. cy/mm
    scale : float
        scale of the jitter, in same units as "dx"
        e.g., if fr has units cy/mm, then scale has units mm

    Returns
    -------
    numpy.ndarray
        transfer function of the jitter

    """
    core = (np.pi*scale*fr)
    out = np.exp(-2 * (core*core))
    return out
