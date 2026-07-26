"""Degradations in the image chain."""

from .mathops import np
from .conf import config


def smear_ft(fx, fy, width, height):
    """Compute the analytic Fourier transform of smear.

    Parameters
    ----------
    fx : ndarray
        X spatial frequencies.
    fy : ndarray
        Y spatial frequencies.
    width : float
        Horizontal smear extent.
    height : float
        Vertical smear extent.

    Returns
    -------
    ndarray
        Smear transfer function.

    """
    if width == 0 and height == 0:
        raise ValueError('one of width or height must be nonzero')
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
    """Compute the analytic Fourier transform of Gaussian jitter.

    Parameters
    ----------
    fr : ndarray
        Radial spatial frequency.
    scale : float
        Jitter standard deviation.

    Returns
    -------
    ndarray
        Jitter transfer function.

    """
    core = np.pi*scale*fr
    return np.exp(-2 * core*core)


__all__ = ['jitter_ft', 'smear_ft']
