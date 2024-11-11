"""Code for working with refractive index data."""
from .mathops import np


def cauchy(wvl, A, *args):
    """Cauchy's equation for the (real) index of refraction of transparent materials.

    Parameters
    ----------
    wvl : number
        wavelength of light, microns
    A : number
        the first term in Cauchy's equation
    args : number
        B, C, ... terms in Cauchy's equation

    Returns
    -------
    numpy.ndarray
        array of refractive indices of the same shape as wvl

    """
    seed = A

    for idx, arg in enumerate(args):
        # compute the power from the index, want to map:
        # 0 -> 2
        # 1 -> 4
        # 2 -> 6
        # ...
        power = 2*idx + 2
        seed = seed + arg / wvl ** power

    return seed


def sellmeier(wvl, A, B):
    """Sellmeier glass equation.

    Parameters
    ----------
    wvl : numpy.ndarray
        wavelengths, microns
    A : Iterable
        sequence of "A" coefficients
    B : Iterable
        sequence of "B" coefficients

    Returns
    -------
    numpy.ndarray
        refractive index

    """
    wvlsq = wvl ** 2
    seed = np.ones_like(wvl)
    for a, b, in zip(A, B):
        num = a * wvlsq
        den = wvlsq - b
        seed += (num/den)

    return np.sqrt(seed)


def internal_transmission(t, k, wvl):
    """Internal transmission of a glass slab.

    Parameters
    ----------
    t : ndarray
        thickness of the plate, millimeters
    k : ndarray
        the complex part of the refractive index, k, in the expression  n + ik
    wvl : ndarray
        wavelength of light, microns

    Returns
    -------
    complex transmission T

    """
    # convert wavelength to millimeters
    wvl = wvl / 1e3
    return np.exp(-4*np.pi*k*t/wvl)
