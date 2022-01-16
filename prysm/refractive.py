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
