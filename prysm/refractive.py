"""Code for working with refractive index data."""
from .mathops import np


def cauchy(wvl, A, *args):
    """Cauchy's equation for the real index of refraction of transparent materials."""
    seed = A

    for idx, arg in enumerate(args):
        power = 2*idx + 2
        seed = seed + arg / np.power(wvl, power)

    return seed


def sellmeier(wvl, A, B):
    """Sellmeier glass equation."""
    wvlsq = np.square(wvl)
    seed = wvlsq * 0 + 1.0
    for a, b, in zip(A, B):
        num = a * wvlsq
        den = wvlsq - b
        seed += (num/den)

    return np.sqrt(seed)


def internal_transmission(t, k, wvl):
    """Internal transmission of a glass slab."""
    wvl = wvl / 1e3
    return np.exp(-4*np.pi*k*t/wvl)
