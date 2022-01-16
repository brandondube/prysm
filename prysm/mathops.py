"""A submodule which allows the user to swap out the backend for mathematics."""
import numpy as np
from scipy import ndimage, interpolate, special, fft


class BackendShim:
    """A shim that allows a backend to be swapped at runtime."""
    def __init__(self, src):
        self._srcmodule = src

    def __getattr__(self, key):
        if key == '_srcmodule':
            return self._srcmodule

        return getattr(self._srcmodule, key)

np = BackendShim(np)
ndimage = BackendShim(ndimage)
special = BackendShim(special)
fft = BackendShim(fft)
interpolate = BackendShim(interpolate)


def jinc(r):
    """Jinc.

    The first zero of jinc occurs at r=pi

    Parameters
    ----------
    r : number
        radial distance

    Returns
    -------
    float
        the value of j1(x)/x for x != 0, 0.5 at 0

    """
    if not hasattr(r, '__iter__'):
        # scalar case
        if r < 1e-8 and r > -1e-8:  # value of jinc for x < 1/2 machine precision  is 0.5
            return 0.5
        else:
            return special.j1(r) / r
    else:
        mask = (r < 1e-8) & (r > -1e-8)
        out = special.j1(r) / r
        out[mask] = 0.5
        return out


def is_odd(int):
    """Determine if an interger is odd using binary operations.

    Parameters
    ----------
    int : int
        an integer

    Returns
    -------
    bool
        true if odd, False if even

    """
    return int & 0x1


def is_power_of_2(value):
    """Check if a value is a power of 2 using binary operations.

    Parameters
    ----------
    value : number
        value to check

    Returns
    -------
    bool
        true if the value is a power of two, False if the value is no

    Notes
    -----
    c++ inspired implementation, see SO:
    https://stackoverflow.com/questions/29480680/finding-if-a-number-is-a-power-of-2-using-recursion

    """
    if value == 1:
        return False
    else:
        return bool(value and not value & (value - 1))


def sign(x):
    """Sign of a number.  Note only works for single values, not arrays."""
    return -1 if x < 0 else 1


def kronecker(i, j):
    """Kronecker delta function, 1 if i = j, otherwise 0."""
    return 1 if i == j else 0


def gamma(n, m):
    """Gamma function."""
    if n == 1 and m == 2:
        return 3 / 8
    elif n == 1 and m > 2:
        mm1 = m - 1
        numerator = 2 * mm1 + 1
        denominator = 2 * (mm1 - 1)
        coef = numerator / denominator
        return coef * gamma(1, mm1)
    else:
        nm1 = n - 1
        num = (nm1 + 1) * (2 * m + 2 * nm1 - 1)
        den = (m + nm1 - 2) * (2 * nm1 + 1)
        coef = num / den
        return coef * gamma(nm1, m)
