"""A submodule which allows the user to swap out the backend for mathematics."""
import warnings

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


_np = np
_ndimage = ndimage
_special = special
_fft = fft
_interpolate = interpolate
np = BackendShim(np)
ndimage = BackendShim(ndimage)
special = BackendShim(special)
fft = BackendShim(fft)
interpolate = BackendShim(interpolate)


def set_backend_to_cupy():
    """Convenience method to automatically configure prysm's backend to cupy."""
    import cupy as cp
    from cupyx.scipy import (
        fft as cpfft,
        ndimage as cpndimage,
        special as cpspecial,
        interpolate as cpinterpolate,
    )

    np._srcmodule = cp
    fft._srcmodule = cpfft
    ndimage._srcmodule = cpndimage
    special._srcmodule = cpspecial
    interpolate._srcmodule = cpinterpolate
    return


def set_backend_to_defaults():
    """Convenience method to restore prysm's default backend options."""
    np._srcmodule = _np
    fft._srcmodule = _fft
    ndimage._srcmodule = _ndimage
    special._srcmodule = _special
    interpolate._srcmodule = interpolate
    return


def set_backend_to_pytorch():
    """Convenience method to automatically configure prysm's backend to PyTorch."""
    import pytorch as torch

    np._srcmodule = torch
    fft._srcmodule = torch.fft
    special._srcmodule = torch.special
    warnings.warn('set_backend_to_pytorch: only np, fft, special remapped; ndimage, interpolate do not have known torch equivalents.')
    return


def set_fft_backend_to_mkl_fft():
    from mkl_fft import _numpy_fft as mklfft

    fft._srcmodule = mklfft
    return


def array_to_true_numpy(*args):
    """convert one or more arrays from an alternate backend to numpy.

    Needed for plotting, serialization, etc.

    Does nothing if given an actual numpy array

    Parameters
    ----------
    args : any number of arrays, of any dimension and dtype

    Returns
    -------
    array, or list of bonefide numpy arrays

    """
    if len(args) == 0:
        return

    out = []
    for arg in args:
        if isinstance(arg, _np.ndarray):
            out.append(arg)
        # cupy
        if hasattr(arg, 'get'):
            out.append(arg.get())

        # PyTorch
        if hasattr(arg, 'numpy'):
            out.append(arg.numpy(force=True))

    if len(out) == 1:
        return out[0]

    return out


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
