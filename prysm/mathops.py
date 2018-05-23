"""A submodule which imports and exports math functions from different libraries.

The intend is to make the backend for prysm interoperable, allowing users to
utilize more high performance engines if they have them installed, or fall
back to more widely available options in the case that they do not.
"""

from math import (
    nan,
    pi,
    floor,
    ceil,
)
import numpy as np
from numpy import (
    sqrt,
    sin,
    cos,
    tan,
    arctan,
    arctan2,
    arccos,
    arcsin,
    sinc,
    radians,
    exp,
    log,
    log10,
    newaxis,
)
from numpy.fft import fftshift, ifftshift, fftfreq

from scipy.special import j1

atan2 = arctan2
atan = arctan

# numba funcs
try:
    from numba import jit, vectorize
except ImportError:
    # if Numba is not installed, create the jit decorator and have it return the
    # original function.
    def jit(signature_or_function=None, locals={}, target='cpu', cache=False, **options):
        """Passthrough duplicate of numba jit.

        Parameters
        ----------
        signature_or_function : None, optional
            a function or signature to compile
        locals : dict, optional
            local variables for the compiled code
        target : str, optional
            architecture to compile for
        cache : bool, optional
            whether to cache compilation to disk
        **options
            various options

        Returns
        -------
        callable
            a callable

        """
        if signature_or_function is None:
            def _jit(function):
                return function
            return _jit
        else:
            return signature_or_function

    vectorize = jit

# cuda
try:
    import cupy as cu
    cuda_compatible = True
except ImportError:
    cuda_compatible = False

allfuncs = frozenset((
    'sqrt',
    'sin',
    'cos',
    'tan',
    'arctan',
    'arctan2',
    'arccos',
    'arcsin',
    'sinc',
    'radians',
    'exp',
    'log',
    'log10',
    'linspace',
    'meshgrid',
    'angle',
    'zeros',
    'ones',
    'empty',
    'sign',
    'isfinite',
    'asarray',
    'arange',
    'stack',
    'mean',
    'unique',
    'swapaxes',
    'rollaxis',
    'searchsorted',
    'concatenate',
    'cumsum',
    'gradient',
    'any',
    'isfinite',
))

fftfuncs = frozenset((
    'fft2',
    'ifft2',
    'fftshift',
    'ifftshift',
    'fftfreq',
))

linalgfuncs = frozenset((
    'lstsq',
))

constants = frozenset((
    'ndarray',
    'float32',
    'float64',
    'complex64',
    'complex128',
))

def jinc(r):
    """Jinc.

    Parameters
    ----------
    r : `number`
        radial distance

    Returns
    -------
    `float`
        the value of j1(x)/x for x != 0, 0.5 at 0

    """
    if r == 0:
        return 0.5
    else:
        return j1(r) / r


jinc = np.vectorize(jinc)

# export control
# thanks, ITAR


fft2, ifft2 = np.fft.fft2, np.fft.ifft2

ndarray = np.ndarray

# silence pyflakes
assert [nan, pi, sqrt, sin, cos, tan, arccos, arcsin, sinc, radians, exp, log, log10]
assert [fftshift, ifftshift, fft2, ifft2, fftfreq]
assert [floor, ceil]


def change_backend(to):
    if to.lower() == 'cu':
        if not cuda_compatible:
            raise ValueError('installation lacks cuda support.')
        else:
            for func in allfuncs:
                exec(f'from cupy import {func}')
                globals()[func] = eval(func)

            for func in fftfuncs:
                exec(f'from cupy.fft import {func}')
                globals()[func] = eval(func)

    elif to.lower() == 'np':
        for func in allfuncs:
            exec(f'from numpy import {func}')
            globals()[func] = eval(func)

        for func in fftfuncs:
            exec(f'from numpy.fft import {func}')
            globals()[func] = eval(func)
