"""A submodule which imports and exports math functions from different libraries.

The intend is to make the backend for prysm interoperable, allowing users to
utilize more high performance engines if they have them installed, or fall
back to more widely available options in the case that they do not.
"""

from math import (
    nan,
    pi,
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
    floor,
    ceil,
)
from numpy.fft import fftshift, ifftshift, fftfreq

atan2 = arctan2
atan = arctan

# numba funcs, cuda
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


# export control
# thanks, ITAR


fft2, ifft2 = np.fft.fft2, np.fft.ifft2

# silence pyflakes
assert [nan, pi, sqrt, sin, cos, tan, arccos, arcsin, sinc, radians, exp, log, log10]
assert [fftshift, ifftshift, fft2, ifft2, fftfreq]
assert [floor, ceil]
