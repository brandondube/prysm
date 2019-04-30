"""A submodule which imports and exports math functions from different libraries.

The intend is to make the backend for prysm interoperable, allowing users to
utilize more high performance engines if they have them installed, or fall
back to more widely available options in the case that they do not.
"""
import numpy as np

from scipy.special import j1, j0  # NOQA

from prysm.conf import config

# numba funcs
try:
    from numba import jit, vectorize
    numba_installed = True
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
    numba_installed = False

# cuda
try:
    import cupy as cp
    from cupy import fuse
    assert cp
    cuda_compatible = True
except ImportError:
    cuda_compatible = False

    def fuse(*args, **kwargs):
        if 'func' in kwargs:
            return kwargs['func']
        else:
            def wrapper(function):
                return function

            return wrapper


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
    if r < 1e-8 and r > -1e-8:  # value of jinc for x < 1/2 machine precision  is 0.5
        return 0.5
    else:
        return j1(r) / r


if numba_installed is True:
    # one day split numba jit and numpy jit
    jinc = np.vectorize(jinc)
else:
    jinc = np.vectorize(jinc)


class MathEngine:
    """An engine allowing an interchangeable backend for mathematical functions."""
    def __init__(self, source=np):
        """Create a new math engine.

        Parameters
        ----------
        source : `module`
            a python module.

        """
        self.source = source

    def __getattr__(self, key):
        """Get attribute.

        Parameters
        ----------
        key : `str` attribute name

        """
        try:
            return getattr(self.source, key)
        except AttributeError:
            # function not found, fall back to numpy
            # this will actually work nicely for numpy 1.16+
            # due to the __array_function__ and __array_ufunc__ interfaces
            # that were implemented
            return getattr(self.source, key)  # this can raise, but we don't *need* to catch

    def change_backend(self, backend):
        """Function to run when changing the backend."""
        self.source = backend


engine = MathEngine()
config.chbackend_observers.append(engine.change_backend)
