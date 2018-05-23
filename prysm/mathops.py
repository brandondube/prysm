"""A submodule which imports and exports math functions from different libraries.

The intend is to make the backend for prysm interoperable, allowing users to
utilize more high performance engines if they have them installed, or fall
back to more widely available options in the case that they do not.
"""

import numpy as np

from scipy.special import j1

from prysm.conf import config

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
    import cupy as cp
    assert cp
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
    'uint8',
    'uint16',
    'uint32',
    'float32',
    'float64',
    'complex64',
    'complex128',
    'pi',
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


def change_backend(to):
    if to.lower() == 'cu':
        if not cuda_compatible:
            raise ValueError('installation lacks cuda support.')
        else:
            target_base = 'cupy'
            target_fft = 'cupy.fft'
            target_linalg = 'cupy.linalg'
            # target_scipy = 'cupyx.scipy'

    elif to.lower() == 'np':
        target_base = 'numpy'
        target_fft = 'numpy.fft'
        target_linalg = 'numpy.linalg'
        # target_scipy = 'scipy'

    for func in allfuncs:
        exec(f'from {target_base} import {func}')
        globals()[func] = eval(func)

    for const in constants:
        exec(f'from {target_base} import {const}')
        globals()[const] = eval(const)

    for func in fftfuncs:
        exec(f'from {target_fft} import {func}')
        globals()[func] = eval(func)

    for func in linalgfuncs:
        exec(f'from {target_linalg} import {func}')
        globals()[func] = eval(func)


config.chbackend_observers.append(change_backend)
config.backend = config.backend  # trigger import of math functions
