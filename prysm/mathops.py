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

allfuncs = set((
    'sort',
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
    'flip',
    'swapaxes',
    'rollaxis',
    'concatenate',
    'cumsum',
    'any',
    'all',
    'isfinite',
    'isnan',
    'ceil',
    'floor',
    'outer',
    'inner',
    'argmin',
    'argmax',
    'allclose',
    'frombuffer',
    'count_nonzero',
    'trapz',
    'hanning',
    'full',
))

allfuncs_cupy_missing = set((
    'searchsorted',
    'gradient',
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
    'int32',
    'int64',
    'float32',
    'float64',
    'complex64',
    'complex128',
    'newaxis',
    'pi',
    'nan',
    'inf',
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
    if r < 1e-8 and r > -1e-8:  # value of jinc for x < 1/2 machine precision  is 0.5
        return 0.5
    else:
        return j1(r) / r


if numba_installed is True:
    # one day split numba jit and numpy jit
    jinc = np.vectorize(jinc)
else:
    jinc = np.vectorize(jinc)


def change_backend(to):
    if to == 'cu':
        if not cuda_compatible:
            raise ValueError('installation lacks cuda support.')
        else:
            target_base = 'cupy'
            target_fft = 'cupy.fft'
            target_linalg = 'cupy.linalg'
            # target_scipy = 'cupyx.scipy'

    elif to == 'np':
        target_base = 'numpy'
        target_fft = 'numpy.fft'
        target_linalg = 'numpy.linalg'
        # target_scipy = 'scipy'

        # two sets of functionality unavailable via cupy
        for func in allfuncs_cupy_missing:
            exec(f'from {target_base} import {func}')
            globals()[func] = eval(func)

        for func in linalgfuncs:
            exec(f'from {target_linalg} import {func}')
            globals()[func] = eval(func)

    for func in allfuncs:
        exec(f'from {target_base} import {func}')
        globals()[func] = eval(func)

    for const in constants:
        exec(f'from {target_base} import {const}')
        globals()[const] = eval(const)

    for func in fftfuncs:
        exec(f'from {target_fft} import {func}')
        globals()[func] = eval(func)


config.chbackend_observers.append(change_backend)
config.backend = config.backend  # trigger import of math functions
