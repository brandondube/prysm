"""A submodule which allows the user to swap out the backend for mathematics."""
from numbers import Number
import warnings

import numpy as np
from scipy import ndimage, interpolate, fft, optimize, signal
from scipy.special import j1 as _besselj1


class BackendShim:
    """A shim that allows a backend to be swapped at runtime."""
    def __init__(self, src):
        self._srcmodule = src

    def __getattr__(self, key):
        if key == '_srcmodule':
            return self._srcmodule

        return getattr(self._srcmodule, key)


_np = np
_scalar_types = (Number, _np.generic)
_ndimage = ndimage
_fft = fft
_interpolate = interpolate
_optimize = optimize
_signal = signal
np = BackendShim(np)
ndimage = BackendShim(ndimage)
fft = BackendShim(fft)
interpolate = BackendShim(interpolate)
optimize = BackendShim(optimize)
signal = BackendShim(signal)


def set_backend_to_cupy():
    """Convenience method to automatically configure prysm's backend to cupy."""
    import cupy as cp
    from cupyx.scipy import (
        fft as cpfft,
        ndimage as cpndimage,
        interpolate as cpinterpolate,
    )

    np._srcmodule = cp
    fft._srcmodule = cpfft
    ndimage._srcmodule = cpndimage
    interpolate._srcmodule = cpinterpolate
    # cupyx.scipy.signal exists but cupyx.scipy.optimize generally does not;
    # opportunistically remap signal if present, leave optimize on scipy.
    try:
        from cupyx.scipy import signal as cpsignal
        signal._srcmodule = cpsignal
    except ImportError:
        pass
    return


def set_backend_to_defaults():
    """Convenience method to restore prysm's default backend options."""
    np._srcmodule = _np
    fft._srcmodule = _fft
    ndimage._srcmodule = _ndimage
    interpolate._srcmodule = _interpolate
    optimize._srcmodule = _optimize
    signal._srcmodule = _signal
    return


def set_backend_to_pytorch():
    """Convenience method to automatically configure prysm's backend to PyTorch."""
    import pytorch as torch

    np._srcmodule = torch
    fft._srcmodule = torch.fft
    warnings.warn('set_backend_to_pytorch: only np and fft remapped; ndimage, interpolate, optimize, and signal do not have known torch equivalents.')
    return


def set_fft_backend_to_mkl_fft():
    """Convenience method to automatically configure prysm's backend to MKL_FFT for FFTs."""
    from mkl_fft import _numpy_fft as mklfft

    fft._srcmodule = mklfft
    return


def array_to_true_numpy(*args):
    """Convert one or more arrays from an alternate backend to numpy.

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
        if isinstance(arg, _scalar_types):
            out.append(arg)
            continue

        if isinstance(arg, _np.ndarray):
            out.append(arg)
            continue

        # cupy
        if hasattr(arg, 'get'):
            out.append(arg.get())
            continue

        # PyTorch
        if hasattr(arg, 'numpy'):
            out.append(arg.numpy(force=True))
            continue

    if len(out) == 1:
        return out[0]

    return out


def row_dot(a, b):
    """Batched dot product along the last (row) axis: ``sum(a * b, axis=-1)``.

    For inputs of shape ``(N, K)`` returns shape ``(N,)``.  Implementation
    uses ``einsum`` because it is the fastest route on numpy and works
    unchanged on the cupy / torch backends.  See the prior incarnation in
    ``prysm/x/raytracing/spencer_and_murty.py`` (``_multi_dot``) for
    benchmark notes — this task is memory-bandwidth limited.

    Parameters
    ----------
    a : ndarray
        shape (N, K)
    b : ndarray
        shape (N, K)

    Returns
    -------
    ndarray
        shape (N,)

    """
    # Implementation will change over time to track the fastest way to do this
    # with numpy. (maybe)
    #
    # There is no BLAS level 1/2/3 function for a batch of dot products.
    #
    # For a (1024*1024)*3, aka 1 million dot batch, the fastest function below
    # takes 4.23 ms on a dual channel laptop (~40GB/s bandwidth from RAM).
    #
    # The dot product is simply sum += a[i]*b[i], which touches three values for
    # each element of the input array, i.e.
    # sum = 0
    # sum += a[0] + b[0]
    # sum += a[1] + b[1]
    # sum += a[2] + b[2]
    #
    # It also performs one flop (floating-point operation) per element.
    #
    # with 6,291,456 elements and eight bytes per element, this is 50,331,648 bytes
    # of computation in 4.23 ms, 11,898,734,751 (about 12GB/sec)
    #
    # So, this can be made faster by using a few threads, but those threads must not
    # perform extra copies, since we are near the memory bandwidth limit of the system
    #
    # But, in a gist
    # https://gist.github.com/brandondube/43ab9e9f173252f5a97e0c0d5c3ca54f
    #
    # with batch size 1 million (einsum is not the fastest below)
    # no parallelism - 7.39 ms ± 424 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
    # thread pool 1  - 8.8 ms ± 259 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
    # thread pool 2  - 5.23 ms ± 68.5 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
    # thread pool 3  - 5.77 ms ± 177 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
    #
    # best case 30% speedup - not worth it
    #
    # rule of thumb, intel CPU can start two floating point operations per clock per core
    # ~= 4 billion clocks per second ~= 8 billion flops/sec/core
    #
    # we need 6.3M flops for our example batch size, and the calc took 4.3 ms, so
    # 1,463,129,302 flops/sec were used (~= 25% of one CPU core)
    #
    # so, the task was memory bandwidth limited, and we go faster with multiple
    # threads only because of some quirk of intel's memory controller and prefetch
    # semantics.  But >10x faster is not living in reality.  Maybe on a system
    # with vast memory bandwidth (say, 4 socket xeon -- 800GB/sec).  But that's
    # $40k of CPUs and one A40 GPU does that for $5k, so why bother.
    return np.einsum('ij,ij->i', a, b)


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
            return _besselj1(r) / r
    else:
        mask = (r < 1e-8) & (r > -1e-8)
        out = _besselj1(r) / r
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
