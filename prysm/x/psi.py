"""Phase Shifting Interferometry."""
from collections import namedtuple

import numpy as truenp

from scipy import signal

from prysm.mathops import np
from prysm.fttools import fftrange
from prysm._richdata import RichData
from prysm.polynomials import sum_of_2d_modes

from skimage.restoration._unwrap_2d import unwrap_2d as sk_unwrap

Scheme = namedtuple('Scheme', ['shifts', 's', 'c'])

ZYGO_THIRTEEN_FRAME = Scheme(
    fftrange(13) * np.pi/4,
    truenp.asarray((-3, -4, 0, 12, 21, 16, 0, -16, -21, -12, 0, 4, 3)),
    truenp.asarray((0, -4, -12, -12, 0, 16, 24, 16, 0, -12, -12, -4, 0)),
)

SCHWIDER = Scheme(
    fftrange(5) * np.pi/2,
    truenp.asarray((0, 2, 0, -2, 0)),
    truenp.asarray((-1, 0, 2, 0, -1)),
)


# def _psi_acc_dtype(gs, ss, cs):
#     # unsigned, or integer
#     kinds = (gs.dtype.kind, ss.dtype.kind, cs.dtype.kind)
#     is_int = all(kind in 'ui' for kind in kinds)
#     # fast path
#     if not is_int:
#         return config.precision

#     sz = gs.dtype.itemsize  # bytes per value
#     if sz < 32:
#         return np.int32
#     else:
#         return np.int64

def psi_accumulate(gs, scheme):
    """Accumulate the numerator and denominator for PSI reconstruction.

    The numerator is the sine of the complex wave, and the denominator the cosine.

    Parameters
    ----------
    gs : iterable
        sequence of images
    scheme : Scheme
        a PSI scheme, or any other object with .s and .c attributes, which are
        the sines and cosines of a sequence of phase shifts

    Returns
    -------
    ndarray, ndarray
        numerator (sine) and denominator (cosine)

    """
    # ss = np.asarray(ss)
    # cs = np.asarray(cs)
    # dtype = _psi_acc_dtype(gs)
    # num = np.zeros(gs[0].shape, dtype=dtype)
    # den = np.zeros(gs[0].shape, dtype=dtype)
    num = sum_of_2d_modes(gs, scheme.s)
    den = sum_of_2d_modes(gs, scheme.c)
    return num, den


def degroot_formalism_psi(gs, scheme):
    """Peter de Groot's formalism for Phase Shifting Interferometry algorithms.

    Parameters
    ----------
    gs : iterable
        sequence of images
    scheme : Scheme
        a PSI scheme, or any other object with .s and .c attributes, which are
        the sines and cosines of a sequence of phase shifts

    Returns
    -------
    ndarray
        wrapped phase estimate

    Notes
    -----
    Ref
    "Measurement of transparent plates with wavelength-tuned
    phase-shifting interferometry"

    Peter de Groot, Appl. Opt,  39, 2658-2663 (2000)
    https://doi.org/10.1364/AO.39.002658
    """
    was_rd = isinstance(gs[0], RichData)
    if was_rd:
        g00 = gs[0]
        gs = [g.data for g in gs]

    num, den = psi_accumulate(gs, scheme)
    out = np.arctan2(num, den)
    if was_rd:
        out = RichData(out, g00.dx, g00.wavelength)

    return out


def design_scheme(N, stepsize=None, window=None):
    """Design a new PSI scheme.

    A scheme is simply the cosine or sine of the phase shifts, each multiplied
    by a window whose job is to reduce spectral leakage and improve robustness
    to imperfect phase shifts and other data acquisition errors.  The window is
    often rounded or made non-ideal, in order for the s and c coefficients
    generated to be integers.  There is no inherent benefit to integer
    coefficients except that they are easier to write down and more sympathetic
    to an implementation of psi_accumulate() on an FPGA or microprocessor with
    reduced floating point performance.

    Parameters
    ----------
    N : int
        number of steps in the scheme; odd numbers are typically preferrable
    stepsize : float
        stepsize, in the sense of 2pi = a full phase shift.
        stepsize = 2pi/(N-1) if stepsize is not given
    window : None or ndarray or str
        if None, no window (equivalent to a window of ones) is used.
        if an ndarray, used as-is
        if a str, a member of scipy.signal.window; note that sym=False will be
        passed

    Returns
    -------
    Scheme
        a complete PSI scheme

    """
    if stepsize is None:
        stepsize = (2*truenp.pi)/(N-1)

    shifts = fftrange(N) * stepsize
    s = truenp.sin(shifts)
    c = truenp.cos(shifts)

    if window is not None:
        if isinstance(window, str):
            window = signal.windows.get_window(window, N)

        s *= window
        c *= window

    return Scheme(shifts, s, c)


def unwrap_phase(wrapped, mask):
    """Unwrap an array containing phase warpped at pi.

    Parameters
    ----------
    wrapped : ndarray
        2D array of phase data, wrapped
    mask : ndarray
        boolean mask array

    Returns
    -------
    ndarray
        unwrapped phase

    Notes
    -----
    currently just a wrapper for scikit-image

    """
    was_rd = isinstance(wrapped, RichData)
    if was_rd:
        w0 = wrapped
        wrapped = wrapped.data

    mask2 = np.zeros(wrapped.shape, dtype=np.uint8, order='C')
    if mask is not None:
        mask2[mask] = 255

    out = np.empty(wrapped.shape, dtype=np.float64, order='C')
    sk_unwrap(wrapped, mask2, out, wrap_around=[False, False], seed=None)
    if was_rd:
        out = RichData(out, w0.dx, w0.wavelength)

    return out
