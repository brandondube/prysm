"""Phase Shifting Interferometry."""
import numpy as truenp

from prysm.mathops import np
from prysm.fttools import fftrange
from prysm._richdata import RichData
from prysm.polynomials import sum_of_2d_modes

from skimage.restoration._unwrap_2d import unwrap_2d as sk_unwrap


FIVE_FRAME_PSI_NOMINAL_SHIFTS = (-np.pi, -np.pi/2, 0, +np.pi/2, +np.pi)
FOUR_FRAME_PSI_NOMINAL_SHIFTS = (0, np.pi/2, np.pi, 3/2*np.pi)

ZYGO_THIRTEEN_FRAME_SHIFTS = fftrange(13) * np.pi/4
ZYGO_THIRTEEN_FRAME_SS = (-3, -4, 0, 12, 21, 16, 0, -16, -21, -12, 0, 4, 3)
ZYGO_THIRTEEN_FRAME_CS = (0, -4, -12, -12, 0, 16, 24, 16, 0, -12, -12, -4, 0)

SCHWIDER_SHIFTS = fftrange(5) * np.pi/2
SCHWIDER_SS = (0, 2, 0, -2, 0)
SCHWIDER_CS = (-1, 0, 2, 0, -1)

# one-time array conversion for dtype lookups in psi acc
ZYGO_THIRTEEN_FRAME_SS = truenp.asarray(ZYGO_THIRTEEN_FRAME_SS)
ZYGO_THIRTEEN_FRAME_CS = truenp.asarray(ZYGO_THIRTEEN_FRAME_CS)
SCHWIDER_SS = truenp.asarray(SCHWIDER_SS)
SCHWIDER_CS = truenp.asarray(SCHWIDER_CS)


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

def psi_accumulate(gs, ss, cs):
    # ss = np.asarray(ss)
    # cs = np.asarray(cs)
    # dtype = _psi_acc_dtype(gs)
    # num = np.zeros(gs[0].shape, dtype=dtype)
    # den = np.zeros(gs[0].shape, dtype=dtype)
    num = sum_of_2d_modes(gs, ss)
    den = sum_of_2d_modes(gs, cs)
    return num, den


def degroot_formalism_psi(gs, ss, cs):
    """Peter de Groot's formalism for Phase Shifting Interferometry algorithms.

    Parameters
    ----------
    gs : iterable
        sequence of images
    ss : iterable
        sequence of numerator weights
    cs : iterable
        sequence of denominator weights

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

    Common/Sample formalisms,
    Schwider-Harihan five-frame algorithms, pi/4 steps
    s = (0, 2, 0, -2, 0)
    c = (-1, 0, 2, 0, -1)

    Zygo 13-frame algorithm, pi/4 steps
    s = (-3, -4, 0, 12, 21, 16, 0, -16, -21, -12, 0, 4, 3)
    c = (0, -4, -12, -12, 0, 16, 24, 16, 0, -12, -12, -4, 0)

    Zygo 15-frame algorithm, pi/2 steps
    s = (-1, 0, 9, 0, -21, 0, 29, 0, -29, 0, 21, 0, -9, 0, 1)
    c = (0, -4, 0, 15, 0, -26, 0, 30, 0, -26, 0, 15, 0, -4, 0)

    """
    was_rd = isinstance(gs[0], RichData)
    if was_rd:
        g00 = gs[0]
        gs = [g.data for g in gs]

    num, den = psi_accumulate(gs, ss, cs)
    out = np.arctan2(num, den)
    if was_rd:
        out = RichData(out, g00.dx, g00.wavelength)

    return out


def unwrap_phase(wrapped, mask):
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
