"""Phase Shifting Interferometry."""

from prysm.mathops import np
from prysm._richdata import RichData
from prysm.fttools import fftrange

from skimage.restoration import unwrap_phase as ski_unwrap_phase


FIVE_FRAME_PSI_NOMINAL_SHIFTS = (-np.pi, -np.pi/2, 0, +np.pi/2, +np.pi)
FOUR_FRAME_PSI_NOMINAL_SHIFTS = (0, np.pi/2, np.pi, 3/2*np.pi)

ZYGO_THIRTEEN_FRAME_SHIFTS = fftrange(13) * np.pi/4
ZYGO_THIRTEEN_FRAME_SS = (-3, -4, 0, 12, 21, 16, 0, -16, -21, -12, 0, 4, 3)
ZYGO_THIRTEEN_FRAME_CS = (0, -4, -12, -12, 0, 16, 24, 16, 0, -12, -12, -4, 0)

SCHWIDER_SHIFTS = fftrange(5) * np.pi/2
SCHWIDER_SS = (0, 2, 0, -2, 0)
SCHWIDER_CS = (-1, 0, 2, 0, -1)


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

    num = \sum {s_m * g_m}
    den = \sum {c_m * g_m}
    theta = arctan(num/dem)

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

    num = np.zeros_like(gs[0])
    den = np.zeros_like(gs[0])
    for gm, sm, cm in zip(gs, ss, cs):
        # PSI algorithms tend to be sparse;
        # optimize against zeros
        if sm != 0:
            num += sm * gm
        if cm != 0:
            den += cm * gm

    out = np.arctan2(num, den)
    if was_rd:
        out = RichData(out, g00.dx, g00.wavelength)

    return out


def unwrap_phase(wrapped):
    was_rd = isinstance(wrapped, RichData)
    if was_rd:
        w0 = wrapped
        wrapped = wrapped.data

    out = ski_unwrap_phase(wrapped)
    if was_rd:
        out = RichData(out, w0.dx, w0.wavelength)

    return out
