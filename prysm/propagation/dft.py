"""Matrix-DFT / chirp-Z based pupil <-> focal propagation with arbitrary sampling.
"""
from collections.abc import Iterable

from ..mathops import np
from ..conf import config
from ..fttools import fftrange, MDFT, CZT


def coordinates_for_focus(pupil_dx, pupil_samples, focal_dx, focal_samples,
                          wavelength, efl, focal_shift=(0, 0)):
    """Coordinate / frequency vectors for an MDFT-based pupil ↔ focal propagation.

    The Fraunhofer kernel is exp(-2πi · x_pupil · x_focal / (λ · efl)). This
    returns the input pupil coordinates (x, y) and the spatial frequencies
    (fx, fy) that pair with them, where fx = x_focal / (λ · efl).

    For end users, prefer prepare_executor, which wraps this and bakes
    the optical normalization into the executor. If you do build the executor
    by hand, multiply its result by pupil_dx * focal_dx / (wavelength * efl).

    Parameters
    ----------
    pupil_dx : float
        pupil-plane sample spacing, mm
    pupil_samples : int or (int, int)
        pupil samples; a single int is treated as square, a tuple as (rows, cols)
    focal_dx : float
        focal-plane sample spacing, microns
    focal_samples : int or (int, int)
        focal samples; a single int is treated as square, a tuple as (rows, cols)
    wavelength : float
        wavelength of light, microns
    efl : float
        effective focal length, mm
    focal_shift : (float, float)
        (x, y) translation of the focal grid center, microns

    Returns
    -------
    x, y : ndarray
        pupil coordinates along the column and row axes, mm
    fx, fy : ndarray
        spatial frequencies along the column and row axes, 1/mm

    """
    if not isinstance(pupil_samples, Iterable):
        pupil_samples = (pupil_samples, pupil_samples)
    if not isinstance(focal_samples, Iterable):
        focal_samples = (focal_samples, focal_samples)

    pny, pnx = pupil_samples
    fny, fnx = focal_samples
    fsx, fsy = focal_shift

    dtype = config.precision
    x = fftrange(pnx, dtype=dtype) * pupil_dx
    y = fftrange(pny, dtype=dtype) * pupil_dx
    # focal positions in microns, then convert to spatial frequency 1/mm:
    # fx = x_focal_mm / (lambda_mm * efl) = x_focal_um / (wavelength_um * efl_mm)
    inv_lz = 1.0 / (wavelength * efl)
    fx = (fftrange(fnx, dtype=dtype) * focal_dx + fsx) * inv_lz
    fy = (fftrange(fny, dtype=dtype) * focal_dx + fsy) * inv_lz
    return x, y, fx, fy


def prepare_executor(pupil_dx, pupil_samples, focal_dx, focal_samples,
                     wavelength, efl, focal_shift=(0, 0), kind='mdft'):
    """Build a reusable MDFT or CZT operator for a pupil ↔ focal propagation.

    Wraps coordinates_for_focus and the executor constructor in one
    call. The optical normalization scalar
    pupil_dx * focal_dx / (wavelength * efl) is baked into the executor's
    norm, so applying the executor produces a unitary-equivalent
    propagated field. The returned operator is in the focus orientation:

    - Focus:    executor(pupil_data) produces focal data
    - Unfocus:  executor.adjoint(focal_data) produces pupil data

    The pupil and focal sample spacings are also stashed on the returned
    operator as executor.pupil_dx and executor.focal_dx for callers
    that need them (e.g. to label an output Wavefront).

    Parameters
    ----------
    pupil_dx, pupil_samples, focal_dx, focal_samples, wavelength, efl, focal_shift
        See coordinates_for_focus.
    kind : {'mdft', 'czt'}, optional
        Executor type to build. Default 'mdft'.

    Returns
    -------
    MDFT or CZT
        operator suitable for passing to focus_dft, unfocus_dft, etc.

    """
    x, y, fx, fy = coordinates_for_focus(
        pupil_dx, pupil_samples, focal_dx, focal_samples,
        wavelength, efl, focal_shift,
    )
    norm = (pupil_dx * focal_dx) / (wavelength * efl)
    if kind == 'mdft':
        op = MDFT(x, y, fx, fy, sign=-1, norm=norm)
    elif kind == 'czt':
        op = CZT(x, y, fx, fy, sign=-1, norm=norm)
    else:
        raise ValueError(f"kind must be 'mdft' or 'czt', got {kind!r}")
    op.pupil_dx = pupil_dx
    op.focal_dx = focal_dx
    return op


def _smootherstep(t):
    """C2 smoothstep 6t^5 - 15t^4 + 10t^3, clipped to [0, 1]; 0 at t<=0, 1 at t>=1."""
    t = np.clip(t, 0, 1)
    return t * t * t * (t * (t * 6 - 15) + 10)


def _cumulative_window(r, a, b):
    """Radial taper that is 1 for r < a and 0 for r > b, with a C2 transition.

    Used to assemble the partition-of-unity hand-off windows of a
    multi-resolution focal-plane propagation.
    """
    return 1 - _smootherstep((r - a) / (b - a))


class MultiResolutionExecutor:
    """A stack of arbitrary-sampling executors plus partition-of-unity windows.

    Each level forward-propagates the pupil to a focal grid of progressively
    finer sampling and smaller field of view, so the singular core of a focal
    plane mask (e.g. a vortex phase ramp) is sampled densely while the coarsest
    level still spans the full field of view and captures every spatial
    frequency. The per-level windows form a partition of unity over the focal
    plane; summing each level's masked, inverse-propagated contribution
    reconstructs the full diffraction integral with the singular region
    integrated at the finest resolution.

    Build instances with prepare_multiresolution. Hand them to
    to_fpm_and_back_multiresolution along with a focal-plane-mask callable.

    Attributes
    ----------
    executors : list of MDFT or CZT
        per-level pupil to focal operators, coarsest first
    windows : list of ndarray
        per-level real partition-of-unity windows, summing to one over the
        focal plane
    xf, yf : list of ndarray
        per-level focal-plane coordinate meshgrids, microns; pass these to a
        focal-plane-mask callable to evaluate the mask on each level's grid

    """

    __slots__ = ('executors', 'windows', 'xf', 'yf')

    def __init__(self, executors, windows, xf, yf):
        self.executors = executors
        self.windows = windows
        self.xf = xf
        self.yf = yf

    def __len__(self):
        return len(self.executors)


def prepare_multiresolution(pupil_dx, pupil_samples, focal_dx, focal_samples,
                            wavelength, efl, num_levels, scaling=4.0,
                            fine_samples=None, window=(0.2, 0.7), kind='mdft'):
    """Build a MultiResolutionExecutor for focal-plane-mask propagation.

    The coarsest level is specified exactly as for prepare_executor and should
    span the full field of view (focal_dx * focal_samples large enough to reach
    the edge of the propagated field) at or above Nyquist, so no spatial
    frequencies are truncated. Each finer level divides the sample spacing and
    the field of view by scaling, zooming into the singular core of the mask.

    Parameters
    ----------
    pupil_dx, pupil_samples, focal_dx, focal_samples, wavelength, efl
        coarsest-level geometry; see coordinates_for_focus. focal_dx and
        focal_samples describe level 0 only.
    num_levels : int
        number of resolution levels. One level reduces to an ordinary
        single executor (no hand-off windows).
    scaling : float, optional
        ratio of consecutive levels' sample spacings and fields of view.
        Default 4.
    fine_samples : int, optional
        focal_samples for every level past the coarsest. Their field of view
        shrinks with scaling, so fewer samples than the coarsest level still
        oversample. Defaults to focal_samples.
    window : (float, float), optional
        inner and outer radii of the hand-off transition, as fractions of each
        level's focal-plane half-width. The transition tapers from one to zero
        across this annulus. Default (0.2, 0.7).
    kind : {'mdft', 'czt'}, optional
        executor type. Default 'mdft'.

    Returns
    -------
    MultiResolutionExecutor

    """
    if fine_samples is None:
        fine_samples = focal_samples
    inner, outer = window

    executors = []
    xfs = []
    yfs = []
    radii = []
    halves = []
    for k in range(num_levels):
        nf = focal_samples if k == 0 else fine_samples
        fdx = focal_dx / scaling**k
        shift = fdx / 2.0  # half-pixel: keep the singular origin off-grid
        ex = prepare_executor(pupil_dx, pupil_samples, fdx, nf,
                              wavelength, efl, focal_shift=(shift, shift), kind=kind)
        line = fftrange(nf, dtype=config.precision) * fdx + shift
        xf, yf = np.meshgrid(line, line)
        executors.append(ex)
        xfs.append(xf)
        yfs.append(yf)
        radii.append(np.hypot(xf, yf))
        halves.append(nf / 2.0 * fdx)

    # each level owns the annulus between its own hand-off (to the finer level
    # inside it) and the coarser level's hand-off (outside it).  The coarsest
    # level extends outward forever (here = 1) and the finest reaches the origin
    # (nxt = 0); the per-level contributions telescope to a partition of unity.
    windows = []
    for k in range(num_levels):
        r = radii[k]
        here = 1.0 if k == 0 else _cumulative_window(r, inner * halves[k], outer * halves[k])
        nxt = 0.0 if k == num_levels - 1 else _cumulative_window(r, inner * halves[k + 1], outer * halves[k + 1])
        windows.append(here - nxt)

    return MultiResolutionExecutor(executors, windows, xfs, yfs)


def focus_dft(wavefunction, executor):
    """Propagate a pupil field to the PSF plane via a precomputed executor.

    Parameters
    ----------
    wavefunction : ndarray
        the pupil-plane field; shape must match what the executor was built for.
    executor : MDFT or CZT
        (semi-)arbitrary sampling fourier transform executor

    Returns
    -------
    ndarray
        focal-plane field

    """
    return executor(wavefunction)


def focus_dft_adjoint(wavefunction, executor):
    """Apply the adjoint of focus_dft.

    Parameters
    ----------
    wavefunction : ndarray
        gradient at the PSF plane
    executor : MDFT or CZT
        (semi-)arbitrary sampling fourier transform executor

    Returns
    -------
    ndarray
        gradient at the pupil plane

    """
    return executor.adjoint(wavefunction)


def unfocus_dft(wavefunction, executor):
    """Propagate an image-plane field to the pupil via a precomputed executor.

    Parameters
    ----------
    wavefunction : ndarray
        the focal-plane field
    executor : MDFT or CZT
        (semi-)arbitrary sampling fourier transform executor

    Returns
    -------
    ndarray
        pupil-plane field

    """
    return executor.adjoint(wavefunction)


def unfocus_dft_adjoint(wavefunction, executor):
    """Apply the adjoint of unfocus_dft.

    Parameters
    ----------
    wavefunction : ndarray
        gradient at the pupil plane
    executor : MDFT or CZT
        (semi-)arbitrary sampling fourier transform executor

    Returns
    -------
    ndarray
        gradient at the focal plane

    """
    return executor(wavefunction)
