"""Matrix-DFT / chirp-Z based pupil <-> focal propagation with arbitrary sampling.
"""
from collections.abc import Iterable

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
