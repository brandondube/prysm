"""Specialized propagation for lyot family coronagraphs.

Can be expanded to add vortex, too.
"""
from ..mathops import np, ndimage
from ._kernels import _adjoint_multiply
from .dft import focus_dft, focus_dft_adjoint, unfocus_dft, unfocus_dft_adjoint


def to_fpm_and_back(wavefunction, fpm, executor, return_more=False):
    """Propagate to a focal plane mask, apply it, and return.

    Composition of focus_dft, multiplication by fpm, and
    unfocus_dft. The same MDFT executor is used for both legs (its
    adjoint provides the inverse). To invoke Babinet's principle, pass
    fpm=1 - fpm.

    Parameters
    ----------
    wavefunction : ndarray
        complex pupil-plane field to propagate
    fpm : ndarray
        the focal plane mask
    executor : MDFT or CZT
        (semi-)arbitrary sampling fourier transform executor
    return_more : bool, optional
        if True, return (new_wavefront, field_at_fpm, field_after_fpm)
        else return new_wavefront

    Returns
    -------
    ndarray, [ndarray, ndarray]
        next pupil; optionally also field at fpm and field after fpm

    """
    field_at_fpm = focus_dft(wavefunction, executor)
    field_after_fpm = field_at_fpm * fpm
    field_at_next_pupil = unfocus_dft(field_after_fpm, executor)

    if return_more:
        return field_at_next_pupil, field_at_fpm, field_after_fpm
    return field_at_next_pupil


def to_fpm_and_back_adjoint(wavefunction, fpm, executor, return_more=False,
                            return_fpm_grad=False, field_at_fpm=None):
    """Apply the adjoint of to_fpm_and_back.

    Parameters
    ----------
    wavefunction : ndarray
        gradient at the next pupil plane (output of the forward call)
    fpm : ndarray
        the focal plane mask used in the forward propagation
    executor : MDFT or CZT
        (semi-)arbitrary sampling fourier transform executor
    return_more : bool, optional
        if True, return (Eabar, Ebbar, intermediate)
        else return Eabar
    return_fpm_grad : bool, optional
        if True, also return the gradient with respect to fpm. Requires
        field_at_fpm from the matching forward propagation.
    field_at_fpm : ndarray, optional
        focal-plane field before the FPM from the forward propagation

    Returns
    -------
    ndarray or tuple of ndarray
        gradient at the input pupil; optionally also the intermediate gradients
        and/or the gradient with respect to fpm

    """
    if return_fpm_grad and field_at_fpm is None:
        raise ValueError('return_fpm_grad=True requires field_at_fpm from the forward propagation')

    fpm_is_complex = np.iscomplexobj(fpm)

    Ebbar = unfocus_dft_adjoint(wavefunction, executor)
    intermediate = _adjoint_multiply(Ebbar, fpm)
    Eabar = focus_dft_adjoint(intermediate, executor)

    if return_fpm_grad:
        fpm_bar = _adjoint_multiply(Ebbar, field_at_fpm, real=not fpm_is_complex)

    if return_more:
        if return_fpm_grad:
            return Eabar, Ebbar, intermediate, fpm_bar
        return Eabar, Ebbar, intermediate
    elif return_fpm_grad:
        return Eabar, fpm_bar
    else:
        return Eabar


def vortex_phase_mask(charge):
    """Build a focal-plane-mask callable for a charge-charge optical vortex.

    The returned callable evaluates exp(i * charge * theta), the azimuthal phase
    ramp of a vortex coronagraph, on focal-plane coordinate grids. Pass it to
    to_fpm_and_back_multiresolution, whose per-level grids resolve the on-axis
    phase singularity.

    Parameters
    ----------
    charge : int
        topological charge of the vortex; even charges null a clear circular
        aperture in the downstream Lyot plane

    Returns
    -------
    callable
        fpm(xf, yf) -> ndarray, with xf and yf focal-plane coordinate grids

    """
    def fpm(xf, yf):
        return np.exp((1j * charge) * np.arctan2(yf, xf))

    return fpm


def prepare_measured_fpm(measurement, dx, center=(0, 0), charge=None,
                         fill=None, order=1):
    """Wrap a measured complex focal-plane-mask map as an fpm callable.

    A high-resolution metrology map of a physically realized mask (e.g. a
    fabricated vortex, with its real surface, etch-depth, and amplitude errors)
    lives on its own uniform grid. The multi-resolution executor evaluates the
    focal-plane mask on a different grid at every level, so the map must be
    resampled on demand. This returns a callable fpm(xf, yf) that bilinearly
    (or higher-order) interpolates the measured complex transmission at the
    requested focal coordinates and falls back to an ideal continuation outside
    the measured extent — letting to_fpm_and_back_multiresolution propagate the
    as-built mask to assess real manufacturing errors.

    The measurement is assumed centered per the make_xy_grid / fftrange
    convention: array index n // 2 along each axis maps to focal coordinate
    center. The coarse levels span a far larger field of view than a real
    measurement, and the finest levels zoom below its resolution; the partition
    windows confine each level to the annulus its sampling resolves, so simple
    interpolation per level is appropriate.

    Parameters
    ----------
    measurement : ndarray
        complex transmission (amplitude times exp(1j * phase)) of the realized
        mask. For a measured phase map phi in radians, pass np.exp(1j * phi).
    dx : float
        sample spacing of the measurement, in the focal-plane coordinate units
        of the executor (microns for prepare_multiresolution).
    center : (float, float), optional
        (x, y) focal coordinate of the measurement's center sample, microns.
        The mask singularity should sit here. Default (0, 0).
    charge : int, optional
        if given, focal points outside the measured extent fall back to an
        ideal charge-charge vortex phase, the natural continuation of a
        vortex mask beyond the measured region. Ignored if fill is given.
    fill : scalar or callable, optional
        value, or fpm(xf, yf) callable, for points outside the measured extent.
        Overrides charge. Defaults to an ideal vortex if charge is given, else
        1 (no effect outside the measured region).
    order : int, optional
        spline order for the interpolation, passed to map_coordinates. 1
        (bilinear, default) is local and overshoot-free; 3 is smoother for
        clean maps.

    Returns
    -------
    callable
        fpm(xf, yf) -> complex ndarray, suitable for
        to_fpm_and_back_multiresolution

    """
    meas = np.asarray(measurement)
    ny, nx = meas.shape
    cx, cy = center
    re = np.real(meas)
    im = np.imag(meas)

    if fill is None:
        fill = vortex_phase_mask(charge) if charge is not None else 1.0
    fill_is_callable = callable(fill)

    def fpm(xf, yf):
        col = (xf - cx) / dx + nx // 2
        row = (yf - cy) / dx + ny // 2
        coords = np.stack([row.reshape(-1), col.reshape(-1)])
        ri = ndimage.map_coordinates(re, coords, order=order, mode='nearest')
        ii = ndimage.map_coordinates(im, coords, order=order, mode='nearest')
        interp = (ri + 1j * ii).reshape(xf.shape)
        inside = (row >= 0) & (row <= ny - 1) & (col >= 0) & (col <= nx - 1)
        fillv = fill(xf, yf) if fill_is_callable else fill
        return np.where(inside, interp, fillv)

    return fpm


def to_fpm_and_back_multiresolution(wavefunction, fpm, executor):
    """Propagate to a focal plane mask and back at multiple resolutions.

    The multi-resolution analogue of to_fpm_and_back. Each level of executor
    forward-propagates the pupil to its focal grid, applies the mask times the
    level's partition-of-unity window, and inverse-propagates; the level
    contributions are summed. This densely samples the singular core of a mask
    such as a vortex phase ramp without truncating any spatial frequency.

    Parameters
    ----------
    wavefunction : ndarray
        complex pupil-plane field to propagate
    fpm : callable
        fpm(xf, yf) -> ndarray, the focal plane mask evaluated on focal-plane
        coordinate grids (microns). See vortex_phase_mask.
    executor : MultiResolutionExecutor
        stack of executors and hand-off windows from prepare_multiresolution

    Returns
    -------
    ndarray
        field at the next pupil (Lyot) plane

    """
    out = None
    for ex, win, xf, yf in zip(executor.executors, executor.windows,
                               executor.xf, executor.yf):
        field_at_fpm = focus_dft(wavefunction, ex)
        field_after_fpm = field_at_fpm * fpm(xf, yf) * win
        contribution = unfocus_dft(field_after_fpm, ex)
        out = contribution if out is None else out + contribution
    return out


def to_fpm_and_back_multiresolution_adjoint(wavefunction, fpm, executor):
    """Apply the adjoint of to_fpm_and_back_multiresolution.

    Parameters
    ----------
    wavefunction : ndarray
        gradient at the next pupil (Lyot) plane
    fpm : callable
        the focal plane mask callable used in the forward propagation
    executor : MultiResolutionExecutor
        stack of executors and hand-off windows from prepare_multiresolution

    Returns
    -------
    ndarray
        gradient at the input pupil plane

    """
    out = None
    for ex, win, xf, yf in zip(executor.executors, executor.windows,
                               executor.xf, executor.yf):
        Ebbar = unfocus_dft_adjoint(wavefunction, ex)
        intermediate = _adjoint_multiply(Ebbar, fpm(xf, yf) * win)
        contribution = focus_dft_adjoint(intermediate, ex)
        out = contribution if out is None else out + contribution
    return out


def babinet(wavefunction, lyot, fpm, executor, return_more=False):
    """Propagate through a Lyot-style coronagraph using Babinet's principle.

    Parameters
    ----------
    wavefunction : ndarray
        complex pupil-plane field to propagate
    lyot : ndarray or None
        the Lyot stop; if None, equivalent to ones_like(wavefunction)
    fpm : ndarray
        the focal plane mask (1 inside the spot); the Babinet complement
        1 - fpm is formed internally (see Soummer et al 2007)
    executor : MDFT or CZT
        (semi-)arbitrary sampling fourier transform executor
    return_more : bool
        if True, return each plane in the propagation
        else return new_wavefront

    Notes
    -----
    if the substrate's reflectivity or transmissivity is not unity, and/or
    the mask's density is not infinity, babinet's principle works as follows:

    suppose we're modeling a Lyot focal plane mask;
    rr = radial coordinates of the image plane, in lambda/d units
    mask = rr < 5  # 1 inside FPM, 0 outside (babinet-style)

    now create some scalars for background transmission and mask transmission

    tau = 0.9 # background
    tmask = 0.1 # mask

    mask = tau - tau*mask + rmask*mask

    the mask variable now contains 0.9 outside the spot, and 0.1 inside

    Returns
    -------
    ndarray, [ndarray, ndarray, ndarray]
        field after lyot, [field at fpm, field after fpm, field at lyot]

    """
    fpm = 1 - fpm
    result = to_fpm_and_back(wavefunction, fpm=fpm, executor=executor, return_more=return_more)
    if return_more:
        field, field_at_fpm, field_after_fpm = result
    else:
        field = result

    field_at_lyot = wavefunction - field
    if lyot is not None:
        field_after_lyot = lyot * field_at_lyot
    else:
        field_after_lyot = field_at_lyot

    if return_more:
        return field_after_lyot, field_at_fpm, field_after_fpm, field_at_lyot
    return field_after_lyot


def babinet_adjoint(wavefunction, lyot, fpm, executor, field_at_fpm=None,
                    field_at_lyot=None, return_fpm_grad=False,
                    return_lyot_grad=False):
    """Apply the adjoint of babinet.

    Parameters
    ----------
    wavefunction : ndarray
        gradient at the field-after-lyot plane (output of the forward call)
    lyot : ndarray or None
        the Lyot stop; if None, equivalent to ones_like(wavefunction)
    fpm : ndarray
        the focal plane mask used in the forward propagation
    executor : MDFT or CZT
        (semi-)arbitrary sampling fourier transform executor
    field_at_fpm : ndarray, optional
        focal-plane field before the FPM from the matching forward call.
        Required when return_fpm_grad is True.
    field_at_lyot : ndarray, optional
        pupil-plane field before the Lyot stop from the matching forward
        call. Required when return_lyot_grad is True.
    return_fpm_grad : bool, optional
        if True, also return the gradient with respect to the original
        fpm argument passed to babinet.
    return_lyot_grad : bool, optional
        if True, also return the gradient with respect to lyot.

    Returns
    -------
    ndarray or tuple of ndarray
        adjoint-propagated gradient; optionally followed by FPM and/or Lyot
        gradients in the order requested by the keyword names

    """
    # Forward algebra:
    # B = 1 - fpm
    # c = to_fpm_and_back(a, B)
    # e = a - c
    # d = lyot * e
    if return_lyot_grad and field_at_lyot is None:
        raise ValueError('return_lyot_grad=True requires field_at_lyot from the forward propagation')

    lyot_is_complex = True if lyot is None else np.iscomplexobj(lyot)
    fpm = 1 - fpm

    dbar = wavefunction
    if lyot is not None:
        cbar = _adjoint_multiply(dbar, lyot)
    else:
        cbar = dbar

    if return_fpm_grad:
        abar, fpm_bar = to_fpm_and_back_adjoint(
            cbar, fpm=fpm, executor=executor,
            return_fpm_grad=True, field_at_fpm=field_at_fpm,
        )
    else:
        abar = to_fpm_and_back_adjoint(cbar, fpm=fpm, executor=executor)

    abar = cbar - abar
    if not (return_fpm_grad or return_lyot_grad):
        return abar

    out = [abar]
    if return_fpm_grad:
        out.append(fpm_bar)
    if return_lyot_grad:
        lyot_bar = _adjoint_multiply(dbar, field_at_lyot, real=not lyot_is_complex)
        out.append(lyot_bar)
    return tuple(out)
