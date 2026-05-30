"""Specialized propagation for lyot family coronagraphs.

Can be expanded to add vortex, too.
"""
from ..mathops import np
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
