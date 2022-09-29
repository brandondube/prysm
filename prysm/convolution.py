"""Recipes for numerical convolution."""
import inspect

from .mathops import fft
from .coordinates import optimize_xy_separable, cart_to_polar
from .fttools import forward_ft_unit


def conv(obj, psf):
    """Convolve an object and psf.

    Parameters
    ----------
    obj : numpy.ndarray
        array representing the object, of shape (M, N)
    psf : numpy.ndarray
        array representing the psf, of shape (M, N)

    Returns
    -------
    numpy.ndarray
        ndarray after undergoing convolution

    """
    # notation:
    o = obj
    h = psf
    O = fft.fft2(fft.ifftshift(o))  # NOQA : O ambiguous (not, lowercase => uppercase notation)
    H = fft.fft2(fft.ifftshift(h))
    i = fft.fftshift(fft.ifft2(O*H)).real  # i = image
    return i


def apply_transfer_functions(obj, dx, tfs, fx=None, fy=None, ft=None, fr=None, shift=False):
    """Blur an object by N transfer functions.

    Parameters
    ----------
    obj : numpy.ndarray
        array representing the object, of shape (M, N)
    dx : float
        sample spacing of the object.  Ignored if fx, etc are defined.
    tfs : sequence of callables, or arrays
        transfer functions.
        If a callable, should be  functions which
        take arguments of any of fx, fy, ft, fr.  Use functools partial or
        class methods to curry other parameters
    fx, fy, ft, fr : numpy.ndarray
        arrays defining the frequency domain, of shape (M, N)
            cartesian X frequency
            cartesian Y frequency
            azimuthal frequency
            radial frequency
        The latter two are simply the atan2 of the former two.
    shift : bool, optional
        if True, fx, fy, ft, fr are assumed to have the origin in the center
        of the array, and tfs are expected to be consistent with that.
        If False, the origin is assumed to be the [0,0]th sample of fr, fx, fy.

    Returns
    -------
    numpy.ndarray
        image after being blurred by each transfer function

    """
    if any(callable(tf) for tf in tfs):
        if fx is None:
            fy, fx = [forward_ft_unit(dx, n) for n in obj.shape]

        fx, fy = optimize_xy_separable(fx, fy)
        fr, ft = cart_to_polar(fx, fy)

    o = obj
    if shift:
        O = fft.fftshift(fft.fft2(fft.ifftshift(o)))  # NOQA
    else:
        O = fft.fft2(o)  # NOQA

    for tf in tfs:
        if callable(tf):
            sig = inspect.signature(tf)
            params = sig.parameters
            kwargs = {}
            if 'fx' in params:
                kwargs['fx'] = fx
            if 'fy' in params:
                kwargs['fy'] = fy
            if 'fr' in params:
                kwargs['fr'] = fr
            if 'ft' in params:
                kwargs['ft'] = ft

            tf = tf(**kwargs)

        O = O * tf  # NOQA

    if shift:
        return fft.fftshift(fft.ifft2(fft.ifftshift(O))).real
    # no if shift on this side, [i]fft will always place the origin at [0,0]
    # real inside shift - 2x faster to shift real than to shift complex
    i = fft.fftshift(fft.ifft2(O).real)
    return i
