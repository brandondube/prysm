"""Recipes for numerical convolution."""
import inspect

from .mathops import np
from .coordinates import optimize_xy_separable, cart_to_polar
from .fttools import forward_ft_unit


def conv(obj, psf):
    """Convolve an object and psf.

    Parameters
    ----------
    obj : `numpy.ndarray`
        array representing the object, of shape (M, N)
    psf : `numpy.ndarray`
        array representing the psf, of shape (M, N)

    Returns
    -------
    `numpy.ndarray`
        ndarray after undergoing convolution

    """
    # notation:
    o = obj
    h = psf
    O = np.fft.fft2(np.fft.ifftshift(o))  # NOQA : O ambiguous (not, lowercase => uppercase notation)
    H = np.fft.fft2(np.fft.ifftshift(h))
    i = np.fft.fftshift(np.fft.ifft2(O*H)).real  # i = image
    return i


def apply_transfer_functions(obj, dx, *tfs, fx=None, fy=None, ft=None, fr=None):
    """Blur an object by N transfer functions.

    Parameters
    ----------
    obj : `numpy.ndarray`
        array representing the object, of shape (M, N)
    dx : `float`
        sample spacing of the object.  Ignored if fx, etc are defined.
    tfs : sequence of `callable`s, or arrays
        transfer functions.  If an array, should be fftshifted with the origin
        in the center of the array.  If a callable, should be  functions which
        take arguments of any of fx, fy, ft, fr.  Use functools partial or
        class methods to curry other parameters
    fx, fy, ft, fr : `numpy.ndarray`
        arrays defining the frequency domain, of shape (M, N)
            cartesian X frequency
            cartesian Y frequency
            azimuthal frequency
            radial frequency
        The latter two are simply the atan2 of the former two.

    Returns
    -------
    `numpy.ndarray`
        image after being blurred by each transfer function

    """
    if fx is None:
        fy, fx = [forward_ft_unit(dx, n) for n in obj.shape]

    fx, fy = optimize_xy_separable(fx, fy)
    fr, ft = cart_to_polar(fx, fy)

    o = obj
    O = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(o)))  # NOQA

    for tf in tfs:
        if callable(tf):
            sig = inspect.signature(tf)
            params = sig.parameters
            kwargs = {}
            if fx in params:
                kwargs['fx'] = fx
            if fy in params:
                kwargs['fy'] = fy
            if fr in params:
                kwargs['fr'] = fr
            if ft in params:
                kwargs['ft'] = ft

            tf = tf(**kwargs)

        O = O * tf  # NOQA

    i = np.fft.fftshift(np.fft.ifft2(O)).real
    return i
