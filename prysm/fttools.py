"""Supplimental tools for computing fourier transforms."""
from prysm import mathops as m


def pad2d(array, Q=2, value=0):
    """Symmetrically pads a 2D array with a value.

    Parameters
    ----------
    array : `numpy.ndarray`
        source array
    Q : `float` or `int`
        oversampling factor; ratio of input to output array widths
    value : `float` or `int`
        value with which to pad the array

    Returns
    -------
    `numpy.ndarray`
        padded array

    Notes
    -----
    padding will be symmetric.

    """
    if Q is 1:
        return array
    else:
        y, x = array.shape
        out_x = int(x * Q)
        out_y = int(y * Q)
        factor_x = (out_x - x) / 2
        factor_y = (out_y - y) / 2
        pad_shape = (
            (int(m.floor(factor_y)), int(m.ceil(factor_y))),
            (int(m.floor(factor_x)), int(m.ceil(factor_x))))
        if value is 0:
            out = m.zeros((out_y, out_x), dtype=array.dtype)
        else:
            out = m.zeros((out_y, out_x), dtype=array.dtype) + value
        yy, xx = pad_shape
        out[yy[0]:yy[0] + y, xx[0]:xx[0] + x] = array
        return out


def unpad2d(array, Q=2):
    """Unpad an array after applying pad2d, above.  Will not work for original arrays of odd length.

    Parameters
    ----------
    array : `numpy.ndarray`
        array of data
    Q : `float`
        oversampling factor, same as pad2d

    Returns
    -------
    `numpy.ndarray`
        unpadded data with shape = array.shape // 2

    """
    iw, ih = array.shape
    ow, oh = iw / Q, ih / Q
    dw, dh = iw - ow, ih - oh
    cut_w, cut_h = int(dw // 2), int(dh // 2)
    return array[cut_w:-cut_w, cut_h:-cut_h].copy()


def forward_ft_unit(sample_spacing, samples):
    """Compute the units resulting from a fourier transform.

    Parameters
    ----------
    sample_spacing : `float`
        center-to-center spacing of samples in an array
    samples : `int`
        number of samples in the data

    Returns
    -------
    `numpy.ndarray`
        array of sample frequencies in the output of an fft

    """
    return m.fftshift(m.fftfreq(samples, sample_spacing))
