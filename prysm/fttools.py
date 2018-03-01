"""Supplimental tools for computing fourier transforms."""
import numpy as np

from prysm.mathops import floor, ceil, ifftshift, fftfreq


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
    x, y = array.shape
    out_x = x * Q
    out_y = y * Q
    factor_x = (out_x - x) / 2
    factor_y = (out_y - x) / 2
    pad_shape = ((floor(factor_x), ceil(factor_x)), (floor(factor_y), ceil(factor_y)))
    if value is 0:
        out = np.zeros((out_x, out_y), dtype=array.dtype)
    else:
        out = np.ones((out_x, out_y), dtype=array.dtype) * value
    x_idx1, x_idx2 = pad_shape[0][0], pad_shape[0][1]
    y_idx1, y_idx2 = pad_shape[1][0], pad_shape[1][1]
    out[x_idx1:x_idx2 + x, y_idx1:y_idx2 + y] = array
    return out


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
    return ifftshift(fftfreq(samples, sample_spacing / 1e3))
