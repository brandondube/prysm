"""Supplimental tools for computing fourier transforms."""
from prysm import mathops as m


def pad2d(array, Q=2, value=0, mode='constant'):
    """Symmetrically pads a 2D array with a value.

    Parameters
    ----------
    array : `numpy.ndarray`
        source array
    Q : `float`, optional
        oversampling factor; ratio of input to output array widths
    value : `float`, optioanl
        value with which to pad the array
    mode : `str`, optional
        mode, passed directly to np.pad

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
        if mode == 'constant':
            pad_shape, out_x, out_y = _padshape(array, Q)
            y, x = array.shape
            if value is 0:
                out = m.zeros((out_y, out_x), dtype=array.dtype)
            else:
                out = m.zeros((out_y, out_x), dtype=array.dtype) + value
            yy, xx = pad_shape
            out[yy[0]:yy[0] + y, xx[0]:xx[0] + x] = array
            return out
        else:
            pad_shape, *_ = _padshape(array, Q)

            if mode == 'constant':
                kwargs = {'constant_values': value, 'mode': mode}
            else:
                kwargs = {'mode': mode}
            return m.pad(array, pad_shape, **kwargs)


def _padshape(array, Q):
    y, x = array.shape
    out_x = int(m.ceil(x * Q))
    out_y = int(m.ceil(y * Q))
    factor_x = (out_x - x) / 2
    factor_y = (out_y - y) / 2
    return (
        (int(m.floor(factor_y)), int(m.ceil(factor_y))),
        (int(m.floor(factor_x)), int(m.ceil(factor_x)))), out_x, out_y


def forward_ft_unit(sample_spacing, samples, shift=True):
    """Compute the units resulting from a fourier transform.

    Parameters
    ----------
    sample_spacing : `float`
        center-to-center spacing of samples in an array
    samples : `int`
        number of samples in the data
    shift : `bool`, optional
        whether to shift the output.  If True, first element is a negative freq
        if False, first element is 0 freq.

    Returns
    -------
    `numpy.ndarray`
        array of sample frequencies in the output of an fft

    """
    unit = m.fftfreq(samples, sample_spacing)

    if shift:
        return m.fftshift(unit)
    else:
        return unit
