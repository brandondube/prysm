"""Detector-related simulations."""
import numbers
import functools
import itertools

from .mathops import np


class Detector:
    """Basic model of a detector, no fuss."""

    def __init__(self, dark_current, read_noise, bias, fwc, conversion_gain, bits, exposure_time, prnu=None, dcnu=None):
        """Initialize a new camera model.

        Parameters
        ----------
        dark_current : float
            e-/sec, charge accumulated with no light reaching the sensor.
        read_noise : float
            e-, random gaussian noise associated with readout
        bias : float
            e-, uniform value added to readout to avoid negative numbers
        fwc : float
            e-, maximum number of electrons that can be held by one pixel
        conversion_gain : float
            e-/DN gain converting e- to DN,
        bits : int
            number of bits for the ADC, multiples of 2 in 8..16 are contemporary
        exposure_time : float
            exposure time, seconds
        prnu : numpy.ndarray, optional
            relative pixel response nonuiformity, a fixed map that the
            input field is multiplied by.  ones_like is perfectly uniform.
        dcnu : numpy.ndarray, optional
            dark current nonuniformity, a fixed map that the dark current
            is multiplied by.  ones_like is perfectly uniform.

        """
        self.dark_current = dark_current
        self.read_noise = read_noise
        self.bias = bias
        self.fwc = fwc
        self.conversion_gain = conversion_gain
        self.bits = bits
        self.exposure_time = exposure_time
        self.prnu = prnu
        self.dcnu = dcnu

    def expose(self, aerial_img, frames=1):
        """Form an exposure of an aerial image.

        Parameters
        ----------
        aerial_img : numpy.ndarray
            aerial image, with units of e-/sec.  Should include any QE as part
            of its Z scaling
        frames : int
            number of images to expose, > 1 is functionally equivalent to
            calling with frames=1 in a loop, but the random values are all drawn
            at once which can much improve performance in GPU-based modeling.

        Returns
        -------
        numpy.ndarray
            of shape (frames, *aerial_img.shape), if frames=1 the first dim
            is squeezed, and output shape is same as input shape.
            dtype=uint8 if nbits <= 8, else uint16 for <= 16, etc
            not scaled to fill containers, i.e. a 12-bit image will have peak
            DN of 4095 in a container that can reach 65535.

            has units of DN.

        """
        electrons = aerial_img * self.exposure_time
        dark = self.dark_current * self.exposure_time
        # if the dark is not uniform, scale by nonuniformity
        if self.dcnu is not None:
            dark = dark * self.dcnu

        # ravel so that the random generation can be batched
        electrons = (electrons + dark).ravel()
        shot_noise = np.random.poisson(electrons, (frames, electrons.size))

        if self.prnu is not None:
            shot_noise = shot_noise * self.prnu

        # 0 is zero mean
        read_noise = np.random.normal(0, self.read_noise, shot_noise.shape)

        # invert conversion gain, mul is faster than div
        scaling = 1 / self.conversion_gain
        input_to_adc = (shot_noise + read_noise + self.bias)
        input_to_adc[input_to_adc > self.fwc] = self.fwc
        output = input_to_adc * scaling
        adc_cap = 2 ** self.bits
        output[output < 0] = 0
        output[output > adc_cap] = adc_cap
        # output will be of type int64, only good for 63 unsigned bits
        if self.bits <= 8:
            output = output.astype(np.uint8)
        elif self.bits <= 16:
            output = output.astype(np.uint16)
        elif self.bits <= 32:
            output = output.astype(np.uint32)
        else:
            raise ValueError('numpy''s random functionality is inadequate for > 32 unsigned bits')

        output = output.reshape((frames, *aerial_img.shape))
        if frames == 1:
            output = output[0, :, :]

        return output


def olpf_ft(fx, fy, width_x, width_y):
    """Analytic FT of an optical low-pass filter, two or four pole.

    Parameters
    ----------
    fx : numpy.ndarray
        x spatial frequency, in cycles per micron
    fy : numpy.ndarray
        y spatial frequency, in cycles per micron
    width_x : float
        x diameter of the pixel, in microns
    width_y : float
        y diameter of the pixel, in microns

    Returns
    -------
    numpy.ndarray
        FT of the OLPF

    """
    return np.cos(2 * width_x * fx) * np.cos(2 * width_y * fy)


def pixel_ft(fx, fy, width_x, width_y):
    """Analytic FT of a rectangular pixel aperture.

    Parameters
    ----------
    fx : numpy.ndarray
        x spatial frequency, in cycles per micron
    fy : numpy.ndarray
        y spatial frequency, in cycles per micron
    width_x : float
        x diameter of the pixel, in microns
    width_y : float
        y diameter of the pixel, in microns

    Returns
    -------
    numpy.ndarray
        FT of the pixel

    """
    return np.sinc(fx * width_x) * np.sinc(fy * width_y)


def pixel(x, y, width_x, width_y):
    """Spatial representation of a pixel.

    Parameters
    ----------
    x : numpy.ndarray
        x coordinates
    y : numpy.ndarray
        y coordinates
    width_x : float
        x diameter of the pixel, in microns
    width_y : float
        y diameter of the pixel, in microns

    Returns
    -------
    numpy.ndarray
        spatial representation of the pixel

    """
    width_x = width_x / 2
    width_y = width_y / 2
    return (x <= width_x) & (x >= -width_x) & (y <= width_y) & (y >= -width_y)


def bindown(array, factor, mode='avg'):
    """Bin (resample) an array.

    Parameters
    ----------
    array : numpy.ndarray
        array of values
    factor : int or sequence of int
        binning factor.  If an integer, broadcast to each axis of array,
        else unique factors may be used for each axis.
    mode : str, {'avg', 'sum'}
        sum or avg, how to adjust the output signal

    Returns
    -------
    numpy.ndarray
        ndarray binned by given number of samples

    Notes
    -----
    For each axis of array, shape must be an integer multiple of factor.

    array may be ND, a scalar factor will broadcast to all dimensions.

    To bin an image cube e.g. of shape (3, m, n),  use bindown(img, [1, factor, factor])

    Raises
    ------
    ValueError
        invalid mode

    """
    if isinstance(factor, numbers.Number):
        factor = tuple([factor] * array.ndim)

    # these two lines look very complicated
    # we want to take an array of shape (m, n) and a binning factor of say, 2
    # and reshape the array to (m/2, 2, n/2, 2)
    # these lines do that, for an arbitrary number of dimensions
    output_shape = tuple(s//n for s, n in zip(array.shape, factor))
    output_shape = tuple(itertools.chain(*zip(output_shape, factor)))
    intermediate_view = array.reshape(output_shape)
    # reduction axes produces (1, 3) for 2D, or (1, 3, 5) for 3D, etc.
    reduction_axes = tuple(range(1, 2*array.ndim, 2))

    if mode.lower() in ('avg', 'average', 'mean'):
        output_data = intermediate_view.mean(axis=reduction_axes)
    elif mode.lower() == 'sum':
        output_data = intermediate_view.sum(axis=reduction_axes)
    else:
        raise ValueError('mode must be average or sum.')

    return output_data


def tile(array, factor, scaling='sum'):
    """Tile (repeat) an array by factor.

    Parameters
    ----------
    array : numpy.ndarray
        array of values
    factor : int or sequence of int
        binning factor.  If an integer, broadcast to each axis of array,
        else unique factors may be used for each axis.
    scaling : str, {'avg', 'sum'}
        sum or avg, how to adjust the output signal

    Returns
    -------
    numpy.ndarray
        ndarray binned by given number of samples

    Notes
    -----
    This function is the adjoint operation for bindown.

    It works with ND arrays, with the same rules as bindown.

    In 2D, it is equivalent to array.repeat(factor, axis=0).repeat(factor, axis=1)
    (and is more generally equivalent for higher dimensionality, but it runs
    about ndim times faster (twice as fast in 2D, 3x in 3D, etc).

    The return may be a view into the argument and is mutated after
    calling tile at the user's risk

    """
    if isinstance(factor, numbers.Number):
        factor = tuple([factor] * array.ndim)

    intermediate = [None] * len(factor)

    slc = (slice(s) for s in array.shape)
    shape1 = tuple(itertools.chain(*zip(slc, intermediate)))
    shape2 = tuple(itertools.chain(*zip(array.shape, factor)))
    output_shape = tuple(s*n for s, n in zip(array.shape, factor))

    # view an array of shape
    # (m, n)
    # => (m, 1, n, 1)
    # => (m, factor, n, factor) (via broadcast_to)
    view = np.broadcast_to(array[shape1], shape2)
    view = view.reshape(output_shape)
    if scaling == 'sum':
        # due to our ND nature, we can't just do factor ** array.ndim,
        # although in the majority of cases this line will do the same
        # thing that that would
        sf = functools.reduce(lambda x, y: x*y, factor)
        sf = 1 / sf
    elif scaling in ('avg', 'average', 'mean'):
        sf = 1
    else:
        raise ValueError('scaling must be average or sum')

    if sf != 1:
        view = view * sf

    return view
