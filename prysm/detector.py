"""Detector-related simulations."""
from collections import deque

from .conf import config
from .mathops import np
from .convolution import Convolvable
from .mathops import is_odd


class Detector(object):
    """Model of a image sensor."""

    def __init__(self, pitch_x=None, pitch_y=None, pixel='rectangle',
                 resolution=(1024, 1024), nbits=16, framebuffer=10):
        """Create a new Detector object.

        Parameters
        ----------
        pixel_size : `float`
            size of pixels, in um
        resolution : `iterable`
            (x,y) resolution in pixels
        nbits : `int`
            number of bits to digitize to
        framebuffer : `int`
            number of frames of data to store

        """
        if isinstance(pixel, str):
            pixel = pixel.lower()
            if pixel == 'rectangle' and pitch_x is None and pitch_y is None:
                raise ValueError('must provide at least x pitch for rectangular pixels.')

            if pixel == 'rectangle':
                pixel = PixelAperture(pitch_x, pitch_y)
                self.rectangular_100pct_fillfactor_pix = True
        else:
            self.rectangular_100pct_fillfactor_pix = False

        self.pixel = pixel

        if pitch_y is None:
            pitch_y = pitch_x

        if not hasattr(resolution, '__iter__'):
            resolution = (resolution, resolution)

        self.pitch_x = pitch_x
        self.pitch_y = pitch_y
        self.resolution = resolution
        self.bit_depth = nbits
        self.captures = deque(maxlen=framebuffer)


class OLPF(Convolvable):
    """Optical Low Pass Filter."""

    def __init__(self, width_x, width_y=None, sample_spacing=0, samples_x=None, samples_y=None):
        """Create a new OLPF object.

        Parameters
        ----------
        width_x : `float`
            blur width in the x direction, microns
        width_y : `float`
            blur width in the y direction, microns
        sample_spacing : `float`, optional
            center to center spacing of samples
        samples_x : `int`, optional
            number of samples along x axis
        samples_y : `int`, optional
            number of samples along y axis; duplicates x if None

        """
        # compute relevant spacings
        if width_y is None:
            width_y = width_x
        if samples_y is None:
            samples_y = samples_x

        self.width_x = width_x
        self.width_y = width_y

        if samples_x is None:  # do no math
            data, ux, uy = None, None, None
        else:
            space_x = width_x / 2
            space_y = width_y / 2
            shift_x = int(space_x // sample_spacing)
            shift_y = int(space_y // sample_spacing)
            center_x = samples_x // 2
            center_y = samples_y // 2

            data = np.zeros((samples_x, samples_y))

            data[center_y - shift_y, center_x - shift_x] = 1
            data[center_y - shift_y, center_x + shift_x] = 1
            data[center_y + shift_y, center_x - shift_x] = 1
            data[center_y + shift_y, center_x + shift_x] = 1
            ux = np.linspace(-space_x, space_x, samples_x)
            uy = np.linspace(-space_y, space_y, samples_y)

        super().__init__(data=data, x=ux, y=uy, has_analytic_ft=True)

    def analytic_ft(self, x, y):
        """Analytic fourier transform of a pixel aperture.

        Parameters
        ----------
        x : `numpy.ndarray`
            sample points in x axis
        y : `numpy.ndarray`
            sample points in y axis

        Returns
        -------
        `numpy.ndarray`
            2D numpy array containing the analytic fourier transform

        """
        return (np.cos(2 * self.width_x * x) *
                np.cos(2 * self.width_y * y)).astype(config.precision)


class PixelAperture(Convolvable):
    """The aperture of a rectangular pixel."""
    def __init__(self, width_x, width_y=None, sample_spacing=0, samples_x=None, samples_y=None):
        """Create a new `PixelAperture` object.

        Parameters
        ----------
        width_x : `float`
            width of the aperture in the x dimension, in microns.
        width_y : `float`, optional
            siez of the aperture in the y dimension, in microns
        sample_spacing : `float`, optional
            spacing of samples, in microns
        samples_x : `int`, optional
            number of samples in the x dimension
        samples_y : `int`, optional
            number of samples in the y dimension

        """
        if width_y is None:
            width_y = width_x
        if samples_y is None:
            samples_y = samples_x

        self.width_x = width_x
        self.width_y = width_y

        if samples_x is None:  # do no math
            data, ux, uy = None, None, None
        else:  # build PixelAperture model
            center_x = samples_x // 2
            center_y = samples_y // 2
            half_width = width_x / 2
            half_height = width_y / 2
            steps_x = int(half_width // sample_spacing)
            steps_y = int(half_height // sample_spacing)

            data = np.zeros((samples_x, samples_y))
            data[center_y - steps_y:center_y + steps_y,
                 center_x - steps_x:center_x + steps_x] = 1
            extx, exty = samples_x // 2 * sample_spacing, samples_y // 2 * sample_spacing
            ux, uy = np.linspace(-extx, extx, samples_x), np.linspace(-exty, exty, samples_y)
        super().__init__(data=data, x=ux, y=uy, has_analytic_ft=True)

    def analytic_ft(self, x, y):
        """Analytic fourier transform of a pixel aperture.

        Parameters
        ----------
        x : `numpy.ndarray`
            sample points in x axis
        y : `numpy.ndarray`
            sample points in y axis

        Returns
        -------
        `numpy.ndarray`
            2D numpy array containing the analytic fourier transform

        """
        return pixelaperture_analytic_otf(self.width_x, self.width_y, x, y)


def pixelaperture_analytic_otf(width_x, width_y, freq_x, freq_y):
    """Analytic MTF of a rectangular pixel aperture.

    Parameters
    ----------
    width_x : `float`
        x diameter of the pixel, in microns
    width_y : `float`
        y diameter of the pixel, in microns
    freq_x : `numpy.ndarray`
        x spatial frequency, in cycles per micron
    freq_y : `numpy.ndarray`
        y spatial frequency, in cycles per micron

    Returns
    -------
    `numpy.ndarray`
        MTF of the pixel aperture

    """
    return np.sinc(freq_x * width_x) * np.sinc(freq_y * width_y)


def bindown(array, nsamples_x, nsamples_y=None, mode='avg'):
    """Bin (resample) an array.

    Parameters
    ----------
    array : `numpy.ndarray`
        array of values
    nsamples_x : `int`
        number of samples in x axis to bin by
    nsamples_y : `int`
        number of samples in y axis to bin by.  If None, duplicates value from nsamples_x
    mode : `str`, {'avg', 'sum'}
        sum or avg, how to adjust the output signal

    Returns
    -------
    `numpy.ndarray`
        ndarray binned by given number of samples

    Notes
    -----
    Array should be 2D.  TODO: patch to allow 3D data.

    If the size of `array` is not evenly divisible by the number of samples,
    the algorithm will trim around the border of the array.  If the trim
    length is odd, one extra sample will be lost on the left side as opposed
    to the right side.

    Raises
    ------
    ValueError
        invalid mode

    """
    if nsamples_y is None:
        nsamples_y = nsamples_x

    if nsamples_x == 1 and nsamples_y == 1:
        return array

    # determine amount we need to trim the array
    samples_x, samples_y = array.shape
    total_samples_x = samples_x // nsamples_x
    total_samples_y = samples_y // nsamples_y
    final_idx_x = total_samples_x * nsamples_x
    final_idx_y = total_samples_y * nsamples_y

    residual_x = int(samples_x - final_idx_x)
    residual_y = int(samples_y - final_idx_y)

    # if the amount to trim is symmetric, trim symmetrically.
    if not is_odd(residual_x) and not is_odd(residual_y):
        samples_to_trim_x = residual_x // 2
        samples_to_trim_y = residual_y // 2
        trimmed_data = array[samples_to_trim_x:final_idx_x + samples_to_trim_x,
                             samples_to_trim_y:final_idx_y + samples_to_trim_y]
    # if not, trim more on the left.
    else:
        samples_tmp_x = (samples_x - final_idx_x) // 2
        samples_tmp_y = (samples_y - final_idx_y) // 2
        samples_top = int(np.floor(samples_tmp_y))
        samples_bottom = int(np.ceil(samples_tmp_y))
        samples_left = int(np.ceil(samples_tmp_x))
        samples_right = int(np.floor(samples_tmp_x))
        trimmed_data = array[samples_left:final_idx_x + samples_right,
                             samples_bottom:final_idx_y + samples_top]

    intermediate_view = trimmed_data.reshape(total_samples_x, nsamples_x,
                                             total_samples_y, nsamples_y)

    if mode.lower() in ('avg', 'average', 'mean'):
        output_data = intermediate_view.mean(axis=(1, 3))
    elif mode.lower() == 'sum':
        output_data = intermediate_view.sum(axis=(1, 3))
    else:
        raise ValueError('mode must be average of sum.')

    # trim as needed to make even number of samples.
    # TODO: allow work with images that are of odd dimensions
    px_x, px_y = output_data.shape
    trim_x, trim_y = 0, 0
    if is_odd(px_x):
        trim_x = 1
    if is_odd(px_y):
        trim_y = 1

    return output_data[:px_x - trim_x, :px_y - trim_y]


def bindown_with_units(px_x, px_y, source_spacing, source_data):
    """Perform bindown, returning unit axes and data.

    Parameters
    ----------
    px_x : `float`
        pixel pitch in the x direction, microns
    px_y : `float`
        pixel pitch in the y direction, microns
    source_spacing : `float`
        pixel pitch in the source data, microns
    source_data : `numpy.ndarray`
        ndarray of regularly spaced data

    Returns
    -------
    ux : `numpy.ndarray`
        1D array of sample coordinates in the x direction
    uy : `numpy.ndarray`
        1D array of sample coordinates in the y direction
    data : `numpy.ndarray`
        binned-down data

    """
    # we assume the pixels are bigger than the samples in the source
    spp_x = px_x / source_spacing
    spp_y = px_y / source_spacing
    if min(spp_x, spp_y) < 1:
        raise ValueError('Pixels smaller than samples, bindown not possible.')
    else:
        spp_x, spp_y = int(np.ceil(spp_x)), int(np.ceil(spp_y))

    data = bindown(source_data, spp_x, spp_y, 'avg')
    s = data.shape
    extx, exty = s[0] * px_x // 2, s[1] * px_y // 2
    ux, uy = np.arange(-extx, extx, px_x), np.arange(-exty, exty, px_y)
    return ux, uy, data
