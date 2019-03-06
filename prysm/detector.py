"""Detector-related simulations."""
from collections import deque

from .conf import config
from .convolution import Convolvable
from .util import is_odd
from prysm import mathops as m


class Detector(object):
    """Model of a image sensor."""

    def __init__(self, pixel_size, resolution=(1024, 1024), nbits=16, framebuffer=24):
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
        self.pixel_size = pixel_size
        self.resolution = resolution
        self.bit_depth = nbits
        self.captures = deque(maxlen=framebuffer)

    def capture(self, convolvable):
        """Sample a convolvable, mimics capturing a photo of an oversampled representation of an image.

        Parameters
        ----------
        convolvable : `prysm.Convolvable`
            a convolvable object

        Returns
        -------
        `prysm.convolvable`
            a new convolvable object, as it would be sampled by the detector

        Raises
        ------
        ValueError
            if the convolvable would have to become supersampled by the detector;
            this would lead to an inaccurate result and is not supported

        """
        # we assume the pixels are bigger than the samples in the convolvable
        samples_per_pixel = self.pixel_size / convolvable.sample_spacing
        if samples_per_pixel < 1:
            raise ValueError('Pixels smaller than samples, bindown not possible.')
        else:
            samples_per_pixel = int(m.ceil(samples_per_pixel))

        data = bindown(convolvable.data, samples_per_pixel)
        s = data.shape
        extx, exty = s[0] * self.pixel_size // 2, s[1] * self.pixel_size // 2
        ux, uy = m.arange(-extx, exty, self.pixel_size), m.arange(-exty, exty, self.pixel_size)
        self.captures.append(Convolvable(data=data, unit_x=ux, unit_y=uy, has_analytic_ft=False))
        return self.captures[-1]

    def save_image(self, path, which='last'):
        """Save an image captured by the detector.

        Parameters
        ----------
        path : `string`
            path to save the image to

        which : `string` or `int`
            if string, "first" or "last", otherwise index into the capture buffer of the camera.

        Raises
        ------
        ValueError
            bad target frame to save; should always be the a valid int < buffer_depth

        """
        if which.lower() == 'last':
            self.captures[-1].save(path, self.bit_depth)
        elif type(which) is int:
            self.captures[which].save(path, self.bit_depth)
        else:
            raise ValueError('invalid "which" provided')

    def show_image(self, which='last', fig=None, ax=None):
        """Show an image captured by the detector.

        Parameters
        ----------
        which : `string` or `int`
            if string, "first" or "last", otherwise index into the capture buffer of the camera
        fig : `matplotlib.figure.Figure`, optional:
            Figure containing the plot
        ax : `matplotlib.axes.Axis`, optional:
            Axis containing the plot

        Returns
        -------
        fig : `matplotlib.figure.Figure:
            Figure containing the plot
        ax : `matplotlib.axes.Axis`
            Axis containing the plot

        """
        if which.lower() == 'last':
            fig, ax = self.captures[-1].show(fig=fig, ax=ax)
        elif type(which) is int:
            fig, ax = self.captures[which].show(fig=fig, ax=ax)
        return fig, ax

    @property
    def fs(self):
        """Sampling frequency in cy/mm."""
        return 1 / self.pixel_size * 1e3

    @property
    def nyquist(self):
        """Nyquist frequency in cy/mm."""
        return self.fs / 2


class OLPF(Convolvable):
    """Optical Low Pass Filter.

    Applies blur to an image to suppress high frequency MTF and aliasing when combined with a PixelAperture.

    Attributes
    ----------
    width_x : `float`
        x width parameter
    width_y : `float`
        y width parameter

    """
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

            data = m.zeros((samples_x, samples_y))

            data[center_y - shift_y, center_x - shift_x] = 1
            data[center_y - shift_y, center_x + shift_x] = 1
            data[center_y + shift_y, center_x - shift_x] = 1
            data[center_y + shift_y, center_x + shift_x] = 1
            ux = m.linspace(-space_x, space_x, samples_x)
            uy = m.linspace(-space_y, space_y, samples_y)

        super().__init__(data=data, unit_x=ux, unit_y=uy, has_analytic_ft=True)

    def analytic_ft(self, unit_x, unit_y):
        """Analytic fourier transform of a pixel aperture.

        Parameters
        ----------
        unit_x : `numpy.ndarray`
            sample points in x axis
        unit_y : `numpy.ndarray`
            sample points in y axis

        Returns
        -------
        `numpy.ndarray`
            2D numpy array containing the analytic fourier transform

        """
        xq, yq = m.meshgrid(unit_x, unit_y)
        return (m.cos(2 * xq * self.width_x) *
                m.cos(2 * yq * self.width_y)).astype(config.precision)


class PixelAperture(Convolvable):
    """The aperture of a pixel.

    Attributes
    ----------
    center_x : `int`
        x axis center pixel
    center_y : `int`
        y axis center pixel
    width_x : `float`
        x-width of the pixel aperture, um, full wide
    width_y : `float`
        y-width of the pixel aperture, um, full width

    """
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

            data = m.zeros((samples_x, samples_y))
            data[center_y - steps_y:center_y + steps_y,
                 center_x - steps_x:center_x + steps_x] = 1
            extx, exty = samples_x // 2 * sample_spacing, samples_y // 2 * sample_spacing
            ux, uy = m.linspace(-extx, extx, samples_x), m.linspace(-exty, exty, samples_y)
        super().__init__(data=data, unit_x=ux, unit_y=uy, has_analytic_ft=True)

    def analytic_ft(self, unit_x, unit_y):
        """Analytic fourier transform of a pixel aperture.

        Parameters
        ----------
        unit_x : `numpy.ndarray`
            sample points in x axis
        unit_y : `numpy.ndarray`
            sample points in y axis

        Returns
        -------
        `numpy.ndarray`
            2D numpy array containing the analytic fourier transform

        """
        xq, yq = m.meshgrid(unit_x, unit_y)
        return abs(pixelaperture_analytic_otf(self.width_x, self.width_y, xq, yq))


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
    return m.sinc(freq_x * width_x) * m.sinc(freq_y * width_y)


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
        samples_top = int(m.floor(samples_tmp_y))
        samples_bottom = int(m.ceil(samples_tmp_y))
        samples_left = int(m.ceil(samples_tmp_x))
        samples_right = int(m.floor(samples_tmp_x))
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

    return output_data[:px_y - trim_y, :px_x - trim_x]
