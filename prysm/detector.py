"""Detector-related simulations."""
from collections import deque

import numpy as np

from .conf import config
from .convolution import Convolvable
from .mathops import floor, ceil, cos, sinc
from .objects import Image
from .util import is_odd


class Detector(object):
    """Model of a image sensor.

    Attributes
    ----------
    bit_depth : `int`
        number of bits to quantize to
    captures : `deque`
        stack of frames that have been captured
    pixel_size : `float`
        size of pixels, in microns
    resolution : `tuple` of ints
        resolution in px

    """

    def __init__(self, pixel_size, resolution=(1024, 1024), nbits=14, framebuffer=24):
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

    def sample_psf(self, psf):
        """Sample a PSF, mimics capturing a photo of an oversampled representation of an image.

        Parameters
        ----------
        psf : `PSF`
            a PSF object

        Returns
        -------
        `PSF`
            a new PSF object, as it would be sampled by the detector

        Raises
        ------
        ValueError
            if the PSF would have to become supersampled by the detector;
            this would lead to an inaccurate result and is not supported

        """
        # we assume the pixels are bigger than the samples in the PSF
        samples_per_pixel = self.pixel_size / psf.sample_spacing
        if samples_per_pixel < 1:
            raise ValueError('Pixels smaller than samples, bindown not possible.')
        else:
            samples_per_pixel = int(ceil(samples_per_pixel))

        data = bindown(psf.data, samples_per_pixel)
        self.captures.append(Image(data=data, sample_spacing=self.pixel_size))
        return self.captures[-1]

    def sample_image(self, image):
        """Sample an image.

        Parameters
        ----------
        image : `Image`
            an Image object

        Returns
        -------
        `Image`
            a new, sampled image

        """
        intermediate_psf = self.sample_psf(image.as_psf())
        self.captures.append(Image(data=intermediate_psf.data,
                                   sample_spacing=intermediate_psf.sample_spacing))
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
    def __init__(self, width_x, width_y=None, sample_spacing=0.1, samples_x=384, samples_y=None):
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
            data, ux, uy = None, np.zeros(2), np.zeros(2)
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
        xq, yq = np.meshgrid(unit_x, unit_y)
        return (cos(2 * xq * self.width_x) *
                cos(2 * yq * self.width_y)).astype(config.precision)


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
            data, ux, uy = None, np.zeros(2), np.zeros(2)
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
        xq, yq = np.meshgrid(unit_x, unit_y)
        return (sinc(xq * self.width_x) *
                sinc(yq * self.width_y)).astype(config.precision)


def generate_mtf(pixel_aperture=1, azimuth=0, num_samples=128):
    """Generate the 1D diffraction-limited MTF for a given pixel width and azimuth.

    Parameters
    ----------
    pixel_aperture : `float`
        aperture of the pixel, microns.  Pixel is assumed to be square
    azimuth : `float`
        azimuth to retrieve the MTF at, in degrees
    num_samples : `int`
        number of samples in the output array

    Returns
    -------
    frequencies : `numpy.ndarray`
        unit axis, cy/mm
    mtf : `numpy.ndarray`
        MTF values (rel. 1.0).

    Notes
    -----
    Azimuth is not actually implemented yet.

    """
    pitch_unit = pixel_aperture / 1e3
    normalized_frequencies = np.linspace(0, 2, num_samples)
    otf = np.sinc(normalized_frequencies)
    mtf = np.abs(otf)
    return normalized_frequencies / pitch_unit, mtf


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
        samples_top = int(floor(samples_tmp_y))
        samples_bottom = int(ceil(samples_tmp_y))
        samples_left = int(ceil(samples_tmp_x))
        samples_right = int(floor(samples_tmp_x))
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
