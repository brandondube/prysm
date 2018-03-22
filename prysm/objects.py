''' Object to convolve lens PSFs with
'''

from multiprocessing import Pool

from functools import lru_cache

import numpy as np

from .conf import config
from .mathops import (
    fft2,
    fftshift,
    sin,
    cos,
    sqrt,
)
from .coordinates import cart_to_polar
from .psf import PSF
from .fttools import forward_ft_unit, pad2d
from .util import share_fig_ax, is_odd


class Image(object):
    ''' Images of an object
    '''
    def __init__(self, data, sample_spacing, synthetic=True):
        ''' Creates a new Image object.

        Args:
            data (`numpy.ndarray`): data that represents the image, 2D.

            sample_spacing (`float`): pixel pitch of the data.

        '''
        self.data = data
        self.sample_spacing = sample_spacing
        self.samples_y, self.samples_x = data.shape
        self.center_x, self.center_y = self.samples_x // 2, self.samples_y // 2
        self.ext_x = sample_spacing * self.center_x
        self.ext_y = sample_spacing * self.center_y
        self.synthetic = synthetic

    def show(self, interp_method=None, fig=None, ax=None):
        ''' Displays the image.

        Args:
            interp_method (`string`): interpolation technique used in display.

            fig (`matplotlib.figure`): figure to display in.

            ax (`matplotlib.axis`): axis to display in.

        Returns:
            `tuple` containing:

                `matplotlib.figure` figure containing the plot.

                `matplotlib.axis` axis containing the plot.

        '''
        lims = (0, 1)
        fig, ax = share_fig_ax(fig, ax)

        ax.imshow(self.data,
                  cmap='Greys_r',
                  interpolation=interp_method,
                  clim=lims,
                  origin='lower')
        ax.set_axis_off()
        return fig, ax

    def show_fourier(self, interp_method='lanczos', fig=None, ax=None):
        ''' Displays the fourier transform of the image.

        Args:
            interp_method (`string`): method used to interpolate the data
                for display.

            fig (`matplotlib.figure`): figure to plot in.

            ax (`matplotlib.axis`): axis to plot in.

        Returns:
            `tuple` containing:

                `matplotlib.figure` figure containing the plot.

                `matplotlib.axis` axis containing the plot.

        '''
        dat = abs(fftshift(fft2(pad2d(self.data))))
        dat /= dat.max()
        unit_x = forward_ft_unit(self.sample_spacing, self.samples_x)
        unit_y = forward_ft_unit(self.sample_spacing, self.samples_y)
        xmin, xmax = unit_x[0], unit_x[-1]
        ymin, ymax = unit_y[0], unit_y[-1]

        fig, ax = share_fig_ax(fig, ax)
        im = ax.imshow(dat**0.1,
                       extent=[xmin, xmax, ymin, ymax],
                       cmap='Greys_r',
                       interpolation=interp_method,
                       origin='lower')
        fig.colorbar(im)
        ax.set(xlabel='Spatial Frequency X [cy/mm]',
               ylabel='Spatial Frequency Y [cy/mm]',
               title='Normalized FT of image, to 0.1 power')
        return fig, ax

    def as_psf(self):
        ''' Converts this image to a PSF object.
        '''
        return PSF(self.data, self.sample_spacing)

    def convpsf(self, psf):
        ''' Convolves with a PSF for image simulation

        Args:
            psf (`PSF`): a PSF

        Returns:
            `Image` A new, blurred image.

        '''
        img_psf = self.as_psf()
        conved_image = _unequal_spacing_conv_core(img_psf, psf)
        # return conved_image
        return Image(data=conved_image.data,
                     sample_spacing=self.sample_spacing,
                     synthetic=self.synthetic)

    def save(self, path, nbits=8):
        ''' Write the image to a png, jpg, tiff, etc.

        Args:
            path (`string`): path to write the image to.

            nbits (`int`): number of bits in the output image.

        Returns:
            null: no return

        '''
        dat = (self.data * 255).astype(np.uint8)

        if self.synthetic is False:
            # was a real image, need to flip vertically.
            dat = np.flip(dat, axis=0)
        imsave(path, dat)

    @staticmethod
    def from_file(path, scale):
        ''' Reads a file into a new Image instance, always monochrome

        Args:
            path (`string`): path to a file.

            scale (`float`): pixel scale, in microns.

        Returns:
            `Image` a new image object.

        Notes:
            TODO: proper handling of images with more than 8bpp.
        '''
        imgarr = imread(path, flatten=True, mode='F')

        return Image(data=np.flip(imgarr, axis=0) / 255, sample_spacing=scale, synthetic=False)


class RGBImage(object):
    ''' RGB images
    '''
    def __init__(self, r, g, b, sample_spacing, synthetic=True):
        ''' creates a new RGB image

        Args:
            r (`Image`): array for the red channel.

            g (`Image`): array for the green channel.

            b (`Image`): array for the blue channel.

            sample_spacing (`float`): spacing between samples, in microns.

            synthetic (`bool`): whether or not the image is synthetic.  Real
                images are upside-down compared to synthetic ones.

        Returns:
            `RGBImage`, new RGB image.
        '''
        # load in data
        self.R = r
        self.G = g
        self.B = b

        # load in sampling information
        self.sample_spacing = sample_spacing
        self.samples_x, self.samples_y = r.shape
        self.center_x = self.samples_x // 2
        self.center_y = self.samples_y // 2
        self.ext_x = sample_spacing * self.center_x
        self.ext_y = sample_spacing * self.center_y

        # load in synthetic or not
        self.synthetic = synthetic

    def show(self, interp_method=None, fig=None, ax=None):
        ''' Displays the image.

        Args:
            interp_method (`string`): interpolation technique used in display.

            fig (`matplotlib.figure`): figure to display in.

            ax (`matplotlib.axis`): axis to display in.

        Returns:
            `tuple` containing:

                `matplotlib.figure` figure containing the plot.

                `matplotlib.axis` axis containing the plot.

        '''
        lims = (0, 1)
        fig, ax = share_fig_ax(fig, ax)

        dat = rgbimage_to_datacube(self)
        ax.imshow(dat,
                  interpolation=interp_method,
                  clims=lims,
                  origin='lower')
        ax.set_axis_off()
        return fig, ax

    def as_psf(self, color='g'):
        ''' Converts a color plane of this image to a PSF object.

        Args:
            color (`string`): red, green, or blue

        Returns:
            `PSF` a PSF object for the given color plane.

        '''
        if color.lower() in ('g', 'green'):
            dat = self.G
        elif color.lower() in ('r', 'red'):
            dat = self.R
        elif color.lower() in ('b', 'blue'):
            dat = self.B
        else:
            raise ValueError('invalid color selected')

        return PSF(dat, self.sample_spacing)

    def save(self, path, nbits=8):
        ''' Write the image to a png, jpg, tiff, etc.

        Args:
            path (`string`): path to write the image to.

            nbits (`int`): number of bits in the output image.

        Returns:
            null: no return

        '''
        dat = rgbimage_to_datacube(self)

        if self.synthetic is not True:
            # was a real image, need to flip vertically.
            dat = np.flip(dat, axis=0)

        imsave(path, dat)

    def convpsf(self, rgbpsf):
        ''' Convolves with a PSF for image simulation

        Args:
            rgbpsf (`RGBPSF`): an RGBPSF

        Returns:
            `RGBImage` A new, blurred image.

        '''
        img_r = self.as_psf('r')
        img_g = self.as_psf('g')
        img_b = self.as_psf('b')

        if config.parallel_rgb:
            psf_r = rgbpsf.r_psf._renorm(to='total')
            psf_g = rgbpsf.g_psf._renorm(to='total')
            psf_b = rgbpsf.b_psf._renorm(to='total')

            imgs = [img_r, img_g, img_b]
            psfs = [psf_r, psf_g, psf_b]
            with Pool(3) as pool:
                r_conv, g_conv, b_conv = pool.map(_unequal_spacing_conv_core, imgs, psfs)
        else:
            r_conv = _unequal_spacing_conv_core(img_r, rgbpsf.r_psf._renorm(to='total'))
            g_conv = _unequal_spacing_conv_core(img_g, rgbpsf.g_psf._renorm(to='total'))
            b_conv = _unequal_spacing_conv_core(img_b, rgbpsf.b_psf._renorm(to='total'))

        return RGBImage(r=r_conv.data, g=g_conv.data, b=b_conv.data,
                        sample_spacing=self.sample_spacing,
                        synthetic=self.synthetic)

    @staticmethod
    def from_file(path, scale):
        ''' Reads a file into a new RGBImage instance, must be 24bpp/8bpc

        Args:
            path (`string`): path to a file.

            scale (`float`): pixel scale, in microns.

        Returns:
            `RGBImage` a new image object.

        Notes:
            TODO: proper handling of images with more than 8bpp.
        '''
        # img is an mxnx3 array of unit8s
        img = imread(path).astype(config.precision) / 255

        img = np.flip(img, axis=0)

        # crop the image if it has an odd dimension.
        # TODO: change this an understand why it is an issue
        # fftshift vs ifftshift?
        if is_odd(img.shape[0]):
            img = img[0:-1, :, :]
        if is_odd(img.shape[1]):
            img = img[:, 0:-1, :]
        return RGBImage(r=img[:, :, 0], g=img[:, :, 1], b=img[:, :, 2],
                        sample_spacing=scale, synthetic=False)


def rgbimage_to_datacube(rgbimage):
    ''' Creates an mxnx3 array from an RGBImage

    Args:
        rgbimage (`RGBImage`): an RGBImage object.

    Returns:
        `numpy.ndarray` an ndarray of shape m x n x 3.
    '''
    dat = np.empty((rgbimage.samples_x, rgbimage.samples_y, 3), dtype=np.uint8)
    dat[:, :, 0] = rgbimage.R * 255
    dat[:, :, 1] = rgbimage.G * 255
    dat[:, :, 2] = rgbimage.B * 255
    return dat


class Slit(Image):
    ''' Representation of a slit or pair of slits.
    '''
    @lru_cache()
    def __init__(self, width, orientation='Vertical', sample_spacing=0.075, samples=384):
        ''' Creates a new Slit instance.

        Args:
            width (`float`): the width of the slit.

            orientation (`string`): the orientation of the slit:
                Horizontal, Vertical, Crossed, or Both.  Crossed and Both
                produce the same results.

            sample_spacing (`float`): spacing of samples in the synthetic image.

            samples (`int`): number of samples per dimension in the synthetic image.

        '''

        # produce coordinate arrays
        ext = samples / 2 * sample_spacing
        x, y = np.linspace(-ext, ext, samples), np.linspace(-ext, ext, samples)
        w = width / 2

        # produce the background
        arr = np.zeros((samples, samples))

        # paint in the slit
        if orientation.lower() in ('v', 'vert', 'vertical'):
            arr[:, abs(x) < w] = 1
            self.orientation = 'Vertical'
            self.width_x = width
        elif orientation.lower() in ('h', 'horiz', 'horizontal'):
            arr[abs(y) < w, :] = 1
            self.width_y = width
            self.orientation = 'Horizontal'
        elif orientation.lower() in ('b', 'both', 'c', 'crossed'):
            arr[abs(y) < w, :] = 1
            arr[:, abs(x) < w] = 1
            self.orientation = 'Crossed'
            self.width_x, self.width_y = width, width

        super().__init__(data=arr, sample_spacing=sample_spacing, synthetic=True)

    def analytical_ft(self, unit_x, unit_y):
        ''' Analytic fourier transform of a Slit

        Args:
            unit_x (numpy.ndarray): sample points in x axis.

            unit_y (numpy.ndarray): sample points in y axis.

        Returns:
            `numpy.ndarray` 2D numpy array containing the analytic fourier transform.

        '''
        xq, yq = np.meshgrid(unit_x, unit_y)
        return (sinc(xq * self.width_x / 1e3) +
                sinc(yq * self.width_y / 1e3)).astype(config.precision)


class Pinhole(Image):
    ''' Representation of a pinhole object.
    '''
    @lru_cache()
    def __init__(self, width, sample_spacing=0.025, samples=384):
        ''' Produces a Pinhole.

        Args:
            width (`float`): the width of the pinhole.

            sample_spacing (`float`): spacing of samples in the synthetic image.

            samples (`int`): number of samples per dimension in the synthetic image.

        '''
        self.width = width

        # produce coordinate arrays
        ext = samples / 2 * sample_spacing
        x, y = np.linspace(-ext, ext, samples), np.linspace(-ext, ext, samples)
        xv, yv = np.meshgrid(x, y)
        w = width / 2

        # paint a circle on a black background
        arr = np.zeros((samples, samples))
        arr[sqrt(xv**2 + yv**2) < w] = 1
        super().__init__(data=arr, sample_spacing=sample_spacing, synthetic=True)


class SiemensStar(Image):
    ''' Representation of a Siemen's star object.
    '''
    @lru_cache()
    def __init__(self, num_spokes, sinusoidal=True, background='black', sample_spacing=2, samples=384):
        ''' Produces a Siemen's Star.

        Args:
            num_spokes (`int`): number of spokes in the star.

            sinusoidal (`bool`): if True, generates a sinusoidal Siemen' star.
                If false, generates a bar/block siemen's star.

            background ('string'): "black" or "white".

            sample_spacing (`float`): Spacing of samples, in microns.

            samples (`int`): number of samples per dimension in the synthetic image.

        '''
        relative_width = 0.9
        self.num_spokes = num_spokes

        # generate a coordinate grid
        x = np.linspace(-1, 1, samples)
        y = np.linspace(-1, 1, samples)
        xx, yy = np.meshgrid(x, y)
        rv, pv = cart_to_polar(xx, yy)

        # generate the siemen's star as a (rho,phi) polynomial
        arr = cos(num_spokes / 2 * pv)

        if not sinusoidal:  # make binary
            arr[arr < 0] = -1
            arr[arr > 0] = 1

        # scale to (0,1) and clip into a disk
        arr = (arr + 1) / 2
        if background.lower() in ('b', 'black'):
            arr[rv > relative_width] = 0
        elif background.lower() in ('w', 'white'):
            arr[rv > relative_width] = 1
        else:
            raise ValueError('invalid background color')

        super().__init__(data=arr, sample_spacing=sample_spacing, synthetic=True)


class TiltedSquare(Image):
    ''' Describes a tilted square for e.g. slanted-edge
    '''
    def __init__(self, angle=8, background='white', sample_spacing=2, samples=384):
        ''' Creates a new TitledSquare instance.

        Args:
            angle (`float`): angle in degrees to tilt w.r.t. the x axis.

            background (`string`): white or black; the square will be the opposite
                color of the background.

            sample_spacing (`float`): spacing of samples.

            samples (`int`): number of samples.

        Returns:
            `TiltedSquare` new TiltedSquare instance.

        '''
        radius = 0.3
        if background.lower() == 'white':
            arr = np.ones((samples, samples))
            fill_with = 0
        else:
            arr = np.zeros((samples, samples))
            fill_with = 1

        # TODO: optimize by working with index numbers directly and avoid
        # creation of X,Y arrays for performance.
        x = np.linspace(-0.5, 0.5, samples)
        y = np.linspace(-0.5, 0.5, samples)
        xx, yy = np.meshgrid(x, y)

        # TODO: convert inline operation to use of rotation matrix
        angle = np.radians(angle)
        xp = xx * cos(angle) - yy * sin(angle)
        yp = xx * sin(angle) + yy * cos(angle)
        mask = (abs(xp) < radius) * (abs(yp) < radius)
        arr[mask] = fill_with
        super().__init__(data=arr, sample_spacing=sample_spacing, synthetic=True)
