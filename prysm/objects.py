'''Object to convolve lens PSFs with.
'''
import numpy as np

from imageio import imread, imsave

from .conf import config
from .convolution import Convolvable
from .coordinates import cart_to_polar

from prysm import mathops as m


class Image(Convolvable):
    '''Image of an object.
    '''
    def __init__(self, data, sample_spacing, has_analytic_ft=False):
        '''Creates a new Image object.

        Parameters
        ----------
        data : `numpy.ndarray`
            data that represents the image, 2D.
        sample_spacing : `float`
            pixel pitch of the data.
        has_analytic_ft : `bool`
            whether the image has an analytical Fourier transform

        '''
        samples_y, samples_x = data.shape
        center_x, center_y = samples_x // 2, samples_y // 2
        ext_x = sample_spacing * center_x
        ext_y = sample_spacing * center_y
        unit_x, unit_y = np.linspace(-ext_x, ext_x, samples_x), np.linspace(-ext_y, ext_y, samples_y)
        super().__init__(data, unit_x, unit_y, has_analytic_ft)

    def save(self, path, nbits=8):
        '''Write the image to a png, jpg, tiff, etc.

        Parameters
        ----------
        path : `string`
            path to write the image to
        nbits : `int`
            number of bits in the output image

        '''
        dat = (self.data * 255).astype(np.uint8)
        imsave(path, dat)

    @staticmethod
    def from_file(path, scale):
        '''Read a monochrome 8 bit per pixel file into a new Image instance.

        Parameters
        ----------
        path : `string`
            path to a file
        scale : `float`
            pixel scale, in microns

        Returns
        -------
        `Image`
            a new image object

        Notes
        -----
        TODO: proper handling of images with more than 8bpp.

        '''
        imgarr = imread(path, flatten=True, pilmode='F')

        return Image(data=np.flip(imgarr, axis=0) / 255, sample_spacing=scale, synthetic=False)


class Slit(Convolvable):
    '''Representation of a slit or pair of slits.

    Attributes
    ----------
    orientation : `str`
        orientation of the slit
    width_x : `float`
        x-width of the slit
    width_y : `float`
        y-width of the slit
    '''

    def __init__(self, width, orientation='Vertical', sample_spacing=0.075, samples=0):
        '''Create a new Slit instance.

        Parameters
        ----------
        width : `float`
            the width of the slit in microns
        orientation : `string`, {'Horizontal', 'Vertical', 'Crossed', 'Both'}
            the orientation of the slit; Crossed and Both produce the same results
        sample_spacing : `float`
            spacing of samples in the synthetic image
        samples : `int`
            number of samples per dimension in the synthetic image

        Notes
        -----
        Default of 0 samples allows quick creation for convolutions without
        generating the image; use samples > 0 for an actual image.

        '''
        w = width / 2

        if samples > 0:
            ext = samples / 2 * sample_spacing
            x, y = np.linspace(-ext, ext, samples), np.linspace(-ext, ext, samples)
            arr = np.zeros((samples, samples))
        else:
            arr, x, y = None, np.zeros(2), np.zeros(2)

        # paint in the slit
        if orientation.lower() in ('v', 'vert', 'vertical'):
            if samples > 0:
                arr[:, abs(x) < w] = 1
            self.orientation = 'Vertical'
            self.width_x = width
            self.width_y = 0
        elif orientation.lower() in ('h', 'horiz', 'horizontal'):
            if samples > 0:
                arr[abs(y) < w, :] = 1
            self.width_x = 0
            self.width_y = width
            self.orientation = 'Horizontal'
        elif orientation.lower() in ('b', 'both', 'c', 'crossed'):
            if samples > 0:
                arr[abs(y) < w, :] = 1
                arr[:, abs(x) < w] = 1
            self.orientation = 'Crossed'
            self.width_x, self.width_y = width, width

        super().__init__(arr, x, y, has_analytic_ft=True)

    def analytic_ft(self, unit_x, unit_y):
        '''Analytic fourier transform of a slit.

        Parameters
        ----------
        unit_x : `numpy.ndarray`
            sample points in x frequency axis
        unit_y : `numpy.ndarray`
            sample points in y frequency axis

        Returns
        -------
        `numpy.ndarray`
            2D numpy array containing the analytic fourier transform

        '''
        xq, yq = np.meshgrid(unit_x, unit_y)
        if self.width_x > 0 and self.width_y > 0:
            return (m.sinc(xq * self.width_x) +
                    m.sinc(yq * self.width_y)).astype(config.precision)
        elif self.width_x > 0 and self.width_y is 0:
            return m.sinc(xq * self.width_x).astype(config.precision)
        else:
            return m.sinc(yq * self.width_y).astype(config.precision)


class Pinhole(Convolvable):
    '''Representation of a pinhole.

    Attributes
    ----------
    width : `float`
        diameter of the pinhole

    '''
    def __init__(self, width, sample_spacing=0.025, samples=0):
        '''Produce a Pinhole.

        Parameters
        ----------
        width : `float`
            the width of the pinhole
        sample_spacing : `float`
            spacing of samples in the synthetic image
        samples : `int`
            number of samples per dimension in the synthetic image

        Notes
        -----
        Default of 0 samples allows quick creation for convolutions without
        generating the image; use samples > 0 for an actual image.

        '''
        self.width = width

        # produce coordinate arrays
        if samples > 0:
            ext = samples / 2 * sample_spacing
            x, y = np.linspace(-ext, ext, samples), np.linspace(-ext, ext, samples)
            xv, yv = np.meshgrid(x, y)
            w = width / 2
            # paint a circle on a black background
            arr = np.zeros((samples, samples))
            arr[m.sqrt(xv**2 + yv**2) < w] = 1
        else:
            arr, x, y = None, np.zeros(2), np.zeros(2)

        super().__init__(data=arr, unit_x=x, unit_y=y, has_analytic_ft=True)

    def analytic_ft(self, unit_x, unit_y):
        '''Analytic fourier transform of a slit.

        Parameters
        ----------
        unit_x : `numpy.ndarray`
            sample points in x frequency axis
        unit_y : `numpy.ndarray`
            sample points in y frequency axis

        Returns
        -------
        `numpy.ndarray`
            2D numpy array containing the analytic fourier transform

        '''
        xq, yq = np.meshgrid(unit_x, unit_y)
        rho = m.sqrt(xq**2 + yq**2) / self.width / 1e3
        return m.jinc(rho).astype(config.precision)


class SiemensStar(Convolvable):
    '''Representation of a Siemen's star object.

    Attributes
    ----------
    num_spokes : `int`
        number of spokes present in the star

    '''
    def __init__(self, num_spokes, sinusoidal=True, background='black', sample_spacing=2, samples=384):
        '''Produces a Siemen's Star.

        Parameters
        ----------
        num_spokes : `int`
            number of spokes in the star.
        sinusoidal : `bool`
            if True, generates a sinusoidal Siemen' star, else, generates a bar/block siemen's star
        background : 'string', {'black', 'white'}
            background color
        sample_spacing : `float`
            spacing of samples, in microns
        samples : `int`
            number of samples per dimension in the synthetic image

        Raises
        ------
        ValueError
            background other than black or white

        '''
        relative_width = 0.9
        self.num_spokes = num_spokes

        # generate a coordinate grid
        x = np.linspace(-1, 1, samples)
        y = np.linspace(-1, 1, samples)
        xx, yy = np.meshgrid(x, y)
        rv, pv = cart_to_polar(xx, yy)

        # generate the siemen's star as a (rho,phi) polynomial
        arr = m.cos(num_spokes / 2 * pv)

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

        super().__init__(data=arr, unit_x=x, unit_y=y, has_analytic_ft=False)


class TiltedSquare(Convolvable):
    '''Describes a tilted square for e.g. slanted-edge '''

    def __init__(self, angle=8, background='white', sample_spacing=2, samples=384):
        '''Create a new TitledSquare instance.

        Parameters
        ----------
        angle : `float`
            angle in degrees to tilt w.r.t. the x axis
        background : `string`
            white or black; the square will be the opposite color of the background
        sample_spacing : `float`
            spacing of samples
        samples : `int`
            number of samples

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
        xp = xx * m.cos(angle) - yy * m.sin(angle)
        yp = xx * m.sin(angle) + yy * m.cos(angle)
        mask = (abs(xp) < radius) * (abs(yp) < radius)
        arr[mask] = fill_with
        super().__init__(data=arr, unit_x=x, unit_y=y, has_analytic_ft=False)
