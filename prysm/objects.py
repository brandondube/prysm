"""Objects for image simulation with."""

from .conf import config
from .mathops import engine as e, jinc
from .convolution import Convolvable
from .coordinates import cart_to_polar


class Slit(Convolvable):
    """Representation of a slit or pair of slits."""

    def __init__(self, width, orientation='Vertical', sample_spacing=None, samples=0):
        """Create a new Slit instance.

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

        """
        w = width / 2

        if samples > 0:
            ext = samples / 2 * sample_spacing
            x, y = e.arange(-ext, ext, sample_spacing), e.arange(-ext, ext, sample_spacing)
            arr = e.zeros((samples, samples))
        else:
            arr, x, y = None, None, None

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

        super().__init__(data=arr, x=x, y=y, has_analytic_ft=True)

    def analytic_ft(self, x, y):
        """Analytic fourier transform of a slit.

        Parameters
        ----------
        x : `numpy.ndarray`
            sample points in x frequency axis
        y : `numpy.ndarray`
            sample points in y frequency axis

        Returns
        -------
        `numpy.ndarray`
            2D numpy array containing the analytic fourier transform

        """
        xq, yq = e.meshgrid(x, y)
        if self.width_x > 0 and self.width_y > 0:
            return (e.sinc(xq * self.width_x) +
                    e.sinc(yq * self.width_y)).astype(config.precision)
        elif self.width_x > 0 and self.width_y == 0:
            return e.sinc(xq * self.width_x).astype(config.precision)
        else:
            return e.sinc(yq * self.width_y).astype(config.precision)


class Pinhole(Convolvable):
    """Representation of a pinhole."""
    def __init__(self, width, sample_spacing=None, samples=0):
        """Create a Pinhole instance.

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

        """
        self.width = width

        # produce coordinate arrays
        if samples > 0:
            ext = samples / 2 * sample_spacing
            x, y = e.arange(-ext, ext, sample_spacing), e.arange(-ext, ext, sample_spacing)
            xv, yv = e.meshgrid(x, y)
            w = width / 2
            # paint a circle on a black background
            arr = e.zeros((samples, samples))
            arr[e.sqrt(xv**2 + yv**2) < w] = 1
        else:
            arr, x, y = None, None, None

        super().__init__(data=arr, x=x, y=y, has_analytic_ft=True)

    def analytic_ft(self, x, y):
        """Analytic fourier transform of a slit.

        Parameters
        ----------
        x : `numpy.ndarray`
            sample points in x frequency axis
        y : `numpy.ndarray`
            sample points in y frequency axis

        Returns
        -------
        `numpy.ndarray`
            2D numpy array containing the analytic fourier transform

        """
        xq, yq = e.meshgrid(x, y)

        # factor of pi corrects for jinc being modulo pi
        # factor of 2 converts radius to diameter
        rho = e.sqrt(xq**2 + yq**2) * self.width * 2 * e.pi
        return jinc(rho).astype(config.precision)


class SiemensStar(Convolvable):
    """Representation of a Siemen's star object."""
    def __init__(self, spokes, sinusoidal=True, background='black', sample_spacing=2, samples=256):
        """Produce a Siemen's Star.

        Parameters
        ----------
        spokes : `int`
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

        """
        relative_width = 0.9
        self.spokes = spokes

        # generate a coordinate grid
        x = e.linspace(-1, 1, samples)
        y = e.linspace(-1, 1, samples)
        xx, yy = e.meshgrid(x, y)
        rv, pv = cart_to_polar(xx, yy)
        ext = sample_spacing * (samples / 2)
        ux, uy = e.arange(-ext, ext, sample_spacing), e.arange(-ext, ext, sample_spacing)

        # generate the siemen's star as a (rho,phi) polynomial
        arr = e.cos(spokes / 2 * pv)

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

        super().__init__(data=arr, x=ux, y=uy, has_analytic_ft=False)


class TiltedSquare(Convolvable):
    """Represents a tilted square for e.g. slanted-edge MTF calculation."""
    def __init__(self, angle=4, background='white', sample_spacing=2, samples=256, radius=0.3, contrast=0.9):
        """Create a new TitledSquare instance.

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
        radius : `float`
            fractional
        contrast : `float`
            contrast, anywhere from 0 to 1

        """
        if background.lower() == 'white':
            arr = e.ones((samples, samples))
            fill_with = 1 - contrast
        else:
            arr = e.zeros((samples, samples))
            fill_with = 1

        ext = samples / 2 * sample_spacing
        radius = radius * ext * 2
        x = y = e.arange(-ext, ext, sample_spacing)
        xx, yy = e.meshgrid(x, y)

        # TODO: convert inline operation to use of rotation matrix
        angle = e.radians(angle)
        xp = xx * e.cos(angle) - yy * e.sin(angle)
        yp = xx * e.sin(angle) + yy * e.cos(angle)
        mask = (abs(xp) < radius) * (abs(yp) < radius)
        arr[mask] = fill_with
        super().__init__(data=arr, x=x, y=y, has_analytic_ft=False)


class SlantedEdge(Convolvable):
    """Representation of a slanted edge."""
    def __init__(self, angle=4, contrast=0.9, crossed=False, sample_spacing=2, samples=256):
        """Create a new TitledSquare instance.

        Parameters
        ----------
        angle : `float`
            angle in degrees to tilt w.r.t. the y axis
        contrast : `float`
            difference between minimum and maximum values in the image
        crossed : `bool`, optional
            whether to make a single edge (crossed=False) or pair of crossed edges (crossed=True)
            aka a "BMW target"
        sample_spacing : `float`
            spacing of samples
        samples : `int`
            number of samples

        """
        diff = (1 - contrast) / 2
        arr = e.full((samples, samples), 1 - diff)
        ext = samples / 2 * sample_spacing
        x = y = e.arange(-ext, ext, sample_spacing)
        xx, yy = e.meshgrid(x, y)

        angle = e.radians(angle)
        xp = xx * e.cos(angle) - yy * e.sin(angle)
        # yp = xx * e.sin(angle) + yy * e.cos(angle)  # do not need this
        mask = xp > 0  # single edge
        if crossed:
            mask = xp > 0  # set of 4 edges
            upperright = mask & e.rot90(mask)
            lowerleft = e.rot90(upperright, 2)
            mask = upperright | lowerleft

        arr[mask] = diff
        self.contrast = contrast
        self.black = diff
        self.white = 1 - diff
        super().__init__(data=arr, x=x, y=y, has_analytic_ft=False)
