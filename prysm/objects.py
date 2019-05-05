"""Objects for image simulation with."""

from scipy.signal import chirp

from .conf import config
from .mathops import engine as e, jinc
from .convolution import Convolvable
from .coordinates import cart_to_polar, polar_to_cart


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
            x = e.arange(-ext, ext, sample_spacing, dtype=config.precision)
            y = e.arange(-ext, ext, sample_spacing, dtype=config.precision)
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
        if self.width_x > 0 and self.width_y > 0:
            return (e.sinc(x * self.width_x) +
                    e.sinc(y * self.width_y)).astype(config.precision)
        elif self.width_x > 0 and self.width_y == 0:
            return e.sinc(x * self.width_x).astype(config.precision)
        else:
            return e.sinc(y * self.width_y).astype(config.precision)


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
            x = e.arange(-ext, ext, sample_spacing, dtype=config.precision)
            y = e.arange(-ext, ext, sample_spacing, dtype=config.precision)
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
    def __init__(self, spokes, sinusoidal=True, radius=0.9, background='black', sample_spacing=2, samples=256):
        """Produce a Siemen's Star.

        Parameters
        ----------
        spokes : `int`
            number of spokes in the star.
        sinusoidal : `bool`
            if True, generates a sinusoidal Siemen' star, else, generates a bar/block siemen's star
        radius : `float`,
            radius of the star, relative to the array width (default 90%)
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
        self.spokes = spokes
        self.radius = radius

        # generate a coordinate grid
        x = e.linspace(-1, 1, samples, dtype=config.precision)
        y = e.linspace(-1, 1, samples, dtype=config.precision)
        xx, yy = e.meshgrid(x, y)
        rv, pv = cart_to_polar(xx, yy)
        ext = sample_spacing * (samples / 2)
        ux = e.arange(-ext, ext, sample_spacing, dtype=config.precision)
        uy = e.arange(-ext, ext, sample_spacing, dtype=config.precision)

        # generate the siemen's star as a (rho,phi) polynomial
        arr = e.cos(spokes / 2 * pv)

        if not sinusoidal:  # make binary
            arr[arr < 0] = -1
            arr[arr > 0] = 1

        # scale to (0,1) and clip into a disk
        arr = (arr + 1) / 2
        if background.lower() in ('b', 'black'):
            arr[rv > radius] = 0
        elif background.lower() in ('w', 'white'):
            arr[rv > radius] = 1
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
            arr = e.ones((samples, samples), dtype=config.precision)
            fill_with = 1 - contrast
        else:
            arr = e.zeros((samples, samples), dtype=config.precision)
            fill_with = 1

        ext = samples / 2 * sample_spacing
        radius = radius * ext * 2
        x = e.arange(-ext, ext, sample_spacing, dtype=config.precision)
        y = e.arange(-ext, ext, sample_spacing, dtype=config.precision)
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
        x = e.arange(-ext, ext, sample_spacing, dtype=config.precision)
        y = e.arange(-ext, ext, sample_spacing, dtype=config.precision)
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


class Grating(Convolvable):
    """A grating with a given ruling."""
    def __init__(self, period, angle=0, sinusoidal=False, sample_spacing=2, samples=256):
        """Create a new Grating object

        Parameters
        ----------
        period : `float`
            period of the grating in microns
        angle : `float`, optional
            clockwise angle of the grating w.r.t. the X axis, degrees
        sinusoidal : `bool`, optional
            if True, the grating is a sinusoid, else it has square edges
        sample_spacing : `float`, optional
            center-to-center sample spacing in microns
        samples : `int`, optional
            number of samples across the diameter of the array

        """
        self.period = period
        self.sinusoidal = sinusoidal

        ext = samples / 2 * sample_spacing
        x = e.arange(-ext, ext, sample_spacing, dtype=config.precision)
        y = e.arange(-ext, ext, sample_spacing, dtype=config.precision)
        xx, yy = e.meshgrid(x, y)
        if angle != 0:
            rho, phi = cart_to_polar(xx, yy)
            phi += e.radians(angle)
            xx, yy = polar_to_cart(rho, phi)

        data = e.cos(2 * e.pi / period * xx)
        if sinusoidal:
            data += 1
            data /= 2
        else:
            data[data > 0] = 1
            data[data < 0] = 0

        super().__init__(data=data, x=x, y=y, has_analytic_ft=False)


class GratingArray(Convolvable):
    """An array of gratings with given rulings."""
    def __init__(self, periods, angles=None, sinusoidal=False, sample_spacing=2, samples=256):
        # if angles not provided, angles are 0
        if angles is None:
            angles = [0] * len(periods)

        self.periods = periods
        self.angles = angles
        self.sinusoidal = sinusoidal

        # calculate the basic grid things are defined on
        ext = samples / 2 * sample_spacing
        x = e.arange(-ext, ext, sample_spacing, dtype=config.precision)
        y = e.arange(-ext, ext, sample_spacing, dtype=config.precision)
        xx, yy = e.meshgrid(x, y)
        xxx, yyy = xx, yy

        # compute the grid parameters; number of columns, number of samples per column
        squareness = e.sqrt(len(periods))
        ncols = int(e.ceil(squareness))
        samples_per_patch = int(e.floor(samples / ncols))
        low_idx_x = 0
        high_idx_x = samples_per_patch
        low_idx_y = 0
        high_idx_y = samples_per_patch
        curr_row = 0

        out = e.zeros(xx.shape)
        for idx, (period, angle) in enumerate(zip(periods, angles)):
            # if we're off at an off angle, adjust the coordinates
            if angle != 0:
                rho, phi = cart_to_polar(xxx, yyy)
                phi += e.radians(angle)
                xxx, yyy = polar_to_cart(rho, phi)

            # compute the sinusoid
            data = e.cos(2 * e.pi / period * xxx)

            # compute the indices to embed it into the final array;
            # every time the current column advances, advance the X coordinates
            sy = slice(low_idx_y, high_idx_y)
            sx = slice(low_idx_x, high_idx_x)
            out[sy, sx] += data[sy, sx]

            # advance the indices are needed
            if (idx > 0) & ((idx + 1) % ncols == 0):
                offset = samples_per_patch * curr_row
                low_idx_x = 0
                high_idx_x = samples_per_patch
                low_idx_y = samples_per_patch + offset
                high_idx_y = samples_per_patch * 2 + offset
                curr_row += 1
            else:
                low_idx_x += samples_per_patch
                high_idx_x += samples_per_patch

            xxx = xx

        if sinusoidal:
            out += 1
            out /= 2
        else:
            out[out > 0] = 1
            out[out < 0] = 0
        super().__init__(data=out, x=x, y=y, has_analytic_ft=False)


class Chirp(Convolvable):
    """A frequency chirp."""
    def __init__(self, p0, p1, angle=0, method='linear', binary=True, sample_spacing=2, samples=256):
        """Create a new Chirp instance.

        Parameters
        ----------
        p0 : `float`
            first period, units of microns
        p1 : `float`
            second period, units of microns
        angle : `float`
            clockwise angle between the X axis and the chirp, units of degrees
        method : `str`, optional, {'linear', 'quadratic', 'logarithmic', 'hyperbolic'}
            type of chirp, passed directly to scipy.signal.chirp
        binary : `bool`, optional
            if True, the chirp is a square bar target, not a sinusoidal target.
        sample_spacing : `float`, optional
            center-to-center spacing of samples in the array
        samples : `float`, optional
            number of samples

        """
        p0 *= 2
        p1 *= 2
        ext = samples / 2 * sample_spacing
        x = e.arange(-ext, ext, sample_spacing, dtype=config.precision)
        y = e.arange(-ext, ext, sample_spacing, dtype=config.precision)
        xx, yy = e.meshgrid(x, y)

        if angle != 0:
            rho, phi = cart_to_polar(xx, yy)
            phi += e.radians(angle)
            xx, yy = polar_to_cart(rho, phi)

        sig = chirp(xx, 1 / p1, ext, 1 / p0, method=method)
        if binary:
            sig[sig < 0] = 0
            sig[sig > 0] = 1
        else:
            sig = (sig + 1) / 2

        super().__init__(x=x, y=y, data=sig, has_analytic_ft=False)
