"""Degredations in the image chain."""

from .conf import config
from .mathops import engine as e
from .coordinates import cart_to_polar, polar_to_cart
from .convolution import Convolvable


class Smear(Convolvable):
    """Smear (motion blur)."""
    def __init__(self, width, angle=0):
        """Create a new Smear model.

        Parameters
        ----------
        width : `float`
            width of the blur in microns
        angle : `float`
            clockwise angle of the blur with respect to the x axis in degrees.

        """
        super().__init__(None, None, None, True)
        self.width = width
        self.angle = angle

    def analytic_ft(self, x, y):
        """Analytic FT of the smear.

        Parameters
        ----------
        x : `numpy.ndarray`
            x Cartesian spatial frequency, cy/um
        y : `numpy.ndarray`
            y Cartesian spatial frequency, cy/um

        Returns
        -------
        `numpy.ndarray`
            analytical FT of the smear.

        """
        if self.angle != 0:
            rho, phi = cart_to_polar(x, y)
            phi += e.radians(self.angle)
            x, y = polar_to_cart(rho, phi)

        return e.sinc(x * self.width)


class Jitter(Convolvable):
    """Jitter (high frequency motion)."""
    def __init__(self, scale, sample_spacing=None, samples=None):
        """Create a new Jitter instance.

        Parameters
        ----------
        scale : `float`
            scale of the jitter, units of microns
        sample_spacing : `float`, optional
            center-to-center sample spacing, units of microns
        samples : `int`, optional
            number of samples in X and Y

        """
        self.scale = scale
        if samples is not None:
            ext = (samples - 1) * sample_spacing / 2
            x = e.arange(-ext, ext, sample_spacing, dtype=config.precision)
            y = e.arange(-ext, ext, sample_spacing, dtype=config.precision)

            coef = 1 / (scale * e.sqrt(2 * e.pi))
            xx, yy = e.meshgrid(x, y)
            rho, _ = cart_to_polar(xx, yy)
            kernel = rho ** 2 / (2 * scale ** 2)
            z = coef * e.exp(-kernel)
        else:
            x, y, z = None, None, None

        super().__init__(data=z, x=x, y=y, has_analytic_ft=True)

    def analytic_ft(self, x, y):
        """Analytic FT of jitter.

        Parameters
        ----------
        x : `numpy.ndarray`
            x Cartesian spatial frequency, units of cy/um
        y : `numpy.ndarray`
            y Cartesian spatial frequency, units of cy/um

        Returns
        -------
        `numpy.ndarray`
            value of analytic FT

        """
        rho, _ = cart_to_polar(x, y)
        kernel = e.pi * self.scale * rho
        return e.exp(-2 * kernel**2)
