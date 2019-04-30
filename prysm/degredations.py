"""Degredations in the image chain."""

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
        xq, yq = e.meshgrid(x, y)
        if self.angle != 0:
            rho, phi = cart_to_polar(xq, yq)
            phi += e.radians(self.angle)
            xq, yq = polar_to_cart(rho, phi)

        return e.sinc(xq * self.width)
