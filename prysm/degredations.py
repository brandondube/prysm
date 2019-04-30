"""Degredations in the image chain."""


from .mathops import engine as e
from .coordinates import cart_to_polar, polar_to_cart
from .convolution import Convolvable


class Smear(Convolvable):
    """Smear (motion blur)."""
    def __init__(self, width, angle=0):
        super().__init__(None, None, None, True)
        self.width = width
        self.angle = angle

    def analytic_ft(self, x, y):
        xq, yq = e.meshgrid(x, y)
        rho, phi = cart_to_polar(xq, yq)
        phi += e.radians(self.angle)
        xq, yq = polar_to_cart(rho, phi)
        return e.sinc(xq * self.width)
