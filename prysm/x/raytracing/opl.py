"""Optical-path-length functions attached to raytracing surfaces."""

from prysm.conf import config
from prysm.mathops import np

from .sags import fd_step


class OPLFunc:
    """Base class for wavelength-aware optical-path modifiers.

    Local coordinates are millimeters, wavelength is microns, and returned OPL
    is millimeters.  Spatial gradients therefore have units mm/mm and Hessians
    have units 1/mm.
    """

    finite_difference_step = None

    def opl(self, x, y, wavelength):
        """Optical path length in millimeters."""
        raise NotImplementedError

    def _fd_step(self, *arrs):
        """Central-difference step scaled to the coordinate magnitude."""
        return fd_step(self.finite_difference_step, *arrs)

    def opl_and_gradient(self, x, y, wavelength):
        """Return OPL and its local x/y spatial gradient."""
        x = np.asarray(x)
        y = np.asarray(y)
        h = self._fd_step(x, y)
        opl = self.opl(x, y, wavelength)
        gx = (self.opl(x + h, y, wavelength)
              - self.opl(x - h, y, wavelength)) / (2.0 * h)
        gy = (self.opl(x, y + h, wavelength)
              - self.opl(x, y - h, wavelength)) / (2.0 * h)
        return opl, gx, gy

    def opl_hessian(self, x, y, wavelength):
        """Return local (OPL_xx, OPL_xy, OPL_yy)."""
        x = np.asarray(x)
        y = np.asarray(y)
        h = self._fd_step(x, y)
        _, gxxp, _ = self.opl_and_gradient(x + h, y, wavelength)
        _, gxxm, _ = self.opl_and_gradient(x - h, y, wavelength)
        _, gxyp, gyyp = self.opl_and_gradient(x, y + h, wavelength)
        _, gxym, gyym = self.opl_and_gradient(x, y - h, wavelength)
        pxx = (gxxp - gxxm) / (2.0 * h)
        pyy = (gyyp - gyym) / (2.0 * h)
        pxy = (gxyp - gxym) / (2.0 * h)
        return pxx, pxy, pyy


class LinearGrating(OPLFunc):
    """Ideal linear grating expressed as a wavelength-dependent OPL ramp.

    Parameters
    ----------
    period : float
        Grating period in millimeters.
    g_vec : array_like
        In-plane grating-vector direction.  The first two components are used.
    order : int or float
        Diffracted order.
    """

    def __init__(self, period, g_vec=(1.0, 0.0), order=1):
        self._period = None
        self._order = None
        self._g_vec = None
        self.period = period
        self.order = order
        self.g_vec = g_vec

    @property
    def period(self):
        return self._period

    @period.setter
    def period(self, value):
        value = float(value)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError('grating period must be finite and positive')
        self._period = value

    @property
    def order(self):
        return self._order

    @order.setter
    def order(self, value):
        value = float(value)
        if not np.isfinite(value):
            raise ValueError('grating order must be finite')
        self._order = value

    @property
    def g_vec(self):
        return self._g_vec

    @g_vec.setter
    def g_vec(self, value):
        g = np.atleast_1d(np.asarray(value, dtype=float)).ravel()
        if g.size == 0:
            raise ValueError('g_vec must contain at least one component')
        gx = float(g[0])
        gy = float(g[1]) if g.size > 1 else 0.0
        if not np.isfinite(gx) or not np.isfinite(gy):
            raise ValueError('g_vec components must be finite')
        self._g_vec = (gx, gy)

    def _gradient(self, wavelength):
        wavelength_mm = float(wavelength) * 1e-3
        scale = self.order * wavelength_mm / self.period
        return scale * self.g_vec[0], scale * self.g_vec[1]

    def opl(self, x, y, wavelength):
        """Unwrapped grating OPL ramp in millimeters."""
        gx, gy = self._gradient(wavelength)
        return gx * x + gy * y

    def opl_and_gradient(self, x, y, wavelength):
        """Return the OPL ramp and its constant spatial gradient."""
        x = np.asarray(x, dtype=config.precision)
        y = np.asarray(y, dtype=config.precision)
        gx, gy = self._gradient(wavelength)
        opl = gx * x + gy * y
        gx = np.full(x.shape, gx, dtype=config.precision)
        gy = np.full(x.shape, gy, dtype=config.precision)
        return opl, gx, gy

    def opl_hessian(self, x, y, wavelength):
        """The Hessian of a linear OPL ramp is zero."""
        z = np.zeros_like(np.asarray(x, dtype=config.precision))
        return z, z, z

    def __repr__(self):
        order = int(self.order) if self.order.is_integer() else self.order
        return (f'LinearGrating(period={self.period!r}, '
                f'g_vec={self.g_vec!r}, order={order!r})')


class CallableOPL(OPLFunc):
    """OPLFunc backed by wavelength-aware user callables."""

    def __init__(self, opl, opl_and_gradient=None, opl_hessian=None):
        if not callable(opl):
            raise TypeError('CallableOPL requires an OPL callable')
        self._opl = opl
        self._oag = opl_and_gradient
        self._ohess = opl_hessian

    def opl(self, x, y, wavelength):
        return self._opl(x, y, wavelength)

    def opl_and_gradient(self, x, y, wavelength):
        if self._oag is None:
            return super().opl_and_gradient(x, y, wavelength)
        return self._oag(x, y, wavelength)

    def opl_hessian(self, x, y, wavelength):
        if self._ohess is None:
            return super().opl_hessian(x, y, wavelength)
        return self._ohess(x, y, wavelength)


__all__ = ['OPLFunc', 'LinearGrating', 'CallableOPL']

