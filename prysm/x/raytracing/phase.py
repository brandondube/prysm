"""Diffractive phase-function providers for raytracing surfaces."""

from prysm.conf import config
from prysm.mathops import np

from .sags import fd_step


class PhaseFunction:
    """Base class for wavelength-independent diffractive phase functions."""

    finite_difference_step = None

    def phase(self, x, y):
        """Diffractive phase in waves at local coordinates x, y."""
        raise NotImplementedError

    def _fd_step(self, *arrs):
        """Central-difference step, scaled to the coordinate magnitude."""
        return fd_step(self.finite_difference_step, *arrs)

    def phase_and_gradient(self, x, y):
        """Phase and its in-plane gradient (phase, dphase/dx, dphase/dy).

        The base implementation central-differences phase.

        """
        x = np.asarray(x)
        y = np.asarray(y)
        h = self._fd_step(x, y)
        ph = self.phase(x, y)
        gx = (self.phase(x + h, y) - self.phase(x - h, y)) / (2.0 * h)
        gy = (self.phase(x, y + h) - self.phase(x, y - h)) / (2.0 * h)
        return ph, gx, gy

    def phase_hessian(self, x, y):
        """Phase Hessian (phase_xx, phase_xy, phase_yy) at x, y.

        The base implementation central-differences the gradient.

        """
        x = np.asarray(x)
        y = np.asarray(y)
        h = self._fd_step(x, y)
        _, gxxp, _ = self.phase_and_gradient(x + h, y)
        _, gxxm, _ = self.phase_and_gradient(x - h, y)
        _, gxyp, gyyp = self.phase_and_gradient(x, y + h)
        _, gxym, gyym = self.phase_and_gradient(x, y - h)
        pxx = (gxxp - gxxm) / (2.0 * h)
        pyy = (gyyp - gyym) / (2.0 * h)
        pxy = (gxyp - gxym) / (2.0 * h)
        return pxx, pxy, pyy


class LinearGrating(PhaseFunction):
    """Constant-gradient (linear) diffraction grating phase.

    Parameters
    ----------
    period : float
        grating period, same length units as the ray coordinates.
    g_vec : array_like
        grating-vector direction; only its first two (x, y) components are used.
    order : int or float
        diffraction order; scales the phase (and thus the bend) linearly.

    """

    def __init__(self, period, g_vec=(1.0, 0.0), order=1):
        g = np.atleast_1d(np.asarray(g_vec, dtype=float)).ravel()
        gx = float(g[0])
        gy = float(g[1]) if g.size > 1 else 0.0
        self.period = float(period)
        self.order = order
        self.g_vec = (gx, gy)
        self._kx = order * gx / self.period
        self._ky = order * gy / self.period

    def phase(self, x, y):
        """Linear phase order * (g . (x, y)) / period."""
        return self._kx * x + self._ky * y

    def phase_and_gradient(self, x, y):
        """Phase and its constant in-plane gradient."""
        x = np.asarray(x, dtype=config.precision)
        ph = self._kx * x + self._ky * np.asarray(y, dtype=config.precision)
        gx = np.full(x.shape, self._kx, dtype=config.precision)
        gy = np.full(x.shape, self._ky, dtype=config.precision)
        return ph, gx, gy

    def phase_hessian(self, x, y):
        """Zero Hessian -- the gradient is constant."""
        z = np.zeros_like(np.asarray(x, dtype=config.precision))
        return z, z, z

    def __repr__(self):
        return (f'LinearGrating(period={self.period!r}, g_vec={self.g_vec!r}, '
                f'order={self.order!r})')


class CallablePhase(PhaseFunction):
    """Phase function backed by user-supplied callables.

    Only phase is required; gradient and Hessian default to finite differences.
    """

    def __init__(self, phase, phase_and_gradient=None, phase_hessian=None):
        if phase is None:
            raise TypeError('CallablePhase requires a phase callable')
        self._phase = phase
        self._pag = phase_and_gradient
        self._phess = phase_hessian

    def phase(self, x, y):
        """Evaluate the wrapped phase callable."""
        return self._phase(x, y)

    def phase_and_gradient(self, x, y):
        """Phase and gradient from the wrapped callable or finite differences."""
        if self._pag is None:
            return super().phase_and_gradient(x, y)
        return self._pag(x, y)

    def phase_hessian(self, x, y):
        """Hessian from the wrapped callable or finite differences."""
        if self._phess is None:
            return super().phase_hessian(x, y)
        return self._phess(x, y)


def as_phase_function(value):
    """Coerce a grating spec to a PhaseFunction or None."""
    if value is None or isinstance(value, PhaseFunction):
        return value
    try:
        period, g_vec, order = value
    except (TypeError, ValueError):
        raise TypeError(
            'grating must be a PhaseFunction, a (period, g_vec, order) tuple, '
            f'or None; got {value!r}')
    return LinearGrating(period, g_vec, order)


__all__ = [
    'PhaseFunction',
    'LinearGrating',
    'CallablePhase',
    'as_phase_function',
]
