"""Optimization problem protocol and callable adapter."""
from prysm.mathops import np  # noqa: F401  (kept for downstream subclasses)


_FD_METHODS = ('forward', 'central')


class Problem:
    """Base class for optimization problems.

    Subclasses set has_* flags for the private hooks they implement.  Missing
    derivatives are finite-differenced when lower-order information is present.

    Parameters
    ----------
    fd_method : {'forward', 'central'}, optional
        Finite difference stencil used for fallback derivatives.  The class
        default is 'central'.
    fd_step : float or ndarray, optional
        Relative finite difference step.  If omitted, a dtype-dependent default
        is selected from the active finite difference method.

    Notes
    -----
    f, g, fg, h, and hvp are the public API.

    """

    has_f = False
    has_g = False
    has_fg = False
    has_h = False
    has_hvp = False

    fd_method = 'central'
    fd_step = None

    def __init__(self, fd_method=None, fd_step=None):
        if fd_method is not None:
            self.fd_method = fd_method
        if fd_step is not None:
            self.fd_step = fd_step
        self._validate_fd_method()

    def f(self, x):
        """Evaluate the scalar objective."""
        if self.has_f:
            return self._f(x)
        if self.has_fg:
            return self._fg(x)[0]
        raise NotImplementedError('Problem needs _f(x) or _fg(x)')

    def g(self, x):
        """Evaluate the objective gradient."""
        if self.has_g:
            return self._g(x)
        if self.has_fg:
            return self._fg(x)[1]
        if self.has_f:
            return self._finite_difference_g(x)
        raise NotImplementedError('Problem needs _g(x), _fg(x), or _f(x)')

    def fg(self, x):
        """Evaluate objective and gradient."""
        if self.has_fg:
            return self._fg(x)
        return self.f(x), self.g(x)

    def h(self, x):
        """Evaluate the dense Hessian."""
        if self.has_h:
            return self._h(x)
        return self._finite_difference_h(x)

    def hvp(self, x, v):
        """Evaluate the Hessian-vector product H(x) @ v."""
        if self.has_hvp:
            return self._hvp(x, v)
        if self.has_h:
            return self.h(x) @ v
        return self._finite_difference_hvp(x, v)

    def _validate_fd_method(self):
        if self.fd_method not in _FD_METHODS:
            raise ValueError(f'fd_method must be one of {_FD_METHODS}; got {self.fd_method!r}')

    def _as_float_array(self, x):
        x = np.asarray(x)
        if not np.issubdtype(x.dtype, np.floating):
            x = x.astype(float)
        return x

    def _fd_exponent(self):
        if self.fd_method == 'forward':
            return 0.5
        return 1 / 3

    def _fd_steps(self, x):
        base = self.fd_step
        if base is None:
            base = np.finfo(x.dtype).eps ** self._fd_exponent()
        return base * np.maximum(1, np.abs(x))

    def _fd_direction_step(self, x, v):
        base = self.fd_step
        if base is None:
            base = np.finfo(x.dtype).eps ** self._fd_exponent()
        v_norm = np.linalg.norm(v)
        if v_norm == 0:
            return 0
        return base * max(1, np.linalg.norm(x)) / v_norm

    def _finite_difference_g(self, x):
        x = self._as_float_array(x)
        g = np.empty_like(x)
        steps = self._fd_steps(x)

        xf = x.ravel()
        gf = g.ravel()
        hf = steps.ravel()

        if self.fd_method == 'forward':
            f0 = self.f(x)
            for j in range(x.size):
                xp = x.copy()
                xp.ravel()[j] = xf[j] + hf[j]
                gf[j] = (self.f(xp) - f0) / hf[j]
        else:
            for j in range(x.size):
                xp = x.copy()
                xm = x.copy()
                xp.ravel()[j] = xf[j] + hf[j]
                xm.ravel()[j] = xf[j] - hf[j]
                gf[j] = (self.f(xp) - self.f(xm)) / (2 * hf[j])

        return g

    def _finite_difference_h(self, x):
        x = self._as_float_array(x)
        n = x.size
        hess = np.empty((n, n), dtype=x.dtype)
        steps = self._fd_steps(x)

        xf = x.ravel()
        hf = steps.ravel()

        if self.fd_method == 'forward':
            g0 = self.g(x).ravel()
            for j in range(n):
                xp = x.copy()
                xp.ravel()[j] = xf[j] + hf[j]
                hess[:, j] = (self.g(xp).ravel() - g0) / hf[j]
        else:
            for j in range(n):
                xp = x.copy()
                xm = x.copy()
                xp.ravel()[j] = xf[j] + hf[j]
                xm.ravel()[j] = xf[j] - hf[j]
                hess[:, j] = (self.g(xp).ravel() - self.g(xm).ravel()) / (2 * hf[j])

        return hess

    def _finite_difference_hvp(self, x, v):
        x = self._as_float_array(x)
        v = self._as_float_array(v)
        step = self._fd_direction_step(x, v)
        if step == 0:
            return np.zeros_like(v)

        if self.fd_method == 'forward':
            return (self.g(x + step * v) - self.g(x)) / step
        return (self.g(x + step * v) - self.g(x - step * v)) / (2 * step)


class _CallableProblem(Problem):
    """Wrap fg(x) -> (f, g) as a Problem.

    The most recent x identity is cached for f-then-g reuse.
    """

    has_fg = True

    def __init__(self, fg, fd_method=None, fd_step=None):
        super().__init__(fd_method=fd_method, fd_step=fd_step)
        self._callable_fg = fg
        self._last_x = None
        self._last_fg = None

    def _fg(self, x):
        if self._last_x is x:
            return self._last_fg
        fg = self._callable_fg(x)
        self._last_x = x
        self._last_fg = fg
        return fg


def as_problem(obj):
    """Return a Problem-like object for obj.

    Parameters
    ----------
    obj : Problem-like or callable
        Problem instance, object exposing fg(x), or callable fg(x) -> (f, g).

    Returns
    -------
    Problem or Problem-like
        obj itself or a _CallableProblem wrapper.

    Raises
    ------
    TypeError
        if obj is neither a Problem-shaped object nor a callable.

    """
    if isinstance(obj, Problem):
        return obj
    if hasattr(obj, 'fg'):
        return obj
    if callable(obj):
        return _CallableProblem(obj)
    raise TypeError(f'expected a Problem-shaped object or a callable; got {type(obj).__name__}')  # NOQA - length
