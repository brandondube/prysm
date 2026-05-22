"""Sample optimization problems.

The functions in this module return objective and gradient pairs, matching the
callable interface accepted by prysm.x.optym optimizers.  The Problem classes
provide analytic objective, gradient, Hessian, and Hessian-vector product hooks
for optimizers that can use richer derivative information.

"""
from prysm.mathops import np

from .problem import Problem


__all__ = (
    'SphereProblem',
    'RosenbrockProblem',
    'RastriginProblem',
    'HimmelblauProblem',
    'sphere',
    'rosenbrock',
    'rastrigin',
    'himmelblau',
)


def _as_float_array(x):
    x = np.asarray(x)
    if not np.issubdtype(x.dtype, np.floating):
        x = x.astype(float)
    return x


class SphereProblem(Problem):
    """Sphere optimization problem.

    The global minimum is f(0) = 0.

    """

    has_f = True
    has_g = True
    has_fg = True
    has_h = True
    has_hvp = True

    def _f(self, x):
        x = _as_float_array(x)
        return (x * x).sum()

    def _g(self, x):
        x = _as_float_array(x)
        return 2 * x

    def _fg(self, x):
        x = _as_float_array(x)
        return (x * x).sum(), 2 * x

    def _h(self, x):
        x = _as_float_array(x)
        return 2 * np.eye(x.size, dtype=x.dtype)

    def _hvp(self, x, v):
        x = _as_float_array(x)
        v = _as_float_array(v)
        if v.size != x.size:
            raise ValueError('v must have the same number of elements as x')
        return 2 * v


class RosenbrockProblem(Problem):
    """Rosenbrock optimization problem.

    The global minimum is f([1, ..., 1]) = 0.

    """

    has_f = True
    has_g = True
    has_fg = True
    has_h = True
    has_hvp = True

    def _validate_x(self, x):
        x = _as_float_array(x)
        if x.size < 2:
            raise ValueError('rosenbrock requires at least two variables')
        return x

    def _f(self, x):
        x = self._validate_x(x)
        xf = x.ravel()
        diff = xf[1:] - xf[:-1] * xf[:-1]
        offset = 1 - xf[:-1]
        return (100 * diff * diff + offset * offset).sum()

    def _g(self, x):
        x = self._validate_x(x)
        return self._gradient(x)

    def _fg(self, x):
        x = self._validate_x(x)
        xf = x.ravel()
        diff = xf[1:] - xf[:-1] * xf[:-1]
        offset = 1 - xf[:-1]
        f = (100 * diff * diff + offset * offset).sum()
        return f, self._gradient_from_terms(x, diff, offset)

    def _h(self, x):
        x = self._validate_x(x)
        xf = x.ravel()
        n = xf.size
        hess = np.zeros((n, n), dtype=x.dtype)
        idx = np.arange(n)
        hess[0, 0] = 1200 * xf[0] * xf[0] - 400 * xf[1] + 2
        if n > 2:
            hess[idx[1:-1], idx[1:-1]] = (
                1200 * xf[1:-1] * xf[1:-1] - 400 * xf[2:] + 202
            )
        hess[-1, -1] = 200
        offdiag = -400 * xf[:-1]
        offdiag_idx = np.arange(n - 1)
        hess[offdiag_idx, offdiag_idx + 1] = offdiag
        hess[offdiag_idx + 1, offdiag_idx] = offdiag
        return hess

    def _hvp(self, x, v):
        x = self._validate_x(x)
        v = _as_float_array(v)
        if v.size != x.size:
            raise ValueError('v must have the same number of elements as x')

        xf = x.ravel()
        vf = v.ravel()
        hv = np.empty_like(vf)
        hv[0] = (
            (1200 * xf[0] * xf[0] - 400 * xf[1] + 2) * vf[0]
            - 400 * xf[0] * vf[1]
        )
        if xf.size > 2:
            hv[1:-1] = (
                -400 * xf[:-2] * vf[:-2]
                + (1200 * xf[1:-1] * xf[1:-1] - 400 * xf[2:] + 202) * vf[1:-1]
                - 400 * xf[1:-1] * vf[2:]
            )
        hv[-1] = -400 * xf[-2] * vf[-2] + 200 * vf[-1]
        return hv.reshape(v.shape)

    def _gradient(self, x):
        xf = x.ravel()
        diff = xf[1:] - xf[:-1] * xf[:-1]
        offset = 1 - xf[:-1]
        return self._gradient_from_terms(x, diff, offset)

    def _gradient_from_terms(self, x, diff, offset):
        xf = x.ravel()
        g = np.zeros_like(xf)
        g[:-1] += -400 * xf[:-1] * diff - 2 * offset
        g[1:] += 200 * diff
        return g.reshape(x.shape)


class RastriginProblem(Problem):
    """Rastrigin optimization problem.

    The global minimum is f(0) = 0.

    """

    has_f = True
    has_g = True
    has_fg = True
    has_h = True
    has_hvp = True

    def _f(self, x):
        x = _as_float_array(x)
        arg = 2 * np.pi * x
        return 10 * x.size + (x * x - 10 * np.cos(arg)).sum()

    def _g(self, x):
        x = _as_float_array(x)
        arg = 2 * np.pi * x
        return 2 * x + 20 * np.pi * np.sin(arg)

    def _fg(self, x):
        x = _as_float_array(x)
        arg = 2 * np.pi * x
        f = 10 * x.size + (x * x - 10 * np.cos(arg)).sum()
        g = 2 * x + 20 * np.pi * np.sin(arg)
        return f, g

    def _h(self, x):
        x = _as_float_array(x)
        diag = self._hessian_diagonal(x).ravel()
        return np.diag(diag)

    def _hvp(self, x, v):
        x = _as_float_array(x)
        v = _as_float_array(v)
        if v.size != x.size:
            raise ValueError('v must have the same number of elements as x')
        return (self._hessian_diagonal(x).ravel() * v.ravel()).reshape(v.shape)

    def _hessian_diagonal(self, x):
        return 2 + 40 * np.pi * np.pi * np.cos(2 * np.pi * x)


class HimmelblauProblem(Problem):
    """Himmelblau optimization problem.

    One global minimum is f([3, 2]) = 0.

    """

    has_f = True
    has_g = True
    has_fg = True
    has_h = True
    has_hvp = True

    def _validate_x(self, x):
        x = _as_float_array(x)
        if x.size != 2:
            raise ValueError('himmelblau requires exactly two variables')
        return x

    def _f(self, x):
        x = self._validate_x(x)
        x0, x1 = x.ravel()
        a = x0 * x0 + x1 - 11
        b = x0 + x1 * x1 - 7
        return a * a + b * b

    def _g(self, x):
        x = self._validate_x(x)
        return self._gradient(x)

    def _fg(self, x):
        x = self._validate_x(x)
        x0, x1 = x.ravel()
        a = x0 * x0 + x1 - 11
        b = x0 + x1 * x1 - 7
        f = a * a + b * b
        return f, self._gradient_from_terms(x, x0, x1, a, b)

    def _h(self, x):
        x = self._validate_x(x)
        x0, x1 = x.ravel()
        cross = 4 * (x0 + x1)
        return np.array([
            [12 * x0 * x0 + 4 * x1 - 42, cross],
            [cross, 4 * x0 + 12 * x1 * x1 - 26],
        ], dtype=x.dtype)

    def _hvp(self, x, v):
        x = self._validate_x(x)
        v = _as_float_array(v)
        if v.size != x.size:
            raise ValueError('v must have the same number of elements as x')

        x0, x1 = x.ravel()
        vf = v.ravel()
        cross = 4 * (x0 + x1)
        hv = np.empty_like(vf)
        hv[0] = (12 * x0 * x0 + 4 * x1 - 42) * vf[0] + cross * vf[1]
        hv[1] = cross * vf[0] + (4 * x0 + 12 * x1 * x1 - 26) * vf[1]
        return hv.reshape(v.shape)

    def _gradient(self, x):
        x0, x1 = x.ravel()
        a = x0 * x0 + x1 - 11
        b = x0 + x1 * x1 - 7
        return self._gradient_from_terms(x, x0, x1, a, b)

    def _gradient_from_terms(self, x, x0, x1, a, b):
        g = np.empty(x.size, dtype=x.dtype)
        g[0] = 4 * x0 * a + 2 * b
        g[1] = 2 * a + 4 * x1 * b
        return g.reshape(x.shape)


_SPHERE = SphereProblem()
_ROSENBROCK = RosenbrockProblem()
_RASTRIGIN = RastriginProblem()
_HIMMELBLAU = HimmelblauProblem()


def sphere(x):
    """Evaluate the sphere function.

    The global minimum is f(0) = 0.

    Parameters
    ----------
    x : ndarray
        optimization variables

    Returns
    -------
    float, ndarray
        objective value and gradient

    """
    return _SPHERE.fg(x)


def rosenbrock(x):
    """Evaluate the Rosenbrock function.

    The global minimum is f([1, ..., 1]) = 0.

    Parameters
    ----------
    x : ndarray
        optimization variables, at least two elements

    Returns
    -------
    float, ndarray
        objective value and gradient

    """
    return _ROSENBROCK.fg(x)


def rastrigin(x):
    """Evaluate the Rastrigin function.

    The global minimum is f(0) = 0.

    Parameters
    ----------
    x : ndarray
        optimization variables

    Returns
    -------
    float, ndarray
        objective value and gradient

    """
    return _RASTRIGIN.fg(x)


def himmelblau(x):
    """Evaluate Himmelblau's function.

    One global minimum is f([3, 2]) = 0.

    Parameters
    ----------
    x : ndarray
        optimization variables, exactly two elements

    Returns
    -------
    float, ndarray
        objective value and gradient

    """
    return _HIMMELBLAU.fg(x)
