"""Sample optimization problems.

The functions in this module return objective and gradient pairs, matching the
callable interface accepted by prysm.x.optym optimizers.

"""
from prysm.mathops import np


def _as_float_array(x):
    x = np.asarray(x)
    if not np.issubdtype(x.dtype, np.floating):
        x = x.astype(float)
    return x


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
    x = _as_float_array(x)
    return (x * x).sum(), 2 * x


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
    x = _as_float_array(x)
    if x.size < 2:
        raise ValueError('rosenbrock requires at least two variables')

    xf = x.ravel()
    diff = xf[1:] - xf[:-1] * xf[:-1]
    offset = 1 - xf[:-1]
    f = (100 * diff * diff + offset * offset).sum()

    g = np.zeros_like(xf)
    g[:-1] += -400 * xf[:-1] * diff - 2 * offset
    g[1:] += 200 * diff
    return f, g.reshape(x.shape)


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
    x = _as_float_array(x)
    arg = 2 * np.pi * x
    f = 10 * x.size + (x * x - 10 * np.cos(arg)).sum()
    g = 2 * x + 20 * np.pi * np.sin(arg)
    return f, g


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
    x = _as_float_array(x)
    if x.size != 2:
        raise ValueError('himmelblau requires exactly two variables')

    xf = x.ravel()
    x0, x1 = xf
    a = x0 * x0 + x1 - 11
    b = x0 + x1 * x1 - 7
    f = a * a + b * b

    g = np.empty_like(xf)
    g[0] = 4 * x0 * a + 2 * b
    g[1] = 2 * a + 4 * x1 * b
    return f, g.reshape(x.shape)
