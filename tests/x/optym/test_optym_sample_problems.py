"""Tests for prysm.x.optym.sample_problems."""
import numpy as np
import pytest

from prysm.x.optym import himmelblau, rastrigin, rosenbrock, sphere
from prysm.x.optym.sample_problems import (
    himmelblau as himmelblau_from_module,
    rastrigin as rastrigin_from_module,
    rosenbrock as rosenbrock_from_module,
    sphere as sphere_from_module,
)


def finite_difference_gradient(fg, x, step=1e-6):
    g = np.empty_like(x)
    xf = x.ravel()
    gf = g.ravel()
    for j in range(x.size):
        xp = x.copy()
        xm = x.copy()
        xp.ravel()[j] = xf[j] + step
        xm.ravel()[j] = xf[j] - step
        fp, _ = fg(xp)
        fm, _ = fg(xm)
        gf[j] = (fp - fm) / (2 * step)
    return g


@pytest.mark.parametrize(
    'func, x',
    [
        (sphere, np.array([1.5, -2.0, 0.25])),
        (rosenbrock, np.array([-1.2, 1.0, 0.5])),
        (rastrigin, np.array([0.25, -0.5, 1.25])),
        (himmelblau, np.array([-2.5, 3.0])),
    ],
)
def test_sample_problem_gradients_match_finite_difference(func, x):
    _, g = func(x)
    g_fd = finite_difference_gradient(func, x)
    np.testing.assert_allclose(g, g_fd, rtol=1e-6, atol=1e-6)


def test_sphere_minimum():
    x = np.zeros(4)
    f, g = sphere(x)
    np.testing.assert_allclose(f, 0)
    np.testing.assert_allclose(g, 0)


def test_rosenbrock_minimum():
    x = np.ones(5)
    f, g = rosenbrock(x)
    np.testing.assert_allclose(f, 0)
    np.testing.assert_allclose(g, 0)


def test_rastrigin_minimum():
    x = np.zeros((2, 3))
    f, g = rastrigin(x)
    np.testing.assert_allclose(f, 0)
    np.testing.assert_allclose(g, 0)


def test_himmelblau_minimum():
    x = np.array([3.0, 2.0])
    f, g = himmelblau(x)
    np.testing.assert_allclose(f, 0)
    np.testing.assert_allclose(g, 0)


def test_sample_problems_preserve_gradient_shape():
    for func, x in [
        (sphere, np.zeros((2, 3))),
        (rosenbrock, np.ones((2, 3))),
        (rastrigin, np.zeros((2, 3))),
        (himmelblau, np.zeros((1, 2))),
    ]:
        _, g = func(x)
        assert g.shape == x.shape


def test_sample_problems_reject_invalid_dimensions():
    with pytest.raises(ValueError, match='at least two'):
        rosenbrock(np.array([1.0]))

    with pytest.raises(ValueError, match='exactly two'):
        himmelblau(np.array([1.0, 2.0, 3.0]))


def test_sample_problems_are_exported_from_package_and_module():
    assert sphere is sphere_from_module
    assert rosenbrock is rosenbrock_from_module
    assert rastrigin is rastrigin_from_module
    assert himmelblau is himmelblau_from_module
