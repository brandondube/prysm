"""Tests for prysm.x.optym.sample_problems."""
import numpy as np
import pytest

from prysm.x.optym import himmelblau, rastrigin, rosenbrock, sphere
from prysm.x.optym.sample_problems import (
    HimmelblauProblem,
    RastriginProblem,
    RosenbrockProblem,
    SphereProblem,
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


def finite_difference_hvp(problem, x, v, step=1e-6):
    xp = x + step * v
    xm = x - step * v
    return (problem.g(xp) - problem.g(xm)) / (2 * step)


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


@pytest.mark.parametrize(
    'problem, func, x, v',
    [
        (
            SphereProblem(),
            sphere,
            np.array([1.5, -2.0, 0.25]),
            np.array([0.5, -1.0, 2.0]),
        ),
        (
            RosenbrockProblem(),
            rosenbrock,
            np.array([-1.2, 1.0, 0.5]),
            np.array([0.25, -1.5, 0.75]),
        ),
        (
            RastriginProblem(),
            rastrigin,
            np.array([0.25, -0.5, 1.25]),
            np.array([1.0, -0.25, 0.5]),
        ),
        (
            HimmelblauProblem(),
            himmelblau,
            np.array([-2.5, 3.0]),
            np.array([0.5, -1.5]),
        ),
    ],
)
def test_sample_problem_classes_provide_analytic_hooks(problem, func, x, v):
    assert problem.has_f
    assert problem.has_g
    assert problem.has_fg
    assert problem.has_h
    assert problem.has_hvp

    f, g = func(x)
    pf, pg = problem.fg(x)
    np.testing.assert_allclose(problem.f(x), f)
    np.testing.assert_allclose(problem.g(x), g)
    np.testing.assert_allclose(pf, f)
    np.testing.assert_allclose(pg, g)

    h = problem.h(x)
    hv = problem.hvp(x, v)
    np.testing.assert_allclose(h, h.T)
    np.testing.assert_allclose(h @ v.ravel(), hv.ravel())
    np.testing.assert_allclose(
        hv,
        finite_difference_hvp(problem, x, v),
        rtol=1e-5,
        atol=1e-5,
    )


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
