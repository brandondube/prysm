"""Tests for prysm.x.optym.problem."""
import numpy as np
import pytest

from prysm.x.optym import Adam, Problem, as_problem
from prysm.x.optym.problem import _CallableProblem


def _quadratic_fg(x):
    """f(x) = 0.5 * ||x||^2; gradient = x."""
    return float(0.5 * np.sum(x * x)), x


class FOnly(Problem):
    """Subclass providing only f; derivatives are finite differenced."""

    has_f = True

    def __init__(self, fd_method='central'):
        super().__init__(fd_method=fd_method)
        self.calls = 0

    def _f(self, x):
        self.calls += 1
        return float(0.5 * np.sum(x * x))


class GOnly(Problem):
    """Subclass providing only g."""

    has_g = True

    def _g(self, x):
        return x


class FGOnly(Problem):
    """Subclass providing only fg; f/g are cheap slices."""

    has_fg = True

    def __init__(self):
        self.calls = 0

    def _fg(self, x):
        self.calls += 1
        return _quadratic_fg(x)


class SeparateFG(Problem):
    """Subclass providing f and g separately; fg calls both."""

    has_f = True
    has_g = True

    def __init__(self):
        self.f_calls = 0
        self.g_calls = 0

    def _f(self, x):
        self.f_calls += 1
        return float(0.5 * np.sum(x * x))

    def _g(self, x):
        self.g_calls += 1
        return x


def test_problem_default_f_calls_fg():
    p = FGOnly()
    x = np.array([3.0, 4.0])
    f = p.f(x)
    np.testing.assert_allclose(f, 12.5)
    assert p.calls == 1


def test_problem_default_g_calls_fg():
    p = FGOnly()
    x = np.array([3.0, 4.0])
    g = p.g(x)
    np.testing.assert_allclose(g, x)
    assert p.calls == 1


def test_problem_default_fg_calls_f_and_g():
    p = SeparateFG()
    x = np.array([3.0, 4.0])
    f, g = p.fg(x)
    np.testing.assert_allclose(f, 12.5)
    np.testing.assert_allclose(g, x)
    assert p.f_calls == 1 and p.g_calls == 1


def test_forward_difference_gradient_from_f():
    p = FOnly(fd_method='forward')
    x = np.array([3.0, 4.0])
    np.testing.assert_allclose(p.g(x), x, rtol=1e-6, atol=1e-6)


def test_central_difference_gradient_from_f():
    p = FOnly(fd_method='central')
    x = np.array([3.0, 4.0])
    np.testing.assert_allclose(p.g(x), x, rtol=1e-9, atol=1e-9)


def test_central_difference_hessian_from_g():
    p = GOnly()
    x = np.array([3.0, 4.0])
    np.testing.assert_allclose(p.h(x), np.eye(2), rtol=1e-9, atol=1e-9)


def test_central_difference_hvp_from_g():
    p = GOnly()
    x = np.array([3.0, 4.0])
    v = np.array([1.0, -2.0])
    np.testing.assert_allclose(p.hvp(x, v), v, rtol=1e-9, atol=1e-9)


def test_problem_rejects_unknown_fd_method():
    with pytest.raises(ValueError):
        Problem(fd_method='backward')


def test_as_problem_returns_problem_unchanged():
    p = FGOnly()
    assert as_problem(p) is p


def test_as_problem_accepts_duck_typed_object():
    class Duck:
        def fg(self, x):
            return 0.0, x
    d = Duck()
    assert as_problem(d) is d


def test_as_problem_wraps_callable():
    p = as_problem(_quadratic_fg)
    assert isinstance(p, _CallableProblem)
    x = np.array([1.0, -2.0])
    f, g = p.fg(x)
    np.testing.assert_allclose(f, 2.5)
    np.testing.assert_allclose(g, x)


def test_as_problem_rejects_non_callable_non_problem():
    with pytest.raises(TypeError):
        as_problem(42)


def test_adam_accepts_problem_instance():
    """End-to-end: Adam can be constructed from a Problem subclass and converges on a quadratic in a handful of steps."""
    p = FGOnly()
    x0 = np.array([5.0, -3.0])
    opt = Adam(p, x0, alpha=0.5)
    for _ in range(200):
        x, f, g = opt.step()
    # not asserting bit-equal convergence -- Adam's momentum makes it oscillate
    # somewhat -- but cost should drop by orders of magnitude
    f0, _ = _quadratic_fg(x0)
    assert f < f0 * 1e-3
