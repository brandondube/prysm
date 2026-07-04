"""Diffractive phase-function protocol tests."""
import numpy as np
import pytest

from prysm.x import materials
from prysm.x.raytracing.phase import (
    PhaseFunction,
    LinearGrating,
    CallablePhase,
)
from prysm.x.raytracing.spencer_and_murty import raytrace
from tests.x.raytracing.surface_helpers import plane


# ---------- LinearGrating closed forms --------------------------------------

def test_linear_grating_phase_gradient_hessian():
    g = LinearGrating(4.0, [1.0, 0.0], order=2)
    x = np.array([0.0, 1.0, -2.0])
    y = np.array([0.0, 3.0, 5.0])
    # phase = order * (g_x x + g_y y) / period = 2 * x / 4 = x / 2
    np.testing.assert_allclose(g.phase(x, y), x / 2.0)
    ph, gx, gy = g.phase_and_gradient(x, y)
    np.testing.assert_allclose(ph, x / 2.0)
    np.testing.assert_allclose(gx, 0.5)          # constant gradient
    np.testing.assert_allclose(gy, 0.0)
    pxx, pxy, pyy = g.phase_hessian(x, y)
    np.testing.assert_allclose(pxx, 0.0)         # linear -> zero Hessian
    np.testing.assert_allclose(pxy, 0.0)
    np.testing.assert_allclose(pyy, 0.0)


def test_linear_grating_uses_only_in_plane_components():
    """A legacy 3-vector grating direction ignores z."""
    g3 = LinearGrating(2.0, [1.0, 0.0, 7.0], order=1)
    g2 = LinearGrating(2.0, [1.0, 0.0], order=1)
    x = np.linspace(-3, 3, 5)
    y = np.linspace(2, -2, 5)
    np.testing.assert_allclose(g3.phase(x, y), g2.phase(x, y))


# ---------- first-class grating objects -------------------------------------

def test_surface_grating_property_requires_phase_function():
    """Surface.grating accepts a PhaseFunction or None and rejects tuples."""
    s = plane(interaction='refl', P=[0, 0, 0])
    s.grating = LinearGrating(2.0e-3, [1.0, 0.0, 0.0], 1)
    assert isinstance(s.grating, PhaseFunction)
    s.grating = None
    assert s.grating is None
    with pytest.raises(TypeError, match='PhaseFunction'):
        s.grating = (2.0e-3, [1.0, 0.0, 0.0], 1)
    with pytest.raises(TypeError, match='PhaseFunction'):
        s.grating = 42.0


# ---------- base finite-difference fallbacks --------------------------------

def test_callable_phase_gradient_fd_fallback():
    """CallablePhase central-differences missing gradients."""
    def phi(x, y):
        return 0.3 * x * x + 0.1 * y * y + 0.05 * x * y

    cp = CallablePhase(phi)
    x = np.array([0.5, -1.0, 2.0])
    y = np.array([1.0, 0.5, -1.5])
    ph, gx, gy = cp.phase_and_gradient(x, y)
    np.testing.assert_allclose(ph, phi(x, y))
    np.testing.assert_allclose(gx, 0.6 * x + 0.05 * y, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(gy, 0.2 * y + 0.05 * x, rtol=1e-6, atol=1e-6)


def test_callable_phase_hessian_fd_fallback():
    """CallablePhase central-differences missing Hessians."""
    def phi(x, y):
        return 0.3 * x * x + 0.1 * y * y + 0.05 * x * y

    def pag(x, y):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        return phi(x, y), 0.6 * x + 0.05 * y, 0.2 * y + 0.05 * x

    cp = CallablePhase(phi, phase_and_gradient=pag)
    x = np.array([0.5, -1.0, 2.0])
    y = np.array([1.0, 0.5, -1.5])
    pxx, pxy, pyy = cp.phase_hessian(x, y)
    np.testing.assert_allclose(pxx, 0.6, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(pxy, 0.05, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(pyy, 0.2, rtol=1e-5, atol=1e-5)


def test_callable_phase_prefers_supplied_derivatives():
    sentinel = object()

    def pag(x, y):
        return sentinel

    cp = CallablePhase(lambda x, y: x, phase_and_gradient=pag)
    assert cp.phase_and_gradient(np.array([0.0]), np.array([0.0])) is sentinel


# ---------- nonlinear (diffractive-lens) physics ----------------------------

class _RadialPhase(PhaseFunction):
    """phi = a/2 (x^2 + y^2): a rotationally symmetric diffractive lens."""

    def __init__(self, a):
        self.a = a

    def phase(self, x, y):
        return 0.5 * self.a * (x * x + y * y)

    def phase_and_gradient(self, x, y):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        return self.phase(x, y), self.a * x, self.a * y

    def phase_hessian(self, x, y):
        x = np.asarray(x, float)
        o = np.full(x.shape, self.a)
        z = np.zeros_like(x)
        return o, z, o


def test_radial_phase_focuses_like_a_lens():
    """A negative radial phase bends rays toward the axis."""
    g = materials.ConstantMaterial(1.5)
    s = plane(interaction='refr', P=[0, 0, 0], material=g)
    s.grating = _RadialPhase(-1e-3)
    img = plane(interaction='eval', P=[0, 0, 50.0])
    P = np.array([[5.0, 0.0, -5.0], [-5.0, 0.0, -5.0], [0.0, 4.0, -5.0]])
    S = np.broadcast_to(np.array([0.0, 0.0, 1.0]), (3, 3)).copy()
    r = raytrace([s, img], P, S, wvl=0.55)
    Sx = r.S[1, :, 0]
    Sy = r.S[1, :, 1]
    assert Sx[0] < 0.0          # +x ray bends toward axis
    assert Sx[1] > 0.0          # -x ray bends toward axis
    assert Sy[2] < 0.0          # +y ray bends toward axis
    # the symmetric pair converges
    np.testing.assert_allclose(abs(r.P[-1, 0, 0]), abs(r.P[-1, 1, 0]))
    assert abs(r.P[-1, 0, 0]) < 5.0
