"""Optical-path-function protocol tests."""
import numpy as np
import pytest

from prysm.x import materials
from prysm.x.raytracing.opl import (
    OPLFunc,
    LinearGrating,
    CallableOPL,
)
from prysm.x.raytracing.spencer_and_murty import raytrace
from tests.x.raytracing.surface_helpers import plane


# ---------- LinearGrating closed forms --------------------------------------

def test_linear_grating_opl_gradient_hessian():
    g = LinearGrating(4e-3, [1.0, 0.0], order=2)
    x = np.array([0.0, 1.0, -2.0])
    y = np.array([0.0, 3.0, 5.0])
    # OPL = order * wavelength_mm * x / period = x / 4 at 0.5 um.
    np.testing.assert_allclose(g.opl(x, y, 0.5), x / 4.0)
    opl, gx, gy = g.opl_and_gradient(x, y, 0.5)
    np.testing.assert_allclose(opl, x / 4.0)
    np.testing.assert_allclose(gx, 0.25)         # constant gradient
    np.testing.assert_allclose(gy, 0.0)
    pxx, pxy, pyy = g.opl_hessian(x, y, 0.5)
    np.testing.assert_allclose(pxx, 0.0)         # linear -> zero Hessian
    np.testing.assert_allclose(pxy, 0.0)
    np.testing.assert_allclose(pyy, 0.0)


def test_linear_grating_uses_only_in_plane_components():
    """A legacy 3-vector grating direction ignores z."""
    g3 = LinearGrating(2e-3, [1.0, 0.0, 7.0], order=1)
    g2 = LinearGrating(2e-3, [1.0, 0.0], order=1)
    x = np.linspace(-3, 3, 5)
    y = np.linspace(2, -2, 5)
    np.testing.assert_allclose(g3.opl(x, y, 0.55), g2.opl(x, y, 0.55))


def test_linear_grating_mutation_keeps_gradient_coherent():
    g = LinearGrating(2e-3, [1.0, 0.0], order=1)
    _, gx0, _ = g.opl_and_gradient(np.array([0.0]), np.array([0.0]), 0.5)
    g.period = 4e-3
    g.order = 2
    g.g_vec = (0.0, 1.0)
    _, gx1, gy1 = g.opl_and_gradient(
        np.array([0.0]), np.array([0.0]), 0.5)
    np.testing.assert_allclose(gx0, 0.25)
    np.testing.assert_allclose(gx1, 0.0)
    np.testing.assert_allclose(gy1, 0.25)


# ---------- first-class grating objects -------------------------------------

def test_surface_grating_property_requires_opl_func():
    """Surface.grating accepts an OPLFunc or None and rejects tuples."""
    s = plane(interaction='refl', P=[0, 0, 0])
    s.grating = LinearGrating(2.0e-3, [1.0, 0.0, 0.0], 1)
    assert isinstance(s.grating, OPLFunc)
    s.grating = None
    assert s.grating is None
    with pytest.raises(TypeError, match='OPLFunc'):
        s.grating = (2.0e-3, [1.0, 0.0, 0.0], 1)
    with pytest.raises(TypeError, match='OPLFunc'):
        s.grating = 42.0


# ---------- base finite-difference fallbacks --------------------------------

def test_callable_opl_gradient_fd_fallback():
    """CallableOPL central-differences missing gradients."""
    def fn(x, y, wavelength):
        return 0.3 * x * x + 0.1 * y * y + 0.05 * x * y

    cp = CallableOPL(fn)
    x = np.array([0.5, -1.0, 2.0])
    y = np.array([1.0, 0.5, -1.5])
    opl, gx, gy = cp.opl_and_gradient(x, y, 0.55)
    np.testing.assert_allclose(opl, fn(x, y, 0.55))
    np.testing.assert_allclose(gx, 0.6 * x + 0.05 * y, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(gy, 0.2 * y + 0.05 * x, rtol=1e-6, atol=1e-6)


def test_callable_opl_hessian_fd_fallback():
    """CallableOPL central-differences missing Hessians."""
    def fn(x, y, wavelength):
        return 0.3 * x * x + 0.1 * y * y + 0.05 * x * y

    def oag(x, y, wavelength):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        return fn(x, y, wavelength), 0.6 * x + 0.05 * y, 0.2 * y + 0.05 * x

    cp = CallableOPL(fn, opl_and_gradient=oag)
    x = np.array([0.5, -1.0, 2.0])
    y = np.array([1.0, 0.5, -1.5])
    pxx, pxy, pyy = cp.opl_hessian(x, y, 0.55)
    np.testing.assert_allclose(pxx, 0.6, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(pxy, 0.05, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(pyy, 0.2, rtol=1e-5, atol=1e-5)


def test_callable_opl_prefers_supplied_derivatives():
    sentinel = object()

    def oag(x, y, wavelength):
        return sentinel

    cp = CallableOPL(lambda x, y, wavelength: x, opl_and_gradient=oag)
    assert cp.opl_and_gradient(
        np.array([0.0]), np.array([0.0]), 0.55) is sentinel


# ---------- nonlinear (diffractive-lens) physics ----------------------------

class _RadialOPL(OPLFunc):
    """OPL = a/2 (x^2 + y^2): a rotationally symmetric thin power."""

    def __init__(self, a):
        self.a = a

    def opl(self, x, y, wavelength):
        return 0.5 * self.a * (x * x + y * y)

    def opl_and_gradient(self, x, y, wavelength):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        return self.opl(x, y, wavelength), self.a * x, self.a * y

    def opl_hessian(self, x, y, wavelength):
        x = np.asarray(x, float)
        o = np.full(x.shape, self.a)
        z = np.zeros_like(x)
        return o, z, o


def test_radial_opl_focuses_like_a_lens():
    """A negative radial OPL gradient bends rays toward the axis."""
    g = materials.ConstantMaterial(1.5)
    s = plane(interaction='refr', P=[0, 0, 0], material=g)
    s.grating = _RadialOPL(-1e-3)
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
