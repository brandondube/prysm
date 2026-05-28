"""Tests for prysm.x.raytracing.sensitivity — merit Jacobian over a LensData's
dense free vector (``merit_jacobian_free``).

The Problem-level wiring (FD vs numeric merit, restore, autograd gating through
``Problem.jacobian``) lives in test_raytracing_lensdata_design.py; this module
pins ``merit_jacobian_free`` directly against analytic paraxial derivatives.
"""
import numpy as np
import pytest

from prysm.x.raytracing import LensData
from prysm.x.raytracing.surfaces import Conic
from prysm.x.raytracing.sensitivity import merit_jacobian_free
from prysm.x.raytracing.paraxial import (
    paraxial_image_distance,
    effective_focal_length,
)


# ---------- FD against analytic d(BFD)/dc for a single sphere ----------

def test_fd_jacobian_single_sphere_curvature():
    """For a single refracting sphere, BFD = n*R/(n-1) = n/((n-1)*c).
    d(BFD)/dc = -n/((n-1) c^2).
    """
    n_glass = 1.5
    c0 = 1.0 / 50.0
    expected = -n_glass / ((n_glass - 1.0) * c0 * c0)

    ld = LensData(wavelengths=[0.55e-3]).add(
        Conic(c0, 0.0), typ='refr', material=lambda wvl: n_glass,
        thickness=0.0)
    ld.vary('curvature', surfaces=0)
    J = merit_jacobian_free(
        ld, lambda: float(paraxial_image_distance(ld, wvl=0.55e-3)), step=1e-7)
    np.testing.assert_allclose(J[0], expected, rtol=1e-5)


def test_fd_jacobian_efl_doublet_curvatures():
    """Two-sphere thin doublet — Jacobian of EFL w.r.t. (c1, c2) should match
    the lensmaker derivative: 1/f = (n-1)(c1 - c2) so df/dc1 = -f^2*(n-1).
    """
    n_glass = 1.5
    c1, c2 = 1.0 / 100.0, -1.0 / 100.0
    f = 1.0 / ((n_glass - 1.0) * (c1 - c2))
    expected_dfdc1 = -f * f * (n_glass - 1.0)
    expected_dfdc2 = +f * f * (n_glass - 1.0)
    ld = (LensData(wavelengths=[0.55e-3])
          .add(Conic(c1, 0.0), typ='refr', material=lambda wvl: n_glass,
               thickness=1e-9)
          .add(Conic(c2, 0.0), typ='refr', material=lambda wvl: 1.0,
               thickness=0.0))
    ld.vary('curvature', surfaces=[0, 1])
    J = merit_jacobian_free(
        ld, lambda: float(effective_focal_length(ld, wvl=0.55e-3)), step=1e-7)
    np.testing.assert_allclose(J[0], expected_dfdc1, rtol=1e-5)
    np.testing.assert_allclose(J[1], expected_dfdc2, rtol=1e-5)


def test_fd_jacobian_restores_free_vector():
    """After Jacobian evaluation the LensData is back to its nominal free
    vector (no transient perturbation leaks out)."""
    ld = LensData(wavelengths=[0.55e-3]).add(
        Conic(1 / 50.0, 0.0), typ='refr', material=lambda wvl: 1.5,
        thickness=0.0)
    ld.vary('curvature', surfaces=0)
    x0 = ld.pack()
    merit_jacobian_free(
        ld, lambda: float(paraxial_image_distance(ld, wvl=0.55e-3)))
    np.testing.assert_allclose(ld.pack(), x0)


def test_fd_jacobian_unknown_method_raises():
    ld = LensData(wavelengths=[0.55e-3]).add(
        Conic(1 / 50.0, 0.0), typ='refr', material=lambda wvl: 1.5,
        thickness=0.0)
    ld.vary('curvature', surfaces=0)
    with pytest.raises(ValueError, match="method must be"):
        merit_jacobian_free(
            ld, lambda: float(paraxial_image_distance(ld, wvl=0.55e-3)),
            method='nope')


# ---------- autograd backend gating ----------

def test_autograd_method_requires_torch_backend():
    """With the default numpy backend, asking for 'autograd' must error
    helpfully rather than silently producing nonsense."""
    ld = LensData(wavelengths=[0.55e-3]).add(
        Conic(1 / 50.0, 0.0), typ='refr', material=lambda wvl: 1.5,
        thickness=0.0)
    ld.vary('curvature', surfaces=0)
    with pytest.raises(RuntimeError, match='backend to be torch'):
        merit_jacobian_free(
            ld, lambda: float(paraxial_image_distance(ld, wvl=0.55e-3)),
            method='autograd')
