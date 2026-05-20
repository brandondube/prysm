"""Tests for prysm.x.raytracing.sensitivity — merit Jacobians."""
import numpy as np
import pytest

from tests.x.raytracing.surface_helpers import (
    plane, sphere, conic, off_axis_conic, even_asphere, q2d, zernike, xy,
    chebyshev, jacobi, toroid, biconic,
)

from prysm.x.raytracing.surfaces import Surface
from prysm.x.raytracing.sensitivity import (
    merit_jacobian,
    surface_param,
    vertex_z_param,
)
from prysm.x.raytracing.paraxial import paraxial_image_distance, effective_focal_length


# ---------- surface_param helper ----------

def test_surface_param_round_trips_setter():
    rx = [
        sphere(c=1 / 50.0, typ='refr', P=np.array([0., 0., 0.]),
                       n=lambda wvl: 1.5),
    ]
    g, s = surface_param(rx, 0, 'c')
    assert g() == pytest.approx(1 / 50.0)
    s(1 / 100.0)
    assert g() == pytest.approx(1 / 100.0)
    # the FFp closure should see the new value next call
    assert rx[0].params['c'] == pytest.approx(1 / 100.0)


def test_surface_param_unknown_key_raises():
    rx = [plane(typ='eval', P=np.array([0., 0., 0.]))]
    with pytest.raises(KeyError):
        surface_param(rx, 0, 'c')  # plane has no params


def test_vertex_z_param_round_trips():
    rx = [plane(typ='eval', P=np.array([0., 0., 5.]))]
    g, s = vertex_z_param(rx, 0)
    assert g() == pytest.approx(5.0)
    s(7.0)
    assert g() == pytest.approx(7.0)
    assert rx[0].P[2] == pytest.approx(7.0)


# ---------- FD against analytic d(BFD)/dc for a single sphere ----------

def test_fd_jacobian_single_sphere_curvature():
    """For a single refracting sphere, BFD = n*R/(n-1) = n/((n-1)*c).
    d(BFD)/dc = -n/((n-1) c^2).
    """
    n_glass = 1.5
    c0 = 1.0 / 50.0
    expected = -n_glass / ((n_glass - 1.0) * c0 * c0)

    rx = [
        sphere(c=c0, typ='refr', P=np.array([0., 0., 0.]),
                       n=lambda wvl: n_glass),
    ]
    params = [surface_param(rx, 0, 'c')]
    J = merit_jacobian(rx, params, paraxial_image_distance, step=1e-7)
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
    rx = [
        sphere(c=c1, typ='refr', P=np.array([0., 0., 0.]),
                       n=lambda wvl: n_glass),
        sphere(c=c2, typ='refr', P=np.array([0., 0., 1e-9]),
                       n=lambda wvl: 1.0),
    ]
    params = [surface_param(rx, 0, 'c'), surface_param(rx, 1, 'c')]
    J = merit_jacobian(rx, params, effective_focal_length, step=1e-7)
    np.testing.assert_allclose(J[0], expected_dfdc1, rtol=1e-5)
    np.testing.assert_allclose(J[1], expected_dfdc2, rtol=1e-5)


def test_fd_jacobian_restores_prescription_state():
    """After Jacobian evaluation, the prescription must be back to its
    nominal state (no transient mutation leaks out)."""
    rx = [
        sphere(c=1 / 50.0, typ='refr', P=np.array([0., 0., 0.]),
                       n=lambda wvl: 1.5),
    ]
    c_before = rx[0].params['c']
    params = [surface_param(rx, 0, 'c')]
    merit_jacobian(rx, params, paraxial_image_distance)
    assert rx[0].params['c'] == pytest.approx(c_before)


def test_fd_jacobian_handles_zero_valued_parameter():
    """A parameter starting at zero must use an absolute step (h = step), not
    a relative step that would collapse to 0."""
    n_glass = 1.5
    rx = [
        sphere(c=1 / 50.0, typ='refr', P=np.array([0., 0., 0.]),
                       n=lambda wvl: n_glass),
        plane(typ='eval', P=np.array([0., 0., 0.])),  # z=0, perturb me
    ]
    # perturbing the eval plane's z does not change the BFD (paraxial image
    # is independent of where you stick a downstream eval plane in the
    # matrix walk because BFD is from the last surface vertex; moving the
    # eval plane *is* moving that vertex though, so dBFD/dz_eval = -1).
    params = [vertex_z_param(rx, 1)]
    J = merit_jacobian(rx, params, paraxial_image_distance, step=1e-6)
    np.testing.assert_allclose(J[0], -1.0, atol=1e-6)


def test_fd_jacobian_unknown_method_raises():
    rx = [
        sphere(c=1 / 50.0, typ='refr', P=np.array([0., 0., 0.]),
                       n=lambda wvl: 1.5),
    ]
    params = [surface_param(rx, 0, 'c')]
    with pytest.raises(ValueError, match="method must be"):
        merit_jacobian(rx, params, paraxial_image_distance, method='nope')


# ---------- autograd backend gating ----------

def test_autograd_method_requires_torch_backend():
    """With the default numpy backend, asking for 'autograd' must error
    helpfully rather than silently producing nonsense."""
    rx = [
        sphere(c=1 / 50.0, typ='refr', P=np.array([0., 0., 0.]),
                       n=lambda wvl: 1.5),
    ]
    params = [surface_param(rx, 0, 'c')]
    with pytest.raises(RuntimeError, match='backend to be torch'):
        merit_jacobian(rx, params, paraxial_image_distance, method='autograd')
