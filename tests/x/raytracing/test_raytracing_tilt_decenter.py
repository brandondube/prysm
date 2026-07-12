"""Tests for first-class per-surface tilt and decenter on Surface."""
import numpy as np
import pytest

from prysm.x import materials
from tests.x.raytracing.surface_helpers import (
    plane, sphere, conic, off_axis_conic, even_asphere, q2d
)

from prysm.coordinates import make_rotation_matrix
from prysm.x.raytracing.spencer_and_murty import (
    raytrace,
    transform_to_global_coords,
    transform_to_local_coords,
)


# ---------- decenter ----------

def test_decenter_shifts_vertex_by_amount():
    """Surface.P should equal nominal P + decenter."""
    nominal = np.array([0., 0., 5.])
    decenter = np.array([1., 2., 0.5])
    surf = plane(interaction='eval', P=nominal, decenter=decenter)
    np.testing.assert_allclose(surf.P, nominal + decenter)


def test_decenter_changes_intersection_point():
    """A flat eval plane decentered in z by +d should be hit at z = d
    (relative to the nominal vertex), not z = 0."""
    nominal = np.array([0., 0., 0.])
    surf = plane(interaction='eval', P=nominal, decenter=[0., 0., 3.0])
    P0 = np.array([0., 0., -10.])
    S0 = np.array([0., 0., 1.])
    trace = raytrace([surf], P0, S0, wvl=0.55)
    phist = trace.P
    np.testing.assert_allclose(phist[-1, 2], 3.0, atol=1e-12)


def test_decenter_validates_shape():
    with pytest.raises(ValueError, match='length-3'):
        plane(interaction='eval', P=np.array([0., 0., 0.]),
                      decenter=[1.0, 2.0])


# ---------- tilt ----------

def test_coordinate_transforms_preserve_single_ray_rank_with_rotation():
    """A rotated vector round trip remains shape (3,), not (1, 3)."""
    R = make_rotation_matrix((3.0, -7.0, 11.0))
    origin = np.array([1.0, -2.0, 3.0])
    point = np.array([4.0, 5.0, 6.0])
    direction = np.array([0.1, -0.2, 0.97])
    local_point, local_direction = transform_to_local_coords(
        point, origin, direction, R)
    assert local_point.shape == (3,)
    assert local_direction.shape == (3,)
    roundtrip_point, roundtrip_direction = transform_to_global_coords(
        local_point, origin, local_direction, R)
    assert roundtrip_point.shape == (3,)
    assert roundtrip_direction.shape == (3,)
    np.testing.assert_allclose(roundtrip_point, point, atol=1e-12)
    np.testing.assert_allclose(roundtrip_direction, direction, atol=1e-12)


def test_tilt_alone_sets_R_to_rotation_matrix():
    """With R=None and tilt=(rz, ry, rx), Surface.R should equal
    make_rotation_matrix(tilt)."""
    tilt = (10.0, 5.0, 2.0)
    surf = plane(interaction='eval', P=np.array([0., 0., 0.]), tilt=tilt)
    R_expected = make_rotation_matrix(tilt)
    np.testing.assert_allclose(surf.R, R_expected, atol=1e-12)


def test_tilt_radians_kwarg():
    """tilt_radians=True must produce a different rotation than degrees."""
    angle = 0.1
    surf_deg = plane(interaction='eval', P=np.array([0., 0., 0.]),
                             tilt=(0., 0., angle), tilt_radians=False)
    surf_rad = plane(interaction='eval', P=np.array([0., 0., 0.]),
                             tilt=(0., 0., angle), tilt_radians=True)
    # the angle is small enough that the two interpretations differ by ~57x
    assert not np.allclose(surf_deg.R, surf_rad.R)
    # cross-check: passing the same value in degrees-converted-to-radians
    # gives matching R
    surf_deg_eq = plane(interaction='eval', P=np.array([0., 0., 0.]),
                                tilt=(0., 0., np.degrees(angle)))
    np.testing.assert_allclose(surf_deg_eq.R, surf_rad.R, atol=1e-12)


def test_tilt_composes_with_existing_R():
    """If both R and tilt are provided, R_total = R @ R_tilt."""
    R_base = make_rotation_matrix((0., 45., 0.))
    R_tilt = make_rotation_matrix((0., 5., 0.))
    surf = plane(interaction='eval', P=np.array([0., 0., 0.]),
                         R=R_base, tilt=(0., 5., 0.))
    np.testing.assert_allclose(surf.R, R_base @ R_tilt, atol=1e-12)


def test_tilted_mirror_reflects_at_double_angle():
    """A planar mirror tilted by alpha about y deflects an axial input ray
    by 2*alpha.  Use radians to avoid degree<->rad confusion in the check.
    """
    alpha = np.radians(10.0)  # 10 deg about y-axis
    surf = plane(interaction='refl', P=np.array([0., 0., 0.]),
                         tilt=(0., np.degrees(alpha), 0.))
    P0 = np.array([0., 0., -10.])
    S0 = np.array([0., 0., 1.])
    trace = raytrace([surf], P0, S0, wvl=0.55)
    shist = trace.S
    S_out = shist[-1]
    # for an input along +z hitting a mirror tilted by alpha about y, the
    # reflected ray makes angle 2*alpha with the -z axis in the xz plane
    # i.e. S_out should be approximately (sin(2a), 0, -cos(2a))
    expected = np.array([np.sin(2 * alpha), 0.0, -np.cos(2 * alpha)])
    np.testing.assert_allclose(S_out, expected, atol=1e-9)


def test_zero_tilt_zero_decenter_match_no_perturbation():
    """tilt=(0,0,0) and decenter=(0,0,0) must match the un-perturbed surface
    bit-for-bit on intersection."""
    P_vertex = np.array([0., 0., 5.])
    surf_a = sphere(c=1 / 50.0, interaction='refr', P=P_vertex,
                            material=materials.ConstantMaterial(1.5))
    surf_b = sphere(c=1 / 50.0, interaction='refr', P=P_vertex,
                            material=materials.ConstantMaterial(1.5),
                            tilt=(0., 0., 0.), decenter=(0., 0., 0.))
    P0 = np.array([1., 0., -10.])
    S0 = np.array([0., 0., 1.])
    trace_a = raytrace([surf_a], P0, S0, wvl=0.55)
    trace_b = raytrace([surf_b], P0, S0, wvl=0.55)
    pa = trace_a.P
    sa = trace_a.S
    pb = trace_b.P
    sb = trace_b.S
    np.testing.assert_allclose(pa, pb, atol=1e-12)
    np.testing.assert_allclose(sa, sb, atol=1e-12)


def test_tilt_decenter_threaded_through_all_factories():
    """Smoke check: every classmethod factory accepts tilt/decenter without
    TypeError and produces a Surface whose P/R reflect the perturbation."""
    P0 = np.array([0., 0., 0.])
    decenter = (0.1, 0.2, 0.3)
    tilt = (1., 2., 3.)
    factories = [
        plane(interaction='eval', P=P0, tilt=tilt, decenter=decenter),
        sphere(c=1 / 100., interaction='refr', P=P0, material=materials.ConstantMaterial(1.5),
                       tilt=tilt, decenter=decenter),
        conic(c=1 / 100., k=-1., interaction='refl', P=P0,
                      tilt=tilt, decenter=decenter),
        off_axis_conic(c=1 / 100., k=-1., interaction='refl', P=P0,
                               dx=10., dy=0.,
                               tilt=tilt, decenter=decenter),
        even_asphere(c=1 / 100., k=0.0, coefs=[1e-8],
                             interaction='refr', P=P0, material=materials.ConstantMaterial(1.5),
                             tilt=tilt, decenter=decenter),
        q2d(c=1 / 100., k=0.0, normalization_radius=10.,
                    cm0=(0.,), ams=(), bms=(),
                    interaction='refr', P=P0, material=materials.ConstantMaterial(1.5),
                    tilt=tilt, decenter=decenter),
    ]
    expected_P = P0 + np.array(decenter)
    expected_R = make_rotation_matrix(tilt)
    for surf in factories:
        np.testing.assert_allclose(surf.P, expected_P, atol=1e-12)
        np.testing.assert_allclose(surf.R, expected_R, atol=1e-12)
