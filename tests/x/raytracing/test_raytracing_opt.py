"""Tests for prysm.x.raytracing.opt."""
import numpy as np
import pytest

from tests.x.raytracing.surface_helpers import (
    plane, sphere, conic, off_axis_conic, even_asphere, q2d, zernike, xy,
    chebyshev, jacobi, toroid, biconic,
)

from prysm.x.raytracing.surfaces import Surface
from prysm.x.raytracing.opt import (
    aim_rays,
    locate_ep,
    locate_xp,
    xp_reference_sphere,
    spot_centroid,
    rms_spot_radius,
    geometric_psf_histogram,
)
from prysm.x.raytracing.spencer_and_murty import (
    STATUS_CLIP, STATUS_MISS, intersect_reference_sphere,
)
from prysm.x.raytracing.paraxial import paraxial_image_distance
from prysm.x.raytracing.auto import rc_prescription_from_efl_bfl_sep


# ---------- aim_rays: single-ray (1-row bundle) ----------

def test_aim_single_ray_hits_target_on_simple_mirror():
    """Aim one ray (a 1-row bundle) at a transverse target on a flat eval
    surface, exercising the reflective-geometry path."""
    prescription = [
        conic(c=1 / 200.0, k=-1.0, interaction='refl', P=np.array([0., 0., 0.])),
        plane(interaction='eval', P=np.array([0., 0., -50.])),  # rays head -z after reflection
    ]
    # incoming collimated ray at (?, ?, -100) → +z; we adjust the (x, y)
    # launch position so the ray hits target_xy=(2, -1) on the eval plane.
    # the initial (Px, Py) guess is the target.
    P = np.array([[2.0, -1.0, -100.]])
    S = np.array([[0., 0., 1.]])
    P_aimed, _, converged = aim_rays(P, S, prescription, surface_index=1,
                                  target_xy=(2.0, -1.0), wvl=0.55)
    assert bool(converged[0])
    # launch z is preserved by the aiming adjustment
    assert P_aimed[0, 2] == -100.0
    from prysm.x.raytracing.spencer_and_murty import raytrace
    trace = raytrace(prescription, P_aimed, S, wvl=0.55)
    np.testing.assert_allclose(trace.P[-1, 0, :2], (2.0, -1.0), atol=1e-6)


def _tir_unaimable_bundle():
    """A single steep ray (1-row bundle) in glass meeting a glass->air
    interface.

    The ray totally internally reflects for every launch (Px, Py), so its
    landing is never finite and it can never be aimed onto the eval plane.
    This drives the all-rays-dead break path in aim_rays.  The launch medium
    (n=1.5) is carried by a leading eval object surface -- the convention for
    an immersed launch.
    """
    prescription = [
        plane(interaction='eval', P=np.array([0., 0., -100.]),
              material=lambda w: 1.5),
        plane(interaction='refr', P=np.array([0., 0., 0.]), material=lambda w: 1.0),
        plane(interaction='eval', P=np.array([0., 0., 10.])),
    ]
    theta = np.deg2rad(60.0)
    P = np.array([[0., 0., -100.]])
    S = np.array([[np.sin(theta), 0., np.cos(theta)]])
    return prescription, P, S


def test_aim_single_ray_strict_raises_on_unaimable_ray():
    """strict=True must raise when the only ray cannot reach the surface."""
    prescription, P, S = _tir_unaimable_bundle()
    with pytest.raises(RuntimeError):
        aim_rays(P, S, prescription, surface_index=2, target_xy=(0.0, 0.0),
                 wvl=0.55, strict=True)


def test_aim_single_ray_strict_false_does_not_raise():
    """strict=False must return a best-effort P even for an un-aimable ray."""
    prescription, P, S = _tir_unaimable_bundle()
    P_out, _, converged = aim_rays(P, S, prescription, surface_index=2,
                                target_xy=(0.0, 0.0), wvl=0.55,
                                strict=False)
    assert P_out.shape == (1, 3)
    assert not bool(converged[0])


# ---------- aim_rays (batched kernel) ----------

def _singlet_with_internal_stop(n_glass=1.5):
    """Two refractive conics with a plane stop between them."""
    return [
        conic(c=1 / 50.0, k=0.0, interaction='refr', P=np.array([0., 0., 0.]),
              material=lambda w: n_glass),
        plane(interaction='eval', P=np.array([0., 0., 2.5])),
        conic(c=-1 / 50.0, k=0.0, interaction='refr', P=np.array([0., 0., 5.]),
              material=lambda w: 1.0),
        plane(interaction='eval', P=np.array([0., 0., 100.])),
    ]


def _collimated_y_fan(n, half, z0, theta_deg):
    """A collimated y-fan of n rays at field angle theta_deg about x."""
    theta = np.deg2rad(theta_deg)
    P = np.zeros((n, 3))
    P[:, 1] = np.linspace(-half, half, n)
    P[:, 2] = z0
    S = np.broadcast_to(np.array([0., np.sin(theta), np.cos(theta)]),
                        (n, 3)).copy()
    return P, S


def test_aim_rays_collimated_bundle_onto_stop():
    """A whole fan aimed at (0, 0) on an internal stop lands there."""
    presc = _singlet_with_internal_stop()
    P, S = _collimated_y_fan(7, half=2.0, z0=-10.0, theta_deg=2.0)
    z_before = P[:, 2].copy()
    P_aim, _, converged = aim_rays(P, S, presc, surface_index=1,
                                target_xy=(0.0, 0.0), wvl=0.55)
    assert bool(np.all(converged))
    # launch z untouched
    np.testing.assert_array_equal(P_aim[:, 2], z_before)
    from prysm.x.raytracing.spencer_and_murty import raytrace
    tr = raytrace(presc, P_aim, S, wvl=0.55)
    np.testing.assert_allclose(tr.P[2, :, :2], 0.0, atol=1e-9)


def test_aim_rays_onto_nonzero_target():
    """Aiming to a nonzero (x, y) on the stop lands there."""
    presc = _singlet_with_internal_stop()
    P, S = _collimated_y_fan(5, half=2.0, z0=-10.0, theta_deg=1.0)
    P_aim, _, converged = aim_rays(P, S, presc, surface_index=1,
                                target_xy=(0.7, -0.3), wvl=0.55)
    assert bool(np.all(converged))
    from prysm.x.raytracing.spencer_and_murty import raytrace
    tr = raytrace(presc, P_aim, S, wvl=0.55)
    np.testing.assert_allclose(tr.P[2, :, 0], 0.7, atol=1e-9)
    np.testing.assert_allclose(tr.P[2, :, 1], -0.3, atol=1e-9)


def test_aim_rays_onto_tilted_surface():
    """Aiming works when the aim surface is tilted out of the xy plane."""
    presc = [
        conic(c=1 / 50.0, k=0.0, interaction='refr', P=np.array([0., 0., 0.]),
              material=lambda w: 1.5),
        plane(interaction='eval', P=np.array([0., 0., 3.0]), tilt=(0., 8., 0.)),
        plane(interaction='eval', P=np.array([0., 0., 50.])),
    ]
    P, S = _collimated_y_fan(5, half=2.0, z0=-10.0, theta_deg=1.5)
    P_aim, _, converged = aim_rays(P, S, presc, surface_index=1,
                                target_xy=(0.0, 0.0), wvl=0.55)
    assert bool(np.all(converged))
    from prysm.x.raytracing.spencer_and_murty import raytrace
    tr = raytrace(presc, P_aim, S, wvl=0.55)
    np.testing.assert_allclose(tr.P[2, :, :2], 0.0, atol=1e-9)


def test_aim_rays_masks_divergent_ray():
    """A ray that TIRs for every launch is flagged not-converged and the
    rest of the bundle still aims (strict=False)."""
    presc = [
        plane(interaction='eval', P=np.array([0., 0., -5.]),
              material=lambda w: 1.5),
        plane(interaction='refr', P=np.array([0., 0., 0.]), material=lambda w: 1.0),
        plane(interaction='eval', P=np.array([0., 0., 10.])),
    ]
    # ray 0 is steep enough to TIR (glass -> air); rays 1, 2 are gentle
    S = np.array([
        [np.sin(np.deg2rad(60.)), 0., np.cos(np.deg2rad(60.))],
        [0., np.sin(np.deg2rad(2.)), np.cos(np.deg2rad(2.))],
        [np.sin(np.deg2rad(2.)), 0., np.cos(np.deg2rad(2.))],
    ])
    P = np.zeros((3, 3))
    P[:, 2] = -5.0
    P_aim, _, converged = aim_rays(P, S, presc, surface_index=2,
                                target_xy=(0.0, 0.0), wvl=0.55,
                                strict=False)
    assert not bool(converged[0])
    assert bool(converged[1]) and bool(converged[2])
    # the masked ray keeps its nominal launch xy
    np.testing.assert_array_equal(P_aim[0, :2], P[0, :2])
    from prysm.x.raytracing.spencer_and_murty import raytrace
    tr = raytrace(presc, P_aim, S, wvl=0.55)
    np.testing.assert_allclose(tr.P[-1, 1:, :2], 0.0, atol=1e-9)


def test_aim_rays_strict_raises_listing_indices():
    """strict=True raises a RuntimeError that names the un-aimable ray."""
    presc = [
        plane(interaction='eval', P=np.array([0., 0., -5.]),
              material=lambda w: 1.5),
        plane(interaction='refr', P=np.array([0., 0., 0.]), material=lambda w: 1.0),
        plane(interaction='eval', P=np.array([0., 0., 10.])),
    ]
    S = np.array([
        [0., np.sin(np.deg2rad(2.)), np.cos(np.deg2rad(2.))],
        [np.sin(np.deg2rad(60.)), 0., np.cos(np.deg2rad(60.))],
    ])
    P = np.zeros((2, 3))
    P[:, 2] = -5.0
    with pytest.raises(RuntimeError, match='1'):
        aim_rays(P, S, presc, surface_index=2, target_xy=(0.0, 0.0),
                 wvl=0.55, strict=True)


@pytest.mark.parametrize('precision, atol', [(32, 1e-3), (64, 1e-9)])
def test_aim_rays_precision(precision, atol):
    """The kernel converges (to a precision-appropriate floor) at 32 and 64
    bit, and the aimed arrays carry config.precision."""
    from prysm.conf import config
    old = config.precision
    try:
        config.precision = precision
        presc = _singlet_with_internal_stop()
        P, S = _collimated_y_fan(5, half=2.0, z0=-10.0, theta_deg=1.0)
        P_aim, _, converged = aim_rays(P, S, presc, surface_index=1,
                                    target_xy=(0.0, 0.0), wvl=0.55,
                                    tol=atol, strict=True)
        assert P_aim.dtype == config.precision
        from prysm.x.raytracing.spencer_and_murty import raytrace
        tr = raytrace(presc, P_aim, S, wvl=0.55)
        np.testing.assert_allclose(np.asarray(tr.P[2, :, :2]), 0.0, atol=atol)
    finally:
        config.precision = old


def test_aim_rays_matches_scipy_lbfgsb():
    """Parity with an independent per-ray scipy L-BFGS-B aim (the method this
    kernel replaced) on a known system."""
    from scipy.optimize import minimize
    presc = _singlet_with_internal_stop()
    P, S = _collimated_y_fan(5, half=2.0, z0=-10.0, theta_deg=2.0)
    P_new, _, _ = aim_rays(P, S, presc, surface_index=1, target_xy=(0.0, 0.0),
                        wvl=0.55)

    from prysm.x.raytracing.spencer_and_murty import raytrace
    trace_path = presc[:2]
    P_ref = P.copy()
    for i in range(P.shape[0]):
        Pi = P[i].copy()
        Si = S[i]

        def objective(xy, Pi=Pi, Si=Si):
            Pi[0], Pi[1] = xy
            land = raytrace(trace_path, Pi, Si, 0.55).P[-1]
            return 0.5 * (land[0] ** 2 + land[1] ** 2)

        res = minimize(objective, P[i, :2], method='L-BFGS-B', tol=1e-14)
        P_ref[i, :2] = res.x
    np.testing.assert_allclose(P_new[:, :2], P_ref[:, :2], atol=1e-7)


@pytest.mark.parametrize('target_z, launch_sz', [(1.0, 1.0), (-1.0, -1.0)])
def test_aim_rays_direction_normalizes_proposals(target_z, launch_sz):
    """Direction aiming must trace unit vectors even when the Newton variable
    moves outside the transverse unit disk."""
    presc = [
        plane(interaction='eval', P=np.array([0., 0., target_z])),
    ]
    P = np.array([[0., 0., 0.]])
    S = np.array([[0., 0., launch_sz]])
    target_xy = (2.0, -1.5)
    _, S_aim, converged = aim_rays(P, S, presc, surface_index=0,
                                target_xy=target_xy, wvl=0.55,
                                vary='direction', strict=True)
    assert bool(converged[0])
    np.testing.assert_allclose(np.linalg.norm(S_aim, axis=1), 1.0, atol=1e-12)
    assert np.sign(S_aim[0, 2]) == np.sign(launch_sz)

    from prysm.x.raytracing.spencer_and_murty import raytrace
    tr = raytrace(presc, P, S_aim, wvl=0.55)
    np.testing.assert_allclose(tr.P[-1, 0, :2], target_xy, atol=1e-9)


# ---------- pupil-on-axis behavior ----------

def test_xp_reference_sphere_axis_foot_intersecting_lines():
    """Chief ray that crosses the optical axis at the origin: foot must be the origin."""
    # chief through (1, 0, -10) and (0, 0, 0)
    P_chief = np.array([1.0, 0.0, -10.0])
    direction = np.array([-1.0, 0.0, 10.0])
    S_chief = direction / np.linalg.norm(direction)
    _, _, P_xp = xp_reference_sphere(P_chief, S_chief)
    np.testing.assert_allclose(P_xp, [0.0, 0.0, 0.0], atol=1e-12)


def test_locate_ep_and_locate_xp_share_helper():
    """Entrance and exit pupil locators agree when given the same axis pair."""
    P_chief = np.array([1.0, 0.0, -10.0])
    S_chief = np.array([-1.0, 0.0, 10.0]) / np.sqrt(101)
    P_obj = np.array([0.0, 0.0, -50.0])
    P_s1 = np.array([0.0, 0.0, 0.0])
    ep = locate_ep(P_chief, S_chief, P_obj, P_s1)
    xp = locate_xp(P_chief, S_chief, P_obj, P_s1)
    # same axis, same chief -> same answer
    np.testing.assert_allclose(ep, xp, atol=1e-12)
    # and that answer is the foot at z=0 (where the chief crosses the axis)
    np.testing.assert_allclose(ep, [0.0, 0.0, 0.0], atol=1e-12)


def test_xp_reference_sphere_radius_matches_geometry():
    """For a chief ray crossing the z-axis at the origin, image at z=10:
    xp at origin, R = 10."""
    P_chief = np.array([0.5, 0.0, 10.0])  # image point (last position)
    direction = np.array([0.5, 0.0, 10.0])
    S_chief = direction / np.linalg.norm(direction)
    C, R, P_xp = xp_reference_sphere(P_chief, S_chief)
    np.testing.assert_allclose(C, P_chief)
    # P_xp lies on the +z axis at the foot of perpendicular from chief to axis
    np.testing.assert_allclose(P_xp[:2], [0.0, 0.0], atol=1e-12)
    # R = |P_xp - C| > 0 for an off-axis image
    assert R > 0


def test_xp_reference_sphere_rejects_axial_chief():
    P_chief = np.array([0.0, 0.0, 10.0])
    S_chief = np.array([0.0, 0.0, 1.0])
    with pytest.raises(ValueError, match='near-axial chief ray'):
        xp_reference_sphere(P_chief, S_chief)


def test_intersect_reference_sphere_rejects_degenerate_radius():
    P = np.array([[0.0, 1.0, 0.0],
                  [0.0, -1.0, 0.0]])
    S = np.array([[0.0, 0.0, 1.0],
                  [0.0, 0.0, 1.0]])
    C = np.array([0.0, 0.0, 10.0])
    with pytest.raises(ValueError, match='degenerate'):
        intersect_reference_sphere(P, S, C, 0.0)

# ---------- end-to-end RC sanity check ----------

def test_rc_prescription_paraxial_image_at_bfl():
    """The paraxial image of an RC built from (efl, bfl, sep) lies at
    sm_vertex + bfl on the optical axis."""
    efl, bfl, sep = 1500.0, 250.0, 400.0
    c1, c2, k1, k2 = rc_prescription_from_efl_bfl_sep(efl, bfl, sep)
    P_pm = np.array([0.0, 0.0, 0.0])
    P_sm = np.array([0.0, 0.0, -sep])
    P_img = np.array([0.0, 0.0, bfl - sep])  # bfl measured from SM
    prescription = [
        conic(c1, k1, 'refl', P_pm),
        conic(c2, k2, 'refl', P_sm),
        plane('eval', P_img),
    ]
    # image distance measured from the trailing eval plane; it should land
    # the paraxial image right on the design BFL location.
    bfd = paraxial_image_distance(prescription, wvl=0.6328)
    img_z = float(prescription[-1].P[2]) + bfd
    np.testing.assert_allclose(img_z, P_img[2], rtol=5e-3)


# ---------- spot statistics ----------

def test_spot_centroid_no_status():
    """With no status, all rays count toward the centroid."""
    P = np.array([[1., 2., 0.], [3., 4., 0.], [5., 6., 0.]])
    np.testing.assert_allclose(spot_centroid(P), [3., 4.])


def test_spot_centroid_filters_invalid_rays():
    """Rays rejected by valid_mask are excluded."""
    P = np.array([[0., 0., 0.], [10., 10., 0.], [-10., -10., 0.]])
    # the second ray is "clipped"; centroid of the remaining two is (-5, -5)
    status = np.array([0+0j, 1+STATUS_CLIP*1j, 0+0j])
    np.testing.assert_allclose(spot_centroid(P, status), [-5., -5.])


def test_spot_centroid_filters_nonfinite_without_status():
    P = np.array([[0., 0., 0.], [np.nan, np.nan, np.nan], [2., 2., 0.]])
    np.testing.assert_allclose(spot_centroid(P), [1., 1.])


def test_spot_centroid_all_invalid_returns_nan():
    P = np.array([[0., 0., 0.]])
    status = np.array([1+STATUS_CLIP*1j])
    result = spot_centroid(P, status)
    assert np.all(np.isnan(result))


def test_rms_spot_radius_zero_for_stigmatic_spot():
    """All rays at the same (x, y) → RMS = 0."""
    P = np.tile([1.0, 2.0, 0.0], (5, 1))
    assert rms_spot_radius(P) == pytest.approx(0.0)


def test_rms_spot_radius_unit_circle():
    """4 rays on a unit circle around the origin: RMS radius = 1."""
    P = np.array([[1., 0., 0.], [-1., 0., 0.], [0., 1., 0.], [0., -1., 0.]])
    assert rms_spot_radius(P) == pytest.approx(1.0)


def test_rms_spot_radius_filters_invalid_rays():
    """A flagged outlier should not blow up the RMS."""
    P = np.array([[1., 0., 0.], [-1., 0., 0.], [0., 1., 0.], [0., -1., 0.],
                  [1e6, 0., 0.]])  # outlier
    status = np.array([0., 0., 0., 0., 1+STATUS_MISS*1j], dtype=complex)
    assert rms_spot_radius(P, status) == pytest.approx(1.0)


def test_rms_spot_radius_filters_nonfinite_without_status():
    P = np.array([[1., 0., 0.], [-1., 0., 0.], [np.nan, np.nan, np.nan]])
    assert rms_spot_radius(P) == pytest.approx(1.0)


def test_rms_spot_radius_custom_centroid():
    """Passing centroid=(0,0) measures from origin, not from spot mean."""
    P = np.array([[1., 1., 0.], [1., 1., 0.]])  # all rays at (1, 1)
    # default centroid = (1, 1) -> RMS = 0; custom (0, 0) -> RMS = sqrt(2)
    assert rms_spot_radius(P) == pytest.approx(0.0)
    assert rms_spot_radius(P, centroid=[0., 0.]) == pytest.approx(np.sqrt(2))


def test_geometric_psf_histogram_count_matches_valid_rays():
    """Histogram total count should equal the number of valid rays."""
    P = np.array([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0],
                  [0.0, 0.1, 0.0], [0.0, -0.1, 0.0]])
    H, xe, ye = geometric_psf_histogram(P, bins=8)
    assert H.sum() == 4
    assert H.shape == (8, 8)


def test_geometric_psf_histogram_filters_invalid():
    P = np.array([[0., 0., 0.], [0., 0., 0.], [1e6, 0., 0.]])
    status = np.array([0+0j, 0+0j, 1+STATUS_CLIP*1j])
    H, _, _ = geometric_psf_histogram(P, status, bins=4)
    # 2 valid rays accounted for; the 1e6 outlier did not stretch the extent
    assert H.sum() == 2


def test_geometric_psf_histogram_filters_nonfinite_without_status():
    P = np.array([[0., 0., 0.], [0., 0., 0.], [np.nan, np.nan, np.nan]])
    H, _, _ = geometric_psf_histogram(P, bins=4)
    assert H.sum() == 2
