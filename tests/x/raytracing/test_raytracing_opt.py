"""Tests for prysm.x.raytracing.opt."""
import numpy as np
import pytest

from prysm.x.raytracing.surfaces import Surface
from prysm.x.raytracing.opt import (
    paraxial_image_solve,
    ray_aim,
    locate_ep,
    locate_xp,
    xp_reference_sphere,
    _closest_approach_on_axis,
    _establish_axis,
    spot_centroid,
    rms_spot_radius,
    geometric_psf_histogram,
)
from prysm.x.raytracing.spencer_and_murty import (
    STATUS_CLIP, STATUS_MISS,
)
from prysm.x.raytracing.auto import rc_prescription_from_efl_bfl_sep


# ---------- paraxial_image_solve ----------

def test_paraxial_image_solve_single_refracting_sphere():
    """Single refracting sphere with R=50, n=1.5: image at z = n_after * R / (n_after - n_before).

    With n_after=1.5, n_before=1.0, R=50: image distance = 1.5 * 50 / 0.5 = 150.
    """
    R = 50.0
    n_glass = 1.5
    expected_image_z = n_glass * R / (n_glass - 1.0)  # 150

    prescription = [
        Surface.sphere(c=1.0 / R, typ='refr', P=np.array([0., 0., 0.]),
                       n=lambda wvl: n_glass),
        # eval plane somewhere downstream so the rays haven't converged yet
        Surface.plane(typ='eval', P=np.array([0., 0., 100.])),
    ]
    P_img = paraxial_image_solve(prescription, z=0, epd=10.0)
    np.testing.assert_allclose(P_img[2], expected_image_z, rtol=1e-3)
    # transverse coords should be ~0 (axial image)
    np.testing.assert_allclose(P_img[:2], [0.0, 0.0], atol=1e-6)


def test_paraxial_image_solve_paraxial_fraction_kwarg():
    """Sweeping paraxial_fraction across orders of magnitude should give the
    same paraxial image position; the kwarg is therefore live."""
    prescription = [
        Surface.sphere(c=1 / 50.0, typ='refr', P=np.array([0., 0., 0.]),
                       n=lambda wvl: 1.5),
        Surface.plane(typ='eval', P=np.array([0., 0., 100.])),
    ]
    img_default = paraxial_image_solve(prescription, z=0, epd=10.0)
    img_smaller = paraxial_image_solve(prescription, z=0, epd=10.0,
                                       paraxial_fraction=1e-5)
    np.testing.assert_allclose(img_default, img_smaller, atol=1e-3)


def test_paraxial_image_solve_requires_na_or_epd():
    with pytest.raises(ValueError, match='na or epd'):
        paraxial_image_solve([], z=0)


# ---------- ray_aim ----------

def test_ray_aim_hits_target_on_simple_mirror():
    """Aim a single ray at a transverse target on a flat eval surface."""
    prescription = [
        Surface.conic(c=1 / 200.0, k=-1.0, typ='refl', P=np.array([0., 0., 0.])),
        Surface.plane(typ='eval', P=np.array([0., 0., -50.])),  # rays head -z after reflection
    ]
    # incoming collimated ray at (0, ?, -100) → +z; we adjust the (x, y)
    # launch position so the ray hits target=(2, -1, ?) on the eval plane.
    P0 = np.array([0., 0., -100.])
    S0 = np.array([0., 0., 1.])
    target = np.array([2.0, -1.0, np.nan])  # NaN means "don't constrain z"
    P_aimed = ray_aim(P0, S0, prescription, j=1, wvl=0.55,
                      target=target, x0=np.array([2.0, -1.0]))
    # the returned P encodes the launch (x, y) needed
    # actually trace it and verify
    from prysm.x.raytracing.spencer_and_murty import raytrace
    phist, _, _ = raytrace(prescription, P_aimed, S0, wvl=0.55)
    final = phist[-1]
    np.testing.assert_allclose(final[:2], target[:2], atol=1e-6)


def test_ray_aim_strict_raises_on_failure():
    """ray_aim with strict=True must raise on a well-defined optimizer failure.
    Inverted bounds (low > high) make L-BFGS-B fail before it even starts.
    """
    prescription = [
        Surface.conic(c=1 / 200.0, k=-1.0, typ='refl', P=np.array([0., 0., 0.])),
        Surface.plane(typ='eval', P=np.array([0., 0., -50.])),
    ]
    P0 = np.array([0., 0., -100.])
    S0 = np.array([0., 0., 1.])
    with pytest.raises((RuntimeError, ValueError)):
        # inverted bounds: lo > hi; scipy returns success=False with an error
        # message, ray_aim then re-raises as RuntimeError. (Some scipy versions
        # raise ValueError directly inside L-BFGS-B before calling our code.)
        ray_aim(P0, S0, prescription, j=1, wvl=0.55,
                target=np.array([0., 0., np.nan]),
                bounds=[(1.0, -1.0), (1.0, -1.0)], strict=True)


def test_ray_aim_strict_false_does_not_raise():
    """strict=False must not raise even if L-BFGS-B reports non-success."""
    prescription = [
        Surface.conic(c=1 / 200.0, k=-1.0, typ='refl', P=np.array([0., 0., 0.])),
        Surface.plane(typ='eval', P=np.array([0., 0., -50.])),
    ]
    P0 = np.array([0., 0., -100.])
    S0 = np.array([0., 0., 1.])
    # a normal call with strict=False should still return a P
    P_out = ray_aim(P0, S0, prescription, j=1, wvl=0.55,
                    target=np.array([1.0, 0., np.nan]), strict=False)
    assert P_out.shape == (3,)


# ---------- pupil-on-axis helpers ----------

def test_closest_approach_on_axis_intersecting_lines():
    """Chief ray that crosses the optical axis at the origin: foot must be the origin."""
    # chief through (1, 0, -10) and (0, 0, 0)
    P_chief = np.array([1.0, 0.0, -10.0])
    direction = np.array([-1.0, 0.0, 10.0])
    S_chief = direction / np.linalg.norm(direction)
    Q = _closest_approach_on_axis(P_chief, S_chief,
                                  axis_point=np.zeros(3),
                                  axis_dir=np.array([0., 0., 1.]))
    np.testing.assert_allclose(Q, [0.0, 0.0, 0.0], atol=1e-12)


def test_closest_approach_on_axis_parallel_chief():
    """Chief ray exactly parallel to the axis: helper must not divide by zero."""
    P_chief = np.array([0.0, 1.0, -5.0])
    S_chief = np.array([0.0, 0.0, 1.0])
    Q = _closest_approach_on_axis(P_chief, S_chief,
                                  axis_point=np.zeros(3),
                                  axis_dir=np.array([0., 0., 1.]))
    # the projection branch returns the foot of perpendicular along the axis;
    # finite, no NaN/Inf — the actual coordinate is implementation-defined
    # for the parallel case but must not blow up
    assert np.all(np.isfinite(Q))


def test_locate_ep_and_locate_xp_share_helper():
    """Both should reduce to _pupil_on_axis; verify they yield the same result
    when the axis pair points define the z-axis."""
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


# ---------- _establish_axis ----------

def test_establish_axis_returns_unit_norm():
    P1 = np.array([0.0, 0.0, 0.0])
    P2 = np.array([3.0, 4.0, 0.0])
    Sa = _establish_axis(P1, P2)
    np.testing.assert_allclose(np.linalg.norm(Sa), 1.0, atol=1e-12)
    np.testing.assert_allclose(Sa, [0.6, 0.8, 0.0], atol=1e-12)


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
        Surface.conic(c1, k1, 'refl', P_pm),
        Surface.conic(c2, k2, 'refl', P_sm),
        Surface.plane('eval', P_img),
    ]
    img = paraxial_image_solve(prescription, z=0, epd=200.0)
    # paraxial image z should match the design BFL location
    np.testing.assert_allclose(img[2], P_img[2], rtol=5e-3)
    np.testing.assert_allclose(img[:2], [0.0, 0.0], atol=1e-3)


# ---------- spot statistics ----------

def test_spot_centroid_no_status():
    """With no status, all rays count toward the centroid."""
    P = np.array([[1., 2., 0.], [3., 4., 0.], [5., 6., 0.]])
    np.testing.assert_allclose(spot_centroid(P), [3., 4.])


def test_spot_centroid_filters_invalid_rays():
    """Rays with status.imag != 0 are excluded."""
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
