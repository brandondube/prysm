"""Tests for the launch ergonomics layer."""
import numpy as np
import pytest

from prysm.x import materials
from tests.x.raytracing.surface_helpers import plane, conic

from prysm.x.raytracing.spencer_and_murty import raytrace
from prysm.x.raytracing.launch import Field, Sampling, launch


# ---------- Field -----------------------------------------------------------

def test_field_angle_radians_deg():
    f = Field(hx=10.0, hy=-5.0, kind='angle', unit='deg')
    ax, ay = f.angle_radians()
    np.testing.assert_allclose(ax, np.deg2rad(10.0))
    np.testing.assert_allclose(ay, np.deg2rad(-5.0))


def test_field_angle_radians_rad():
    f = Field(hx=0.1, hy=-0.05, kind='angle', unit='rad')
    ax, ay = f.angle_radians()
    np.testing.assert_allclose(ax, 0.1)
    np.testing.assert_allclose(ay, -0.05)


def test_field_height_rejects_angle_radians():
    f = Field(hx=2.0, hy=0.0, kind='height', object_z=-10.0)
    with pytest.raises(ValueError):
        f.angle_radians()


def test_field_height_requires_object_z():
    with pytest.raises(ValueError):
        Field(hx=2.0, hy=0.0, kind='height')


def test_field_rejects_bad_kind():
    with pytest.raises(ValueError):
        Field(kind='bogus')


def test_field_rejects_bad_unit():
    with pytest.raises(ValueError):
        Field(unit='furlongs')


def test_field_vignetting_stored_verbatim():
    # negative factors are Code V pupil expansion and tiny values are
    # harmless optimizer noise; both are kept as-is (the compression map
    # scales each side by 1 - factor)
    f = Field(0.0, 0.0, vignetting={'vuy': 0.3, 'vly': -0.25})
    assert f.vignetting == {'vux': 0.0, 'vlx': 0.0, 'vuy': 0.3, 'vly': -0.25}

    f = Field(0.0, 0.0, vignetting={'vux': 0.0, 'vuy': 0.0})
    assert f.vignetting is None  # all-zero -> no vignetting


def test_field_vignetting_rejects_degenerate_factor():
    with pytest.raises(ValueError):
        Field(0.0, 0.0, vignetting={'vuy': 1.0})


def test_vignetting_compresses_pupil_samples_per_side():
    from prysm.x.raytracing.launch import _apply_vignetting

    f = Field(0.0, 0.0, vignetting={'vux': 0.5, 'vlx': -0.5,
                                    'vuy': 0.3, 'vly': 0.1})
    xy = np.asarray([
        [1.0, 0.0],
        [-1.0, 0.0],
        [0.0, 1.0],
        [0.0, -1.0],
        [0.0, 0.0],
    ])
    out = _apply_vignetting(xy, f)
    # full length, each side scaled by 1 - factor, center (chief) untouched
    assert out.shape == xy.shape
    np.testing.assert_allclose(out[0], [0.5, 0.0])    # upper x compressed
    np.testing.assert_allclose(out[1], [-1.5, 0.0])   # lower x expanded
    np.testing.assert_allclose(out[2], [0.0, 0.7])
    np.testing.assert_allclose(out[3], [0.0, -0.9])
    np.testing.assert_allclose(out[4], [0.0, 0.0])


def test_sampling_points_scales_normalized_coordinates():
    xy = np.asarray([[0.0, 1.0], [0.5, -0.5], [0.0, 0.0]])
    s = Sampling.points(xy)
    np.testing.assert_allclose(s.build(4.0), xy * 4.0)


# ---------- Sampling --------------------------------------------------------

def test_sampling_chief():
    s = Sampling.chief()
    xy = s.build(extent=10.0)
    assert xy.shape == (1, 2)
    np.testing.assert_array_equal(xy[0], [0., 0.])


def test_sampling_fan_y_axis():
    s = Sampling.fan(n=11, axis='y')
    xy = s.build(extent=5.0)
    assert xy.shape == (11, 2)
    np.testing.assert_allclose(xy[:, 0], 0.0, atol=1e-12)
    assert xy[0, 1] == pytest.approx(-5.0)
    assert xy[-1, 1] == pytest.approx(5.0)


def test_sampling_fan_x_axis():
    s = Sampling.fan(n=7, axis='x')
    xy = s.build(extent=2.0)
    np.testing.assert_allclose(xy[:, 1], 0.0, atol=1e-12)
    assert xy[0, 0] == pytest.approx(-2.0)
    assert xy[-1, 0] == pytest.approx(2.0)


def test_sampling_fan_rejects_bad_axis():
    with pytest.raises(ValueError):
        Sampling.fan(n=5, axis='z')


def test_sampling_cross_count_is_2n():
    s = Sampling.cross(n=11)
    xy = s.build(extent=3.0)
    assert xy.shape == (22, 2)


def test_sampling_rect_grid_count():
    s = Sampling.rect(n=5)
    xy = s.build(extent=1.0)
    assert xy.shape == (25, 2)


def test_sampling_hex_count():
    # 1 + 3 * nrings * (nrings + 1)
    s = Sampling.hex(nrings=3)
    xy = s.build(extent=10.0)
    assert xy.shape == (1 + 3 * 3 * 4, 2)


def test_sampling_unknown_kind_raises():
    s = Sampling('bogus')
    with pytest.raises(ValueError):
        s.build(extent=1.0)


def test_sampling_obscuration_drops_central_samples():
    extent = 10.0
    eps = 0.3
    full = Sampling.hex(nrings=4).build(extent=extent)
    annular = Sampling.hex(nrings=4, obscuration=eps).build(extent=extent)
    r_full = np.hypot(full[:, 0], full[:, 1])
    r_ann = np.hypot(annular[:, 0], annular[:, 1])
    # only the samples inside the obscuration are removed; the rest are kept
    assert len(annular) == int((r_full >= eps * extent).sum())
    assert len(annular) < len(full)
    assert r_ann.min() >= eps * extent - 1e-9


def test_sampling_fan_obscuration_opens_a_gap():
    xy = Sampling.fan(n=21, axis='y', obscuration=0.25).build(extent=4.0)
    assert np.all(np.abs(xy[:, 1]) >= 0.25 * 4.0 - 1e-9)


# ---------- launch ---------------------------------------------------------

def _simple_collimating_mirror():
    """One concave mirror at z=0 with c<0 and rays at -z; image at z=-40."""
    c = -1 / 80.0
    f = 1.0 / (2.0 * c)  # = -40
    s = conic(c=c, k=-1.0, interaction='refl', P=[0, 0, 0])
    img = plane(interaction='eval', P=[0, 0, f])
    return [s, img]


def test_launch_chief_direction_zero_field_is_pure_z():
    presc = _simple_collimating_mirror()
    P, S = launch(presc, Field(0., 0.), 0.55e-3, Sampling.chief(), epd=0.0)
    assert P.shape == (1, 3) and S.shape == (1, 3)
    np.testing.assert_allclose(S[0], [0., 0., 1.], atol=1e-15)


def test_launch_field_tilt_in_y_sets_Sy():
    presc = _simple_collimating_mirror()
    P, S = launch(presc, Field(0., 5., unit='deg'), 0.55e-3,
                  Sampling.chief(), epd=0.0)
    np.testing.assert_allclose(S[0, 0], 0.0, atol=1e-12)
    np.testing.assert_allclose(S[0, 1], np.sin(np.deg2rad(5.0)))


def test_launch_pupil_z_defaults_to_first_surface():
    presc = _simple_collimating_mirror()
    P, _ = launch(presc, Field(0., 0.), 0.55e-3,
                  Sampling.fan(n=5), epd=10.0)
    np.testing.assert_array_equal(P[:, 2], presc[0].P[2])


def test_launch_pupil_z_override():
    presc = _simple_collimating_mirror()
    P, _ = launch(presc, Field(0., 0.), 0.55e-3,
                  Sampling.chief(), epd=0.0, pupil_z=-50.0)
    assert P[0, 2] == -50.0


def test_launch_requires_epd_for_non_chief():
    presc = _simple_collimating_mirror()
    with pytest.raises(ValueError):
        launch(presc, Field(0., 0.), 0.55e-3, Sampling.fan(n=5))


def test_launch_pupil_extent_overrides_epd():
    presc = _simple_collimating_mirror()
    P, _ = launch(presc, Field(0., 0.), 0.55e-3,
                  Sampling.fan(n=5, axis='y'), epd=2.0,
                  pupil_extent=7.0)
    assert P[:, 1].max() == pytest.approx(7.0)


def test_launch_finite_conjugate_starts_at_object_point():
    presc = _simple_collimating_mirror()
    field = Field(0.5, -0.25, kind='height', object_z=-20.0)
    P, S = launch(presc, field, 0.55e-3, Sampling.rect(n=4), epd=2.0)
    np.testing.assert_allclose(P[:, 0], 0.5)
    np.testing.assert_allclose(P[:, 1], -0.25)
    np.testing.assert_allclose(P[:, 2], -20.0)
    np.testing.assert_allclose(np.linalg.norm(S, axis=-1), 1.0)


def test_launch_finite_conjugate_directions_pass_through_pupil():
    presc = _simple_collimating_mirror()
    field = Field(0.0, 0.0, kind='height', object_z=-15.0)
    P, S = launch(presc, field, 0.55e-3, Sampling.rect(n=3), epd=2.0,
                  pupil_z=0.0)
    dt = (0.0 - P[:, 2]) / S[:, 2]
    arrived = P + dt[:, np.newaxis] * S
    np.testing.assert_allclose(arrived[:, :2],
                               Sampling.rect(n=3).build(1.0),
                               atol=1e-12)


def test_launch_collimated_beam_traces_to_focus():
    """A collimated y-fan launched into the parabolic mirror collapses to a
    near-zero spot at z=-40 (the parabola focus)."""
    presc = _simple_collimating_mirror()
    P, S = launch(presc, Field(0., 0.), 0.55e-3,
                  Sampling.fan(n=11), epd=10.0, pupil_z=-50.0)
    trace = raytrace(presc, P, S, 0.55e-3)
    # at the image plane (last surface), all rays should be near (0, 0)
    spot_y = trace.P[-1, :, 1]
    assert float(np.max(np.abs(spot_y))) < 1e-10


# ---------- aim-to-surface -------------------------------------------------

def _refractive_singlet_with_internal_stop(n_glass=1.5):
    """A two-surface lens with an extra plane between them acting as a stop."""
    s1 = conic(c=1 / 50.0, k=0.0, interaction='refr',
                       P=[0, 0, 0], material=materials.ConstantMaterial(n_glass))
    stop = plane(interaction='eval', P=[0, 0, 2.5])
    s2 = conic(c=-1 / 50.0, k=0.0, interaction='refr',
                       P=[0, 0, 5.0], material=materials.air)
    img = plane(interaction='eval', P=[0, 0, 100.0])
    return [s1, stop, s2, img]


def test_launch_with_aim_to_stop_chief_lands_at_zero():
    """launch(..., aim_to=...) integration smoke test."""
    presc = _refractive_singlet_with_internal_stop()
    P, S = launch(presc, Field(0., 1., unit='deg'), 0.55,
                  Sampling.chief(), epd=4.0, pupil_z=-10.0,
                  aim_to=1)
    trace = raytrace(presc, P, S, 0.55)
    np.testing.assert_allclose(trace.P[2, 0, :2], (0., 0.), atol=1e-7)


def test_launch_aim_to_finite_bundle_keeps_object_point():
    """Finite-conjugate explicit aiming varies direction, not object point."""
    presc = _refractive_singlet_with_internal_stop()
    fld = Field(0.0, 1.0, kind='height', object_z=-10.0)
    P, S = launch(presc, fld, 0.55, Sampling.fan(n=3), epd=10.0,
                  aim_to=1, aim_strict=False)
    np.testing.assert_allclose(P, np.array([[0.0, 1.0, -10.0]] * 3),
                               atol=1e-12)
    trace = raytrace(presc, P, S, 0.55)
    np.testing.assert_allclose(trace.P[2, :, :2], 0.0, atol=1e-7)
