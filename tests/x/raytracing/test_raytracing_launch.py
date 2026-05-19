"""Tests for the launch ergonomics layer."""
import numpy as np
import pytest

from prysm.x.raytracing.surfaces import Surface
from prysm.x.raytracing.spencer_and_murty import raytrace
from prysm.x.raytracing.launch import Field, Sampling, launch, aim_bundle_to_surface


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
    f = Field(hx=2.0, hy=0.0, kind='height')
    with pytest.raises(ValueError):
        f.angle_radians()


def test_field_rejects_bad_kind():
    with pytest.raises(ValueError):
        Field(kind='bogus')


def test_field_rejects_bad_unit():
    with pytest.raises(ValueError):
        Field(unit='furlongs')


# ---------- Sampling --------------------------------------------------------

def test_sampling_chief():
    s = Sampling.chief()
    P, S = s.build(extent=10.0)
    assert P.shape == (1, 3)
    np.testing.assert_array_equal(P[0], [0., 0., 0.])
    np.testing.assert_array_equal(S[0], [0., 0., 1.])


def test_sampling_fan_y_axis():
    s = Sampling.fan(n=11, axis='y')
    P, S = s.build(extent=5.0)
    assert P.shape == (11, 3)
    # azimuth=90 → fan in y
    np.testing.assert_allclose(P[:, 0], 0.0, atol=1e-12)
    assert P[0, 1] == pytest.approx(-5.0)
    assert P[-1, 1] == pytest.approx(5.0)


def test_sampling_fan_x_axis():
    s = Sampling.fan(n=7, axis='x')
    P, _ = s.build(extent=2.0)
    np.testing.assert_allclose(P[:, 1], 0.0, atol=1e-12)
    assert P[0, 0] == pytest.approx(-2.0)
    assert P[-1, 0] == pytest.approx(2.0)


def test_sampling_fan_rejects_bad_axis():
    with pytest.raises(ValueError):
        Sampling.fan(n=5, axis='z')


def test_sampling_cross_count_is_2n():
    s = Sampling.cross(n=11)
    P, _ = s.build(extent=3.0)
    assert P.shape == (22, 3)


def test_sampling_rect_grid_count():
    s = Sampling.rect(n=5)
    P, _ = s.build(extent=1.0)
    assert P.shape == (25, 3)


def test_sampling_hex_count():
    # 1 + 3 * nrings * (nrings + 1)
    s = Sampling.hex(nrings=3)
    P, _ = s.build(extent=10.0)
    assert P.shape[0] == 1 + 3 * 3 * 4


def test_sampling_unknown_kind_raises():
    s = Sampling('bogus')
    with pytest.raises(ValueError):
        s.build(extent=1.0)


# ---------- launch ---------------------------------------------------------

def _simple_collimating_mirror():
    """One concave mirror at z=0 with c<0 and rays at -z; image at z=-40."""
    c = -1 / 80.0
    f = 1.0 / (2.0 * c)  # = -40
    s = Surface.conic(c=c, k=-1.0, typ='refl', P=[0, 0, 0])
    img = Surface.plane(typ='eval', P=[0, 0, f])
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


def test_launch_rejects_height_field_in_v1():
    presc = _simple_collimating_mirror()
    with pytest.raises(NotImplementedError):
        launch(presc, Field(2.0, 0.0, kind='height'),
               0.55e-3, Sampling.chief(), epd=0.0)


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


# ---------- aim_bundle_to_surface ------------------------------------------

def _refractive_singlet_with_internal_stop(n_glass=1.5):
    """A two-surface lens with an extra plane between them acting as a stop."""
    s1 = Surface.conic(c=1 / 50.0, k=0.0, typ='refr',
                       P=[0, 0, 0], n=lambda w: n_glass)
    stop = Surface.plane(typ='eval', P=[0, 0, 2.5])
    s2 = Surface.conic(c=-1 / 50.0, k=0.0, typ='refr',
                       P=[0, 0, 5.0], n=lambda w: 1.0)
    img = Surface.plane(typ='eval', P=[0, 0, 100.0])
    return [s1, stop, s2, img]


def test_aim_chief_to_stop_center_lands_at_zero():
    presc = _refractive_singlet_with_internal_stop()
    # chief at modest field
    P, S = launch(presc, Field(0., 2., unit='deg'), 0.55,
                  Sampling.chief(), epd=4.0, pupil_z=-10.0)
    P_aim = aim_bundle_to_surface(P, S, presc, surface_index=1,
                                  target_xy=(0.0, 0.0),
                                  wavelength=0.55, tol=1e-12)
    trace = raytrace(presc, P_aim, S, 0.55)
    # at stop (surface index 1 in the prescription), x and y should be near 0
    np.testing.assert_allclose(trace.P[2, 0, :2], (0., 0.), atol=1e-7)


def test_aim_preserves_launch_z():
    """aim_bundle_to_surface must restore the launch z (legacy ray_aim resets to 0)."""
    presc = _refractive_singlet_with_internal_stop()
    P, S = launch(presc, Field(0., 1., unit='deg'), 0.55,
                  Sampling.fan(n=3), epd=4.0, pupil_z=-10.0)
    z_before = P[:, 2].copy()
    P_aim = aim_bundle_to_surface(P, S, presc, surface_index=1,
                                  target_xy=(0.0, 0.0),
                                  wavelength=0.55, tol=1e-9)
    np.testing.assert_array_equal(P_aim[:, 2], z_before)


def test_launch_with_aim_to_stop_chief_lands_at_zero():
    """launch(..., aim_to=...) integration smoke test."""
    presc = _refractive_singlet_with_internal_stop()
    P, S = launch(presc, Field(0., 1., unit='deg'), 0.55,
                  Sampling.chief(), epd=4.0, pupil_z=-10.0,
                  aim_to=1)
    trace = raytrace(presc, P, S, 0.55)
    np.testing.assert_allclose(trace.P[2, 0, :2], (0., 0.), atol=1e-7)
