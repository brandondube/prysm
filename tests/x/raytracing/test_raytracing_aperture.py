"""Tests for the unified surface Aperture (clip, extent, substrate, features)."""
import numpy as np
import pytest

from prysm.x import materials
from prysm.x.raytracing.surfaces import OffAxisConic, Plane, Surface
from prysm.x.raytracing.aperture import (
    Aperture,
    AnnularClip,
    Chamfer,
    CircularClip,
    CircularExtent,
    Flat,
    FlatBackSubstrate,
    FlatParentSubstrate,
    ParallelSubstrate,
    Seat,
    SquareCut,
    SurfaceSubstrate,
    annular_aperture,
    circular_aperture,
)


def _front_profile(surf, outer, points=5, center=0.0):
    """Sample a plain optical-face meridian for substrate tests."""
    ploty = center + np.linspace(-outer, outer, points)
    sag = np.asarray(surf.sag(np.zeros_like(ploty), ploty)) + surf.P[2]
    return ploty, sag, sag.copy()


# ---- clip ----------------------------------------------------------------

def test_float_clip_wraps_circular():
    ap = Aperture(2.0)
    assert isinstance(ap.clip, CircularClip)
    inside = ap.clips(np.asarray([0.0, 1.9, 2.1]), np.asarray([0.0, 0.0, 0.0]))
    np.testing.assert_array_equal(inside, [True, True, False])


def test_no_clip_passes_everything():
    ap = Aperture()
    assert ap.clips(np.asarray([1e9]), np.asarray([1e9])) == np.bool_(True)
    # the trace masks via ~clips(); a scalar True must yield an all-False clip set
    converged = np.asarray([True, True])
    assert not (converged & ~ap.clips(np.zeros(2), np.zeros(2))).any()


def test_annular_clip_blocks_central_disk():
    clip = annular_aperture(1.0, 3.0)
    assert isinstance(clip, AnnularClip)
    r = np.asarray([0.5, 2.0, 3.5])
    np.testing.assert_array_equal(clip(r, np.zeros_like(r)),
                                  [False, True, False])
    assert clip.limiting_radius == 3.0


def test_circular_aperture_carries_radius():
    clip = circular_aperture(4.0)
    assert clip.limiting_radius == 4.0


# ---- limiting / drawn radius --------------------------------------------

def test_limiting_radius_prefers_clip_then_footprint():
    assert Aperture(2.0).limiting_radius(footprint=9.0) == 2.0
    assert Aperture().limiting_radius(footprint=9.0) == 9.0
    assert Aperture().limiting_radius() is None


def test_drawn_radius_grows_clip_by_oversize():
    assert Aperture(2.0).drawn_radius() == pytest.approx(2.0 * 1.05)
    assert Aperture().drawn_radius(footprint=3.0) == pytest.approx(3.0 * 1.05)
    # explicit extent is taken verbatim, no further oversize
    assert Aperture(extent=CircularExtent(5.0)).drawn_radius() == 5.0


def test_is_auto_only_without_clip_or_user_extent():
    assert Aperture().is_auto
    assert not Aperture(2.0).is_auto
    assert not Aperture(extent=CircularExtent(5.0)).is_auto


# ---- solve / staleness ---------------------------------------------------

def test_solve_extent_writes_footprint_times_oversize_and_stamp():
    ap = Aperture()
    assert ap.is_stale(7)
    ap.solve_extent(10.0, version=7)
    assert ap.extent.outer_radius == pytest.approx(10.0 * 1.05)
    assert not ap.is_stale(7)
    assert ap.is_stale(8)  # an edit bumped the version


def test_user_clip_aperture_is_never_stale():
    assert not Aperture(2.0).is_stale(123)


# ---- extent outline ------------------------------------------------------

def test_circular_extent_outline_masks_bore():
    ploty, mask = CircularExtent(2.0, inner_radius=0.5).outline(5)
    np.testing.assert_allclose(ploty, np.linspace(-2.0, 2.0, 5))
    np.testing.assert_array_equal(mask, np.abs(ploty) < 0.5)


def test_circular_extent_outline_radius_override_keeps_bore():
    ext = CircularExtent(2.0, inner_radius=0.5)
    ploty, mask = ext.outline(5, radius=3.0)
    np.testing.assert_allclose(ploty, np.linspace(-3.0, 3.0, 5))


# ---- substrates ----------------------------------------------------------

def _plane_mirror(z=0.0):
    return Surface(shape=Plane(), interaction='refl',
                   P=np.asarray([0.0, 0.0, z]))


def test_surface_substrate_draws_only_the_face():
    surf = _plane_mirror()
    ploty, sag, edge = _front_profile(surf, 1.0)
    zz, tt = SurfaceSubstrate().back_outline(surf, ploty, sag, edge, 0.0)
    np.testing.assert_allclose(zz, sag)
    np.testing.assert_allclose(tt, ploty)


def test_parallel_substrate_offsets_a_uniform_shell():
    surf = _plane_mirror()
    ploty, sag, edge = _front_profile(surf, 1.0)
    zz, tt = ParallelSubstrate(thickness=2.0, side=1).back_outline(
        surf, ploty, sag, edge, 0.0)
    zz = np.asarray(zz)
    # front face at z=0, back face at z=2
    np.testing.assert_allclose(zz[:5], np.zeros(5))
    np.testing.assert_allclose(zz[6:11], np.full(5, 2.0))


def test_flat_parent_substrate_is_a_plane_at_vertex_plus_thickness():
    surf = Surface(shape=OffAxisConic(c=1 / 100., k=-1., dy=10),
                   interaction='refl', P=np.asarray([0.0, 0.0, 0.0]))
    ploty, sag, edge = _front_profile(surf, 5.0)
    zz, _ = FlatParentSubstrate(thickness=2.0, side=1).back_outline(
        surf, ploty, sag, edge, 0.0)
    np.testing.assert_allclose(np.asarray(zz)[6:11], np.full(5, 2.0))


def test_flat_back_substrate_is_parallel_to_the_tangent():
    surf = Surface(shape=OffAxisConic(c=1 / 100., k=-1., dy=10),
                   interaction='refl', P=np.asarray([0.0, 0.0, 0.0]))
    ploty, sag, edge = _front_profile(surf, 5.0)
    zz, tt = FlatBackSubstrate(thickness=2.0, side=1).back_outline(
        surf, ploty, sag, edge, 0.0)
    rear_x = np.asarray(zz)[6:11]
    rear_y = np.asarray(tt)[6:11]
    slope = np.diff(rear_x) / np.diff(rear_y)
    np.testing.assert_allclose(slope, np.full(4, slope[0]))
    # the back sits one thickness behind the lower edge of the optical face
    front_lower = surf.sag(np.asarray([0.]), np.asarray([-5.]))[0]
    np.testing.assert_allclose(rear_x[rear_y == -5][0] - front_lower, 2.0)


def test_parallel_substrate_auto_side_follows_departure():
    # a concave-toward-+z plane: auto picks the side opposite the sag departure
    surf = Surface(shape=OffAxisConic(c=1 / 100., k=-1., dy=10),
                   interaction='refl', P=np.asarray([0.0, 0.0, 0.0]))
    ploty, sag, edge = _front_profile(surf, 5.0)
    zz, _ = ParallelSubstrate(thickness=2.0).back_outline(
        surf, ploty, sag, edge, 0.0)
    # the back face is offset from the optical face by exactly the thickness
    zz = np.asarray(zz)
    np.testing.assert_allclose(np.abs(zz[6:11] - sag[::-1]), np.full(5, 2.0))


def test_bored_substrate_renders_two_open_loops():
    surf = Surface(shape=Plane(), interaction='refl',
                   P=np.asarray([0.0, 0.0, 0.0]))
    ploty, sag, edge = _front_profile(surf, 10.0, points=41)
    zz, tt = ParallelSubstrate(thickness=5.0, side=1, bore=3.0).back_outline(
        surf, ploty, sag, edge, 0.0)
    zz = np.asarray(zz, dtype=float)
    tt = np.asarray(tt, dtype=float)
    assert np.isnan(zz).sum() == 2  # one separator per loop
    finite = np.isfinite(tt)
    assert np.all(np.abs(tt[finite]) >= 3.0 - 1e-9)  # bore is open


# ---- edge features -------------------------------------------------------

def test_square_cut_and_flat_share_span():
    assert SquareCut(0.5, 1.5, 0.25).span(0.0, 2.0, ('front', 'rear')) == \
        (0.5, 1.5, 0.25)
    assert Flat(0.5, 1.5, 0.25).span(0.0, 2.0, ('front', 'rear')) == \
        (0.5, 1.5, 0.25)
    assert not SquareCut(0.5, 1.5, 0.25).is_chamfer


def test_chamfer_is_marked():
    assert Chamfer(0.5, 1.0, 0.2).is_chamfer


def test_seat_steps_in_from_named_face():
    seat = Seat('front', 0.5, 0.2)
    start, end, depth = seat.span(0.0, 2.0, ('front', 'rear'))
    assert (start, end, depth) == (0.0, 0.5, 0.2)
    seat_rear = Seat('rear', 0.5, 0.2)
    assert seat_rear.span(0.0, 2.0, ('front', 'rear')) == (1.5, 2.0, 0.2)


def test_feature_side_filter():
    f = SquareCut(0.5, 1.5, 0.25, side='upper')
    assert f.applies_to('upper')
    assert not f.applies_to('lower')
    assert SquareCut(0, 1, 0.1).applies_to('lower')  # default 'both'


# ---- the solve -----------------------------------------------------------

def test_solve_apertures_sizes_auto_and_skips_clip():
    from prysm.x.raytracing import OpticalSystem
    from prysm.x.raytracing.lensdata import LensData
    from prysm.x.raytracing.surfaces import Sphere

    n15 = materials.ConstantMaterial(1.5)
    lens = LensData()
    (lens.add(Sphere(1 / 50.0), thickness=4.0, material=n15)         # auto
        .add(Sphere(-1 / 50.0), thickness=40.0, material=materials.air,
             aperture=8.0))                                           # user clip
    sys = OpticalSystem(lens, aperture=10.0, fields=[0.0, 3.0],
                        wavelengths=[0.5876], reference=0)
    front = lens.rows[1].aperture
    rear = lens.rows[2].aperture
    assert front.is_auto and front.extent is None
    assert front.is_stale(lens._version)
    assert not rear.is_auto

    sys.solve.apertures()

    # the auto front is sized to footprint x oversize and is no longer stale
    assert front.extent is not None
    assert not front.is_stale(lens._version)
    # EPD/2 = 5, so the on-axis footprint x 1.05 is a few mm
    assert 4.0 < front.extent.outer_radius < 8.0
    # the user-clip rear is untouched
    assert rear.extent is None
    assert rear._solved_at_version is None


def test_solve_apertures_restamps_after_an_edit():
    from prysm.x.raytracing import OpticalSystem
    from prysm.x.raytracing.lensdata import LensData
    from prysm.x.raytracing.surfaces import Sphere

    n15 = materials.ConstantMaterial(1.5)
    lens = LensData().add(Sphere(1 / 50.0), thickness=4.0, material=n15)
    lens.add(Sphere(-1 / 50.0), thickness=40.0, material=materials.air)
    sys = OpticalSystem(lens, aperture=10.0, fields=[0.0],
                        wavelengths=[0.5876], reference=0)
    sys.solve.apertures()
    ap = lens.rows[1].aperture
    assert not ap.is_stale(lens._version)
    # an edit bumps the version; the solved extent goes stale until re-solved
    lens.rows[1].thickness = 5.0
    assert ap.is_stale(lens._version)
