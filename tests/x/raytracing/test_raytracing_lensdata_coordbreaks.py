"""Tests for the general coordinate-break pose scan and break kinds.

Covers basic, decenter-and-return, return, reverse, bend, and mirror folding.
"""

import numpy as np
import pytest

from prysm.x import materials
from prysm.coordinates import make_rotation_matrix
from prysm.x.raytracing import LensData, raytrace, valid_mask
from prysm.x.raytracing.lensdata import R_rh, _ben_auto_gamma
from prysm.x.raytracing.raygen import generate_collimated_ray_fan
from prysm.x.raytracing.surfaces import Conic, Plane


_air = materials.air


# ---------------------------------------------------------------------------
# basic break
# ---------------------------------------------------------------------------

def test_single_tilt_break_matches_surface_tilt_convention():
    ld = (LensData()
          .add_coordbreak(tilt=(0.0, 7.0, 3.0))
          .add(Plane(), typ='eval'))
    R = np.asarray(ld.surfaces[1].R)   # surfaces[0] is OBJECT (before the break)
    # the kernel consumes surf.R as global->local; a tilt CB must reproduce the
    # existing Surface(tilt=...) convention exactly
    np.testing.assert_allclose(R, np.asarray(R_rh(0, 7, 3)))
    np.testing.assert_allclose(R, make_rotation_matrix((0, 7, 3)))


def test_basic_decenter_shifts_origin_along_local_axes():
    ld = (LensData()
          .add_coordbreak(decenter=(2.0, 3.0, 0.0), thickness=5.0)
          .add(Plane(), typ='eval'))
    np.testing.assert_allclose(np.asarray(ld.surfaces[1].P), [2.0, 3.0, 5.0])


def test_basic_tilt_is_cumulative_for_downstream_surfaces():
    ld = (LensData()
          .add(Plane(), typ='refr', material=_air, thickness=2.0)
          .add_coordbreak(tilt=(0.0, 10.0, 0.0), thickness=1.0, kind='basic')
          .add(Plane(), typ='refr', material=_air, thickness=2.0)
          .add(Plane(), typ='eval'))
    s = ld.surfaces       # s[0] is OBJECT; s[1] is the pre-break refr plane
    assert s[1].R is None
    # both surfaces after the break carry the tilt
    np.testing.assert_allclose(np.asarray(s[2].R), np.asarray(R_rh(0, 10, 0)))
    np.testing.assert_allclose(np.asarray(s[3].R), np.asarray(R_rh(0, 10, 0)))


def test_identity_break_does_not_change_axial_layout():
    # a zero break routes through the general path but must agree with the
    # axial path it would otherwise take
    base = (LensData()
            .add(Conic(1 / 50.0, 0.0), typ='refr', material=_air,
                 thickness=4.0)
            .add(Plane(), typ='eval'))
    withcb = (LensData()
              .add(Conic(1 / 50.0, 0.0), typ='refr', material=_air,
                   thickness=4.0)
              .add_coordbreak(thickness=0.0)
              .add(Plane(), typ='eval'))
    for a, b in zip(base.surfaces, withcb.surfaces):
        np.testing.assert_allclose(np.asarray(a.P), np.asarray(b.P))
        assert (a.R is None) == (b.R is None)


# ---------------------------------------------------------------------------
# rev break
# ---------------------------------------------------------------------------

def test_rev_inverts_a_matching_basic():
    ld = (LensData()
          .add(Plane(), typ='refr', material=_air)
          .add_coordbreak(decenter=(1.0, 2.0, 0.0), tilt=(0.0, 10.0, 5.0),
                          kind='basic')
          .add_coordbreak(decenter=(1.0, 2.0, 0.0), tilt=(0.0, 10.0, 5.0),
                          kind='rev')
          .add(Plane(), typ='eval'))
    s = ld.surfaces       # s[0]=OBJECT, s[1]=refr, s[2]=post-rev eval
    # after basic + matching rev, the frame returns to the origin/identity
    np.testing.assert_allclose(np.asarray(s[2].P), [0.0, 0.0, 0.0], atol=1e-12)
    assert s[2].R is None


# ---------------------------------------------------------------------------
# ret break
# ---------------------------------------------------------------------------

def test_ret_restores_a_named_prior_frame():
    ld = (LensData()
          .add(Plane(), typ='refr', material=_air, thickness=3.0)  # row 1
          .add_coordbreak(tilt=(0.0, 20.0, 0.0), thickness=4.0, kind='basic')
          .add(Plane(), typ='refr', material=_air, thickness=2.0)  # row 3
          .add_coordbreak(kind='ret', ret_target=1)
          .add(Plane(), typ='eval'))                               # row 5
    s = ld.surfaces  # s0=OBJECT, row1->s1, row3->s2, row5->s3
    assert s[2].R is not None  # the intervening surface is tilted
    np.testing.assert_allclose(np.asarray(s[3].P), np.asarray(s[1].P))
    assert s[3].R is None


def test_ret_with_unplaced_target_raises():
    ld = (LensData()
          .add(Plane(), typ='refr', material=_air)
          .add_coordbreak(kind='ret', ret_target=7)
          .add(Plane(), typ='eval'))
    with pytest.raises(ValueError):
        ld.to_surfaces()


# ---------------------------------------------------------------------------
# dar break
# ---------------------------------------------------------------------------

def test_dar_applies_to_next_surface_only():
    ld = (LensData()
          .add(Plane(), typ='refr', material=_air, thickness=2.0)
          .add_coordbreak(tilt=(0.0, 15.0, 0.0), thickness=3.0, kind='dar')
          .add(Plane(), typ='refr', material=_air, thickness=2.0)  # tilted
          .add(Plane(), typ='eval'))                               # NOT tilted
    s = ld.surfaces       # s[0]=OBJECT, s[1]=refr, s[2]=dar-tilted, s[3]=eval
    assert s[2].R is not None
    assert s[3].R is None


# ---------------------------------------------------------------------------
# ben break + mirror fold
# ---------------------------------------------------------------------------

def test_ben_auto_gamma_matches_manual():
    assert float(_ben_auto_gamma(45, 45)) == pytest.approx(-19.471, abs=1e-3)
    assert float(_ben_auto_gamma(30, 0)) == pytest.approx(0.0, abs=1e-12)
    assert float(_ben_auto_gamma(0, 22)) == pytest.approx(0.0, abs=1e-12)


def test_ben_90_degree_fold_places_and_traces_centered():
    ld = (LensData()
          .add(Plane(), typ='refr', material=_air, thickness=10.0)
          .add_coordbreak(tilt=(0.0, 0.0, 45.0), kind='ben')
          .add(Plane(), typ='refl', thickness=8.0)
          .add(Plane(), typ='eval'))
    s = ld.surfaces       # s[0]=OBJECT, s[1]=refr, s[2]=mirror, s[3]=eval
    np.testing.assert_allclose(np.asarray(s[2].P), [0.0, 0.0, 10.0], atol=1e-9)
    # 45 deg tilt about x bends the +z axis to +y; eval sits 8 mm along +y
    np.testing.assert_allclose(np.asarray(s[3].P), [0.0, 8.0, 10.0], atol=1e-9)

    P0, S0 = generate_collimated_ray_fan(7, maxr=2.0, z=-5.0)
    r = raytrace(ld, P0, S0, wvl=0.55)
    assert valid_mask(r.status, r.P[-1]).all()
    # frame fold agrees with the kernel reflection: rays land on the folded
    # eval plane and the chief lands at its origin
    np.testing.assert_allclose(np.asarray(r.P[-1])[:, 1], 8.0, atol=1e-9)
    chief = np.asarray(r.P[-1])[3]
    np.testing.assert_allclose(chief, [0.0, 8.0, 10.0], atol=1e-9)


def test_normal_incidence_mirror_fold_in_general_path():
    # a fold (reverse-z) at an on-axis mirror, reached through the general path
    # via a (zero) coordinate break, steps downstream to decreasing global z
    ld = (LensData()
          .add_coordbreak(thickness=0.0)  # forces the general path
          .add(Conic(1 / 200.0, -1.0), typ='refl', thickness=50.0)
          .add(Plane(), typ='eval'))
    s = ld.surfaces       # s[0]=OBJECT, s[1]=mirror, s[2]=eval
    np.testing.assert_allclose(np.asarray(s[1].P), [0.0, 0.0, 0.0])
    np.testing.assert_allclose(np.asarray(s[2].P), [0.0, 0.0, -50.0])


def test_unknown_coordbreak_kind_raises():
    ld = (LensData()
          .add(Plane(), typ='refr', material=_air)
          .add_coordbreak(kind='bogus')
          .add(Plane(), typ='eval'))
    with pytest.raises(ValueError):
        ld.to_surfaces()
