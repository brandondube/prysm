"""Tests for pickups (dependent DOFs) and the paraxial image-distance
solve, both resolved on compile so layout/trace/merit always agree."""

import numpy as np
import pytest

from prysm.x.raytracing import OpticalSystem
from prysm.x.raytracing import FRAUNHOFER_LINES_UM, LensData
from prysm.x import materials
from prysm.x.raytracing.design import EFL, Problem
from prysm.x.raytracing.paraxial import (
    effective_focal_length,
    paraxial_image_distance,
)
from prysm.x.raytracing.surfaces import Conic, EvenAsphere, Plane


n_bk7 = materials.ConstantMaterial(1.5168, name='N-BK7')


def make_singlet(c0=1 / 102.0, c1=-1 / 102.0, gap=95.0):
    # OBJECT/IMAGE endpoints are implicit (ADR-0006); the two conics are rows
    # 1 and 2, and the IMAGE plane is always present.
    lens = LensData()
    (lens.add(Conic(c0, 0.0), thickness=6.0, material=n_bk7,
              semidiameter=10.0)
         .add(Conic(c1, 0.0), thickness=gap, material=materials.air,
              semidiameter=10.0))
    return OpticalSystem(lens, aperture=20.0, wavelengths=list(FRAUNHOFER_LINES_UM.values()),
                         reference=1)


# ---------------------------------------------------------------------------
# pickups
# ---------------------------------------------------------------------------

def test_symmetry_pickup_freezes_dependent_and_follows_source():
    ld = make_singlet(c1=0.0)
    ld.opt.pickup('curvature', 2, from_surface=1, scale=-1.0)
    ld.opt.vary('curvature', surfaces=[1, 2])
    # the dependent (surface 2) curvature is frozen -> only surface 1 is free
    assert len(ld.opt.pack()) == 1
    s = ld.surfaces
    assert s[2].params['c'] == pytest.approx(-s[1].params['c'])


def test_pickup_tracks_source_under_update():
    ld = make_singlet(c1=0.0)
    ld.opt.pickup('curvature', 2, from_surface=1, scale=-1.0)
    ld.opt.vary('curvature', surfaces=1)
    ld.opt.update(np.array([1 / 80.0]))
    s = ld.surfaces
    assert s[1].params['c'] == pytest.approx(1 / 80.0)
    assert s[2].params['c'] == pytest.approx(-1 / 80.0)


def test_pickup_with_scale_and_offset():
    ld = make_singlet()
    ld.opt.pickup('thickness', 2, from_surface=1, scale=2.0, offset=1.0)
    ld.opt.vary('thickness', surfaces=1)
    ld.opt.update(np.array([4.0]))
    # thickness on the second conic = 2*4 + 1 = 9
    assert ld.rows[2].thickness == pytest.approx(9.0)


def test_pickup_length_mismatch_raises():
    coefs = (1e-4, -2e-6)
    ld = OpticalSystem(LensData()
          .add(EvenAsphere(1 / 50.0, 0.0, coefs), thickness=2.0,
               material=n_bk7, semidiameter=8.0)
          .add(Conic(1 / 80.0, 0.0), thickness=2.0, material=materials.air,
               semidiameter=8.0))
    with pytest.raises(ValueError):
        # 2 coefs cannot be picked up from 1 curvature (rows[0] is OBJECT)
        ld.opt.pickup('coefs', 1, from_surface=2, from_category='curvature')


def test_coef_symmetry_pickup_elementwise():
    coefs = (1e-4, -2e-6, 3e-9)
    ld = OpticalSystem(LensData()
          .add(EvenAsphere(1 / 50.0, 0.0, coefs), thickness=2.0,
               material=n_bk7, semidiameter=8.0)
          .add(EvenAsphere(1 / 50.0, 0.0, (0.0, 0.0, 0.0)), thickness=2.0,
               material=materials.air, semidiameter=8.0))
    ld.opt.pickup('coefs', 2, from_surface=1, scale=-1.0)
    np.testing.assert_allclose(
        np.asarray(ld.surfaces[2].params['coefs']),
        [-1e-4, 2e-6, -3e-9])


# ---------------------------------------------------------------------------
# image-distance solve
# ---------------------------------------------------------------------------

def test_image_solve_places_eval_at_paraxial_image():
    ld = make_singlet(gap=10.0)
    ld.solve.image_distance()
    s = ld.surfaces
    lens = s[:-1]                          # everything up to (not incl.) IMAGE
    pid = float(paraxial_image_distance(lens, wvl=ld.wavelength()))
    gap = float(np.asarray(s[-1].P)[2]) - float(np.asarray(s[-2].P)[2])
    assert gap == pytest.approx(pid)


def test_image_solve_preserves_leading_object_medium():
    # the object-space medium now lives on the OBJECT row (ADR-0006)
    lens = LensData()
    lens.object_row.material = materials.ConstantMaterial(1.33)
    lens.object_row.thickness = 40.0
    (lens.add(Conic(1 / 100.0, 0.0), thickness=5.0, material=n_bk7,
              semidiameter=10.0)
         .add(Conic(-1 / 100.0, 0.0), thickness=10.0,
              material=materials.air, semidiameter=10.0))
    sys = OpticalSystem(lens, aperture=20.0, wavelengths=list(FRAUNHOFER_LINES_UM.values()),
                        reference=1)
    sys.solve.image_distance()
    s = sys.surfaces
    wvl = sys.wavelength()
    expected = float(paraxial_image_distance(s[:-1], wvl=wvl))
    gap = float(np.asarray(s[-1].P)[2]) - float(np.asarray(s[-2].P)[2])
    assert gap == pytest.approx(expected)


def test_image_solve_freezes_the_solved_gap():
    ld = make_singlet(gap=10.0)
    ld.opt.vary('thickness', surfaces=2)   # the last powered gap (rows[0]=OBJECT)
    assert len(ld.opt.pack()) == 1
    ld.solve.image_distance(surface=2)
    assert len(ld.opt.pack()) == 0  # the solved gap is frozen


def test_clear_image_solve_releases_the_solved_gap():
    ld = make_singlet(gap=10.0)
    ld.solve.image_distance(surface=2)
    assert len(ld.opt.pack()) == 0
    ld.solve.clear_image_distance()
    assert ld._design._image_solve is None
    ld.opt.vary('thickness', surfaces=2)
    assert len(ld.opt.pack()) == 1


def test_vary_thickness_clears_matching_image_solve():
    ld = make_singlet(gap=10.0)
    ld.solve.image_distance(surface=2)
    ld.opt.vary('thickness', surfaces=2)
    assert ld._design._image_solve is None
    assert len(ld.opt.pack()) == 1


def test_image_solve_tracks_curvature_changes():
    ld = make_singlet(gap=10.0)
    ld.solve.image_distance()
    ld.opt.vary('curvature', surfaces=1)   # front conic (rows[0]=OBJECT)
    ld.opt.update(np.array([1 / 70.0]))  # stronger front -> shorter back focus
    s = ld.surfaces
    lens = s[:-1]
    pid = float(paraxial_image_distance(lens, wvl=ld.wavelength()))
    gap = float(np.asarray(s[-1].P)[2]) - float(np.asarray(s[-2].P)[2])
    assert gap == pytest.approx(pid)


def test_image_solve_without_powered_surface_raises():
    # an empty system (only OBJECT/IMAGE endpoints) has nothing to focus
    ld = OpticalSystem(LensData(), aperture=20.0,
                       wavelengths=list(FRAUNHOFER_LINES_UM.values()),
                       reference=1)
    with pytest.raises(ValueError):
        ld.solve.image_distance()


def test_solve_and_pickup_compose_in_optimization():
    # symmetric singlet (pickup) with an auto-focused image (solve); optimize
    # the shared curvature to hit a target EFL
    ld = make_singlet(c1=0.0, gap=10.0)
    ld.opt.pickup('curvature', 2, from_surface=1, scale=-1.0)
    ld.solve.image_distance()
    ld.opt.vary('curvature', surfaces=1)
    wvl = ld.wavelength()
    prob = Problem(ld, constraints=[EFL(wvl, target=120.0)])
    res = prob.solve(damping=1e-8, maxiter=10)
    assert res.success
    s = ld.surfaces
    lens = s[:-1]
    assert effective_focal_length(lens, wvl=wvl) == pytest.approx(120.0,
                                                                  rel=1e-5)
    # pickup still holds and image is still at paraxial focus after optimizing
    assert s[2].params['c'] == pytest.approx(-s[1].params['c'])
    pid = float(paraxial_image_distance(lens, wvl=wvl))
    gap = float(np.asarray(s[-1].P)[2]) - float(np.asarray(s[-2].P)[2])
    assert gap == pytest.approx(pid)


def test_copy_preserves_pickups_and_solves():
    ld = make_singlet(c1=0.0, gap=10.0)
    ld.opt.pickup('curvature', 2, from_surface=1, scale=-1.0)
    ld.solve.image_distance()
    clone = ld.copy()
    assert len(clone._design._pickups) == 1
    assert clone._design._image_solve is not None
    clone.opt.vary('curvature', surfaces=1).update(np.array([1 / 75.0]))
    assert clone.surfaces[2].params['c'] == pytest.approx(-1 / 75.0)
