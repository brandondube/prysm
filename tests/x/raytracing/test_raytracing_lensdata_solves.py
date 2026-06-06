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


def n_bk7(wvl):
    return 1.5168


def make_singlet(c0=1 / 102.0, c1=-1 / 102.0, gap=95.0, with_image=True):
    lens = LensData()
    (lens.add(Conic(c0, 0.0), thickness=6.0, material=n_bk7,
              semidiameter=10.0)
         .add(Conic(c1, 0.0), thickness=gap, material=materials.air,
              semidiameter=10.0))
    if with_image:
        lens.add(Plane(), typ='eval', material=materials.air,
                 semidiameter=10.0)
    return OpticalSystem(lens, aperture=20.0, wavelengths=FRAUNHOFER_LINES_UM,
                         reference_wavelength='d')


# ---------------------------------------------------------------------------
# pickups
# ---------------------------------------------------------------------------

def test_symmetry_pickup_freezes_dependent_and_follows_source():
    ld = make_singlet(c1=0.0, with_image=False)
    ld.lens.pickup('curvature', 1, from_surface=0, scale=-1.0)
    ld.lens.vary('curvature', surfaces=[0, 1])
    # the dependent (surface 1) curvature is frozen -> only surface 0 is free
    assert len(ld.lens.pack()) == 1
    s = ld.surfaces
    assert s[1].params['c'] == pytest.approx(-s[0].params['c'])


def test_pickup_tracks_source_under_update():
    ld = make_singlet(c1=0.0, with_image=False)
    ld.lens.pickup('curvature', 1, from_surface=0, scale=-1.0)
    ld.lens.vary('curvature', surfaces=0)
    ld.lens.update(np.array([1 / 80.0]))
    s = ld.surfaces
    assert s[0].params['c'] == pytest.approx(1 / 80.0)
    assert s[1].params['c'] == pytest.approx(-1 / 80.0)


def test_pickup_with_scale_and_offset():
    ld = make_singlet(with_image=False)
    ld.lens.pickup('thickness', 1, from_surface=0, scale=2.0, offset=1.0)
    ld.lens.vary('thickness', surfaces=0)
    ld.lens.update(np.array([4.0]))
    # thickness1 = 2*4 + 1 = 9
    assert ld.rows[1].thickness == pytest.approx(9.0)


def test_pickup_length_mismatch_raises():
    coefs = (1e-4, -2e-6)
    ld = (LensData()
          .add(EvenAsphere(1 / 50.0, 0.0, coefs), thickness=2.0,
               material=n_bk7, semidiameter=8.0)
          .add(Conic(1 / 80.0, 0.0), thickness=2.0, material=materials.air,
               semidiameter=8.0))
    with pytest.raises(ValueError):
        # 2 coefs cannot be picked up from 1 curvature
        ld.pickup('coefs', 0, from_surface=1, from_category='curvature')


def test_coef_symmetry_pickup_elementwise():
    coefs = (1e-4, -2e-6, 3e-9)
    ld = (LensData()
          .add(EvenAsphere(1 / 50.0, 0.0, coefs), thickness=2.0,
               material=n_bk7, semidiameter=8.0)
          .add(EvenAsphere(1 / 50.0, 0.0, (0.0, 0.0, 0.0)), thickness=2.0,
               material=materials.air, semidiameter=8.0))
    ld.pickup('coefs', 1, from_surface=0, scale=-1.0)
    np.testing.assert_allclose(
        np.asarray(ld.surfaces[1].params['coefs']),
        [-1e-4, 2e-6, -3e-9])


# ---------------------------------------------------------------------------
# image-distance solve
# ---------------------------------------------------------------------------

def test_image_solve_places_eval_at_paraxial_image():
    ld = make_singlet(gap=10.0)
    ld.solve_image_distance()
    s = ld.surfaces
    lens = [x for x in s if x.typ != -3]
    pid = float(paraxial_image_distance(lens, wvl=ld.wavelength('d')))
    gap = float(np.asarray(s[-1].P)[2]) - float(np.asarray(s[-2].P)[2])
    assert gap == pytest.approx(pid)


def test_image_solve_preserves_leading_object_eval_medium():
    lens = LensData()
    (lens.add(Plane(), typ='eval', material=lambda wvl: 1.33, thickness=40.0)
         .add(Conic(1 / 100.0, 0.0), thickness=5.0, material=n_bk7,
              semidiameter=10.0)
         .add(Conic(-1 / 100.0, 0.0), thickness=10.0,
              material=materials.air, semidiameter=10.0)
         .add(Plane(), typ='eval', material=materials.air,
              semidiameter=10.0))
    sys = OpticalSystem(lens, aperture=20.0, wavelengths=FRAUNHOFER_LINES_UM,
                        reference_wavelength='d')
    sys.solve_image_distance()
    s = sys.surfaces
    wvl = sys.wavelength('d')
    expected = float(paraxial_image_distance(s[:-1], wvl=wvl))
    gap = float(np.asarray(s[-1].P)[2]) - float(np.asarray(s[-2].P)[2])
    assert gap == pytest.approx(expected)


def test_image_solve_freezes_the_solved_gap():
    ld = make_singlet(gap=10.0)
    ld.lens.vary('thickness', surfaces=1)
    assert len(ld.lens.pack()) == 1
    ld.solve_image_distance(surface=1)
    assert len(ld.lens.pack()) == 0  # the solved gap is frozen


def test_image_solve_tracks_curvature_changes():
    ld = make_singlet(gap=10.0)
    ld.solve_image_distance()
    ld.lens.vary('curvature', surfaces=0)
    ld.lens.update(np.array([1 / 70.0]))  # stronger front -> shorter back focus
    s = ld.surfaces
    lens = [x for x in s if x.typ != -3]
    pid = float(paraxial_image_distance(lens, wvl=ld.wavelength('d')))
    gap = float(np.asarray(s[-1].P)[2]) - float(np.asarray(s[-2].P)[2])
    assert gap == pytest.approx(pid)


def test_image_solve_without_eval_plane_raises():
    ld = make_singlet(with_image=False)
    with pytest.raises(ValueError):
        ld.solve_image_distance()


def test_solve_and_pickup_compose_in_optimization():
    # symmetric singlet (pickup) with an auto-focused image (solve); optimize
    # the shared curvature to hit a target EFL
    ld = make_singlet(c1=0.0, gap=10.0)
    ld.lens.pickup('curvature', 1, from_surface=0, scale=-1.0)
    ld.solve_image_distance()
    ld.lens.vary('curvature', surfaces=0)
    wvl = ld.wavelength('d')
    prob = Problem(ld, equality_constraints=[EFL(wvl, target=120.0)])
    res = prob.solve(damping=1e-8, maxiter=10)
    assert res.success
    s = ld.surfaces
    lens = [x for x in s if x.typ != -3]
    assert effective_focal_length(lens, wvl=wvl) == pytest.approx(120.0,
                                                                  rel=1e-5)
    # pickup still holds and image is still at paraxial focus after optimizing
    assert s[1].params['c'] == pytest.approx(-s[0].params['c'])
    pid = float(paraxial_image_distance(lens, wvl=wvl))
    gap = float(np.asarray(s[-1].P)[2]) - float(np.asarray(s[-2].P)[2])
    assert gap == pytest.approx(pid)


def test_copy_preserves_pickups_and_solves():
    ld = make_singlet(c1=0.0, gap=10.0)
    ld.lens.pickup('curvature', 1, from_surface=0, scale=-1.0)
    ld.solve_image_distance()
    clone = ld.copy()
    assert len(clone.lens._pickups) == 1
    assert clone.lens._image_solve is not None
    clone.lens.vary('curvature', surfaces=0).update(np.array([1 / 75.0]))
    assert clone.surfaces[1].params['c'] == pytest.approx(-1 / 75.0)
