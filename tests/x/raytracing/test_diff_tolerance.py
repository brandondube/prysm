"""Perturbation-to-DiffSeed mapping vs FD sensitivity_table."""
import numpy as np
import pytest

from prysm.x import materials
from prysm.x.raytracing import OpticalSystem
from prysm.x.raytracing import LensData
from prysm.x.raytracing.launch import Field, Sampling, launch
from prysm.x.raytracing.surfaces import Conic, Plane
from prysm.x.raytracing.spencer_and_murty import _is_measurement_surf, raytrace
from prysm.x.raytracing.paraxial import paraxial_image_distance
from prysm.x.raytracing.design import WavefrontRMS
from prysm.x.raytracing.tolerance import (
    Perturbation, sensitivity_table,
)
from prysm.x.raytracing._diff_raytrace import (
    seed_from_perturbation, seeds_from_perturbations, wavefront_with_tangents,
)


WVL = 0.5
NG = 1.6


_glass = materials.ConstantMaterial(NG)


_air = materials.air


def _place_image(ld, gap_row):
    """Set gap_row's thickness to the paraxial image distance (fixed plane)."""
    lens = [s for s in ld.to_surfaces() if not _is_measurement_surf(s.typ)]
    bfd = float(paraxial_image_distance(lens, wvl=WVL))
    ld.rows[gap_row].thickness = bfd
    ld.lens._invalidate()
    return ld


def singlet():
    """Axial thick singlet, image plane at the paraxial focus."""
    lens = LensData()
    (lens.add(Conic(1 / 30.0, 0.0), typ='refr', thickness=4.0, material=_glass)
         .add(Conic(-1 / 30.0, 0.0), typ='refr', thickness=20.0, material=_air))
    ld = OpticalSystem(lens, aperture=10.0, wavelengths=[WVL])
    return _place_image(ld, gap_row=2)


def singlet_cb():
    """Same singlet with a basic coordinate break before S1."""
    lens = LensData()
    (lens.add(Conic(1 / 30.0, 0.0), typ='refr', thickness=4.0, material=_glass)
         .add_coordbreak(decenter=(0., 0., 0.), tilt=(0., 0., 0.),
                         kind='basic', thickness=0.0)
         .add(Conic(-1 / 30.0, 0.0), typ='refr', thickness=20.0, material=_air))
    ld = OpticalSystem(lens, aperture=10.0, wavelengths=[WVL])
    return _place_image(ld, gap_row=3)


def singlet_solved():
    """Axial singlet whose image gap is an image-distance solve (compensator)."""
    lens = LensData()
    (lens.add(Conic(1 / 30.0, 0.0), typ='refr', thickness=4.0, material=_glass)
         .add(Conic(-1 / 30.0, 0.0), typ='refr', thickness=20.0, material=_air))
    ld = OpticalSystem(lens, aperture=10.0, wavelengths=[WVL])
    return ld.solve.image_distance(wavelength=WVL)


def bundle(ld):
    """Diagonal off-axis collimated 2D grid."""
    return launch(ld, Field(2.5, 2.5), WVL, Sampling.rect(n=7),
                  epd=10.0, pupil_z=-5.0)


# ---------- sensitivity engines --------------------------------------------

def wd_rms_sensitivities(ld, P, S, perturbations):
    """WD prediction of d(RMS WFE)/dtau from one differential trace."""
    seeds = seeds_from_perturbations(perturbations)
    opd, _, _, dW = wavefront_with_tangents(
        ld.to_surfaces(), P, S, WVL, seeds,
        output='length')
    rms = float(np.sqrt(np.mean(opd * opd)))
    return np.mean(opd[:, None] * dW, axis=0) / rms, rms


def fd_rms_sensitivities(ld, P, S, perturbations):
    """FD sensitivity_table of the WavefrontRMS merit."""
    op = WavefrontRMS()

    def merit(prescription):
        return float(op.value(raytrace(prescription, P, S, WVL),
                              prescription, WVL))

    table = sensitivity_table(ld, perturbations, merit)
    return table.sensitivities(), table.merit_nominal


def check(ld, perturbations, rtol=2e-3, atol=1e-8):
    P, S = bundle(ld)
    wd, rms = wd_rms_sensitivities(ld, P, S, perturbations)
    fd, m_nom = fd_rms_sensitivities(ld, P, S, perturbations)
    # the WD nominal RMS must equal the merit it predicts the sensitivity of
    np.testing.assert_allclose(rms, m_nom, rtol=1e-10)
    np.testing.assert_allclose(wd, fd, rtol=rtol, atol=atol)
    return wd, fd


# ---------- per-tolerance validation ---------------------------------------

def test_curvature_surface0():
    ld = singlet()
    wd, _ = check(ld, [Perturbation.normal(ld, 'curvature', 1, 1e-6, name='c1')])
    assert abs(wd[0]) > 1e-4  # a genuinely non-vanishing sensitivity


def test_curvature_surface1():
    ld = singlet()
    check(ld, [Perturbation.normal(ld, 'curvature', 2, 1e-6, name='c2')])


def test_radius_alias_maps_to_curvature():
    ld = singlet()
    # 'radius' and 'curvature' both address the stored DOF 'c'
    check(ld, [Perturbation.normal(ld, 'radius', 1, 1e-6, name='r1')])


def test_conic_surface0():
    ld = singlet()
    check(ld, [Perturbation.normal(ld, 'conic', 1, 1e-5, name='k1')])


def test_thickness_surface0_fanout():
    """Perturbing the lens gap fans out to S1 and the image plane downstream."""
    ld = singlet()
    wd, _ = check(ld, [Perturbation.normal(ld, 'thickness', 1, 1e-5, name='t0')])
    assert abs(wd[0]) > 1e-4


def test_tilt_coordbreak_rx():
    ld = singlet_cb()
    # tilt component 2 == rx on the coordinate break (row 2)
    pert = Perturbation.normal(ld, 'tilt', 2, 1e-4, name='btx', component=2)
    wd, _ = check(ld, [pert], rtol=3e-3)
    assert abs(wd[0]) > 1e-4


def test_decenter_coordbreak_dx():
    ld = singlet_cb()
    pert = Perturbation.normal(ld, 'decenter', 2, 1e-5, name='dsx', component=0)
    wd, _ = check(ld, [pert], rtol=3e-3)
    assert abs(wd[0]) > 1e-4


def test_curvature_with_image_solve_is_compensator_aware():
    """With an image-distance solve, perturbing curvature also moves the image
    plane; the layout-FD pose tangent must capture that so WD tracks the FD
    sensitivity_table (which re-solves on every recompile)."""
    ld = singlet_solved()
    seed = seed_from_perturbation(
        Perturbation.normal(ld, 'curvature', 1, 1e-6, name='c1'))
    # the solve makes the image-plane vertex move with c -> a pose tangent
    assert seed.pose, 'expected a solve-induced image-plane pose tangent'
    check(ld, [Perturbation.normal(ld, 'curvature', 1, 1e-6, name='c1')])


def test_all_perturbations_one_trace():
    """One differential trace yields every tolerance's RMS sensitivity at once."""
    ld = singlet_cb()
    # rows: 0=OBJECT, 1=S0, 2=coordbreak, 3=S1, 4=IMAGE
    perts = [
        Perturbation.normal(ld, 'curvature', 1, 1e-6, name='c1'),
        Perturbation.normal(ld, 'conic', 1, 1e-5, name='k1'),
        Perturbation.normal(ld, 'curvature', 3, 1e-6, name='c2'),
        Perturbation.normal(ld, 'thickness', 1, 1e-5, name='t0'),
        Perturbation.normal(ld, 'tilt', 2, 1e-4, name='btx', component=2),
        Perturbation.normal(ld, 'decenter', 2, 1e-5, name='dsx', component=0),
    ]
    wd, fd = check(ld, perts, rtol=3e-3)
    assert wd.shape == (6,)


# ---------- mapping mechanics ----------------------------------------------

def test_shape_seed_resolves_surface_index_past_coordbreak():
    """A shape perturbation on S1 (row 3, after a coordbreak row) maps to the
    compiled surface index 2 (OBJECT, S0, then S1), not the row index 3."""
    ld = singlet_cb()
    seed = seed_from_perturbation(
        Perturbation.normal(ld, 'curvature', 3, 1e-6, name='c2'))
    assert seed.shape == (2, 'c')


def test_conic_seed_names_k_dof():
    ld = singlet()
    seed = seed_from_perturbation(
        Perturbation.normal(ld, 'conic', 1, 1e-5, name='k2'))
    assert seed.shape == (1, 'k')


def test_pose_perturbation_has_no_shape_activation():
    ld = singlet()
    seed = seed_from_perturbation(
        Perturbation.normal(ld, 'thickness', 1, 1e-5, name='t0'))
    assert seed.shape is None
    assert seed.pose  # fans out to downstream surfaces


def test_component_kwarg_required_for_multi_dof_category():
    ld = singlet_cb()
    with pytest.raises(ValueError, match='exactly one'):
        Perturbation.normal(ld, 'tilt', 2, 1e-4)  # 3 DOFs without component
