"""compensators -- SVD/least-squares projection vs FD re-optimization.

The compensated wavefront-differential model projects the nominal wavefront and
every tolerance map onto the orthogonal complement of the compensator
derivative maps (one extra column per compensator in the same nominal trace).
That is the linear-least-squares analog of re-solving the back focus per
perturbation, so it is validated against an FD that actually re-optimizes the
compensator DOF -- here the image-gap despace (focus) -- by Gauss-Newton on the
real analysis.wavefront RMS for the nominal and the +/- perturbed systems.

The test system is run with a small off-axis field (so the exit pupil is well
defined, unlike the on-axis chief-parallel-to-axis degeneracy) and its field
tilt removed, with the image plane deliberately defocused so focus is a strong
but small compensator -- the small-degradation regime where the linearized
wavefront-error quadratic compensation and a true nonlinear re-optimization
agree.
"""
import numpy as np
import pytest

from prysm.x.raytracing import OpticalSystem
from prysm.x.raytracing import LensData
from prysm.x.raytracing.launch import Field, Sampling, launch
from prysm.x.raytracing.surfaces import Conic, Plane
from prysm.x.raytracing.spencer_and_murty import STYPE_EVAL
from prysm.x.raytracing.paraxial import paraxial_image_distance
from prysm.x.raytracing.tolerance import Perturbation
from prysm.x.raytracing.wavefront_differential import (
    wavefront_differential, compensate, project_out,
)
from tests.x.raytracing.surface_helpers import wf_auto


WVL = 0.5
EPD = 5.0
FLD = Field(0.8, 0.0)
DEFOCUS = 0.2   # image plane displaced from paraxial focus -> focus-compensable


def _glass(w):
    return 1.6


def _air(w):
    return 1.0


def singlet():
    ld_data = LensData()
    (ld_data.add(Conic(1 / 24.0, 0.0), typ='refr', thickness=5.0, material=_glass)
            .add(Conic(-1 / 80.0, 0.0), typ='refr', thickness=20.0, material=_air)
            .add(Plane(), typ='eval'))
    ld = OpticalSystem(ld_data, aperture=EPD, wavelengths=[WVL])
    lens = [s for s in ld.to_surfaces() if s.typ != STYPE_EVAL]
    bfd = float(paraxial_image_distance(lens, wvl=WVL))
    ld.rows[1].thickness = bfd + DEFOCUS
    ld.lens._invalidate()
    return ld


def bundle(ld):
    return launch(ld, FLD, WVL, Sampling.rect(n=9),
                  epd=EPD, pupil_z=-5.0)


def focus_compensator(ld):
    """The image-gap despace (row 1 thickness) -- moves the image plane."""
    return Perturbation.normal(ld, 'thickness', 1, 1e-3, name='focus')


def wd(ld, tols, P, S, comps=None):
    return wavefront_differential(ld, tols, P, S, WVL, field=FLD,
                                  compensators=comps)


# ---------- FD re-optimization of the compensator(s) -----------------------

def reoptimize_rms(ld, comps, P, S, n_iter=12):
    """min over the compensator DOFs of the field-corrected wavefront RMS, by
    Gauss-Newton on the OPD residual.  Returns (rms_min, c_opt); restores
    nominal on exit."""
    comps = list(comps)
    c = np.array([cp.nominal for cp in comps], dtype=float)

    def opd_at(cvals):
        for cp, v in zip(comps, cvals):
            cp.set(float(v))
        opd, _, _ = wf_auto(ld.to_surfaces(), P, S, WVL,
                            field=FLD)
        return opd

    try:
        for _ in range(n_iter):
            r = opd_at(c)
            J = np.empty((r.size, len(comps)))
            for i in range(len(comps)):
                h = 1e-4 * max(1.0, abs(c[i]))
                cp_ = c.copy(); cp_[i] += h
                cm_ = c.copy(); cm_[i] -= h
                J[:, i] = (opd_at(cp_) - opd_at(cm_)) / (2 * h)
            dc = -np.linalg.lstsq(J, r, rcond=None)[0]
            c = c + dc
            if np.max(np.abs(dc)) < 1e-13:
                break
        r = opd_at(c)
        rms = float(np.sqrt(np.mean(r * r)))
    finally:
        for cp in comps:
            cp.reset()
    return rms, c


def fd_compensated_sensitivity(ld, tol, comps, P, S):
    """Central-FD d(re-optimized RMS)/dtau and dc_opt/dtau for one tolerance."""
    h = tol.step
    try:
        tol.set(tol.nominal + h)
        rms_p, c_p = reoptimize_rms(ld, comps, P, S)
        tol.set(tol.nominal - h)
        rms_m, c_m = reoptimize_rms(ld, comps, P, S)
    finally:
        tol.reset()
    return (rms_p - rms_m) / (2 * h), (c_p - c_m) / (2 * h)


# ---------- projection mechanics -------------------------------------------

def test_projected_maps_are_orthogonal_to_compensators():
    ld = singlet()
    P, S = bundle(ld)
    tols = [Perturbation.normal(ld, 'curvature', 0, 1e-5, name='c1'),
            Perturbation.normal(ld, 'conic', 0, 1e-4, name='k1')]
    m = wd(ld, tols, P, S, comps=[focus_compensator(ld)])
    assert m.is_compensated
    M = m.comp_maps                        # (N, 1)
    # every projected column (and W0) is orthogonal to the compensator span
    np.testing.assert_allclose(M.T @ m.W0, 0.0, atol=1e-9)
    np.testing.assert_allclose(M.T @ m.dW, 0.0, atol=1e-9)


def test_compensate_helper_matches_manual_projection():
    rng = np.random.default_rng(0)
    opd = rng.normal(size=40)
    D = rng.normal(size=(40, 3))
    M = rng.normal(size=(40, 2))
    opd_c, D_c, basis = compensate(opd, D, M)
    # residual must be orthogonal to M, and equal v - basis(basis^T v)
    np.testing.assert_allclose(M.T @ opd_c, 0.0, atol=1e-12)
    np.testing.assert_allclose(D_c, project_out(D, basis), rtol=0, atol=0)


def test_empty_compensators_matches_uncompensated_model():
    ld = singlet()
    P, S = bundle(ld)
    tols = [Perturbation.normal(ld, 'curvature', 0, 1e-5, name='c1')]
    m0 = wd(ld, tols, P, S)
    m1 = wd(ld, tols, P, S, comps=[])
    assert not m0.is_compensated and not m1.is_compensated
    np.testing.assert_allclose(m0.W0, m1.W0)
    np.testing.assert_allclose(m0.dW, m1.dW)


# ---------- the headline: compensated RMS vs FD re-optimization ------------

def test_compensated_nominal_rms_matches_reoptimized_focus():
    ld = singlet()
    P, S = bundle(ld)
    tols = [Perturbation.normal(ld, 'curvature', 0, 1e-5, name='c1')]
    comp = focus_compensator(ld)
    m = wd(ld, tols, P, S, comps=[comp])
    rms_fd, _ = reoptimize_rms(ld, [comp], P, S)
    np.testing.assert_allclose(m.rms_nominal, rms_fd, rtol=1e-2)


def test_compensation_substantially_lowers_nominal_rms():
    ld = singlet()
    P, S = bundle(ld)
    tols = [Perturbation.normal(ld, 'curvature', 0, 1e-5, name='c1')]
    m_un = wd(ld, tols, P, S)
    m_co = wd(ld, tols, P, S, comps=[focus_compensator(ld)])
    # projecting out a subspace removes energy -> RMS cannot increase
    assert m_co.rms_nominal <= m_un.rms_nominal + 1e-12
    # the deliberate defocus is mostly removed by the focus compensator
    assert m_co.rms_nominal < 0.5 * m_un.rms_nominal


def test_compensated_sensitivity_matches_fd_reoptimization():
    ld = singlet()
    P, S = bundle(ld)
    tol = Perturbation.normal(ld, 'curvature', 0, 1e-6, name='c1')
    comp = focus_compensator(ld)
    m = wd(ld, [tol], P, S, comps=[comp])
    fd_sens, _ = fd_compensated_sensitivity(ld, tol, [comp], P, S)
    np.testing.assert_allclose(m.sensitivity()[0], fd_sens, rtol=2e-2,
                               atol=1e-9)


def test_compensator_motions_match_fd():
    ld = singlet()
    P, S = bundle(ld)
    tol = Perturbation.normal(ld, 'curvature', 0, 1e-6, name='c1')
    comp = focus_compensator(ld)
    m = wd(ld, [tol], P, S, comps=[comp])
    motions = m.compensator_motions()
    assert motions.shape == (1, 1)
    assert abs(motions[0, 0]) > 1e-6      # curvature drives a real refocus
    _, dc_fd = fd_compensated_sensitivity(ld, tol, [comp], P, S)
    np.testing.assert_allclose(motions[0, 0], dc_fd[0], rtol=3e-2)


def test_compensator_motions_without_compensators_raises():
    ld = singlet()
    P, S = bundle(ld)
    m = wd(ld, [Perturbation.normal(ld, 'curvature', 0, 1e-6, name='c1')], P, S)
    with pytest.raises(ValueError, match='no compensators'):
        m.compensator_motions()


def test_compensated_sensitivity_below_uncompensated_for_focus_like_tol():
    """A lens-gap thickness tolerance moves the image, largely absorbed by the
    focus compensator, so its compensated RMS sensitivity is smaller than the
    uncompensated one and tracks the FD re-optimization."""
    ld = singlet()
    P, S = bundle(ld)
    tol = Perturbation.normal(ld, 'thickness', 0, 1e-4, name='t0')
    comp = focus_compensator(ld)
    m_un = wd(ld, [tol], P, S)
    m_co = wd(ld, [tol], P, S, comps=[comp])
    assert abs(m_co.sensitivity()[0]) < abs(m_un.sensitivity()[0])
    fd_sens, _ = fd_compensated_sensitivity(ld, tol, [comp], P, S)
    np.testing.assert_allclose(m_co.sensitivity()[0], fd_sens, rtol=3e-2,
                               atol=1e-9)
