"""Tests for optimizing a LensData directly through design.Problem over
its dense free vector (numpy FD path; the autograd path is gated on torch)."""

import numpy as np
import pytest

from prysm.mathops import optimize

from prysm.x.raytracing import OpticalSystem
from prysm.x.raytracing import FRAUNHOFER_LINES_UM, LensData
from prysm.x import materials
from prysm.x.raytracing.design import EFL, Problem, RmsSpotRadius
from prysm.x.raytracing.launch import Sampling
from prysm.x.raytracing.paraxial import effective_focal_length
from prysm.x.raytracing.surfaces import Conic, Plane


n_bk7 = materials.ConstantMaterial('N-BK7', 1.5168)


def make_singlet(image_gap=95.0, with_image=True):
    lens = LensData()
    (lens.add(Conic(1 / 102.0, 0.0), thickness=6.0, material=n_bk7,
              semidiameter=12.0)
         .add(Conic(-1 / 102.0, 0.0), thickness=image_gap,
              material=materials.air, semidiameter=12.0))
    if with_image:
        lens.add(Plane(), typ='eval', material=materials.air,
                 semidiameter=12.0)
    return OpticalSystem(lens, aperture=20.0, fields=[0],
                         wavelengths=list(FRAUNHOFER_LINES_UM.values()),
                         reference=1)


def test_problem_x0_is_the_packed_free_vector():
    ld = make_singlet(with_image=False)
    ld.lens.vary('curvature', surfaces=[0, 1])
    prob = Problem(ld, [EFL(ld.wavelength(), target=100.0)])
    np.testing.assert_allclose(prob.x0(), [1 / 102.0, -1 / 102.0])


def test_problem_residuals_track_the_free_vector():
    ld = make_singlet(with_image=False)
    ld.lens.vary('curvature', surfaces=1)
    wvl = ld.wavelength()
    target = 100.0
    prob = Problem(ld, [EFL(wvl, target=target)])
    r0 = prob.residuals(prob.x0())
    # residual = EFL(current) - target
    assert r0[0] == pytest.approx(effective_focal_length(ld, wvl=wvl) - target)


def test_lensdata_efl_optimization_converges():
    ld = make_singlet(with_image=False)
    ld.lens.vary('curvature', surfaces=1)  # one DOF, one operand -> well posed
    wvl = ld.wavelength()
    target = 80.0
    prob = Problem(ld, constraints=[EFL(wvl, target=target)])
    res = prob.solve(damping=1e-8, xtol=1e-12, ftol=1e-12,
                     constraint_tol=1e-12)
    assert res.success
    assert effective_focal_length(ld, wvl=wvl) == pytest.approx(target, rel=1e-6)


def test_lensdata_thickness_and_curvature_jointly_varied():
    # vary a curvature AND a thickness; merit drives EFL to target.  Confirms
    # the free vector mixes shape and gap DOFs and the optimizer moves both.
    ld = make_singlet(with_image=False)
    ld.lens.vary('curvature', surfaces=1).vary('thickness', surfaces=0)
    wvl = ld.wavelength()
    prob = Problem(ld, constraints=[EFL(wvl, target=90.0)])
    x0 = prob.x0()
    assert len(x0) == 2
    res = prob.solve(x0, damping=1e-8, maxiter=10)
    assert res.success
    assert effective_focal_length(ld, wvl=wvl) == pytest.approx(90.0, rel=1e-5)


def test_focal_length_constraint_is_not_an_objective_residual():
    ld = make_singlet(with_image=False)
    ld.lens.vary('curvature', surfaces=1)
    wvl = ld.wavelength()
    prob = Problem(ld, constraints=[EFL(wvl, target=90.0)])
    assert prob.residuals(prob.x0()).size == 0
    assert prob.equalities(prob.x0()).shape == (1,)


def test_fd_free_jacobian_matches_numeric_merit_gradient():
    ld = make_singlet(with_image=False)
    ld.lens.vary('curvature', surfaces=[0, 1])
    wvl = ld.wavelength()
    prob = Problem(ld, [EFL(wvl, target=100.0)])
    x = prob.x0()
    J = prob.jacobian(x, method='fd', step=1e-7)

    # independent central-difference reference on the scalar merit
    ref = np.empty_like(J)
    for i in range(len(x)):
        h = 1e-7 * abs(x[i])
        xp = x.copy(); xp[i] += h
        xm = x.copy(); xm[i] -= h
        ref[i] = (prob.merit(xp) - prob.merit(xm)) / (2 * h)
    ld.lens.update(x)
    np.testing.assert_allclose(J, ref, rtol=1e-4)


def test_jacobian_restores_free_vector():
    ld = make_singlet(with_image=False)
    ld.lens.vary('curvature', surfaces=[0, 1])
    prob = Problem(ld, [EFL(ld.wavelength(), target=100.0)])
    x0 = prob.x0()
    prob.jacobian(x0)
    np.testing.assert_allclose(ld.lens.pack(), x0)


def test_rms_spot_operand_decreases_under_optimization():
    ld = make_singlet(image_gap=96.0)
    wvl = ld.wavelength()
    op = RmsSpotRadius(ld.field(0), wvl, Sampling.hex(nrings=3))
    ld.lens.vary('curvature', surfaces=[0, 1])
    prob = Problem(ld, [op])
    spot0 = op(ld, _fresh_cache(prob))
    res = optimize.least_squares(prob.residuals, prob.x0(), jac='3-point',
                                 max_nfev=60)
    ld.lens.update(res.x)
    spot1 = op(ld, _fresh_cache(prob))
    assert spot1 <= spot0


def test_lensdata_autograd_jacobian_requires_torch():
    ld = make_singlet(with_image=False)
    ld.lens.vary('curvature', surfaces=1)
    prob = Problem(ld, [EFL(ld.wavelength(), target=100.0)])
    with pytest.raises(RuntimeError, match='backend to be torch'):
        prob.jacobian(prob.x0(), method='autograd')


def _fresh_cache(prob):
    from prysm.x.raytracing.design import _TraceCache
    return _TraceCache(prob.prescription)
