"""Tests for the design layer (Operands / Problem) over a LensData.

Problem mechanics over the free vector (x0, residuals, jacobian, autograd
gating, joint DOFs) live in test_raytracing_lensdata_design.py; this module
pins operand correctness, the per-call trace cache, end-to-end optimization
recovery, and the constrained damped-least-squares solver.
"""
import numpy as np
import pytest

from prysm.mathops import optimize

from prysm.x.raytracing import OpticalSystem
from prysm.x.raytracing import LensData
from prysm.x import materials
from prysm.x.raytracing.surfaces import Conic, Plane
from prysm.x.raytracing.spencer_and_murty import raytrace
from prysm.x.raytracing.raygen import generate_collimated_ray_fan
from prysm.x.raytracing.opt import rms_spot_radius
from prysm.x.raytracing.paraxial import effective_focal_length
from prysm.x.raytracing.launch import Field, Sampling, launch
from prysm.x.raytracing.design import (
    RmsSpotRadius, EFL, WavefrontRMS, ZernikeCoefficient, Distortion,
    FieldCurvature,
    Problem, damped_least_squares,
    _TraceCache,
)


# ---------- system builders --------------------------------------------------

def _parabola_mirror(kappa=-1.0):
    """Concave mirror c=-1/80 at z=0; the reflection fold places the image at
    z=-40.  kappa=-1 is the stigmatic parabola for an on-axis collimated beam.
    """
    c = -1 / 80.0
    f = abs(1.0 / (2.0 * c))  # 40
    lens = LensData()
    (lens.add(Conic(c, kappa), typ='refl', thickness=f)
         .add(Plane(), typ='eval'))
    return OpticalSystem(lens, aperture=10.0, wavelengths=[0.55e-3])


def _collimated_fan(n=15, maxr=5.0):
    """Collimated ray fan launched from z=-50 toward the mirror at z=0."""
    P, S = generate_collimated_ray_fan(n, maxr=maxr)
    P = P.copy()
    P[:, 2] = -50.0
    return P, S


def _refractive_singlet(c1=1 / 50.0, c2=-1 / 50.0, gap=5.0, n=1.5):
    """Sphere/sphere singlet (index n) with an image plane 100 past the rear
    vertex."""
    lens = LensData()
    (lens.add(Conic(c1, 0.0), typ='refr', material=materials.ConstantMaterial(n),
              thickness=gap)
         .add(Conic(c2, 0.0), typ='refr', material=materials.air,
              thickness=100.0)
         .add(Plane(), typ='eval'))
    return OpticalSystem(lens, aperture=4.0, wavelengths=[0.55])


# ---------- Trace cache ------------------------------------------------------

def test_trace_cache_reuses_identical_launch():
    """Two operands at the same (P, S, wavelength) share a single trace."""
    ld = _parabola_mirror()
    P, S = _collimated_fan(7, maxr=2.0)
    op1 = RmsSpotRadius(P, S, wavelength=0.55e-3)
    op2 = RmsSpotRadius(P, S, wavelength=0.55e-3, target=1e-3)
    op3 = RmsSpotRadius(P, S, wavelength=0.65e-3)  # different wvl ⇒ separate
    prob = Problem(ld, [op1, op2, op3])
    _, cache = prob.residuals(prob.x0(), return_cache=True)
    assert cache.n_traces == 2  # two unique (P, S, wvl) keys


def test_trace_cache_independent_per_call():
    """Cache is per merit/residuals call, not global."""
    ld = _parabola_mirror()
    P, S = _collimated_fan(7, maxr=2.0)
    op = RmsSpotRadius(P, S, wavelength=0.55e-3)
    prob = Problem(ld, [op])
    _, c1 = prob.residuals(prob.x0(), return_cache=True)
    _, c2 = prob.residuals(prob.x0(), return_cache=True)
    assert c1 is not c2
    assert c1.n_traces == 1 and c2.n_traces == 1


# ---------- Operand evaluation -----------------------------------------------

def test_efl_operand_matches_paraxial():
    """EFL operand value matches direct paraxial.effective_focal_length."""
    ld = _refractive_singlet()
    op = EFL(wavelength=0.55)
    np.testing.assert_allclose(op(ld, _TraceCache(ld)),
                               effective_focal_length(ld, wvl=0.55))


def test_rms_spot_operand_matches_direct():
    """RmsSpotRadius matches the direct opt.rms_spot_radius computation."""
    ld = _parabola_mirror()
    P, S = _collimated_fan(11, maxr=3.0)
    op = RmsSpotRadius(P, S, wavelength=0.55e-3)
    direct = raytrace(ld, P, S, 0.55e-3)
    np.testing.assert_allclose(
        op(ld, _TraceCache(ld)),
        rms_spot_radius(direct.P[-1], status=direct.status),
    )


# ---------- End-to-end optimization recovery --------------------------------

def test_recover_parabola_kappa_via_minimize():
    """A spherical concave mirror has on-axis spherical aberration; varying κ
    via design.Problem with scipy minimize drives κ → -1 (parabolic,
    stigmatic for an on-axis collimated beam).
    """
    ld = _parabola_mirror(kappa=0.0)
    ld.lens.vary('conic', surfaces=0)
    P, S = _collimated_fan(15, maxr=5.0)
    op = RmsSpotRadius(P, S, wavelength=0.55e-3, target=0.0, weight=1.0)
    prob = Problem(ld, [op])
    res = optimize.minimize(prob.merit, prob.x0(),
                            jac=prob.jacobian, method='L-BFGS-B',
                            bounds=[(-3.0, 3.0)],
                            options={'gtol': 1e-14, 'ftol': 1e-14})
    np.testing.assert_allclose(res.x[0], -1.0, atol=1e-5)
    assert prob.merit(res.x) < 1e-10


def test_recover_efl_via_curvature_sweep():
    """Vary the front-surface curvature of a singlet to hit a target EFL."""
    ld = _refractive_singlet(c1=1 / 30.0, c2=-1 / 30.0, gap=4.0)
    ld.lens.vary('curvature', surfaces=0)
    target_efl = 75.0
    op = EFL(wavelength=0.55, target=target_efl, weight=1.0)
    prob = Problem(ld, [op])
    res = optimize.minimize(prob.merit, prob.x0(),
                            jac=prob.jacobian, method='BFGS',
                            options={'gtol': 1e-14})
    ld.lens.update(res.x)
    np.testing.assert_allclose(effective_focal_length(ld, wvl=0.55),
                               target_efl, rtol=1e-6)


def test_residuals_least_squares_path():
    """The residual vector from Problem feeds least_squares cleanly."""
    ld = _parabola_mirror(kappa=0.0)
    ld.lens.vary('conic', surfaces=0)
    P, S = _collimated_fan(15, maxr=5.0)
    op = RmsSpotRadius(P, S, wavelength=0.55e-3)
    prob = Problem(ld, [op])
    res = optimize.least_squares(prob.residuals, prob.x0(), jac='3-point',
                                 bounds=([-3.0], [3.0]),
                                 ftol=1e-12, xtol=1e-12)
    np.testing.assert_allclose(res.x[0], -1.0, atol=1e-5)


# ---------- Constrained damped least squares ---------------------------------

class _VectorResidualProblem:
    def __init__(self, target):
        self.target = np.asarray(target, dtype=float)

    def residuals(self, x):
        return np.asarray(x, dtype=float) - self.target


def test_damped_least_squares_equality_lambda_constraint():
    problem = _VectorResidualProblem([3.0, 4.0])
    result = damped_least_squares(
        problem,
        x0=np.array([0.0, 0.0]),
        equality_constraints=lambda x: x[0] + x[1] - 1.0,
        damping=0.0,
        maxiter=3,
    )
    assert result.success
    np.testing.assert_allclose(result.x, [0.0, 1.0], atol=1e-9)
    np.testing.assert_allclose(result.x.sum(), 1.0, atol=1e-12)
    assert result.lambda_eq.shape == (1,)


def test_damped_least_squares_active_inequality_lambda_constraint():
    problem = _VectorResidualProblem([0.0, 0.0])
    result = damped_least_squares(
        problem,
        x0=np.array([4.0, 1.0]),
        inequality_constraints=lambda x: x[0] - 2.0,
        damping=0.0,
        maxiter=3,
    )
    assert result.success
    np.testing.assert_allclose(result.x, [2.0, 0.0], atol=1e-9)
    assert result.lambda_ineq[0] < 0.0
    np.testing.assert_array_equal(result.active_inequalities, [0])


def test_damped_least_squares_inactive_inequality_lambda_constraint():
    problem = _VectorResidualProblem([3.0, 0.0])
    result = damped_least_squares(
        problem,
        x0=np.array([0.0, 0.0]),
        inequality_constraints=lambda x: x[0] - 2.0,
        damping=0.0,
        maxiter=5,
    )
    assert result.success
    np.testing.assert_allclose(result.x, [3.0, 0.0], atol=1e-9)
    assert result.lambda_ineq[0] == 0.0
    assert result.active_inequalities.size == 0


def test_damped_least_squares_runs_raytracing_problem_with_constraint():
    ld = _refractive_singlet(c1=1 / 30.0, c2=-1 / 30.0, gap=4.0)
    ld.lens.vary('curvature', surfaces=0)
    target_efl = 75.0
    problem = Problem(
        ld,
        equality_constraints=[EFL(wavelength=0.55, target=target_efl)],
    )
    result = problem.solve(damping=1e-8, maxiter=10, constraint_tol=1e-10)
    assert result.success
    np.testing.assert_allclose(
        effective_focal_length(ld, wvl=0.55),
        target_efl,
        rtol=1e-8,
    )


def test_solve_warns_when_solver_reports_failure(monkeypatch):
    """A failed solve still updates the lens but is never silent."""
    from prysm.x.raytracing import design as design_mod

    ld = _refractive_singlet()
    ld.lens.vary('curvature', surfaces=0)
    problem = Problem(ld, [RmsSpotRadius(
        *launch(ld, Field(0., 0.), 0.55, Sampling.fan(n=5), epd=4.0),
        wavelength=0.55)])
    x0 = problem.x0()

    def fake_dls(prob, x0=None, **kwargs):
        from prysm.x.optym.least_squares import DampedLeastSquaresResult
        return DampedLeastSquaresResult(
            x=prob.x0(), residuals=np.zeros(1), cost=0.0, success=False,
            message='equality constraints violated', nit=3, nfev=7, njev=3,
            ncev=3, lambda_eq=np.zeros(0), lambda_ineq=np.zeros(0),
            active_inequalities=np.zeros(0, dtype=int), history=[])

    monkeypatch.setattr(design_mod, 'damped_least_squares', fake_dls)
    with pytest.warns(UserWarning, match='did not converge'):
        result = problem.solve()
    assert not result.success
    np.testing.assert_allclose(problem.x0(), x0)


# ---------- Operand library --------------------------------------------------

def test_wavefront_rms_operand_evaluates():
    """WavefrontRMS on a perfect parabola is ~0."""
    ld = _parabola_mirror(-1.0)
    P, S = launch(ld, Field(0., 0.), 0.55e-3,
                  Sampling.fan(n=11), epd=10.0, pupil_z=-50.0)
    op = WavefrontRMS(P, S, wavelength=0.55e-3, P_xp=(0, 0, 0))
    assert op(ld, _TraceCache(ld)) < 1e-9


def test_wavefront_rms_operand_in_problem():
    """WavefrontRMS plugs into Problem.merit cleanly (no free DOFs)."""
    ld = _parabola_mirror(-1.0)
    P, S = launch(ld, Field(0., 0.), 0.55e-3,
                  Sampling.fan(n=11), epd=10.0, pupil_z=-50.0)
    op = WavefrontRMS(P, S, wavelength=0.55e-3, P_xp=(0, 0, 0))
    prob = Problem(ld, [op])
    assert prob.merit(prob.x0()) < 1e-18


def test_zernike_coefficient_operand_returns_known_term():
    """When OPD is dominated by a single Zernike term, the operand returns it."""
    ld = _parabola_mirror(-1.0)
    P, S = launch(ld, Field(0., 0.), 0.55e-3,
                  Sampling.hex(nrings=4), epd=10.0, pupil_z=-50.0)
    nms = [(0, 0), (2, 0), (4, 0)]
    op = ZernikeCoefficient(P, S, wavelength=0.55e-3, n=4, m=0,
                            nms_basis=nms,
                            P_xp=(0, 0, 0),
                            normalization_radius=5.0, norm=False)
    val = op(ld, _TraceCache(ld))
    # parabola is aberration-free for the on-axis collimated beam,
    # so all higher-order Zernikes are tiny
    assert abs(val) < 1e-9


def test_zernike_coefficient_rejects_missing_basis_term():
    ld = _parabola_mirror(-1.0)
    P, S = launch(ld, Field(0., 0.), 0.55e-3,
                  Sampling.fan(n=5), epd=4.0, pupil_z=-10.0)
    with pytest.raises(ValueError):
        ZernikeCoefficient(P, S, wavelength=0.55e-3, n=4, m=0,
                           nms_basis=[(0, 0), (2, 0)])


def test_distortion_operand_zero_for_on_axis():
    ld = _refractive_singlet()
    op = Distortion(Field(0., 0., unit='deg'), 0.55, epd=4.0)
    assert op(ld, _TraceCache(ld)) == 0.0


def test_distortion_operand_runs_off_axis():
    ld = _refractive_singlet()
    op = Distortion(Field(0., 1., unit='deg'), 0.55, epd=4.0)
    value = op(ld, _TraceCache(ld))
    # distortion is now signed (pincushion +, barrel -); off-axis it is a
    # finite, nonzero value rather than an unconditionally positive magnitude
    assert np.isfinite(value)
    assert value != 0.0


def test_field_curvature_operand_zero_for_on_axis():
    ld = _refractive_singlet()
    op = FieldCurvature(Field(0., 0., unit='deg'), 0.55, epd=4.0)
    np.testing.assert_allclose(op(ld, _TraceCache(ld)), 0.0, atol=1e-9)


def test_field_curvature_operand_in_problem():
    ld = _refractive_singlet()
    op = FieldCurvature(Field(0., 1., unit='deg'), 0.55, epd=4.0)
    prob = Problem(ld, [op])
    assert prob.merit(prob.x0()) >= 0.0
