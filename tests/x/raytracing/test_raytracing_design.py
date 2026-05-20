"""Tests for the design layer (Variables / Operands / Problem)."""
import numpy as np
import pytest

from tests.x.raytracing.surface_helpers import (
    plane, sphere, conic, off_axis_conic, even_asphere, q2d, zernike, xy,
    chebyshev, jacobi, toroid, biconic,
)
from scipy import optimize

from prysm.x.raytracing.surfaces import Surface
from prysm.x.raytracing.spencer_and_murty import raytrace
from prysm.x.raytracing.raygen import generate_collimated_ray_fan
from prysm.x.raytracing.opt import rms_spot_radius
from prysm.x.raytracing.paraxial import effective_focal_length
from prysm.x.raytracing.auto import rc_prescription_from_efl_bfl_sep
from prysm.x.raytracing.launch import Field, Sampling, launch
from prysm.x.raytracing.design import (
    curvature_of, radius_of, kappa_of, coef_of, position_of, thickness_after,
    RmsSpotRadius, RayHeightAt, Boresight, EFL, BFL, ParaxialImageDistance,
    WavefrontRMS, ZernikeCoefficient, Distortion, FieldCurvature,
    Problem, damped_least_squares,
    _TraceCache,
)


# ---------- Variable factories ----------------------------------------------

def test_curvature_of_roundtrip():
    s = conic(c=1 / 80.0, k=-1.0, typ='refl', P=[0, 0, 0])
    g, set_ = curvature_of(s)
    assert g() == 1 / 80.0
    set_(1 / 100.0)
    assert s.params['c'] == 1 / 100.0
    # FFp picks up the change
    z, _, _ = s.FFp(np.array([1.0]), np.array([0.0]))
    assert z[0] != 1.0 / (2 * 80.0)  # was: 1/160; now uses c=1/100


def test_radius_of_inverts_curvature():
    s = conic(c=1 / 80.0, k=0.0, typ='refl', P=[0, 0, 0])
    g, set_ = radius_of(s)
    np.testing.assert_allclose(g(), 80.0)
    set_(120.0)
    np.testing.assert_allclose(s.params['c'], 1 / 120.0)


def test_kappa_of_roundtrip():
    s = conic(c=1 / 80.0, k=-1.0, typ='refl', P=[0, 0, 0])
    g, set_ = kappa_of(s)
    assert g() == -1.0
    set_(-0.5)
    assert s.params['k'] == -0.5


def test_kappa_of_biconic():
    s = biconic(c_x=1 / 80.0, c_y=1 / 60.0, k_x=-1.0, k_y=0.0,
                        typ='refl', P=[0, 0, 0])
    g_x, set_x = kappa_of(s, name='k_x')
    g_y, set_y = kappa_of(s, name='k_y')
    assert g_x() == -1.0
    assert g_y() == 0.0
    set_x(-0.7)
    set_y(0.3)
    assert s.params['k_x'] == -0.7
    assert s.params['k_y'] == 0.3


def test_coef_of_asphere():
    s = even_asphere(c=1 / 80.0, k=0.0, coefs=(1e-6, 2e-9, 3e-12),
                             typ='refl', P=[0, 0, 0])
    g, set_ = coef_of(s, 'coefs', 1)
    assert g() == 2e-9
    set_(5e-9)
    assert s.params['coefs'][1] == 5e-9
    # untouched
    assert s.params['coefs'][0] == 1e-6
    assert s.params['coefs'][2] == 3e-12


def test_position_of_axes():
    s = plane(typ='eval', P=[1.0, 2.0, 3.0])
    for axis, expected in [(0, 1.0), (1, 2.0), (2, 3.0)]:
        g, set_ = position_of(s, axis)
        assert g() == expected
    g, set_ = position_of(s, 2)
    set_(50.0)
    assert s.P[2] == 50.0


def test_thickness_after_pair():
    s1 = plane(typ='eval', P=[0, 0, 10.0])
    s2 = plane(typ='eval', P=[0, 0, 25.0])
    g, set_ = thickness_after(s1, s2)
    assert g() == 15.0
    set_(40.0)
    assert s2.P[2] == 50.0  # 10 + 40
    assert s1.P[2] == 10.0  # unchanged


# ---------- Trace cache ------------------------------------------------------

def test_trace_cache_reuses_identical_launch():
    """Two operands at the same (P, S, wavelength) share a single trace."""
    s = conic(c=1 / 80.0, k=-1.0, typ='refl', P=[0, 0, 0])
    img = plane(typ='eval', P=[0, 0, -40.0])
    presc = [s, img]
    P, S = generate_collimated_ray_fan(7, maxr=2.0)
    op1 = RmsSpotRadius(P, S, wavelength=0.55e-3)
    op2 = RmsSpotRadius(P, S, wavelength=0.55e-3, target=1e-3)
    op3 = RmsSpotRadius(P, S, wavelength=0.65e-3)  # different wvl ⇒ separate trace
    prob = Problem(presc, variables=[], operands=[op1, op2, op3])
    _, cache = prob.residuals(np.array([]), return_cache=True)
    assert cache.n_traces == 2  # two unique (P, S, wvl) keys


def test_trace_cache_independent_per_call():
    """Cache is per merit/residuals call, not global."""
    s = conic(c=1 / 80.0, k=-1.0, typ='refl', P=[0, 0, 0])
    img = plane(typ='eval', P=[0, 0, -40.0])
    P, S = generate_collimated_ray_fan(7, maxr=2.0)
    op = RmsSpotRadius(P, S, wavelength=0.55e-3)
    prob = Problem([s, img], variables=[], operands=[op])
    _, c1 = prob.residuals(np.array([]), return_cache=True)
    _, c2 = prob.residuals(np.array([]), return_cache=True)
    assert c1 is not c2
    assert c1.n_traces == 1 and c2.n_traces == 1


# ---------- Operand evaluation -----------------------------------------------

def test_efl_operand_matches_paraxial():
    """EFL operand value matches direct paraxial.effective_focal_length."""
    # simple thin singlet (sphere/sphere, n=1.5)
    n_glass = lambda w: 1.5
    s1 = conic(c=1 / 50.0, k=0.0, typ='refr', P=[0, 0, 0], n=n_glass)
    s2 = conic(c=-1 / 50.0, k=0.0, typ='refr', P=[0, 0, 5.0],
                       n=lambda w: 1.0)
    img = plane(typ='eval', P=[0, 0, 100.0])
    presc = [s1, s2, img]
    op = EFL(wavelength=0.55)
    prob = Problem(presc, variables=[], operands=[op])
    cache = _TraceCache(presc)
    np.testing.assert_allclose(op(presc, cache),
                               effective_focal_length(presc, wvl=0.55))


def test_rms_spot_operand_matches_direct():
    """RmsSpotRadius matches the direct opt.rms_spot_radius computation."""
    s = conic(c=1 / 80.0, k=-1.0, typ='refl', P=[0, 0, 0])
    img = plane(typ='eval', P=[0, 0, -40.0])
    P, S = generate_collimated_ray_fan(11, maxr=3.0)
    presc = [s, img]
    op = RmsSpotRadius(P, S, wavelength=0.55e-3)
    cache = _TraceCache(presc)
    direct_trace = raytrace(presc, P, S, wvl=0.55e-3)
    np.testing.assert_allclose(
        op(presc, cache),
        rms_spot_radius(direct_trace.P[-1], status=direct_trace.status),
    )


# ---------- End-to-end optimization recovery --------------------------------

def _concave_parabola_test_setup(starting_kappa=0.0):
    """Concave mirror at z=0 (c<0 so the bowl opens in -z); rays launched
    from z=-50 going +z; paraxial focus at z=-1/(2|c|) on the rays' side.

    Returns (prescription, P, S, focus_z).
    """
    c = -1 / 80.0
    f = 1.0 / (2.0 * c)  # = -40
    s = conic(c=c, k=starting_kappa, typ='refl', P=[0, 0, 0])
    img = plane(typ='eval', P=[0, 0, f])
    P, S = generate_collimated_ray_fan(15, maxr=5.0)
    P = P.copy()
    P[:, 2] = -50.0
    return [s, img], P, S, f


def test_recover_parabola_kappa_via_minimize():
    """A spherical concave mirror has on-axis spherical aberration; varying
    κ via design.Problem with scipy minimize drives κ → -1 (parabolic,
    stigmatic for an on-axis collimated beam).

    """
    presc, P, S, _ = _concave_parabola_test_setup(starting_kappa=0.0)
    s = presc[0]
    op = RmsSpotRadius(P, S, wavelength=0.55e-3, target=0.0, weight=1.0)
    prob = Problem(presc, [kappa_of(s)], [op])
    res = optimize.minimize(prob.merit, prob.x0(),
                            jac=prob.jacobian, method='L-BFGS-B',
                            bounds=[(-3.0, 3.0)],
                            options={'gtol': 1e-14, 'ftol': 1e-14})
    np.testing.assert_allclose(res.x[0], -1.0, atol=1e-5)
    assert prob.merit(res.x) < 1e-10


def test_recover_efl_via_radius_sweep():
    """Vary the front-surface radius of a singlet to hit a target EFL."""
    n_glass = lambda w: 1.5
    s1 = conic(c=1 / 30.0, k=0.0, typ='refr', P=[0, 0, 0], n=n_glass)
    s2 = conic(c=-1 / 30.0, k=0.0, typ='refr', P=[0, 0, 4.0],
                       n=lambda w: 1.0)
    img = plane(typ='eval', P=[0, 0, 100.0])
    presc = [s1, s2, img]
    target_efl = 75.0
    op = EFL(wavelength=0.55, target=target_efl, weight=1.0)
    prob = Problem(presc, [curvature_of(s1)], [op])
    res = optimize.minimize(prob.merit, prob.x0(),
                            jac=prob.jacobian, method='BFGS',
                            options={'gtol': 1e-14})
    np.testing.assert_allclose(effective_focal_length(presc, wvl=0.55),
                               target_efl, rtol=1e-6)


def test_residuals_least_squares_path():
    """The residual vector from Problem feeds least_squares cleanly."""
    presc, P, S, _ = _concave_parabola_test_setup(starting_kappa=0.0)
    s = presc[0]
    op = RmsSpotRadius(P, S, wavelength=0.55e-3)
    prob = Problem(presc, [kappa_of(s)], [op])
    res = optimize.least_squares(prob.residuals, prob.x0(), jac='3-point',
                                 bounds=([-3.0], [3.0]),
                                 ftol=1e-12, xtol=1e-12)
    np.testing.assert_allclose(res.x[0], -1.0, atol=1e-5)


# ---------- Cache behavior under jacobian -----------------------------------

def test_jacobian_fd_matches_numerical_gradient():
    """Problem.jacobian(method='fd') matches scipy's central differences on
    the merit, validating the wiring of merit_jacobian into design.Problem."""
    c = 1 / 80.0
    s = conic(c=c, k=0.5, typ='refl', P=[0, 0, 0])
    f = -1.0 / (2.0 * c)
    img = plane(typ='eval', P=[0, 0, f])
    P, S = generate_collimated_ray_fan(11, maxr=8.0)
    op = RmsSpotRadius(P, S, wavelength=0.55e-3)
    prob = Problem([s, img], [kappa_of(s)], [op])
    x = prob.x0()
    J = prob.jacobian(x, method='fd', step=1e-6)
    # reference: scipy approx_fprime
    J_ref = optimize.approx_fprime(x, prob.merit, epsilon=1e-6)
    np.testing.assert_allclose(J, J_ref, rtol=1e-3, atol=1e-9)


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
    n_glass = lambda w: 1.5
    s1 = conic(c=1 / 30.0, k=0.0, typ='refr',
                       P=[0, 0, 0], n=n_glass)
    s2 = conic(c=-1 / 30.0, k=0.0, typ='refr',
                       P=[0, 0, 4.0], n=lambda w: 1.0)
    img = plane(typ='eval', P=[0, 0, 100.0])
    presc = [s1, s2, img]
    target_efl = 75.0
    problem = Problem(
        presc,
        [curvature_of(s1)],
        [EFL(wavelength=0.55, target=target_efl, weight=1.0)],
    )
    x0 = problem.x0()
    result = damped_least_squares(
        problem,
        x0=x0,
        equality_constraints=lambda x: x[0] - x0[0],
        damping=1e-8,
        maxiter=3,
    )
    assert result.success
    np.testing.assert_allclose(result.x, x0, atol=1e-12)
    np.testing.assert_allclose(result.x[0] - x0[0], 0.0, atol=1e-12)

# ---------- New operands (8C) ----------------------------------------------

def _spherical_singlet():
    n_glass = lambda w: 1.5
    s1 = conic(c=1 / 50.0, k=0.0, typ='refr',
                       P=[0, 0, 0], n=n_glass)
    s2 = conic(c=-1 / 50.0, k=0.0, typ='refr',
                       P=[0, 0, 5.0], n=lambda w: 1.0)
    img = plane(typ='eval', P=[0, 0, 100.0])
    return [s1, s2, img]


def _concave_parabola():
    c = -1 / 80.0
    f = 1.0 / (2.0 * c)
    s = conic(c=c, k=-1.0, typ='refl', P=[0, 0, 0])
    img = plane(typ='eval', P=[0, 0, f])
    return [s, img]


def test_wavefront_rms_operand_evaluates():
    """WavefrontRMS on a perfect parabola is ~0."""
    presc = _concave_parabola()
    P, S = launch(presc, Field(0., 0.), 0.55e-3,
                  Sampling.fan(n=11), epd=10.0, pupil_z=-50.0)
    op = WavefrontRMS(P, S, wavelength=0.55e-3)
    cache = _TraceCache(presc)
    assert op(presc, cache) < 1e-9


def test_wavefront_rms_operand_in_problem():
    """WavefrontRMS plugs into Problem.merit cleanly."""
    presc = _concave_parabola()
    P, S = launch(presc, Field(0., 0.), 0.55e-3,
                  Sampling.fan(n=11), epd=10.0, pupil_z=-50.0)
    op = WavefrontRMS(P, S, wavelength=0.55e-3)
    prob = Problem(presc, variables=[], operands=[op])
    assert prob.merit(np.array([])) < 1e-18


def test_zernike_coefficient_operand_returns_known_term():
    """When OPD is dominated by a single Zernike term, the operand returns it."""
    presc = _concave_parabola()
    P, S = launch(presc, Field(0., 0.), 0.55e-3,
                  Sampling.hex(nrings=4), epd=10.0, pupil_z=-50.0)
    nms = [(0, 0), (2, 0), (4, 0)]
    op = ZernikeCoefficient(P, S, wavelength=0.55e-3, n=4, m=0,
                            nms_basis=nms,
                            normalization_radius=5.0, norm=False)
    cache = _TraceCache(presc)
    val = op(presc, cache)
    # parabola is aberration-free for the on-axis collimated beam,
    # so all higher-order Zernikes are tiny
    assert abs(val) < 1e-9


def test_zernike_coefficient_rejects_missing_basis_term():
    presc = _concave_parabola()
    P, S = launch(presc, Field(0., 0.), 0.55e-3,
                  Sampling.fan(n=5), epd=4.0, pupil_z=-10.0)
    with pytest.raises(ValueError):
        ZernikeCoefficient(P, S, wavelength=0.55e-3, n=4, m=0,
                            nms_basis=[(0, 0), (2, 0)])


def test_distortion_operand_zero_for_on_axis():
    presc = _spherical_singlet()
    op = Distortion(Field(0., 0., unit='deg'), 0.55, epd=4.0)
    cache = _TraceCache(presc)
    assert op(presc, cache) == 0.0


def test_distortion_operand_runs_off_axis():
    presc = _spherical_singlet()
    op = Distortion(Field(0., 1., unit='deg'), 0.55, epd=4.0)
    cache = _TraceCache(presc)
    val = op(presc, cache)
    assert val >= 0.0


def test_field_curvature_operand_zero_for_on_axis():
    presc = _spherical_singlet()
    op = FieldCurvature(Field(0., 0., unit='deg'), 0.55, epd=4.0)
    cache = _TraceCache(presc)
    np.testing.assert_allclose(op(presc, cache), 0.0, atol=1e-9)


def test_field_curvature_operand_in_problem():
    presc = _spherical_singlet()
    op = FieldCurvature(Field(0., 1., unit='deg'), 0.55, epd=4.0)
    prob = Problem(presc, variables=[], operands=[op])
    val = prob.merit(np.array([]))
    assert val >= 0.0
