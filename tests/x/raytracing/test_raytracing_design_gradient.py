"""Adjoint gradient routing in Problem: residual_jacobian, the
damped-least-squares seam, fallbacks, and the merit-gradient auto path."""
import numpy as np

import pytest

from prysm.x import materials
from prysm.x.raytracing import LensData, OpticalSystem
from prysm.x.raytracing.surfaces import Conic, Plane, EvenAsphere
from prysm.x.raytracing.launch import Field, Sampling
from prysm.x.raytracing.design import (
    Problem, RmsSpotRadius, WavefrontRMS, RayHeightAt,
)


def _singlet(c1=1 / 50.0, c2=-1 / 50.0, gap=5.0, back=100.0, shape=Conic):
    lens = LensData()
    if shape is Conic:
        front = Conic(c1, 0.0)
    else:
        front = shape(c1, 0.0, coefs=[0.0, 0.0])
    (lens.add(front, typ='refr', material=materials.ConstantMaterial(1.5),
              thickness=gap)
         .add(Conic(c2, 0.0), typ='refr', material=materials.air,
              thickness=back))
    return OpticalSystem(lens, aperture=4.0, wavelengths=[0.55])


def _two_bundle_problem(sys_, **prob_kwargs):
    # On-axis launch is invariant to these DOFs.
    f = Field(0., 0.)
    fan = Sampling.fan(n=9)
    ops = [RmsSpotRadius(f, 0.55, fan, weight=2.0),
           RmsSpotRadius(f, 0.55, Sampling.hex(3)),
           WavefrontRMS(f, 0.55, fan, P_xp=(0., 0., 80.0))]
    return Problem(sys_, ops, **prob_kwargs)


def _fd_jacobian(prob, x, step=1e-7):
    r0 = prob.residuals(x)
    J = np.empty((r0.size, x.size))
    for j in range(x.size):
        h = step * max(1.0, abs(x[j]))
        xp = x.copy()
        xm = x.copy()
        xp[j] += h
        xm[j] -= h
        J[:, j] = (prob.residuals(xp) - prob.residuals(xm)) / (2 * h)
    return J


def test_residual_jacobian_matches_fd_mixed_dofs_and_bundles():
    sys_ = _singlet()
    sys_.opt.vary('curvature', surfaces=[1, 2])
    sys_.opt.vary('thickness', surfaces=2)
    prob = _two_bundle_problem(sys_)
    x = prob.x0()
    J = prob.residual_jacobian(x)
    assert J is not None
    assert J.shape == (3, 3)
    Jfd = _fd_jacobian(prob, x)
    np.testing.assert_allclose(J, Jfd, rtol=5e-5, atol=1e-10)


def test_residual_jacobian_declines_on_unseedable_operand():
    sys_ = _singlet()
    sys_.opt.vary('curvature', surfaces=1)
    f = Field(0., 0.)
    fan = Sampling.fan(n=5)
    ops = [RmsSpotRadius(f, 0.55, fan),
           RayHeightAt(f, 0.55, fan, surface_index=-1, axis=1)]
    prob = Problem(sys_, ops)
    assert prob.residual_jacobian(prob.x0()) is None
    # DLS falls back to FD and still solves.
    result = prob.solve(maxiter=5)
    assert result.x.size == 1


def test_residual_jacobian_declines_on_vector_shape_dof():
    sys_ = _singlet(shape=EvenAsphere)
    sys_.opt.vary('coefs', surfaces=1)
    prob = _two_bundle_problem(sys_)
    assert prob.residual_jacobian(prob.x0()) is None


def test_residual_jacobian_declines_when_gradient_fd():
    sys_ = _singlet()
    sys_.opt.vary('curvature', surfaces=1)
    prob = _two_bundle_problem(sys_, gradient='fd')
    assert prob.residual_jacobian(prob.x0()) is None


def _clipped_singlet(semidia):
    lens = LensData()
    (lens.add(Conic(1 / 50.0, 0.0), typ='refr',
              material=materials.ConstantMaterial(1.5), thickness=5.0,
              semidiameter=semidia)
         .add(Conic(-1 / 50.0, 0.0), typ='refr', material=materials.air,
              thickness=95.0))
    return OpticalSystem(lens, aperture=8.0, wavelengths=[0.55],
                         fields=[Field(0., 0.), Field(0., 18.)])


def test_residual_jacobian_declines_on_nonfinite_adjoint():
    # a clipping aperture vignettes part of the off-axis fan; the adjoint sweep
    # runs non-finite through the dropped rays, so residual_jacobian must decline
    # to FD rather than hand the solver a NaN that stalls the line search.
    sys_ = _clipped_singlet(3.0)
    sys_.opt.vary('thickness', surfaces=2)
    prob = Problem(sys_, [RmsSpotRadius(Field(0., 18.), 0.55, Sampling.fan(n=15))])
    assert prob.residual_jacobian(prob.x0()) is None
    # the same bundle unclipped keeps the analytic route
    wide = _clipped_singlet(50.0)
    wide.opt.vary('thickness', surfaces=2)
    pw = Problem(wide, [RmsSpotRadius(Field(0., 18.), 0.55, Sampling.fan(n=15))])
    Jw = pw.residual_jacobian(pw.x0())
    assert Jw is not None and np.all(np.isfinite(Jw))
    # and DLS still steps via FD on the clipped problem
    result = prob.solve(maxiter=10)
    assert result.x.size == 1


def test_gradient_kwarg_validated():
    sys_ = _singlet()
    with pytest.raises(ValueError, match='gradient'):
        Problem(sys_, [], gradient='exact')


def test_merit_jacobian_auto_matches_fd():
    sys_ = _singlet()
    sys_.opt.vary('curvature', surfaces=[1, 2])
    prob = _two_bundle_problem(sys_)
    x = prob.x0()
    g_auto = prob.jacobian(x, method='auto')
    g_fd = prob.jacobian(x, method='fd')
    np.testing.assert_allclose(g_auto, g_fd, rtol=5e-5)


def test_solve_with_adjoint_routing_matches_fd_and_cuts_nfev():
    def build():
        sys_ = _singlet(back=90.0)
        sys_.opt.vary('thickness', surfaces=2)
        return sys_, [RmsSpotRadius(Field(0., 0.), 0.55, Sampling.fan(n=11))]

    sys_a, ops_a = build()
    prob_a = Problem(sys_a, ops_a)
    res_a = prob_a.solve(maxiter=10)

    sys_f, ops_f = build()
    prob_f = Problem(sys_f, ops_f, gradient='fd')
    res_f = prob_f.solve(maxiter=10)

    assert res_a.success and res_f.success
    np.testing.assert_allclose(res_a.x, res_f.x, rtol=1e-6)
    # Adjoint path skips the 2*n FD stencil per iteration.
    assert res_a.nfev < res_f.nfev
