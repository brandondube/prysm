"""Adjoint gradient routing in Problem: residual_jacobian, the
damped-least-squares seam, fallbacks, and the merit-gradient auto path."""
import numpy as np

import pytest

from prysm.x import materials
from prysm.x.raytracing import LensData, OpticalSystem
from prysm.x.raytracing.surfaces import Conic, Plane, EvenAsphere
from prysm.x.raytracing.launch import Field, Sampling, launch
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
              thickness=back)
         .add(Plane(), typ='eval'))
    return OpticalSystem(lens, aperture=4.0, wavelengths=[0.55])


def _two_bundle_problem(sys_, **prob_kwargs):
    P1, S1 = launch(sys_, Field(0., 0.), 0.55, Sampling.fan(n=9), epd=4.0)
    P2, S2 = launch(sys_, Field(0., 0.5), 0.55, Sampling.hex(3), epd=4.0)
    ops = [RmsSpotRadius(P1, S1, wavelength=0.55, weight=2.0),
           RmsSpotRadius(P2, S2, wavelength=0.55),
           WavefrontRMS(P1, S1, wavelength=0.55, P_xp=(0., 0., 80.0))]
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
    sys_.lens.vary('curvature', surfaces=[0, 1])
    sys_.lens.vary('thickness', surfaces=1)
    prob = _two_bundle_problem(sys_)
    x = prob.x0()
    J = prob.residual_jacobian(x)
    assert J is not None
    assert J.shape == (3, 3)
    Jfd = _fd_jacobian(prob, x)
    np.testing.assert_allclose(J, Jfd, rtol=5e-5, atol=1e-10)


def test_residual_jacobian_declines_on_unseedable_operand():
    sys_ = _singlet()
    sys_.lens.vary('curvature', surfaces=0)
    P, S = launch(sys_, Field(0., 0.), 0.55, Sampling.fan(n=5), epd=4.0)
    ops = [RmsSpotRadius(P, S, wavelength=0.55),
           RayHeightAt(P, S, 0.55, surface_index=-1, axis=1)]
    prob = Problem(sys_, ops)
    assert prob.residual_jacobian(prob.x0()) is None
    # the DLS seam falls back to FD and still solves
    result = prob.solve(maxiter=5)
    assert result.x.size == 1


def test_residual_jacobian_declines_on_vector_shape_dof():
    sys_ = _singlet(shape=EvenAsphere)
    sys_.lens.vary('coefs', surfaces=0)
    prob = _two_bundle_problem(sys_)
    assert prob.residual_jacobian(prob.x0()) is None


def test_residual_jacobian_declines_when_gradient_fd():
    sys_ = _singlet()
    sys_.lens.vary('curvature', surfaces=0)
    prob = _two_bundle_problem(sys_, gradient='fd')
    assert prob.residual_jacobian(prob.x0()) is None


def test_gradient_kwarg_validated():
    sys_ = _singlet()
    with pytest.raises(ValueError, match='gradient'):
        Problem(sys_, [], gradient='exact')


def test_merit_jacobian_auto_matches_fd():
    sys_ = _singlet()
    sys_.lens.vary('curvature', surfaces=[0, 1])
    prob = _two_bundle_problem(sys_)
    x = prob.x0()
    g_auto = prob.jacobian(x, method='auto')
    g_fd = prob.jacobian(x, method='fd')
    np.testing.assert_allclose(g_auto, g_fd, rtol=5e-5)


def test_solve_with_adjoint_routing_matches_fd_and_cuts_nfev():
    def build():
        sys_ = _singlet(back=90.0)
        sys_.lens.vary('thickness', surfaces=1)
        P, S = launch(sys_, Field(0., 0.), 0.55, Sampling.fan(n=11), epd=4.0)
        return sys_, [RmsSpotRadius(P, S, wavelength=0.55)]

    sys_a, ops_a = build()
    prob_a = Problem(sys_a, ops_a)
    res_a = prob_a.solve(maxiter=10)

    sys_f, ops_f = build()
    prob_f = Problem(sys_f, ops_f, gradient='fd')
    res_f = prob_f.solve(maxiter=10)

    assert res_a.success and res_f.success
    np.testing.assert_allclose(res_a.x, res_f.x, rtol=1e-6)
    # adjoint linearization skips the 2 * n FD stencil per iteration
    assert res_a.nfev < res_f.nfev
