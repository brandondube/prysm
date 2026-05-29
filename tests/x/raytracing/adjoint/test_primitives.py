"""Dot-product (transpose) tests for the adjoint primitives.

For each adjoint A^T of a forward primitive A, the inner-product identity

    <y_bar, A x> == <A^T y_bar, x>

holds to floating-point precision at any admissible nominal point.  Each test
draws random nominal data, random tangent inputs x, and a random cotangent
y_bar, evaluates both sides, and asserts equality.
"""
import numpy as np
import pytest

from prysm.x.raytracing._diff_raytrace import (
    d_transform_local,
    d_intersect,
    d_refract,
    d_reflect,
    d_transform_global,
    d_opl_segment,
    d_intersect_reference_sphere,
    d_closest_point_on_axis,
)
from prysm.x.raytracing.adjoint.primitives import (
    adj_transform_local,
    adj_intersect,
    adj_refract,
    adj_reflect,
    adj_transform_global,
    adj_opl_segment,
    adj_intersect_reference_sphere,
    adj_intersect_reference_sphere_full,
    adj_closest_point_on_axis,
)

RTOL = 1e-10
N = 17


@pytest.fixture
def rng():
    return np.random.default_rng(0xC0FFEE)


def _vecs(rng, n, k):
    """k random (n, 3) vectors."""
    return [rng.standard_normal((n, 3)) for _ in range(k)]


def _unit_normals(rng, n):
    """Random unit normals with positive z (so n_hat_z != 0)."""
    v = rng.standard_normal((n, 3))
    v[:, 2] = np.abs(v[:, 2]) + 0.5
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _rot(rng):
    """A random orthonormal 3x3 (so R^T R = I, like a real surface rotation)."""
    q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    return q


def _vdot(a, b):
    return float(np.sum(a * b))


# ---------- 1.1 -------------------------------------------------------------

def test_adj_opl_segment(rng):
    seg = rng.standard_normal((N, 3))
    n_pre = 1.37
    n_pre_dot = rng.standard_normal(1)
    dseg = rng.standard_normal((N, 3, 1))
    L_bar = rng.standard_normal(N)

    dL = d_opl_segment(n_pre, n_pre_dot, seg, dseg)[:, 0]
    n_bar, dseg_bar = adj_opl_segment(n_pre, seg, L_bar)

    lhs = _vdot(L_bar, dL)
    rhs = float(n_bar) * float(n_pre_dot[0]) + _vdot(dseg_bar, dseg[..., 0])
    assert np.isclose(lhs, rhs, rtol=RTOL)


# ---------- 1.2 -------------------------------------------------------------

def test_adj_transform_global(rng):
    Reff = _rot(rng)
    Q = rng.standard_normal(3)
    Q_loc, Sprime = _vecs(rng, N, 2)
    dPj = rng.standard_normal((N, 3, 1))
    dSprime = rng.standard_normal((N, 3, 1))
    Qdot = rng.standard_normal((3, 1))
    Rdot = rng.standard_normal((3, 3, 1))

    dPjp1, dSjp1 = d_transform_global(Reff, Q, Q_loc, Sprime,
                                      dPj, dSprime, Qdot, Rdot)
    P_bar = rng.standard_normal((N, 3))
    S_bar = rng.standard_normal((N, 3))
    dPj_bar, dSprime_bar, Qdot_bar, Rdot_bar = adj_transform_global(
        Reff, Q_loc, Sprime, P_bar, S_bar)

    lhs = _vdot(P_bar, dPjp1[..., 0]) + _vdot(S_bar, dSjp1[..., 0])
    rhs = (_vdot(dPj_bar, dPj[..., 0]) + _vdot(dSprime_bar, dSprime[..., 0])
           + _vdot(Qdot_bar, Qdot[..., 0]) + _vdot(Rdot_bar, Rdot[..., 0]))
    assert np.isclose(lhs, rhs, rtol=RTOL)


# ---------- 1.3 -------------------------------------------------------------

def test_adj_refract(rng):
    n, nprime = 1.0, 1.51        # mu < 1 => no TIR for any angle
    S_loc = _unit_normals(rng, N)
    n_hat = _unit_normals(rng, N)
    S_locdot = rng.standard_normal((N, 3, 1))
    dn_hat = rng.standard_normal((N, 3, 1))
    ndot_pre = rng.standard_normal(1)
    ndot_post = rng.standard_normal(1)

    Sprime, dSprime = d_refract(n, nprime, S_loc, n_hat,
                                S_locdot, dn_hat, ndot_pre, ndot_post)
    dSprime_bar = rng.standard_normal((N, 3))
    S_locdot_bar, dn_hat_bar, ndp_bar, ndpost_bar = adj_refract(
        n, nprime, S_loc, n_hat, dSprime_bar)

    lhs = _vdot(dSprime_bar, dSprime[..., 0])
    rhs = (_vdot(S_locdot_bar, S_locdot[..., 0])
           + _vdot(dn_hat_bar, dn_hat[..., 0])
           + float(ndp_bar) * float(ndot_pre[0])
           + float(ndpost_bar) * float(ndot_post[0]))
    assert np.isclose(lhs, rhs, rtol=RTOL)


# ---------- 1.4 -------------------------------------------------------------

def test_adj_reflect(rng):
    S_loc = _unit_normals(rng, N)
    n_hat = _unit_normals(rng, N)
    S_locdot = rng.standard_normal((N, 3, 1))
    dn_hat = rng.standard_normal((N, 3, 1))

    Sprime, dSprime = d_reflect(S_loc, n_hat, S_locdot, dn_hat)
    dSprime_bar = rng.standard_normal((N, 3))
    S_locdot_bar, dn_hat_bar = adj_reflect(S_loc, n_hat, dSprime_bar)

    lhs = _vdot(dSprime_bar, dSprime[..., 0])
    rhs = _vdot(S_locdot_bar, S_locdot[..., 0]) + _vdot(dn_hat_bar, dn_hat[..., 0])
    assert np.isclose(lhs, rhs, rtol=RTOL)


# ---------- 1.5 -------------------------------------------------------------

def test_adj_intersect(rng):
    P0 = rng.standard_normal((N, 3))
    S_loc = _unit_normals(rng, N)        # nonzero g_dot_S
    Q_loc = rng.standard_normal((N, 3))
    n_hat = _unit_normals(rng, N)
    hessian = (rng.standard_normal(N), rng.standard_normal(N),
               rng.standard_normal(N))
    P0dot = rng.standard_normal((N, 3, 1))
    S_locdot = rng.standard_normal((N, 3, 1))
    dsag_param = rng.standard_normal((N, 1))
    dgx_param = rng.standard_normal((N, 1))
    dgy_param = rng.standard_normal((N, 1))

    dPj, dn_hat = d_intersect(P0, S_loc, Q_loc, n_hat, P0dot, S_locdot,
                              hessian, dsag_param, dgx_param, dgy_param)
    dPj_bar = rng.standard_normal((N, 3))
    dn_hat_bar = rng.standard_normal((N, 3))
    (P0dot_bar, S_locdot_bar, dsag_bar, dgx_bar,
     dgy_bar) = adj_intersect(P0, S_loc, Q_loc, n_hat, hessian,
                              dPj_bar, dn_hat_bar)

    lhs = _vdot(dPj_bar, dPj[..., 0]) + _vdot(dn_hat_bar, dn_hat[..., 0])
    rhs = (_vdot(P0dot_bar, P0dot[..., 0]) + _vdot(S_locdot_bar, S_locdot[..., 0])
           + _vdot(dsag_bar, dsag_param[..., 0])
           + _vdot(dgx_bar, dgx_param[..., 0])
           + _vdot(dgy_bar, dgy_param[..., 0]))
    assert np.isclose(lhs, rhs, rtol=RTOL)


# ---------- 1.6 -------------------------------------------------------------

def test_adj_transform_local(rng):
    Reff = _rot(rng)
    Q = rng.standard_normal(3)
    P = rng.standard_normal((N, 3))
    S = rng.standard_normal((N, 3))
    Pdot = rng.standard_normal((N, 3, 1))
    Sdot = rng.standard_normal((N, 3, 1))
    Qdot = rng.standard_normal((3, 1))
    Rdot = rng.standard_normal((3, 3, 1))

    P0, S_loc, P0dot, S_locdot = d_transform_local(
        Reff, Q, P, S, Pdot, Sdot, Qdot, Rdot)
    P0dot_bar = rng.standard_normal((N, 3))
    S_locdot_bar = rng.standard_normal((N, 3))
    Pdot_bar, Sdot_bar, Qdot_bar, Rdot_bar = adj_transform_local(
        Reff, P, Q, S, P0dot_bar, S_locdot_bar)

    lhs = _vdot(P0dot_bar, P0dot[..., 0]) + _vdot(S_locdot_bar, S_locdot[..., 0])
    rhs = (_vdot(Pdot_bar, Pdot[..., 0]) + _vdot(Sdot_bar, Sdot[..., 0])
           + _vdot(Qdot_bar, Qdot[..., 0]) + _vdot(Rdot_bar, Rdot[..., 0]))
    assert np.isclose(lhs, rhs, rtol=RTOL)


# ---------- 1.7 -------------------------------------------------------------

def test_adj_intersect_reference_sphere(rng):
    # benign converging geometry: rays near origin aimed +z, center ahead.
    P = rng.standard_normal((N, 3)) * 0.3
    S = np.tile(np.array([0.0, 0.0, 1.0]), (N, 1))
    S = S + rng.standard_normal((N, 3)) * 0.02
    S /= np.linalg.norm(S, axis=1, keepdims=True)
    C = np.array([0.0, 0.0, 50.0])
    R = 50.0
    Pdot = rng.standard_normal((N, 3, 1))
    Sdot = rng.standard_normal((N, 3, 1))
    Cdot = np.zeros((3, 1))
    Rdot = np.zeros(1)

    tdot = d_intersect_reference_sphere(P, S, Pdot, Sdot, C, Cdot, R, Rdot)
    t_bar = rng.standard_normal(N)
    P_bar, S_bar = adj_intersect_reference_sphere(P, S, C, R, t_bar)

    lhs = _vdot(t_bar, tdot[..., 0])
    rhs = _vdot(P_bar, Pdot[..., 0]) + _vdot(S_bar, Sdot[..., 0])
    assert np.isclose(lhs, rhs, rtol=RTOL)


def test_adj_intersect_reference_sphere_full(rng):
    """Full transpose: ray AND sphere (C, R) cotangents."""
    P = rng.standard_normal((N, 3)) * 0.3
    S = np.tile(np.array([0.0, 0.0, 1.0]), (N, 1)) + rng.standard_normal((N, 3)) * 0.02
    S /= np.linalg.norm(S, axis=1, keepdims=True)
    C = np.array([0.0, 0.0, 50.0])
    R = 50.0
    Pdot = rng.standard_normal((N, 3, 1))
    Sdot = rng.standard_normal((N, 3, 1))
    Cdot = rng.standard_normal((3, 1))
    Rdot = rng.standard_normal(1)

    tdot = d_intersect_reference_sphere(P, S, Pdot, Sdot, C, Cdot, R, Rdot)
    t_bar = rng.standard_normal(N)
    P_bar, S_bar, C_bar, R_bar = adj_intersect_reference_sphere_full(
        P, S, C, R, t_bar)

    lhs = _vdot(t_bar, tdot[..., 0])
    rhs = (_vdot(P_bar, Pdot[..., 0]) + _vdot(S_bar, Sdot[..., 0])
           + _vdot(C_bar, Cdot[..., 0]) + float(R_bar) * float(Rdot[0]))
    assert np.isclose(lhs, rhs, rtol=RTOL)


def test_adj_closest_point_on_axis(rng):
    P = rng.standard_normal(3) * 0.5
    S = np.array([0.02, -0.03, 1.0])
    S /= np.linalg.norm(S)
    axis_point = np.zeros(3)
    axis_dir = np.array([0.0, 0.0, 1.0])
    Pdot = rng.standard_normal((3, 1))
    Sdot = rng.standard_normal((3, 1))

    P_xp, P_xp_dot = d_closest_point_on_axis(P, S, Pdot, Sdot,
                                             axis_point, axis_dir)
    P_xp_bar = rng.standard_normal(3)
    P_bar, S_bar = adj_closest_point_on_axis(P, S, axis_point, axis_dir,
                                             P_xp_bar)

    lhs = _vdot(P_xp_bar, P_xp_dot[..., 0])
    rhs = _vdot(P_bar, Pdot[..., 0]) + _vdot(S_bar, Sdot[..., 0])
    assert np.isclose(lhs, rhs, rtol=RTOL)
