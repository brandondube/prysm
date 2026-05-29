"""Merit-head seeds: adjoint gradient vs forward mode and central differences.

Each merit is differentiated three ways and cross-checked:
  - the adjoint backward sweep (one sweep, all parameters)
  - (RMS WFE) prysm's forward-mode wavefront_with_tangents, contracted with the
    merit's OPD cotangent
  - central finite differences of the merit recomputed on perturbed systems
The FD recomputes the chief ray and (auto) exit pupil per perturbation, so it is
the faithful total derivative the adjoint must reproduce.
"""
import numpy as np
import pytest

from prysm.x.raytracing.spencer_and_murty import raytrace, intersect_reference_sphere
from prysm.x.raytracing._diff_raytrace import (
    raytrace_with_tangents,
    wavefront_with_tangents,
    seed_curvature,
    seed_conic,
    seed_despace,
    seed_decenter,
    seed_index,
)
from prysm.x.raytracing.opt import _valid_mask
from prysm.x.raytracing.analysis import _pupil_center_chief_index

from prysm.x.raytracing.adjoint.backward_sweep import adjoint_gradient
from prysm.x.raytracing.adjoint.merit_heads import (
    RmsSpotSizeSeed, DistortionSeed, RmsWfeSeed, _closest_point_on_axis,
)
from tests.x.raytracing.adjoint.conftest import make_system, ray_bundle, BASE, NG, WVL


# ---------- nominal merit evaluators (mirror the heads' conventions) --------

def _merit_spot(system, P, S):
    tr = raytrace(system, P, S, WVL)
    valid = _valid_mask(tr.status, tr.P[-1])
    xy = tr.P[-1][valid, :2]
    centroid = xy.mean(axis=0)
    return np.mean(np.sum((xy - centroid) ** 2, axis=1))


def _merit_distortion(system, P, S, axis):
    tr = raytrace(system, P, S, WVL)
    chief = _pupil_center_chief_index(tr.P[0])
    return tr.P[-1][chief, {'x': 0, 'y': 1}[axis]]


def _merit_wfe(system, P, S, n_image=1.0):
    tr = raytrace(system, P, S, WVL)
    valid = _valid_mask(tr.status, tr.P[-1])
    chief = _pupil_center_chief_index(tr.P[0])
    C = tr.P[-1][chief]
    P_xp = _closest_point_on_axis(C, tr.S[-1][chief],
                                  np.zeros(3), np.array([0., 0., 1.]))
    R = float(np.sqrt(np.sum((P_xp - C) ** 2)))
    _, t = intersect_reference_sphere(tr.P[-1][valid], tr.S[-1][valid], C, R)
    OPL_total = tr.OPL[:, valid].sum(axis=0) + n_image * t
    valid_idx = np.nonzero(valid)[0]
    chief_v = int(np.nonzero(valid_idx == chief)[0][0])
    opd = OPL_total - OPL_total[chief_v]
    return np.mean(opd ** 2)


# ---------- FD harness ------------------------------------------------------

SEEDS_AND_OVERRIDES = [
    (seed_curvature(0), 'c0', 1e-6),
    (seed_conic(0), 'k0', 1e-5),
    (seed_curvature(1), 'c1', 1e-6),
    (seed_conic(1), 'k1', 1e-5),
    (seed_despace([(1, +1)]), 'z1', 1e-6),
    (seed_decenter(1, 'x'), 'x1', 1e-6),
    (seed_decenter(1, 'y'), 'y1', 1e-6),
    (seed_index(0), 'ng', 1e-6),
]


def _fd_grad(merit_fn, P, S):
    grad = np.empty(len(SEEDS_AND_OVERRIDES))
    for p, (_, key, h) in enumerate(SEEDS_AND_OVERRIDES):
        base = BASE[key]
        mp = merit_fn(make_system(**{key: base + h}), P, S)
        mm = merit_fn(make_system(**{key: base - h}), P, S)
        grad[p] = (mp - mm) / (2 * h)
    return grad


def _seeds():
    return [s for (s, _, _) in SEEDS_AND_OVERRIDES]


# ---------- spot size -------------------------------------------------------

def test_spot_size_vs_fd():
    P, S = ray_bundle()
    grad_adj = adjoint_gradient(make_system(), P, S, WVL, _seeds(),
                                RmsSpotSizeSeed())
    grad_fd = _fd_grad(_merit_spot, P, S)
    np.testing.assert_allclose(grad_adj, grad_fd, rtol=2e-5, atol=1e-8)


# ---------- distortion ------------------------------------------------------

@pytest.mark.parametrize('axis', ['x', 'y'])
def test_distortion_vs_fd(axis):
    P, S = ray_bundle()
    grad_adj = adjoint_gradient(make_system(), P, S, WVL, _seeds(),
                                DistortionSeed(axis=axis))
    grad_fd = _fd_grad(lambda sys, p, s: _merit_distortion(sys, p, s, axis), P, S)
    np.testing.assert_allclose(grad_adj, grad_fd, rtol=2e-5, atol=1e-8)


def test_distortion_vs_forward_mode():
    P, S = ray_bundle()
    seeds = _seeds()
    res = raytrace_with_tangents(make_system(), P, S, WVL, seeds)
    chief = _pupil_center_chief_index(P)
    for axis, idx in (('x', 0), ('y', 1)):
        grad_adj = adjoint_gradient(make_system(), P, S, WVL, seeds,
                                    DistortionSeed(axis=axis))
        grad_fwd = res.Pdot[-1][chief, idx, :]
        np.testing.assert_allclose(grad_adj, grad_fwd, rtol=1e-9, atol=1e-11)


# ---------- RMS WFE ---------------------------------------------------------

def test_wfe_vs_forward_mode():
    P, S = ray_bundle()
    seeds = _seeds()
    opd, xp, yp, dW = wavefront_with_tangents(make_system(), P, S, WVL, seeds,
                                              output='length')
    nv = opd.shape[0]
    opd_bar = (2.0 / nv) * opd
    grad_fwd = np.einsum('v,vp->p', opd_bar, dW)

    grad_adj = adjoint_gradient(make_system(), P, S, WVL, seeds, RmsWfeSeed())
    np.testing.assert_allclose(grad_adj, grad_fwd, rtol=1e-8, atol=1e-11)


def test_wfe_vs_fd():
    P, S = ray_bundle()
    grad_adj = adjoint_gradient(make_system(), P, S, WVL, _seeds(), RmsWfeSeed())
    grad_fd = _fd_grad(_merit_wfe, P, S)
    np.testing.assert_allclose(grad_adj, grad_fd, rtol=2e-5, atol=1e-9)
