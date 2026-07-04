"""Seedable merit gradients vs forward mode and central differences."""
import numpy as np
import pytest

from prysm.x.raytracing.spencer_and_murty import (
    raytrace, valid_mask,
)
from prysm.x.raytracing._diff_raytrace import (
    wavefront_with_tangents,
    seed_curvature,
    seed_conic,
    seed_despace,
    seed_decenter,
    seed_index,
)
from prysm.x.raytracing.analysis import _pupil_center_chief_index
from prysm.x.raytracing.opt import _closest_approach_on_axis, rms_spot_radius

from prysm.x.raytracing.adjoint.backward_sweep import adjoint_gradient
from prysm.x.raytracing.design import (
    Merit, RmsSpotRadius, WavefrontRMS, Distortion,
)
from tests.x.raytracing.adjoint.conftest import make_system, ray_bundle, BASE, WVL


# ---------- nominal merit evaluators (mirror the merits' conventions) -------

def _merit_spot(system, P, S):
    tr = raytrace(system, P, S, WVL)
    valid = valid_mask(tr.status, tr.P[-1])
    xy = tr.P[-1][valid, :2]
    centroid = xy.mean(axis=0)
    return float(np.sqrt(np.mean(np.sum((xy - centroid) ** 2, axis=1))))


def _merit_wfe(system, P, S, n_image=1.0):
    tr = raytrace(system, P, S, WVL)
    valid = valid_mask(tr.status, tr.P[-1])
    chief = _pupil_center_chief_index(tr.P[0])
    C = tr.P[-1][chief]
    P_xp = _closest_approach_on_axis(C, tr.S[-1][chief],
                                     np.zeros(3), np.array([0., 0., 1.]))
    R = float(np.sqrt(np.sum((P_xp - C) ** 2)))
    # independent reference-sphere oracle (the deleted -b - sqrt root), inlined
    # so this stays decoupled from the production EIC closing it validates.
    d = tr.P[-1][valid] - C
    b = np.sum(tr.S[-1][valid] * d, axis=-1)
    cc = np.sum(d * d, axis=-1) - R * R
    t = -b - np.sqrt(b * b - cc)
    OPL_total = tr.OPL[:, valid].sum(axis=0) + n_image * t
    valid_idx = np.nonzero(valid)[0]
    chief_v = int(np.nonzero(valid_idx == chief)[0][0])
    opd = OPL_total - OPL_total[chief_v]
    return float(np.sqrt(np.mean(opd ** 2)))


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
                                RmsSpotRadius())
    grad_fd = _fd_grad(_merit_spot, P, S)
    np.testing.assert_allclose(grad_adj, grad_fd, rtol=2e-5, atol=1e-8)


# ---------- RMS WFE ---------------------------------------------------------

def test_wfe_vs_forward_mode():
    P, S = ray_bundle()
    seeds = _seeds()
    opd, xp, yp, dW = wavefront_with_tangents(make_system(), P, S, WVL, seeds,
                                              output='length')
    nv = opd.shape[0]
    rms = float(np.sqrt(np.mean(opd ** 2)))
    opd_bar = opd / (nv * rms)
    grad_fwd = np.einsum('v,vp->p', opd_bar, dW)

    grad_adj = adjoint_gradient(make_system(), P, S, WVL, seeds,
                                WavefrontRMS())
    np.testing.assert_allclose(grad_adj, grad_fwd, rtol=1e-8, atol=1e-11)


def test_wfe_vs_fd():
    P, S = ray_bundle()
    grad_adj = adjoint_gradient(make_system(), P, S, WVL, _seeds(),
                                WavefrontRMS())
    grad_fd = _fd_grad(_merit_wfe, P, S)
    np.testing.assert_allclose(grad_adj, grad_fd, rtol=2e-5, atol=1e-9)


# ---------- Merit contract (the shared interface) ---------------------------

def test_seeded_merits_are_merits():
    P, S = ray_bundle()
    for merit in (RmsSpotRadius(), WavefrontRMS()):
        assert isinstance(merit, Merit)
        assert merit.has_value
        assert merit.seedable


def test_distortion_is_optimizer_only():
    # The adjoint chief-landing seed was dropped; a bare landing coordinate is
    # not a usable figure of merit without a target.
    d = Distortion(field=None, wavelength=WVL, epd=10.0)
    assert isinstance(d, Merit)
    assert not d.seedable
    assert not d.has_value


def test_merit_base_stubs_raise():
    bare = Merit()
    assert not bare.has_value
    assert not bare.seedable
    with pytest.raises(NotImplementedError):
        bare.value(None, None, None)
    with pytest.raises(NotImplementedError):
        bare.seed(None, None, None)
    with pytest.raises(NotImplementedError):
        bare(None, None)


def test_value_only_merit_flags():
    class ValueOnly(Merit):
        name = 'value_only'

        def value(self, trace, prescription, wavelength):
            return 1.0

    m = ValueOnly()
    assert m.has_value
    assert not m.seedable


def test_spot_value_matches_rms_spot_radius():
    P, S = ray_bundle()
    sys = make_system()
    tr = raytrace(sys, P, S, WVL)
    val = RmsSpotRadius().value(tr, sys, WVL)
    assert np.isclose(val, float(rms_spot_radius(tr.P[-1], status=tr.status)))
    assert np.isclose(val, _merit_spot(sys, P, S))
