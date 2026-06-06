"""End-to-end: adjoint backward sweep vs the validated forward-mode engine.

The defining adjoint identity is, for any cotangent (P_bar, S_bar, L_bar) on the
image-plane ray state,

    grad[p] = <P_bar, dP_last[:, :, p]> + <S_bar, dS_last[:, :, p]>
              + <L_bar, dOPL_total[:, p]>

where the right-hand-side tangents come from prysm's forward-mode
raytrace_with_tangents.  This checks the whole backward loop (all six primitives
+ seed contraction) against an independently validated reference.
"""
import numpy as np

from prysm.x.raytracing._diff_raytrace import (
    raytrace_with_tangents,
    seed_curvature,
    seed_conic,
    seed_despace,
    seed_decenter,
    seed_tilt,
    seed_index,
)
from prysm.x.raytracing.spencer_and_murty import valid_mask

from prysm.x.raytracing.adjoint.backward_sweep import (
    _forward_with_intermediates,
    adjoint_gradient,
)
from tests.x.raytracing.adjoint.conftest import make_system, ray_bundle, WVL


class RawSeed:
    """Test head: returns a fixed cotangent on the image-plane ray state."""

    def __init__(self, P_bar, S_bar, L_bar):
        self._P_bar = P_bar
        self._S_bar = S_bar
        self._L_bar = L_bar

    def seed(self, trace, prescription, wavelength):
        return self._P_bar, self._S_bar, self._L_bar


def _all_seeds():
    return [
        seed_curvature(0),
        seed_conic(0),
        seed_curvature(1),
        seed_conic(1),
        seed_despace([(1, +1)]),
        seed_despace([(1, +1), (2, +1)]),   # thickness fan-out
        seed_decenter(1, 'x'),
        seed_decenter(1, 'y'),
        seed_tilt(1, 'x'),
        seed_index(0),
    ]


def _forward_jacobian(surfaces, P, S, seeds):
    res = raytrace_with_tangents(surfaces, P, S, WVL, seeds)
    return res


def test_backward_sweep_matches_forward_mode():
    P, S = ray_bundle()
    surfaces = make_system()
    seeds = _all_seeds()

    res = _forward_jacobian(surfaces, P, S, seeds)
    trace = res.trace
    valid = valid_mask(trace.status, trace.P[-1])

    rng = np.random.default_rng(7)
    n = P.shape[0]
    P_bar = rng.standard_normal((n, 3))
    S_bar = rng.standard_normal((n, 3))
    L_bar = rng.standard_normal(n)
    # zero invalid rays so the forward contraction has no NaNs
    P_bar[~valid] = 0.0
    S_bar[~valid] = 0.0
    L_bar[~valid] = 0.0

    Pdot = res.Pdot[-1][valid]            # (Nv, 3, P)
    Sdot = res.Sdot[-1][valid]
    Ldot = res.Ldot.sum(axis=0)[valid]    # (Nv, P)
    grad_fwd = (np.einsum('ni,nip->p', P_bar[valid], Pdot)
                + np.einsum('ni,nip->p', S_bar[valid], Sdot)
                + np.einsum('n,np->p', L_bar[valid], Ldot))

    head = RawSeed(P_bar, S_bar, L_bar)
    grad_adj = adjoint_gradient(surfaces, P, S, WVL, seeds, head)

    np.testing.assert_allclose(grad_adj, grad_fwd, rtol=1e-9, atol=1e-11)


def test_position_only_cotangent():
    """Position-only seed (e.g. landing point) matches forward mode."""
    P, S = ray_bundle()
    surfaces = make_system()
    seeds = _all_seeds()
    res = _forward_jacobian(surfaces, P, S, seeds)
    valid = valid_mask(res.trace.status, res.trace.P[-1])

    n = P.shape[0]
    P_bar = np.zeros((n, 3))
    P_bar[valid, 0] = 1.0          # sum of x landing positions
    S_bar = np.zeros((n, 3))
    L_bar = np.zeros(n)

    grad_fwd = np.einsum('ni,nip->p', P_bar[valid], res.Pdot[-1][valid])
    head = RawSeed(P_bar, S_bar, L_bar)
    grad_adj = adjoint_gradient(surfaces, P, S, WVL, seeds, head)
    np.testing.assert_allclose(grad_adj, grad_fwd, rtol=1e-9, atol=1e-11)


def test_forward_intermediates_trace_matches_raytrace():
    """The intermediates pass reproduces the authoritative nominal trace."""
    from prysm.x.raytracing.spencer_and_murty import raytrace
    P, S = ray_bundle()
    surfaces = make_system()
    trace_ref = raytrace(surfaces, P, S, WVL)
    trace, _ = _forward_with_intermediates(surfaces, P, S, WVL)
    np.testing.assert_array_equal(trace.P, trace_ref.P)
    np.testing.assert_array_equal(trace.S, trace_ref.S)
