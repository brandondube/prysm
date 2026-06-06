"""Tolerance-analysis helpers on the adjoint Jacobian.

multi_objective_sensitivity is checked against per-head adjoint_gradient (the
shared-forward path must reproduce the per-head sweeps); the linear-algebra
helpers (inverse sensitivity, RSS, compensators) are checked against their
defining relations and a direct Monte Carlo for RSS.
"""
import numpy as np

from prysm.x.raytracing._diff_raytrace import (
    seed_curvature, seed_conic, seed_despace, seed_decenter, seed_index,
)

from prysm.x.raytracing.adjoint.backward_sweep import adjoint_gradient
from prysm.x.raytracing.design import RmsSpotRadius, WavefrontRMS
from prysm.x.raytracing.adjoint.tolerance_analysis import (
    multi_objective_sensitivity,
    ToleranceSensitivityTable,
    inverse_sensitivity,
    rss_prediction,
    compensated_jacobian,
    multi_objective_budget,
)
from tests.x.raytracing.adjoint.conftest import make_system, ray_bundle, WVL


def _seeds():
    return [seed_curvature(0), seed_conic(1), seed_despace([(1, +1)]),
            seed_decenter(1, 'y'), seed_index(0)]


def _heads(P, S):
    # the seedable unified merits; Distortion is value-only (no adjoint seed).
    return [WavefrontRMS(P, S, WVL), RmsSpotRadius(P, S, WVL)]


def test_jacobian_matches_per_head_sweeps():
    P, S = ray_bundle()
    seeds = _seeds()
    heads = _heads(P, S)
    res = multi_objective_sensitivity(make_system(), P, S, WVL, seeds, heads)
    assert res.jacobian.shape == (len(heads), len(seeds))
    for m, head in enumerate(heads):
        g = adjoint_gradient(make_system(), P, S, WVL, seeds, head)
        np.testing.assert_allclose(res.jacobian[m], g, rtol=1e-12, atol=0)
    assert res.param_names == [s.name for s in seeds]
    assert 'rms_wfe' in res.nominals


def test_ranked_by_orders_by_abs_sensitivity():
    P, S = ray_bundle()
    res = multi_objective_sensitivity(make_system(), P, S, WVL, _seeds(),
                                      _heads(P, S))
    ranked = res.ranked_by('rms_wfe')
    mags = [abs(v) for _, v in ranked]
    assert mags == sorted(mags, reverse=True)


def test_inverse_sensitivity_hits_budget():
    J = np.array([[2.0, -0.5, 0.0],
                  [1.0, 3.0, 4.0]])
    budget = 0.1
    tol = inverse_sensitivity(J, budget)
    # zero-sensitivity parameter (col 0 of obj... actually col 2 obj0 == 0)
    deg = np.abs(J) * tol[None, :]
    # the binding objective for each finite-tol parameter hits the budget
    for p in range(J.shape[1]):
        if np.isfinite(tol[p]):
            assert np.isclose(deg[:, p].max(), budget)


def test_inverse_sensitivity_clips():
    J = np.array([[10.0, 0.01]])
    tol = inverse_sensitivity(J, 1.0, steps_max=np.array([5.0, 5.0]))
    assert tol[0] == 0.1            # 1/10
    assert tol[1] == 5.0            # 1/0.01=100 clipped to 5


def test_rss_matches_monte_carlo():
    J = np.array([[2.0, -1.5, 0.7],
                  [0.3, 1.1, -2.2]])
    sigmas = np.array([0.05, 0.08, 0.02])
    rss = rss_prediction(J, sigmas)

    rng = np.random.default_rng(0)
    N = 200000
    taus = rng.normal(0.0, sigmas[None, :], size=(N, J.shape[1]))
    samples = taus @ J.T                       # (N, M)
    mc = samples.std(axis=0, ddof=0)
    np.testing.assert_allclose(rss, mc, rtol=2e-2)


def test_compensated_jacobian_zeros_compensator_columns():
    rng = np.random.default_rng(1)
    M, P, K = 5, 7, 2
    J = rng.standard_normal((M, P))
    J_comp = rng.standard_normal((M, K))
    J_eff, motions = compensated_jacobian(J, J_comp)
    assert motions.shape == (K, P)
    # J_eff lies in the orthogonal complement of the compensator columns
    np.testing.assert_allclose(J_comp.T @ J_eff, np.zeros((K, P)), atol=1e-10)
    # if a tolerance column equals a compensator direction, it is fully removed
    J2 = J.copy()
    J2[:, 0] = J_comp[:, 0]
    J_eff2, _ = compensated_jacobian(J2, J_comp)
    np.testing.assert_allclose(J_eff2[:, 0], 0.0, atol=1e-10)


def test_multi_objective_budget():
    J = np.array([[2.0, 1.0],
                  [1.0, 4.0]])
    budgets = np.array([0.2, 0.4])
    tol = multi_objective_budget(J, budgets)
    # param 0: min(0.2/2, 0.4/1) = 0.1 ; param 1: min(0.2/1, 0.4/4)=0.1
    np.testing.assert_allclose(tol, [0.1, 0.1])


def test_sensitivity_table():
    P, S = ray_bundle()
    res = multi_objective_sensitivity(make_system(), P, S, WVL, _seeds(),
                                      _heads(P, S))
    steps = np.full(len(_seeds()), 1e-3)
    tbl = ToleranceSensitivityTable(res, steps)
    np.testing.assert_allclose(tbl.sensitivity(), np.abs(res.jacobian))
    np.testing.assert_allclose(tbl.degradation_at_step(),
                               res.jacobian * steps[None, :])
