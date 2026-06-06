"""Tests for analytic-gradient refinement (Phase 2).

refine should recover a textbook single-layer quarter-wave AR (whose exact
solution is known) and a known multilayer target from a perturbed start, with
the L-BFGS-B and Levenberg-Marquardt paths agreeing.
"""
import numpy as np
import pytest

from prysm.x.coatings import Stack, RTA, refine
from prysm.x.coatings.merit import Reflectance, MeritFunction


WVL = 0.55
N_SUB = 1.52
N_AR = np.sqrt(N_SUB)               # ideal single-layer AR index -> exact R=0
QWOT = WVL / (4 * N_AR)             # quarter-wave optical thickness


def test_single_layer_qwot_ar_recovered():
    # start well off the QWOT thickness; a perfect zero-R solution exists.
    s0 = Stack([N_AR], [0.07], N_SUB)
    target = Reflectance(WVL, target=0.0)
    result = refine(s0, target)
    assert result.success
    # the exact zero is at QWOT; the default stop leaves a high-quality AR.
    assert result.stack.thicknesses[0] == pytest.approx(QWOT, rel=1e-2)
    R, _, _ = RTA(result.stack, WVL, 0.0, 's')
    assert R < 1e-6
    # driving the tolerances harder drives the reflectance to machine zero.
    tight = refine(s0, target, ftol=0.0, gtol=1e-14, maxiter=500)
    assert tight.stack.thicknesses[0] == pytest.approx(QWOT, rel=1e-3)
    R_tight, _, _ = RTA(tight.stack, WVL, 0.0, 's')
    assert R_tight < 1e-8


def test_multilayer_target_from_perturbed_start():
    # a known design; perturb its thicknesses, then recover the spectrum.
    indices = [1.38, 2.05, 1.38, 2.05]
    truth = np.array([0.10, 0.065, 0.115, 0.07])
    wvls = np.linspace(0.45, 0.65, 11)
    R_target, _, _ = RTA(Stack(indices, truth, N_SUB), wvls, 0.0, 's')

    start = truth + np.array([0.02, -0.015, 0.01, -0.02])
    s0 = Stack(indices, start, N_SUB)
    target = Reflectance(wvls, pol='s', target=R_target)

    result = refine(s0, target, maxiter=300)
    R_fit, _, _ = RTA(result.stack, wvls, 0.0, 's')
    assert np.allclose(R_fit, R_target, atol=1e-4)
    assert result.merit < 1e-8


def test_lbfgsb_and_lm_agree():
    indices = [1.38, 2.05, 1.38]
    truth = np.array([0.10, 0.065, 0.115])
    wvls = np.linspace(0.5, 0.6, 6)
    R_target, _, _ = RTA(Stack(indices, truth, N_SUB), wvls, 0.0, 's')
    start = truth + 0.01

    target = Reflectance(wvls, pol='s', target=R_target)
    r_bfgs = refine(Stack(indices, start, N_SUB), target, method='lbfgsb')
    r_lm = refine(Stack(indices, start, N_SUB), target, method='lm')

    R_b, _, _ = RTA(r_bfgs.stack, wvls, 0.0, 's')
    R_l, _, _ = RTA(r_lm.stack, wvls, 0.0, 's')
    assert np.allclose(R_b, R_target, atol=1e-4)
    assert np.allclose(R_l, R_target, atol=1e-4)


def test_index_variable_refine_recovers_single_index():
    # optimize a layer's index (not thickness) to match a target spectrum -- the
    # rugate / graded-index design path.  Varying one index is well-determined,
    # so the recovery is unique.
    th = [0.10, 0.08, 0.10]
    wvls = np.linspace(0.5, 0.6, 7)
    R_target, _, _ = RTA(Stack([1.40, 2.20, 1.45], th, N_SUB), wvls, 0.0, 's')

    start = Stack([1.40, 1.90, 1.45], th, N_SUB)
    target = Reflectance(wvls, pol='s', target=R_target)
    result = refine(start, target, variables='index', variable_layers=[1],
                    bounds=(1.3, 2.4), maxiter=400)
    R_fit, _, _ = RTA(result.stack, wvls, 0.0, 's')
    assert np.allclose(R_fit, R_target, atol=1e-5)
    assert float(result.stack.indices[1]) == pytest.approx(2.20, rel=1e-3)
    # the frozen layers kept their index
    assert float(result.stack.indices[0]) == pytest.approx(1.40)


def test_index_variable_refine_drives_down_merit_multivariable():
    # several free indices: a low-reflectance target is non-unique, but the
    # analytic index gradient still drives the merit down by orders of magnitude.
    th = [0.10, 0.08, 0.10]
    wvls = np.linspace(0.5, 0.6, 6)
    R_target, _, _ = RTA(Stack([1.40, 2.20, 1.45], th, N_SUB), wvls, 0.0, 's')
    start = Stack([1.6, 1.95, 1.7], th, N_SUB)
    target = Reflectance(wvls, pol='s', target=R_target)
    start_merit = target.value(start)
    result = refine(start, target, variables='index', bounds=(1.3, 2.3),
                    maxiter=400)
    assert result.merit < start_merit / 100
    R_fit, _, _ = RTA(result.stack, wvls, 0.0, 's')
    assert np.allclose(R_fit, R_target, atol=2e-3)


def test_variable_layers_subset_frozen():
    # only the second layer is free; the others stay put.
    indices = [1.38, 2.05, 1.38]
    s0 = Stack(indices, [0.10, 0.05, 0.115], N_SUB)
    target = Reflectance(WVL, target=0.0)
    result = refine(s0, target, variable_layers=[1])
    assert result.stack.thicknesses[0] == pytest.approx(0.10)
    assert result.stack.thicknesses[2] == pytest.approx(0.115)
    assert result.stack.thicknesses[1] != pytest.approx(0.05)
