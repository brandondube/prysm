"""Tests for field-constrained design (Phase 4).

The field merit heads (peak interface intensity, field in a layer, layer
absorptance) get gradient finite-difference checks; field-only optimization is
shown to lower the targeted field; adding a peak-field term lowers the peak
interface intensity versus an R-only design at matched reflectance; and a
per-layer absorptance target is met for a lossy stack.
"""
import numpy as np
import pytest

from prysm.x.coatings import Stack, RTA, refine
from prysm.x.coatings.diff import forward_eval
from prysm.x.coatings.merit import (
    Reflectance, LayerAbsorptance, PeakFieldAtInterfaces, FieldInLayer,
    MeritFunction,
)

W = 0.55
SUB = 1.52


def _peak_field(stack, pol='s', wvl=W):
    return float(np.max(forward_eval(stack, wvl, 0.0, pol).Esq_value))


# ----------------------------------------------------------- gradient vs FD

def _fd_grad(term, th, build, h=1e-7):
    g = np.zeros_like(th)
    for i in range(th.size):
        tp = th.copy(); tp[i] += h
        tm = th.copy(); tm[i] -= h
        g[i] = (term.value(build(tp)) - term.value(build(tm))) / (2 * h)
    return g


@pytest.mark.parametrize('pol', ['s', 'p', 'avg'])
@pytest.mark.parametrize('term_factory', [
    lambda: PeakFieldAtInterfaces(np.array([0.5, 0.6]), target=0.0),
    lambda: PeakFieldAtInterfaces(np.array([0.5, 0.6]), boundaries=[1, 2, 3],
                                  target=0.0),
    lambda: FieldInLayer(1, np.array([0.5, 0.6]), target=0.0),
    lambda: LayerAbsorptance(2, np.array([0.5, 0.6]), target=0.0),
])
def test_field_head_gradient_matches_fd(pol, term_factory):
    indices = [1.46, 2.2, 1.5 + 0.2j, 2.05]
    th = np.array([0.10, 0.07, 0.05, 0.09])
    build = lambda t: Stack(indices, t, SUB)
    term = term_factory()
    term.theta = np.radians(15.0)
    term.pol = pol
    _, g = term.value_and_grad(build(th))
    g_fd = _fd_grad(term, th, build)
    assert np.allclose(g, g_fd, rtol=2e-5, atol=1e-8)


# ----------------------------------------------------------- field-only opt

def test_peak_field_only_optimization_lowers_field():
    rng = np.random.default_rng(5)
    indices = [1.46 if i % 2 else 2.25 for i in range(8)]
    s0 = Stack(indices, 0.05 + 0.08 * rng.random(8), SUB)
    before = _peak_field(s0)
    result = refine(s0, PeakFieldAtInterfaces(W, pol='s', target=0.0), maxiter=300)
    after = _peak_field(result.stack)
    assert after < 0.6 * before


def test_field_in_layer_optimization_lowers_field():
    rng = np.random.default_rng(6)
    indices = [1.46 if i % 2 else 2.25 for i in range(6)]
    s0 = Stack(indices, 0.06 + 0.06 * rng.random(6), SUB)
    term = FieldInLayer(2, W, pol='s', target=0.0)
    before = term.value(s0)
    result = refine(s0, term, maxiter=300)
    assert term.value(result.stack) < 0.5 * before


# ------------------------------------------------ peak field at matched R

def test_peak_field_term_lowers_peak_at_matched_reflectance():
    rng = np.random.default_rng(11)
    indices = [1.46 if i % 2 else 2.25 for i in range(12)]
    wv = np.array([0.50, 0.55, 0.60])
    s0 = Stack(indices, 0.04 + 0.10 * rng.random(12), SUB)

    # R-only baseline
    only_R = MeritFunction([Reflectance(wv, pol='s', target=0.05, weight=1.0)])
    A = refine(s0, only_R, maxiter=500).stack
    R_A = np.asarray(RTA(A, wv, 0.0, 's')[0])
    peak_A = _peak_field(A)

    # add a peak-field term with R held by a heavy weight
    combined = MeritFunction([
        Reflectance(wv, pol='s', target=0.05, weight=1000.0),
        PeakFieldAtInterfaces(W, pol='s', target=0.0, weight=1.0),
    ])
    B = refine(A, combined, maxiter=800).stack
    R_B = np.asarray(RTA(B, wv, 0.0, 's')[0])
    peak_B = _peak_field(B)

    assert np.max(np.abs(R_B - R_A)) < 1e-2          # reflectance still matched
    assert peak_B < 0.95 * peak_A                     # peak interface field lower


# ----------------------------------------------------------- absorptance

def test_layer_absorptance_target_met_for_lossy_stack():
    indices = [1.46, 1.5 + 0.4j, 1.46]
    s0 = Stack(indices, [0.10, 0.06, 0.10], SUB)
    target = 0.15
    result = refine(s0, LayerAbsorptance(1, W, pol='s', target=target), maxiter=300)
    _, _, A = RTA(result.stack, W, 0.0, 's')
    assert A[1] == pytest.approx(target, abs=1e-4)


# ----------------------------------------------------------- multi-objective

def test_mixed_objective_value_and_grad_consistent():
    indices = [1.46, 2.2, 1.5 + 0.2j, 2.05]
    th = np.array([0.10, 0.07, 0.05, 0.09])
    build = lambda t: Stack(indices, t, SUB)
    mf = MeritFunction([
        Reflectance(np.array([0.5, 0.6]), pol='s', target=0.0, weight=2.0),
        PeakFieldAtInterfaces(np.array([0.5, 0.6]), pol='s', target=0.0),
        LayerAbsorptance(2, 0.55, pol='s', target=0.0, weight=0.5),
    ])
    val, grad = mf.value_and_grad(build(th))
    assert val == pytest.approx(sum(t.value(build(th)) for t in mf.terms))
    g_fd = np.zeros_like(th)
    for i in range(th.size):
        tp = th.copy(); tp[i] += 1e-7
        tm = th.copy(); tm[i] -= 1e-7
        g_fd[i] = (mf.value(build(tp)) - mf.value(build(tm))) / 2e-7
    assert np.allclose(grad, g_fd, rtol=2e-5, atol=1e-8)
