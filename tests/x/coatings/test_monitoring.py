"""Tests for monitoring-strategy simulation (Phase 5).

Quarter-wave layers terminate at the turning points of the monitor signal at the
monitor wavelength; a zero-error run reproduces the nominal design; and a
low-sensitivity monitor wavelength yields lower realized-error variance than a
naive (self-quarter-wave) choice.
"""
import numpy as np
import pytest

from prysm.x.coatings import Stack, RTA
from prysm.x.coatings import monitoring as mon

W = 0.55
SUB = 1.52


# ----------------------------------------------------------- turning points

@pytest.mark.parametrize('n1', [1.46, 2.05])
def test_qwot_layer_turns_at_quarter_wave(n1):
    qw = W / (4 * n1)
    s = Stack([n1], [qw], SUB)
    d, sig = mon.monitoring_trace(s, 0, W, mode='R', n_points=2000,
                                  max_factor=2.5)
    tps = mon.turning_points(d, sig)
    assert tps[0] == pytest.approx(qw, rel=2e-3)


def test_each_qwot_layer_turns_at_quarter_wave():
    indices = [1.46, 2.05, 1.46, 2.05]
    th = np.array([W / (4 * n) for n in indices])
    s = Stack(indices, th, SUB)
    for k, n in enumerate(indices):
        d, sig = mon.monitoring_trace(s, k, W, mode='R', n_points=2000,
                                      max_factor=2.0)
        tps = mon.turning_points(d, sig)
        # the first extremum during growth is the layer's quarter-wave point
        assert tps[0] == pytest.approx(th[k], rel=3e-3)


# ----------------------------------------------------------- helpers

def test_level_cut_interpolates_crossing():
    d = np.linspace(0, 1, 101)
    sig = 0.2 + 0.5 * d                       # crosses 0.45 at d = 0.5
    assert mon.level_cut(d, sig, 0.45) == pytest.approx(0.5, abs=1e-6)


def test_level_cut_picks_crossing_nearest_target():
    d = np.linspace(0, 1, 201)
    sig = np.sin(2 * np.pi * d)               # crosses 0 at 0, 0.5, 1.0
    assert mon.level_cut(d, sig, 0.0, target=0.48) == pytest.approx(0.5, abs=1e-2)


# ----------------------------------------------------------- as-built run

def test_zero_error_level_run_reproduces_nominal():
    indices = [1.46, 2.05, 1.46, 2.05]
    th = np.array([0.09, 0.067, 0.10, 0.067])
    des = Stack(indices, th, SUB)
    ab = mon.simulate_run(des, W, strategy='level', n_points=1200)
    assert np.allclose(np.asarray(ab.thicknesses), th, atol=2e-4)


def test_zero_error_turning_run_reproduces_qwot_design():
    indices = [1.46, 2.05, 1.46, 2.05]
    th = np.array([W / (4 * n) for n in indices])
    des = Stack(indices, th, SUB)
    ab = mon.simulate_run(des, W, strategy='turning', n_points=2000)
    assert np.allclose(np.asarray(ab.thicknesses), th, atol=3e-4)


def test_thickness_error_thickens_layer():
    indices = [1.46, 2.05, 1.46, 2.05]
    th = np.array([W / (4 * n) for n in indices])
    des = Stack(indices, th, SUB)
    err = np.zeros(4)
    err[2] = 0.01
    ab = mon.simulate_run(des, W, strategy='turning', thickness_errors=err,
                          n_points=2000)
    out = np.asarray(ab.thicknesses)
    assert out[2] == pytest.approx(th[2] + 0.01, abs=5e-4)
    # other layers essentially unchanged
    assert out[0] == pytest.approx(th[0], abs=5e-4)


# ----------------------------------------------- monitor-wavelength choice

def test_self_quarter_wave_monitor_is_worst_and_avoided():
    # the design layers are quarter-wave at 0.55; level monitoring there lands on
    # an extremum (ill-conditioned), so 0.55 should be the worst candidate.
    indices = [1.46, 2.05, 1.46, 2.05]
    th = np.array([0.09, 0.067, 0.10, 0.067])
    des = Stack(indices, th, SUB)
    candidates = [0.45, 0.50, 0.55, 0.60, 0.65]
    best, scores = mon.choose_monitor_wavelength(
        des, candidates, np.array([W]), strategy='level', design_pol='s')
    worst = candidates[int(np.argmax(scores))]
    assert worst == pytest.approx(0.55)
    assert best != pytest.approx(0.55)
    # the self-QW wavelength is markedly more error-sensitive
    assert np.max(scores) > 3 * np.min(scores)


def test_low_sensitivity_wavelength_lowers_realized_variance():
    indices = [1.46, 2.05, 1.46, 2.05]
    th = np.array([0.09, 0.067, 0.10, 0.067])
    des = Stack(indices, th, SUB)
    candidates = [0.45, 0.50, 0.55, 0.60, 0.65]
    best, scores = mon.choose_monitor_wavelength(
        des, candidates, np.array([W]), strategy='level', design_pol='s')
    worst = candidates[int(np.argmax(scores))]

    noise = 5e-4

    def realized_std(wm, n=150):
        rng = np.random.default_rng(1)
        Rs = []
        for _ in range(n):
            se = noise * rng.standard_normal(len(des))
            ab = mon.simulate_run(des, wm, strategy='level', signal_errors=se,
                                  n_points=800)
            Rs.append(float(RTA(ab, W, 0.0, 's')[0]))
        return np.std(Rs)

    assert realized_std(best) < 0.5 * realized_std(worst)
