"""Tests for thin film calculations."""
import pytest

from prysm import thinfilm

wvl = .587725
n_C7980 = 1.458461
n_MgF2 = 1.3698
n_CeF3 = 1.6290 + 1j * 0.0034836
n_ZrO2 = 2.1588


def test_accuracy_of_monolayer_reflectivity_MgF2_on_C7980():
    indices = [
        n_MgF2,
        n_C7980
    ]
    thicknesses = [
        .150,
        10_000  # 10 mm thick substrate
    ]
    r, _ = thinfilm.multilayer_stack_rt('p', indices, thicknesses, wvl)
    R = abs(r)**2
    assert R == pytest.approx(0.022, abs=0.001)  # 98% transmission


def test_accuracy_of_multilayer_reflectivity_on_C7980():
    indices = [
        n_MgF2,
        n_ZrO2,
        n_CeF3,
        n_C7980
    ]
    thicknesses = [
        wvl/4,
        wvl/2,
        wvl/4,
        10_000
    ]
    r, _ = thinfilm.multilayer_stack_rt('s', indices, thicknesses, wvl)
    R = abs(r)**2
    assert R == pytest.approx(0.0024, abs=0.0005)  # 99.7% transmission


def test_brewsters_accuracy():
    ang = thinfilm.brewsters_angle(1, 1.5)
    assert ang == pytest.approx(56.3, abs=1e-2)


def test_critical_accuracy():
    ang = thinfilm.critical_angle(1, 1.5, deg=True)
    assert ang == pytest.approx(41.8, abs=0.02)
