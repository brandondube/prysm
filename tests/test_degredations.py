"""Tests for degredations."""
from prysm import degredations


def test_smear():
    sm = degredations.Smear(1, 1)
    assert sm.analytic_ft(0, 0) == 1 / sm.width


def test_jitter():
    jt = degredations.Jitter(1)
    assert jt.analytic_ft(0, 0) == 1


def test_jitter_with_spatial_has_bright_origin():
    jt = degredations.Jitter(scale=2, samples=64, sample_spacing=0.5)
    assert jt.data[32, 32] > 0.1
