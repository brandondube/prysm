"""Tests for degredations."""
from prysm import degredations


def test_smear():
    sm = degredations.Smear(1, 1)
    assert sm.analytic_ft(0, 0) == 1 / sm.width


def test_jitter():
    jt = degredations.Jitter(1)
    assert jt.analytic_ft(0, 0) == 1
