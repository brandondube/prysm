"""Tests for degredations."""
from prysm import degredations


def test_smear():
    sm = degredations.Smear(1, 1)
    assert sm.analytic_ft(0, 0) == 1 / sm.width
