"""Tests for special segmented system handling."""
import pytest

from prysm import coordinates, segmented


def test_segmented_hex_functions():
    x, y = coordinates.make_xy_grid(256, diameter=2)
    csa = segmented.CompositeHexagonalAperture(x, y, 2, 0.2, .007, exclude=(0,))
    assert csa
