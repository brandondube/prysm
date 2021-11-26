"""Tests for special segmented system handling."""
import pytest

import numpy as np

from prysm import coordinates, segmented, polynomials


def test_segmented_hex_functions():
    x, y = coordinates.make_xy_grid(256, diameter=2)
    csa = segmented.CompositeHexagonalAperture(x, y, 2, 0.2, .007, exclude=(0,))
    nms = [polynomials.noll_to_nm(j) for j in [1, 2, 3]]
    csa.prepare_opd_bases(polynomials.zernike_nm_sequence, nms)
    csa.compose_opd(np.random.rand(len(csa.segment_ids), len(nms)))
    assert csa
