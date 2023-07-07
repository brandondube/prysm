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


@pytest.mark.skip(reason='pending fixes to prepare_opd_bases and compose_opd with new XY polynomials')
def test_segmented_keystone_functions():
    x, y = coordinates.make_xy_grid(256, diameter=2)
    csa = segmented.CompositeKeystoneAperture(x, y,
        center_circle_diameter=2,
        rings=3,
        segments_per_ring=6,
        ring_radius=0.2,
        segment_gap=.007)  # NOQA
    nms = [polynomials.noll_to_nm(j) for j in [1, 2, 3]]
    csa.prepare_opd_bases(polynomials.zernike_nm_sequence, nms)
    csa.compose_opd(np.random.rand(len(csa.segment_ids), len(nms)))
    assert csa
