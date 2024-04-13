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


def test_segmented_keystone_functions():
    x, y = coordinates.make_xy_grid(256, diameter=8)
    csa = segmented.CompositeKeystoneAperture(x, y,
        center_circle_diameter=2.4,
        rings=3,
        segments_per_ring=[6,12,18],
        ring_radius=0.9,
        segment_gap=.007)  # NOQA
    nms = [polynomials.noll_to_nm(j) for j in [1, 2, 3]]

    nms2 = [polynomials.j_to_xy(j) for j in [2, 3, 4, 5]]
    csa.prepare_opd_bases(polynomials.zernike_nm_sequence, nms, polynomials.xy_polynomial_sequence, nms2, rotate_xyaxes=True, segment_basis_kwargs=dict(cartesian_grid=False))
    center_coefs = np.random.rand(len(nms))
    segment_coefs = np.random.rand(len(csa.segment_ids), len(nms2))
    opd_map = csa.compose_opd(center_coefs, segment_coefs)
    assert csa
