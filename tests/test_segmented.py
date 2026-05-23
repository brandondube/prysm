"""Tests for special segmented system handling."""
import numpy as np

from prysm import coordinates, polynomials, segmented


def test_hex_coordinate_arithmetic_and_wrapped_direction():
    h1 = segmented.Hex(1, -1, 0)
    h2 = segmented.Hex(0, 1, -1)

    assert segmented.add_hex(h1, h2) == segmented.Hex(1, 0, -1)
    assert segmented.sub_hex(h1, h2) == segmented.Hex(1, -2, 1)
    assert segmented.mul_hex(h1, h2) == segmented.Hex(0, -1, 0)
    assert segmented.hex_dir(7) == segmented.hex_dir(1)


def test_composite_hexagonal_aperture_opd_is_limited_to_segments():
    x, y = coordinates.make_xy_grid(96, diameter=2)
    csa = segmented.CompositeHexagonalAperture(x, y, rings=1, segment_diameter=0.6, segment_separation=0.05)
    nms = [polynomials.noll_to_nm(j) for j in [1, 2]]

    grids, bases = csa.prepare_opd_bases(polynomials.zernike_nm_seq, nms)
    coefs = np.ones((len(csa.segment_ids), len(nms)))
    opd = csa.compose_opd(coefs)

    assert len(grids) == len(bases) == len(csa.segment_ids)
    assert np.count_nonzero(opd) > 0
    np.testing.assert_array_equal(opd != 0, csa.amp != 0)


def test_composite_keystone_aperture_center_and_segment_coefficients_contribute():
    x, y = coordinates.make_xy_grid(96, diameter=8)
    csa = segmented.CompositeKeystoneAperture(
        x, y,
        center_circle_diameter=2.4,
        rings=1,
        segments_per_ring=[6],
        ring_radius=0.9,
        radial_gap=.007,
    )
    center_nms = [polynomials.noll_to_nm(1)]
    segment_mns = [polynomials.xy_j_to_mn(2)]

    csa.prepare_opd_bases(
        polynomials.zernike_nm_seq,
        center_nms,
        polynomials.xy_seq,
        segment_mns,
        rotate_xyaxes=True,
        segment_basis_kwargs=dict(cartesian_grid=False),
    )
    opd = csa.compose_opd([1], np.ones((len(csa.segment_windows), len(segment_mns))))

    assert opd.shape == x.shape
    assert np.count_nonzero(opd[csa.center_window] * csa.center_mask) > 0
    assert np.count_nonzero(opd) > np.count_nonzero(opd[csa.center_window] * csa.center_mask)
