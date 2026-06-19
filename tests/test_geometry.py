"""Tests for basic geometry."""
import math

import pytest

from prysm import geometry, coordinates


@pytest.mark.parametrize('sides', [5, 8])
def test_regular_polygon_contains_center_and_excludes_far_corners(sides):
    x, y = coordinates.make_xy_grid(65, diameter=2)

    mask = geometry.regular_polygon(sides, 0.5, x, y)

    assert mask[32, 32]
    assert not mask[0, 0]


def test_gaussian_peaks_at_center_and_falls_off_radially():
    x, y = coordinates.make_xy_grid(65, diameter=2)

    mask = geometry.gaussian(0.5, x, y)

    assert mask[32, 32] == pytest.approx(1)
    assert mask[32, 40] > mask[32, 48]


def test_rotated_ellipse_fails_if_minor_is_bigger_than_major():
    minor = 1
    major = 0.5
    with pytest.raises(ValueError):
        geometry.rotated_ellipse(width_major=major, width_minor=minor, x=None, y=None)


def test_rotated_ellipse_major_axis_rotation_changes_support():
    x, y = coordinates.make_xy_grid(65, diameter=2)

    horizontal = geometry.rotated_ellipse(0.8, 0.2, x, y, major_axis_angle=0)
    vertical = geometry.rotated_ellipse(0.8, 0.2, x, y, major_axis_angle=90)

    assert horizontal[32, 50]
    assert not horizontal[50, 32]
    assert vertical[50, 32]
    assert not vertical[32, 50]


def test_circle_correct_area():
    x, y = coordinates.make_xy_grid(256, diameter=2)
    dx = x[0, 1] - x[0, 0]
    r_samples = 100
    r_circle = dx*r_samples
    r, _ = coordinates.cart_to_polar(x, y)
    mask = geometry.circle(r_circle, r)
    expected_area_of_circle = r_samples*r_samples * math.pi
    assert mask.sum() == pytest.approx(expected_area_of_circle, abs=3)


def test_truecircle_correct_area():
    x, y = coordinates.make_xy_grid(256, diameter=2)
    dx = x[0, 1] - x[0, 0]
    r_samples = 100
    r_circle = dx*r_samples
    r, _ = coordinates.cart_to_polar(x, y)
    mask = geometry.truecircle(r_circle, r)
    expected_area_of_circle = r_samples*r_samples * math.pi
    assert mask.sum() == pytest.approx(expected_area_of_circle, abs=1.5)


def test_truecircle_physical_grid_area_and_registration():
    # on a physical (non-normalized) grid the historical 2/samples pitch is
    # wrong; passing dx anti-aliases the edge correctly. area is right to a
    # fraction of an edge pixel and the mask stays centered (no half-pixel shift)
    import numpy as np
    dx = 0.05
    x, y = coordinates.make_xy_grid(256, dx=dx)
    r = np.hypot(x, y)
    r_samples = 80
    mask = geometry.truecircle(dx * r_samples, r, dx=dx)
    assert mask.sum() == pytest.approx(r_samples * r_samples * math.pi, abs=1.5)
    # centered: first moment is zero to machine precision
    assert abs((mask * x).sum() / mask.sum()) < 1e-12
    assert abs((mask * y).sum() / mask.sum()) < 1e-12
    # the default (dx=None) infers 2/samples and is wrong here: a tiny pitch
    # makes the transition collapse toward a hard edge, so the areas differ
    hard_like = geometry.truecircle(dx * r_samples, r)
    assert abs(hard_like.sum() - mask.sum()) > 1


def test_truecircle_normalized_grid_matches_legacy():
    # dx=None reproduces the historical normalized-grid behavior bit-for-bit
    import numpy as np
    x, y = coordinates.make_xy_grid(256, diameter=2)
    r = np.hypot(x, y)
    radius = 100 * (2 / 256)
    new = geometry.truecircle(radius, r)
    one_pixel = 2 / 256
    legacy = np.minimum(np.maximum((radius + one_pixel / 2 - r) * (256 / 2), 0), 1)
    np.testing.assert_array_equal(new, legacy)


def test_rectangle_correct_area():
    # really this test should be done for a rectangle that is less than the
    # entire array
    x, y = coordinates.make_xy_grid(256, diameter=2)
    mask = geometry.rectangle(1, x, y)
    expected = x.size
    assert mask.sum() == expected


def test_rectangle_angle_90_swaps_width_and_height():
    x, y = coordinates.make_xy_grid(65, diameter=2)

    horizontal = geometry.rectangle(0.8, x, y, height=0.2)
    vertical = geometry.rectangle(0.8, x, y, height=0.2, angle=90)

    assert horizontal[32, 50]
    assert not horizontal[50, 32]
    assert vertical[50, 32]
    assert not vertical[32, 50]


def test_offset_circle():
    # [-16, 15] grid
    x, y = coordinates.make_xy_grid(32, dx=1)
    c = geometry.offset_circle(3, x, y, center=(2, 2))
    s = c.sum()
    assert s == 29  # 29 = roundup of 3^2 * pi


def test_annulus_excludes_center_and_outer_region():
    x, y = coordinates.make_xy_grid(65, diameter=2)
    r, _ = coordinates.cart_to_polar(x, y)

    mask = geometry.annulus(0.2, 0.5, r)

    assert not mask[32, 32]
    assert mask[32, 48]
    assert not mask[32, 0]


def test_spider_blocks_expected_axis():
    x, y = coordinates.make_xy_grid(65, diameter=2)

    mask = geometry.spider(1, 0.2, x, y)

    assert not mask[32, 48]
    assert mask[32, 16]
    assert mask[48, 32]


def test_spider_rotation_degrees_matches_radians():
    x, y = coordinates.make_xy_grid(65, diameter=2)

    deg = geometry.spider(4, 0.05, x, y, rotation=15)
    rad = geometry.spider(4, 0.05, x, y, rotation=math.radians(15),
                          rotation_is_rad=True)

    assert (deg == rad).all()


def test_rectangle_with_corner_fillets_removes_corners():
    x, y = coordinates.make_xy_grid(65, dx=1)

    mask = geometry.rectangle_with_corner_fillets(20, 20, 4, x, y)

    assert mask[32, 32]
    assert not mask[12, 12]
