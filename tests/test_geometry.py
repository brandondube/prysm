"""Tests for basic geometry."""
import math

import pytest

import numpy as np

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


def test_antialias_circle_correct_area():
    x, y = coordinates.make_xy_grid(256, diameter=2)
    dx = x[0, 1] - x[0, 0]
    r_samples = 100
    r_circle = dx*r_samples
    r, _ = coordinates.cart_to_polar(x, y)
    mask = geometry.antialias(geometry.circle_sdf(r_circle, r), dx)
    expected_area_of_circle = r_samples*r_samples * math.pi
    assert mask.sum() == pytest.approx(expected_area_of_circle, abs=1.5)


def test_antialias_physical_grid_area_and_registration():
    dx = 0.05
    x, y = coordinates.make_xy_grid(256, dx=dx)
    r = np.hypot(x, y)
    r_samples = 80
    mask = geometry.antialias(geometry.circle_sdf(dx * r_samples, r), dx)
    assert mask.sum() == pytest.approx(r_samples * r_samples * math.pi, abs=1.5)
    # centered: first moment is zero to machine precision
    assert abs((mask * x).sum() / mask.sum()) < 1e-12
    assert abs((mask * y).sum() / mask.sum()) < 1e-12


def test_antialias_hexagon_correct_area():
    dx = 0.05
    x, y = coordinates.make_xy_grid(256, dx=dx)
    radius = 4.0
    d = geometry.regular_polygon_sdf(6, radius, x, y, rotation=15)
    cover = geometry.antialias(d, dx)
    analytic = 3 * math.sqrt(3) / 2 * radius * radius
    assert cover.sum() * dx * dx == pytest.approx(analytic, rel=1e-4)


def test_sdf_pairing_invariant():
    # every binary rasterizer is its SDF thresholded at zero
    x, y = coordinates.make_xy_grid(65, diameter=2)
    r = np.hypot(x, y)
    pairs = [
        (geometry.circle(0.5, r),
         geometry.circle_sdf(0.5, r)),
        (geometry.annulus(0.2, 0.5, r),
         geometry.annulus_sdf(0.2, 0.5, r)),
        (geometry.rectangle(0.5, x, y, height=0.25, angle=17),
         geometry.rectangle_sdf(0.5, x, y, height=0.25, angle=17)),
        (geometry.rotated_ellipse(0.6, 0.3, x, y, major_axis_angle=30),
         geometry.rotated_ellipse_sdf(0.6, 0.3, x, y, major_axis_angle=30)),
        (geometry.regular_polygon(5, 0.5, x, y, rotation=10),
         geometry.regular_polygon_sdf(5, 0.5, x, y, rotation=10)),
        (geometry.spider(3, 0.05, x, y, rotation=20),
         geometry.spider_sdf(3, 0.05, x, y, rotation=20)),
        (geometry.rectangle_with_corner_fillets(0.5, 0.4, 0.1, x, y),
         geometry.rectangle_with_corner_fillets_sdf(0.5, 0.4, 0.1, x, y)),
    ]
    for binary, d in pairs:
        assert (np.asarray(binary) == (d <= 0)).all()


def test_polygon_tolerates_empty_windows():
    # segmented apertures rasterize per-segment windows; off-grid segments are empty
    x = np.zeros((0, 5))
    y = np.zeros((0, 5))
    mask = geometry.regular_polygon(6, 1, x, y)
    assert mask.shape == (0, 5)


def test_polygon_sdf_winding_invariant():
    x, y = coordinates.make_xy_grid(65, diameter=2)
    verts = geometry._generate_vertices(6, 0.6, rotation=15)
    cw = geometry.polygon_sdf(verts, x, y)
    ccw = geometry.polygon_sdf(verts[::-1], x, y)
    np.testing.assert_allclose(cw, ccw, atol=1e-12)


def test_composition_helpers():
    x, y = coordinates.make_xy_grid(65, diameter=2)
    r = np.hypot(x, y)
    inner = geometry.circle_sdf(0.2, r)
    outer = geometry.circle_sdf(0.5, r)
    composed = geometry.subtract(outer, inner)
    np.testing.assert_allclose(composed, geometry.annulus_sdf(0.2, 0.5, r), atol=1e-15)
    # union of the annulus with its hole fills the disk back in
    refilled = geometry.union(composed, inner)
    assert ((refilled <= 0) == (outer <= 0)).all()
    # intersection with the hole is empty
    assert not (geometry.intersect(composed, inner) <= 0).any()


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


def test_spider_masks_vane_region():
    x, y = coordinates.make_xy_grid(65, diameter=2)

    mask = geometry.spider(1, 0.2, x, y)

    assert mask[32, 48]      # on the +x vane
    assert not mask[32, 16]  # -x, no vane
    assert not mask[48, 32]  # +y, no vane


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


def test_multisample_centered_and_matches_antialias():
    dx = 0.05
    x, y = coordinates.make_xy_grid(128, dx=dx)
    radius = 40 * dx

    def member(xx, yy):
        return np.hypot(xx, yy) <= radius

    cover = geometry.multisample(member, x, y, samples=8)
    assert cover.sum() == pytest.approx(40 * 40 * math.pi, abs=1.5)
    # centered subsample offsets: no registration bias
    assert abs((cover * x).sum() / cover.sum()) < 1e-12
    assert abs((cover * y).sum() / cover.sum()) < 1e-12
    # agrees with the analytic ramp to within the two edge models
    r = np.hypot(x, y)
    analytic = geometry.antialias(geometry.circle_sdf(radius, r), dx)
    assert float(abs(cover - analytic).max()) < 0.1
