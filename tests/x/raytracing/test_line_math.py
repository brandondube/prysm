"""Tests for prysm.x.raytracing._line_math."""

import numpy as np

from prysm.x.raytracing import _line_math


def test_unit_vector_between():
    np.testing.assert_allclose(
        _line_math.unit_vector_between([0., 0., 0.], [0, 0, 3]), [0, 0, 1])


def test_closest_point_on_line_to_line():
    # query line along x at y=1; axis is the z-axis. Closest point is origin.
    P = np.array([0., 1., 0.])
    S = np.array([1., 0., 0.])
    axis_point = np.array([0., 0., 0.])
    axis_dir = np.array([0., 0., 1.])
    pt = _line_math.closest_point_on_line_to_line(P, S, axis_point, axis_dir)
    np.testing.assert_allclose(pt, [0., 0., 0.], atol=1e-12)
