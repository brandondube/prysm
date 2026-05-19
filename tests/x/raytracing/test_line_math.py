"""Tests for prysm.x.raytracing._line_math."""

import numpy as np

from prysm.x.raytracing import _line_math


def test_line_intersection_params_and_unit_vector_between():
    P1 = np.array([0., 0., 0.])
    S1 = np.array([1., 0., 0.])
    P2 = np.array([2., -1., 0.])
    S2 = np.array([0., 1., 0.])
    s = _line_math.line_intersection_params(P1, S1, P2, S2)
    np.testing.assert_allclose(P1 + s[0] * S1, P2 + s[1] * S2)
    np.testing.assert_allclose(_line_math.unit_vector_between(P1, [0, 0, 3]),
                               [0, 0, 1])


def test_line_intersection_params_parallel_rays():
    P1 = np.array([0., 0., 0.])
    S1 = np.array([1., 0., 0.])
    P2 = np.array([0., 1., 0.])
    S2 = np.array([1., 0., 0.])
    s = _line_math.line_intersection_params(P1, S1, P2, S2)
    assert np.isfinite(s).all()


def test_closest_point_on_line_to_line():
    # query line along x at y=1; axis is the z-axis. Closest point is origin.
    P = np.array([0., 1., 0.])
    S = np.array([1., 0., 0.])
    axis_point = np.array([0., 0., 0.])
    axis_dir = np.array([0., 0., 1.])
    pt = _line_math.closest_point_on_line_to_line(P, S, axis_point, axis_dir)
    np.testing.assert_allclose(pt, [0., 0., 0.], atol=1e-12)
