import pytest

import numpy as np

from prysm.x.optym.operators import SpatialGradient2D


@pytest.mark.parametrize('shape', [(5, 7), (7, 5)])
def test_spatial_gradient_x_adjoint_is_adjoint(shape):
    rng = np.random.default_rng(1234)
    op = SpatialGradient2D()
    x = rng.normal(size=shape)
    y = rng.normal(size=shape)

    lhs = np.vdot(op.forward_x(x), y)
    rhs = np.vdot(x, op.adjoint_x(y))

    np.testing.assert_allclose(lhs, rhs, atol=1e-12)


@pytest.mark.parametrize('shape', [(5, 7), (7, 5)])
def test_spatial_gradient_y_adjoint_is_adjoint(shape):
    rng = np.random.default_rng(4321)
    op = SpatialGradient2D()
    x = rng.normal(size=shape)
    y = rng.normal(size=shape)

    lhs = np.vdot(op.forward_y(x), y)
    rhs = np.vdot(x, op.adjoint_y(y))

    np.testing.assert_allclose(lhs, rhs, atol=1e-12)
