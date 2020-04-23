"""Jacobi submodule tests."""

import numpy as np

import pytest

from scipy.special import jacobi as sps_jac

from prysm import jacobi as pjac

@pytest.mark.parametrize('n', [0, 1, 2, 3, 4])
@pytest.mark.parametrize('alpha, beta', [
    (0,0),
    (1,1),
    (-0.75,0),
    (1,-0.75)])
def test_jacobi_1_4_match_scipy(n, alpha, beta):
    x = np.linspace(-1, 1, 32)
    prysm_ = pjac.jacobi(n=n, alpha=alpha, beta=beta, x=x)
    scipy_ = sps_jac(n=n, alpha=alpha, beta=beta)(x)
    assert np.allclose(prysm_, scipy_)
