"""tests for all polynomials."""
import pytest

import numpy as np

from prysm.coordinates import cart_to_polar
from prysm import polynomials

from scipy.special import jacobi as sps_jac


# TODO: add regression tests against scipy.special.eval_legendre etc

SAMPLES = 32
X, Y = np.linspace(-1, 1, SAMPLES), np.linspace(-1, 1, SAMPLES)


@pytest.fixture
def rho():
    rho, phi = cart_to_polar(X, Y)
    return rho


@pytest.fixture
def phi():
    rho, phi = cart_to_polar(X, Y)
    return phi


# - Q poly


@pytest.mark.parametrize('n', [0, 1, 2, 3, 4, 5, 6])
def test_qbfs_functions(n, rho):
    sag = polynomials.Qbfs(n, rho)
    assert sag.any()


def test_qbfs_sequence_functions(rho):
    ns = [1, 2, 3, 4, 5, 6]
    gen = polynomials.Qbfs_sequence(ns, rho)
    assert len(list(gen)) == len(ns)


@pytest.mark.parametrize('n', [0, 1, 2, 3, 4, 5, 6])
def test_qcon_functions(n, rho):
    sag = polynomials.Qcon(n, rho)
    assert sag.any()


def test_qcon_sequence_functions(rho):
    ns = [1, 2, 3, 4, 5, 6]
    gen = polynomials.Qcon_sequence(ns, rho)
    assert len(list(gen)) == len(ns)

# there are truth tables in the paper, which are not used here.  Some of them contain
# typos, so the test would have to be very loose, e.g. 0.05 atol.  A visual check
# is equally valuable, so we only check functionality here.


@pytest.mark.parametrize('nm', [
    (1, 1),
    (2, 0),
    (3, 1),
    (2, 2),
    (2, -2),
    (4, 0),
    (7, 7),
])
def test_2d_Q(nm, rho, phi):
    sag = polynomials.Q2d(*nm, rho, phi)
    assert sag.any()


def test_2d_Q_sequence_functions(rho, phi):
    nms = [polynomials.noll_to_nm(i) for i in range(1, 11)]
    modes = list(polynomials.Q2d_sequence(nms, rho, phi))
    assert len(modes) == len(nms)


# - zernike


@pytest.mark.parametrize('n', [2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
def test_zero_separation_gives_correct_array_sizes(n):
    sep = polynomials.zernike_zero_separation(n)
    assert int(1/sep) == int(n**2)


@pytest.mark.parametrize('fringe_idx', range(1, 100))
def test_nm_to_fringe_round_trips(fringe_idx):
    n, m = polynomials.fringe_to_nm(fringe_idx)
    j = polynomials.nm_to_fringe(n, m)
    assert j == fringe_idx


def test_ansi_2_term_can_construct(rho, phi):
    ary = polynomials.zernike_nm(3, 1, rho, phi)
    assert ary.any()


@pytest.mark.parametrize('n', [0, 1, 2, 3, 4])
@pytest.mark.parametrize('alpha, beta', [
    (0, 0),
    (1, 1),
    (-0.75, 0),
    (1, -0.75)])
def test_jacobi_1_4_match_scipy(n, alpha, beta):
    x = np.linspace(-1, 1, 32)
    prysm_ = polynomials.jacobi(n=n, alpha=alpha, beta=beta, x=x)
    scipy_ = sps_jac(n=n, alpha=alpha, beta=beta)(x)
    assert np.allclose(prysm_, scipy_)
