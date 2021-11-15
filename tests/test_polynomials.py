"""tests for all polynomials."""
import pytest

import numpy as np

from prysm.coordinates import cart_to_polar, make_xy_grid
from prysm import polynomials

from scipy.special import (
    jacobi as sps_jac,
    legendre as sps_leg,
    chebyt as sps_cheby1,
    chebyu as sps_cheby2
)


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


@pytest.mark.parametrize('j', range(1, 100))
def test_ansij_roudn_trips(j):
    n, m = polynomials.ansi_j_to_nm(j)
    jj = polynomials.nm_to_ansi_j(n, m)
    assert j == jj


def test_ansi_2_term_can_construct(rho, phi):
    ary = polynomials.zernike_nm(3, 1, rho, phi)
    assert ary.any()


def test_zernike_sequence_same_as_loop(rho, phi):
    nms = (
        (2, 0),  # defocus
        (4, 0),  # sph1
        (6, 0),
        (8, 0),  # sph3
        (2, 2),  # ast, cma, trefoil sort of out of order, test there isn't some implicit assumption about ordering
        (2, -2),
        (3, 1),
        (3, 3),
        (3, -1),
        (3, -3),
    )
    seq = list(polynomials.zernike_nm_sequence(nms, rho, phi))
    for elem, nm in zip(seq, nms):
        exp = polynomials.zernike_nm(*nm, rho, phi)
        assert np.allclose(exp, elem)


def test_zernike_to_magang_functions():
    # data has piston, tt, power, sph, ast, cma, tre = 7 unique things
    data = [
        (0, 0, 1),
        (1, 1, 1),
        (1, -1, 1),
        (2, 0, 1),
        (4, 0, 1),
        (2, 2, 1),
        (2, -2, 1),
        (3, 1, 1),
        (3, -1, 1),
        (3, 3, 1),
        (3, -3, 1)
    ]
    magang = polynomials.zernikes_to_magnitude_angle(data)
    # TODO: also test correct magnitude and angle
    assert len(magang) == 7


def test_zernike_topn_correct():
    data = {
        (3, 1): 1,
        (3, -1): -1,
        (2, 0): 10,  # mag 10 index 1 term == (2,0)
        (4, 0): 9,
        (6, 0): 12,
        (2, 2): 8,
        (3, 3): 7,
    }
    exp = [
        (12, 4, 'Secondary Spherical'),
        (10, 2, 'Defocus'),
        (9, 3, 'Primary Spherical'),
        (8, 5, 'Primary Astigmatism 00°'),
        (7, 6, 'Primary Trefoil X')
    ]
    res = polynomials.top_n(data, 5)
    assert exp == res


def test_barplot_functions():
    data = {
        2: 1,
        3: 2,
        4: 3,
        5: 4,
        6: 5
    }
    fig, ax = polynomials.zernike_barplot(data)
    assert fig, ax


def test_barplot_magnitudes_functions():
    data = {
        2: 1,
        3: 2,
        4: 3,
        5: 4,
        6: 5
    }
    fig, ax = polynomials.zernike_barplot_magnitudes(data)
    assert fig, ax


@pytest.mark.parametrize('n', [0, 1, 2, 3, 4])
@pytest.mark.parametrize('alpha, beta', [
    (0, 0),
    (1, 1),
    (-0.75, 0),
    (1, -0.75)])
def test_jacobi_1_4_match_scipy(n, alpha, beta):
    prysm_ = polynomials.jacobi(n=n, alpha=alpha, beta=beta, x=X)
    scipy_ = sps_jac(n=n, alpha=alpha, beta=beta)(X)
    assert np.allclose(prysm_, scipy_)


def test_jacobi_weight_correct():
    from prysm.polynomials.jacobi import weight
    # these are cheby1 weights
    alpha = -0.5
    beta = -0.5
    x = X
    res = weight(alpha, beta, x)
    exp = (1-x)**alpha * (1+x)**beta
    assert np.allclose(res, exp)


@pytest.mark.parametrize('n', [0, 1, 2, 3, 4, 5])
def test_legendre_matches_scipy(n):
    prysm_ = polynomials.legendre(n, X)
    scipy_ = sps_leg(n)(X)
    assert np.allclose(prysm_, scipy_)


def test_legendre_sequence_matches_loop():
    ns = [1, 2, 3, 4, 5]
    seq = polynomials.legendre_sequence(ns, X)
    loop = [polynomials.legendre(n, X) for n in ns]
    for elem, exp in zip(seq, loop):
        assert np.allclose(elem, exp)


@pytest.mark.parametrize('n', [0, 1, 2, 3, 4, 5])
def test_cheby1_matches_scipy(n):
    prysm_ = polynomials.cheby1(n, X)
    scipy_ = sps_cheby1(n)(X)
    assert np.allclose(prysm_, scipy_)


@pytest.mark.parametrize('n', [0, 1, 2, 3, 4, 5])
def test_cheby2_matches_scipy(n):
    prysm_ = polynomials.cheby2(n, X)
    scipy_ = sps_cheby2(n)(X)
    assert np.allclose(prysm_, scipy_)


def test_cheby1_seq_matches_loop():
    ns = [0, 1, 2, 3, 4, 5]
    seq = list(polynomials.cheby1_sequence(ns, X))
    for elem, n in zip(seq, ns):
        exp = polynomials.cheby1(n, X)
        assert np.allclose(exp, elem)


def test_cheby2_seq_matches_loop():
    ns = [0, 1, 2, 3, 4, 5]
    seq = list(polynomials.cheby2_sequence(ns, X))
    for elem, n in zip(seq, ns):
        exp = polynomials.cheby2(n, X)
        assert np.allclose(exp, elem)


@pytest.mark.parametrize('n', [1, 2, 3, 4, 8])
def test_dickson1_alpha0_powers(n):
    d = polynomials.dickson1(n, 0, X)
    exp = X ** n
    assert np.allclose(exp, d)


@pytest.mark.parametrize('n', [1, 2, 3, 4, 8])
def test_dickson1_alpha1_cheby(n):
    d = polynomials.dickson1(n, 1, 2*X)
    c = polynomials.cheby1(n, X)
    assert np.allclose(d, 2*c)


# no known identities
@pytest.mark.parametrize('n', [1, 2, 3, 4, 5])
def test_dickson2_functions(n):
    d = polynomials.dickson2(n, 1, X)
    assert d.any()


def test_dickson1_seq_matches_loop():
    ns = [0, 1, 2, 3, 4, 5]
    seq = list(polynomials.dickson1_sequence(ns, 1, X))
    for elem, n in zip(seq, ns):
        exp = polynomials.dickson1(n, 1, X)
        assert np.allclose(exp, elem)


def test_dickson2_seq_matches_loop():
    ns = [0, 1, 2, 3, 4, 5]
    seq = list(polynomials.dickson2_sequence(ns, 1, X))
    for elem, n in zip(seq, ns):
        exp = polynomials.dickson2(n, 1, X)
        assert np.allclose(exp, elem)


@pytest.mark.parametrize('n', [1, 2, 3, 4, 5])
def test_jacobi_der_matches_finite_diff(n):
    # need more points for accurate finite diff
    x = np.linspace(-1, 1, 128)
    Pn = polynomials.jacobi(n, 1, 1, x)
    Pnprime = polynomials.jacobi_der(n, 1, 1, x)
    dx = x[1] - x[0]
    Pnprime_numerical = np.gradient(Pn, dx)
    ratio = Pnprime / Pnprime_numerical
    assert abs(ratio-1).max() < 0.1  # 10% relative error


def test_jacobi_der_sequence_same_as_loop():
    ns = [0, 1, 2, 3, 4, 5]
    seq = list(polynomials.jacobi_der_sequence(ns, 0.5, 0.5, X))
    for elem, n in zip(seq, ns):
        exp = polynomials.jacobi_der(n, 0.5, 0.5, X)
        assert np.allclose(exp, elem)


@pytest.mark.parametrize('n', [1, 2, 3, 4, 5])
def test_cheby1_der_matches_finite_diff(n):
    # need more points for accurate finite diff
    x = np.linspace(-1, 1, 128)
    Pn = polynomials.cheby1(n, x)
    Pnprime = polynomials.cheby1_der(n, x)
    dx = x[1] - x[0]
    Pnprime_numerical = np.gradient(Pn, dx)
    ratio = Pnprime / Pnprime_numerical
    assert abs(ratio-1).max() < 0.15  # 15% relative error


def test_cheby1_der_sequence_same_as_loop():
    ns = [0, 1, 2, 3, 4, 5]
    seq = list(polynomials.cheby1_der_sequence(ns, X))
    for elem, n in zip(seq, ns):
        exp = polynomials.cheby1_der(n, X)
        assert np.allclose(exp, elem)


@pytest.mark.parametrize('n', [1, 2, 3, 4, 5])
def test_cheby2_der_matches_finite_diff(n):
    # need more points for accurate finite diff
    x = np.linspace(-1, 1, 128)
    Pn = polynomials.cheby2(n, x)
    Pnprime = polynomials.cheby2_der(n, x)
    dx = x[1] - x[0]
    Pnprime_numerical = np.gradient(Pn, dx)
    ratio = Pnprime / Pnprime_numerical
    assert abs(ratio-1).max() < 0.15  # 15% relative error


def test_cheby2_der_sequence_same_as_loop():
    ns = [0, 1, 2, 3, 4, 5]
    seq = list(polynomials.cheby2_der_sequence(ns, X))
    for elem, n in zip(seq, ns):
        exp = polynomials.cheby2_der(n, X)
        assert np.allclose(exp, elem)


@pytest.mark.parametrize('n', [1, 2, 3, 4, 5])
def test_legendre_der_matches_finite_diff(n):
    # need more points for accurate finite diff
    x = np.linspace(-1, 1, 128)
    Pn = polynomials.legendre(n, x)
    Pnprime = polynomials.legendre_der(n, x)
    dx = x[1] - x[0]
    Pnprime_numerical = np.gradient(Pn, dx)
    ratio = Pnprime / Pnprime_numerical
    assert abs(ratio-1).max() < 0.15  # 15% relative error


def test_legendre_der_sequence_same_as_loop():
    ns = [0, 1, 2, 3, 4, 5]
    seq = list(polynomials.legendre_der_sequence(ns, X))
    for elem, n in zip(seq, ns):
        exp = polynomials.legendre_der(n, X)
        assert np.allclose(exp, elem)


# - higher order routines

def test_sum_and_lstsq():
    x, y = make_xy_grid(100, diameter=2)
    ns = [0, 1, 2, 3, 4, 5]
    ms = [1, 2, 3, 4, 5, 6, 7]
    weights_x = np.random.rand(len(ns))
    weights_y = np.random.rand(len(ms))
    # "fun" thing, mix first and second kind chebyshev polynomials
    mx, my = polynomials.separable_2d_sequence(ns, ms, x, y,
                                               polynomials.cheby1_sequence,
                                               polynomials.cheby2_sequence)

    data = polynomials.sum_of_xy_modes(mx, my, x, y, weights_x, weights_y)
    mx = [polynomials.mode_1d_to_2d(m, x, y, 'x') for m in mx]
    my = [polynomials.mode_1d_to_2d(m, x, y, 'y') for m in my]
    modes = mx + my  # concat
    exp = list(weights_x) + list(weights_y)  # concat
    coefs = polynomials.lstsq(modes, data)
    assert np.allclose(coefs, exp)


@pytest.mark.parametrize(['a', 'b', 'c'], [
    [1, 1, 1],
    [1, 3, 1],
    [0, 2, 0],
    [0, 4, 0],
    [2, 2, 2]])
def test_hopkins_correct(a, b, c, rho, phi):
    H = np.sqrt(2)/2
    res = polynomials.hopkins(a, b, c, rho, phi, H)
    exp = np.cos(a*phi) * rho ** b * H ** c  # H =
    assert np.allclose(res, exp)
