"""tests for all polynomials."""
import pytest

import numpy as np
from prysm import coordinates

from prysm.coordinates import cart_to_polar
from prysm import polynomials

from scipy.special import (
    jacobi as sps_jac,
    legendre as sps_leg,
    chebyt as sps_cheby1,
    chebyu as sps_cheby2,
    hermite as sps_H,
    hermitenorm as sps_He,
    genlaguerre as sps_laguerre,
)


SAMPLES = 32
X, Y = np.linspace(-1, 1, SAMPLES), np.linspace(-1, 1, SAMPLES)
rho, phi = cart_to_polar(X, Y)

# for Laguerre, orthogonal on [0,inf]; cutoff at 10 is arbitrary
XLEFT = np.linspace(0, 10, SAMPLES)


@pytest.fixture
def rho():
    rho, phi = cart_to_polar(X, Y)
    return rho


@pytest.fixture
def phi():
    rho, phi = cart_to_polar(X, Y)
    return phi


# XY poly

xy_poly_truth_table = [  # NOQA
    [np.nan,  2,  4,  7, 11, 16, 22, 29, 37, 46, 56],  # NOQA
    [     3,  5,  8, 12, 17, 23, 30, 38, 47, 57],  # NOQA
    [     6,  9, 13, 18, 24, 31, 39, 48, 58],  # NOQA
    [    10, 14, 19, 25, 32, 40, 49, 59],  # NOQA
    [    15, 20, 26, 33, 41, 50, 60],  # NOQA
    [    21, 27, 34, 42, 51, 61],  # NOQA
    [    28, 35, 43, 52, 62],  # NOQA
    [    36, 44, 53, 63],  # NOQA
    [    45, 54, 64],  # NOQA
    [    55, 65],  # NOQA
    [    66],  # NOQA
]


@pytest.mark.parametrize('j', np.arange(2, 67))
def test_xy_poly_mapping_roundtrip(j):
    n, m = polynomials.xy_j_to_mn(j)
    assert xy_poly_truth_table[m][n] == j


def test_xy_poly_first_cross_term():
    m = n = 1
    xx, yy = np.meshgrid(X, Y)
    prysm_calc = polynomials.xy(m, n, xx, yy)
    truth = xx * yy
    assert np.allclose(prysm_calc, truth)


def test_xy_poly_later_cross_term():
    m = 1
    n = 3
    xx, yy = np.meshgrid(X, Y)
    prysm_calc = polynomials.xy(m, n, xx, yy)
    truth = xx * yy**3
    assert np.allclose(prysm_calc, truth)


def test_xy_poly_seq_cross_terms():
    mns = [
        (1, 1),
        (1, 3),
    ]
    xx, yy = np.meshgrid(X, Y)
    prysm_calc1, prysm_calc2 = polynomials.xy_seq(mns, xx, yy)
    truth1 = xx * yy
    truth2 = xx * yy ** 3
    assert np.allclose(prysm_calc1, truth1)
    assert np.allclose(prysm_calc2, truth2)
# - Q poly


def test_qbfs_first_two_orders_match_closed_form(rho):
    rho2 = rho * rho
    c_q = rho2 * (1 - rho2)

    np.testing.assert_allclose(polynomials.Qbfs(0, rho), c_q)
    np.testing.assert_allclose(polynomials.Qbfs(1, rho), (13 - 16 * rho2) * c_q / np.sqrt(19))


def test_qbfs_and_qcon_sequences_match_scalar_evaluation(rho):
    ns = [0, 1, 2, 3]

    qbfs_seq = list(polynomials.Qbfs_seq(ns, rho))
    qcon_seq = list(polynomials.Qcon_seq(ns, rho))

    for i, n in enumerate(ns):
        np.testing.assert_allclose(qbfs_seq[i], polynomials.Qbfs(n, rho))
        np.testing.assert_allclose(qcon_seq[i], polynomials.Qcon(n, rho))


def test_2d_Q_seq_same_as_loop(rho, phi):
    nms = [polynomials.noll_to_nm(i) for i in range(1, 11)]
    modes = list(polynomials.Q2d_seq(nms, rho, phi))
    iterated = [polynomials.Q2d(n, m, rho, phi) for n, m in nms]
    for m, i in zip(modes, iterated):
        assert np.allclose(m, i)


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


def test_zernike_seq_same_as_loop(rho, phi):
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
    seq = list(polynomials.zernike_nm_seq(nms, rho, phi))
    for elem, nm in zip(seq, nms):
        exp = polynomials.zernike_nm(*nm, rho, phi)
        assert np.allclose(exp, elem)


@pytest.mark.parametrize('norm', [True, False])
def test_zernike_der_seq_same_as_loop(norm, rho, phi):
    nms = [polynomials.noll_to_nm(j) for j in range(0, 12)]
    loop = []
    for n, m in nms:
        loop.append(polynomials.zernike_nm_der(n, m, rho, phi, norm=norm))

    non_loop = polynomials.zernike_nm_der_seq(nms, rho, phi, norm=norm)
    for looped, not_looped in zip(loop, non_loop):
        rl, tl = looped
        rnl, tnl = not_looped
        assert np.allclose(rl, rnl)
        assert np.allclose(tl, tnl)


_zernike_xy_nms = [(0, 0), (1, 1), (1, -1), (2, 0), (2, 2), (2, -2),
                   (3, 1), (3, -1), (3, 3), (3, -3), (4, 0), (4, 2), (4, -4),
                   (5, 1), (5, 5), (6, 0)]


@pytest.mark.parametrize('n,m', _zernike_xy_nms)
@pytest.mark.parametrize('norm', [True, False])
def test_zernike_nm_der_xy_matches_numerical(n, m, norm):
    # stay strictly inside the unit disk so finite differences don't escape it
    xs = np.linspace(-0.7, 0.7, 41)
    X, Y = np.meshgrid(xs, xs, indexing='xy')
    h = 1e-5

    def zern_xy(xx, yy):
        rho_ = np.sqrt(xx * xx + yy * yy)
        phi_ = np.arctan2(yy, xx)
        return polynomials.zernike_nm(n, m, rho_, phi_, norm=norm)

    dzdx_num = (zern_xy(X + h, Y) - zern_xy(X - h, Y)) / (2 * h)
    dzdy_num = (zern_xy(X, Y + h) - zern_xy(X, Y - h)) / (2 * h)
    dzdx_an, dzdy_an = polynomials.zernike_nm_der_xy(n, m, X, Y, norm=norm)
    np.testing.assert_allclose(dzdx_an, dzdx_num, atol=1e-7, rtol=1e-5)
    np.testing.assert_allclose(dzdy_an, dzdy_num, atol=1e-7, rtol=1e-5)


@pytest.mark.parametrize('n,m', _zernike_xy_nms)
def test_zernike_nm_der_xy_finite_at_origin(n, m):
    """Cartesian derivatives must be finite at x=y=0 (no 1/r singularity)."""
    zero = np.array(0.0)
    dx, dy = polynomials.zernike_nm_der_xy(n, m, zero, zero)
    assert np.isfinite(dx) and np.isfinite(dy)


def test_zernike_nm_der_xy_seq_matches_loop():
    nms = [polynomials.noll_to_nm(j) for j in range(0, 12)]
    xs = np.linspace(-0.7, 0.7, 21)
    X, Y = np.meshgrid(xs, xs, indexing='xy')
    looped = [polynomials.zernike_nm_der_xy(n, m, X, Y) for n, m in nms]
    seq = polynomials.zernike_nm_der_xy_seq(nms, X, Y)
    for (dx_l, dy_l), entry in zip(looped, seq):
        assert np.allclose(dx_l, entry[0])
        assert np.allclose(dy_l, entry[1])


def _zernike_sum_loop(coefs, nms, X, Y, norm):
    rho, phi = coordinates.cart_to_polar(X, Y)
    W = np.zeros_like(X)
    dWx = np.zeros_like(X)
    dWy = np.zeros_like(X)
    for c, (n, m) in zip(coefs, nms):
        W += c * polynomials.zernike_nm(n, m, rho, phi, norm=norm)
        dx, dy = polynomials.zernike_nm_der_xy(n, m, X, Y, norm=norm)
        dWx += c * dx
        dWy += c * dy
    return W, dWx, dWy


@pytest.mark.parametrize('norm', [True, False])
def test_zernike_sum_der_xy_matches_loop(norm):
    nms = [(0, 0), (1, 1), (1, -1), (2, 0), (2, 2), (2, -2),
           (3, 1), (3, -1), (3, 3), (3, -3),
           (4, 0), (4, 2), (4, -4), (5, 1), (5, -3), (5, 5),
           (6, 0), (6, 4), (6, -6)]
    rng = np.random.default_rng(42)
    coefs = rng.normal(size=len(nms))
    xs = np.linspace(-0.7, 0.7, 41)
    X, Y = np.meshgrid(xs, xs, indexing='xy')

    W_l, dWx_l, dWy_l = _zernike_sum_loop(coefs, nms, X, Y, norm)
    W_c, dWx_c, dWy_c = polynomials.zernike_sum_der_xy(coefs, nms, X, Y, norm=norm)
    W_s = polynomials.zernike_sum(coefs, nms, X, Y, norm=norm)
    np.testing.assert_allclose(W_s, W_l, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(W_c, W_l, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(dWx_c, dWx_l, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(dWy_c, dWy_l, atol=1e-12, rtol=1e-12)


def test_weighted_surface_sum_helpers_match_basis_loops():
    xs = np.linspace(-0.5, 0.5, 9)
    X, Y = np.meshgrid(xs, xs, indexing='xy')

    mns = [(0, 0), (1, 0), (0, 2), (2, 1)]
    coefs = [0.5, -0.25, 0.1, 0.05]
    z_xy, dx_xy, dy_xy = polynomials.xy_sum_der_xy(
        coefs, mns, X, Y, cartesian_grid=False)
    z_ref = sum(c * X**m * Y**n for c, (m, n) in zip(coefs, mns))
    dx_ref = sum(c * m * X**(m - 1) * Y**n
                 for c, (m, n) in zip(coefs, mns) if m)
    dy_ref = sum(c * n * X**m * Y**(n - 1)
                 for c, (m, n) in zip(coefs, mns) if n)
    np.testing.assert_allclose(polynomials.xy_sum(
        coefs, mns, X, Y, cartesian_grid=False), z_ref)
    np.testing.assert_allclose(z_xy, z_ref)
    np.testing.assert_allclose(dx_xy, dx_ref)
    np.testing.assert_allclose(dy_xy, dy_ref)

    orders = [0, 1, 2]
    jcoefs = [0.2, -0.1, 0.05]
    z_j, dx_j, dy_j = polynomials.jacobi_radial_sum_der_xy(
        jcoefs, orders, 0.0, 0.0, X, Y, 1.0)
    np.testing.assert_allclose(
        z_j,
        polynomials.jacobi_radial_sum(jcoefs, orders, 0.0, 0.0, X, Y, 1.0),
    )
    assert np.isfinite(dx_j).all()
    assert np.isfinite(dy_j).all()


def test_xy_sum_cartesian_grid_matches_generic_path_with_duplicates():
    xs = np.linspace(-0.5, 0.5, 11)
    X, Y = np.meshgrid(xs, xs, indexing='xy')
    mns = [(0, 0), (1, 0), (0, 2), (2, 1), (2, 1)]
    coefs = [0.5, -0.25, 0.1, 0.05, -0.02]

    z_fast = polynomials.xy_sum(coefs, mns, X, Y, cartesian_grid=True)
    z_ref = polynomials.xy_sum(coefs, mns, X, Y, cartesian_grid=False)
    np.testing.assert_allclose(z_fast, z_ref)

    fast = polynomials.xy_sum_der_xy(coefs, mns, X, Y, cartesian_grid=True)
    ref = polynomials.xy_sum_der_xy(coefs, mns, X, Y, cartesian_grid=False)
    for fast_elem, ref_elem in zip(fast, ref):
        np.testing.assert_allclose(fast_elem, ref_elem)


def test_zernike_sum_der_xy_finite_at_origin():
    nms = [(2, 0), (3, 1), (3, -1), (4, 0), (5, 5)]
    coefs = [0.3, -1.2, 0.5, 0.1, 0.7]
    zero = np.array(0.0)
    W, dx, dy = polynomials.zernike_sum_der_xy(coefs, nms, zero, zero)
    assert np.isfinite(W) and np.isfinite(dx) and np.isfinite(dy)


def test_zernike_sum_der_xy_handles_single_mode_and_duplicates():
    xs = np.linspace(-0.7, 0.7, 17)
    X, Y = np.meshgrid(xs, xs, indexing='xy')
    # single mode (exercises the len-1 padding inside the Clenshaw helper)
    W, dx, dy = polynomials.zernike_sum_der_xy([2.0], [(0, 0)], X, Y, norm=True)
    np.testing.assert_allclose(W, 2.0 * np.ones_like(X))
    np.testing.assert_allclose(dx, np.zeros_like(X), atol=0)
    np.testing.assert_allclose(dy, np.zeros_like(X), atol=0)
    # duplicate (n,m) entries should sum
    W2, dx2, dy2 = polynomials.zernike_sum_der_xy([1.0, 1.0], [(2, 2), (2, 2)], X, Y, norm=True)
    Wref, dxref, dyref = polynomials.zernike_sum_der_xy([2.0], [(2, 2)], X, Y, norm=True)
    np.testing.assert_allclose(W2, Wref, atol=1e-14)
    np.testing.assert_allclose(dx2, dxref, atol=1e-14)
    np.testing.assert_allclose(dy2, dyref, atol=1e-14)
    # trailing zero coefficients should not force high-order recurrence work
    high_zero = [(2, 2), (12, 0), (15, -3)]
    W3, dx3, dy3 = polynomials.zernike_sum_der_xy([2.0, 0.0, 0.0], high_zero, X, Y, norm=True)
    np.testing.assert_allclose(W3, Wref, atol=1e-14)
    np.testing.assert_allclose(dx3, dxref, atol=1e-14)
    np.testing.assert_allclose(dy3, dyref, atol=1e-14)


def test_zernikes_to_magnitude_angle_combines_sine_cosine_pairs():
    data = [
        (2, 2, 3),
        (2, -2, 4),
        (2, 0, 5),
    ]

    magang = polynomials.zernikes_to_magnitude_angle(data)

    assert magang['Defocus'] == (5, 0)
    assert magang['Primary Astigmatism'][0] == pytest.approx(5)
    assert magang['Primary Astigmatism'][1] == pytest.approx(np.degrees(np.arctan2(3, 4)))


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
    with np.testing.suppress_warnings() as sup:
        sup.filter(RuntimeWarning)
        res = weight(alpha, beta, x)
        exp = (1-x)**alpha * (1+x)**beta
    assert np.allclose(res, exp)


@pytest.mark.parametrize('n', [0, 1, 2, 3, 4, 5])
def test_legendre_matches_scipy(n):
    prysm_ = polynomials.legendre(n, X)
    scipy_ = sps_leg(n)(X)
    assert np.allclose(prysm_, scipy_)


def test_legendre_seq_matches_loop():
    ns = [1, 2, 3, 4, 5]
    seq = polynomials.legendre_seq(ns, X)
    loop = [polynomials.legendre(n, X) for n in ns]
    for elem, exp in zip(seq, loop):
        assert np.allclose(elem, exp)


@pytest.mark.parametrize('n', [0, 1, 2, 3, 4, 5])
def test_hermite_he_matches_scipy(n):
    prysm_ = polynomials.hermite_He(n, X)
    scipy_ = sps_He(n)(X)
    assert np.allclose(prysm_, scipy_)


def test_hermite_he_seq_matches_loop():
    ns = [1, 2, 3, 4, 5]
    seq = polynomials.hermite_He_seq(ns, X)
    loop = [polynomials.hermite_He(n, X) for n in ns]
    for elem, exp in zip(seq, loop):
        assert np.allclose(elem, exp)


@pytest.mark.parametrize('n', [0, 1, 2, 3, 4, 5])
def test_hermite_h_matches_scipy(n):
    prysm_ = polynomials.hermite_H(n, X)
    scipy_ = sps_H(n)(X)
    assert np.allclose(prysm_, scipy_)


def test_hermite_h_seq_matches_loop():
    ns = [1, 2, 3, 4, 5]
    seq = polynomials.hermite_H_seq(ns, X)
    loop = [polynomials.hermite_H(n, X) for n in ns]
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
    seq = list(polynomials.cheby1_seq(ns, X))
    for elem, n in zip(seq, ns):
        exp = polynomials.cheby1(n, X)
        assert np.allclose(exp, elem)


def test_cheby2_seq_matches_loop():
    ns = [0, 1, 2, 3, 4, 5]
    seq = list(polynomials.cheby2_seq(ns, X))
    for elem, n in zip(seq, ns):
        exp = polynomials.cheby2(n, X)
        assert np.allclose(exp, elem)


@pytest.mark.parametrize('seq_fn', [
    polynomials.cheby1_seq, polynomials.cheby1_der_seq,
    polynomials.cheby2_seq, polynomials.cheby2_der_seq,
    polynomials.cheby3_seq, polynomials.cheby3_der_seq,
    polynomials.cheby4_seq, polynomials.cheby4_der_seq,
])
def test_cheby_seq_2d_input(seq_fn):
    """Cheby seq/der_seq broadcast correctly for 2-D x (docstring claim)."""
    ns = [0, 1, 2, 3, 4]
    x1 = np.linspace(-0.4, 0.4, 9)
    x2 = np.stack([x1] * 7)  # shape (7, 9)
    seq2 = seq_fn(ns, x2)
    assert seq2.shape == (len(ns), 7, 9)
    seq1 = seq_fn(ns, x1)
    np.testing.assert_allclose(seq2[:, 0, :], seq1)


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


@pytest.mark.parametrize('n', [2, 3, 4, 5])
def test_dickson2_satisfies_recurrence(n):
    alpha = 1.25

    lhs = polynomials.dickson2(n, alpha, X)
    rhs = X * polynomials.dickson2(n - 1, alpha, X) - alpha * polynomials.dickson2(n - 2, alpha, X)

    np.testing.assert_allclose(lhs, rhs)


def test_dickson1_seq_matches_loop():
    ns = [0, 1, 2, 3, 4, 5]
    seq = list(polynomials.dickson1_seq(ns, 1, X))
    for elem, n in zip(seq, ns):
        exp = polynomials.dickson1(n, 1, X)
        assert np.allclose(exp, elem)


def test_dickson2_seq_matches_loop():
    ns = [0, 1, 2, 3, 4, 5]
    seq = list(polynomials.dickson2_seq(ns, 1, X))
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


def test_jacobi_der_seq_same_as_loop():
    ns = [0, 1, 2, 3, 4, 5]
    seq = list(polynomials.jacobi_der_seq(ns, 0.5, 0.5, X))
    for elem, n in zip(seq, ns):
        exp = polynomials.jacobi_der(n, 0.5, 0.5, X)
        assert np.allclose(exp, elem)


@pytest.mark.parametrize('n', [1, 4])
def test_cheby1_der_matches_finite_diff(n):
    # need more points for accurate finite diff
    x = np.linspace(-1, 1, 128)
    Pn = polynomials.cheby1(n, x)
    Pnprime = polynomials.cheby1_der(n, x)
    dx = x[1] - x[0]
    Pnprime_numerical = np.gradient(Pn, dx)
    ratio = Pnprime / Pnprime_numerical
    assert abs(ratio-1).max() < 0.15  # 15% relative error


def test_cheby1_der_seq_same_as_loop():
    ns = [0, 1, 2, 3, 4, 5]
    seq = list(polynomials.cheby1_der_seq(ns, X))
    for elem, n in zip(seq, ns):
        exp = polynomials.cheby1_der(n, X)
        assert np.allclose(exp, elem)


@pytest.mark.parametrize('n', [1, 4])
def test_cheby2_der_matches_finite_diff(n):
    # need more points for accurate finite diff
    x = np.linspace(-1, 1, 128)
    Pn = polynomials.cheby2(n, x)
    Pnprime = polynomials.cheby2_der(n, x)
    dx = x[1] - x[0]
    Pnprime_numerical = np.gradient(Pn, dx)
    ratio = Pnprime / Pnprime_numerical
    assert abs(ratio-1).max() < 0.15  # 15% relative error


def test_cheby2_der_seq_same_as_loop():
    ns = [0, 1, 2, 3, 4, 5]
    seq = list(polynomials.cheby2_der_seq(ns, X))
    for elem, n in zip(seq, ns):
        exp = polynomials.cheby2_der(n, X)
        assert np.allclose(exp, elem)


@pytest.mark.parametrize('n', [1, 4])
def test_cheby3_der_matches_finite_diff(n):
    # need more points for accurate finite diff
    x = np.linspace(-1, 1, 128)
    Pn = polynomials.cheby3(n, x)
    Pnprime = polynomials.cheby3_der(n, x)
    dx = x[1] - x[0]
    Pnprime_numerical = np.gradient(Pn, dx)
    ratio = Pnprime / Pnprime_numerical
    assert abs(ratio-1).max() < 0.15  # 15% relative error


def test_cheby3_der_seq_same_as_loop():
    ns = [0, 1, 2, 3, 4, 5]
    seq = list(polynomials.cheby3_der_seq(ns, X))
    for elem, n in zip(seq, ns):
        exp = polynomials.cheby3_der(n, X)
        assert np.allclose(exp, elem)


@pytest.mark.parametrize('n', [1, 4])
def test_cheby4_der_matches_finite_diff(n):
    # need more points for accurate finite diff
    x = np.linspace(-1, 1, 128)
    Pn = polynomials.cheby4(n, x)
    Pnprime = polynomials.cheby4_der(n, x)
    dx = x[1] - x[0]
    Pnprime_numerical = np.gradient(Pn, dx)
    ratio = Pnprime / Pnprime_numerical
    assert abs(ratio-1).max() < 0.15  # 15% relative error


def test_cheby4_der_seq_same_as_loop():
    ns = [0, 1, 2, 3, 4, 5]
    seq = list(polynomials.cheby4_der_seq(ns, X))
    for elem, n in zip(seq, ns):
        exp = polynomials.cheby4_der(n, X)
        assert np.allclose(exp, elem)


@pytest.mark.parametrize('n', [1, 4])
def test_legendre_der_matches_finite_diff(n):
    # need more points for accurate finite diff
    x = np.linspace(-1, 1, 128)
    Pn = polynomials.legendre(n, x)
    Pnprime = polynomials.legendre_der(n, x)
    dx = x[1] - x[0]
    Pnprime_numerical = np.gradient(Pn, dx)
    ratio = Pnprime / Pnprime_numerical
    assert abs(ratio-1).max() < 0.35  # 35% relative error


def test_legendre_der_seq_same_as_loop():
    ns = [0, 1, 2, 3, 4, 5]
    seq = list(polynomials.legendre_der_seq(ns, X))
    for elem, n in zip(seq, ns):
        exp = polynomials.legendre_der(n, X)
        assert np.allclose(exp, elem)


@pytest.mark.parametrize('n', [1, 4])
def test_hermite_He_der_matches_finite_diff(n):
    # need more points for accurate finite diff
    x = np.linspace(-1, 1, 128)
    Pn = polynomials.hermite_He(n, x)
    Pnprime = polynomials.hermite_He_der(n, x)
    dx = x[1] - x[0]
    Pnprime_numerical = np.gradient(Pn, dx)
    diff = Pnprime - Pnprime_numerical
    assert abs(diff).max() < 0.35  # 10%


def test_hermite_He_der_seq_same_as_loop():
    ns = [0, 1, 2, 3, 4, 5]
    seq = list(polynomials.hermite_He_der_seq(ns, X))
    for elem, n in zip(seq, ns):
        exp = polynomials.hermite_He_der(n, X)
        assert np.allclose(exp, elem)


@pytest.mark.parametrize('n', [1, 4])
def test_hermite_H_der_matches_finite_diff(n):
    # need more points for accurate finite diff
    x = np.linspace(-1, 1, 128)
    Pn = polynomials.hermite_H(n, x)
    Pnprime = polynomials.hermite_H_der(n, x)
    dx = x[1] - x[0]
    Pnprime_numerical = np.gradient(Pn, dx)
    ratio = Pnprime / Pnprime_numerical
    assert abs(ratio-1).max() < 0.1  # 10%


def test_hermite_H_der_seq_same_as_loop():
    ns = [0, 1, 2, 3, 4, 5]
    seq = list(polynomials.hermite_H_der_seq(ns, X))
    for elem, n in zip(seq, ns):
        exp = polynomials.hermite_H_der(n, X)
        assert np.allclose(exp, elem)


def test_clenshaw_matches_standard_way():
    # pseudorandom numbers
    # this test fails sometimes when random coefs are used?
    cs = np.random.rand(5)
    basis = list(polynomials.jacobi_seq([0, 1, 2, 3, 4], .5, .5, X))
    exp = np.dot(cs, basis)
    clenshaw = polynomials.jacobi_sum_clenshaw(cs, .5, .5, X)
    assert np.allclose(exp, clenshaw, atol=1e-8)


@pytest.mark.parametrize('a, b', [
    [0, 0],
    [0, 1],
    [1, 0],
    [-.5, -.5],
    [-.5, .5],
    [.5, .5],
    [0, 4]
])
def test_clenshaw_matches_standard_way_der(a, b):
    # this test fails sometimes when random coefs are used?
    cs = np.random.rand(7)
    basis = list(polynomials.jacobi_der_seq([0, 1, 2, 3, 4, 5, 6], a, b, X))
    exp = np.dot(cs, basis)
    clenshaw = polynomials.jacobi_sum_clenshaw_der(cs, a, b, X)
    clenshaw = clenshaw[1][0]
    assert np.allclose(exp, clenshaw, atol=1e-8)


@pytest.mark.parametrize('a, b', [(0, 0), (0, 1), (-.5, .5), (.5, .5), (0, 4)])
def test_jacobi_fused_value_derivative_matches_existing_apis(a, b):
    ns = [0, 1, 2, 3, 4, 5, 6]
    vals, ders = polynomials.jacobi_seq_with_der(ns, a, b, X)
    np.testing.assert_allclose(vals, polynomials.jacobi_seq(ns, a, b, X))
    np.testing.assert_allclose(ders, polynomials.jacobi_der_seq(ns, a, b, X))

    val, der = polynomials.jacobi_with_der(4, a, b, X)
    np.testing.assert_allclose(val, polynomials.jacobi(4, a, b, X))
    np.testing.assert_allclose(der, polynomials.jacobi_der(4, a, b, X))


def test_clenshaw_handles_single_coefficient():
    # piston: only P_0 contributes; sum = s[0], all derivatives are 0
    res = polynomials.jacobi_sum_clenshaw([2.5], 0, 0, X)
    np.testing.assert_allclose(res, 2.5 * np.ones_like(X))
    res_der = polynomials.jacobi_sum_clenshaw_der([2.5], 0, 0, X, j=2)
    np.testing.assert_allclose(res_der[0, 0], 2.5 * np.ones_like(X))
    np.testing.assert_allclose(res_der[1, 0], np.zeros_like(X))
    np.testing.assert_allclose(res_der[2, 0], np.zeros_like(X))


@pytest.mark.parametrize('a, b', [(0, 0), (0, 1), (-.5, .5)])
def test_clenshaw_higher_derivatives_match_finite_diff(a, b):
    # j>=2 was previously broken: line 406 used `j` instead of `jj`,
    # corrupting derivative orders 1..j-1.  Validate against central diffs.
    rng = np.random.default_rng(0)
    cs = rng.normal(size=7)
    xs = np.linspace(-0.6, 0.6, 51)
    h = 1e-4

    def sum_at(x):
        return polynomials.jacobi_sum_clenshaw(cs, a, b, x)

    d1_num = (sum_at(xs + h) - sum_at(xs - h)) / (2 * h)
    d2_num = (sum_at(xs + h) - 2 * sum_at(xs) + sum_at(xs - h)) / (h * h)

    res = polynomials.jacobi_sum_clenshaw_der(cs, a, b, xs, j=2)
    np.testing.assert_allclose(res[1, 0], d1_num, atol=1e-7, rtol=1e-6)
    np.testing.assert_allclose(res[2, 0], d2_num, atol=1e-3, rtol=1e-4)


def test_clenshaw_der_zeros_above_polynomial_degree():
    # derivative order > polynomial degree must return zeros, not crash
    xs = np.linspace(-0.5, 0.5, 11)
    res = polynomials.jacobi_sum_clenshaw_der([1.0, 2.0], 0, 0, xs, j=3)
    # P_1(x) = x for alpha=beta=0, sum = 1 + 2x, d/dx = 2, d2/dx2 = 0, d3/dx3 = 0
    np.testing.assert_allclose(res[0, 0], 1.0 + 2.0 * xs)
    np.testing.assert_allclose(res[1, 0], 2.0 * np.ones_like(xs))
    np.testing.assert_allclose(res[2, 0], np.zeros_like(xs))
    np.testing.assert_allclose(res[3, 0], np.zeros_like(xs))


def test_cheby3_seq_matches_loop():
    ns = [1, 2, 3, 4, 5]
    seq = polynomials.cheby3_seq(ns, X)
    loop = [polynomials.cheby3(n, X) for n in ns]
    for elem, exp in zip(seq, loop):
        assert np.allclose(elem, exp)


def test_cheby4_seq_matches_loop():
    ns = [1, 2, 3, 4, 5]
    seq = polynomials.cheby4_seq(ns, X)
    loop = [polynomials.cheby4(n, X) for n in ns]
    for elem, exp in zip(seq, loop):
        assert np.allclose(elem, exp)


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


def test_qbfs_zzprime_grads():
    # decent number of points, so that finite diff isn't awful
    r = np.linspace(-1, 1, 512)
    coefs = np.random.rand(5)
    z, zprime = polynomials.qpoly.compute_z_zprime_Qbfs(coefs, r, r*r)
    dx = r[1] - r[0]
    fd = np.gradient(z, dx)
    assert np.allclose(zprime[1:-1], fd[1:-1], atol=2e-1)


def test_clenshaw_qbfs_returns_alphas_like_siblings():
    # clenshaw_qbfs returns the alphas table, matching clenshaw_qbfs_der's
    # j=0 plane (and clenshaw_q2d / clenshaw_q2d_der); the sag assembly
    # u^2(1-u^2) 2(a0+a1) lives in compute_z_Qbfs
    u = np.linspace(0, 1, 32)
    usq = u * u
    coefs = [0.1, -0.2, 0.05]
    alphas = polynomials.qpoly.clenshaw_qbfs(coefs, usq)
    alphas_der = polynomials.qpoly.clenshaw_qbfs_der(coefs, usq, j=1)
    np.testing.assert_allclose(alphas, alphas_der[0])

    z = polynomials.qpoly.compute_z_Qbfs(coefs, u, usq)
    z2, _ = polynomials.qpoly.compute_z_zprime_Qbfs(coefs, u, usq)
    np.testing.assert_allclose(z, z2)

    # len-1 and empty coefficient edge cases
    z1 = polynomials.qpoly.compute_z_Qbfs([0.1], u, usq)
    z1_ref, _ = polynomials.qpoly.compute_z_zprime_Qbfs([0.1], u, usq)
    np.testing.assert_allclose(z1, z1_ref)
    z0 = polynomials.qpoly.compute_z_Qbfs([], u, usq)
    np.testing.assert_allclose(z0, 0)


def test_qcon_zzprime_grads():
    # decent number of points, so that finite diff isn't awful
    r = np.linspace(-1, 1, 512)
    coefs = np.random.rand(5)
    z, zprime = polynomials.qpoly.compute_z_zprime_Qcon(coefs, r, r*r)
    dx = r[1] - r[0]
    fd = np.gradient(z, dx)
    # tends to be about 6e-4, permit 10x higher so sporadic failures don't happen
    assert np.allclose(zprime[1:-1], fd[1:-1], atol=5e-1)


def test_qpoly_summed_paths_ignore_trailing_zeros():
    r = np.linspace(0, 1, 64)
    t = np.linspace(0, 2*np.pi, 64)
    rr, tt = np.meshgrid(r, t)

    coefs = [0.1, -0.2, 0.05]
    coefs_padded = [0.1, -0.2, 0.05, 0, 0, 0]
    z, zp = polynomials.qpoly.compute_z_zprime_Qbfs(coefs, r, r*r)
    z_pad, zp_pad = polynomials.qpoly.compute_z_zprime_Qbfs(coefs_padded, r, r*r)
    np.testing.assert_allclose(z_pad, z)
    np.testing.assert_allclose(zp_pad, zp)

    z, zp = polynomials.qpoly.compute_z_zprime_Qcon(coefs, r, r*r)
    z_pad, zp_pad = polynomials.qpoly.compute_z_zprime_Qcon(coefs_padded, r, r*r)
    np.testing.assert_allclose(z_pad, z)
    np.testing.assert_allclose(zp_pad, zp)

    cm0 = [0.1, -0.05]
    ams = [[0.2], [], [0.15]]
    bms = [[], [0.3], []]
    cm0_padded = [0.1, -0.05, 0, 0]
    ams_padded = [[0.2, 0, 0], [0, 0], [0.15, 0, 0], [0, 0, 0]]
    bms_padded = [[0, 0], [0.3, 0, 0], [0], [0, 0, 0]]
    out = polynomials.qpoly.compute_z_zprime_Q2d(cm0, ams, bms, rr, tt)
    out_padded = polynomials.qpoly.compute_z_zprime_Q2d(cm0_padded, ams_padded, bms_padded, rr, tt)
    for elem, elem_padded in zip(out, out_padded):
        np.testing.assert_allclose(elem_padded, elem)

    zero = polynomials.qpoly.compute_z_zprime_Q2d([0, 0], [[0, 0]], [[0, 0]], rr, tt)
    for elem in zero:
        np.testing.assert_allclose(elem, np.zeros_like(rr))


def test_q2d_coefficient_restructure_skips_zero_terms():
    nms = [(0, 0), (5, 0), (1, 1), (8, 1), (2, -2), (7, -2)]
    coefs = [0.25, 0, -0.5, 0, 0.75, 0]
    cm0, ams, bms = polynomials.qpoly.Q2d_nm_c_to_a_b(nms, coefs)
    assert cm0 == [0.25]
    assert ams == [[0, -0.5], []]
    assert bms == [[], [0, 0, 0.75]]

    cm0, ams, bms = polynomials.qpoly.Q2d_nm_c_to_a_b(nms, [0, 0, 0, 0, 0, 0])
    assert cm0 == []
    assert ams == []
    assert bms == []


@pytest.mark.parametrize(['n', 'alpha'], [
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1],
    [2, 1],
    [3, 1],
    [4, 1],
    [0, 2],
    [1, 2],
    [2, 2],
    [3, 2],
    [4, 2]])
def test_laguerre_matches_scipy(n, alpha):
    prysm_lag = polynomials.laguerre(n, alpha, XLEFT)
    scipy_lag = sps_laguerre(n, alpha)(XLEFT)
    assert np.allclose(prysm_lag, scipy_lag)


def test_laguerre_seq_matches_loop():
    ns = [0, 1, 2, 3, 4, 5]
    seq = polynomials.laguerre_seq(ns, 1.0, XLEFT)
    loop = [polynomials.laguerre(n, 1.0, XLEFT) for n in ns]
    for elem, exp in zip(seq, loop):
        assert np.allclose(elem, exp)


@pytest.mark.parametrize('n', [1, 2, 3, 4, 5])
@pytest.mark.parametrize('alpha', [0.0, 1.0, 2.0])
def test_laguerre_der_matches_scipy(n, alpha):
    # d/dx L_n^alpha = -L_{n-1}^{alpha+1}; compare to scipy's exact derivative
    prysm_ = polynomials.laguerre_der(n, alpha, XLEFT)
    scipy_ = sps_laguerre(n, alpha).deriv()(XLEFT)
    assert np.allclose(prysm_, scipy_)


def test_laguerre_der_n_zero():
    # the constant L_0 differentiates to identically zero
    assert np.allclose(polynomials.laguerre_der(0, 1.0, XLEFT), 0)


def test_laguerre_der_seq_same_as_loop():
    ns = [0, 1, 2, 3, 5]
    seq = list(polynomials.laguerre_der_seq(ns, 1.0, XLEFT))
    for elem, n in zip(seq, ns):
        exp = polynomials.laguerre_der(n, 1.0, XLEFT)
        assert np.allclose(exp, elem)


# derivative additions: dickson / xy / qpoly

@pytest.mark.parametrize('n', [1, 2, 3, 4, 5])
@pytest.mark.parametrize('alpha', [-1.0, 0.0, 1.0])
def test_dickson1_der_matches_finite_diff(n, alpha):
    x = np.linspace(-0.9, 0.9, 256)
    h = 1e-5
    der = polynomials.dickson1_der(n, alpha, x)
    fd = (polynomials.dickson1(n, alpha, x + h)
          - polynomials.dickson1(n, alpha, x - h)) / (2 * h)
    assert np.allclose(der, fd, atol=1e-6, rtol=1e-5)


def test_dickson1_der_n_zero_and_one():
    x = np.linspace(-1, 1, 32)
    np.testing.assert_array_equal(polynomials.dickson1_der(0, 1.0, x), 0)
    np.testing.assert_array_equal(polynomials.dickson1_der(1, 1.0, x), 1)


def test_dickson1_der_seq_same_as_loop():
    x = np.linspace(-0.9, 0.9, 64)
    ns = [0, 1, 2, 3, 5]
    seq = polynomials.dickson1_der_seq(ns, 1.0, x)
    for i, n in enumerate(ns):
        assert np.allclose(seq[i], polynomials.dickson1_der(n, 1.0, x))


@pytest.mark.parametrize('n', [1, 2, 3, 4, 5])
@pytest.mark.parametrize('alpha', [-1.0, 0.0, 1.0])
def test_dickson2_der_matches_finite_diff(n, alpha):
    x = np.linspace(-0.9, 0.9, 256)
    h = 1e-5
    der = polynomials.dickson2_der(n, alpha, x)
    fd = (polynomials.dickson2(n, alpha, x + h)
          - polynomials.dickson2(n, alpha, x - h)) / (2 * h)
    assert np.allclose(der, fd, atol=1e-6, rtol=1e-5)


def test_dickson2_der_seq_same_as_loop():
    x = np.linspace(-0.9, 0.9, 64)
    ns = [0, 1, 2, 3, 5]
    seq = polynomials.dickson2_der_seq(ns, 1.0, x)
    for i, n in enumerate(ns):
        assert np.allclose(seq[i], polynomials.dickson2_der(n, 1.0, x))


@pytest.mark.parametrize(('m', 'n'), [(0, 0), (1, 0), (0, 1), (1, 1),
                                       (2, 3), (4, 2)])
def test_xy_der_x_matches_truth(m, n):
    x = np.linspace(-1, 1, 32)
    y = np.linspace(-1, 1, 32).reshape(-1, 1)
    der = polynomials.xy_der_x(m, n, x, y, cartesian_grid=False)
    if m == 0:
        truth = np.zeros_like(x * y)
    else:
        truth = m * x**(m-1) * y**n
    assert np.allclose(der, np.broadcast_to(truth, der.shape))


@pytest.mark.parametrize(('m', 'n'), [(0, 0), (1, 0), (0, 1), (1, 1),
                                       (2, 3), (4, 2)])
def test_xy_der_y_matches_truth(m, n):
    x = np.linspace(-1, 1, 32)
    y = np.linspace(-1, 1, 32).reshape(-1, 1)
    der = polynomials.xy_der_y(m, n, x, y, cartesian_grid=False)
    if n == 0:
        truth = np.zeros_like(x * y)
    else:
        truth = n * x**m * y**(n-1)
    assert np.allclose(der, np.broadcast_to(truth, der.shape))


@pytest.mark.parametrize(('m', 'n'), [(0, 0), (1, 0), (0, 1), (1, 1),
                                       (2, 3), (4, 2)])
def test_xy_der_xy_matches_truth(m, n):
    x = np.linspace(-1, 1, 32)
    y = np.linspace(-1, 1, 32).reshape(-1, 1)
    der = polynomials.xy_der_xy(m, n, x, y, cartesian_grid=False)
    if m == 0 or n == 0:
        truth = np.zeros_like(x * y)
    else:
        truth = (m * n) * x**(m-1) * y**(n-1)
    assert np.allclose(der, np.broadcast_to(truth, der.shape))


def test_xy_der_seq_matches_loop():
    x = np.linspace(-1, 1, 16)
    y = np.linspace(-1, 1, 16).reshape(-1, 1)
    mns = [(0, 0), (1, 0), (0, 1), (2, 1), (3, 4)]
    for seq_fn, single_fn in [
        (polynomials.xy_der_x_seq, polynomials.xy_der_x),
        (polynomials.xy_der_y_seq, polynomials.xy_der_y),
        (polynomials.xy_der_xy_seq, polynomials.xy_der_xy),
    ]:
        seq = seq_fn(mns, x, y, cartesian_grid=False)
        for elem, (m, n) in zip(seq, mns):
            ref = single_fn(m, n, x, y, cartesian_grid=False)
            assert np.allclose(elem, ref)


def test_xy_seq_piston_returns_one():
    # latent bug fix: prior xy_seq used dickson1_seq(alpha=0) which gives D_0=2,
    # so xy_seq([(0,0)], ...) returned 4. The single-mode xy(0, 0, ...) has
    # always returned 1; the seq variant now agrees.
    x = np.linspace(-1, 1, 8)
    y = np.linspace(-1, 1, 8).reshape(-1, 1)
    (elem,) = polynomials.xy_seq([(0, 0)], x, y, cartesian_grid=False)
    assert np.allclose(elem, np.ones_like(x * y))


@pytest.mark.parametrize('n', [0, 1, 2, 3, 4, 5])
def test_Qbfs_der_matches_finite_diff(n):
    x = np.linspace(0.05, 0.95, 256)
    h = 1e-5
    der = polynomials.Qbfs_der(n, x)
    fd = (polynomials.Qbfs(n, x + h) - polynomials.Qbfs(n, x - h)) / (2 * h)
    assert np.allclose(der, fd, atol=1e-6, rtol=1e-4)


def test_Qbfs_der_seq_same_as_loop():
    x = np.linspace(0.05, 0.95, 64)
    ns = [0, 1, 2, 3, 5]
    seq = polynomials.Qbfs_der_seq(ns, x)
    for i, n in enumerate(ns):
        assert np.allclose(seq[i], polynomials.Qbfs_der(n, x))


@pytest.mark.parametrize('n', [0, 1, 2, 3, 4, 5])
def test_Qcon_der_matches_finite_diff(n):
    x = np.linspace(0.05, 0.95, 256)
    h = 1e-5
    der = polynomials.Qcon_der(n, x)
    fd = (polynomials.Qcon(n, x + h) - polynomials.Qcon(n, x - h)) / (2 * h)
    # Qcon involves jacobi(0, 4), high-order derivatives swing in magnitude,
    # so accept a slightly looser tolerance than Qbfs.
    assert np.allclose(der, fd, atol=1e-5, rtol=1e-3)


def test_Qcon_der_seq_same_as_loop():
    x = np.linspace(0.05, 0.95, 64)
    ns = [0, 1, 2, 3, 5]
    seq = polynomials.Qcon_der_seq(ns, x)
    for i, n in enumerate(ns):
        assert np.allclose(seq[i], polynomials.Qcon_der(n, x))


# Q2d polar and Cartesian derivatives

_Q2D_DER_NMS = [(0, 0), (1, 0), (2, 1), (3, -2), (2, 3), (1, -1), (4, 1)]


@pytest.mark.parametrize(('n', 'm'), _Q2D_DER_NMS)
def test_Q2d_der_polar_matches_finite_diff(n, m):
    r1 = np.linspace(0.05, 0.95, 64)
    t1 = np.linspace(0.1, 2 * np.pi - 0.1, 64)
    R, T = np.meshgrid(r1, t1)
    h = 1e-5
    dr_an, dt_an = polynomials.Q2d_der(n, m, R, T)
    dr_fd = (polynomials.Q2d(n, m, R + h, T)
             - polynomials.Q2d(n, m, R - h, T)) / (2 * h)
    dt_fd = (polynomials.Q2d(n, m, R, T + h)
             - polynomials.Q2d(n, m, R, T - h)) / (2 * h)
    assert np.allclose(dr_an, dr_fd, atol=1e-5, rtol=1e-3)
    assert np.allclose(dt_an, dt_fd, atol=1e-5, rtol=1e-3)


def test_Q2d_der_seq_same_as_loop():
    r1 = np.linspace(0.05, 0.95, 32)
    t1 = np.linspace(0.1, 2 * np.pi - 0.1, 32)
    R, T = np.meshgrid(r1, t1)
    dr_seq, dt_seq = polynomials.Q2d_der_seq(_Q2D_DER_NMS, R, T)
    for i, (n, m) in enumerate(_Q2D_DER_NMS):
        dr_s, dt_s = polynomials.Q2d_der(n, m, R, T)
        assert np.allclose(dr_seq[i], dr_s)
        assert np.allclose(dt_seq[i], dt_s)


@pytest.mark.parametrize(('n', 'm'), _Q2D_DER_NMS)
def test_Q2d_der_xy_matches_finite_diff(n, m):
    # use a disk-interior Cartesian grid that stays away from r=1
    x1 = np.linspace(-0.6, 0.6, 48)
    y1 = np.linspace(-0.6, 0.6, 48).reshape(-1, 1)
    X, Y = np.meshgrid(x1.ravel(), y1.ravel())
    h = 1e-5
    dx_an, dy_an = polynomials.Q2d_der_xy(n, m, X, Y)

    def Z_at(xa, ya):
        ra = np.sqrt(xa * xa + ya * ya)
        ta = np.arctan2(ya, xa)
        return polynomials.Q2d(n, m, ra, ta)

    dx_fd = (Z_at(X + h, Y) - Z_at(X - h, Y)) / (2 * h)
    dy_fd = (Z_at(X, Y + h) - Z_at(X, Y - h)) / (2 * h)
    assert np.allclose(dx_an, dx_fd, atol=1e-5, rtol=1e-3)
    assert np.allclose(dy_an, dy_fd, atol=1e-5, rtol=1e-3)


def test_Q2d_der_xy_seq_same_as_loop():
    x1 = np.linspace(-0.6, 0.6, 24)
    y1 = np.linspace(-0.6, 0.6, 24).reshape(-1, 1)
    X, Y = np.meshgrid(x1.ravel(), y1.ravel())
    dx_seq, dy_seq = polynomials.Q2d_der_xy_seq(_Q2D_DER_NMS, X, Y)
    for i, (n, m) in enumerate(_Q2D_DER_NMS):
        dx_s, dy_s = polynomials.Q2d_der_xy(n, m, X, Y)
        assert np.allclose(dx_seq[i], dx_s)
        assert np.allclose(dy_seq[i], dy_s)


def test_Q2d_der_xy_finite_at_origin():
    # the harmonic decomposition keeps the Cartesian derivative finite at r=0
    x = np.array([[0.0, 1e-12], [1e-12, 0.0]])
    y = np.array([[0.0, 0.0], [1e-12, 1e-12]])
    for n, m in [(0, 0), (2, 1), (3, -2), (4, 3)]:
        dx, dy = polynomials.Q2d_der_xy(n, m, x, y)
        assert np.isfinite(dx).all()
        assert np.isfinite(dy).all()
