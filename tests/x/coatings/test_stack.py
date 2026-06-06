"""Tests for the coating stack field / partial-product engine (Phase 1)."""
import pytest

from prysm import thinfilm
from prysm.mathops import np
from prysm.x.coatings import stack as st
from prysm.x.coatings import (
    Stack,
    stack_rt,
    internal_fields,
    field_at_depth,
    forward_products,
    backward_products,
    stack_characteristic_matrices,
    RTA,
)

wvl = 0.587725
n_C7980 = 1.458461
n_MgF2 = 1.3698
n_CeF3 = 1.6290 + 1j * 0.0034836
n_ZrO2 = 2.1588

MONO = ([n_MgF2], [0.150])
MULTI = ([n_MgF2, n_ZrO2, n_CeF3], [wvl / 4, wvl / 2, wvl / 4])


# ---------------------------------------------------------------- r/t crosscheck

@pytest.mark.parametrize('pol', ['s', 'p'])
@pytest.mark.parametrize('aoi', [0, 15, 45])
@pytest.mark.parametrize('layers', [MONO, MULTI])
def test_r_matches_multilayer_stack_rt(pol, aoi, layers):
    indices, thicknesses = layers
    s = Stack(indices, thicknesses, n_C7980)
    r, _ = stack_rt(s, wvl, np.radians(aoi), pol)
    r_ref, _ = thinfilm.multilayer_stack_rt(indices, thicknesses, wvl, pol, n_C7980, aoi=aoi)
    assert np.allclose(r, r_ref)


@pytest.mark.parametrize('aoi', [0, 15, 45])
@pytest.mark.parametrize('layers', [MONO, MULTI])
def test_s_pol_t_matches_multilayer_stack_rt(aoi, layers):
    # for s polarization the substrate-column normalization agrees with thinfilm,
    # so t is identical (the p-pol t differs by a cos ratio -- validated via energy).
    indices, thicknesses = layers
    s = Stack(indices, thicknesses, n_C7980)
    _, t = stack_rt(s, wvl, np.radians(aoi), 's')
    _, t_ref = thinfilm.multilayer_stack_rt(indices, thicknesses, wvl, 's', n_C7980, aoi=aoi)
    assert np.allclose(t, t_ref)


@pytest.mark.parametrize('pol', ['s', 'p'])
def test_full_product_consistency_and_unimodular(pol):
    indices, thicknesses = MULTI
    s = Stack(indices, thicknesses, n_C7980)
    mats = stack_characteristic_matrices(s, wvl, np.radians(20), pol)
    L = forward_products(mats)
    R = backward_products(mats)
    # the full assembly product appears at both ends.
    assert np.allclose(L[-1], R[0])
    # each characteristic matrix is unimodular, so the product is too.
    det = L[-1][..., 0, 0] * L[-1][..., 1, 1] - L[-1][..., 0, 1] * L[-1][..., 1, 0]
    assert np.allclose(det, 1.0)


# ---------------------------------------------------------------- energy budget

@pytest.mark.parametrize('pol', ['s', 'p'])
@pytest.mark.parametrize('aoi', [0, 30])
def test_energy_conservation_lossless(pol, aoi):
    indices = [n_MgF2, n_ZrO2, n_C7980]
    thicknesses = [wvl / 4, wvl / 2, wvl / 4]
    s = Stack(indices, thicknesses, n_C7980)
    R, T, A = RTA(s, wvl, np.radians(aoi), pol)
    assert R + T == pytest.approx(1.0, abs=1e-12)
    assert np.sum(A) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize('pol', ['s', 'p'])
@pytest.mark.parametrize('aoi', [0, 30])
def test_energy_conservation_lossy(pol, aoi):
    # a clearly absorbing layer in the middle of a dielectric stack.
    indices = [n_MgF2, 1.5 + 0.5j, n_ZrO2]
    thicknesses = [wvl / 4, 0.080, wvl / 4]
    s = Stack(indices, thicknesses, n_C7980)
    R, T, A = RTA(s, wvl, np.radians(aoi), pol)
    assert R + np.sum(A) + T == pytest.approx(1.0, abs=1e-12)
    # passive media absorb non-negative power; the lossy layer dominates.
    assert np.all(A >= -1e-12)
    assert np.sum(A) > 0.05
    assert A[1] == pytest.approx(np.sum(A), abs=1e-3)


@pytest.mark.parametrize('pol', ['s', 'p'])
@pytest.mark.parametrize('aoi', [0, 40])
def test_bare_substrate_is_fresnel(pol, aoi):
    s = Stack([], [], n_C7980)
    theta0 = np.radians(aoi)
    r, _ = stack_rt(s, wvl, theta0, pol)
    theta1 = thinfilm.snell_aor(1.0, n_C7980, aoi, deg=True)
    fresnel = thinfilm.fresnel_rs if pol == 's' else thinfilm.fresnel_rp
    assert np.allclose(r, fresnel(1.0, n_C7980, theta0, theta1))
    R, T, A = RTA(s, wvl, theta0, pol)
    assert A.shape == (0,)
    assert R + T == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------- internal field

@pytest.mark.parametrize('pol', ['s', 'p'])
def test_field_at_depth_matches_boundaries(pol):
    indices, thicknesses = MULTI
    s = Stack(indices, thicknesses, n_C7980)
    theta0 = np.radians(25)
    E, H = internal_fields(s, wvl, theta0, pol)
    Z = np.concatenate([np.zeros(1), np.cumsum(np.asarray(thicknesses))])
    Ez, Hz = field_at_depth(s, Z, wvl, theta0, pol)
    assert np.allclose(Ez, E)
    assert np.allclose(Hz, H)


def test_qwot_field_swap():
    # a quarter-wave layer at normal incidence swaps and scales (E, H) by eta:
    # the matrix is [[0, -i/eta],[-i eta, 0]], so |E_front| eta == |H_sub| and
    # |H_front| == eta |E_sub|.
    n1 = 1.38
    s = Stack([n1], [wvl / (4 * n1)], 1.52)
    E, H = internal_fields(s, wvl, 0.0, 's')  # normal incidence: eta = n1
    assert np.abs(E[0]) * n1 == pytest.approx(np.abs(H[1]), rel=1e-9)
    assert np.abs(H[0]) == pytest.approx(n1 * np.abs(E[1]), rel=1e-9)


@pytest.mark.parametrize('aoi', [0, 35])
@pytest.mark.parametrize('pol', ['s', 'p'])
def test_standing_wave_node_spacing(aoi, pol):
    # inside a thick homogeneous layer the standing-wave |E|^2 period is
    # lambda / (2 n cos(theta_layer)).
    n1 = 2.0
    d = 3.0
    s = Stack([n1], [d], 1.5)
    theta0 = np.radians(aoi)
    z = np.linspace(0, d, 12001)
    Ez, _ = field_at_depth(s, z, wvl, theta0, pol)
    intensity = np.abs(Ez) ** 2
    peaks = np.where((intensity[1:-1] > intensity[:-2]) & (intensity[1:-1] > intensity[2:]))[0] + 1
    spacing = np.diff(z[peaks])
    cost1 = np.cos(thinfilm.snell_aor(1.0, n1, aoi, deg=True))
    expected = wvl / (2 * n1 * cost1)
    assert np.allclose(spacing, expected, atol=2 * (z[1] - z[0]))


# ---------------------------------------------------------------- vectorization

@pytest.mark.parametrize('pol', ['s', 'p'])
def test_vectorized_over_wavelength_matches_loop(pol):
    indices, thicknesses = MULTI
    s = Stack(indices, thicknesses, n_C7980)
    wvls = np.array([0.45, 0.55, 0.65])
    theta0 = np.radians(20)
    R, T, A = RTA(s, wvls, theta0, pol)
    for i, w in enumerate(wvls):
        Ri, Ti, Ai = RTA(s, float(w), theta0, pol)
        assert Ri == pytest.approx(R[i])
        assert Ti == pytest.approx(T[i])
        assert np.allclose(Ai, A[:, i])


@pytest.mark.parametrize('pol', ['s', 'p'])
def test_vectorized_over_angle_matches_loop(pol):
    indices, thicknesses = MULTI
    s = Stack(indices, thicknesses, n_C7980)
    thetas = np.radians(np.array([0.0, 15.0, 30.0, 45.0]))
    R, T, A = RTA(s, wvl, thetas, pol)
    E, H = internal_fields(s, wvl, thetas, pol)
    for i, th in enumerate(thetas):
        Ri, Ti, Ai = RTA(s, wvl, float(th), pol)
        Ei, Hi = internal_fields(s, wvl, float(th), pol)
        assert Ri == pytest.approx(R[i])
        assert Ti == pytest.approx(T[i])
        assert np.allclose(Ai, A[:, i])
        assert np.allclose(Ei, E[:, i])
        assert np.allclose(Hi, H[:, i])


def test_stack_length_and_validation():
    s = Stack(MULTI[0], MULTI[1], n_C7980)
    assert len(s) == 3
    with pytest.raises(ValueError, match='same number of layers'):
        Stack([n_MgF2, n_ZrO2], [0.1], n_C7980)
