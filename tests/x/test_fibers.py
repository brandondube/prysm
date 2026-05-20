"""Tests for prysm.x.fibers step-index mode finding."""
import numpy as np
import pytest

from prysm.x.fibers import (
    _BESSELJ_ZERO_CACHE,
    _besselj_positive_zeros,
    _ghatak_eq_8_40,
    find_all_modes,
)


def _expected_mode_count(ell, V, zero_table):
    """LP_{ell,m} count from cutoff theory.

    LP_{0,1} has no cutoff; LP_{0,m>=2} cuts off at V = j_{1, m-1}.
    LP_{l>=1, m} cuts off at V = j_{l-1, m}.
    """
    if ell == 0:
        return 1 + int((zero_table[1] < V).sum())
    return int((zero_table[ell - 1] < V).sum())


@pytest.fixture(scope='module')
def jn_zeros():
    # Pre-tabulated zeros of J_n for n=0..60, k=1..60 (only needs scipy at
    # test-collection time; the prysm code under test does not depend on it).
    sps = pytest.importorskip('scipy.special')
    return {n: sps.jn_zeros(n, 60) for n in range(61)}


@pytest.mark.parametrize('V', [3.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0])
def test_find_all_modes_count_matches_cutoff_theory(V, jn_zeros):
    """Every LP_{l,m} predicted by cutoff theory must appear in the dict.

    Regression for the pre-refactor bug where modes with propagation
    constants close to a Bessel pole were silently dropped by the
    grid-scan + pole-discarding heuristic.
    """
    modes = find_all_modes(V)
    for ell, bs in modes.items():
        if ell < 0:
            continue
        expected = _expected_mode_count(ell, V, jn_zeros)
        assert len(bs) == expected, (
            f'V={V} ell={ell}: got {len(bs)} modes, expected {expected}'
        )


@pytest.mark.parametrize('V', [5.0, 10.0, 20.0, 40.0])
def test_find_all_modes_roots_satisfy_equation(V):
    """Every reported b must zero the dispersion equation.

    The tolerance is intentionally looser at high V because the dispersion
    residual is steep for weakly-confined modes close to cutoff.  This test
    guards root validity, not the last bit of the Bessel evaluation.
    """
    tol = 1e-4 if V <= 25 else 1e-2
    modes = find_all_modes(V)
    for ell, bs in modes.items():
        if ell < 0:
            continue
        for b in bs:
            r = _ghatak_eq_8_40(b, V, ell)
            assert abs(r) < tol, f'V={V} ell={ell} b={b}: residual {r}'


@pytest.mark.parametrize('V', [3.0, 8.0, 15.0, 25.0])
def test_negative_ell_mirrors_positive(V):
    """Sign-degenerate modes (l, -l) must have identical b lists."""
    modes = find_all_modes(V)
    for ell, bs in modes.items():
        if ell <= 0:
            continue
        assert -ell in modes
        np.testing.assert_array_equal(np.asarray(bs), np.asarray(modes[-ell]))


def test_v_below_first_cutoff_returns_single_mode():
    """V < 2.405 (first zero of J_0): only LP_{0,1} propagates."""
    modes = find_all_modes(2.0)
    assert set(modes.keys()) == {0}
    assert len(modes[0]) == 1


@pytest.mark.parametrize('V', [0.5, 1.0, 2.0, 2.3])
def test_single_mode_fast_path_roots_satisfy_equation(V):
    """The LP_01-only region should return the single physical root."""
    modes = find_all_modes(V)
    assert set(modes.keys()) == {0}
    assert len(modes[0]) == 1
    assert abs(_ghatak_eq_8_40(modes[0][0], V, 0)) < 1e-6


@pytest.mark.parametrize('V', [2.5, 3.0, 3.7])
def test_low_v_two_family_fast_path_roots_satisfy_equation(V):
    """Between the first J_0 and J_1 zeros only LP_01 and LP_11 exist."""
    modes = find_all_modes(V)
    assert set(modes.keys()) == {0, 1, -1}
    assert len(modes[0]) == 1
    assert len(modes[1]) == 1
    np.testing.assert_array_equal(np.asarray(modes[1]), np.asarray(modes[-1]))
    assert abs(_ghatak_eq_8_40(modes[0][0], V, 0)) < 1e-6
    assert abs(_ghatak_eq_8_40(modes[1][0], V, 1)) < 1e-6


@pytest.mark.parametrize('V', [0.5, 2.5, 5.0, 10.0, 20.0, 40.0])
def test_count_only_matches_mode_lengths(V):
    """count_only returns cutoff-theory counts without root solving."""
    modes = find_all_modes(V)
    counts = find_all_modes(V, count_only=True)
    assert counts == {ell: len(bs) for ell, bs in modes.items()}


def test_besselj_zeros_first_zero_large_order(jn_zeros):
    """First zero of high-order J_l is found (was missed by McMahon-only)."""
    for l in (10, 20, 27, 35):
        zeros = _besselj_positive_zeros(l, 50.0)
        scipy_zeros = jn_zeros[l]
        scipy_zeros = scipy_zeros[scipy_zeros < 50.0]
        assert zeros.shape == scipy_zeros.shape, (
            f'l={l}: got {len(zeros)} zeros, expected {len(scipy_zeros)}'
        )
        np.testing.assert_allclose(zeros, scipy_zeros, atol=1e-5, rtol=1e-5)


def test_besselj_zero_cache_extends_after_empty_result(jn_zeros):
    """A below-cutoff cache entry must not mask later larger requests."""
    _BESSELJ_ZERO_CACHE.clear()
    assert len(_besselj_positive_zeros(20, 10.0)) == 0

    zeros = _besselj_positive_zeros(20, 50.0)
    scipy_zeros = jn_zeros[20]
    scipy_zeros = scipy_zeros[scipy_zeros < 50.0]
    assert zeros.shape == scipy_zeros.shape
    np.testing.assert_allclose(zeros, scipy_zeros, atol=1e-5, rtol=1e-5)
