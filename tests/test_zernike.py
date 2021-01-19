"""Tests for the Zernike submodule."""
import pytest

import numpy as np

from prysm.coordinates import cart_to_polar
from prysm import zernike

import matplotlib
matplotlib.use('Agg')

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


@pytest.fixture
def fit_data():
    p = zernike.FringeZernike(Z9=1, samples=SAMPLES)
    return p.phase, p.coefs


@pytest.fixture
def sample():
    return zernike.NollZernike(np.random.rand(9), samples=64)


def test_fit_agrees_with_truth(fit_data):
    data, real_coefs = fit_data
    coefs = zernike.zernikefit(data, map_='Fringe')
    assert coefs[8] == pytest.approx(real_coefs[9])  # compare 8 (0-based index 9) to 9 (dict key)


def test_fit_does_not_throw_on_normalize(fit_data):
    data, real_coefs = fit_data
    coefs = zernike.zernikefit(data, norm=True, map_='Noll')
    assert coefs[10] != 0


def test_names_functions(sample):
    assert any(sample.names)


def test_magnitudes_functions(sample):
    assert any(sample.magnitudes)


@pytest.mark.parametrize('orientation', ['h', 'v'])
def test_barplot_functions(sample, orientation):
    fig, ax = sample.barplot(orientation=orientation)
    assert fig
    assert ax


@pytest.mark.parametrize('orientation, sort', [['h', True], ['v', False]])
def test_barplot_magnitudes_functions(sample, orientation, sort):
    fig, ax = sample.barplot_magnitudes(orientation=orientation, sort=sort)
    assert fig
    assert ax


@pytest.mark.parametrize('orientation', ['h', 'v'])
def test_barplot_topn_functions(sample, orientation):
    fig, ax = sample.barplot_topn(orientation=orientation)
    assert fig
    assert ax


@pytest.mark.parametrize('n', [2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
def test_zero_separation_gives_correct_array_sizes(n):
    sep = zernike.zero_separation(n)
    assert int(1/sep) == int(n**2)


@pytest.mark.parametrize('fringe_idx', range(1, 100))
def test_nm_to_fringe_round_trips(fringe_idx):
    n, m = zernike.fringe_to_n_m(fringe_idx)
    j = zernike.n_m_to_fringe(n, m)
    assert j == fringe_idx


def test_ansi_2_term_can_construct():
    ary = zernike.zernike_nm(3, 1, rho, phi)
    assert ary.any()
