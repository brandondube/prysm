"""Tests for the Zernike submodule."""
import pytest

import numpy as np

from prysm.coordinates import cart_to_polar
from prysm import zernike

import matplotlib
matplotlib.use('TkAgg')

SAMPLES = 32

X, Y = np.linspace(-1, 1, SAMPLES), np.linspace(-1, 1, SAMPLES)

all_zernikes = [
    zernike.piston,
    zernike.tilt,
    zernike.tip,
    zernike.defocus,
    zernike.primary_astigmatism_00,
    zernike.primary_astigmatism_45,
    zernike.primary_coma_y,
    zernike.primary_coma_x,
    zernike.primary_spherical,
    zernike.primary_trefoil_y,
    zernike.primary_trefoil_x,
]


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


def test_all_zernfcns_run_without_error_or_nans(rho, phi):
    for func in all_zernikes:
        assert func(rho, phi).all()


def test_can_build_fringezernike_pupil_with_vector_args():
    abers = np.random.rand(48)
    p = zernike.FringeZernike(abers, samples=SAMPLES)
    assert p


def test_repr_is_a_str():
    p = zernike.FringeZernike()
    assert type(repr(p)) is str


def test_fringezernike_rejects_base_not_0_or_1():
    with pytest.raises(ValueError):
        zernike.FringeZernike(base=2)
    with pytest.raises(ValueError):
        zernike.FringeZernike(base=-1)


def test_fringezernike_takes_all_named_args():
    params = {
        'norm': True,
        'base': 1,
    }
    p = zernike.FringeZernike(**params)
    assert p


def test_fringezernike_will_pass_pupil_args():
    params = {
        'samples': 32,
        'dia': 50,
    }
    p = zernike.FringeZernike(**params)
    assert p


def test_fit_agrees_with_truth(fit_data):
    data, real_coefs = fit_data
    coefs = zernike.zernikefit(data, map_='Fringe')
    real_coefs = np.asarray(real_coefs)
    assert coefs[8] == pytest.approx(real_coefs[8])


def test_fit_does_not_throw_on_normalize(fit_data):
    data, real_coefs = fit_data
    coefs = zernike.zernikefit(data, norm=True, map_='Noll')
    assert coefs[8] != 0


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


def test_truncate_functions(sample):
    assert sample.truncate(9)


def test_truncate_topn_functions(sample):
    assert sample.truncate_topn(9)
