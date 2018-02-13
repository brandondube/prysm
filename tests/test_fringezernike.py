''' Unit tests for the fringezernike submodule.
'''
import pytest

import numpy as np

from prysm.coordinates import cart_to_polar
from prysm import fringezernike


Z = fringezernike.zernfcns
norms = fringezernike._normalizations
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
    p = fringezernike.FringeZernike(Z9=1, samples=SAMPLES)
    return p.phase, p.coefs


def test_all_zernfcns_run_without_error_or_nans(rho, phi):
    for _, zernike in Z.items():
        assert zernike(rho, phi).all()


def test_all_zernfcns_run_without_errors_or_nans_with_norms(rho, phi):
    for (_, zernike), norm in zip(Z.items(), norms):
        assert (zernike(rho, phi) * norm).all()


def test_can_build_fringezernike_pupil_with_vector_args():
    abers = np.random.rand(48)
    p = fringezernike.FringeZernike(abers, samples=SAMPLES)
    assert p


def test_repr_is_a_str():
    p = fringezernike.FringeZernike()
    assert type(repr(p)) is str


def test_fringezernike_rejects_base_not_0_or_1():
    with pytest.raises(ValueError):
        fringezernike.FringeZernike(base=2)
    with pytest.raises(ValueError):
        fringezernike.FringeZernike(base=-1)


def test_fringezernike_takes_all_named_args():
    params = {
        'rms_norm': True,
        'base': 1,
    }
    p = fringezernike.FringeZernike(**params)
    assert p


def test_fringezernike_will_pass_pupil_args():
    params = {
        'samples': 32,
        'wavelength': 0.5,
    }
    p = fringezernike.FringeZernike(**params)
    assert p


def test_fit_agrees_with_truth(fit_data):
    data, real_coefs = fit_data
    coefs = fringezernike.fit(data)
    real_coefs = np.asarray(real_coefs)
    assert coefs[8] == pytest.approx(real_coefs[8])


def test_fit_does_not_throw_on_normalize(fit_data):
    data, real_coefs = fit_data
    coefs = fringezernike.fit(data, rms_norm=True)
    assert coefs[8] != 0


def test_fit_raises_on_too_many_terms(fit_data):
    data, real_coefs = fit_data
    with pytest.raises(ValueError):
        fringezernike.fit(data, num_terms=100)
