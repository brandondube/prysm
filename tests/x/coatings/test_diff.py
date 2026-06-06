"""Tests for the differentiable transfer-matrix engine (Phase 2).

The defining checks are (1) the per-layer matrix vjp satisfies the transpose
dot-product identity, and (2) the assembled thickness gradient matches central
finite differences for reflectance, transmittance, absorptance, and field
merits, vectorized over a spectral/angular grid and both polarizations.
"""
import numpy as np
import pytest

from prysm.x.coatings import Stack
from prysm.x.coatings import diff
from prysm.x.coatings.merit import (
    Reflectance, Transmittance, LayerAbsorptance, FieldIntensityAtBoundary,
    MeritFunction,
)

wvl = 0.55
# dielectric stack with one absorbing layer so A and the lossy gradient are live
INDICES = [1.38, 2.05, 1.5 + 0.3j, 2.05]
TH = np.array([0.12, 0.08, 0.05, 0.10])
SUB = 1.52


def _stack(th=TH):
    return Stack(INDICES, th, SUB)


# ----------------------------------------------------------- transpose test

def test_char_matrix_vjp_transpose():
    rng = np.random.default_rng(0xBEEF)
    shape = (6,)
    beta = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    eta = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)

    dbeta = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    deta = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    M_bar = (rng.standard_normal(shape + (2, 2))
             + 1j * rng.standard_normal(shape + (2, 2)))

    dMdb = diff._dchar_dbeta(beta, eta)
    dMde = diff._dchar_deta(beta, eta)
    dM = dMdb * dbeta[..., None, None] + dMde * deta[..., None, None]

    c_beta, c_eta = diff.char_matrix_vjp(beta, eta, M_bar)

    lhs = np.real(np.sum(np.conj(M_bar) * dM))
    rhs = np.real(np.sum(np.conj(c_beta) * dbeta + np.conj(c_eta) * deta))
    assert np.isclose(lhs, rhs, rtol=1e-12)


# ----------------------------------------------------------- gradient vs FD

def _merit_terms():
    return {
        'R': Reflectance(np.array([0.45, 0.55, 0.65]), target=0.0),
        'T': Transmittance(np.array([0.45, 0.55, 0.65]), target=0.9),
        'A': LayerAbsorptance(2, np.array([0.5, 0.6]), target=0.0),
        'E': FieldIntensityAtBoundary(2, np.array([0.5, 0.6]), target=0.0),
    }


def _fd_grad(term, th, h=1e-7):
    g = np.zeros_like(th)
    for i in range(th.size):
        tp = th.copy(); tp[i] += h
        tm = th.copy(); tm[i] -= h
        g[i] = (term.value(_stack(tp)) - term.value(_stack(tm))) / (2 * h)
    return g


@pytest.mark.parametrize('pol', ['s', 'p', 'avg'])
@pytest.mark.parametrize('key', ['R', 'T', 'A', 'E'])
def test_thickness_gradient_matches_fd(pol, key):
    term = _merit_terms()[key]
    term.theta = np.radians(20.0)
    term.pol = pol
    _, g_analytic = term.value_and_grad(_stack())
    g_fd = _fd_grad(term, TH)
    assert np.allclose(g_analytic, g_fd, rtol=2e-5, atol=1e-8)


@pytest.mark.parametrize('pol', ['s', 'p'])
@pytest.mark.parametrize('aoi', [0.0, 30.0])
@pytest.mark.parametrize('key', ['R', 'T', 'A', 'E'])
def test_index_gradient_matches_fd(pol, aoi, key):
    # the index gradient must be correct off normal incidence too (the angle
    # depends on index through Snell's law).
    indices = [1.38, 2.05, 1.5 + 0.2j, 2.2]
    th = np.array([0.10, 0.07, 0.05, 0.06])
    theta = np.radians(aoi)
    wv = np.array([0.5, 0.55, 0.6])

    def fwd(idx):
        return diff.forward_eval(Stack(idx, th, SUB), wv, theta, pol)

    def value(idx):
        f = fwd(idx)
        q = {'R': f.R_value, 'T': f.T_value, 'A': f.A_value, 'E': f.Esq_value}[key]
        return float(np.sum(q ** 2))

    f0 = fwd(indices)
    if key == 'R':
        g = diff.index_gradient(f0, dR=2 * f0.R_value)
    elif key == 'T':
        g = diff.index_gradient(f0, dT=2 * f0.T_value)
    elif key == 'A':
        g = diff.index_gradient(f0, dA=2 * f0.A_value)
    else:
        g = diff.index_gradient(f0, dEsq=2 * f0.Esq_value)

    g_fd = np.zeros(len(indices))
    h = 1e-7
    for i in range(len(indices)):
        ip = list(indices); ip[i] = ip[i] + h
        im = list(indices); im[i] = im[i] - h
        g_fd[i] = (value(ip) - value(im)) / (2 * h)
    assert np.allclose(g, g_fd, rtol=3e-5, atol=1e-8)


def test_merit_function_sums_terms():
    terms = list(_merit_terms().values())
    mf = MeritFunction(terms)
    val, grad = mf.value_and_grad(_stack())
    # value equals the sum of term values
    assert val == pytest.approx(sum(t.value(_stack()) for t in terms))
    # gradient equals central FD of the aggregate value
    g_fd = np.zeros_like(TH)
    for i in range(TH.size):
        tp = TH.copy(); tp[i] += 1e-7
        tm = TH.copy(); tm[i] -= 1e-7
        g_fd[i] = (mf.value(_stack(tp)) - mf.value(_stack(tm))) / 2e-7
    assert np.allclose(grad, g_fd, rtol=2e-5, atol=1e-8)


def test_value_matches_rta():
    from prysm.x.coatings import RTA
    s = _stack()
    R, T, A = RTA(s, np.array([0.45, 0.55]), np.radians(10.0), 's')
    fwd = diff.forward_eval(s, np.array([0.45, 0.55]), np.radians(10.0), 's')
    assert np.allclose(fwd.R_value, R)
    assert np.allclose(fwd.T_value, T)
    assert np.allclose(fwd.A_value, A)
