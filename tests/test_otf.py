"""Optical Transfer Function (OTF) unit tests."""
import pytest

import numpy as np

from prysm import otf
from prysm.fttools import forward_ft_unit

import matplotlib
matplotlib.use('Agg')

SAMPLES = 32
LIM = 1e3


def test_mtf_calc_correct():
    x, y = forward_ft_unit(1/1e3, 128), forward_ft_unit(1/1e3, 128)
    xx, yy = np.meshgrid(x, y)
    dat = np.sin(xx)
    mtf = otf.mtf_from_psf(dat, x[1]-x[0])
    center = tuple(s//2 for s in mtf.shape)
    assert mtf.data[center] == 1


def test_ptf_calc_correct():
    x, y = forward_ft_unit(1/1e3, 128), forward_ft_unit(1/1e3, 128)
    xx, yy = np.meshgrid(x, y)
    dat = np.sin(xx)
    ptf = otf.ptf_from_psf(dat, x[1]-x[0])
    center = tuple(s//2 for s in ptf.shape)
    assert ptf.data[center] == 0


def test_otf_calc_correct():
    x, y = forward_ft_unit(1/1e3, 128), forward_ft_unit(1/1e3, 128)
    xx, yy = np.meshgrid(x, y)
    dat = np.sin(xx)
    otf_ = otf.otf_from_psf(dat, x[1]-x[0])
    center = tuple(s//2 for s in otf_.shape)
    assert otf_.data[center] == 1+0j


def _shifted_gaussian(n=14, sig=0.6, x0=0.8, y0=-0.4):
    """An off-center PSF for adjoint tests.

    Narrow in real space so its OTF is broad and its modulus stays well away from
    zero everywhere -- the modulus/angle gradients have kinks where |OTF| -> 0,
    so a floor-bounded modulus is what makes the finite-difference check valid.
    The off-center shift gives the OTF a nonzero phase to exercise the PTF path.
    """
    c = np.arange(n) - n // 2
    xx, yy = np.meshgrid(c, c)
    return np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * sig ** 2))


def test_transform_psf_adjoint_dot_test():
    # <A x, y> == <x, A^H y> for the linear FT and its adjoint
    rng = np.random.default_rng(0)
    n = 16
    x = rng.standard_normal((n, n))
    y = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    Ax, _ = otf.transform_psf(x, dx=1.0)
    Aty = otf.transform_psf_adjoint(y)
    lhs = np.sum(np.conj(Ax) * y)
    rhs = np.sum(np.conj(x) * Aty)
    assert np.allclose(lhs, rhs, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize('which', ['mtf', 'ptf', 'otf'])
def test_from_psf_adjoint_matches_fd(which):
    rng = np.random.default_rng(1)
    psf = _shifted_gaussian()
    dx = 1.0
    v = rng.standard_normal(psf.shape)

    if which == 'mtf':
        fwd, adj = otf.mtf_from_psf, otf.mtf_from_psf_adjoint
        bar = rng.standard_normal(psf.shape)
        loss = lambda p: float(np.sum(bar * fwd(p, dx).data))  # noqa: E731
    elif which == 'ptf':
        fwd, adj = otf.ptf_from_psf, otf.ptf_from_psf_adjoint
        bar = rng.standard_normal(psf.shape)
        loss = lambda p: float(np.sum(bar * fwd(p, dx).data))  # noqa: E731
    else:
        fwd, adj = otf.otf_from_psf, otf.otf_from_psf_adjoint
        bar = rng.standard_normal(psf.shape) + 1j * rng.standard_normal(psf.shape)
        loss = lambda p: float(np.real(np.sum(np.conj(bar) * fwd(p, dx).data)))  # noqa: E731

    psf_bar = adj(bar, psf, dx)
    analytic = float(np.sum(psf_bar * v))

    eps = 1e-6
    fd = (loss(psf + eps * v) - loss(psf - eps * v)) / (2 * eps)
    assert np.allclose(analytic, fd, rtol=1e-5, atol=1e-7)

    # passing the cached transform from return_more must reproduce the recompute path
    _, data = fwd(psf, dx, return_more=True)
    psf_bar_cached = adj(bar, data=data)
    assert np.allclose(psf_bar_cached, psf_bar, rtol=1e-12, atol=1e-12)


def test_encircled_energy_monotonic_and_bounded():
    psf = _shifted_gaussian(n=64, sig=2.0, x0=0.0, y0=0.0)
    psf = psf / psf.sum()
    radii = np.array([2.0, 5.0, 10.0, 20.0, 40.0])
    ee = otf.encircled_energy(psf, dx=1.0, radius=radii)
    assert np.all(np.diff(ee) > 0)          # more energy in a bigger circle
    assert ee[-1] <= 1.0 + 1e-6             # cannot exceed the total
    # scalar and vector forms agree
    assert np.isclose(otf.encircled_energy(psf, 1.0, 10.0), ee[2])


@pytest.mark.parametrize('radius', [12.0, [6.0, 18.0, 35.0]])
def test_encircled_energy_adjoint_matches_fd(radius):
    rng = np.random.default_rng(3)
    psf = _shifted_gaussian()
    dx = 1.0
    v = rng.standard_normal(psf.shape)

    scalar = np.isscalar(radius)
    if scalar:
        ee_bar = rng.standard_normal()
        loss = lambda p: float(ee_bar * otf.encircled_energy(p, dx, radius))  # noqa: E731
    else:
        ee_bar = rng.standard_normal(len(radius))
        loss = lambda p: float(np.sum(ee_bar * otf.encircled_energy(p, dx, radius)))  # noqa: E731

    psf_bar = otf.encircled_energy_adjoint(ee_bar, psf, dx, radius)
    analytic = float(np.sum(psf_bar * v))

    eps = 1e-6
    fd = (loss(psf + eps * v) - loss(psf - eps * v)) / (2 * eps)
    assert np.allclose(analytic, fd, rtol=1e-5, atol=1e-7)

    # cached transform path reproduces the recompute path
    _, data = otf.encircled_energy(psf, dx, radius, return_more=True)
    psf_bar_cached = otf.encircled_energy_adjoint(ee_bar, dx=dx, radius=radius, data=data)
    assert np.allclose(psf_bar_cached, psf_bar, rtol=1e-12, atol=1e-12)


def test_mtf_ptf_otf_from_psf_matches_individual():
    # the combined single-FT routine must agree bit-for-bit with the three
    # per-quantity functions, and its return_more transform must match
    psf = _shifted_gaussian()
    dx = 1.0
    mtf, ptf, otf_, data = otf.mtf_ptf_otf_from_psf(psf, dx, return_more=True)

    mtf_ref, data_ref = otf.mtf_from_psf(psf, dx, return_more=True)
    ptf_ref = otf.ptf_from_psf(psf, dx)
    otf_ref = otf.otf_from_psf(psf, dx)

    assert np.array_equal(mtf.data, mtf_ref.data)
    assert np.array_equal(ptf.data, ptf_ref.data)
    assert np.array_equal(otf_.data, otf_ref.data)
    assert np.array_equal(data, data_ref)
    assert mtf.dx == mtf_ref.dx
