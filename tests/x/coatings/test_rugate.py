"""Tests for rugate / inhomogeneous-index synthesis (Phase 6).

A single-sinusoid rugate produces a reflectance notch at its design wavelength
with about the predicted width; apodization suppresses the sidebands; a
discretized two-level profile reproduces the equivalent homogeneous stack; and
the Fourier (Q-function) inverse method places a notch at the targeted
wavenumber.
"""
import numpy as np
import pytest

from prysm.x.coatings import Stack, RTA
from prysm.x.coatings import rugate as ru


def _spectrum(stack, lams, pol='s'):
    R, _, _ = RTA(stack, lams, 0.0, pol)
    return np.asarray(R)


# ----------------------------------------------------------- single sinusoid

def test_sinusoid_notch_center_and_width():
    n_avg, n_amp, lam0 = 1.8, 0.10, 0.55
    s = ru.sinusoidal_rugate(n_avg, n_amp, lam0, n_periods=30,
                             sublayers_per_period=30)
    lams = np.linspace(0.45, 0.70, 800)
    R = _spectrum(s, lams)
    ipk = int(np.argmax(R))
    # notch centered at the first-order design wavelength
    assert lams[ipk] == pytest.approx(lam0, abs=2e-3)
    assert R[ipk] > 0.9                               # strong rejection
    # width within a factor of 2 of the coupled-mode estimate n_amp/n_avg*lam0
    half = R[ipk] / 2
    band = lams[R >= half]
    fwhm = band.max() - band.min()
    predicted = n_amp / n_avg * lam0
    assert 0.5 * predicted < fwhm < 2.0 * predicted


def test_notch_wavelength_round_trips_period():
    Lam = ru.rugate_period(1.8, 0.55)
    assert ru.notch_wavelength(1.8, Lam) == pytest.approx(0.55)


# ----------------------------------------------------------- apodization

def test_apodization_reduces_sidebands():
    n_avg, n_amp, lam0 = 1.8, 0.10, 0.55
    plain = ru.sinusoidal_rugate(n_avg, n_amp, lam0, n_periods=30,
                                 sublayers_per_period=30)
    apod = ru.sinusoidal_rugate(n_avg, n_amp, lam0, n_periods=30,
                                sublayers_per_period=30,
                                apodization=ru.quintic_taper(0.4))
    lams = np.linspace(0.45, 0.70, 800)
    Rp = _spectrum(plain, lams)
    Ra = _spectrum(apod, lams)
    # measure sideband ripple outside the central notch
    fwhm = 0.04
    mask = np.abs(lams - lam0) > 1.5 * fwhm
    assert Ra[mask].max() < 0.85 * Rp[mask].max()


def test_apodize_preserves_mean_and_tapers_excursion():
    n_avg = 1.8
    base = lambda z: n_avg + 0.1 * np.sin(2 * np.pi * z / 0.1)
    win = ru.quintic_taper(0.5)
    tapered = ru.apodize(base, n_avg, 1.0, win)
    # at the very edge the window is ~0, so the profile collapses to the mean
    assert tapered(0.0) == pytest.approx(n_avg, abs=1e-9)
    # at the center the window is 1, so the excursion is preserved
    assert tapered(0.5) == pytest.approx(base(0.5), rel=1e-9)


# ----------------------------------------------------------- discretization

def test_discretized_two_level_matches_homogeneous_stack():
    def two_level(z):
        return 1.46 if (z % 0.2) < 0.1 else 2.2

    s = ru.discretize_profile(two_level, 0.4, 4, 1.52)
    assert [round(float(n), 3) for n in s.indices] == [1.46, 2.2, 1.46, 2.2]
    manual = Stack([1.46, 2.2, 1.46, 2.2], [0.1, 0.1, 0.1, 0.1], 1.52)
    lams = np.linspace(0.45, 0.70, 50)
    assert np.allclose(_spectrum(s, lams), _spectrum(manual, lams))


def test_discretize_profile_thickness_and_count():
    s = ru.discretize_profile(lambda z: 1.5, 1.0, 25, 1.52)
    assert len(s) == 25
    assert np.allclose(np.asarray(s.thicknesses), 0.04)


# ----------------------------------------------------------- inverse method

def test_rugate_from_target_places_notch():
    n_avg, lam0 = 1.8, 0.55
    k0 = 2 * np.pi / lam0
    k = np.linspace(0.5 * k0, 1.5 * k0, 2000)
    target = 0.3 * np.exp(-((k - k0) / (0.03 * k0)) ** 2)
    s = ru.rugate_from_target(k, target, n_avg, total_optical_thickness=40.0,
                              n_sublayers=1500)
    lams = np.linspace(0.45, 0.70, 600)
    R = _spectrum(s, lams)
    ipk = int(np.argmax(R))
    assert lams[ipk] == pytest.approx(lam0, abs=4e-3)
