"""the wavefront-differential front end (the wavefront-error quadratic).

Validates wavefront_differential / WavefrontDifferential built from one nominal
differential trace:

- the assembled quadratic (C, B, Gram) reproduces design.WavefrontRMS' nominal
  RMS and the FD sensitivity_table slope;
- rms_at(p, T) tracks the re-traced RMS and beats the pure-linear extrapolation
  at finite T (the point of the quadratic);
- inverse_sensitivity round-trips against rms_at;
- expected_rms is the exact independent-zero-mean roll-up of the quadratic;
- fast_monte_carlo matches the slow re-tracing tolerance.monte_carlo in
  distribution (same seed, same perturbations) -- the headline deliverable;
- cumulative_probability is a proper empirical CDF.
"""
import numpy as np
import pytest

from prysm.x.raytracing import OpticalSystem
from prysm.x.raytracing import LensData
from prysm.x.raytracing.launch import Field, Sampling, launch
from prysm.x.raytracing.surfaces import Conic, Plane
from prysm.x.raytracing.spencer_and_murty import STYPE_EVAL
from prysm.x.raytracing.paraxial import paraxial_image_distance
from prysm.x.raytracing.design import WavefrontRMS
from prysm.x.raytracing.tolerance import (
    Perturbation, sensitivity_table, monte_carlo, operand_as_merit,
)
from prysm.x.raytracing.wavefront_differential import (
    wavefront_differential, WavefrontDifferential, cumulative_probability,
)


WVL = 0.5
NG = 1.6


def _glass(w):
    return NG


def _air(w):
    return 1.0


def _place_image(ld, gap_row):
    lens = [s for s in ld.to_surfaces() if s.typ != STYPE_EVAL]
    bfd = float(paraxial_image_distance(lens, wvl=WVL))
    ld.rows[gap_row].thickness = bfd
    ld.lens._invalidate()
    return ld


def singlet():
    lens = LensData()
    (lens.add(Conic(1 / 30.0, 0.0), typ='refr', thickness=4.0, material=_glass)
         .add(Conic(-1 / 30.0, 0.0), typ='refr', thickness=20.0, material=_air)
         .add(Plane(), typ='eval'))
    ld = OpticalSystem(lens, aperture=10.0, wavelengths=[WVL])
    return _place_image(ld, gap_row=1)


def singlet_cb():
    lens = LensData()
    (lens.add(Conic(1 / 30.0, 0.0), typ='refr', thickness=4.0, material=_glass)
         .add_coordbreak(decenter=(0., 0., 0.), tilt=(0., 0., 0.),
                         kind='basic', thickness=0.0)
         .add(Conic(-1 / 30.0, 0.0), typ='refr', thickness=20.0, material=_air)
         .add(Plane(), typ='eval'))
    ld = OpticalSystem(lens, aperture=10.0, wavelengths=[WVL])
    return _place_image(ld, gap_row=2)


def bundle(ld):
    return launch(ld, Field(2.5, 2.5), WVL, Sampling.rect(n=7),
                  epd=10.0, pupil_z=-5.0)


def basic_perts(ld):
    return [
        Perturbation.normal(ld, 'curvature', 0, 1e-5, name='c1'),
        Perturbation.normal(ld, 'conic', 0, 1e-4, name='k1'),
        Perturbation.normal(ld, 'thickness', 0, 5e-4, name='t0'),
    ]


def merit_of(ld, P, S):
    return operand_as_merit(WavefrontRMS(P, S, WVL))


# ---------- the model reproduces the gate ----------------------------------

def test_nominal_rms_matches_wavefrontrms():
    ld = singlet()
    P, S = bundle(ld)
    wd = wavefront_differential(ld, basic_perts(ld), P, S, WVL)
    m_nom = merit_of(ld, P, S)(ld)
    np.testing.assert_allclose(wd.rms_nominal, m_nom, rtol=1e-10)


def test_wavefront_differential_resolves_system_wavelength_name():
    def dispersive(w):
        return 1.5 + 0.02 * (w - 0.55)

    lens = LensData()
    (lens.add(Conic(1 / 40.0, 0.0), typ='refr', thickness=4.0,
              material=dispersive)
         .add(Conic(-1 / 40.0, 0.0), typ='refr', thickness=20.0,
              material=_air)
         .add(Plane(), typ='eval'))
    sys = OpticalSystem(lens, aperture=10.0, wavelengths={'d': 0.55},
                        reference_wavelength='d')
    sys.solve_image_distance(wavelength='d')
    P, S = launch(sys, Field(0.0, 0.0), sys.wavelength('d'),
                  Sampling.rect(n=3), epd=10.0, pupil_z=-5.0)
    perts = [Perturbation.normal(sys, 'curvature', 0, 1e-5, name='c1')]
    with pytest.raises(ValueError, match='near-axial chief ray'):
        wavefront_differential(sys, perts, P, S, 'd')
    by_name = wavefront_differential(sys, perts, P, S, 'd', P_xp=(0, 0, 0))
    by_value = wavefront_differential(sys, perts, P, S, 0.55, P_xp=(0, 0, 0))
    np.testing.assert_allclose(by_name.W0, by_value.W0)
    np.testing.assert_allclose(by_name.dW, by_value.dW)


def test_sensitivity_matches_fd_sensitivity_table():
    ld = singlet()
    P, S = bundle(ld)
    perts = basic_perts(ld)
    wd = wavefront_differential(ld, perts, P, S, WVL)
    fd = sensitivity_table(ld, perts, merit_of(ld, P, S)).sensitivities()
    np.testing.assert_allclose(wd.sensitivity(), fd, rtol=3e-3, atol=1e-9)


def test_gram_is_symmetric_with_A_on_the_diagonal():
    ld = singlet()
    P, S = bundle(ld)
    wd = wavefront_differential(ld, basic_perts(ld), P, S, WVL)
    G = wd.gram()
    np.testing.assert_allclose(G, G.T, rtol=0, atol=1e-18)
    np.testing.assert_allclose(np.diag(G), wd.A, rtol=0, atol=0)


def test_predict_rms_sq_zero_tau_is_nominal():
    ld = singlet()
    P, S = bundle(ld)
    wd = wavefront_differential(ld, basic_perts(ld), P, S, WVL)
    np.testing.assert_allclose(wd.predict_rms(np.zeros(3)), wd.rms_nominal,
                               rtol=1e-12)


# ---------- the quadratic vs a re-trace ------------------------------------

def _retrace_rms(ld, P, S, pert, T):
    """Actual RMS with pert set to nominal + T (restored after)."""
    merit = merit_of(ld, P, S)
    try:
        pert.set(pert.nominal + T)
        return float(merit(ld))
    finally:
        pert.reset()


def test_rms_at_tracks_retrace_and_beats_linear():
    """At finite T the quadratic rms_at is closer to the re-traced RMS than the
    pure-linear extrapolation rms_nominal + T*sensitivity."""
    ld = singlet()
    P, S = bundle(ld)
    pert = Perturbation.normal(ld, 'curvature', 0, 1e-5, name='c1')
    wd = wavefront_differential(ld, [pert], P, S, WVL)
    T = 2e-3  # well beyond the FD step, so curvature of RMS(T) matters
    true_rms = _retrace_rms(ld, P, S, pert, T)
    quad = float(wd.rms_at(0, T))
    linear = wd.rms_nominal + T * float(wd.sensitivity()[0])
    assert abs(quad - true_rms) < abs(linear - true_rms)
    np.testing.assert_allclose(quad, true_rms, rtol=5e-3)


def test_full_quadratic_form_matches_linearized_wavefront():
    """predict_rms_sq(tau) must equal mean((W0 + dW@tau)**2) by construction."""
    ld = singlet_cb()
    P, S = bundle(ld)
    perts = [
        Perturbation.normal(ld, 'curvature', 0, 1e-5, name='c1'),
        Perturbation.normal(ld, 'curvature', 2, 1e-5, name='c2'),
        Perturbation.normal(ld, 'thickness', 0, 5e-4, name='t0'),
    ]
    wd = wavefront_differential(ld, perts, P, S, WVL)
    rng = np.random.default_rng(0)
    tau = rng.normal(size=3) * np.array([1e-3, 1e-3, 5e-2])
    W = wd.W0 + wd.dW @ tau
    np.testing.assert_allclose(wd.predict_rms_sq(tau), np.mean(W * W),
                               rtol=1e-10)


# ---------- inverse sensitivity --------------------------------------------

def test_inverse_sensitivity_round_trips():
    ld = singlet()
    P, S = bundle(ld)
    perts = basic_perts(ld)
    wd = wavefront_differential(ld, perts, P, S, WVL)
    target = 0.25 * wd.rms_nominal
    t_lo, t_hi = wd.inverse_sensitivity(target)
    want = wd.rms_nominal + target
    for p in range(len(perts)):
        np.testing.assert_allclose(float(wd.rms_at(p, t_hi[p])), want, rtol=1e-7)
        np.testing.assert_allclose(float(wd.rms_at(p, t_lo[p])), want, rtol=1e-7)
        assert t_lo[p] <= 0.0 <= t_hi[p]


def test_inverse_sensitivity_linear_only_tolerance_is_one_sided():
    # a fabricated model: A=0, B>0 -> a single finite bound, inf on the slack
    wd = WavefrontDifferential.__new__(WavefrontDifferential)
    wd.A = np.array([0.0])
    wd.B = np.array([2.0])
    wd.C = 1.0
    wd.rms_nominal = 1.0
    wd.n_params = 1
    t_lo, t_hi = wd.inverse_sensitivity(0.5)
    # RMS_target=1.5 -> 1.5^2 = 1 + 2T -> T = 0.625
    np.testing.assert_allclose(t_hi[0], 0.625, rtol=1e-12)
    assert t_lo[0] == -np.inf


# ---------- RSS roll-up -----------------------------------------------------

def test_expected_rms_sq_is_quadratic_mean_over_independent_mc():
    """E[RMS^2] = C + sum sigma^2 A is exactly the mean of the quadratic model's
    RMS^2 over independent zero-mean draws (no linearization gap: both evaluate
    the same quadratic)."""
    ld = singlet()
    P, S = bundle(ld)
    perts = basic_perts(ld)
    wd = wavefront_differential(ld, perts, P, S, WVL)
    res = wd.fast_monte_carlo(perts, n_trials=40000, seed=7)
    mc_mean_sq = float(np.mean(res.merits ** 2))
    np.testing.assert_allclose(wd.expected_rms_sq(), mc_mean_sq, rtol=2e-2)


def test_rms_change_per_tolerance_nonnegative_and_named():
    ld = singlet()
    P, S = bundle(ld)
    perts = basic_perts(ld)
    wd = wavefront_differential(ld, perts, P, S, WVL)
    drms = wd.rms_change_per_tolerance()
    assert drms.shape == (3,)
    rows = wd.rows()
    assert [r['name'] for r in rows] == ['c1', 'k1', 't0']


# ---------- fast MC vs slow re-tracing MC (the headline) --------------------

def test_fast_mc_matches_slow_monte_carlo_distribution():
    ld = singlet()
    P, S = bundle(ld)
    perts = basic_perts(ld)
    wd = wavefront_differential(ld, perts, P, S, WVL)

    n, seed = 3000, 12345
    fast = wd.fast_monte_carlo(perts, n_trials=n, seed=seed)
    slow = monte_carlo(ld, perts, merit_of(ld, P, S), n_trials=n, seed=seed)

    fs, ss = fast.summary(), slow.summary()
    np.testing.assert_allclose(fs['mean'], ss['mean'], rtol=1e-2)
    np.testing.assert_allclose(fs['std'], ss['std'], rtol=5e-2)
    np.testing.assert_allclose(fs['p95'], ss['p95'], rtol=2e-2)
    # shared seed + matched RNG order -> per-trial agreement to linearization
    np.testing.assert_allclose(fast.merits, slow.merits, rtol=2e-2, atol=1e-6)


def test_fast_mc_requires_matching_param_count():
    ld = singlet()
    P, S = bundle(ld)
    perts = basic_perts(ld)
    wd = wavefront_differential(ld, perts, P, S, WVL)
    with pytest.raises(ValueError, match='match the model'):
        wd.fast_monte_carlo(perts[:2], n_trials=10)


# ---------- cumulative probability ------------------------------------------

def test_cumulative_probability_is_a_cdf():
    ld = singlet()
    P, S = bundle(ld)
    perts = basic_perts(ld)
    wd = wavefront_differential(ld, perts, P, S, WVL)
    res = wd.fast_monte_carlo(perts, n_trials=2000, seed=1)
    thresh, prob = cumulative_probability(res)
    assert np.all(np.diff(thresh) >= 0)        # sorted
    assert np.all(np.diff(prob) >= 0)          # monotone
    np.testing.assert_allclose(prob[-1], 1.0, rtol=0, atol=1e-12)
    # the curve agrees with the result's own yield_at at a probe threshold
    probe = float(np.median(res.merits))
    np.testing.assert_allclose(res.yield_at(probe),
                               prob[np.searchsorted(thresh, probe, 'right') - 1],
                               atol=1.0 / res.n_trials + 1e-12)


def test_cumulative_probability_accepts_raw_array():
    m, p = cumulative_probability(np.array([3.0, 1.0, 2.0]))
    np.testing.assert_allclose(m, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(p, [1 / 3, 2 / 3, 1.0])
