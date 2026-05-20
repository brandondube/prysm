"""Tests for the tolerancing layer."""
import numpy as np
import pytest

from tests.x.raytracing.surface_helpers import (
    plane, sphere, conic, off_axis_conic, even_asphere, q2d, zernike, xy,
    chebyshev, jacobi, toroid, biconic,
)

from prysm.x.raytracing.surfaces import Surface
from prysm.x.raytracing.spencer_and_murty import raytrace
from prysm.x.raytracing.launch import Field, Sampling, launch
from prysm.x.raytracing.design import (
    curvature_of, kappa_of, thickness_after, position_of,
    RmsSpotRadius, EFL,
)
from prysm.x.raytracing.opt import rms_spot_radius
from prysm.x.raytracing.tolerance import (
    Perturbation, SensitivityTable, MonteCarloResult,
    sensitivity_table, monte_carlo, operand_as_merit,
)


# ---------- helpers --------------------------------------------------------

def _spherical_singlet():
    n_glass = lambda w: 1.5
    s1 = conic(c=1 / 50.0, k=0.0, typ='refr',
                       P=[0, 0, 0], n=n_glass)
    s2 = conic(c=-1 / 50.0, k=0.0, typ='refr',
                       P=[0, 0, 5.0], n=lambda w: 1.0)
    img = plane(typ='eval', P=[0, 0, 100.0])
    return [s1, s2, img]


def _concave_parabola():
    c = -1 / 80.0
    f = 1.0 / (2.0 * c)
    s = conic(c=c, k=-1.0, typ='refl', P=[0, 0, 0])
    img = plane(typ='eval', P=[0, 0, f])
    return [s, img]


# ---------- Perturbation factories ----------------------------------------

def test_perturbation_normal_captures_nominal_and_sigma():
    presc = _spherical_singlet()
    p = Perturbation.normal(curvature_of(presc[0]), sigma=1e-4, name='c1')
    assert p.name == 'c1'
    np.testing.assert_allclose(p.nominal, 1 / 50.0)
    assert p.step == 1e-4


def test_perturbation_normal_relative_scales_with_nominal():
    presc = _spherical_singlet()
    p = Perturbation.normal_relative(
        curvature_of(presc[0]), sigma_rel=0.001, name='c1',
    )
    np.testing.assert_allclose(p.step, abs(1 / 50.0) * 0.001)


def test_perturbation_uniform_half_width():
    presc = _spherical_singlet()
    p = Perturbation.uniform(kappa_of(presc[0]), half_width=0.05, name='k1')
    assert p.step == 0.05
    assert p.nominal == 0.0


def test_perturbation_triangular_uses_nominal_as_peak():
    presc = _spherical_singlet()
    rng = np.random.default_rng(0)
    p = Perturbation.triangular(kappa_of(presc[0]), half_width=0.1, name='k1')
    # draw many; mean should be at nominal (within MC noise)
    samples = np.array([p.sample(rng) for _ in range(5000)])
    np.testing.assert_allclose(samples.mean(), 0.0, atol=2e-3)
    assert float(samples.min()) >= -0.1
    assert float(samples.max()) <= 0.1


def test_perturbation_sample_normal_centers_on_nominal():
    presc = _spherical_singlet()
    p = Perturbation.normal(kappa_of(presc[0]), sigma=0.05, name='k1')
    rng = np.random.default_rng(42)
    samples = np.array([p.sample(rng) for _ in range(10000)])
    np.testing.assert_allclose(samples.mean(), 0.0, atol=2e-3)
    np.testing.assert_allclose(samples.std(), 0.05, rtol=0.05)


def test_perturbation_reset_restores_value():
    presc = _spherical_singlet()
    p = Perturbation.normal(kappa_of(presc[0]), sigma=0.05, name='k1')
    p.setter(0.5)
    assert presc[0].params['k'] == 0.5
    p.reset()
    assert presc[0].params['k'] == 0.0


# ---------- operand_as_merit ----------------------------------------------

def test_operand_as_merit_runs():
    presc = _concave_parabola()
    P, S = launch(presc, Field(0., 0.), 0.55e-3,
                  Sampling.fan(n=11), epd=10.0, pupil_z=-50.0)
    op = RmsSpotRadius(P, S, wavelength=0.55e-3)
    merit = operand_as_merit(op)
    val = merit(presc)
    assert val >= 0.0
    # the perfect parabola should give a near-zero spot
    assert val < 1e-9


# ---------- sensitivity_table ----------------------------------------------

def test_sensitivity_table_zero_step_yields_zero_sensitivity():
    presc = _concave_parabola()
    pert = Perturbation.normal(kappa_of(presc[0]), sigma=0.0, name='k')
    merit = lambda p: float(p[0].params['k'])
    table = sensitivity_table(presc, [pert], merit)
    assert table.rows[0]['sensitivity'] == 0.0


def test_sensitivity_table_matches_analytic_for_linear_merit():
    """If merit is a linear function of a parameter, the centered FD
    sensitivity is exactly the linear coefficient."""
    presc = _concave_parabola()
    pert = Perturbation.normal(kappa_of(presc[0]), sigma=1e-3, name='k')
    # merit(prescription) = 3.7 * k + 2.0
    merit = lambda p: 3.7 * float(p[0].params['k']) + 2.0
    table = sensitivity_table(presc, [pert], merit)
    np.testing.assert_allclose(table.rows[0]['sensitivity'], 3.7, rtol=1e-12)


def test_sensitivity_table_restores_parameter():
    """After computing sensitivities, the prescription must be unchanged."""
    presc = _concave_parabola()
    nominal_k = presc[0].params['k']
    nominal_c = presc[0].params['c']
    perts = [
        Perturbation.normal(kappa_of(presc[0]), sigma=0.1, name='k'),
        Perturbation.normal(curvature_of(presc[0]), sigma=1e-4, name='c'),
    ]
    merit = lambda p: float(p[0].params['c'])
    _ = sensitivity_table(presc, perts, merit)
    assert presc[0].params['k'] == nominal_k
    assert presc[0].params['c'] == nominal_c


def test_sensitivity_table_records_per_row_deltas():
    presc = _concave_parabola()
    pert = Perturbation.normal(kappa_of(presc[0]), sigma=0.01, name='k')
    merit = lambda p: 2.0 * float(p[0].params['k'])  # linear
    table = sensitivity_table(presc, [pert], merit)
    row = table.rows[0]
    # delta_plus = merit(+h) - merit(0) = 2*0.01 = 0.02
    np.testing.assert_allclose(row['delta_plus'], 0.02, rtol=1e-12)
    np.testing.assert_allclose(row['delta_minus'], -0.02, rtol=1e-12)


def test_sensitivity_table_global_step_overrides_per_row():
    presc = _concave_parabola()
    pert = Perturbation.normal(kappa_of(presc[0]), sigma=0.01, name='k')
    merit = lambda p: float(p[0].params['k'])
    table = sensitivity_table(presc, [pert], merit, step=0.1)
    assert table.rows[0]['step'] == 0.1


def test_sensitivity_table_repr_lists_each_row():
    presc = _concave_parabola()
    perts = [
        Perturbation.normal(kappa_of(presc[0]), sigma=0.05, name='k'),
        Perturbation.normal(curvature_of(presc[0]), sigma=1e-4, name='c'),
    ]
    merit = lambda p: float(p[0].params['k']) + float(p[0].params['c'])
    table = sensitivity_table(presc, perts, merit)
    s = repr(table)
    assert 'SensitivityTable' in s
    assert 'k' in s
    assert 'c' in s


def test_sensitivity_table_sensitivities_array_shape():
    presc = _concave_parabola()
    perts = [
        Perturbation.normal(kappa_of(presc[0]), sigma=0.01, name='k'),
        Perturbation.normal(curvature_of(presc[0]), sigma=1e-4, name='c'),
    ]
    merit = lambda p: float(p[0].params['k'])
    table = sensitivity_table(presc, perts, merit)
    s = table.sensitivities()
    assert s.shape == (2,)


# ---------- monte_carlo ----------------------------------------------------

def test_monte_carlo_zero_sigma_returns_nominal_merit():
    """All-zero-sigma perturbations leave the prescription untouched."""
    presc = _spherical_singlet()
    perts = [Perturbation.normal(kappa_of(presc[0]), sigma=0.0, name='k')]

    def merit(p):
        return float(p[0].params['k'])

    result = monte_carlo(presc, perts, merit, n_trials=50, seed=0)
    np.testing.assert_array_equal(result.merits, np.zeros(50))


def test_monte_carlo_reproducible_with_seed():
    presc = _spherical_singlet()
    perts = [Perturbation.normal(kappa_of(presc[0]), sigma=0.01, name='k')]
    merit = lambda p: float(p[0].params['k'])
    r1 = monte_carlo(presc, perts, merit, n_trials=100, seed=12345)
    r2 = monte_carlo(presc, perts, merit, n_trials=100, seed=12345)
    np.testing.assert_array_equal(r1.merits, r2.merits)


def test_monte_carlo_different_seeds_differ():
    presc = _spherical_singlet()
    perts = [Perturbation.normal(kappa_of(presc[0]), sigma=0.01, name='k')]
    merit = lambda p: float(p[0].params['k'])
    r1 = monte_carlo(presc, perts, merit, n_trials=100, seed=1)
    r2 = monte_carlo(presc, perts, merit, n_trials=100, seed=2)
    assert not np.array_equal(r1.merits, r2.merits)


def test_monte_carlo_restores_nominal_on_completion():
    presc = _spherical_singlet()
    nominal_k = presc[0].params['k']
    perts = [Perturbation.normal(kappa_of(presc[0]), sigma=0.1, name='k')]
    merit = lambda p: float(p[0].params['k'])
    _ = monte_carlo(presc, perts, merit, n_trials=20, seed=0)
    assert presc[0].params['k'] == nominal_k


def test_monte_carlo_restores_nominal_on_merit_exception():
    """Even if merit raises mid-trial, the prescription is restored."""
    presc = _spherical_singlet()
    nominal_k = presc[0].params['k']
    perts = [Perturbation.normal(kappa_of(presc[0]), sigma=0.1, name='k')]
    call_counter = {'n': 0}

    def merit(p):
        call_counter['n'] += 1
        if call_counter['n'] == 3:
            raise RuntimeError('synthetic failure mid-trial')
        return float(p[0].params['k'])

    with pytest.raises(RuntimeError):
        monte_carlo(presc, perts, merit, n_trials=20, seed=0)
    assert presc[0].params['k'] == nominal_k


def test_monte_carlo_summary_keys():
    presc = _spherical_singlet()
    perts = [Perturbation.normal(kappa_of(presc[0]), sigma=0.01, name='k')]
    merit = lambda p: float(p[0].params['k']) ** 2
    result = monte_carlo(presc, perts, merit, n_trials=200, seed=0)
    summary = result.summary()
    for k in ('n_trials', 'min', 'max', 'mean', 'std',
              'median', 'p95', 'p99'):
        assert k in summary


def test_monte_carlo_yield_at_threshold_edges():
    presc = _spherical_singlet()
    perts = [Perturbation.normal(kappa_of(presc[0]), sigma=0.01, name='k')]
    merit = lambda p: float(p[0].params['k']) ** 2  # always >= 0
    result = monte_carlo(presc, perts, merit, n_trials=200, seed=0)
    # all merits are non-negative, so threshold +inf -> 100% yield
    assert result.yield_at(float('inf')) == 1.0
    # threshold -inf -> 0% yield
    assert result.yield_at(float('-inf')) == 0.0


def test_monte_carlo_record_samples_shape():
    presc = _spherical_singlet()
    perts = [
        Perturbation.normal(kappa_of(presc[0]), sigma=0.01, name='k'),
        Perturbation.normal(curvature_of(presc[0]), sigma=1e-4, name='c'),
    ]
    merit = lambda p: float(p[0].params['k'])
    result = monte_carlo(presc, perts, merit, n_trials=30, seed=0,
                         record_samples=True)
    assert result.sampled_x.shape == (30, 2)


def test_monte_carlo_sampled_x_is_none_by_default():
    presc = _spherical_singlet()
    perts = [Perturbation.normal(kappa_of(presc[0]), sigma=0.01, name='k')]
    merit = lambda p: float(p[0].params['k'])
    result = monte_carlo(presc, perts, merit, n_trials=10, seed=0)
    assert result.sampled_x is None


# ---------- end-to-end optical example ------------------------------------

def test_monte_carlo_on_spherical_aberration_recovery():
    """The parabola is stigmatic; perturbing kappa increases the spot
    RMS, so most MC samples have non-zero (positive) merit."""
    presc = _concave_parabola()
    P, S = launch(presc, Field(0., 0.), 0.55e-3,
                  Sampling.fan(n=11), epd=10.0, pupil_z=-50.0)
    op = RmsSpotRadius(P, S, wavelength=0.55e-3)
    merit = operand_as_merit(op)
    pert = Perturbation.normal(kappa_of(presc[0]), sigma=0.05, name='k')
    result = monte_carlo(presc, [pert], merit, n_trials=200, seed=7)
    summary = result.summary()
    # mean RMS spot should be measurably larger than the nominal 0 value
    assert summary['mean'] > 1e-4
    # nominal kappa is restored
    assert presc[0].params['k'] == -1.0
