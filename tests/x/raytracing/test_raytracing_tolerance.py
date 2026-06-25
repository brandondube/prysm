"""Tests for the tolerancing layer (perturbations over LensData DOF slots)."""
import numpy as np
import pytest

from prysm.x import materials
from prysm.x.raytracing import OpticalSystem
from prysm.x.raytracing import LensData
from prysm.x.raytracing.launch import Sampling
from prysm.x.raytracing.design import RmsSpotRadius
from prysm.x.raytracing.surfaces import Conic, Plane
from prysm.x.raytracing.tolerance import (
    Perturbation, sensitivity_table, monte_carlo, operand_as_merit,
)


_n_glass = materials.ConstantMaterial(1.5)


# ---------- helpers --------------------------------------------------------

def _spherical_singlet():
    # OBJECT/IMAGE endpoints implicit (ADR-0006); the conics are rows 1 and 2.
    lens = LensData()
    (lens.add(Conic(1 / 50.0, 0.0), typ='refr', thickness=5.0,
              material=_n_glass)
         .add(Conic(-1 / 50.0, 0.0), typ='refr', thickness=95.0,
              material=materials.air))
    return OpticalSystem(lens, aperture=10.0, wavelengths=[0.55e-3])


def _concave_parabola():
    c = -1 / 80.0
    f = abs(1.0 / (2.0 * c))  # 40; the fold places the image at -f
    lens = LensData()
    lens.add(Conic(c, -1.0), typ='refl', thickness=f)   # the mirror is row 1
    return OpticalSystem(lens, aperture=10.0, wavelengths=[0.55e-3])


# ---------- Perturbation factories ----------------------------------------

def test_perturbation_normal_captures_nominal_and_sigma():
    ld = _spherical_singlet()
    p = Perturbation.normal(ld, 'curvature', 1, sigma=1e-4, name='c1')
    assert p.name == 'c1'
    np.testing.assert_allclose(p.nominal, 1 / 50.0)
    assert p.step == 1e-4


def test_perturbation_normal_relative_scales_with_nominal():
    ld = _spherical_singlet()
    p = Perturbation.normal_relative(ld, 'curvature', 1, sigma_rel=0.001,
                                     name='c1')
    np.testing.assert_allclose(p.step, abs(1 / 50.0) * 0.001)


def test_perturbation_uniform_half_width():
    ld = _spherical_singlet()
    p = Perturbation.uniform(ld, 'conic', 1, half_width=0.05, name='k1')
    assert p.step == 0.05
    assert p.nominal == 0.0


def test_perturbation_triangular_uses_nominal_as_peak():
    ld = _spherical_singlet()
    rng = np.random.default_rng(0)
    p = Perturbation.triangular(ld, 'conic', 1, half_width=0.1, name='k1')
    samples = np.array([p.sample(rng) for _ in range(5000)])
    np.testing.assert_allclose(samples.mean(), 0.0, atol=2e-3)
    assert float(samples.min()) >= -0.1
    assert float(samples.max()) <= 0.1


def test_perturbation_sample_normal_centers_on_nominal():
    ld = _spherical_singlet()
    p = Perturbation.normal(ld, 'conic', 1, sigma=0.05, name='k1')
    rng = np.random.default_rng(42)
    samples = np.array([p.sample(rng) for _ in range(10000)])
    np.testing.assert_allclose(samples.mean(), 0.0, atol=2e-3)
    np.testing.assert_allclose(samples.std(), 0.05, rtol=0.05)


def test_perturbation_reset_restores_value():
    ld = _spherical_singlet()
    p = Perturbation.normal(ld, 'conic', 1, sigma=0.05, name='k1')
    p.set(0.5)
    assert ld.surfaces[1].params['k'] == 0.5
    p.reset()
    assert ld.surfaces[1].params['k'] == 0.0


def test_perturbation_target_must_be_single_dof():
    ld = _spherical_singlet()
    with pytest.raises(ValueError):
        # 'curvature' over 'all' resolves to multiple DOFs
        Perturbation.normal(ld, 'curvature', 'all', sigma=1e-4)


# ---------- operand_as_merit ----------------------------------------------

def test_operand_as_merit_runs():
    ld = _concave_parabola()
    op = RmsSpotRadius(wavelength=0.55e-3, sampling=Sampling.fan(n=11))
    merit = operand_as_merit(op)
    val = merit(ld)
    assert val >= 0.0
    assert val < 1e-9


# ---------- sensitivity_table ----------------------------------------------

def test_sensitivity_table_zero_step_yields_zero_sensitivity():
    ld = _concave_parabola()
    pert = Perturbation.normal(ld, 'conic', 1, sigma=0.0, name='k')
    merit = lambda p: float(p[1].params['k'])
    table = sensitivity_table(ld, [pert], merit)
    assert table.rows[0]['sensitivity'] == 0.0


def test_sensitivity_table_matches_analytic_for_linear_merit():
    ld = _concave_parabola()
    pert = Perturbation.normal(ld, 'conic', 1, sigma=1e-3, name='k')
    merit = lambda p: 3.7 * float(p[1].params['k']) + 2.0
    table = sensitivity_table(ld, [pert], merit)
    np.testing.assert_allclose(table.rows[0]['sensitivity'], 3.7, rtol=1e-12)


def test_sensitivity_table_restores_parameter():
    ld = _concave_parabola()
    nominal_k = ld.surfaces[1].params['k']
    nominal_c = ld.surfaces[1].params['c']
    perts = [
        Perturbation.normal(ld, 'conic', 1, sigma=0.1, name='k'),
        Perturbation.normal(ld, 'curvature', 1, sigma=1e-4, name='c'),
    ]
    merit = lambda p: float(p[1].params['c'])
    _ = sensitivity_table(ld, perts, merit)
    assert ld.surfaces[1].params['k'] == nominal_k
    assert ld.surfaces[1].params['c'] == nominal_c


def test_sensitivity_table_records_per_row_deltas():
    ld = _concave_parabola()
    pert = Perturbation.normal(ld, 'conic', 1, sigma=0.01, name='k')
    merit = lambda p: 2.0 * float(p[1].params['k'])
    table = sensitivity_table(ld, [pert], merit)
    row = table.rows[0]
    np.testing.assert_allclose(row['delta_plus'], 0.02, rtol=1e-12)
    np.testing.assert_allclose(row['delta_minus'], -0.02, rtol=1e-12)


def test_sensitivity_table_global_step_overrides_per_row():
    ld = _concave_parabola()
    pert = Perturbation.normal(ld, 'conic', 1, sigma=0.01, name='k')
    merit = lambda p: float(p[1].params['k'])
    table = sensitivity_table(ld, [pert], merit, step=0.1)
    assert table.rows[0]['step'] == 0.1


def test_sensitivity_table_repr_lists_each_row():
    ld = _concave_parabola()
    perts = [
        Perturbation.normal(ld, 'conic', 1, sigma=0.05, name='k'),
        Perturbation.normal(ld, 'curvature', 1, sigma=1e-4, name='c'),
    ]
    merit = lambda p: float(p[1].params['k']) + float(p[1].params['c'])
    table = sensitivity_table(ld, perts, merit)
    s = repr(table)
    assert 'SensitivityTable' in s
    assert 'k' in s
    assert 'c' in s


# ---------- monte_carlo ----------------------------------------------------

def test_monte_carlo_zero_sigma_returns_nominal_merit():
    ld = _spherical_singlet()
    perts = [Perturbation.normal(ld, 'conic', 1, sigma=0.0, name='k')]
    merit = lambda p: float(p[1].params['k'])
    result = monte_carlo(ld, perts, merit, n_trials=50, seed=0)
    np.testing.assert_array_equal(result.merits, np.zeros(50))


def test_monte_carlo_reproducible_with_seed():
    ld = _spherical_singlet()
    perts = [Perturbation.normal(ld, 'conic', 1, sigma=0.01, name='k')]
    merit = lambda p: float(p[1].params['k'])
    r1 = monte_carlo(ld, perts, merit, n_trials=100, seed=12345)
    r2 = monte_carlo(ld, perts, merit, n_trials=100, seed=12345)
    np.testing.assert_array_equal(r1.merits, r2.merits)


def test_monte_carlo_different_seeds_differ():
    ld = _spherical_singlet()
    perts = [Perturbation.normal(ld, 'conic', 1, sigma=0.01, name='k')]
    merit = lambda p: float(p[1].params['k'])
    r1 = monte_carlo(ld, perts, merit, n_trials=100, seed=1)
    r2 = monte_carlo(ld, perts, merit, n_trials=100, seed=2)
    assert not np.array_equal(r1.merits, r2.merits)


def test_monte_carlo_restores_nominal_on_completion():
    ld = _spherical_singlet()
    nominal_k = ld.surfaces[1].params['k']
    perts = [Perturbation.normal(ld, 'conic', 1, sigma=0.1, name='k')]
    merit = lambda p: float(p[1].params['k'])
    _ = monte_carlo(ld, perts, merit, n_trials=20, seed=0)
    assert ld.surfaces[1].params['k'] == nominal_k


def test_monte_carlo_restores_nominal_on_merit_exception():
    ld = _spherical_singlet()
    nominal_k = ld.surfaces[1].params['k']
    perts = [Perturbation.normal(ld, 'conic', 1, sigma=0.1, name='k')]
    call_counter = {'n': 0}

    def merit(p):
        call_counter['n'] += 1
        if call_counter['n'] == 3:
            raise RuntimeError('synthetic failure mid-trial')
        return float(p[1].params['k'])

    with pytest.raises(RuntimeError):
        monte_carlo(ld, perts, merit, n_trials=20, seed=0)
    assert ld.surfaces[1].params['k'] == nominal_k


def test_monte_carlo_summary_keys():
    ld = _spherical_singlet()
    perts = [Perturbation.normal(ld, 'conic', 1, sigma=0.01, name='k')]
    merit = lambda p: float(p[1].params['k']) ** 2
    result = monte_carlo(ld, perts, merit, n_trials=200, seed=0)
    summary = result.summary()
    for k in ('n_trials', 'min', 'max', 'mean', 'std',
              'median', 'p95', 'p99'):
        assert k in summary


def test_monte_carlo_yield_at_threshold_edges():
    ld = _spherical_singlet()
    perts = [Perturbation.normal(ld, 'conic', 1, sigma=0.01, name='k')]
    merit = lambda p: float(p[1].params['k']) ** 2
    result = monte_carlo(ld, perts, merit, n_trials=200, seed=0)
    assert result.yield_at(float('inf')) == 1.0
    assert result.yield_at(float('-inf')) == 0.0


def test_monte_carlo_record_samples_shape():
    ld = _spherical_singlet()
    perts = [
        Perturbation.normal(ld, 'conic', 1, sigma=0.01, name='k'),
        Perturbation.normal(ld, 'curvature', 1, sigma=1e-4, name='c'),
    ]
    merit = lambda p: float(p[1].params['k'])
    result = monte_carlo(ld, perts, merit, n_trials=30, seed=0,
                         record_samples=True)
    assert result.sampled_x.shape == (30, 2)


# ---------- end-to-end optical example ------------------------------------

def test_monte_carlo_on_spherical_aberration_recovery():
    ld = _concave_parabola()
    op = RmsSpotRadius(wavelength=0.55e-3, sampling=Sampling.fan(n=11))
    merit = operand_as_merit(op)
    pert = Perturbation.normal(ld, 'conic', 1, sigma=0.05, name='k')
    result = monte_carlo(ld, [pert], merit, n_trials=200, seed=7)
    summary = result.summary()
    assert summary['mean'] > 1e-4
    assert ld.surfaces[1].params['k'] == -1.0
