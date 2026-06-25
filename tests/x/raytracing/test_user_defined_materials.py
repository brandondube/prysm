"""Tests for first-class user-defined raytracing materials."""

import numpy as np
import pytest

from prysm.x.raytracing import LensData, surface_table
from prysm.x import materials
from prysm.x.raytracing.io import write_seq, write_zmx
from prysm.x.raytracing.surfaces import Conic, Plane


def test_material_objects_compile_through_lensdata_verbatim():
    # Refractive surfaces consume MaterialProtocol objects directly: the object
    # is carried to the compiled surface verbatim (identity holds, no wrapping).
    # Bare numbers / callables and glass-name strings are intentionally not
    # accepted -- resolve them to a material object first.
    mat = materials.ConstantMaterial(1.5, name='CONST')
    other = materials.ConstantMaterial(1.6, name='OTHER')

    ld = (LensData()
          .add(Plane(), thickness=1.0, material=mat)
          .add(Plane(), thickness=1.0, material=other)
          .add(Plane(), typ='eval'))
    assert ld.surfaces[1].material is mat      # surfaces[0] is the OBJECT plane
    assert ld.surfaces[1].material.n(0.55) == pytest.approx(1.5)
    assert ld.surfaces[2].material is other


def test_tabulated_material_scalar_vector_and_linear_interpolation():
    mat = materials.TabulatedMaterial(
        name='MYGLASS',
        wavelengths=[0.5, 0.6, 0.7],
        n=[1.6, 1.5, 1.4],
        method='linear',
    )
    assert mat(0.55) == pytest.approx(1.55)
    np.testing.assert_allclose(mat(np.array([0.55, 0.65])), [1.55, 1.45])
    assert mat.n(0.6) == pytest.approx(1.5)
    assert mat.wavelength_range == pytest.approx((0.5, 0.7))


def test_tabulated_material_rejects_out_of_range_unless_extrapolated():
    mat = materials.TabulatedMaterial('MYGLASS', [0.5, 0.6, 0.7],
                                      [1.6, 1.5, 1.4])
    with pytest.raises(ValueError, match='outside material range'):
        mat(0.45)

    extrapolated = materials.TabulatedMaterial(
        'MYGLASS', [0.5, 0.6, 0.7], [1.6, 1.5, 1.4], extrapolate=True,
    )
    assert extrapolated(0.4) == pytest.approx(1.7)
    assert extrapolated.metadata['extrapolate'] is True


def test_tabulated_material_rejects_bad_wavelength_data():
    with pytest.raises(ValueError, match='strictly increasing'):
        materials.TabulatedMaterial('BAD', [0.5, 0.5], [1.5, 1.6])
    with pytest.raises(ValueError, match='strictly increasing'):
        materials.TabulatedMaterial('BAD', [0.6, 0.5], [1.5, 1.6])
    with pytest.raises(ValueError, match='positive'):
        materials.TabulatedMaterial('BAD', [0.0, 0.5], [1.5, 1.6])


def test_k_interpolation_and_nk_work_for_scalars_and_vectors():
    mat = materials.TabulatedMaterial(
        name='ABSORBING',
        wavelengths=[0.5, 0.6, 0.7],
        n=[1.6, 1.5, 1.4],
        k=[1e-5, 3e-6, 1e-6],
        k_interpolation='log',
    )
    expected_k = np.exp((np.log(1e-5) + np.log(3e-6)) / 2)
    assert mat.k(0.55) == pytest.approx(expected_k)
    assert mat.nk(0.55) == pytest.approx(1.55 + 1j * expected_k)

    nk = mat.nk(np.array([0.5, 0.7]))
    np.testing.assert_allclose(nk.real, [1.6, 1.4])
    np.testing.assert_allclose(nk.imag, [1e-5, 1e-6])

    no_k = materials.TabulatedMaterial('NOABS', [0.5, 0.6],
                                       [1.5, 1.4])
    np.testing.assert_allclose(no_k.k(np.array([0.5, 0.6])), [0.0, 0.0])
    no_k_raise = materials.TabulatedMaterial(
        'NOABS', [0.5, 0.6], [1.5, 1.4], missing_k='raise',
    )
    with pytest.raises(ValueError, match='has no k samples'):
        no_k_raise.k(0.55)


def test_log_k_zero_policy_is_explicit_and_negative_k_is_rejected():
    with pytest.raises(ValueError, match='nonnegative'):
        materials.TabulatedMaterial('BADK', [0.5, 0.6], [1.5, 1.4],
                                    k=[1e-6, -1e-6])

    with pytest.raises(ValueError, match='positive k samples'):
        materials.TabulatedMaterial(
            'ZEROK', [0.5, 0.6], [1.5, 1.4], k=[0.0, 1e-6],
            k_interpolation='log',
        )

    explicit = materials.TabulatedMaterial(
        'ZEROK', [0.5, 0.6, 0.7], [1.5, 1.4, 1.3],
        k=[0.0, 1e-6, 2e-6], k_interpolation='log',
        k_zero_policy='linear',
    )
    assert explicit.k(0.55) == pytest.approx(0.5e-6)


def test_cauchy_fit_recovers_synthetic_data_and_enforces_domain():
    wvl = np.array([0.45, 0.5, 0.6, 0.7, 0.8])
    n = 1.5 + 0.01 / wvl ** 2 + 0.001 / wvl ** 4
    mat = materials.FittedMaterial.from_samples(
        name='CAUCHY',
        wavelengths=wvl,
        n=n,
        model='cauchy',
        terms=3,
        max_abs_error=1e-12,
    )
    np.testing.assert_allclose(mat.coefficients, [1.5, 0.01, 0.001],
                               atol=1e-12)
    np.testing.assert_allclose(mat(wvl), n, atol=1e-12)
    assert mat.fit_report.model == 'cauchy'
    assert mat.fit_report.parameter_count == 3
    assert mat.fit_report.degrees_of_freedom == 2
    with pytest.raises(ValueError, match='outside material range'):
        mat(0.9)

    extrapolated = materials.FittedMaterial.from_samples(
        'CAUCHY', wvl, n, model='cauchy', terms=3, extrapolate=True,
    )
    assert extrapolated(0.9) == pytest.approx(1.5 + 0.01 / 0.9 ** 2
                                              + 0.001 / 0.9 ** 4)


def test_underdetermined_high_order_fit_is_rejected():
    with pytest.raises(ValueError, match='underdetermined'):
        materials.FittedMaterial.from_samples(
            'SELL', [0.5, 0.6, 0.7], [1.5, 1.49, 1.48],
            model='sellmeier1', terms=2,
        )
    with pytest.raises(ValueError, match='underdetermined'):
        materials.FittedMaterial.from_samples(
            'SCHOTT', [0.45, 0.5, 0.6, 0.7, 0.8],
            [1.53, 1.52, 1.51, 1.50, 1.49], model='schott',
        )


def test_fitted_material_direct_coefficients_infer_terms():
    mat = materials.FittedMaterial(
        'DIRECT', 'cauchy', [1.5, 0.01, 0.001],
        wavelength_range=(0.4, 0.8),
    )
    assert mat.terms == 3
    assert mat(0.5) == pytest.approx(1.5 + 0.01 / 0.5 ** 2
                                     + 0.001 / 0.5 ** 4)


def test_overdetermined_fit_reports_residuals_and_threshold_failures():
    wvl = np.linspace(0.45, 0.8, 9)
    clean = 1.5 + 0.01 / wvl ** 2
    n = clean + np.array([0.0, 2e-5, -1e-5, 1e-5, 0.0,
                          -2e-5, 1e-5, 0.0, -1e-5])
    mat = materials.FittedMaterial.from_samples(
        'NOISY', wvl, n, model='cauchy', terms=2,
        max_abs_error=1e-3, rms_error=1e-3,
    )
    report = mat.fit_report
    assert report.sample_count == 9
    assert report.parameter_count == 2
    assert report.degrees_of_freedom == 7
    assert report.residuals.shape == (9,)
    assert report.max_abs_error > 0
    assert report.rms_error > 0

    with pytest.raises(ValueError, match='max_abs_error'):
        materials.FittedMaterial.from_samples(
            'NOISY', wvl, n, model='cauchy', terms=2,
            max_abs_error=1e-12,
        )


def test_page_info_supports_listing_and_best_effort_io_names():
    mat = materials.TabulatedMaterial(
        'USERGLASS', [0.5, 0.6, 0.7], [1.6, 1.5, 1.4],
    )
    ld = (LensData()
          .add(Conic(0.01, 0.0), thickness=1.0, material=mat)
          .add(Plane(), typ='eval'))
    assert surface_table(ld).records[1]['material'] == 'USERGLASS'  # [0] is OBJECT
    assert 'GLAS USERGLASS' in write_zmx(ld)
    assert 'GLA USERGLASS' in write_seq(ld)


def test_convenience_constructors_create_expected_materials():
    tab = materials.from_samples('TAB', [0.5, 0.6], [1.5, 1.4])
    assert isinstance(tab, materials.TabulatedMaterial)
    assert tab(0.55) == pytest.approx(1.45)

    fit = materials.fit_material(
        'FIT', [0.5, 0.6, 0.7], [1.5, 1.49, 1.48],
        model='cauchy', terms=2,
    )
    assert isinstance(fit, materials.FittedMaterial)
    assert fit.fit_report.sample_count == 3
