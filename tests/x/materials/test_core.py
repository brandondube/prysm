import pytest

from prysm.mathops import np
from prysm.conf import config
from prysm.x.materials import (
    ConstantMaterial,
    FormulaMaterial,
    MaterialRangeError,
    MissingKError,
)


def test_constant_material_scalar_vector_and_nk():
    material = ConstantMaterial('absorber', 1.5, k=0.02)
    assert material(0.55) == pytest.approx(1.5)
    np.testing.assert_allclose(material.n([0.5, 0.6]), [1.5, 1.5])
    assert material.k(0.55) == pytest.approx(0.02)
    assert material.nk(0.55) == pytest.approx(1.5 + 0.02j)


def test_missing_k_policies_are_explicit():
    transparent = ConstantMaterial('transparent', 1.5, missing_k='zero')
    assert transparent.k(0.55) == pytest.approx(0)
    absorbing_unknown = ConstantMaterial('unknown', 1.5, missing_k='raise')
    with pytest.raises(MissingKError):
        absorbing_unknown.k(0.55)


def test_wavelength_and_temperature_ranges_raise_by_default():
    material = ConstantMaterial(
        'limited',
        1.5,
        wavelength_range=(0.4, 0.8),
        temperature_range=(80, 300),
    )
    with pytest.raises(MaterialRangeError, match='wavelength'):
        material.n(0.3)
    with pytest.raises(MaterialRangeError, match='temperature'):
        material.n(0.55, temperature=20)


def test_formula_material_supports_metrics():
    material = FormulaMaterial(
        'linear',
        lambda w, a, b: a + b * w,
        (1.4, 0.2),
        wavelength_range=(0.4, 0.8),
    )
    assert material.n(0.5) == pytest.approx(1.5)
    assert material.dispersion(0.6, 0.5) == pytest.approx(0.02)
    assert material.dn_dlambda(0.5) == pytest.approx(0.2, rel=1e-6)


def test_formula_material_forwards_temperature_when_callable_accepts_it():
    def dispersion(wvl_um, base, slope, temperature=None):
        n = base + slope * wvl_um
        if temperature is not None:
            n = n + 1e-3 * (temperature - 300)
        return n

    material = FormulaMaterial('thermo', dispersion, (1.4, 0.2))
    # no temperature -> base behavior
    assert material.n(0.5) == pytest.approx(1.5)
    # temperature flows into the callable
    assert material.n(0.5, temperature=400) == pytest.approx(1.5 + 1e-3 * 100)


def test_formula_material_does_not_force_temperature_on_plain_callables():
    material = FormulaMaterial('plain', lambda w, a, b: a + b * w, (1.4, 0.2))
    # a wavelength-only formula is unaffected by a temperature argument
    assert material.n(0.5, temperature=400) == pytest.approx(1.5)


def test_dn_dlambda_defined_at_range_boundary():
    # central differencing would step out of range at a closed band edge; the
    # one-sided fallback keeps the derivative finite and correct.
    material = FormulaMaterial(
        'linear', lambda w, a, b: a + b * w, (1.4, 0.2), wavelength_range=(0.4, 0.8),
    )
    assert material.dn_dlambda(0.8) == pytest.approx(0.2, rel=1e-6)
    assert material.dn_dlambda(0.4) == pytest.approx(0.2, rel=1e-6)


def test_dn_dlambda_zero_width_range_is_zero_not_nan():
    # a degenerate single-point range collapses the difference interval; the
    # derivative of a locally-constant sample is 0, not 0/0 = nan.
    material = FormulaMaterial('z', lambda w, a: a + 0 * w, (1.5,), wavelength_range=(0.5, 0.5))
    assert float(material.dn_dlambda(0.5)) == pytest.approx(0.0)


def test_dn_dT_single_temperature_grid_is_zero_not_nan():
    from prysm.x.materials import TemperatureGridMaterial

    # one temperature -> zero-width temperature_range; dn_dT must be 0, not nan.
    material = TemperatureGridMaterial(
        'g', [0.5, 1.0], [300], [[1.5, 1.6]], layout=('temperature', 'wavelength'),
    )
    assert float(material.dn_dT(0.75, 300)) == pytest.approx(0.0)


def test_constant_material_plain_sequence_uses_config_precision():
    old_precision = config.precision
    try:
        config.precision = np.float32
        material = ConstantMaterial('constant', 1.5, k=0.01)

        assert material.n([0.5, 0.6]).dtype == np.dtype(np.float32)
        assert material.k([0.5, 0.6]).dtype == np.dtype(np.float32)
    finally:
        config.precision = old_precision
