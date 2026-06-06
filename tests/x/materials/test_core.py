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


def test_constant_material_plain_sequence_uses_config_precision():
    old_precision = config.precision
    try:
        config.precision = np.float32
        material = ConstantMaterial('constant', 1.5, k=0.01)

        assert material.n([0.5, 0.6]).dtype == np.dtype(np.float32)
        assert material.k([0.5, 0.6]).dtype == np.dtype(np.float32)
    finally:
        config.precision = old_precision
