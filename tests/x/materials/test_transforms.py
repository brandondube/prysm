import pytest

from prysm.x.materials import (
    ConstantMaterial,
    IndexOffsetMaterial,
    StressOpticMaterial,
    TemperatureGridMaterial,
    TemperatureShiftedMaterial,
)


def test_correction_accepts_scalar_and_wavelength_only_callable():
    base = ConstantMaterial('base', 1.5)
    # a bare scalar correction is a constant offset
    assert IndexOffsetMaterial(base, 0.01).n(0.55) == pytest.approx(1.51)
    # a wavelength-only callable is bound to its (wvl) shape, not handed a temperature
    sloped = IndexOffsetMaterial(base, lambda wvl: 0.1 * wvl)
    assert sloped.n(0.5, temperature=300) == pytest.approx(1.55)
    # a (wvl, temperature) positional callable receives the query temperature
    stressed = StressOpticMaterial(base, lambda wvl, temperature: temperature * 1e-4, stress=2.0)
    assert stressed.n(0.5, temperature=300) == pytest.approx(1.5 + 300 * 1e-4 * 2.0)


def test_material_correction_receives_temperature():
    parent = TemperatureGridMaterial(
        'base',
        [0.5, 1.0],
        [100, 300],
        [
            [1.5, 1.5],
            [1.5, 1.5],
        ],
        layout=('temperature', 'wavelength'),
    )
    correction = TemperatureGridMaterial(
        'dn_dT',
        [0.5, 1.0],
        [100, 300],
        [
            [1e-3, 2e-3],
            [3e-3, 4e-3],
        ],
        layout=('temperature', 'wavelength'),
    )
    shifted = TemperatureShiftedMaterial(parent, correction, reference_temperature=100)

    assert shifted.n(0.75, temperature=200) == pytest.approx(1.75)


def test_callable_correction_typeerror_is_not_masked():
    def correction(wvl_um, temperature):
        raise TypeError('internal failure')

    material = IndexOffsetMaterial(ConstantMaterial('base', 1.5), correction)

    with pytest.raises(TypeError, match='internal failure'):
        material.n(0.55, temperature=300)
