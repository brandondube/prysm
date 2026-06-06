import pytest

from prysm.x.materials import (
    ConstantMaterial,
    IndexOffsetMaterial,
    TemperatureGridMaterial,
    TemperatureShiftedMaterial,
)


def test_material_correction_receives_temperature():
    parent = TemperatureGridMaterial(
        'base',
        [0.5, 1.0],
        [100, 300],
        [
            [1.5, 1.5],
            [1.5, 1.5],
        ],
    )
    correction = TemperatureGridMaterial(
        'dn_dT',
        [0.5, 1.0],
        [100, 300],
        [
            [1e-3, 2e-3],
            [3e-3, 4e-3],
        ],
    )
    shifted = TemperatureShiftedMaterial(parent, correction, reference_temperature=100)

    assert shifted.n(0.75, temperature=200) == pytest.approx(1.75)


def test_callable_correction_typeerror_is_not_masked():
    def correction(wvl_um, temperature):
        raise TypeError('internal failure')

    material = IndexOffsetMaterial(ConstantMaterial('base', 1.5), correction)

    with pytest.raises(TypeError, match='internal failure'):
        material.n(0.55, temperature=300)
