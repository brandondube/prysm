import pytest

from prysm.mathops import np
from prysm.conf import config
from prysm.x.materials import MaterialRangeError, TabulatedMaterial, TemperatureGridMaterial


def test_tabulated_material_interpolates_n_and_k():
    material = TabulatedMaterial(
        'film',
        [0.4, 0.6, 0.8],
        [1.4, 1.5, 1.6],
        k=[1e-4, 1e-3, 1e-2],
        k_interpolation='log',
    )
    assert material.n(0.5) == pytest.approx(1.45)
    assert material.k(0.5) == pytest.approx(np.sqrt(1e-7))
    nk = material.nk([0.4, 0.8])
    np.testing.assert_allclose(nk, [1.4 + 1e-4j, 1.6 + 1e-2j])


def test_tabulated_material_supports_nearest_and_range_checks():
    material = TabulatedMaterial(
        'nearest',
        [0.4, 0.6, 0.8],
        [1.4, 1.5, 1.6],
        interpolation='nearest',
    )
    assert material.n(0.51) == pytest.approx(1.5)
    with pytest.raises(MaterialRangeError):
        material.n(0.2)


def test_tabulated_material_linear_extrapolates_when_enabled():
    material = TabulatedMaterial(
        'linear',
        [1.0, 2.0],
        [1.0, 3.0],
        extrapolate=True,
    )
    assert material.n(3.0) == pytest.approx(5.0)
    assert material.n(0.0) == pytest.approx(-1.0)


def test_temperature_grid_interpolates_wavelength_and_temperature():
    wavelengths = [0.5, 1.0]
    temperatures = [100, 300]
    # n = 1 + wavelength + temperature / 1000
    n = [
        [1.6, 2.1],
        [1.8, 2.3],
    ]
    material = TemperatureGridMaterial('grid', wavelengths, temperatures, n)
    assert material.n(0.75, temperature=200) == pytest.approx(1.95)
    np.testing.assert_allclose(
        material.n([0.5, 1.0], temperature=100),
        [1.6, 2.1],
    )
    np.testing.assert_allclose(
        material.k(0.75, temperature=[100, 300]),
        [0, 0],
    )


def test_temperature_grid_extrapolates_wavelength_and_temperature_when_enabled():
    wavelengths = [1.0, 2.0]
    temperatures = [10.0, 20.0]
    # n = wavelength + temperature / 10
    n = [
        [2.0, 3.0],
        [3.0, 4.0],
    ]
    material = TemperatureGridMaterial(
        'grid',
        wavelengths,
        temperatures,
        n,
        extrapolate=True,
    )
    assert material.n(3.0, temperature=30.0) == pytest.approx(6.0)


def test_temperature_grid_uses_derivative_grids_or_finite_difference():
    wavelengths = [0.5, 1.0]
    temperatures = [100, 300]
    n = [
        [1.6, 2.1],
        [1.8, 2.3],
    ]
    material = TemperatureGridMaterial(
        'grid',
        wavelengths,
        temperatures,
        n,
        dn_dT=[
            [1e-3, 1e-3],
            [1e-3, 1e-3],
        ],
    )
    assert material.dn_dT(0.75, 200) == pytest.approx(1e-3)
    assert material.dn_dlambda(0.75, temperature=200) == pytest.approx(1.0, rel=1e-6)


def test_tabulated_material_respects_config_precision_and_query_dtype():
    old_precision = config.precision
    try:
        config.precision = np.float32
        material = TabulatedMaterial('film', [0.4, 0.6, 0.8], [1.4, 1.5, 1.6])

        assert material.wavelengths.dtype == np.dtype(np.float32)
        assert material.n([0.5]).dtype == np.dtype(np.float32)
        assert material.n(np.array([0.5], dtype=np.float32)).dtype == np.dtype(np.float32)
        assert material.n(np.array([0.5], dtype=np.float64)).dtype == np.dtype(np.float64)
    finally:
        config.precision = old_precision


def test_temperature_grid_material_respects_config_precision():
    old_precision = config.precision
    try:
        config.precision = np.float32
        material = TemperatureGridMaterial(
            'grid',
            [0.5, 1.0],
            [100, 300],
            [
                [1.6, 2.1],
                [1.8, 2.3],
            ],
        )

        assert material.n_grid.dtype == np.dtype(np.float32)
        assert material.n(0.75, temperature=200).dtype == np.dtype(np.float32)
    finally:
        config.precision = old_precision
import pytest

from prysm.x.materials import (
    AmbiguousMaterialError,
    Catalog,
    CatalogChain,
    ConstantMaterial,
    MaterialRegistry,
    TabulatedMaterial,
)


def test_catalog_chain_namespace_lookup_and_ambiguity():
    schott = Catalog.from_materials([
        ConstantMaterial('N-BK7', 1.5, catalog='SCHOTT', metadata={'aliases': ('BK7',)}),
    ])
    ohara = Catalog.from_materials([
        ConstantMaterial('S-BSL7', 1.52, catalog='OHARA', metadata={'aliases': ('BK7',)}),
    ])
    chain = CatalogChain([schott, ohara])
    assert chain['SCHOTT:N-BK7'].n(0.55) == pytest.approx(1.5)
    with pytest.raises(AmbiguousMaterialError):
        chain.material_for_name('BK7')


def test_registry_metadata_and_computed_search():
    low = TabulatedMaterial(
        'low',
        [0.4, 0.8],
        [1.45, 1.46],
        k=[0, 0],
        catalog='LAB',
        process='IBS',
    )
    high = TabulatedMaterial(
        'high',
        [0.4, 0.8],
        [2.0, 2.1],
        k=[0.1, 0.1],
        catalog='LAB',
        process='ebeam',
    )
    registry = MaterialRegistry.from_catalogs(Catalog.from_materials([low, high]))
    records = registry.search(
        wavelength_range_contains=(0.45, 0.65),
        process='IBS',
        n_at=(0.55, 1.44, 1.47),
        k_max=(0.55, 1e-6),
    )
    assert [record.name for record in records] == ['low']


def test_registry_uses_catalog_matching_semantics():
    material = ConstantMaterial(
        'N-BK7',
        1.5,
        catalog='SCHOTT',
        process='IBS',
        metadata={'aliases': ('BK7',)},
    )
    registry = MaterialRegistry.from_catalogs(Catalog.from_materials([material]))

    assert [record.name for record in registry.search(query='N BK7')] == ['N-BK7']
    assert [record.name for record in registry.search(process='ibs')] == ['N-BK7']
    assert [record.name for record in registry.search(catalog='schott')] == ['N-BK7']


def test_registry_computed_criteria_validate_arity():
    registry = MaterialRegistry.from_catalogs(Catalog.from_materials([
        ConstantMaterial('glass', 1.5),
    ]))

    with pytest.raises(ValueError, match='n_at criterion expects'):
        registry.search(n_at=(0.55,))
    with pytest.raises(ValueError, match='n_at criterion must be a sequence'):
        registry.search(n_at=0.55)
    with pytest.raises(ValueError, match='k_max criterion expects'):
        registry.search(k_max=(0.55, 1e-6, None, 'extra'))
