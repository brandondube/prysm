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


def test_registry_resolves_by_name_via_shared_record_query():
    # the registry shares the RecordSet query seam, so name resolution and the
    # namespace:name getitem work the same as on a Catalog or a chain.
    registry = MaterialRegistry.from_catalogs(Catalog.from_materials([
        ConstantMaterial('N-BK7', 1.5, catalog='SCHOTT'),
        ConstantMaterial('S-BSL7', 1.52, catalog='OHARA'),
    ]))
    assert registry.material_for_name('N-BK7').n(0.55) == pytest.approx(1.5)
    assert registry['OHARA:S-BSL7'].n(0.55) == pytest.approx(1.52)


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


def test_registry_k_max_treats_missing_k_as_transparent():
    # a missing_k='raise' member must not abort the k_max filter.
    opaque_unknown = ConstantMaterial('X', 2.0, missing_k='raise', catalog='LAB')
    clear = ConstantMaterial('Y', 1.5, missing_k='zero', catalog='LAB')
    registry = MaterialRegistry.from_catalogs(
        Catalog.from_materials([opaque_unknown, clear])
    )
    names = [record.name for record in registry.search(k_max=(0.55, 1e-6))]
    assert set(names) == {'X', 'Y'}
