"""The index-spec resolver: the single owner of the glass/index spec grammar."""

import pytest

from prysm.x.materials import MIRROR, air, vacuum, glass, lookup, resolve_index
from prysm.x.materials import Catalog, ConstantMaterial


def test_resolve_index_sentinels_and_air():
    assert resolve_index(None) is None
    assert resolve_index(MIRROR) is MIRROR
    assert resolve_index('MIRROR') is MIRROR
    assert resolve_index('mirror') is MIRROR
    assert resolve_index('') is air
    assert resolve_index('   ') is air
    assert resolve_index('AIR') is air
    assert resolve_index('vacuum') is air


def test_air_singleton_is_a_material_protocol():
    # air / vacuum are MaterialProtocol singletons, not bare callables: they
    # carry .n (real), .nk (complex), and __call__ aliasing .n.
    assert air.n(0.55) == 1.0
    assert air.nk(0.55) == 1.0 + 0j
    assert air(0.55) == 1.0
    assert vacuum.n(0.55) == 1.0
    assert vacuum.nk(0.55) == 1.0 + 0j
    assert vacuum(0.55) == 1.0


def test_resolve_index_numbers_and_callables():
    assert resolve_index(1.5)(0.55) == 1.5
    # complex index is preserved (a complex-aware caller keeps n + 1j*k)
    assert resolve_index(1.2 + 0.3j)(0.55) == 1.2 + 0.3j
    f = lambda wvl: 2.0
    assert resolve_index(f) is f
    material = ConstantMaterial(1.7, name='glass')
    assert resolve_index(material) is material


def test_resolve_index_name_requires_resolver():
    # a glass name needs a catalog; without one it refuses rather than guessing
    with pytest.raises(TypeError, match='without a catalog'):
        resolve_index('N-BK7')
    catalog = Catalog.from_materials([ConstantMaterial(1.5168, name='N-BK7')])
    resolved = resolve_index('N-BK7', name_resolver=catalog.material_for_name)
    assert resolved.n(0.55) == pytest.approx(1.5168)


def test_lookup_projects_blank_to_air_and_resolves_names():
    catalog = Catalog.from_materials([ConstantMaterial(1.5168, name='N-BK7')])
    assert lookup(None) is air
    assert lookup('') is air
    assert lookup('AIR') is air
    assert lookup('MIRROR') is MIRROR
    assert lookup('N-BK7', database=catalog).n(0.55) == pytest.approx(1.5168)
