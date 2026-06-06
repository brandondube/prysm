from pathlib import Path

import pytest

from prysm.mathops import np
from prysm.x.materials import AGFCatalog, AGFMaterial, AmbiguousMaterialError


DATA = Path(__file__).parents[1] / 'raytracing' / 'data' / 'materials'


def agf_catalog():
    return AGFCatalog.from_files([
        DATA / 'tiny_schott.agf',
        DATA / 'tiny_ohara.agf',
    ])


def test_agf_catalog_parses_materials_and_metadata():
    catalog = AGFCatalog.from_file(DATA / 'tiny_schott.agf')
    material = catalog.material_for_name('N-BK7')
    assert material.name == 'N-BK7'
    assert material.catalog == 'SCHOTT'
    assert material.page_info['page'] == 'N-BK7'
    assert material.k(0.55) == pytest.approx(0)


def test_agf_sellmeier_scalar_vector_and_range():
    material = AGFCatalog.from_file(DATA / 'tiny_schott.agf').material_for_name('N-BK7')
    assert material.n(0.5875618) == pytest.approx(1.5168000345)
    n = material.n(np.array([0.4861327, 0.6562725]))
    np.testing.assert_allclose(n, [1.52237629, 1.51432235], rtol=1e-6)
    with pytest.raises(ValueError, match='outside AGF range'):
        material.n(0.25)


def test_agf_from_file_accepts_utf16_and_keeps_extended_metadata(tmp_path):
    path = tmp_path / 'utf16.agf'
    text = """\
CC UTF-16 test catalog
NM TEST 1 0 1.500000 50.0 0
GC test glass
CD 2.25 0 0 0 0 0
MD 82.00 0.21 580 820.000 1.19
BD 0.588 2.77 0.80 3.57
LD 0.4 0.8
"""
    path.write_bytes(text.encode('utf-16'))
    material = AGFCatalog.from_file(path, namespace='TESTCAT').material_for_name('TEST')
    assert material.n(0.55) == pytest.approx(1.5)
    assert material.metadata['MD'] == ('82.00 0.21 580 820.000 1.19',)
    assert material.metadata['BD'] == ('0.588 2.77 0.80 3.57',)


def test_agf_formula_subset():
    material = AGFMaterial(
        name='SAMPLE',
        catalog='HIKARI',
        formula=13,
        coefficients=(
            2.45448839, -0.00867148963,
            -0.00010471524, 0.0176039752,
            0.000154610243, 0.0000559918259,
            -0.00000501297284, 0.00000031755799,
            0, 0,
        ),
    )
    assert material.n(0.5875618) == pytest.approx(1.582670, abs=1e-6)


def test_agf_ambiguous_lookup_is_explicit():
    text = """\
NM N-BK7 1
CD 2.25 0 0 0 0 0
LD 0.4 0.8
"""
    one = AGFCatalog.from_text(text, namespace='ONE')
    two = AGFCatalog.from_text(text, namespace='TWO')
    from prysm.x.materials import CatalogChain

    chain = CatalogChain([one, two])
    with pytest.raises(AmbiguousMaterialError):
        chain.material_for_name('N-BK7')
    assert chain['ONE:N-BK7'].n(0.55) == pytest.approx(1.5)
