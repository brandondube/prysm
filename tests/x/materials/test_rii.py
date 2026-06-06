import pytest

from prysm.x.materials import MaterialRangeError, RefractiveIndexCatalog
from prysm.x.materials.rii import default_db_path


def test_default_db_path_is_ri_info_database():
    assert default_db_path().name == '.refractiveindex.info-database'


def test_from_database_requires_existing_folder_without_download(tmp_path):
    with pytest.raises(FileNotFoundError):
        RefractiveIndexCatalog.from_database(db_path=tmp_path, download=False)


def test_rii_formula_material_lookup_and_page_info(rii_catalog):
    material = rii_catalog.material_for_name('N-BK7')
    assert material.n(0.5875618) == pytest.approx(1.5168000345005885, rel=1e-12)
    assert material.k(0.5) == pytest.approx(0.0)
    # ranked to the SCHOTT-optical spec page over the plain BK7 book.
    assert material.page_info['book'] == 'SCHOTT-optical'
    assert material.page_info['page'] == 'N-BK7'
    assert set(material.page_info) == {
        'shelf', 'book', 'page', 'filepath', 'rangeMin', 'rangeMax',
    }


def test_rii_formula_out_of_range_raises(rii_catalog):
    material = rii_catalog.material_for_name('N-BK7')
    with pytest.raises(MaterialRangeError):
        material.n(0.2)


def test_rii_case_insensitive_lookup(rii_catalog):
    assert rii_catalog.material_for_name('n-bk7').n(0.5875618) == pytest.approx(
        1.5168000345005885, rel=1e-12
    )


def test_rii_tabulated_nk_and_qualifier(rii_catalog):
    material = rii_catalog.material_for_name('SiO2', page='Malitson')
    assert material.n(0.5) == pytest.approx(1.45)
    assert material.k(0.5) == pytest.approx(0.001)
    assert material.nk(0.6) == pytest.approx(1.46 + 0.002j)
    assert material.page_info['page'] == 'Malitson'


def test_rii_ambiguous_name_resolves_to_ranked_best(rii_catalog):
    # two SiO2 pages tie on rank; the (shelf, book, page) tiebreak picks Malitson.
    best = rii_catalog.material_for_name('SiO2')
    assert best.page_info['page'] == 'Malitson'
    other = rii_catalog.material_for_name('SiO2', page='Other')
    assert other.n(0.6) == pytest.approx(1.60)


def test_rii_unknown_name_raises(rii_catalog):
    with pytest.raises(KeyError):
        rii_catalog.material_for_name('UNOBTAINIUM')


def test_rii_formula_n_with_tabulated_k_keeps_n_analytic(rii_catalog):
    from prysm.x.materials.core import FormulaMaterial

    material = rii_catalog.material_for_name('HYBRID')
    # n stays an analytic formula instead of being resampled onto the k grid.
    assert isinstance(material, FormulaMaterial)
    # exact Sellmeier value; a 3-point linear interp of n would be far off here.
    assert material.n(0.5875618) == pytest.approx(1.5168000345005885, rel=1e-9)
    # k comes from the tabulated grid: np.interp(0.65, [0.3,1.0,2.5], [0.1,0.2,0.3]).
    assert float(material.k(0.65)) == pytest.approx(0.15)


def test_rii_single_sample_page_is_constant():
    from prysm.x.materials.rii import RefractiveIndexMaterial

    # a one-row tabulated page is a constant index, valid at any wavelength,
    # rather than a load-time crash on the >=2-samples interpolation rule.
    material = RefractiveIndexMaterial('X', [0.55], [2.0], k=[0.01])
    assert float(material.n(0.4)) == pytest.approx(2.0)
    assert float(material.n(1.0)) == pytest.approx(2.0)
    assert float(material.k(0.7)) == pytest.approx(0.01)
