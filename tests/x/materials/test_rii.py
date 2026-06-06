import sqlite3

import pytest

from prysm.x.materials import AmbiguousMaterialError, RefractiveIndexCatalog
from prysm.x.materials.rii import default_cache_root


def make_rii_db(path):
    with sqlite3.connect(path) as con:
        con.execute(
            'CREATE TABLE pages(pageid int, shelf text, book text, page text, '
            'filepath text, hasrefractive integer, hasextinction integer, '
            'rangeMin real, rangeMax real, points int)'
        )
        con.execute('CREATE TABLE refractiveindex(pageid int, wave real, refindex real)')
        con.execute('CREATE TABLE extcoeff(pageid int, wave real, coeff real)')
        con.executemany(
            'INSERT INTO pages VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
            [
                (1, 'main', 'SiO2', 'Malitson', 'SiO2/nk/Malitson.yml', 1, 0, 0.4, 0.8, 3),
                (2, 'main', 'SiO2', 'Other', 'SiO2/nk/Other.yml', 1, 1, 0.4, 0.8, 3),
            ],
        )
        con.executemany(
            'INSERT INTO refractiveindex VALUES (?, ?, ?)',
            [
                (1, 0.4, 1.4), (1, 0.6, 1.5), (1, 0.8, 1.6),
                (2, 0.4, 1.5), (2, 0.6, 1.6), (2, 0.8, 1.7),
            ],
        )
        con.executemany(
            'INSERT INTO extcoeff VALUES (?, ?, ?)',
            [(2, 0.4, 0.01), (2, 0.6, 0.02), (2, 0.8, 0.03)],
        )


def test_default_cache_root_is_riidb():
    assert default_cache_root().name == '_riidb'


def test_rii_catalog_loads_sqlite_material_by_name_and_page_qualifier(tmp_path):
    db = tmp_path / 'refractive.db'
    make_rii_db(db)
    catalog = RefractiveIndexCatalog.from_sqlite(db)
    material = catalog.material_for_name('SiO2', book='Malitson')
    assert material.n(0.5) == pytest.approx(1.45)
    assert material.k(0.5) == pytest.approx(0)
    assert material.page_info['book'] == 'SiO2'
    assert material.page_info['page'] == 'Malitson'


def test_rii_ambiguous_lookup_requires_qualifier(tmp_path):
    db = tmp_path / 'refractive.db'
    make_rii_db(db)
    catalog = RefractiveIndexCatalog.from_sqlite(db)
    with pytest.raises(AmbiguousMaterialError):
        catalog.material_for_name('SiO2')
    material = catalog.material_for_name('SiO2', page='Other')
    assert material.nk(0.6) == pytest.approx(1.6 + 0.02j)
