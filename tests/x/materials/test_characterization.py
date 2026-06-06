"""Characterization net for the x/materials unification refactor.

These tests pin the externally observable behavior of one representative of
each data source: golden n/k values, the per-source page_info dict, and
__all__ resolvability.  They are the gate the unification phases must not move
(except the deliberate RII page_info re-baseline in Phase 5, which gets its own
characterization alongside the rewritten rii.py).
"""

from pathlib import Path

import pytest

from prysm.mathops import np
import prysm.x.materials as materials
from prysm.x.materials import (
    AGFCatalog,
    CHARMSCoefficientMaterial,
    CHARMSTableMaterial,
    ConstantMaterial,
    FittedMaterial,
    TabulatedMaterial,
)


DATA = Path(__file__).parents[1] / 'raytracing' / 'data' / 'materials'


# --------------------------------------------------------------------------
# golden n / k values
# --------------------------------------------------------------------------

def test_constant_golden():
    m = ConstantMaterial('constant', 1.5)
    assert float(m.n(0.55)) == pytest.approx(1.5)
    assert float(m.k(0.55)) == pytest.approx(0.0)


def test_tabulated_golden():
    m = TabulatedMaterial('tab', [0.5, 0.6, 0.7], [1.6, 1.5, 1.4])
    assert float(m.n(0.55)) == pytest.approx(1.55)
    assert float(m.n(0.65)) == pytest.approx(1.45)


def test_fitted_cauchy_golden():
    wls = np.array([0.45, 0.55, 0.65, 0.75])
    n = 1.5 + 0.01 / wls ** 2
    m = FittedMaterial.from_samples('fit', wls, n, model='cauchy')
    assert isinstance(m, FittedMaterial)
    assert float(m.n(0.55)) == pytest.approx(1.5330578512396698, rel=1e-12)


def test_agf_sellmeier_golden():
    m = AGFCatalog.from_file(DATA / 'tiny_schott.agf').material_for_name('N-BK7')
    assert float(m.n(0.5875618)) == pytest.approx(1.5168000345005885, rel=1e-12)
    assert float(m.n(0.4861327)) == pytest.approx(1.5223762897312285, rel=1e-12)
    assert float(m.n(0.6562725)) == pytest.approx(1.5143223472613747, rel=1e-12)
    assert float(m.k(0.55)) == pytest.approx(0.0)


def test_agf_schott_formula_golden():
    m = AGFCatalog.from_text(
        'NM SCH 1\nCD 2.25 0 0 0 0 0\nLD 0.4 0.8\n', namespace='SCH'
    ).material_for_name('SCH')
    assert float(m.n(0.55)) == pytest.approx(1.5)


def test_charms_coefficient_golden():
    m = CHARMSCoefficientMaterial(
        'test',
        coefficients={'S': [[0.6], [0.2], [0.1]], 'lambda': [[0.1], [0.2], [10.0]]},
        wavelength_range=(0.5, 2.0),
        temperature_range=(20, 300),
    )
    assert float(m.n(1.0, temperature=77)) == pytest.approx(1.3466194111120775, rel=1e-12)


def test_charms_table_golden():
    m = CHARMSTableMaterial(
        'table', [1.0, 2.0], [80, 300], [[1.5, 1.6], [1.7, 1.8]],
        layout=('temperature', 'wavelength'),
    )
    assert float(m.n(1.5, temperature=190)) == pytest.approx(1.65)


def test_rii_formula_golden(rii_catalog):
    # RII path is YAML-sourced (Phase 5); page_info drops the sqlite pageid key.
    m = rii_catalog.material_for_name('N-BK7')
    assert float(m.n(0.5875618)) == pytest.approx(1.5168000345005885, rel=1e-12)
    assert set(m.page_info) == {
        'shelf', 'book', 'page', 'filepath', 'rangeMin', 'rangeMax',
    }
    assert m.page_info['book'] == 'SCHOTT-optical'
    assert m.page_info['page'] == 'N-BK7'


# --------------------------------------------------------------------------
# page_info dicts (the Phase 3 before/after diff gate)
# --------------------------------------------------------------------------

def test_constant_page_info():
    m = ConstantMaterial('constant', 1.5)
    assert m.page_info == {
        'shelf': 'user', 'book': 'USER', 'page': 'constant', 'filepath': '',
        'catalog': 'USER', 'rangeMin': None, 'rangeMax': None, 'model': 'constant',
    }


def test_tabulated_page_info():
    m = TabulatedMaterial('tab', [0.5, 0.6, 0.7], [1.6, 1.5, 1.4])
    assert m.page_info == {
        'shelf': 'user', 'book': 'USER', 'page': 'tab', 'filepath': '',
        'catalog': 'USER', 'rangeMin': 0.5, 'rangeMax': 0.7, 'model': 'linear',
    }


def test_fitted_page_info():
    wls = np.array([0.45, 0.55, 0.65, 0.75])
    m = FittedMaterial.from_samples('fit', wls, 1.5 + 0.01 / wls ** 2, model='cauchy')
    assert m.page_info == {
        'shelf': 'user', 'book': 'USER', 'page': 'fit', 'filepath': '',
        'catalog': 'USER', 'rangeMin': 0.45, 'rangeMax': 0.75, 'model': 'cauchy',
    }


def test_agf_sellmeier_page_info():
    path = DATA / 'tiny_schott.agf'
    m = AGFCatalog.from_file(path).material_for_name('N-BK7')
    assert m.page_info == {
        'shelf': 'agf', 'book': 'SCHOTT-agf', 'page': 'N-BK7', 'filepath': str(path),
        'catalog': 'SCHOTT', 'formula': 2, 'rangeMin': 0.3, 'rangeMax': 2.5,
    }


def test_agf_schott_page_info():
    m = AGFCatalog.from_text(
        'NM SCH 1\nCD 2.25 0 0 0 0 0\nLD 0.4 0.8\n', namespace='SCH'
    ).material_for_name('SCH')
    assert m.page_info == {
        'shelf': 'agf', 'book': 'SCH-agf', 'page': 'SCH', 'filepath': '',
        'catalog': 'SCH', 'formula': 1, 'rangeMin': 0.4, 'rangeMax': 0.8,
    }


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------

def test_all_exports_resolve():
    for name in materials.__all__:
        assert hasattr(materials, name), name
