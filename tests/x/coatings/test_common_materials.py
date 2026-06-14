"""Common-materials tables: structure, resolution routing, and band coverage."""

from pathlib import Path

import pytest

from prysm.x.coatings import common_materials as cm


class StubCatalog:
    """Records every material_for_name call and returns the name."""

    def __init__(self):
        self.calls = []

    def material_for_name(self, name, **qualifiers):
        self.calls.append((name, qualifiers))
        return name


def _all_tables():
    for app in ('AR', 'BANDPASS', 'MIRROR'):
        for band in cm.BANDS:
            yield app, band, cm.names(app, band)


def test_tables_keyed_by_bands_with_wellformed_tokens():
    for table in (cm.ANTIREFLECTION, cm.BANDPASS, cm.MIRROR):
        assert set(table) == set(cm.BANDS)
    for app, band, roles in _all_tables():
        assert roles, (app, band)
        for role, members in roles.items():
            assert isinstance(members, tuple), (app, band, role)
            for token in members:
                if isinstance(token, tuple):
                    book, page = token
                    assert isinstance(book, str) and isinstance(page, str)
                else:
                    assert isinstance(token, str)


def test_names_and_materials_case_insensitive_and_aliased():
    assert cm.names('ar', 'lwir') is cm.names('ANTIREFLECTION', 'LWIR')


def test_materials_routes_tokens_and_page_pins_through_catalog():
    db = StubCatalog()
    mats = cm.materials('BANDPASS', 'LWIR', database=db)
    assert mats['low'] == ('ZnS', 'ZnSe')
    assert mats['high'] == ('PbTe', 'Ge')
    assert ('ZnS', {}) in db.calls
    assert ('ZnSe', {'page': 'Amotchkina'}) in db.calls
    assert ('PbTe', {'page': 'Weiting-300K'}) in db.calls


@pytest.mark.skipif(
    not (Path.home() / '.refractiveindex.info-database').exists(),
    reason='refractiveindex.info database not downloaded',
)
def test_every_entry_evaluates_across_its_band():
    for app, band, roles in _all_tables():
        lo, hi = cm.BANDS[band]
        mats = cm.materials(app, band)
        for role, members in mats.items():
            for token, mat in zip(roles[role], members):
                for wvl in (lo, (lo + hi) / 2, hi):
                    nk = complex(mat.nk(wvl))
                    assert nk.real > 0, (app, band, role, token, wvl)
