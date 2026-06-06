"""Raytracing-facing material lookup helpers."""

from pathlib import Path

from prysm.mathops import np


try:
    from refractivesqlite import Database
except ImportError:  # pragma: no cover - exercised when optional package absent
    Database = None


MIRROR = '__MIRROR__'
_PREFERRED_BOOK_BY_PREFIX = {
    'N-': ('SCHOTT-optical',),
    'P-': ('SCHOTT-optical',),
    'S-': ('OHARA-optical',),
    'J-': ('HIKARI-optical',),
    'H-': ('CDGM-optical',),
    'K-': ('SUMITA-optical',),
}


def air(wvl):
    """Index of air using the pure-vacuum approximation."""
    return 1.0


def vacuum(wvl):
    """Index of vacuum."""
    return 1.0


def _normalize_name(name):
    return ''.join(ch for ch in str(name).strip().upper() if ch not in '-_ ')


class SQLiteMaterial:
    """Callable wrapper around a refractivesqlite Material."""

    def __init__(self, material, page_info=None):
        self.material = material
        if page_info is None and hasattr(material, 'get_page_info'):
            page_info = material.get_page_info()
        self.page_info = dict(page_info or {})
        self.name = self.page_info.get('page') or self.page_info.get('book')
        self.catalog = self.page_info.get('book')

    def __call__(self, wvl):
        """Return refractive index at wavelength(s) in microns."""
        n = self.material.get_refractiveindex(wvl, unit='um')
        if np.isscalar(wvl):
            return float(n)
        return n

    def n(self, wvl_um, temperature=None):
        """Return real refractive index at wavelength in microns."""
        return self(wvl_um)


class RefractiveIndexDatabase:
    """Small adapter around refractivesqlite.Database."""

    def __init__(self, database):
        self.database = database
        self.path = getattr(database, 'path', None)

    def search_pages(self, name):
        """Return pages whose page name matches name by catalog rules."""
        key = name.strip()
        norm = _normalize_name(key)
        rows = _search_custom(
            self.database,
            """
            SELECT pageid, shelf, book, page, filepath,
                   hasrefractive, hasextinction, rangeMin, rangeMax, points
            FROM pages
            WHERE hasrefractive = 1
              AND (
                UPPER(page) = UPPER(?)
                OR REPLACE(REPLACE(REPLACE(UPPER(page), '-', ''), '_', ''), ' ', '') = ?
              )
            """,
            (key, norm),
        )
        rows = [_page_row_to_dict(row) for row in rows]
        return sorted(rows, key=lambda row: self._rank_page(row, key, norm))

    def get_material(self, pageid):
        """Return a callable material for pageid."""
        material = self.database.get_material(pageid)
        page_info = material.get_page_info() if hasattr(material, 'get_page_info') else None
        return SQLiteMaterial(material, page_info=page_info)

    def material_for_name(self, name):
        """Return the preferred material matching a glass name."""
        pages = self.search_pages(name)
        if not pages:
            raise KeyError(f'no refractiveindex.info material named {name!r}')
        return self.get_material(pages[0]['pageid'])

    @staticmethod
    def _rank_page(row, key, norm):
        page = row.get('page', '')
        book = row.get('book', '')
        shelf = row.get('shelf', '')
        page_norm = _normalize_name(page)
        rank = 100
        if page.upper() == key.upper():
            rank -= 50
        if page_norm == norm:
            rank -= 25
        if shelf == 'specs':
            rank -= 10
        for prefix, books in _PREFERRED_BOOK_BY_PREFIX.items():
            if key.upper().startswith(prefix) and book in books:
                rank -= 20
                break
        if book.endswith('-optical'):
            rank -= 5
        return (rank, shelf, book, page)


def _page_row_to_dict(row):
    if isinstance(row, dict):
        return dict(row)
    keys = (
        'pageid',
        'shelf',
        'book',
        'page',
        'filepath',
        'hasrefractive',
        'hasextinction',
        'rangeMin',
        'rangeMax',
        'points',
    )
    return dict(zip(keys, row))


def _search_custom(database, sql, params):
    try:
        return database.search_custom(sql, params)
    except TypeError:
        return database.search_custom(_inline_sql_params(sql, params))


def _inline_sql_params(sql, params):
    for param in params:
        sql = sql.replace('?', _sql_literal(param), 1)
    return sql


def _sql_literal(value):
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    if value is None:
        return 'NULL'
    return str(value)


def _require_database_type(database_type=Database):
    if database_type is None:
        raise ImportError(
            'refractivesqlite is required for raytracing glass lookup; '
            'install HugoGuillen/refractiveindex.info-sqlite'
        )
    return database_type


def load_material_db(database_type=Database):
    """Load the refractiveindex.info database from the prysm repo root."""
    dbtype = _require_database_type(database_type)
    path = Path(__file__).resolve().parents[3] / '_riidb' / 'refractive.db'
    db = dbtype(path)
    if not path.exists():
        db.create_database_from_url()
    return db


def glass(name, database=None, database_type=Database):
    """Resolve a glass name from a material catalog or refractivesqlite database."""
    db = _database_adapter(database, database_type)
    if db is None:
        raise KeyError(
            'no material database supplied; pass a catalog with '
            'material_for_name(name) or a refractivesqlite.Database object'
        )
    return db.material_for_name(name)


def _database_adapter(database, database_type=Database):
    if database is None:
        return None
    if hasattr(database, 'material_for_name'):
        return database
    dbtype = database_type
    if dbtype is not None and isinstance(database, dbtype):
        return RefractiveIndexDatabase(database)
    _require_database_type(dbtype)
    raise TypeError('database must be a material catalog or refractivesqlite.Database instance')


def lookup(name, database=None, database_type=Database):
    """Resolve a glass token to a callable material, air, or MIRROR."""
    if name is None or not str(name).strip():
        return air
    key = str(name).strip().upper()
    if key in ('AIR', 'VACUUM'):
        return air
    if key == 'MIRROR':
        return MIRROR
    return glass(name, database=database, database_type=database_type)
