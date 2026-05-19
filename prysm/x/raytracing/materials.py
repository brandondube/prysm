"""Glass lookup for raytracing prescriptions.

Actual glasses are resolved only from an explicitly supplied refractivesqlite
Database object.

AIR, VACUUM, empty glass names, and MIRROR are special non-glass cases used by
the Zemax and Code V readers.

"""

from prysm.mathops import np

try:
    from refractivesqlite import Database
except ImportError:  # pragma: no cover - exercised when optional package absent
    Database = None


MIRROR = '__MIRROR__'  # sentinel string for IO consumers
_PREFERRED_BOOK_BY_PREFIX = {
    # Common manufacturer prefixes seen in Zemax and Code V glass catalogs.
    'N-': ('SCHOTT-optical',),
    'P-': ('SCHOTT-optical',),
    'S-': ('OHARA-optical',),
    'J-': ('HIKARI-optical',),
    'H-': ('CDGM-optical',),
    'K-': ('SUMITA-optical',),
}


def air(wvl):
    """Index of air (n=1).

    Pure-vacuum approximation; the standard-air dispersion (Edlen) is at
    the ~1e-4 level over the visible and is rarely worth modeling unless
    the system is metrology-grade.

    """
    return 1.0


def vacuum(wvl):
    """Index of vacuum (n=1)."""
    return 1.0


def _normalize_name(name):
    return ''.join(ch for ch in name.strip().upper() if ch not in '-_ ')


class SQLiteMaterial:
    """Callable wrapper around a refractivesqlite Material.

    Parameters
    ----------
    material
        Material returned by Database.get_material.
    page_info : dict, optional
        Row metadata from the refractiveindex.info-sqlite pages table.

    """

    def __init__(self, material, page_info=None):
        self.material = material
        if page_info is None and hasattr(material, 'get_page_info'):
            page_info = material.get_page_info()
        self.page_info = dict(page_info or {})

    def __call__(self, wvl):
        """Return the refractive index at wavelength(s) wvl in microns."""
        n = self.material.get_refractiveindex(wvl, unit='um')
        if np.isscalar(wvl):
            return float(n)
        return n


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
    keys = ('pageid', 'shelf', 'book', 'page', 'filepath',
            'hasrefractive', 'hasextinction', 'rangeMin', 'rangeMax', 'points')
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


def _require_database_type():
    if Database is None:
        raise ImportError(
            'refractivesqlite is required for raytracing glass lookup; '
            'install HugoGuillen/refractiveindex.info-sqlite'
        )
    return Database


def glass(name, database=None):
    """Resolve name from a refractiveindex.info-sqlite database.

    Parameters
    ----------
    name : str
        Glass page name, for example N-BK7 or S-BSL7.
    database : Database
        Refractivesqlite database reader.

    Returns
    -------
    SQLiteMaterial
        Callable n(wvl_um) material.

    """
    db = _database_adapter(database)
    if db is None:
        raise KeyError(
            'no refractiveindex.info-sqlite database supplied; pass '
            'a refractivesqlite.Database object as database'
        )
    return db.material_for_name(name)


def _database_adapter(database):
    if database is None:
        return None
    dbtype = Database
    if dbtype is not None and isinstance(database, dbtype):
        return RefractiveIndexDatabase(database)
    _require_database_type()
    raise TypeError('database must be a refractivesqlite.Database instance')


def lookup(name, database=None):
    """Resolve a glass name to a callable n(wvl_um) -> float.

    Returns the special sentinel MIRROR for reflective surfaces; IO code
    must detect this and set typ='refl' (not 'refr') without an n
    callback.

    Empty / None / 'AIR' / 'VACUUM' all resolve to air().

    Parameters
    ----------
    name : str or None
        glass name from a prescription file (Zemax GLAS, CodeV GLA).
    database : Database
        refractiveindex.info-sqlite database reader.

    Returns
    -------
    callable n(wvl) or the MIRROR sentinel string

    Raises
    ------
        KeyError
        when no configured database contains the named glass.

    """
    if name is None or not name.strip():
        return air
    key = name.strip().upper()
    if key in ('AIR', 'VACUUM'):
        return air
    if key == 'MIRROR':
        return MIRROR
    return glass(name, database=database)
