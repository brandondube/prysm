"""Raytracing-facing material lookup helpers.

A glass token resolves to a callable material via a catalog object that
exposes material_for_name(name).  When no catalog is supplied the
refractiveindex.info database is used (downloaded on first use); AIR / VACUUM /
blank resolve to unit index and MIRROR to a reflective sentinel.
"""


MIRROR = '__MIRROR__'

_DEFAULT_CATALOG = None


def air(wvl):
    """Index of air using the pure-vacuum approximation."""
    return 1.0


def vacuum(wvl):
    """Index of vacuum."""
    return 1.0


def _default_catalog():
    """Return the module-cached refractiveindex.info catalog (downloads once)."""
    global _DEFAULT_CATALOG
    if _DEFAULT_CATALOG is None:
        from .rii import RefractiveIndexCatalog
        _DEFAULT_CATALOG = RefractiveIndexCatalog.from_database()
    return _DEFAULT_CATALOG


def glass(name, database=None):
    """Resolve a glass name through a material catalog or the default database."""
    if database is None:
        database = _default_catalog()
    if hasattr(database, 'material_for_name'):
        return database.material_for_name(name)
    raise TypeError(
        'database must be a material catalog exposing material_for_name(name)'
    )


def lookup(name, database=None):
    """Resolve a glass token to a callable material, air, or the MIRROR sentinel."""
    if name is None or not str(name).strip():
        return air
    key = str(name).strip().upper()
    if key in ('AIR', 'VACUUM'):
        return air
    if key == 'MIRROR':
        return MIRROR
    return glass(name, database=database)
