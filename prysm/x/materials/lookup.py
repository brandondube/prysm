"""Material lookup helpers."""

from .core import ConstantMaterial


MIRROR = '__MIRROR__'

_DEFAULT_CATALOG = None

# Unit-index media as MaterialProtocol singletons.
air = ConstantMaterial(1.0, name='air')
vacuum = ConstantMaterial(1.0, name='vacuum')


def _default_catalog():
    """Return the module-cached refractiveindex.info catalog (downloads once)."""
    global _DEFAULT_CATALOG
    if _DEFAULT_CATALOG is None:
        from .rii import RefractiveIndexCatalog
        _DEFAULT_CATALOG = RefractiveIndexCatalog.from_database()
    return _DEFAULT_CATALOG


def glass(name, database=None, **qualifiers):
    """Resolve a glass name through a material catalog or the default database.

    Qualifiers (e.g. page for the refractiveindex.info catalog) are forwarded
    to the catalog's material_for_name.
    """
    if database is None:
        database = _default_catalog()
    if hasattr(database, 'material_for_name'):
        return database.material_for_name(name, **qualifiers)
    raise TypeError(
        'database must be a material catalog exposing material_for_name(name)'
    )


def resolve_index(spec, name_resolver=None):
    """Resolve any index spec to a callable n(wvl), MIRROR, air, or None.

    Strings 'MIRROR', 'AIR', and 'VACUUM' are special.  Other strings route
    through name_resolver; numeric specs become constant callables.
    """
    if spec is None:
        return None
    if spec is MIRROR:
        return MIRROR
    if isinstance(spec, str):
        key = spec.strip().upper()
        if spec == MIRROR or key == 'MIRROR':
            return MIRROR
        if not key or key in ('AIR', 'VACUUM'):
            return air
        if name_resolver is None:
            raise TypeError(f'cannot resolve glass name {spec!r} without a catalog')
        return name_resolver(spec)
    if callable(spec):
        return spec
    value = spec
    return lambda wvl: value


def lookup(name, database=None, **qualifiers):
    """Resolve a glass token to a callable material, air, or the MIRROR sentinel.

    Qualifiers (e.g. catalog for a vendor) narrow a glass-name resolution.
    """
    resolved = resolve_index(
        name, name_resolver=lambda token: glass(token, database=database, **qualifiers)
    )
    return air if resolved is None else resolved
