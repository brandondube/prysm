"""Catalog containers, namespace lookup, and explicit ambiguity handling."""

from .core import MaterialRecord, _normalize_name, _range_contains


class AmbiguousMaterialError(KeyError):
    """Raised when a material lookup has more than one matching record."""

    def __init__(self, query, candidates):
        self.query = query
        self.candidates = tuple(candidates)
        labels = ', '.join(_record_label(record) for record in self.candidates)
        super().__init__(f'ambiguous material {query!r}; candidates: {labels}')


def _record_label(record):
    parts = []
    if record.catalog:
        parts.append(record.catalog)
    parts.append(record.name)
    if record.variant:
        parts.append(record.variant)
    return ':'.join(parts)


def _matches_name(record, name):
    norm = _normalize_name(name)
    return any(_normalize_name(candidate) == norm for candidate in record.names_for_match())


def _record_matches_query(record, query):
    if query is None:
        return True
    norm = _normalize_name(query)
    for candidate in record.names_for_match():
        cnorm = _normalize_name(candidate)
        if norm == cnorm or norm in cnorm:
            return True
    return False


def _record_matches_filters(record, filters):
    catalog = filters.get('catalog')
    if catalog is not None and _normalize_name(record.catalog or '') != _normalize_name(catalog):
        return False
    variant = filters.get('variant')
    if variant is not None and _normalize_name(record.variant or '') != _normalize_name(variant):
        return False
    process = filters.get('process')
    if process is not None and _normalize_name(record.process or '') != _normalize_name(process):
        return False
    material_class = filters.get('material_class')
    if material_class is not None and record.material_class != material_class:
        return False
    wavelength_range_contains = filters.get('wavelength_range_contains')
    if wavelength_range_contains is not None:
        if not _range_contains(record.wavelength_range, wavelength_range_contains):
            return False
    temperature_range_contains = filters.get('temperature_range_contains')
    if temperature_range_contains is not None:
        if not _range_contains(record.temperature_range, temperature_range_contains):
            return False
    for key, value in filters.items():
        if key in {
            'catalog',
            'variant',
            'process',
            'material_class',
            'wavelength_range_contains',
            'temperature_range_contains',
        }:
            continue
        if value is None:
            continue
        if record.metadata.get(key) != value:
            return False
    return True


class Catalog:
    """In-memory catalog over material records."""

    def __init__(self, records=(), *, namespace=None):
        self.namespace = namespace
        self._records = tuple(records)

    @classmethod
    def from_materials(cls, materials, *, namespace=None):
        """Build a catalog from material instances."""
        records = []
        for material in materials:
            if namespace is not None and not material.catalog:
                material.catalog = namespace
            records.append(material.record())
        return cls(records, namespace=namespace)

    def records(self):
        """Return all material records."""
        return self._records

    def search(self, query=None, **metadata_filters):
        """Search catalog metadata without forcing material instantiation."""
        return [
            record for record in self._records
            if _record_matches_query(record, query)
            and _record_matches_filters(record, metadata_filters)
        ]

    def material_for_name(self, name, **qualifiers):
        """Resolve one material by name or raise on missing/ambiguous matches."""
        catalog = qualifiers.pop('catalog', None)
        if catalog is None:
            catalog = qualifiers.pop('namespace', None)
        matches = [
            record for record in self._records
            if _matches_name(record, name)
            and _record_matches_filters(record, {'catalog': catalog, **qualifiers})
        ]
        if not matches:
            raise KeyError(f'no material named {name!r}')
        if len(matches) > 1:
            raise AmbiguousMaterialError(name, matches)
        return matches[0].load()

    def __getitem__(self, key):
        """Lookup by name, or by namespace:name."""
        if isinstance(key, str) and ':' in key:
            catalog, name = key.split(':', 1)
            return self.material_for_name(name, catalog=catalog)
        return self.material_for_name(key)


class CatalogChain:
    """Bundle several catalogs with explicit ambiguity rules."""

    def __init__(self, catalogs):
        self.catalogs = tuple(catalogs)

    def records(self):
        """Yield records from every catalog in chain order."""
        for catalog in self.catalogs:
            yield from catalog.records()

    def search(self, query=None, **metadata_filters):
        """Search across all catalogs."""
        return [
            record for catalog in self.catalogs
            for record in catalog.search(query, **metadata_filters)
        ]

    def material_for_name(self, name, **qualifiers):
        """Resolve one material by name across the chain."""
        qualifiers = dict(qualifiers)
        catalog_name = qualifiers.pop('catalog', None)
        if catalog_name is None:
            catalog_name = qualifiers.pop('namespace', None)
        matches = [
            record for record in self.records()
            if _matches_name(record, name)
            and _record_matches_filters(record, {'catalog': catalog_name, **qualifiers})
        ]
        if not matches:
            raise KeyError(f'no material named {name!r}')
        if len(matches) > 1:
            raise AmbiguousMaterialError(name, matches)
        return matches[0].load()

    def __getitem__(self, key):
        """Lookup by name, or by namespace:name."""
        if isinstance(key, str) and ':' in key:
            catalog, name = key.split(':', 1)
            return self.material_for_name(name, catalog=catalog)
        return self.material_for_name(key)
