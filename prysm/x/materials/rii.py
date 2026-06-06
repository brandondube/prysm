"""refractiveindex.info catalog backend using prysm's controlled _riidb cache."""

import sqlite3
from pathlib import Path

from prysm.mathops import np
from prysm.conf import config

from .catalog import AmbiguousMaterialError, Catalog
from .core import MaterialRecord, _normalize_name
from .tabulated import TabulatedMaterial


def default_cache_root():
    """Return the deterministic refractiveindex.info cache root."""
    return Path(__file__).resolve().parents[3] / '_riidb'


def _row_to_dict(row):
    if isinstance(row, dict):
        return dict(row)
    return {key: row[key] for key in row.keys()}


def _record_from_page(row, namespace, sqlite_path):
    data = _row_to_dict(row)
    metadata = {
        'pageid': data['pageid'],
        'shelf': data['shelf'],
        'book': data['book'],
        'page': data['page'],
        'filepath': data['filepath'],
        'hasrefractive': bool(data['hasrefractive']),
        'hasextinction': bool(data['hasextinction']),
        'points': data['points'],
        'provenance': 'refractiveindex.info sqlite',
        'sqlite_path': str(sqlite_path),
    }
    aliases = tuple(
        item for item in (data['page'], data['filepath'])
        if item and item != data['book']
    )
    return MaterialRecord(
        name=data['book'],
        catalog=namespace,
        variant=data['page'],
        aliases=aliases,
        source=data['filepath'],
        license='CC0',
        wavelength_range=(data['rangeMin'], data['rangeMax']),
        material_class='RefractiveIndexMaterial',
        metadata=metadata,
        material_id=f'{namespace}:{data["shelf"]}:{data["book"]}:{data["page"]}',
    )


class RefractiveIndexMaterial(TabulatedMaterial):
    """Tabulated material loaded from refractiveindex.info sqlite rows."""

    def __init__(self, record, wavelengths, n, *, k=None):
        metadata = dict(record.metadata)
        metadata['aliases'] = record.aliases
        self.page_info = {
            'pageid': metadata['pageid'],
            'shelf': metadata['shelf'],
            'book': metadata['book'],
            'page': metadata['page'],
            'filepath': metadata['filepath'],
            'rangeMin': record.wavelength_range[0],
            'rangeMax': record.wavelength_range[1],
        }
        super().__init__(
            record.name,
            wavelengths,
            n,
            k=k,
            catalog=record.catalog,
            variant=record.variant,
            source=record.source,
            license=record.license,
            wavelength_range=record.wavelength_range,
            metadata=metadata,
            missing_k='zero' if k is None else 'raise',
            page_info=self.page_info,
        )


class RefractiveIndexCatalog(Catalog):
    """Catalog adapter for refractiveindex.info sqlite data."""

    def __init__(self, records, *, sqlite_path, cache_root=None, namespace='RII'):
        self.sqlite_path = Path(sqlite_path)
        self.cache_root = Path(cache_root) if cache_root is not None else self.sqlite_path.parent
        self.namespace = namespace
        records = list(records)
        for record in records:
            record.loader = lambda record=record: self.material_for_pageid(
                record.metadata['pageid']
            )
        super().__init__(records, namespace=namespace)

    @classmethod
    def from_cache(cls, cache_root=None, *, download=False, namespace='RII'):
        """Load the catalog from _riidb, optionally preparing the cache first."""
        cache_root = Path(cache_root) if cache_root is not None else default_cache_root()
        sqlite_path = cache_root / 'refractive.db'
        if not sqlite_path.exists() and download:
            cache_root.mkdir(parents=True, exist_ok=True)
            _try_external_download(cache_root)
        if not sqlite_path.exists():
            raise FileNotFoundError(
                f'refractiveindex.info sqlite cache not found at {sqlite_path}'
            )
        return cls.from_sqlite(sqlite_path, cache_root=cache_root, namespace=namespace)

    @classmethod
    def from_sqlite(cls, sqlite_path, *, cache_root=None, namespace='RII'):
        """Load metadata records from an existing refractive.db file."""
        sqlite_path = Path(sqlite_path)
        with sqlite3.connect(sqlite_path) as con:
            con.row_factory = sqlite3.Row
            rows = con.execute(
                """
                SELECT pageid, shelf, book, page, filepath,
                       hasrefractive, hasextinction, rangeMin, rangeMax, points
                FROM pages
                WHERE hasrefractive = 1
                ORDER BY shelf, book, page
                """
            ).fetchall()
        records = [_record_from_page(row, namespace, sqlite_path) for row in rows]
        return cls(records, sqlite_path=sqlite_path, cache_root=cache_root, namespace=namespace)

    def material_for_pageid(self, pageid):
        """Load one material by refractiveindex.info page id."""
        record = self._record_for_pageid(pageid)
        with sqlite3.connect(self.sqlite_path) as con:
            n_rows = con.execute(
                """
                SELECT wave, refindex
                FROM refractiveindex
                WHERE pageid = ?
                ORDER BY wave
                """,
                (pageid,),
            ).fetchall()
            k_rows = con.execute(
                """
                SELECT wave, coeff
                FROM extcoeff
                WHERE pageid = ?
                ORDER BY wave
                """,
                (pageid,),
            ).fetchall()
        if not n_rows:
            raise KeyError(f'no refractive index samples for pageid {pageid}')
        wavelengths = np.array([row[0] for row in n_rows], dtype=config.precision)
        n = np.array([row[1] for row in n_rows], dtype=config.precision)
        k = None
        if k_rows:
            k_wavelengths = np.array([row[0] for row in k_rows], dtype=config.precision)
            k_values = np.array([row[1] for row in k_rows], dtype=config.precision)
            if len(k_wavelengths) == len(wavelengths) and np.all(k_wavelengths == wavelengths):
                k = k_values
            else:
                k = np.interp(wavelengths, k_wavelengths, k_values).astype(
                    wavelengths.dtype,
                    copy=False,
                )
        return RefractiveIndexMaterial(record, wavelengths, n, k=k)

    def _record_for_pageid(self, pageid):
        for record in self.records():
            if record.metadata['pageid'] == pageid:
                return record
        raise KeyError(f'no refractiveindex.info pageid {pageid}')

    def material_for_name(self, name, **qualifiers):
        """Resolve one material by material name plus shelf/book/page qualifiers."""
        catalog = qualifiers.pop('catalog', qualifiers.pop('namespace', None))
        if catalog is not None and _normalize_name(catalog) != _normalize_name(self.namespace):
            raise KeyError(f'no material named {name!r} in catalog {catalog!r}')
        shelf = qualifiers.pop('shelf', None)
        book = qualifiers.pop('book', None)
        page = qualifiers.pop('page', None)
        matches = []
        for record in self.records():
            metadata = record.metadata
            if not _rii_name_matches(record, name):
                continue
            if shelf is not None and _normalize_name(metadata['shelf']) != _normalize_name(shelf):
                continue
            if page is not None and _normalize_name(metadata['page']) != _normalize_name(page):
                continue
            if book is not None and not (
                _normalize_name(metadata['book']) == _normalize_name(book)
                or _normalize_name(metadata['page']) == _normalize_name(book)
            ):
                continue
            if any(metadata.get(key) != value for key, value in qualifiers.items()):
                continue
            matches.append(record)
        if not matches:
            raise KeyError(f'no refractiveindex.info material named {name!r}')
        if len(matches) > 1:
            raise AmbiguousMaterialError(name, matches)
        return matches[0].load()


def _rii_name_matches(record, name):
    norm = _normalize_name(name)
    if _normalize_name(record.name) == norm:
        return True
    return any(_normalize_name(alias) == norm for alias in record.aliases)


def _try_external_download(cache_root):
    try:
        from refractivesqlite import Database
    except ImportError as exc:
        raise ImportError(
            'download=True requires the refractivesqlite package'
        ) from exc
    Database(cache_root / 'refractive.db').create_database_from_url()
