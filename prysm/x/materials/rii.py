"""refractiveindex.info catalog backend over the ri.info YAML database.

prysm reads the refractiveindex.info database (catalog-nk.yml plus the per-page
YAML data files) directly and parses each material into a backend-pure
FormulaMaterial (formula ids 1-9) or a tabulated RefractiveIndexMaterial.  The
optional refractiveindex package is used only to auto-download the database
folder when it is absent; an existing folder is read without importing it.
"""

from functools import partial
from pathlib import Path

from prysm.mathops import np
from prysm.conf import config

from .catalog import Catalog
from .core import FormulaMaterial, MaterialRecord, _normalize_name
from .formulas import riinfo_formula
from .tabulated import TabulatedMaterial


_PREFERRED_BOOK_BY_PREFIX = {
    'N-': ('SCHOTT-optical',),
    'P-': ('SCHOTT-optical',),
    'S-': ('OHARA-optical',),
    'J-': ('HIKARI-optical',),
    'H-': ('CDGM-optical',),
    'K-': ('SUMITA-optical',),
}


def default_db_path():
    """Return the refractiveindex package's default database folder."""
    return Path.home() / '.refractiveindex.info-database'


def _rii_page_info(material):
    """page_info shape for a refractiveindex.info-sourced material."""
    wr = material.wavelength_range
    lo, hi = wr if wr is not None else (None, None)
    meta = material.metadata
    return {
        'shelf': meta.get('shelf'),
        'book': meta.get('book'),
        'page': meta.get('page'),
        'filepath': material.source or meta.get('filepath') or '',
        'rangeMin': lo,
        'rangeMax': hi,
    }


def _rank_page(record, name):
    """Sort key preferring the canonical dataset for a glass name.

    Lower is better: an exact page-name match, a manufacturer-spec shelf, and a
    catalog-page book (e.g. SCHOTT-optical for an N- glass) all reduce the rank.
    """
    meta = record.metadata
    page = meta.get('page') or ''
    book = meta.get('book') or ''
    shelf = meta.get('shelf') or ''
    key = str(name)
    norm = _normalize_name(key)
    rank = 100
    if page.upper() == key.upper():
        rank -= 50
    if _normalize_name(page) == norm:
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


def _load_catalog(db_path):
    """Map (shelf, book, page) -> data file path from catalog-nk.yml."""
    import yaml
    catalog_file = Path(db_path) / 'catalog-nk.yml'
    with open(catalog_file, 'rt', encoding='utf-8') as f:
        catalog = yaml.load(f, Loader=yaml.BaseLoader)
    index = {}
    for shelf in catalog:
        if 'DIVIDER' in shelf:
            continue
        shelf_name = shelf['SHELF']
        for book_entry in shelf.get('content', []):
            if 'DIVIDER' in book_entry:
                continue
            book_name = book_entry['BOOK']
            for page_entry in book_entry.get('content', []):
                if 'DIVIDER' in page_entry:
                    continue
                page_name = page_entry['PAGE']
                filepath = Path(db_path) / 'data' / Path(page_entry['data'])
                index[(shelf_name, book_name, page_name)] = filepath
    return index


def _parse_tabulated(data_str):
    """Parse a refractiveindex.info tabulated DATA block into arrays."""
    wavelengths = []
    col1 = []
    col2 = []
    for row in data_str.strip().split('\n'):
        parts = row.split()
        if not parts:
            continue
        wavelengths.append(float(parts[0]))
        col1.append(float(parts[1]))
        if len(parts) > 2:
            col2.append(float(parts[2]))
    wl = np.array(wavelengths, dtype=config.precision)
    c1 = np.array(col1, dtype=config.precision)
    c2 = np.array(col2, dtype=config.precision) if col2 else None
    return wl, c1, c2


def _ensure_database_downloaded(db_path):
    """Populate db_path via the refractiveindex package's auto-download.

    Uses only the package's public API: constructing a material runs its
    _ensure_database (which downloads when the folder is absent) before the
    catalog key check, so a deliberately missing key raises KeyError after the
    folder is in place.
    """
    try:
        from refractiveindex import RefractiveIndexMaterial as _Probe
    except ImportError as exc:
        raise ImportError(
            'the refractiveindex.info database is absent and downloading it '
            'requires the optional refractiveindex package; install it with '
            "pip install 'prysm[glass]' (or pip install refractiveindex), or "
            'pass an existing db_path'
        ) from exc
    # The probe exists only to trigger the package's auto-download side effect.
    # The deliberately-bogus key raises once the folder is in place (KeyError in
    # current versions); tolerate any error and verify the outcome directly, so
    # a changed exception type cannot defeat an otherwise-successful download.
    try:
        _Probe(
            '__prysm__', '__prysm__', '__prysm__',
            db_path=str(db_path), auto_download=True,
        )
    except Exception:
        pass
    if not (Path(db_path) / 'catalog-nk.yml').exists():
        raise FileNotFoundError(
            f'auto-download did not populate the refractiveindex.info database '
            f'at {db_path}'
        )


class RefractiveIndexMaterial(TabulatedMaterial):
    """Tabulated material loaded from a refractiveindex.info data file."""

    def __init__(self, name, wavelengths, n, *, k=None, variant=None,
                 catalog='RII', source=None, metadata=None):
        # a single-sample page is a constant index: nearest interpolation with
        # extrapolation returns that lone value at any wavelength, instead of
        # failing the >=2-samples requirement for linear interpolation.
        single = len(wavelengths) < 2
        super().__init__(
            name,
            wavelengths,
            n,
            k=k,
            catalog=catalog,
            variant=variant,
            source=source,
            license='CC0',
            metadata=dict(metadata or {}),
            missing_k='zero' if k is None else 'raise',
            method='nearest' if single else None,
            extrapolate=bool(single),
        )
        self._page_info_builder = _rii_page_info


def _load_rii_material(shelf, book, page, filepath, namespace):
    """Parse one refractiveindex.info YAML file into a prysm material."""
    import yaml
    with open(filepath, 'rt', encoding='utf-8') as f:
        doc = yaml.load(f, Loader=yaml.BaseLoader)

    metadata = {'shelf': shelf, 'book': book, 'page': page, 'filepath': str(filepath)}
    n_grid = None
    k_grid = None
    formula = None
    for data in doc['DATA']:
        parts = data['type'].split()
        category = parts[0]
        subtype = parts[1] if len(parts) > 1 else None
        if category == 'tabulated':
            wl, c1, c2 = _parse_tabulated(data['data'])
            if subtype == 'n':
                n_grid = (wl, c1)
            elif subtype == 'k':
                k_grid = (wl, c1)
            elif subtype == 'nk':
                n_grid = (wl, c1)
                k_grid = (wl, c2)
        elif category == 'formula':
            fid = int(subtype)
            coeffs = tuple(float(s) for s in data['coefficients'].split())
            rng = data.get('range', data.get('wavelength_range'))
            lo, hi = (float(x) for x in rng.split())
            formula = (fid, coeffs, lo, hi)

    if formula is not None:
        fid, coeffs, lo, hi = formula
        k_formula = None
        if k_grid is not None:
            # keep n analytic and interpolate the tabulated k, rather than
            # degrading n to samples on the k grid.
            wlk, kk = k_grid

            def _k_from_table(wvl):
                return np.interp(wvl, wlk, kk)

            k_formula = _k_from_table
        material = FormulaMaterial(
            book,
            partial(riinfo_formula, fid),
            coeffs,
            k_formula=k_formula,
            catalog=namespace,
            variant=page,
            source=str(filepath),
            license='CC0',
            wavelength_range=(lo, hi),
            metadata=metadata,
        )
        material._page_info_builder = _rii_page_info
        return material

    if n_grid is None:
        raise ValueError(f'refractiveindex.info material {filepath} has no n data')
    wl, nn = n_grid
    kk = None
    if k_grid is not None:
        wlk, kk_raw = k_grid
        if len(wlk) == len(wl) and np.all(wlk == wl):
            kk = kk_raw
        else:
            kk = np.interp(wl, wlk, kk_raw).astype(wl.dtype, copy=False)
    return RefractiveIndexMaterial(
        book, wl, nn, k=kk, variant=page, catalog=namespace,
        source=str(filepath), metadata=metadata,
    )


class RefractiveIndexCatalog(Catalog):
    """Catalog adapter over the refractiveindex.info YAML database."""

    def __init__(self, records, *, db_path=None, namespace='RII'):
        self.db_path = None if db_path is None else Path(db_path)
        self.namespace = namespace
        super().__init__(records, namespace=namespace)
        # normalized-name -> records index so material_for_name is an O(1) hit
        # plus a rank over same-name candidates, not a normalize-and-scan over
        # every page on each lookup.
        index = {}
        for record in self.records():
            for norm in _record_match_names(record):
                index.setdefault(norm, []).append(record)
        self._records_by_norm = index

    @classmethod
    def from_database(cls, db_path=None, *, download=True, namespace='RII'):
        """Build a catalog from the ri.info database, downloading if absent."""
        db_path = Path(db_path) if db_path is not None else default_db_path()
        if not (db_path / 'catalog-nk.yml').exists():
            if download:
                _ensure_database_downloaded(db_path)
            else:
                raise FileNotFoundError(
                    f'refractiveindex.info database not found at {db_path}'
                )
        index = _load_catalog(db_path)
        records = [
            _rii_record(shelf, book, page, filepath, namespace)
            for (shelf, book, page), filepath in index.items()
        ]
        return cls(records, db_path=db_path, namespace=namespace)

    def material_for_name(self, name, **qualifiers):
        """Resolve a glass name to its best-ranked refractiveindex.info page."""
        catalog = qualifiers.pop('catalog', qualifiers.pop('namespace', None))
        if catalog is not None and _normalize_name(catalog) != _normalize_name(self.namespace):
            raise KeyError(f'no material named {name!r} in catalog {catalog!r}')
        shelf = qualifiers.pop('shelf', None)
        book = qualifiers.pop('book', None)
        page = qualifiers.pop('page', None)
        norm = _normalize_name(name)
        matches = []
        for record in self._records_by_norm.get(norm, ()):
            meta = record.metadata
            if shelf is not None and _normalize_name(meta.get('shelf') or '') != _normalize_name(shelf):
                continue
            if book is not None and _normalize_name(meta.get('book') or '') != _normalize_name(book):
                continue
            if page is not None and _normalize_name(meta.get('page') or '') != _normalize_name(page):
                continue
            if any(meta.get(key) != value for key, value in qualifiers.items()):
                continue
            matches.append(record)
        if not matches:
            raise KeyError(f'no refractiveindex.info material named {name!r}')
        best = min(matches, key=lambda record: _rank_page(record, name))
        return best.load()


def _record_match_names(record):
    """Normalized names a record can be looked up by (name, variant, aliases)."""
    candidates = (record.name, record.variant) + tuple(record.aliases)
    return {_normalize_name(candidate) for candidate in candidates if candidate}


def _rii_record(shelf, book, page, filepath, namespace):
    aliases = tuple(item for item in (page, str(filepath)) if item and item != book)
    return MaterialRecord(
        name=book,
        catalog=namespace,
        variant=page,
        aliases=aliases,
        source=str(filepath),
        license='CC0',
        material_class='RefractiveIndexMaterial',
        metadata={'shelf': shelf, 'book': book, 'page': page, 'filepath': str(filepath)},
        loader=partial(_load_rii_material, shelf, book, page, filepath, namespace),
        material_id=f'{namespace}:{shelf}:{book}:{page}',
    )
