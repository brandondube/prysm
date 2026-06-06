"""Standalone Zemax AGF material catalog backend."""

from pathlib import Path

from prysm.mathops import np
from prysm.conf import config

from .catalog import Catalog
from .core import (
    MaterialRecord,
    _normalize_name,
)
from .formulas import agf_formula


_CATALOG_ALIASES = {
    'SCHOTT': 'SCHOTT',
    'SCHOTTGLASS': 'SCHOTT',
    'SCHOTTOPTICAL': 'SCHOTT',
    'OHARA': 'OHARA',
    'OHARAOPTICAL': 'OHARA',
    'HOYA': 'HOYA',
    'HIKARI': 'HIKARI',
    'CDGM': 'CDGM',
    'SUMITA': 'SUMITA',
}

_METADATA_RECORDS = {'GC', 'ED', 'TD', 'IT', 'MD', 'OD', 'BD'}


def _catalog_key(catalog):
    norm = _normalize_name(catalog or '')
    return _CATALOG_ALIASES.get(norm, norm)


def _catalog_from_path(path):
    stem = Path(path).stem
    upper = stem.upper()
    for key in _CATALOG_ALIASES:
        if key in _normalize_name(upper):
            return _CATALOG_ALIASES[key]
    return upper


def _float_tokens(tokens):
    return tuple(float(token) for token in tokens)


def _decode_agf_bytes(data):
    if data.startswith((b'\xff\xfe', b'\xfe\xff')):
        return data.decode('utf-16')
    if data.startswith(b'\xef\xbb\xbf'):
        return data.decode('utf-8-sig')
    try:
        return data.decode('utf-8')
    except UnicodeDecodeError:
        return data.decode('cp1252')


def _aliases_for_agf_name(name):
    aliases = []
    upper = name.upper()
    if upper.startswith('N-'):
        aliases.append(upper[2:])
    return tuple(aliases)


class AGFMaterial:
    """Material parsed from one AGF NM record."""

    __slots__ = (
        'name', 'catalog', 'formula', 'coefficients', 'wavelength_min',
        'wavelength_max', 'metadata', 'source_path', 'variant', 'source',
        'citation', 'license', 'process', 'temperature_range', 'page_info',
    )

    def __init__(
        self,
        name,
        catalog,
        formula,
        coefficients,
        *,
        wavelength_min=None,
        wavelength_max=None,
        metadata=None,
        source_path=None,
        variant=None,
        source=None,
        citation=None,
        license=None,
        process=None,
        temperature_range=None,
    ):
        self.name = name
        self.catalog = catalog or ''
        self.formula = formula
        self.coefficients = tuple(float(c) for c in coefficients)
        self.wavelength_min = None if wavelength_min is None else float(wavelength_min)
        self.wavelength_max = None if wavelength_max is None else float(wavelength_max)
        self.source_path = source_path
        self.variant = variant
        self.source = source or source_path
        self.citation = citation
        self.license = license
        self.process = process
        self.temperature_range = temperature_range
        self.metadata = dict(metadata) if metadata is not None else {}
        self.metadata.setdefault('formula', self.formula)
        self.metadata.setdefault('aliases', _aliases_for_agf_name(self.name))
        self.page_info = {
            'shelf': 'agf',
            'book': f'{self.catalog}-agf' if self.catalog else 'agf',
            'page': self.name,
            'filepath': self.source_path or '',
            'catalog': self.catalog,
            'formula': self.formula,
            'rangeMin': self.wavelength_min,
            'rangeMax': self.wavelength_max,
        }

    @property
    def wavelength_range(self):
        """Return AGF LD wavelength limits in microns."""
        return self.wavelength_min, self.wavelength_max

    def __call__(self, wvl_um):
        """Alias for n(wvl_um)."""
        return self.n(wvl_um)

    def _validate_wavelengths(self, wvl):
        lo, hi = self.wavelength_range
        if lo is None or hi is None:
            return
        out_of_range = np.less(wvl, lo) | np.greater(wvl, hi)
        if np.any(out_of_range):
            raise ValueError(
                f'wavelength for {self.name} outside AGF range {lo:g} to {hi:g} um'
            )

    def n(self, wvl_um, temperature=None):
        """Return refractive index from the AGF dispersion formula."""
        self._validate_wavelengths(wvl_um)
        return agf_formula(self.formula, self.coefficients, wvl_um, self.name)

    def k(self, wvl_um, temperature=None):
        """Return zero extinction for AGF glass records."""
        self._validate_wavelengths(wvl_um)
        if np.isscalar(wvl_um):
            return wvl_um * 0
        if hasattr(wvl_um, 'shape'):
            return np.zeros_like(wvl_um)
        return np.zeros(np.shape(wvl_um), dtype=config.precision)

    def nk(self, wvl_um, temperature=None):
        """Return complex refractive index n + 1j*k."""
        return self.n(wvl_um, temperature=temperature) + 1j * self.k(
            wvl_um, temperature=temperature
        )

    def record(self):
        """Return a lazy material record for catalog indexing."""
        metadata = dict(self.metadata)
        return MaterialRecord(
            name=self.name,
            catalog=self.catalog,
            variant=self.variant,
            aliases=metadata.get('aliases', ()),
            source=self.source_path,
            citation=self.citation,
            license=self.license,
            wavelength_range=self.wavelength_range,
            temperature_range=self.temperature_range,
            process=self.process,
            material_class=type(self).__name__,
            metadata=metadata,
            loader=lambda: self,
        )


class AGFCatalog(Catalog):
    """Collection of AGF materials."""

    def __init__(self, materials, catalog=None, namespace=None, comments=()):
        namespace = namespace if namespace is not None else catalog
        self.materials = tuple(materials)
        self.catalog = namespace or (self.materials[0].catalog if self.materials else '')
        self.comments = tuple(comments)
        super().__init__([material.record() for material in self.materials], namespace=self.catalog)

    @classmethod
    def from_file(cls, path, namespace=None, catalog=None):
        """Parse one AGF file from disk."""
        path = Path(path)
        text = _decode_agf_bytes(path.read_bytes())
        namespace = namespace if namespace is not None else catalog
        namespace = namespace or _catalog_from_path(path)
        return cls.from_text(text, namespace=namespace, source_path=str(path))

    @classmethod
    def from_files(cls, paths, namespace=None):
        """Parse several AGF files into one catalog."""
        materials = []
        comments = []
        for path in paths:
            catalog = cls.from_file(path)
            materials.extend(catalog.materials)
            comments.extend(catalog.comments)
        return cls(materials, namespace=namespace or 'AGF', comments=comments)

    @classmethod
    def from_text(cls, text, namespace='AGF', source_path=None, catalog=None):
        """Parse AGF text into an AGFCatalog."""
        if catalog is not None and namespace == 'AGF':
            namespace = catalog
        namespace = _catalog_key(namespace)
        materials = []
        comments = []
        current = None

        def finish_current():
            if current is None:
                return
            materials.append(AGFMaterial(
                name=current['name'],
                catalog=namespace,
                formula=current['formula'],
                coefficients=current.get('coefficients', ()),
                wavelength_min=current.get('wavelength_min'),
                wavelength_max=current.get('wavelength_max'),
                metadata=current.get('metadata', {}),
                source_path=source_path,
            ))

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith('!'):
                continue
            tokens = line.split()
            record = tokens[0].upper()
            rest = tokens[1:]

            if record == 'CC':
                comments.append(' '.join(rest))
                continue

            if record == 'NM':
                finish_current()
                if len(rest) < 2:
                    raise ValueError(f'malformed AGF NM record: {line!r}')
                current = {
                    'name': rest[0],
                    'formula': int(float(rest[1])),
                    'metadata': {'NM': (' '.join(rest[2:]),)},
                }
                continue

            if current is None:
                continue

            if record == 'CD':
                current['coefficients'] = _float_tokens(rest)
            elif record == 'LD':
                limits = _float_tokens(rest[:2])
                if len(limits) == 2:
                    current['wavelength_min'] = limits[0]
                    current['wavelength_max'] = limits[1]
            elif record in _METADATA_RECORDS:
                current['metadata'].setdefault(record, ())
                current['metadata'][record] += (' '.join(rest),)

        finish_current()
        return cls(materials, namespace=namespace, comments=comments)


def load_agf_catalog(path_or_paths, namespace=None):
    """Load one AGF file or an iterable of AGF files."""
    if isinstance(path_or_paths, (str, Path)):
        return AGFCatalog.from_file(path_or_paths, namespace=namespace)
    return AGFCatalog.from_files(path_or_paths, namespace=namespace)
