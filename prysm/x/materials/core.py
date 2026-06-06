"""Core material protocol and common material helpers.

Wavelengths are in microns.  Temperatures are in Kelvin.  Complex refractive
index follows the convention n + 1j*k.
"""

from prysm.mathops import np
from prysm.conf import config


class MaterialRangeError(ValueError):
    """Raised when a material is evaluated outside its valid range."""


class MissingKError(ValueError):
    """Raised when extinction data is requested but not available."""


def _normalize_name(name):
    return ''.join(ch for ch in str(name).strip().upper() if ch not in '-_ ')


def _range_contains(outer, inner):
    if outer is None or inner is None:
        return False
    lo, hi = outer
    ilo, ihi = inner
    if lo is None or hi is None or ilo is None or ihi is None:
        return False
    return lo <= ilo and hi >= ihi


def _user_page_info(name, catalog, source, wavelength_range, model):
    """Build best-effort page metadata for user-defined materials."""
    lo, hi = wavelength_range if wavelength_range is not None else (None, None)
    return {
        'shelf': 'user',
        'book': catalog or 'USER',
        'page': name,
        'filepath': source or '',
        'catalog': catalog or 'USER',
        'rangeMin': lo,
        'rangeMax': hi,
        'model': model,
    }


def _validate_range(values, valid_range, label, name):
    if valid_range is None:
        return
    lo, hi = valid_range
    if lo is None and hi is None:
        return
    out = False
    if lo is not None:
        out = out | np.less(values, lo)
    if hi is not None:
        out = out | np.greater(values, hi)
    if np.any(out):
        if lo is None:
            range_text = f'<= {hi:g}'
        elif hi is None:
            range_text = f'>= {lo:g}'
        else:
            range_text = f'{lo:g} to {hi:g}'
        raise MaterialRangeError(
            f'{label} for {name} is outside valid range {range_text}'
        )


class MaterialProtocol:
    """Documents the duck-typed optical material interface.

    A material is anything that carries the metadata attributes name, catalog,
    variant, source, citation, license, wavelength_range, temperature_range,
    process, and metadata, and that provides the methods below.  Nothing
    subclasses this; it is a reference for the convention only.
    """

    def n(self, wvl_um, temperature=None):
        """Return real refractive index at wavelength in microns."""

    def k(self, wvl_um, temperature=None):
        """Return extinction coefficient at wavelength in microns."""

    def nk(self, wvl_um, temperature=None):
        """Return complex refractive index n + 1j*k."""

    def __call__(self, wvl_um):
        """Alias for n(wvl_um)."""


class MaterialRecord:
    """Metadata-only catalog entry with a lazy material loader."""

    __slots__ = (
        'name', 'catalog', 'variant', 'aliases', 'source', 'citation',
        'license', 'wavelength_range', 'temperature_range', 'process',
        'material_class', 'metadata', 'loader', 'material_id',
    )

    def __init__(
        self,
        name,
        *,
        catalog=None,
        variant=None,
        aliases=(),
        source=None,
        citation=None,
        license=None,
        wavelength_range=None,
        temperature_range=None,
        process=None,
        material_class=None,
        metadata=None,
        loader=None,
        material_id=None,
    ):
        self.name = name
        self.catalog = catalog
        self.variant = variant
        self.aliases = () if aliases is None else tuple(aliases)
        self.source = source
        self.citation = citation
        self.license = license
        self.wavelength_range = wavelength_range
        self.temperature_range = temperature_range
        self.process = process
        self.metadata = dict(metadata) if metadata is not None else {}
        self.loader = loader
        if material_class is None:
            material_class = self.metadata.get('material_class')
        self.material_class = material_class
        if material_id is None:
            parts = [catalog, name, variant]
            material_id = ':'.join(str(p) for p in parts if p)
        self.material_id = material_id

    def load(self):
        """Instantiate or return the material represented by this record."""
        if self.loader is None:
            raise ValueError(f'material record {self.name!r} has no loader')
        return self.loader()

    def names_for_match(self):
        """Return name and aliases used for normalized lookup."""
        names = [self.name]
        if self.variant:
            names.append(self.variant)
        names.extend(self.aliases)
        return tuple(names)


class BaseMaterial:
    """Base class implementing shared material metadata and metrics."""

    def __init__(
        self,
        name,
        *,
        catalog=None,
        variant=None,
        source=None,
        citation=None,
        license=None,
        wavelength_range=None,
        temperature_range=None,
        process=None,
        metadata=None,
        missing_k='zero',
    ):
        if missing_k not in ('zero', 'raise'):
            raise ValueError("missing_k must be 'zero' or 'raise'")
        self.name = name
        self.catalog = catalog
        self.variant = variant
        self.source = source
        self.citation = citation
        self.license = license
        self.wavelength_range = wavelength_range
        self.temperature_range = temperature_range
        self.process = process
        self.metadata = dict(metadata or {})
        self.missing_k = missing_k

    def __call__(self, wvl_um):
        """Alias for n(wvl_um)."""
        return self.n(wvl_um)

    def _check_wavelength(self, wvl):
        if self.metadata.get('extrapolate_wavelength'):
            return
        _validate_range(wvl, self.wavelength_range, 'wavelength', self.name)

    def _check_temperature(self, temperature):
        if temperature is None:
            return
        if self.metadata.get('extrapolate_temperature'):
            return
        _validate_range(temperature, self.temperature_range, 'temperature', self.name)

    def _missing_k(self, wvl_um):
        if self.missing_k == 'raise':
            raise MissingKError(f'extinction data k is not available for {self.name}')
        if np.isscalar(wvl_um):
            return wvl_um * 0
        if hasattr(wvl_um, 'shape'):
            return np.zeros_like(wvl_um)
        return np.zeros(np.shape(wvl_um), dtype=config.precision)

    def k(self, wvl_um, temperature=None):
        """Return extinction coefficient or apply the configured missing-k policy."""
        self._check_wavelength(wvl_um)
        self._check_temperature(temperature)
        return self._missing_k(wvl_um)

    def nk(self, wvl_um, temperature=None):
        """Return complex refractive index n + 1j*k."""
        n = self.n(wvl_um, temperature=temperature)
        k = self.k(wvl_um, temperature=temperature)
        return n + 1j * k

    def n_at(self, wvl_um, temperature=None):
        """Return n at one wavelength, useful for registry searches."""
        return self.n(wvl_um, temperature=temperature)

    def dispersion(self, wvl1_um, wvl2_um, temperature=None):
        """Return n(wvl1) - n(wvl2)."""
        return self.n(wvl1_um, temperature=temperature) - self.n(
            wvl2_um, temperature=temperature
        )

    def partial_dispersion(
        self,
        wvl1_um,
        wvl2_um,
        wvl3_um,
        wvl4_um,
        temperature=None,
    ):
        """Return a partial dispersion ratio."""
        num = self.dispersion(wvl1_um, wvl2_um, temperature=temperature)
        den = self.dispersion(wvl3_um, wvl4_um, temperature=temperature)
        return num / den

    def abbe(self, wvl_short_um, wvl_center_um, wvl_long_um, temperature=None):
        """Return an Abbe-like number for arbitrary wavelengths."""
        n_center = self.n(wvl_center_um, temperature=temperature)
        n_short = self.n(wvl_short_um, temperature=temperature)
        n_long = self.n(wvl_long_um, temperature=temperature)
        return (n_center - 1) / (n_short - n_long)

    def dn_dlambda(self, wvl_um, temperature=None):
        """Finite-difference derivative of n with respect to wavelength."""
        h = np.maximum(np.abs(wvl_um) * 1e-6, 1e-6)
        return (
            self.n(np.add(wvl_um, h), temperature=temperature)
            - self.n(np.subtract(wvl_um, h), temperature=temperature)
        ) / (2 * h)

    def dn_dT(self, wvl_um, temperature):
        """Finite-difference derivative of n with respect to temperature."""
        h = np.maximum(np.abs(temperature) * 1e-6, 1e-3)
        return (
            self.n(wvl_um, temperature=np.add(temperature, h))
            - self.n(wvl_um, temperature=np.subtract(temperature, h))
        ) / (2 * h)

    def record(self, *, loader=None):
        """Create a metadata record for this material."""
        if loader is None:
            loader = lambda: self
        aliases = tuple(self.metadata.get('aliases', ()))
        return MaterialRecord(
            name=self.name,
            catalog=self.catalog,
            variant=self.variant,
            aliases=aliases,
            source=self.source,
            citation=self.citation,
            license=self.license,
            wavelength_range=self.wavelength_range,
            temperature_range=self.temperature_range,
            process=self.process,
            material_class=type(self).__name__,
            metadata=dict(self.metadata),
            loader=loader,
        )


class ConstantMaterial(BaseMaterial):
    """Material with constant n and optional constant k."""

    def __init__(self, *args, name=None, n=None, k=None, page_info=None, **kwargs):
        name, n = _constant_material_args(args, name, n)
        n = float(n)
        if not np.isfinite(n):
            raise ValueError('n must be finite')
        if k is not None:
            k = float(k)
            if not np.isfinite(k) or k < 0:
                raise ValueError('k must be finite and nonnegative')
        missing_k = kwargs.pop('missing_k', 'zero' if k is None else 'raise')
        super().__init__(name, missing_k=missing_k, **kwargs)
        self.n_value = n
        self.k_value = k
        self.index = n
        self.extinction = 0.0 if k is None else k
        self.fit_report = None
        self.metadata.setdefault('model', 'constant')
        self.metadata.setdefault('extrapolate', True)
        self.page_info = _user_page_info(
            self.name,
            self.catalog,
            self.source,
            self.wavelength_range,
            'constant',
        )
        if page_info:
            self.page_info.update(page_info)

    def n(self, wvl_um, temperature=None):
        """Return constant real refractive index."""
        self._check_wavelength(wvl_um)
        self._check_temperature(temperature)
        if np.isscalar(wvl_um):
            return wvl_um * 0 + self.n_value
        if hasattr(wvl_um, 'shape'):
            return np.zeros_like(wvl_um) + self.n_value
        return np.zeros(np.shape(wvl_um), dtype=config.precision) + self.n_value

    def k(self, wvl_um, temperature=None):
        """Return constant extinction coefficient."""
        self._check_wavelength(wvl_um)
        self._check_temperature(temperature)
        if self.k_value is None:
            return self._missing_k(wvl_um)
        if np.isscalar(wvl_um):
            return wvl_um * 0 + self.k_value
        if hasattr(wvl_um, 'shape'):
            return np.zeros_like(wvl_um) + self.k_value
        return np.zeros(np.shape(wvl_um), dtype=config.precision) + self.k_value


def _constant_material_args(args, name, n):
    if len(args) > 2:
        raise TypeError('ConstantMaterial expects name, n or n with name=')
    if len(args) == 2:
        if name is not None or n is not None:
            raise TypeError('ConstantMaterial got duplicate name or n')
        return args[0], args[1]
    if len(args) == 1:
        if n is not None:
            if name is not None:
                raise TypeError('ConstantMaterial got duplicate name')
            return args[0], n
        if name is None:
            if isinstance(args[0], str):
                raise TypeError('ConstantMaterial missing n')
            return 'CONSTANT', args[0]
        return name, args[0]
    if n is None:
        raise TypeError('ConstantMaterial missing n')
    return 'CONSTANT' if name is None else name, n


class FormulaMaterial(BaseMaterial):
    """Material backed by a wavelength formula callable."""

    def __init__(
        self,
        name,
        formula,
        coefficients=(),
        *,
        k_formula=None,
        k_coefficients=(),
        **kwargs,
    ):
        missing_k = kwargs.pop('missing_k', 'zero' if k_formula is None else 'raise')
        super().__init__(name, missing_k=missing_k, **kwargs)
        self.formula = formula
        self.coefficients = tuple(coefficients)
        self.k_formula = k_formula
        self.k_coefficients = tuple(k_coefficients)

    def n(self, wvl_um, temperature=None):
        """Return formula-derived real refractive index."""
        self._check_wavelength(wvl_um)
        self._check_temperature(temperature)
        return self.formula(wvl_um, *self.coefficients)

    def k(self, wvl_um, temperature=None):
        """Return formula-derived extinction coefficient."""
        self._check_wavelength(wvl_um)
        self._check_temperature(temperature)
        if self.k_formula is None:
            return self._missing_k(wvl_um)
        return self.k_formula(wvl_um, *self.k_coefficients)
