"""Core material protocol and common material helpers.

Wavelengths are in microns.  Temperatures are in Kelvin.  Complex refractive
index follows the convention n + 1j*k.
"""

import inspect

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


def _user_page_info(material):
    """Default page_info shape for a user-defined material."""
    wr = material.wavelength_range
    lo, hi = wr if wr is not None else (None, None)
    meta = material.metadata
    catalog = material.catalog
    return {
        'shelf': 'user',
        'book': catalog or 'USER',
        'page': material.name,
        'filepath': material.source or '',
        'catalog': catalog or 'USER',
        'rangeMin': lo,
        'rangeMax': hi,
        'model': meta.get('model', meta.get('method')),
    }


def _accepts_temperature(func):
    """True if func can be called with a temperature= keyword."""
    if func is None:
        return False
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return False
    for parameter in signature.parameters.values():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            return True
        if parameter.name == 'temperature' and parameter.kind in (
            inspect.Parameter.KEYWORD_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            return True
    return False


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
    """Duck-typed optical material interface."""

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
        self._page_info_builder = _user_page_info

    def __call__(self, wvl_um):
        """Alias for n(wvl_um)."""
        return self.n(wvl_um)

    @property
    def page_info(self):
        """Read-only provenance view derived from this material's attributes."""
        return self._page_info_builder(self)

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

    def _central_difference(self, evaluate, x, h_floor, valid_range, extrapolate):
        """Range-clamped central difference of evaluate() about x.

        Clamping the +/-h points into valid_range degrades the central
        difference to a one-sided difference at a closed band edge instead of
        evaluating out of range; a collapsed interval yields 0 (the derivative
        of a locally-constant sample), not 0/0.
        """
        h = np.maximum(np.abs(x) * 1e-6, h_floor)
        hi_point = np.add(x, h)
        lo_point = np.subtract(x, h)
        if valid_range is not None and not extrapolate:
            lo, hi = valid_range
            if hi is not None:
                hi_point = np.minimum(hi_point, hi)
            if lo is not None:
                lo_point = np.maximum(lo_point, lo)
        num = evaluate(hi_point) - evaluate(lo_point)
        denom = hi_point - lo_point
        return np.where(denom == 0, 0.0, num / np.where(denom == 0, 1.0, denom))

    def dn_dlambda(self, wvl_um, temperature=None):
        """Finite-difference derivative of n with respect to wavelength."""
        return self._central_difference(
            lambda w: self.n(w, temperature=temperature),
            wvl_um, 1e-6, self.wavelength_range,
            self.metadata.get('extrapolate_wavelength'),
        )

    def dn_dT(self, wvl_um, temperature):
        """Finite-difference derivative of n with respect to temperature."""
        return self._central_difference(
            lambda t: self.n(wvl_um, temperature=t),
            temperature, 1e-3, self.temperature_range,
            self.metadata.get('extrapolate_temperature'),
        )

    def record(self, *, loader=None, catalog=None):
        """Create a metadata record for this material.

        A catalog override lets a catalog stamp its namespace onto the record
        without mutating the caller-owned material.
        """
        if loader is None:
            loader = lambda: self
        aliases = tuple(self.metadata.get('aliases', ()))
        return MaterialRecord(
            name=self.name,
            catalog=self.catalog if catalog is None else catalog,
            variant=self.variant,
            aliases=aliases,
            source=self.source,
            citation=self.citation,
            license=self.license,
            wavelength_range=self.wavelength_range,
            temperature_range=self.temperature_range,
            process=self.process,
            material_class=self.metadata.get('material_class', type(self).__name__),
            metadata=dict(self.metadata),
            loader=loader,
        )


class ConstantMaterial(BaseMaterial):
    """Material with constant n and optional constant k."""

    def __init__(self, n, *, name=None, k=None, **kwargs):
        n = float(n)
        if not np.isfinite(n):
            raise ValueError('n must be finite')
        if name is None:
            name = f'const_{n:g}'
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
        self._formula_wants_temperature = _accepts_temperature(formula)
        self._k_formula_wants_temperature = _accepts_temperature(k_formula)

    def n(self, wvl_um, temperature=None):
        """Return formula-derived real refractive index."""
        self._check_wavelength(wvl_um)
        self._check_temperature(temperature)
        if temperature is not None and self._formula_wants_temperature:
            return self.formula(wvl_um, *self.coefficients, temperature=temperature)
        return self.formula(wvl_um, *self.coefficients)

    def k(self, wvl_um, temperature=None):
        """Return formula-derived extinction coefficient."""
        self._check_wavelength(wvl_um)
        self._check_temperature(temperature)
        if self.k_formula is None:
            return self._missing_k(wvl_um)
        if temperature is not None and self._k_formula_wants_temperature:
            return self.k_formula(wvl_um, *self.k_coefficients, temperature=temperature)
        return self.k_formula(wvl_um, *self.k_coefficients)


# d/F/C spectral lines (microns) defining nd and the Abbe number.
_LINE_D, _LINE_F, _LINE_C = 0.5875618, 0.4861327, 0.6562725


def model_glass(nd, vd, name=None):
    """Cauchy model glass reproducing index nd and Abbe number vd at d/F/C.

    The two-term Cauchy A + B/wvl**2 is the unique fit through (nd, Vd); a
    designer's stand-in for a real glass, not a partial-dispersion model.
    """
    from .formulas import cauchy
    B = ((nd - 1.0) / vd) / (1.0 / _LINE_F ** 2 - 1.0 / _LINE_C ** 2)
    A = nd - B / _LINE_D ** 2
    if name is None:
        name = f'model {nd:.4f}/{vd:.2f}'
    return FormulaMaterial(name, cauchy, (A, B),
                           metadata={'model_glass': True, 'nd': nd, 'vd': vd})
