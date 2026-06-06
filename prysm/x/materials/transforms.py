"""Opt-in material transforms for process and environment effects."""

import inspect

from .core import BaseMaterial


def _metadata_with_parent(parent, metadata=None):
    out = dict(getattr(parent, 'metadata', {}) or {})
    out.update(metadata or {})
    chain = list(out.get('parent_chain', ()))
    chain.append({
        'name': getattr(parent, 'name', None),
        'catalog': getattr(parent, 'catalog', None),
        'variant': getattr(parent, 'variant', None),
    })
    out['parent_chain'] = tuple(chain)
    return out


def _call_correction(correction, wvl_um, temperature):
    correction_n = getattr(correction, 'n', None)
    if callable(correction_n):
        return correction_n(wvl_um, temperature=temperature)
    if not callable(correction):
        return correction
    try:
        call_signature = inspect.signature(correction)
    except (TypeError, ValueError):
        try:
            return correction(wvl_um, temperature)
        except TypeError:
            return correction(wvl_um)
    try:
        call_signature.bind(wvl_um, temperature)
    except TypeError:
        try:
            call_signature.bind(wvl_um, temperature=temperature)
        except TypeError:
            return correction(wvl_um)
        return correction(wvl_um, temperature=temperature)
    return correction(wvl_um, temperature)


class MaterialTransform(BaseMaterial):
    """Base wrapper preserving material provenance."""

    def __init__(self, parent, *, name=None, metadata=None, **kwargs):
        self.parent = parent
        super().__init__(
            name or getattr(parent, 'name', type(parent).__name__),
            catalog=kwargs.pop('catalog', getattr(parent, 'catalog', None)),
            variant=kwargs.pop('variant', getattr(parent, 'variant', None)),
            source=kwargs.pop('source', getattr(parent, 'source', None)),
            citation=kwargs.pop('citation', getattr(parent, 'citation', None)),
            license=kwargs.pop('license', getattr(parent, 'license', None)),
            wavelength_range=kwargs.pop(
                'wavelength_range',
                getattr(parent, 'wavelength_range', None),
            ),
            temperature_range=kwargs.pop(
                'temperature_range',
                getattr(parent, 'temperature_range', None),
            ),
            process=kwargs.pop('process', getattr(parent, 'process', None)),
            metadata=_metadata_with_parent(parent, metadata),
            missing_k=kwargs.pop('missing_k', getattr(parent, 'missing_k', 'zero')),
            **kwargs,
        )

    def k(self, wvl_um, temperature=None):
        """Delegate extinction coefficient to the parent material."""
        if hasattr(self.parent, 'k'):
            return self.parent.k(wvl_um, temperature=temperature)
        return super().k(wvl_um, temperature=temperature)


class TemperatureShiftedMaterial(MaterialTransform):
    """Apply an explicit dn/dT correction from a reference temperature."""

    def __init__(self, parent, dn_dT, reference_temperature, **kwargs):
        super().__init__(parent, **kwargs)
        self.dn_dT_model = dn_dT
        self.reference_temperature = reference_temperature

    def n(self, wvl_um, temperature=None):
        """Return parent n shifted by dn/dT times delta T."""
        if temperature is None:
            temperature = self.reference_temperature
        self._check_temperature(temperature)
        base = self.parent.n(wvl_um, temperature=self.reference_temperature)
        slope = _call_correction(self.dn_dT_model, wvl_um, temperature)
        return base + slope * (temperature - self.reference_temperature)


class IndexOffsetMaterial(MaterialTransform):
    """Apply an explicit additive offset to n and optionally k."""

    def __init__(self, parent, offset, *, k_offset=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.offset = offset
        self.k_offset = k_offset

    def n(self, wvl_um, temperature=None):
        """Return parent n plus offset."""
        return self.parent.n(wvl_um, temperature=temperature) + _call_correction(
            self.offset, wvl_um, temperature
        )

    def k(self, wvl_um, temperature=None):
        """Return parent k plus optional offset."""
        out = super().k(wvl_um, temperature=temperature)
        if self.k_offset is None:
            return out
        return out + _call_correction(self.k_offset, wvl_um, temperature)


class StressOpticMaterial(MaterialTransform):
    """Apply a scalar stress-optic index correction."""

    def __init__(self, parent, coefficient, stress, **kwargs):
        super().__init__(parent, **kwargs)
        self.coefficient = coefficient
        self.stress = stress

    def n(self, wvl_um, temperature=None):
        """Return parent n plus stress-optic correction."""
        coefficient = _call_correction(self.coefficient, wvl_um, temperature)
        return self.parent.n(wvl_um, temperature=temperature) + coefficient * self.stress


class ThicknessDependentMaterial(MaterialTransform):
    """Apply an opt-in thickness-dependent index correction."""

    def __init__(self, parent, model, thickness, *, thickness_range=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.model = model
        self.thickness = thickness
        self.thickness_range = thickness_range
        if thickness_range is not None:
            lo, hi = thickness_range
            if (lo is not None and thickness < lo) or (hi is not None and thickness > hi):
                raise ValueError('thickness is outside the model range')

    def n(self, wvl_um, temperature=None):
        """Return parent n plus thickness-dependent correction."""
        if callable(self.model):
            try:
                offset = self.model(self.thickness, wvl_um, temperature)
            except TypeError:
                offset = self.model(self.thickness, wvl_um)
        else:
            offset = self.model
        return self.parent.n(wvl_um, temperature=temperature) + offset


class ProcessVariantMaterial(MaterialTransform):
    """Metadata-rich process variant with unchanged optical behavior."""

    def __init__(self, parent, *, process=None, variant=None, **kwargs):
        super().__init__(parent, process=process, variant=variant, **kwargs)

    def n(self, wvl_um, temperature=None):
        """Delegate n to the parent material."""
        self._check_wavelength(wvl_um)
        self._check_temperature(temperature)
        return self.parent.n(wvl_um, temperature=temperature)
