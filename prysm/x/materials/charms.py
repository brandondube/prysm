"""CHARMS-style cryogenic material support."""

from prysm.mathops import np
from prysm.conf import config

from .catalog import Catalog
from .core import BaseMaterial
from .tabulated import TemperatureGridMaterial


def _polyval_ascending(coefficients, temperature):
    out = np.zeros(np.shape(temperature), dtype=coefficients.dtype)
    power = np.ones_like(out)
    for coefficient in coefficients:
        out = out + coefficient * power
        power = power * temperature
    return out


def _coefficient_array(value, label):
    arr = np.array(value, dtype=config.precision)
    if arr.shape[0] != 3:
        raise ValueError(f'{label} must provide three Sellmeier terms')
    return arr


class TemperatureSellmeierMaterial(BaseMaterial):
    """Temperature-dependent Sellmeier material in the CHARMS form."""

    def __init__(
        self,
        name,
        strength_coefficients,
        resonance_coefficients,
        *,
        residuals=None,
        measurement_uncertainty=None,
        **kwargs,
    ):
        missing_k = kwargs.pop('missing_k', 'zero')
        metadata = dict(kwargs.pop('metadata', {}) or {})
        if residuals is not None:
            metadata['residuals'] = residuals
        if measurement_uncertainty is not None:
            metadata['measurement_uncertainty'] = measurement_uncertainty
        super().__init__(name, metadata=metadata, missing_k=missing_k, **kwargs)
        self.strength_coefficients = _coefficient_array(
            strength_coefficients, 'strength_coefficients'
        )
        self.resonance_coefficients = _coefficient_array(
            resonance_coefficients, 'resonance_coefficients'
        )

    def n(self, wvl_um, temperature=None):
        """Evaluate the temperature-dependent Sellmeier equation.

        Vectorized over the broadcast (wavelength, temperature) query: the
        temperature polynomials and the Sellmeier sum are whole-array ops with
        no per-point loop or item assignment.
        """
        if temperature is None:
            raise ValueError(f'temperature is required for {self.name}')
        self._check_wavelength(wvl_um)
        self._check_temperature(temperature)
        wvl_b, temp_b = np.broadcast_arrays(wvl_um, temperature)
        w2 = wvl_b ** 2
        n2 = 1.0 + wvl_b * 0
        for strength, resonance in zip(
            self.strength_coefficients, self.resonance_coefficients
        ):
            S = _polyval_ascending(strength, temp_b)
            lam = _polyval_ascending(resonance, temp_b)
            n2 = n2 + S * w2 / (w2 - lam ** 2)
        return np.sqrt(n2)


class CHARMSCoefficientMaterial(TemperatureSellmeierMaterial):
    """CHARMS coefficient-table material."""

    def __init__(self, name, coefficients=None, **kwargs):
        if coefficients is not None:
            if isinstance(coefficients, dict):
                strength = coefficients.get('S', coefficients.get('strength'))
                resonance = coefficients.get('lambda', coefficients.get('resonance'))
            else:
                strength, resonance = coefficients
            kwargs.setdefault('strength_coefficients', strength)
            kwargs.setdefault('resonance_coefficients', resonance)
        super().__init__(name, **kwargs)


class CHARMSTableMaterial(TemperatureGridMaterial):
    """CHARMS absolute-index table material."""


class CHARMSDataset(Catalog):
    """Catalog container for CHARMS materials."""

    @classmethod
    def from_materials(cls, materials, *, namespace='CHARMS'):
        """Build a CHARMS dataset from material instances.

        Catalog.from_materials already stamps the namespace onto each record, so
        no per-material mutation is needed here.
        """
        return super().from_materials(materials, namespace=namespace)
