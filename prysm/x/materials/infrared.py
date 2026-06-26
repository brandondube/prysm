"""Infrared optical materials: CHARMS silicon/germanium and Malitson sapphire."""

from .charms import TemperatureSellmeierMaterial
from .core import FormulaMaterial
from .formulas import sellmeier
from .catalog import Catalog
from .transforms import IsothermalMaterial


# Frey, Leviton & Madison, "Temperature-dependent refractive index of silicon
# and germanium", Proc. SPIE 6273, 62732J (2006), Tables 5 and 10.  Three-term
# Sellmeier with 4th-order temperature dependence; rows are ascending powers of
# T (K), one per strength S_i / resonance lam_i.
_CHARMS_CITE = 'Frey, Leviton & Madison, Proc. SPIE 6273, 62732J (2006)'

_SI_STRENGTH = (
    (10.4907, -2.08020e-4, 4.21694e-6, -5.82298e-9, 3.44688e-12),
    (-1346.61, 29.1664, -0.278724, 1.05939e-3, -1.35089e-6),
    (4.42827e7, -1.76213e6, -7.61575e4, 678.414, 103.243),
)
_SI_RESONANCE = (
    (0.299713, -1.14234e-5, 1.67134e-7, -2.51049e-10, 2.32484e-14),
    (-3.51710e3, 42.3892, -0.357957, 1.17504e-3, -1.13212e-6),
    (1.71400e6, -1.44984e5, -6.90744e3, -39.3699, 23.5770),
)
_GE_STRENGTH = (
    (13.9723, 2.52809e-3, -5.02195e-6, 2.22604e-8, -4.86238e-12),
    (0.452096, -3.09197e-3, 2.16895e-5, -6.02290e-8, 4.12038e-11),
    (751.447, -14.2843, -0.238093, 2.96047e-3, -7.73454e-6),
)
_GE_RESONANCE = (
    (0.386367, 2.01871e-4, -5.93448e-7, -2.27923e-10, 5.37423e-12),
    (1.08843, 1.16510e-3, -4.97284e-6, 1.12357e-8, 9.40201e-12),
    (-2893.19, -0.967948, -0.527016, 6.49364e-3, -1.95162e-5),
)


def charms_silicon(name='silicon'):
    """CHARMS temperature-dependent silicon (1.1-5.6 um, 20-300 K)."""
    return TemperatureSellmeierMaterial(
        name, _SI_STRENGTH, _SI_RESONANCE,
        wavelength_range=(1.1, 5.6), temperature_range=(20.0, 300.0),
        catalog='CHARMS', citation=_CHARMS_CITE,
    )


def charms_germanium(name='germanium'):
    """CHARMS temperature-dependent germanium (1.9-5.5 um, 20-300 K).

    dn/dT is enormous (~4e-4 /K near room temperature); a real Ge design must
    fix the operating temperature, not assume the room-temperature index.
    """
    return TemperatureSellmeierMaterial(
        name, _GE_STRENGTH, _GE_RESONANCE,
        wavelength_range=(1.9, 5.5), temperature_range=(20.0, 300.0),
        catalog='CHARMS', citation=_CHARMS_CITE,
    )


# Malitson & Dodge, "Refractive index and birefringence of synthetic sapphire",
# J. Opt. Soc. Am. 62, 1405 (1972); ordinary ray, room temperature, 0.2-5.5 um.
_SAPPHIRE_A = (1.4313493, 0.65054713, 5.3414021)
_SAPPHIRE_B = (0.0726631 ** 2, 0.1193242 ** 2, 18.028251 ** 2)


def sapphire_ordinary(name='sapphire', *, aliases=()):
    """Room-temperature ordinary-ray sapphire (Malitson, 0.2-5.5 um)."""
    return FormulaMaterial(
        name, sellmeier, (_SAPPHIRE_A, _SAPPHIRE_B),
        wavelength_range=(0.2, 5.5), catalog='Malitson',
        citation='Malitson & Dodge, J. Opt. Soc. Am. 62, 1405 (1972)',
        metadata={'aliases': tuple(aliases)},
    )


def infrared_catalog(temperature=295.0):
    """Catalog of MWIR materials, the CHARMS models bound to temperature (K).

    Germanium and silicon are the temperature-dependent CHARMS models frozen at
    temperature so they answer the bare n(wvl) the ray trace makes; sapphire is
    the room-temperature Malitson model.  Names and aliases cover the usual Code
    V tokens (GERMMW, SILICON, SAPHIR).
    """
    ge = IsothermalMaterial(
        charms_germanium(), temperature, name='germanium',
        metadata={'aliases': ('GE', 'GERMANIUM', 'GERMMW')})
    si = IsothermalMaterial(
        charms_silicon(), temperature, name='silicon',
        metadata={'aliases': ('SI', 'SILICON')})
    sap = sapphire_ordinary(aliases=('SAPHIR', 'SAPPHIRE', 'AL2O3'))
    return Catalog.from_materials([ge, si, sap], namespace='IR')
