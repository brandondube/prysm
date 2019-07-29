"""Various laser wavelengths."""
from .conf import mkwvl, HeNe  # NOQA

from astropy import units as u

# IR
CO2 = mkwvl(10.6, u.um)
NdYAP = mkwvl(1080, u.nm)
NdYAG = mkwvl(1064, u.nm)
InGaAs = mkwvl(980, u.nm)

# VIS
Ruby = mkwvl(694, u.nm)
# HeNe
Cu = mkwvl(578, u.nm)

# UV / DUV / EUV / X-Ray
XeF = mkwvl(351, u.nm)
XeCl = mkwvl(308, u.nm)
KrF = mkwvl(248, u.nm)
KrCl = mkwvl(222, u.nm)
ArF = mkwvl(193, u.nm)

__all__ = [
    'CO2',
    'NdYAP',
    'NdYAG',
    'InGaAs',
    'Ruby',
    'HeNe',
    'Cu',
    'XeF',
    'XeCl',
    'KrF',
    'KrCl',
    'ArF',
    'mkwvl',
]
