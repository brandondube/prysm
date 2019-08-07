"""Various laser wavelengths."""

from astropy import units as u


def mkwvl(quantity, base=u.um):
    """Generate a new Wavelength unit.

    Parameters
    ----------
    quantity : `float` or `astropy.units.unit`
        number of (base) for the wavelength, e.g. quantity=632.8 with base=u.nm for HeNe.
        if an astropy unit, simply returned by this function
    base : `astropy.units.Unit`
        base unit, e.g. um or nm

    Returns
    -------
    `astropy.units.Unit`
        new Unit for appropriate wavelength

    """
    if not isinstance(quantity, u.Unit):
        return u.def_unit(['wave', 'wavelength'], quantity * base,
                          format={'latex': r'\lambda', 'unicode': 'λ'})
    else:
        return quantity


# IR
CO2 = mkwvl(10.6, u.um)
NdYAP = mkwvl(1080, u.nm)
NdYAG = mkwvl(1064, u.nm)
InGaAs = mkwvl(980, u.nm)

# VIS
Ruby = mkwvl(694, u.nm)
HeNe = mkwvl(632.8, u.nm)
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
