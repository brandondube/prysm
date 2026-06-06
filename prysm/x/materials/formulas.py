"""Dispersion formula helpers for material models."""

from prysm.mathops import np
from prysm.refractive import cauchy as _cauchy
from prysm.refractive import sellmeier as _sellmeier


def cauchy(wvl_um, A, *args):
    """Cauchy equation using wavelength in microns."""
    return _cauchy(wvl_um, A, *args)


def sellmeier(wvl_um, A, B):
    """Sellmeier equation using wavelength in microns."""
    return _sellmeier(wvl_um, A, B)


def sellmeier_interleaved(wvl_um, *coefficients):
    """Sellmeier equation with A1, B1, A2, B2 coefficient order."""
    return _sellmeier(wvl_um, coefficients[0::2], coefficients[1::2])


def schott(wvl_um, c0, c1, c2, c3, c4, c5):
    """Schott glass equation used by AGF formula 1."""
    wvlsq = np.square(wvl_um)
    n_squared = (
        c0
        + c1 * wvlsq
        + c2 / wvlsq
        + c3 / wvlsq ** 2
        + c4 / wvlsq ** 3
        + c5 / wvlsq ** 4
    )
    return np.sqrt(n_squared)


def extended2(wvl_um, c0, c1, c2, c3, c4, c5, c6, c7):
    """AGF extended formula 2."""
    wvlsq = np.square(wvl_um)
    n_squared = (
        c0
        + c1 * wvlsq
        + c2 / wvlsq
        + c3 / wvlsq ** 2
        + c4 / wvlsq ** 3
        + c5 / wvlsq ** 4
        + c6 * wvlsq ** 2
        + c7 * wvlsq ** 3
    )
    return np.sqrt(n_squared)


def extended3(wvl_um, c0, c1, c2, c3, c4, c5, c6, c7, c8):
    """AGF extended formula 3."""
    wvlsq = np.square(wvl_um)
    n_squared = (
        c0
        + c1 * wvlsq
        + c2 * wvlsq ** 2
        + c3 / wvlsq
        + c4 / wvlsq ** 2
        + c5 / wvlsq ** 3
        + c6 / wvlsq ** 4
        + c7 / wvlsq ** 5
        + c8 / wvlsq ** 6
    )
    return np.sqrt(n_squared)


def agf_formula(formula, coefficients, wvl_um, name='material'):
    """Evaluate the AGF formula subset used by current prysm fixtures."""
    if formula == 1:
        if len(coefficients) < 6:
            raise ValueError(f'AGF Schott formula glass {name} requires six coefficients')
        return schott(wvl_um, *coefficients[:6])
    if formula == 2:
        return _agf_sellmeier(coefficients, wvl_um, name, terms=3)
    if formula == 6:
        return _agf_sellmeier(coefficients, wvl_um, name, terms=4)
    if formula == 12:
        if len(coefficients) < 8:
            raise ValueError(f'AGF Extended 2 formula glass {name} requires eight coefficients')
        return extended2(wvl_um, *coefficients[:8])
    if formula == 13:
        if len(coefficients) < 9:
            raise ValueError(f'AGF Extended 3 formula glass {name} requires nine coefficients')
        return extended3(wvl_um, *coefficients[:9])
    raise NotImplementedError(
        f'AGF dispersion formula {formula} for {name} is not implemented'
    )


def _agf_sellmeier(coefficients, wvl_um, name, terms):
    needed = terms * 2
    if len(coefficients) < needed:
        raise ValueError(f'AGF Sellmeier glass {name} requires {needed} coefficients')
    pairs = coefficients[:needed]
    return _sellmeier(wvl_um, pairs[0::2], pairs[1::2])
