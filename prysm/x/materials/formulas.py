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


def agf_formula(formula, wvl_um, *coefficients, name='material'):
    """Evaluate the AGF formula subset used by current prysm fixtures.

    Coefficients follow the wavelength positionally so that
    partial(agf_formula, formula_id) matches the FormulaMaterial calling
    convention formula(wvl_um, coef0, coef1, ...).
    """
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


def riinfo_formula(formula_id, wvl_um, *coefficients):
    """Evaluate refractiveindex.info dispersion formulas 1-9, wavelength in um.

    Mirrors the refractiveindex package's _compute_formula, but evaluated
    through mathops.np so the consumption path stays backend-pure and
    differentiable.  Coefficients follow the wavelength positionally so that
    partial(riinfo_formula, formula_id) matches the FormulaMaterial calling
    convention formula(wvl_um, coef0, coef1, ...).
    """
    C = coefficients
    # pad so the fixed-index formulas (4, 7, 8, 9) tolerate short coefficient
    # lists; Cp[5] (formula 9) is the highest fixed index read.
    Cp = list(C) + [0.0] * 6
    wl = wvl_um
    if formula_id == 1:  # Sellmeier
        nsq = 1 + Cp[0]
        for i in range(1, len(C), 2):
            nsq = nsq + C[i] * wl ** 2 / (wl ** 2 - C[i + 1] ** 2)
        return np.sqrt(nsq)
    if formula_id == 2:  # Sellmeier-2
        nsq = 1 + Cp[0]
        for i in range(1, len(C), 2):
            nsq = nsq + C[i] * wl ** 2 / (wl ** 2 - C[i + 1])
        return np.sqrt(nsq)
    if formula_id == 3:  # Polynomial
        nsq = Cp[0]
        for i in range(1, len(C), 2):
            nsq = nsq + C[i] * wl ** C[i + 1]
        return np.sqrt(nsq)
    if formula_id == 4:  # RefractiveIndex.INFO
        nsq = Cp[0]
        for i in range(1, min(8, len(C)), 4):
            nsq = nsq + C[i] * wl ** C[i + 1] / (wl ** 2 - C[i + 2] ** C[i + 3])
        if len(C) > 9:
            for i in range(9, len(C), 2):
                nsq = nsq + C[i] * wl ** C[i + 1]
        return np.sqrt(nsq)
    if formula_id == 5:  # Cauchy
        n = Cp[0]
        for i in range(1, len(C), 2):
            n = n + C[i] * wl ** C[i + 1]
        return n
    if formula_id == 6:  # Gases
        n = 1 + Cp[0]
        for i in range(1, len(C), 2):
            n = n + C[i] / (C[i + 1] - wl ** (-2))
        return n
    if formula_id == 7:  # Herzberger
        n = Cp[0]
        n = n + Cp[1] / (wl ** 2 - 0.028)
        n = n + Cp[2] / (wl ** 2 - 0.028) ** 2
        for i in range(3, len(C)):
            n = n + C[i] * wl ** (2 * (i - 2))
        return n
    if formula_id == 8:  # Retro
        tmp = Cp[0] + Cp[1] * wl ** 2 / (wl ** 2 - Cp[2]) + Cp[3] * wl ** 2
        return np.sqrt((2 * tmp + 1) / (1 - tmp))
    if formula_id == 9:  # Exotic
        return np.sqrt(
            Cp[0]
            + Cp[1] / (wl ** 2 - Cp[2])
            + Cp[3] * (wl - Cp[4]) / ((wl - Cp[4]) ** 2 + Cp[5])
        )
    raise ValueError(f'unknown refractiveindex.info dispersion formula {formula_id}')
