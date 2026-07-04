"""Common coating material tokens by spectral band and application."""

from ..materials import glass as _glass


BANDS = {
    'VIS': (0.4, 0.7),
    'VIS-NIR': (0.4, 1.1),
    'VIS-NIR-SWIR': (0.4, 2.5),
    'MWIR': (3.0, 5.0),
    'LWIR': (8.0, 12.0),
}

ANTIREFLECTION = {
    'VIS': {
        'low': ('MgF2', 'SiO2'),
        'mid': ('Al2O3',),
        'high': (('TiO2', 'Sarkar'), ('Ta2O5', 'Gao')),
    },
    'VIS-NIR': {
        'low': ('MgF2', 'SiO2'),
        'mid': ('Al2O3', 'HfO2'),
        'high': ('Nb2O5', ('Ta2O5', 'Gao')),
    },
    'VIS-NIR-SWIR': {
        'low': (('SiO2', 'Malitson'), 'MgF2'),
        'mid': ('Al2O3', ('HfO2', 'Franta'), ('ZrO2', 'Wood')),
        'high': (('Ta2O5', 'Franta-2015'),),
    },
    'MWIR': {
        'low': ('YbF3', ('SiO', 'Hass')),
        'mid': ('ZnS',),
        'high': ('Ge', ('Si', 'Chandler-Horowitz')),
    },
    'LWIR': {
        'low': ('YbF3', ('BaF2', 'Li')),
        'mid': ('ZnS', ('ZnSe', 'Amotchkina')),
        'high': ('Ge',),
    },
}

BANDPASS = {
    'VIS': {
        'low': ('SiO2',),
        'high': (('TiO2', 'Sarkar'), ('Ta2O5', 'Gao')),
    },
    'VIS-NIR': {
        'low': ('SiO2',),
        'high': ('Nb2O5', ('Ta2O5', 'Gao')),
    },
    'VIS-NIR-SWIR': {
        'low': (('SiO2', 'Malitson'),),
        'high': (('Ta2O5', 'Franta-2015'), ('Si', 'Franta-25C')),
    },
    'MWIR': {
        'low': (('SiO', 'Hass'), 'ZnS'),
        'high': ('Ge',),
    },
    'LWIR': {
        'low': ('ZnS', ('ZnSe', 'Amotchkina')),
        'high': (('PbTe', 'Weiting-300K'), 'Ge'),
    },
}

MIRROR = {
    'VIS': {
        'metal': ('Al', 'Ag'),
        'barrier': ('Al2O3', 'Si3N4'),
        'low': ('SiO2',),
        'high': (('TiO2', 'Sarkar'), 'Nb2O5'),
    },
    'VIS-NIR': {
        'metal': ('Ag', 'Au'),
        'barrier': ('Al2O3', 'Si3N4'),
        'low': ('SiO2',),
        'high': ('Nb2O5', ('Ta2O5', 'Gao')),
    },
    'VIS-NIR-SWIR': {
        'metal': ('Ag',),
        'barrier': ('Al2O3',),
        'low': (('SiO2', 'Malitson'),),
        'high': (('Ta2O5', 'Franta-2015'),),
    },
    'MWIR': {
        'metal': ('Au',),
        'barrier': ('Al2O3',),
        'low': ('YbF3',),
        'high': ('ZnS',),
    },
    'LWIR': {
        'metal': ('Au', ('Al', 'Rakic')),
        'barrier': (),
        'low': ('YbF3',),
        'high': ('ZnS', ('ZnSe', 'Amotchkina')),
    },
}

APPLICATIONS = {
    'AR': ANTIREFLECTION,
    'ANTIREFLECTION': ANTIREFLECTION,
    'BANDPASS': BANDPASS,
    'MIRROR': MIRROR,
}


def names(application, band):
    """Common material tokens for an application and band.

    Parameters
    ----------
    application : str
        AR (or ANTIREFLECTION), BANDPASS, or MIRROR; case-insensitive.
    band : str
        a key of BANDS; case-insensitive.

    Returns
    -------
    dict
        role -> tuple of book names or (book, page) pairs.

    """
    table = APPLICATIONS[application.upper()]
    return table[band.upper()]


def materials(application, band, database=None):
    """Common materials for an application and band, resolved to callables.

    Parameters
    ----------
    application : str
        AR (or ANTIREFLECTION), BANDPASS, or MIRROR; case-insensitive.
    band : str
        a key of BANDS; case-insensitive.
    database : material catalog, optional
        forwarded to x/materials glass.

    Returns
    -------
    dict
        role -> tuple of materials, in the same order as names.

    """
    table = names(application, band)
    return {
        role: tuple(_resolve(token, database) for token in members)
        for role, members in table.items()
    }


def _resolve(token, database):
    """Resolve a book name or (book, page) token to a material."""
    if isinstance(token, tuple):
        book, page = token
        return _glass(book, database=database, page=page)
    return _glass(token, database=database)
