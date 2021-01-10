"""Analysis routines for slicing and dicing Zernike coefficient sets."""
from collections import defaultdict

import numpy as np  # not mathops -- nothing in this file operates on anything worthwhile for a GPU

from prysm.util import sort_xy
from prysm.mathops import is_odd, sign
from prysm.plotting import share_fig_ax


def zernikes_to_magnitude_angle_nmkey(coefs):
    """Convert Zernike polynomial set to a magnitude and phase representation.

    Parameters
    ----------
    coefs : `list` of `tuples`
        a list looking like[(1,2,3),] where (1,2) are the n, m indices and 3 the coefficient

    Returns
    -------
    `dict`
        dict keyed by tuples of (n, |m|) with values of (rho, phi) where rho is the magnitudes, and phi the phase

    """
    def mkary():  # default for defaultdict
        return list()

    combinations = defaultdict(mkary)

    # for each name and coefficient, make a len 2 array.  Put the Y or 0 degree values in the first slot
    for n, m, coef in coefs:
        m2 = abs(m)
        key = (n, m2)
        combinations[key].append(coef)

    for key, value in combinations.items():
        if len(value) == 1:
            magnitude = value[0]
            angle = 0
        else:
            magnitude = np.sqrt(sum([v**2 for v in value]))
            angle = np.degrees(np.arctan2(*value))

        combinations[key] = (magnitude, angle)

    return dict(combinations)


def zernikes_to_magnitude_angle(coefs):
    """Convert Zernike polynomial set to a magnitude and phase representation.

    This function is identical to zernikes_to_magnitude_angle_nmkey, except its keys are strings instead of (n, |m|)

    Parameters
    ----------
    coefs : `list` of `tuples`
        a list looking like[(1,2,3),] where (1,2) are the n, m indices and 3 the coefficient

    Returns
    -------
    `dict`
        dict keyed by friendly name strings with values of (rho, phi) where rho is the magnitudes, and phi the phase

    """
    d = zernikes_to_magnitude_angle_nmkey(coefs)
    d2 = {}
    for k, v in d.items():
        # (n,m) -> "Primary Coma X" -> ['Primary', 'Coma', 'X'] -> 'Primary Coma'
        name = n_m_to_name(*k)
        split = name.split(" ")
        if len(split) < 3 and 'Tilt' not in name:  # oh, how special the low orders are
            k2 = name
        else:
            k2 = " ".join(split[:-1])

        d2[k2] = v

    return d2


_names = {
    1: 'Primary',
    2: 'Secondary',
    3: 'Tertiary',
    4: 'Quaternary',
    5: 'Quinary',
}

_names_m = {
    1: 'Coma',
    2: 'Astigmatism',
    3: 'Trefoil',
    4: 'Quadrafoil',
    5: 'Pentafoil',
    6: 'Hexafoil',
    7: 'Septafoil',
    8: 'Octafoil',
}


def _name_accessor(n, m):
    """Convert n, m to "order" n, where Order is 1 primary, 2 secondary, etc.

    "order" is a key to _names

    """
    if m == 0 and n >= 4:
        return int((n / 2) + 1)
    if is_odd(m) and n >= 3:
        return abs(int((n - 3) / 2 + 1))
    else:
        return int(n / abs(m))


def _name_helper(n, m):
    accessor = _name_accessor(n, m)
    prefix = _names.get(accessor, f'{accessor}th')
    name = _names_m.get(abs(m), f'{abs(m)}-foil')
    if n == 1:
        name = 'Tilt'

    if is_odd(m):
        if sign(m) == 1:
            suffix = 'X'
        else:
            suffix = 'Y'
    else:
        if sign(m) == 1:
            suffix = '00°'
        else:
            suffix = '45°'

    return f'{prefix} {name} {suffix}'


def n_m_to_name(n, m):
    """Convert an (n,m) index into a human readable name.

    Parameters
    ----------
    n : `int`
        radial polynomial order
    m : `int`
        azimuthal polynomial order

    Returns
    -------
    `str`
        a name, np.g. Piston or Primary Spherical

    """
    # piston, tip tilt, az invariant order
    if n == 0:
        return 'Piston'
    if n == 1:
        if sign(m) == 1:
            return 'Tilt X'
        else:
            return 'Tilt Y'
    if n == 2 and m == 0:
        return 'Defocus'
    if m == 0:
        accessor = int((n / 2) - 1)
        prefix = _names.get(accessor, f'{accessor}th')
        return f'{prefix} Spherical'
    return _name_helper(n, m)


def top_n(coefs, n=5):
    """Identify the top n terms in the wavefront expansion.

    Parameters
    ----------
    coefs : `dict`
        keys of (n,m), values of magnitudes, e.g. {(3,1): 2} represents 2 of primary coma
    n : `int`, optional
        identify the top n terms.

    Returns
    -------
    `list`
        list of tuples (magnitude, index, term)

    """
    coefsv = np.asarray(coefs.values())
    coefs_work = abs(coefsv)
    oidxs = np.asarray(list(coefs.keys()))
    idxs = np.argpartition(coefs_work, -n)[-n:]  # argpartition does some magic to identify the top n (unsorted)
    idxs = idxs[np.argsort(coefs_work[idxs])[::-1]]  # use argsort to sort them in ascending order and reverse
    big_terms = coefs[idxs]  # finally, take the values from the
    big_idxs = oidxs[idxs]
    names = [n_m_to_name(*p) for p in oidxs][idxs]  # p = pair (n,m)
    return list(zip(big_terms, big_idxs, names))


def barplot(coefs, names=None, orientation='h', buffer=1, zorder=3, number=True, offset=0, width=0.8, fig=None, ax=None):
    """Create a barplot of coefficients and their names.

    Parameters
    ----------
    coefs : `dict`
        with keys of Zn, values of numbers
    names : `dict`
        with keys of Zn, values of names (e.g. Primary Coma X)
    orientation : `str`, {'h', 'v', 'horizontal', 'vertical'}
        orientation of the plot
    buffer : `float`, optional
        buffer to use around the left and right (or top and bottom) bars
    zorder : `int`, optional
        zorder of the bars.  Use zorder > 3 to put bars in front of gridlines
    number : `bool`, optional
        if True, plot numbers along the y=0 line showing indices
    offset : `float`, optional
        offset to apply to bars, useful for before/after Zernike breakdowns
    width : `float`, optional
        width of bars, useful for before/after Zernike breakdowns
    fig : `matplotlib.figurnp.Figure`
        Figure containing the plot
    ax : `matplotlib.axes.Axis`
        Axis containing the plot

    Returns
    -------
    fig : `matplotlib.figurnp.Figure`
        Figure containing the plot
    ax : `matplotlib.axes.Axis`
        Axis containing the plot

    """
    from matplotlib import pyplot as plt
    fig, ax = share_fig_ax(fig, ax)

    coefs = np.asarray(list(coefs.values()))
    idxs = np.asarray(list(coefs.keys()))
    lims = (idxs[0] - buffer, idxs[-1] + buffer)
    if orientation.lower() in ('h', 'horizontal'):
        vmin, vmax = coefs.min(), coefs.max()
        drange = vmax - vmin
        offsetY = drange * 0.01

        ax.bar(idxs + offset, coefs, zorder=zorder, width=width)
        plt.xticks(idxs, names, rotation=90)
        if number:
            for i in idxs:
                ax.text(i, offsetY, str(i), ha='center')
    else:
        ax.barh(idxs + offset, coefs, zorder=zorder, height=width)
        plt.yticks(idxs, names)
        if number:
            for i in idxs:
                ax.text(0, i, str(i), ha='center')

    ax.set(xlim=lims)
    return fig, ax


def barplot_magnitudes(magnitudes, orientation='h', sort=False,
                       buffer=1, zorder=3, offset=0, width=0.8,
                       fig=None, ax=None):
    """Create a barplot of magnitudes of coefficient pairs and their names.

    e.g., astigmatism will get one bar.

    Parameters
    ----------
    magnitudes : `dict`
        keys of names, values of magnitudes.  E.g., {'Primary Coma': 1234567}
    orientation : `str`, {'h', 'v', 'horizontal', 'vertical'}
        orientation of the plot
    sort : `bool`, optional
        whether to sort the zernikes in descending order
    buffer : `float`, optional
        buffer to use around the left and right (or top and bottom) bars
    zorder : `int`, optional
        zorder of the bars.  Use zorder > 3 to put bars in front of gridlines
    offset : `float`, optional
        offset to apply to bars, useful for before/after Zernike breakdowns
    width : `float`, optional
        width of bars, useful for before/after Zernike breakdowns
    fig : `matplotlib.figure.Figure`
        Figure containing the plot
    ax : `matplotlib.axes.Axis`
        Axis containing the plot

    Returns
    -------
    fig : `matplotlib.figure.Figure`
        Figure containing the plot
    ax : `matplotlib.axes.Axis`
        Axis containing the plot

    """
    from matplotlib import pyplot as plt

    mags = magnitudes.values()
    names = magnitudes.keys()
    idxs = np.arange(len(names))
    # idxs = np.asarray(list(range(len(names))))

    if sort:
        mags, names = sort_xy(mags, names)
        mags = list(reversed(mags))
        names = list(reversed(names))

    lims = (idxs[0] - buffer, idxs[-1] + buffer)
    fig, ax = share_fig_ax(fig, ax)
    if orientation.lower() in ('h', 'horizontal'):
        ax.bar(idxs + offset, mags, zorder=zorder, width=width)
        plt.xticks(idxs, names, rotation=90)
        ax.set(xlim=lims)
    else:
        ax.barh(idxs + offset, mags, zorder=zorder, height=width)
        plt.yticks(idxs, names)
        ax.set(ylim=lims)
    return fig, ax
