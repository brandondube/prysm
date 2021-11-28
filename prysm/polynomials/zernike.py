"""Zernike polynomials."""

from collections import defaultdict

import numpy as truenp

from .jacobi import jacobi, jacobi_der, jacobi_sequence

from prysm.mathops import np, kronecker, sign, is_odd
from prysm.util import sort_xy
from prysm.plotting import share_fig_ax


def zernike_norm(n, m):
    """Norm of a Zernike polynomial with n, m indexing."""
    return truenp.sqrt((2 * (n + 1)) / (1 + kronecker(m, 0)))


def zero_separation(n):
    """Zero separation in normalized r based on radial order n."""
    return 1 / n ** 2


def zernike_nm(n, m, r, t, norm=True):
    """Zernike polynomial of radial order n, azimuthal order m at point r, t.

    Parameters
    ----------
    n : int
        radial order
    m : int
        azimuthal order
    r : numpy.ndarray
        radial coordinates
    t : numpy.ndarray
        azimuthal coordinates
    norm : bool, optional
        if True, orthonormalize the result (unit RMS)
        else leave orthogonal (zero-to-peak = 1)

    Returns
    -------
    numpy.ndarray
        zernike mode of order n,m at points r,t

    """
    x = 2 * r ** 2 - 1
    am = abs(m)
    n_j = (n - am) // 2
    out = jacobi(n_j, 0, am, x)
    if m != 0:
        if m < 0:
            out *= (r ** am * np.sin(am*t))
        else:
            out *= (r ** am * np.cos(m*t))

    if norm:
        out *= zernike_norm(n, m)

    return out


def zernike_nm_sequence(nms, r, t, norm=True):
    """Zernike polynomial of radial order n, azimuthal order m at point r, t.

    Parameters
    ----------
    nms : iterable of tuple of int,
        sequence of (n, m); looks like [(1,1), (3,1), ...]
    r : numpy.ndarray
        radial coordinates
    t : numpy.ndarray
        azimuthal coordinates
    norm : bool, optional
        if True, orthonormalize the result (unit RMS)
        else leave orthogonal (zero-to-peak = 1)

    Returns
    -------
    generator
        yields one mode at a time of nms

    """
    # this function deduplicates all possible work.  It uses a connection
    # to the jacobi polynomials to efficiently compute a series of zernike
    # polynomials
    # it follows this basic algorithm:
    # for each (n, m) compute the appropriate Jacobi polynomial order
    # collate the unique values of that for each |m|
    # compute a set of jacobi polynomials for each |m|
    # compute r^|m| , sin(|m|*t), and cos(|m|*t for each |m|
    #
    # benchmarked at 12.26 ns/element (256x256), 4.6GHz CPU = 56 clocks per element
    # ~36% faster than previous impl (12ms => 8.84 ms)
    x = 2 * r ** 2 - 1
    ms = [e[1] for e in nms]
    am = truenp.abs(ms)
    amu = truenp.unique(am)

    def factory():
        return 0

    jacobi_sequences_mjn = defaultdict(factory)
    # jacobi_sequences_mjn is a lookup table from |m| to all orders < max(n_j)
    # for each |m|, i.e. 0 .. n_j_max
    for nm, am_ in zip(nms, am):
        n = nm[0]
        nj = (n-am_) // 2
        if nj > jacobi_sequences_mjn[am_]:
            jacobi_sequences_mjn[am_] = nj

    for k in jacobi_sequences_mjn:
        nj = jacobi_sequences_mjn[k]
        jacobi_sequences_mjn[k] = truenp.arange(nj+1)

    jacobi_sequences = {}

    jacobi_sequences_mjn = dict(jacobi_sequences_mjn)
    for k in jacobi_sequences_mjn:
        n_jac = jacobi_sequences_mjn[k]
        jacobi_sequences[k] = list(jacobi_sequence(n_jac, 0, k, x))

    powers_of_m = {}
    sines = {}
    cosines = {}
    for m in amu:
        powers_of_m[m] = r ** m
        sines[m] = np.sin(m*t)
        cosines[m] = np.cos(m*t)

    for n, m in nms:
        absm = abs(m)
        nj = (n-absm) // 2
        jac = jacobi_sequences[absm][nj]
        if norm:
            jac = jac * zernike_norm(n, m)

        if m == 0:
            # rotationally symmetric Zernikes are jacobi
            yield jac
        else:
            if m < 0:
                azpiece = sines[absm]
            else:
                azpiece = cosines[absm]

            radialpiece = powers_of_m[absm]
            out = jac * azpiece * radialpiece  # jac already contains the norm
            yield out


def zernike_nm_der(n, m, r, t, norm=True):
    """Derivatives of Zernike polynomial of radial order n, azimuthal order m, w.r.t r and t.

    Parameters
    ----------
    n : int
        radial order
    m : int
        azimuthal order
    r : numpy.ndarray
        radial coordinates
    t : numpy.ndarray
        azimuthal coordinates
    norm : bool, optional
        if True, orthonormalize the result (unit RMS)
        else leave orthogonal (zero-to-peak = 1)

    Returns
    -------
    numpy.ndarray, numpy.ndarray
        dZ/dr, dZ/dt

    """
    # x = 2 * r ** 2 - 1
    # R = radial polynomial R_n^m, not dZ/dr
    # R = P_(n-m)//2^(0,|m|) (x)
    # = modified jacobi polynomial
    # dR = 4r R'(x) (chain rule)
    # => use jacobi_der
    # if m == 0, dZ = dR
    # for m != 0, Z = r^|m| * R * cos(mt)
    # the cosine term has no impact on the radial derivative,
    # for which we need the product rule:
    # d/dr(u v) = v(du/dr) + u(dv/dr)
    #      u    = R, which we already have the derivative of
    #        v  = r^|m| = r^k
    #     dv/dr = k r^(k-1)
    # d/dr(Z)   = r^k * (4r * R'(x)) + R * k r^(k-1)
    #             ------------------   -------------
    #                    v du              u dv
    #
    # all of that is multiplied by d/dr( cost ) or sint, which is just a "pass-through"
    # since cost does not depend on r
    #
    # in azimuth it's the other way around: regular old Zernike computation,
    # multiplied by d/dt ( cost )
    x = 2 * r ** 2 - 1
    am = abs(m)
    n_j = (n - am) // 2
    # dv from above == d/dr(R(2r^2-1))
    dv = (4*r) * jacobi_der(n_j, 0, am, x)
    if norm:
        znorm = zernike_norm(n, m)
    if m == 0:
        dr = dv
        dt = np.zeros_like(dv)
    else:
        v = jacobi(n_j, 0, am, x)
        u = r ** am
        du = am * r ** (am-1)
        dr = v * du + u * dv
        if m < 0:
            dt = am * np.cos(am*t)
            dr *= np.sin(am*t)
        else:
            dt = -m * np.sin(m*t)
            dr *= np.cos(m*t)

        # dt = dt * (u * v)
        # = cost * r^|m| * R
        # faster to write it as two in-place ops here
        # (no allocations)
        dt *= u
        dt *= v

        # ugly as this is, we skip one multiply
        # by doing these extra ifs
        if norm:
            dt *= znorm

    if norm:
        dr *= znorm

    return dr, dt


def zernike_nm_der_sequence(nms, r, t, norm=True):
    """Derivatives of Zernike polynomial of radial order n, azimuthal order m, w.r.t r and t.

    Parameters
    ----------
    nms : iterable
        sequence of [(n, m)] radial and azimuthal orders
    m : int
        azimuthal order
    r : numpy.ndarray
        radial coordinates
    t : numpy.ndarray
        azimuthal coordinates
    norm : bool, optional
        if True, orthonormalize the result (unit RMS)
        else leave orthogonal (zero-to-peak = 1)

    Returns
    -------
    list
        length (len(nms)) list of (dZ/dr, dZ/dt)

    """
    # TODO: actually implement the recurrence relation as in zernike_sequence,
    # instead of just using a loop for API homogenaeity
    out = []
    for n, m in nms:
        out.append(zernike_nm_der(n, m, r, t, norm=norm))

    return out


def nm_to_fringe(n, m):
    """Convert (n,m) two term index to Fringe index."""
    term1 = (1 + (n + abs(m))/2)**2
    term2 = 2 * abs(m)
    term3 = (1 + sign(m)) / 2
    return int(term1 - term2 - term3) + 1  # shift 0 base to 1 base


def nm_to_ansi_j(n, m):
    """Convert (n,m) two term index to ANSI single term index."""
    return int((n * (n + 2) + m) / 2)


def ansi_j_to_nm(idx):
    """Convert ANSI single term to (n,m) two-term index."""
    n = int(np.ceil((-3 + np.sqrt(9 + 8*idx))/2))
    m = 2 * idx - n * (n + 2)
    return n, m


def noll_to_nm(idx):
    """Convert Noll Z to (n, m) two-term index."""
    # I don't really understand this code, the math is inspired by POPPY
    # azimuthal order
    n = int(np.ceil((-1 + np.sqrt(1 + 8 * idx)) / 2) - 1)
    if n == 0:
        m = 0
    else:
        # this is sort of a rising factorial to use that term incorrectly
        nseries = int((n + 1) * (n + 2) / 2)
        res = idx - nseries - 1

        if is_odd(idx):
            sign = -1
        else:
            sign = 1

        if is_odd(n):
            ms = [1, 1]
        else:
            ms = [0]

        for i in range(n // 2):
            ms.append(ms[-1] + 2)
            ms.append(ms[-1])

        m = ms[res] * sign

    return n, m


def fringe_to_nm(idx):
    """Convert Fringe Z to (n, m) two-term index."""
    m_n = 2 * (np.ceil(np.sqrt(idx)) - 1)  # sum of n+m
    g_s = (m_n / 2)**2 + 1  # start of each group of equal n+m given as idx index
    n = m_n / 2 + np.floor((idx - g_s) / 2)
    m = (m_n - n) * (1 - np.mod(idx-g_s, 2) * 2)
    return int(n), int(m)


def zernikes_to_magnitude_angle_nmkey(coefs):
    """Convert Zernike polynomial set to a magnitude and phase representation.

    Parameters
    ----------
    coefs : list of tuples
        a list looking like[(1,2,3),] where (1,2) are the n, m indices and 3 the coefficient

    Returns
    -------
    dict
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
    coefs : list of tuples
        a list looking like[(1,2,3),] where (1,2) are the n, m indices and 3 the coefficient

    Returns
    -------
    dict
        dict keyed by friendly name strings with values of (rho, phi) where rho is the magnitudes, and phi the phase

    """
    d = zernikes_to_magnitude_angle_nmkey(coefs)
    d2 = {}
    for k, v in d.items():
        # (n,m) -> "Primary Coma X" -> ['Primary', 'Coma', 'X'] -> 'Primary Coma'
        name = nm_to_name(*k)
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


def nm_to_name(n, m):
    """Convert an (n,m) index into a human readable name.

    Parameters
    ----------
    n : int
        radial polynomial order
    m : int
        azimuthal polynomial order

    Returns
    -------
    str
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
    coefs : dict
        keys of (n,m), values of magnitudes, e.g. {(3,1): 2} represents 2 of primary coma
    n : int, optional
        identify the top n terms.

    Returns
    -------
    list
        list of tuples (magnitude, index, term)

    """
    coefsv = truenp.asarray(list(coefs.values()))
    coefs_work = abs(coefsv)
    oidxs = truenp.asarray(list(coefs.keys()))
    idxs = truenp.argpartition(coefs_work, -n)[-n:]  # argpartition does some magic to identify the top n (unsorted)
    idxs = idxs[truenp.argsort(coefs_work[idxs])[::-1]]  # use argsort to sort them in ascending order and reverse
    big_terms = coefsv[idxs]  # finally, take the values from the
    names = [nm_to_name(*p) for p in oidxs]
    names = truenp.asarray(names)[idxs]  # p = pair (n,m)
    return list(zip(big_terms, idxs, names))


def barplot(coefs, names=None, orientation='h', buffer=1, zorder=3, number=True,
            offset=0, width=0.8, fig=None, ax=None):
    """Create a barplot of coefficients and their names.

    Parameters
    ----------
    coefs : dict
        with keys of Zn, values of numbers
    names : dict
        with keys of Zn, values of names (e.g. Primary Coma X)
    orientation : str, {'h', 'v', 'horizontal', 'vertical'}
        orientation of the plot
    buffer : float, optional
        buffer to use around the left and right (or top and bottom) bars
    zorder : int, optional
        zorder of the bars.  Use zorder > 3 to put bars in front of gridlines
    number : bool, optional
        if True, plot numbers along the y=0 line showing indices
    offset : float, optional
        offset to apply to bars, useful for before/after Zernike breakdowns
    width : float, optional
        width of bars, useful for before/after Zernike breakdowns
    fig : matplotlib.figurnp.Figure
        Figure containing the plot
    ax : matplotlib.axes.Axis
        Axis containing the plot

    Returns
    -------
    fig : matplotlib.figurnp.Figure
        Figure containing the plot
    ax : matplotlib.axes.Axis
        Axis containing the plot

    """
    from matplotlib import pyplot as plt
    fig, ax = share_fig_ax(fig, ax)

    coefs2 = np.asarray(list(coefs.values()))
    idxs = np.asarray(list(coefs.keys()))
    coefs = coefs2
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
    magnitudes : dict
        keys of names, values of magnitudes.  E.g., {'Primary Coma': 1234567}
    orientation : str, {'h', 'v', 'horizontal', 'vertical'}
        orientation of the plot
    sort : bool, optional
        whether to sort the zernikes in descending order
    buffer : float, optional
        buffer to use around the left and right (or top and bottom) bars
    zorder : int, optional
        zorder of the bars.  Use zorder > 3 to put bars in front of gridlines
    offset : float, optional
        offset to apply to bars, useful for before/after Zernike breakdowns
    width : float, optional
        width of bars, useful for before/after Zernike breakdowns
    fig : matplotlib.figure.Figure
        Figure containing the plot
    ax : matplotlib.axes.Axis
        Axis containing the plot

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the plot
    ax : matplotlib.axes.Axis
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
