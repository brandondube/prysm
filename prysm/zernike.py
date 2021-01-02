"""Zernike functions."""
from collections import defaultdict

from .conf import config
from .mathops import engine as np, kronecker, sign
from .util import rms, sort_xy, is_odd
from .plotting import share_fig_ax
from .jacobi import jacobi, jacobi_sequence

# See JCW - http://wp.optics.arizona.edu/jcwyant/wp-content/uploads/sites/13/2016/08/ZernikePolynomialsForTheWeb.pdf


def piston(rho, phi):
    """Zernike Piston."""
    return np.ones(rho.shape)


def tip(rho, phi):
    """Zernike Tilt-Y."""
    return rho * np.cos(phi)


def tilt(rho, phi):
    """Zernike Tilt-X."""
    return rho * np.sin(phi)


def defocus(rho, phi):
    """Zernike defocus."""
    return 2 * rho**2 - 1


def primary_astigmatism_00(rho, phi):
    """Zernike primary astigmatism 0°."""
    return rho**2 * np.cos(2 * phi)


def primary_astigmatism_45(rho, phi):
    """Zernike primary astigmatism 45°."""
    return rho**2 * np.sin(2 * phi)


def primary_coma_y(rho, phi):
    """Zernike primary coma Y."""
    return (3 * rho**3 - 2 * rho) * np.cos(phi)


def primary_coma_x(rho, phi):
    """Zernike primary coma X."""
    return (3 * rho**3 - 2 * rho) * np.sin(phi)


def primary_spherical(rho, phi):
    """Zernike primary Spherical."""
    return 6 * rho**4 - 6 * rho**2 + 1


def primary_trefoil_y(rho, phi):
    """Zernike primary trefoil Y."""
    return rho**3 * np.cos(3 * phi)


def primary_trefoil_x(rho, phi):
    """Zernike primary trefoil X."""
    return rho**3 * np.sin(3 * phi)


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


def zernike_norm(n, m):
    """Norm of a Zernike polynomial with n, m indexing."""
    return np.sqrt((2 * (n + 1)) / (1 + kronecker(m, 0)))


def n_m_to_fringe(n, m):
    """Convert (n,m) two term index to Fringe index."""
    term1 = (1 + (n + abs(m))/2)**2
    term2 = 2 * abs(m)
    term3 = (1 + sign(m)) / 2
    return int(term1 - term2 - term3) + 1  # shift 0 base to 1 base


def n_m_to_ansi_j(n, m):
    """Convert (n,m) two term index to ANSI single term index."""
    return int((n * (n + 2) + m) / 2)


def ansi_j_to_n_m(idx):
    """Convert ANSI single term to (n,m) two-term index."""
    n = int(np.ceil((-3 + np.sqrt(9 + 8*idx))/2))
    m = 2 * idx - n * (n + 2)
    return n, m


def noll_to_n_m(idx):
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


def fringe_to_n_m(idx):
    """Convert Fringe Z to (n, m) two-term index."""
    m_n = 2 * (np.ceil(np.sqrt(idx)) - 1)  # sum of n+m
    g_s = (m_n / 2)**2 + 1  # start of each group of equal n+m given as idx index
    n = m_n / 2 + np.floor((idx - g_s) / 2)
    m = (m_n - n) * (1 - np.mod(idx-g_s, 2) * 2)
    return int(n), int(m)


def zero_separation(n):
    """Zero separation in normalized r based on radial order n."""
    return 1 / n ** 2


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


def zernike_nm(n, m, r, t, norm=True):
    """Zernike polynomial of radial order n, azimuthal order m at point r, t.

    Parameters
    ----------
    n : `int`
        radial order
    m : `int`
        azimuthal order
    r : `numpy.ndarray`
        radial coordinates
    t : `numpy.ndarray`
        azimuthal coordinates
    norm : `bool`, optional
        if True, orthonormalize the result (unit RMS)
        else leave orthogonal (zero-to-peak = 1)

    """
    x = 2 * r ** 2 - 1
    am = abs(m)
    n_j = (n - am) // 2
    out = jacobi(n_j, 0, am, x)
    if m != 0:
        if m < 0:
            out *= (r ** am * np.sin(m*t))
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
    r : `numpy.ndarray`
        radial coordinates
    t : `numpy.ndarray`
        azimuthal coordinates
    norm : `bool`, optional
        if True, orthonormalize the result (unit RMS)
        else leave orthogonal (zero-to-peak = 1)

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
    ms = list(e[1] for e in nms)
    am = np.abs(ms)
    amu = np.unique(am)

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
        jacobi_sequences_mjn[k] = np.arange(nj+1)

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



nm_funcs = {
    'Fringe': fringe_to_n_m,
    'Noll': noll_to_n_m,
    'ANSI': ansi_j_to_n_m,
}


class BaseZernike(Pupil):
    """Basic class implementing Zernike features."""
    _name = None
    _cache = zcachemn

    def __init__(self, *args, **kwargs):
        """Initialize a new Zernike instancnp."""
        self.coefs = {}

        self.normalize = False
        pass_args = {}

        if args is not None:
            if len(args) == 1:
                enumerator = args[0]
            else:
                enumerator = args

            for idx, coef in enumerate(enumerator):
                self.coefs[idx+1] = coef

        if kwargs is not None:
            for key, value in kwargs.items():
                if key[0].lower() == 'z' and key[1].isnumeric():
                    idx = int(key[1:])  # strip 'Z' from index
                    self.coefs[idx] = value
                elif key.lower() == 'norm':
                    self.normalize = value
                else:
                    pass_args[key] = value

        super().__init__(**pass_args)

    def build(self):
        """Use the wavefront coefficients stored in this class instance to build a wavefront model.

        Returns
        -------
        self : `BaseZernike`
            this Zernike instance

        """
        nm_func = nm_funcs.get(self._name, None)
        if nm_func is None:
            raise ValueError("single index notation not understood, modify zerniknp.nm_funcs")

        # build a coordinate system over which to evaluate this function
        self.data = np.zeros((self.samples, self.samples), dtype=config.precision)
        keys = list(sorted(self.coefs.keys()))

        for term in keys:
            coef = self.coefs[term]
            # short circuit for speed
            if coef == 0:
                continue
            else:
                n, m = nm_func(term)
                term = self._cache(n, m, self.samples, self.normalize)
                self.data += coef * term

        return self

    def top_n(self, n=5):
        """Identify the top n terms in the wavefront.

        Parameters
        ----------
        n : `int`, optional
            identify the top n terms.

        Returns
        -------
        `list`
            list of tuples (magnitude, index, term)

        """
        coefs = np.asarray(list(self.coefs.values()))
        coefs_work = abs(coefs)
        oidxs = np.asarray(list(self.coefs.keys()))
        idxs = np.argpartition(coefs_work, -n)[-n:]  # argpartition does some magic to identify the top n (unsorted)
        idxs = idxs[np.argsort(coefs_work[idxs])[::-1]]  # use argsort to sort them in ascending order and reverse
        big_terms = coefs[idxs]  # finally, take the values from the
        big_idxs = oidxs[idxs]
        names = np.asarray(self.names, dtype=str)[big_idxs - 1]
        return list(zip(big_terms, big_idxs, names))

    @property
    def magnitudes(self):
        """Return the magnitude and angles of the zernike components in this wavefront."""
        # need to call through class variable to avoid insertion of self as arg
        nmf = nm_funcs[self._name]
        inp = []
        for k, v in self.coefs.items():
            tup = (*nmf(k), v)
            inp.append(tup)

        return zernikes_to_magnitude_angle(inp)

    @property
    def names(self):
        """Names of the terms in self."""
        # need to call through class variable to avoid insertion of self as arg
        nmf = nm_funcs[self._name]
        return [n_m_to_name(*nmf(i)) for i in self.coefs.keys()]

    def barplot(self, orientation='h', buffer=1, zorder=3, number=True, offset=0, width=0.8, fig=None, ax=None):
        """Create a barplot of coefficients and their names.

        Parameters
        ----------
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

        coefs = np.asarray(list(self.coefs.values()))
        idxs = np.asarray(list(self.coefs.keys()))
        names = self.names
        lab = self.labels.z(self.xy_unit, self.z_unit)
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
            ax.set(ylabel=lab, xlim=lims)
        else:
            ax.barh(idxs + offset, coefs, zorder=zorder, height=width)
            plt.yticks(idxs, names)
            if number:
                for i in idxs:
                    ax.text(0, i, str(i), ha='center')
            ax.set(xlabel=lab, ylim=lims)
        return fig, ax

    def barplot_magnitudes(self, orientation='h', sort=False,
                           buffer=1, zorder=3, offset=0, width=0.8,
                           fig=None, ax=None):
        """Create a barplot of magnitudes of coefficient pairs and their names.

        np.g., astigmatism will get one bar.

        Parameters
        ----------
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

        magang = self.magnitudes
        mags = [m[0] for m in magang.values()]
        names = magang.keys()
        idxs = np.asarray(list(range(len(names))))

        if sort:
            mags, names = sort_xy(mags, names)
            mags = list(reversed(mags))
            names = list(reversed(names))
        lab = self.labels.z(self.xy_unit, self.z_unit)
        lims = (idxs[0] - buffer, idxs[-1] + buffer)
        fig, ax = share_fig_ax(fig, ax)
        if orientation.lower() in ('h', 'horizontal'):
            ax.bar(idxs + offset, mags, zorder=zorder, width=width)
            plt.xticks(idxs, names, rotation=90)
            ax.set(ylabel=lab, xlim=lims)
        else:
            ax.barh(idxs + offset, mags, zorder=zorder, height=width)
            plt.yticks(idxs, names)
            ax.set(xlabel=lab, ylim=lims)
        return fig, ax

    def barplot_topn(self, n=5, orientation='h', buffer=1, zorder=3, fig=None, ax=None):
        """Plot the top n terms in the wavefront.

        Parameters
        ----------
        n : `int`, optional
            plot the top n terms.
        orientation : `str`, {'h', 'v', 'horizontal', 'vertical'}
            orientation of the plot
        buffer : `float`, optional
            buffer to use around the left and right (or top and bottom) bars
        zorder : `int`, optional
            zorder of the bars.  Use zorder > 3 to put bars in front of gridlines
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

        topn = self.top_n(n)
        magnitudes = [n[0] for n in topn]
        names = [n[2] for n in topn]
        idxs = range(len(names))

        fig, ax = share_fig_ax(fig, ax)

        lab = self.labels.z(self.xy_unit, self.z_unit)
        lims = (idxs[0] - buffer, idxs[-1] + buffer)
        if orientation.lower() in ('h', 'horizontal'):
            ax.bar(idxs, magnitudes, zorder=zorder)
            plt.xticks(idxs, names, rotation=90)
            ax.set(ylabel=lab, xlim=lims)
        else:
            ax.barh(idxs, magnitudes, zorder=zorder)
            plt.yticks(idxs, names)
            ax.set(xlabel=lab, ylim=lims)
        return fig, ax

    def truncate(self, n):
        """Truncate the wavefront to the first n terms.

        Parameters
        ----------
        n : `int`
            number of terms to keep.

        Returns
        -------
        `self`
            modified FringeZernike instancnp.

        """
        if n > len(self.coefs):
            return self
        else:
            coefs = {}
            for idx, i in enumerate(sorted(self.coefs.keys())):
                if idx > n:
                    break
                coefs[i] = self.coefs[i]

            self.coefs = coefs
            self.build()
            return self

    def truncate_topn(self, n):
        """Truncate the pupil to only the top n terms.

        Parameters
        ----------
        n : `int`
            number of parameters to keep

        Returns
        -------
        `self`
            modified FringeZernike instancnp.

        """
        topn = self.top_n(n)
        new_coefs = {}
        for coef in topn:
            mag, index, *_ = coef
            new_coefs[index] = mag

        self.coefs = new_coefs
        self.build()
        return self

    def __str__(self):
        """Pretty-print pupil description."""
        if self.normalize is True:
            header = f'rms normalized {self._name} Zernike description with:\n\t'
        else:
            header = f'{self._name} Zernike description with:\n\t'

        strs = []
        keys = list(sorted(self.coefs.keys()))

        for number in keys:
            coef = self.coefs[number]
            # skip 0 terms
            if coef == 0:
                continue

            # positive coefficient, prepend with +
            if np.sign(coef) == 1:
                _ = '+' + f'{coef:.3f}'
            # negative, sign comes from the value
            else:
                _ = f'{coef:.3f}'

            # create the name
            nm = nm_funcs[self._name](number)
            name = n_m_to_name(*nm)
            name = f'Z{number} - {name}'

            strs.append(' '.join([_, name]))
        body = '\n\t'.join(strs)
        unit_str = f'{self.labels.unit_prefix}{self.z_unit}{self.labels.unit_suffix}'
        footer = f'\n\t{self.pv:.3f} PV, {self.rms:.3f} RMS {unit_str}'
        return f'{header}{body}{footer}'


class FringeZernike(BaseZernike):
    """Fringe Zernike description of an optical pupil."""
    _name = 'Fringe'


class NollZernike(BaseZernike):
    """Noll Zernike description of an optical pupil."""
    _name = 'Noll'


class ANSI1TermZernike(BaseZernike):
    """1-term ANSI Zernike description of an optical pupil."""
    _name = 'ANSI'


class ANSI2TermZernike(Pupil):
    """2-term ANSI Zernike description of an optical pupil."""
    _cache = zcachemn

    def __init__(self, *args, **kwargs):
        """Initialize a new Zernike instancnp."""
        self.normalize = True
        pass_args = {}

        self.terms = []
        if kwargs is not None:
            for key, value in kwargs.items():
                k0l = key[0].lower()
                if k0l in ('a', 'b'):
                    # the only kwarg to contain a _ is a Zernike term
                    # the kwarg looks like A<n>_<m>=<coef>

                    # if the term is "A", it is a cosine and m is positive
                    if k0l == 'a':
                        msign = 1
                    elif k0l == 'b':
                        msign = -1

                    if '_' in key:
                        front, back = key.split('_')
                        n = int(front[1:])
                        m = int(back) * msign
                    else:
                        n = int(key[1:])
                        m = 0

                    self.terms.append((n, m, value))  # coef = value

                elif key.lower() == 'norm':
                    self.normalize = value
                else:
                    pass_args[key] = value

        super().__init__(**pass_args)

    def build(self):
        """Use the wavefront coefficients stored in this class instance to build a wavefront model.

        Returns
        -------
        self : `BaseZernike`
            this Zernike instance

        """
        # build a coordinate system over which to evaluate this function
        self.phase = np.zeros((self.samples, self.samples), dtype=config.precision)
        for (n, m, coef) in self.terms:
            # short circuit for speed
            if coef == 0:
                continue
            else:
                zernike = self._cache(n=n, m=m, samples=self.samples, norm=self.normalize) * coef
                self.phase += zernike

        return self


def zernikefit(data, x=None, y=None,
               rho=None, phi=None, terms=16,
               norm=False, residual=False,
               round_at=6, map_='Fringe'):
    """Fits a number of Zernike coefficients to provided data.

    Works by minimizing the mean square error  between each coefficient and the
    given data.  The data should be uniformly sampled in an x,y grid.

    Parameters
    ----------
    data : `numpy.ndarray`
        data to fit to.
    x : `numpy.ndarray`, optional
        x coordinates, same shape as data
    y : `numpy.ndarray`, optional
        y coordinates, same shape as data
    rho : `numpy.ndarray`, optional
        radial coordinates, same shape as data
    phi : `numpy.ndarray`, optional
        azimuthal coordinates, same shape as data
    terms : `int` or iterable, optional
        if an int, number of terms to fit,
        otherwise, specific terms to fit.
        If an iterable of ints, members of the single index set map_,
        else interpreted as (n,m) terms, in which case both m+ and m- must be given.
    norm : `bool`, optional
        if True, normalize coefficients to unit RMS value
    residual : `bool`, optional
        if True, return a tuple of (coefficients, residual)
    round_at : `int`, optional
        decimal place to round values at.
    map_ : `str`, optional, {'Fringe', 'Noll', 'ANSI'}
        which ordering of Zernikes to use

    Returns
    -------
    coefficients : `numpy.ndarray`
        an array of coefficients matching the input data.
    residual : `float`
        RMS error between the input data and the fit.

    Raises
    ------
    ValueError
        too many terms requested.

    """
    data = data.T  # transpose to mimic transpose of zernikes

    # precompute the valid indexes in the original data
    pts = np.isfinite(data)

    # set up an x/y rho/phi grid to evaluate Zernikes on
    if x is None and rho is None:
        rho, phi = make_rho_phi_grid(*reversed(data.shape))
        rho = rho[pts].flatten()
        phi = phi[pts].flatten()
    elif rho is None:
        rho, phi = cart_to_polar(x, y)
        rho, phi = rho[pts].flatten(), phi[pts].flatten()

    # convert indices to (n,m)
    if isinstance(terms, int):
        # case 1, number of terms
        nms = [nm_funcs[map_](i+1) for i in range(terms)]
    elif isinstance(terms[0], int):
        nms = [nm_funcs[map_](i) for i in terms]
    else:
        nms = terms

    # compute each Zernike term
    zerns_raw = []
    for (n, m) in nms:
        zern = zcachemn.grid_bypass(n, m, norm, rho, phi)
        zerns_raw.append(zern)

    zcachemn.grid_bypass_cleanup(rho, phi)
    zerns = np.asarray(zerns_raw).T

    # use least squares to compute the coefficients
    meas_pts = data[pts].flatten()
    coefs = np.linalg.lstsq(zerns, meas_pts, rcond=None)[0]
    if round_at is not None:
        coefs = coefs.round(round_at)

    if residual is True:
        components = []
        for zern, coef in zip(zerns_raw, coefs):
            components.append(coef * zern)

        _fit = np.asarray(components)
        _fit = _fit.sum(axis=0)
        rmserr = rms(data[pts].flatten() - _fit)
        return coefs, rmserr
    else:
        return coefs
