"""Zernike functions."""
import warnings
from collections import defaultdict

from retry import retry

from .conf import config
from .mathops import engine as e, kronecker, sign
from .pupil import Pupil
from .coordinates import make_rho_phi_grid, cart_to_polar, gridcache
from .util import rms, sort_xy, is_odd
from .plotting import share_fig_ax
from .jacobi import jacobi

# See JCW - http://wp.optics.arizona.edu/jcwyant/wp-content/uploads/sites/13/2016/08/ZernikePolynomialsForTheWeb.pdf


def piston(rho, phi):
    """Zernike Piston."""
    return e.ones(rho.shape)


def tip(rho, phi):
    """Zernike Tilt-Y."""
    return rho * e.cos(phi)


def tilt(rho, phi):
    """Zernike Tilt-X."""
    return rho * e.sin(phi)


def defocus(rho, phi):
    """Zernike defocus."""
    return 2 * rho**2 - 1


def primary_astigmatism_00(rho, phi):
    """Zernike primary astigmatism 0°."""
    return rho**2 * e.cos(2 * phi)


def primary_astigmatism_45(rho, phi):
    """Zernike primary astigmatism 45°."""
    return rho**2 * e.sin(2 * phi)


def primary_coma_y(rho, phi):
    """Zernike primary coma Y."""
    return (3 * rho**3 - 2 * rho) * e.cos(phi)


def primary_coma_x(rho, phi):
    """Zernike primary coma X."""
    return (3 * rho**3 - 2 * rho) * e.sin(phi)


def primary_spherical(rho, phi):
    """Zernike primary Spherical."""
    return 6 * rho**4 - 6 * rho**2 + 1


def primary_trefoil_y(rho, phi):
    """Zernike primary trefoil Y."""
    return rho**3 * e.cos(3 * phi)


def primary_trefoil_x(rho, phi):
    """Zernike primary trefoil X."""
    return rho**3 * e.sin(3 * phi)


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
            magnitude = e.sqrt(sum([v**2 for v in value]))
            angle = e.degrees(e.arctan2(*value))

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
    return e.sqrt((2 * (n + 1)) / (1 + kronecker(m, 0)))


def n_m_to_fringe(n, m):
    """Convert (n,m) two term index to Fringe index."""
    term1 = (1 + (n + abs(m))/2)**2
    term2 = 2 * abs(m)
    term3 = (1 + sign(m)) / 2
    return int(term1 - term2 + term3)


def n_m_to_ansi_j(n, m):
    """Convert (n,m) two term index to ANSI single term index."""
    return int((n * (n + 2) + m) / 2)


def ansi_j_to_n_m(idx):
    """Convert ANSI single term to (n,m) two-term index."""
    n = int(e.ceil((-3 + e.sqrt(9 + 8*idx))/2))
    m = 2 * idx - n * (n + 2)
    return n, m


def noll_to_n_m(idx):
    """Convert Noll Z to (n, m) two-term index."""
    # I don't really understand this code, the math is inspired by POPPY
    # azimuthal order
    n = int(e.ceil((-1 + e.sqrt(1 + 8 * idx)) / 2) - 1)
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
    m_n = 2 * (e.ceil(e.sqrt(idx)) - 1)  # sum of n+m
    g_s = (m_n / 2)**2 + 1  # start of each group of equal n+m given as idx index
    n = m_n / 2 + e.floor((idx - g_s) / 2)
    m = (m_n - n) * (1 - e.mod(idx-g_s, 2) * 2)
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
            suffix = 'Y'
        else:
            suffix = 'X'
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
        a name, e.g. Piston or Primary Spherical

    """
    # piston, tip tilt, az invariant order
    if n == 0:
        return 'Piston'
    if n == 1:
        if sign(m) == 1:
            return 'Tilt Y'
        else:
            return 'Tilt X'
    if n == 2 and m == 0:
        return 'Defocus'
    if m == 0:
        accessor = int((n / 2) - 1)
        prefix = _names.get(accessor, f'{accessor}th')
        return f'{prefix} Spherical'
    return _name_helper(n, m)


class ZCacheMN:
    """Cache of Zernike terms evaluated over the unit circle, based on (n, m) indices."""
    def __init__(self, gridcache=gridcache):
        """Create a new ZCache instance."""
        self.normed = {}
        self.regular = {}
        self.jac = {}
        self.cos = {}
        self.sin = {}
        self.gridcache = gridcache
        self.offgridj = {}  # jacobi polynomials
        self.offgrid_shifted_r = {}

    @retry(tries=2)
    def get_zernike(self, n, m, samples, norm):
        """Get an array of phase values for a given radial order n, azimuthal order m, number of samples, and orthonormalization."""  # NOQA
        if is_odd(n - m):
            raise ValueError('Zernike polynomials are only defined for n-m even.')
        key = (n, m, samples)
        if norm:
            d_ = self.normed
        else:
            d_ = self.regular

        try:
            return d_[key]
        except KeyError as e:
            zern = self.get_term(n=n, m=m, samples=samples)
            if norm:
                zern = zern * zernike_norm(n=n, m=m)

            d_[key] = zern
            raise e

    def get_term(self, n, m, samples):
        am = abs(m)
        r, p = self.get_grid(samples=samples, modified=False)
        term = self.get_jacobi(n=n, m=am, samples=samples)

        if m != 0:
            azterm = self.get_azterm(m=m, samples=samples)
            rterm = r ** am
            term = term * azterm * rterm

        return term

    def __call__(self, n, m, samples, norm):
        return self.get_zernike(n=n, m=m, samples=samples, norm=norm)

    def grid_bypass(self, n, m, norm, r, p):
        """Bypass the grid computation, providing radial coordinates directly.

        Notes
        -----
        To avoid memory leaks, you should use grid_bypass_cleanup after you are
        finished with this function for a given pair of r, p arrays

        Parameters
        ----------
        n : `int`
            radial order
        m : `int`
            azimuthal order
        norm : `bool`
            whether to orthonormalize the polynomials
        r : `numpy.ndarray`
            radial coordinates.  Unnormalized in the sense of the coordinate perturbation of the jacobi polynomials.
            Notionally on a regular grid spanning [0,1]
        p : `numpy.ndarray`
            azimuthal coordinates matching r

        Returns
        -------
        `numpy.ndarray`
            zernike polynomial n or m at this coordinate.

        """
        key_ = self._gb_key(r)
        key = (n, m, key_)
        rmod = 2 * r ** 2 - 1
        self.offgrid_shifted_r[key] = rmod

        term = self.get_jacobi(n=n, m=abs(m), samples=0, r=rmod)  # samples not used, dummy value

        if m != 0:
            if sign(m) == -1:
                azterm = e.sin(m * p)
            else:
                azterm = e.cos(m * p)

            rterm = r ** abs(m)
            term = term * azterm * rterm

        if norm:
            norm = zernike_norm(n, m)
            term *= norm

        return term

    def grid_bypass_cleanup(self, r, p):
        """Remove data related to r, p from the cache.

        Parameters
        ----------
        r : `numpy.ndarray`
            radial coordinates
        p : `numpy.ndarray`
            azimuthal coordinates

        """
        key_ = self._gb_key(r)
        for dict_ in (self.offgridj, self.offgrid_shifted_r):
            keys = list(dict_.keys())
            for key in keys:
                if key[2] == key_[0]:
                    del dict_[key]

    def _gb_key(self, r):
        spacing = r[1] - r[0]
        npts = r.shape
        max_ = r[-1]
        return f'{spacing}-{npts}-{max_}'

    @retry(tries=2)
    def get_azterm(self, m, samples):
        key = (m, samples)
        if sign(m) == -1:
            d_ = self.sin
            func = e.sin
        else:
            d_ = self.cos
            func = e.cos

        try:
            return d_[key]
        except KeyError as err:
            _, p = self.get_grid(samples=samples, modified=False)
            d_[key] = func(m * p)
            raise err

    @retry(tries=3)
    def get_jacobi(self, n, m, samples, nj=None, r=None):
        if nj is None:
            nj = (n - m) // 2

        if r is not None:
            key = (nj, m, self._gb_key(r))
            # r provided, grid not wanted
            # this is just a duplication of below with a separate r and cache dict
            try:
                return self.offgridj[key]
            except KeyError as e:
                if nj > 2:
                    jnm2 = self.get_jacobi(n=None, nj=nj - 2, m=m, samples=samples, r=r)
                    jnm1 = self.get_jacobi(n=None, nj=nj - 1, m=m, samples=samples, r=r)
                else:
                    jnm1, jnm2 = None, None

                jac = jacobi(nj, alpha=0, beta=m, Pnm1=jnm1, Pnm2=jnm2, x=r)
                self.offgridj[key] = jac
                raise e

        key = (nj, m, samples)
        try:
            return self.jac[key]
        except KeyError as e:
            r, _ = self.get_grid(samples=samples)
            if nj > 2:
                jnm1 = self.get_jacobi(n=None, nj=nj - 1, m=m, samples=samples)
                jnm2 = self.get_jacobi(n=None, nj=nj - 2, m=m, samples=samples)
            else:
                jnm1, jnm2 = None, None
            jac = jacobi(nj, alpha=0, beta=m, Pnm1=jnm1, Pnm2=jnm2, x=r)
            self.jac[key] = jac
            raise e

    def get_grid(self, samples, modified=True, r=None, p=None):
        if modified:
            res = self.gridcache(samples=samples, radius=1, r='r -> 2r^2 - 1', t='t -> t+90')
        else:
            res = self.gridcache(samples=samples, radius=1, r='r', t='t -> t+90')

        return res['r'], res['t']

    def clear(self, *args):
        """Empty the cache."""
        self.normed = {}
        self.regular = {}
        self.jac = {}
        self.sin = {}
        self.cos = {}
        self.offgrid_shifted_r = {}
        self.offgridj = {}
        self.offgridn = {}
        self.offgridr = {}

    def nbytes(self):
        """Total size in memory of the cache in bytes."""
        total = 0
        dicts = (
            self.normed,
            self.regular,
            self.jac,
            self.sin,
            self.cos,
            self.offgrid_shifted_r,
            self.offgridj,
        )
        for dict_ in dicts:
            for key in dict_:
                total += dict_[key].nbytes

        return total


zcachemn = ZCacheMN()
config.chbackend_observers.append(zcachemn.clear)

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
        """Initialize a new Zernike instance."""
        self.coefs = {}

        self.normalize = False
        pass_args = {}

        bb = kwargs.get('base', config.zernike_base)
        if bb != 1:
            warnings.warn("base of zero is deprecated and will be removed in prysm v0.19")
        if bb > 1:
            raise ValueError('It violates convention to use a base greater than 1.')
        elif bb < 0:
            raise ValueError('It is nonsensical to use a negative base.')
        self.base = bb

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
                    self.coefs[idx - (1-self.base)] = value
                elif key.lower() == 'norm':
                    self.normalize = value
                elif key.lower() == 'base':
                    self.base = value
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
            raise ValueError("single index notation not understood, modify zernike.nm_funcs")

        # build a coordinate system over which to evaluate this function
        self.data = e.zeros((self.samples, self.samples), dtype=config.precision)
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
        coefs = e.asarray(list(self.coefs.values()))
        coefs_work = abs(coefs)
        oidxs = e.asarray(list(self.coefs.keys()))
        idxs = e.argpartition(coefs_work, -n)[-n:]  # argpartition does some magic to identify the top n (unsorted)
        idxs = idxs[e.argsort(coefs_work[idxs])[::-1]]  # use argsort to sort them in ascending order and reverse
        big_terms = coefs[idxs]  # finally, take the values from the
        big_idxs = oidxs[idxs]
        names = e.asarray(self.names, dtype=str)[big_idxs - self.base]
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
        fig, ax = share_fig_ax(fig, ax)

        coefs = e.asarray(list(self.coefs.values()))
        idxs = e.asarray(list(self.coefs.keys()))
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

        E.g., astigmatism will get one bar.

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
        idxs = e.asarray(list(range(len(names))))

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
            modified FringeZernike instance.

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
            modified FringeZernike instance.

        """
        topn = self.top_n(n)
        new_coefs = {}
        for coef in topn:
            mag, index, *_ = coef
            new_coefs[index+(self.base-1)] = mag

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
            if e.sign(coef) == 1:
                _ = '+' + f'{coef:.3f}'
            # negative, sign comes from the value
            else:
                _ = f'{coef:.3f}'

            # create the name
            nm = nm_funcs[self._name](number)
            name = n_m_to_name(*nm)
            name = f'Z{number-(1-self.base)} - {name}'

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
        """Initialize a new Zernike instance."""
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
        self.phase = e.zeros((self.samples, self.samples), dtype=config.precision)
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
    pts = e.isfinite(data)

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
    zerns = e.asarray(zerns_raw).T

    # use least squares to compute the coefficients
    meas_pts = data[pts].flatten()
    coefs = e.linalg.lstsq(zerns, meas_pts, rcond=None)[0]
    if round_at is not None:
        coefs = coefs.round(round_at)

    if residual is True:
        components = []
        for zern, coef in zip(zerns_raw, coefs):
            components.append(coef * zern)

        _fit = e.asarray(components)
        _fit = _fit.sum(axis=0)
        rmserr = rms(data[pts].flatten() - _fit)
        return coefs, rmserr
    else:
        return coefs
