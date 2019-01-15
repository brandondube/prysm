'''A repository of fringe Zernike aberration descriptions used to model pupils of optical systems.'''
from collections import defaultdict

from .conf import config
from .pupil import Pupil
from .coordinates import make_rho_phi_grid, cart_to_polar
from .util import rms, share_fig_ax, sort_xy

from prysm import mathops as m

from prysm import _zernike as z


zernmap = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 8,
    9: 9,
    10: 10,
    11: 11,
    12: 12,
    13: 13,
    14: 14,
    15: 15,
    16: 16,
    17: 17,
    18: 18,
    19: 19,
    20: 20,
    21: 21,
    22: 22,
    23: 23,
    24: 24,
    25: 25,
    26: 26,
    27: 27,
    28: 28,
    29: 29,
    30: 30,
    31: 31,
    32: 32,
    33: 33,
    34: 34,
    35: 35,
    36: 36,
    37: 37,
    38: 38,
    39: 39,
    40: 40,
    41: 41,
    42: 42,
    43: 43,
    44: 44,
    45: 45,
    46: 46,
    47: 47,
    48: 48,
 }


def fzname(idx, base=1):
    """Return the name of a Fringe Zernike with the given index and base."""
    return z.zernikes[zernmap[idx-base]].name


def fzset_to_magnitude_angle(coefs):
    """Convert Fringe Zernike polynomial set to a magnitude and phase representation."""

    def mkary():  # default for defaultdict
        return m.zeros(2)

    # make a list of names to go with the coefficients
    names = [fzname(i, base=0) for i in range(len(coefs))]
    combinations = defaultdict(mkary)

    # for each name and coefficient, make a len 2 array.  Put the Y or 0 degree values in the first slot
    for coef, name in zip(coefs, names):
        if name.endswith(('X', 'Y', '°')):
            newname = ' '.join(name.split(' ')[:-1])
            if name.endswith('Y'):
                combinations[newname][0] = coef
            elif name.endswith('X'):
                combinations[newname][1] = coef
            elif name[-2] == '5':  # 45 degree case
                combinations[newname][1] = coef
            else:
                combinations[newname][0] = coef
        else:
            combinations[name][0] = coef

    # print(combinations)
    # now go over the combinations and compute the L2 norms and angles
    for name in combinations:
        ovals = combinations[name]
        magnitude = m.sqrt((ovals**2).sum())
        phase = m.degrees(m.arctan2(*ovals))
        values = (magnitude, phase)
        combinations[name] = values

    return dict(combinations)  # cast to regular dict for return


class FZCache(object):
    def __init__(self):
        self.normed = defaultdict(dict)
        self.regular = defaultdict(dict)

    def get_zernike(self, number, norm, samples):
        if norm is True:
            target = self.normed
        else:
            target = self.regular

        try:
            zern = target[samples][number]
        except KeyError:
            rho, phi = make_rho_phi_grid(samples, aligned='y')
            func = z.zernikes[zernmap[number]]
            zern = func(rho, phi)
            if norm is True:
                zern *= func.norm

            target[samples][number] = zern.copy()

        return zern

    def __call__(self, number, norm, samples):
        return self.get_zernike(number, norm, samples)

    def clear(self, *args):
        self.normed = defaultdict(dict)
        self.regular = defaultdict(dict)


class FringeZernike(Pupil):
    def __init__(self, *args, **kwargs):
        if args is not None:
            if len(args) is 0:
                self.coefs = m.zeros(len(zernmap), dtype=config.precision)
            else:
                self.coefs = m.asarray([*args[0]], dtype=config.precision)

        self.normalize = False
        pass_args = {}

        self.base = config.zernike_base
        try:
            bb = kwargs['base']
            if bb > 1:
                raise ValueError('It violates convention to use a base greater than 1.')
            elif bb < 0:
                raise ValueError('It is nonsensical to use a negative base.')
            self.base = bb
        except KeyError:
            # user did not specify base
            pass

        if kwargs is not None:
            for key, value in kwargs.items():
                if key[0].lower() == 'z':
                    idx = int(key[1:])  # strip 'Z' from index
                    self.coefs[idx - self.base] = value
                elif key in ('norm'):
                    self.normalize = True
                elif key.lower() == 'base':
                    self.base = value
                else:
                    pass_args[key] = value

        super().__init__(**pass_args)

    def build(self):
        '''Uses the wavefront coefficients stored in this class instance to
            build a wavefront model.

        Returns
        -------
        self.phase : `numpy.ndarray`
            arrays containing the phase associated with the pupil
        self.fcn : `numpy.ndarray`
            array containing the wavefunction of the pupil plane

        '''
        # build a coordinate system over which to evaluate this function
        self.phase = m.zeros((self.samples, self.samples), dtype=config.precision)
        for term, coef in enumerate(self.coefs):
            # short circuit for speed
            if coef == 0:
                continue
            self.phase += coef * zcache(term, self.normalize, self.samples)

        return self

    @property
    def magnitudes(self):
        """Returns the magnitude and angles of the zernike components in this wavefront."""
        return fzset_to_magnitude_angle(self.coefs)

    def barplot(self, orientation='h', buffer=1, zorder=3, fig=None, ax=None):
        """Creates a barplot of coefficients and their names.

        Parameters
        ----------
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
        fig, ax = share_fig_ax(fig, ax)

        coefs = m.asarray(self.coefs)
        idxs = m.asarray(range(len(coefs))) + self.base
        names = [fzname(i) for i in (idxs - self.base)]
        lab = f'{self.zaxis_label} [{self.phase_unit}]'
        lims = (idxs[0] - buffer, idxs[-1] + buffer)
        if orientation.lower() in ('h', 'horizontal'):
            vmin, vmax = coefs.min(), coefs.max()
            drange = vmax - vmin
            offset = drange * 0.01

            ax.bar(idxs, self.coefs, zorder=zorder)
            plt.xticks(idxs, names, rotation=90)
            for i in idxs:
                ax.text(i, offset, str(i), ha='center')
            ax.set(ylabel=lab, xlim=lims)
        else:
            ax.barh(idxs, self.coefs, zorder=zorder)
            plt.yticks(idxs, names)
            for i in idxs:
                ax.text(0, i, str(i), ha='center')
            ax.set(xlabel=lab, ylim=lims)
        return fig, ax

    def barplot_magnitudes(self, orientation='h', sort=False, buffer=1, zorder=3, fig=None, ax=None):
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

        magang = fzset_to_magnitude_angle(self.coefs)
        mags = [m[0] for m in magang.values()]
        names = magang.keys()
        idxs = list(range(len(names)))

        if sort:
            mags, names = sort_xy(mags, names)
            mags = list(reversed(mags))
            names = list(reversed(names))
        lab = f'{self.zaxis_label} [{self.phase_unit}]'
        lims = (idxs[0] - buffer, idxs[-1] + buffer)
        fig, ax = share_fig_ax(fig, ax)
        if orientation.lower() in ('h', 'horizontal'):
            ax.bar(idxs, mags, zorder=zorder)
            plt.xticks(idxs, names, rotation=90)
            ax.set(ylabel=lab, xlim=lims)
        else:
            ax.barh(idxs, mags, zorder=zorder)
            plt.yticks(idxs, names)
            ax.set(xlabel=lab, ylim=lims)
        return fig, ax

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
        coefs = m.asarray(self.coefs)
        coefs_work = abs(coefs)
        oidxs = m.arange(len(coefs)) + self.base  # "original indexes"
        idxs = m.argpartition(coefs_work, -n)[-n:]  # argpartition does some magic to identify the top n (unsorted)
        idxs = idxs[m.argsort(coefs_work[idxs])[::-1]]  # use argsort to sort them in ascending order and reverse
        big_terms = coefs[idxs]  # finally, take the values from the
        big_idxs = oidxs[idxs]
        names = [fzname(i) for i in idxs]
        return list(zip(big_terms, big_idxs, names))

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

        lab = f'{self.zaxis_label} [{self.phase_unit}]'
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
            self.coefs = self.coefs[:n]
            self.build()
            self.mask(self._mask, self.mask_target)
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
        new_coefs = m.zeros(len(self.coefs), dtype=config.precision)
        for coef in topn:
            mag, index, *_ = coef
            new_coefs[index-self.base] = mag

        self.coefs = new_coefs
        self.build()
        self.mask(self._mask, self.mask_target)
        return self

    def __repr__(self):
        '''Pretty-print pupil description.'''
        if self.normalize is True:
            header = 'rms normalized Fringe Zernike description with:\n\t'
        else:
            header = 'Fringe Zernike description with:\n\t'

        strs = []
        for number, (coef, func) in enumerate(zip(self.coefs, z.zernikes)):
            # skip 0 terms
            if coef == 0:
                continue

            # positive coefficient, prepend with +
            if m.sign(coef) == 1:
                _ = '+' + f'{coef:.3f}'
            # negative, sign comes from the value
            else:
                _ = f'{coef:.3f}'

            # create the name
            name = f'Z{number+self.base} - {func.name}'

            strs.append(' '.join([_, name]))
        body = '\n\t'.join(strs)

        footer = f'\n\t{self.pv:.3f} PV, {self.rms:.3f} RMS'
        return f'{header}{body}{footer}'


def fit(data, x=None, y=None, rho=None, phi=None, terms=16, norm=False, residual=False, round_at=6):
    '''Fits a number of Zernike coefficients to provided data by minimizing
        the root sum square between each coefficient and the given data.  The
        data should be uniformly sampled in an x,y grid.

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
        azimuthal
    terms : `int`, optional
        number of terms to fit, fits terms 0~terms
    norm : `bool`, optional
        if True, normalize coefficients to unit RMS value
    residual : `bool`, optional
        if True, return a tuple of (coefficients, residual)
    round_at : `int`
        decimal place to round values at.

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

    '''
    if terms > len(zernmap):
        raise ValueError(f'number of terms must be less than {len(zernmap)}')

    data = data.T  # transpose to mimic transpose of zernikes

    # precompute the valid indexes in the original data
    pts = m.isfinite(data)

    if x is None and rho is None:
        # set up an x/y rho/phi grid to evaluate Zernikes on
        rho, phi = make_rho_phi_grid(*reversed(data.shape))
        rho = rho[pts].flatten()
        phi = phi[pts].flatten()
    elif rho is None:
        rho, phi = cart_to_polar(x, y)
        rho, phi = rho[pts].flatten(), phi[pts].flatten()

    # compute each Zernike term
    zernikes = []
    for i in range(terms):
        func = z.zernikes[zernmap[i]]
        base_zern = func(rho, phi)
        if norm:
            base_zern *= func.norm
        zernikes.append(base_zern)
    zerns = m.asarray(zernikes).T

    # use least squares to compute the coefficients
    meas_pts = data[pts].flatten()
    coefs = m.lstsq(zerns, meas_pts, rcond=None)[0]
    if round_at is not None:
        coefs = coefs.round(round_at)

    if residual is True:
        components = []
        for zern, coef in zip(zernikes, coefs):
            components.append(coef * zern)

        _fit = m.asarray(components)
        _fit = _fit.sum(axis=0)
        rmserr = rms(data[pts].flatten() - _fit)
        return coefs, rmserr
    else:
        return coefs


zcache = FZCache()
config.chbackend_observers.append(zcache.clear)
