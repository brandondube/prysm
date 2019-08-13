"""Basic class holding data, used to recycle code."""
import copy
from collections.abc import Iterable
from numbers import Number

from scipy import interpolate

from .conf import config, sanitize_unit
from .mathops import engine as e
from .wavelengths import mkwvl
from .coordinates import uniform_cart_to_polar, polar_to_cart
from .plotting import share_fig_ax


def fix_interp_pair(x, y):
    if y is None:
        y = 0

    if x is None:
        x = 0

    if isinstance(x, Iterable) and not isinstance(y, Iterable):
        y = [y] * len(x)
    elif isinstance(y, Iterable) and not isinstance(x, Iterable):
        x = [x] * len(y)

    return x, y


class RichData:
    """Abstract base class holding some data properties."""
    _data_attr = 'data'
    _data_type = 'image'
    _default_twosided = True
    _slice_xscale = 'linear'
    _slice_yscale = 'linear'

    def __init__(self, x, y, data, labels, xy_unit=None, z_unit=None, wavelength=None):
        """Initialize a new BasicData instance.

        Parameters
        ----------
        x : `numpy.ndarray`
            x unit axis
        y : `numpy.ndarray`
            y unit axis
        data : `numpy.ndarray`
            data
        labels : `Labels`
            labels instance, can be shared
        xyunit : `astropy.unit` or `str`, optional
            astropy unit or string which satisfies hasattr(astropy.units, xyunit)
        zunit : `astropy.unit` or `str`, optional
             astropy unit or string which satisfies hasattr(astropy.units, xyunit)
        wavelength : `astropy.unit` or `float`
            astropy unit or quantity or float with implicit units of microns

        Returns
        -------
        RichData
            the instance

        """
        if wavelength is None:
            wavelength = config.wavelength

        self.x, self.y = x, y
        setattr(self, self._data_attr, data)
        self.labels = labels
        self.wavelength = mkwvl(wavelength)
        self.xy_unit = sanitize_unit(xy_unit, self.wavelength)
        self.z_unit = sanitize_unit(z_unit, self.wavelength)
        self.interpf_x, self.interpf_y, self.interpf_2d = None, None, None

    @property
    def shape(self):
        """Proxy to phase or data shape."""
        try:
            return getattr(self, self._data_attr).shape
        except AttributeError:
            return (0, 0)

    @property
    def size(self):
        """Proxy to phase or data size."""
        try:
            return getattr(self, self._data_attr).size
        except AttributeError:
            return 0

    @property
    def samples_x(self):
        """Number of samples in the x dimension."""
        return self.shape[1]

    @property
    def samples_y(self):
        """Number of samples in the y dimension."""
        return self.shape[0]

    @property
    def sample_spacing(self):
        """center-to-center sample spacing."""
        try:
            return self.x[1] - self.x[0]
        except TypeError:
            return e.nan

    @property
    def center_x(self):
        """Center "pixel" in x."""
        return self.samples_x // 2

    @property
    def center_y(self):
        """Center "pixel" in y."""
        return self.samples_y // 2

    def copy(self):
        """Return a (deep) copy of this instance."""
        return copy.deepcopy(self)

    def change_xy_unit(self, to, inplace=True):
        """Change the x/y unit to a new one, scaling the data in the process.

        Parameters
        ----------
        to : `astropy.unit` or `str`
            if not an astropy unit, a string that is a valid attribute of astropy.units.
        inplace : `bool`, optional
            if True, returns self.  Otherwise returns the modified data.

        Returns
        -------
        `RichData`
            self, if inplace=True
        `numpy.ndarray`, `numpy.ndarray`
            x, y from self, if inplace=False

        """
        unit = sanitize_unit(to, self.wavelength)
        coef = self.xy_unit.to(unit)
        x, y = self.x * coef, self.y * coef
        if not inplace:
            return x, y
        else:
            self.x, self.y = x, y
            self.xy_unit = unit
            return self

    def change_z_unit(self, to, inplace=True):
        """Change the z unit to a new one, scaling the data in the process.

        Parameters
        ----------
        to : `astropy.unit` or `str`
            if not an astropy unit, a string that is a valid attribute of astropy.units.
        inplace : `bool`, optional
            if True, returns self.  Otherwise returns the modified data.

        Returns
        -------
        `RichData`
            self, if inplace=True
        `numpy.ndarray`
            data from self, if inplace=False

        """
        unit = sanitize_unit(to, self.wavelength)
        coef = self.z_unit.to(unit)
        modified_data = getattr(self, self._data_attr) * coef
        if not inplace:
            return modified_data
        else:
            setattr(self, self._data_attr, modified_data)
            self.units = unit
            return self

    def slices(self, twosided=None):
        """Create a `Slices` instance from this instance.

        Parameters
        ----------
        twosided : `bool`, optional
            if None, copied from self._default_twosided

        Returns
        -------
        `Slices`
            a Slices object

        """
        if twosided is None:
            twosided = self._default_twosided
        return Slices(getattr(self, self._data_attr), x=self.x, y=self.y,
                      twosided=twosided, x_unit=self.xy_unit, z_unit=self.z_unit, labels=self.labels,
                      xscale=self._slice_xscale, yscale=self._slice_yscale)

    def _make_interp_function_2d(self):
        """Generate a 2D interpolation function for this instance, used in sampling with exact_xy.

        Returns
        -------
        `scipy.interpolate.RegularGridInterpolator`
            interpolator instance.

        """
        if self.interpf_2d is None:
            self.interpf_2d = interpolate.RegularGridInterpolator((self.y, self.x), getattr(self, self._data_attr))

        return self.interpf_2d

    def _make_interp_function_xy1d(self):
        """Generate two interpolation functions for the xy slices.

        Returns
        -------
        self.interpf_x : `scipy.interpolate.interp1d`
            x interpolator
        self.interpf_y : `scipy.interpolate.interp1d`
            y interpolator

        """
        if self.interpf_x is None or self.interpf_y is None:
            ux, x = self.slices().x
            uy, y = self.slices().y

            self.interpf_x = interpolate.interp1d(ux, x)
            self.interpf_y = interpolate.interp1d(uy, y)

        return self.interpf_x, self.interpf_y

    def exact_polar(self, rho, phi=None):
        """Retrieve data at the specified radial coordinates pairs.

        Parameters
        ----------
        r : iterable
            radial coordinate(s) to sample
        phi : iterable
            azimuthal coordinate(s) to sample

        Returns
        -------
        `numpy.ndarray`
            data at the given points

        """
        self._make_interp_function_2d()

        rho, phi = fix_interp_pair(rho, phi)
        x, y = polar_to_cart(rho, phi)
        return self.interpf_2d((x, y), method='linear')

    def exact_xy(self, x, y=None):
        """Retrieve data at the specified X-Y frequency pairs.

        Parameters
        ----------
        x : iterable
            X coordinate(s) to retrieve
        y : iterable
            Y coordinate(s) to retrieve

        Returns
        -------
        `numpy.ndarray`
            data at the given points

        """
        self._make_interp_function_2d()

        x, y = fix_interp_pair(x, y)
        return self.interpf_2d((y, x), method='linear')

    def exact_x(self, x):
        """Return data at an exact x coordinate along the y=0 axis.

        Parameters
        ----------
        x : `number` or `numpy.ndarray`
            x coordinate(s) to return

        Returns
        -------
        `numpy.ndarray`
            ndarray of values

        """
        self._make_interp_function_xy1d()
        return self.interpf_x(x)

    def exact_y(self, y):
        """Return data at an exact y coordinate along the x=0 axis.

        Parameters
        ----------
        y : `number` or `numpy.ndarray`
            y coordinate(s) to return

        Returns
        -------
        `numpy.ndarray`
            ndarray of values

        """
        self._make_interp_function_xy1d()
        return self.interpf_y(y)

    def plot2d(self, xlim=None, ylim=None, clim=None, cmap=None,
               log=False, power=1, interpolation=None,
               show_colorbar=True, show_axlabels=True,
               fig=None, ax=None):
        """Plot the data in 2D.

        Parameters
        ----------
        xlim : `float` or iterable, optional
            x axis limits.  If not iterable, symmetric version of the single value
        ylim : `float` or iterable, optional
            y axis limits.  If None and xlim is not None, copied from xlim.
            If not iterable, symmetric version of the single value.
        clim : iterable, optional
            clim passed directly to matplotlib.
            If None, looked up on self._default_clim.
        cmap : `str`, optional
            colormap to use, passed directly to matplotlib if not None.
            If None, looks up the default cmap for self._data_type on config
        log : `bool`, optional
            if True, plot on a log color scale
        power : `float`, optional
            if not 1, plot on a power stretched color scale
        interpolation : `str`, optional
            interpolation method to use, passed directly to matplotlib
        show_colorbar : `bool`, optional
            if True, draws the colorbar
        show_axlabels : `bool`, optional
            if True, draws the axis labels
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
        from matplotlib.colors import PowerNorm, LogNorm
        fig, ax = share_fig_ax(fig, ax)

        # sanitize some inputs
        if cmap is None:
            cmap = getattr(config, f'{self._data_type}_cmap')

        if interpolation is None:
            interpolation = config.interpolation

        if xlim is not None and not isinstance(xlim, Iterable):
            xlim = (-xlim, xlim)

        if ylim is None and xlim is not None:
            ylim = xlim
        elif ylim is not None and not isinstance(ylim, Iterable):
            ylim = (-ylim, ylim)

        if clim is not None and not isinstance(clim, Iterable):
            clim = (-clim, clim)

        norm = None
        if log:
            norm = LogNorm()
        elif power != 1:
            norm = PowerNorm(power)

        im = ax.imshow(getattr(self, self._data_attr),
                       extent=[self.x[0], self.x[-1], self.y[0], self.y[-1]],
                       cmap=cmap,
                       clim=clim,
                       norm=norm,
                       origin='lower',
                       interpolation=interpolation)

        if show_colorbar:
            fig.colorbar(im, label=self.labels.z(self.xy_unit, self.z_unit), ax=ax, fraction=0.046)

        xlab, ylab = None, None
        if show_axlabels:
            xlab = self.labels.x(self.xy_unit, self.z_unit)
            ylab = self.labels.y(self.xy_unit, self.z_unit)
        ax.set(xlabel=xlab, xlim=xlim, ylabel=ylab, ylim=ylim)

        return fig, ax


class Slices:
    """Slices of data."""
    def __init__(self, data, x, y, x_unit, z_unit, labels, xscale, yscale, twosided=True):
        self._source = data
        self._source_polar = None
        self._r = None
        self._p = None
        self._x = x
        self._y = y
        self.x_unit, self.z_unit = x_unit, z_unit
        self.labels = labels
        self.xscale, self.yscale = xscale, yscale
        self.center_y, self.center_x = (int(e.ceil(s / 2)) for s in data.shape)
        self.twosided = twosided

    def check_polar_calculated(self):
        """Ensure that the polar representation of the source data has been calculated."""
        if self._source_polar is None:
            rho, phi, polar = uniform_cart_to_polar(self._x, self._y, self._source)
            self._r, self._p = rho, phi
            self._source_polar = polar

    @property
    def x(self):
        """Slice through the Y=0 axis of the data, i.e. along the X axis.

        Returns
        -------
        x : `numpy.ndarray`
            coordinates
        slice : `numpy.ndarray`
            values of the data array at these coordinates

        """
        if self.twosided:
            return self._x, self._source[self.center_y, :]
        else:
            return self._x[self.center_x:], self._source[self.center_y, self.center_x:]

    @property
    def y(self):
        """Slice through the X=0 axis of the data, i.e., along the Y axis.

        Returns
        -------
        y : `numpy.ndarray`
            coordinates
        slice : `numpy.ndarray`
            values of the data array at these coordinates

        """
        if self.twosided:
            return self._y, self._source[:, self.center_x]
        else:
            return self._y[self.center_y:], self._source[self.center_y:, self.center_x]

    @property
    def azavg(self):
        """Azimuthal average of the data.

        Returns
        -------
        rho : `numpy.ndarray`
            coordinates
        slice : `numpy.ndarray`
            values of the data array at these coordinates

        """
        self.check_polar_calculated()
        return self._r, e.nanmean(self._source_polar, axis=0)

    @property
    def azmedian(self):
        """Azimuthal median of the data.

        Returns
        -------
        rho : `numpy.ndarray`
            coordinates
        slice : `numpy.ndarray`
            values of the data array at these coordinates

        """
        self.check_polar_calculated()
        return self._r, e.nanmedian(self._source_polar, axis=0)

    @property
    def azmin(self):
        """Azimuthal minimum of the data.

        Returns
        -------
        rho : `numpy.ndarray`
            coordinates
        slice : `numpy.ndarray`
            values of the data array at these coordinates

        """
        self.check_polar_calculated()
        return self._r, e.nanmin(self._source_polar, axis=0)

    @property
    def azmax(self):
        """Azimuthal maximum of the data.

        Returns
        -------
        rho : `numpy.ndarray`
            coordinates
        slice : `numpy.ndarray`
            values of the data array at these coordinates

        """
        self.check_polar_calculated()
        return self._r, e.nanmax(self._source_polar, axis=0)

    @property
    def azpv(self):
        """Azimuthal PV of the data.

        Returns
        -------
        rho : `numpy.ndarray`
            coordinates
        slice : `numpy.ndarray`
            values of the data array at these coordinates

        """
        r, mx = self.azmax
        r, mn = self.azmin
        return r, mx - mn

    @property
    def azvar(self):
        """Azimuthal variance of the data.

        Returns
        -------
        rho : `numpy.ndarray`
            coordinates
        slice : `numpy.ndarray`
            values of the data array at these coordinates

        """
        self.check_polar_calculated()
        return self._r, e.nanvar(self._source_polar, axis=0)

    @property
    def azstd(self):
        """Azimuthal standard deviation of the data.

        Returns
        -------
        rho : `numpy.ndarray`
            coordinates
        slice : `numpy.ndarray`
            values of the data array at these coordinates

        """
        self.check_polar_calculated()
        return self._r, e.nanstd(self._source_polar, axis=0)

    def plot(self, slices, lw=None, alpha=None, zorder=None, invert_x=False,
             xlim=(None, None), xscale=None,
             ylim=(None, None), yscale=None,
             show_legend=True, show_axlabels=True,
             fig=None, ax=None):
        """Plot slice(s)

        Parameters
        ----------
        slices : `str` or `Iterable`
            if a string, plots a single slice.  Else, plots several slices.
        lw : `float` or `Iterable`, optional
            line width to use for the slice(s).
            If a single value, used for all slice(s).
            If iterable, used pairwise with the slices
        alpha : `float` or `Iterable`, optional
            alpha (transparency) to use for the slice(s).
            If a single value, used for all slice(s).
            If iterable, used pairwise with the slices
        zorder : `int` or `Iterable`, optional
            zorder (stack height) to use for the slice(s).
            If a single value, used for all slice(s).
            If iterable, used pairwise with the slices
        invert_x : `bool`, optional
            if True, flip x (i.e., Freq => Period or vice-versa)
        xlim : `tuple`, optional
            x axis limits
        xscale : `str`, {'linear', 'log'}, optional
            scale used for the x axis
        ylim : `tuple`, optional
            y axis limits
        yscale : `str`, {'linear', 'log'}, optional
            scale used for the y axis
        show_legend : `bool`, optional
            if True, show the legend
        show_axlabels : `bool`, optional
            if True, show the axis labels
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
        def safely_invert_x(x, v):
            # these values are unsafe for fp32.  Maybe a bit pressimistic, but that's life
            zeros = abs(x) < 1e-9
            x, v = x.copy(), v.copy()
            x[zeros] = e.nan
            v[zeros] = e.nan
            x = 1 / x
            return x, v

        # error check everything
        if alpha is None:
            alpha = config.alpha

        if lw is None:
            lw = config.lw

        if zorder is None:
            zorder = config.zorder

        if isinstance(slices, str):
            slices = [slices]

        if isinstance(alpha, Number):
            alpha = [alpha] * len(slices)

        if isinstance(lw, Number):
            lw = [lw] * len(slices)

        if isinstance(zorder, int):
            zorder = [zorder] * len(slices)

        fig, ax = share_fig_ax(fig, ax)

        for slice_, alpha, lw, zorder in zip(slices, alpha, lw, zorder):
            u, v = getattr(self, slice_)
            if invert_x:
                u, v = safely_invert_x(u, v)

            ax.plot(u, v, label=slice_, lw=lw, alpha=alpha, zorder=zorder)

        if show_legend:
            ax.legend(title='Slice')

        # the x label has some special text manipulation

        if invert_x:
            units = self.units.copy()
            units.x = 1 / units.x
            units.y = 1 / units.y

            xlabel = self.labels.generic(self.x_unit, self.z_unit)
            # ax.invert_xaxis()
            if 'Period' in xlabel:
                xlabel = xlabel.replace('Period', 'Frequency')
            elif 'Frequency' in xlabel:
                xlabel = xlabel.replace('Frequency', 'Period')
        else:
            # slightly unclean code duplication here
            xlabel = self.labels.generic(self.x_unit, self.z_unit)

        ylabel = self.labels.z(self.x_unit, self.z_unit)

        if not show_axlabels:
            xlabel, ylabel = '', ''

        # z looks wrong here, but z from 2D is y in 1D.
        ax.set(xscale=xscale or self.xscale, xlim=xlim, xlabel=xlabel,
               yscale=yscale or self.yscale, ylim=ylim, ylabel=ylabel)
        if invert_x:
            ax.invert_xaxis()

        return fig, ax
