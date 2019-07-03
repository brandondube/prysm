"""Basic class holding data, used to recycle code."""
import copy
from collections.abc import Iterable

from scipy import interpolate

from .conf import config
from .mathops import engine as e
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
    axis_mode = 'period'

    units = {
        'm': 'm',
        'meter': 'm',
        'mm': 'mm',
        'millimeter': 'mm',
        'μm': 'μm',
        'um': 'μm',
        'micron': 'μm',
        'micrometer': 'μm',
        'nm': 'nm',
        'nanometer': 'nm',
        'Å': 'Å',
        'aa': 'Å',
        'angstrom': 'Å',
        'λ': 'λ',
        'waves': 'λ',
        'lambda': 'λ',
        'px': 'px',
        'pixel': 'px',
        'au': 'a.u.',
        'arb': 'a.u.',
        'arbitrary': 'a.u.',
    }
    unit_scales = {
        'm': 1,
        'mm': 1e-3,
        'μm': 1e-6,
        'nm': 1e-9,
        'Å': 1e-10,
    }
    unit_changes = {
        'm_m': lambda x: 1,
        'm_mm': lambda x: 1e-3,
        'm_μm': lambda x: 1e-6,
        'm_nm': lambda x: 1e-9,
        'm_Å': lambda x: 1e-10,
        'm_λ': lambda x: 1e-6 * x,
        'mm_mm': lambda x: 1,
        'mm_m': lambda x: 1e3,
        'mm_μm': lambda x: 1e-3,
        'mm_nm': lambda x: 1e-6,
        'mm_Å': lambda x: 1e-7,
        'mm_λ': lambda x: 1e-3 * x,
        'μm_μm': lambda x: 1,
        'μm_m': lambda x: 1e6,
        'μm_mm': lambda x: 1e3,
        'μm_nm': lambda x: 1e-3,
        'μm_Å': lambda x: 1e-4,
        'μm_λ': lambda x: 1 * x,
        'nm_nm': lambda x: 1,
        'nm_m': lambda x: 1e9,
        'nm_mm': lambda x: 1e6,
        'nm_μm': lambda x: 1e3,
        'nm_Å': lambda x: 1e-1,
        'nm_λ': lambda x: 1e3 * x,
        'Å_Å': lambda x: 1,
        'Å_m': lambda x: 1e10,
        'Å_mm': lambda x: 1e7,
        'Å_μm': lambda x: 1e4,
        'Å_nm': lambda x: 10,
        'Å_λ': lambda x: 1e4 * x,
        'λ_λ': lambda x: 1,
        'λ_m': lambda x: 1e6 / x,
        'λ_mm': lambda x: 1e3 / x,
        'λ_μm': lambda x: x,
        'λ_nm': lambda x: 1e-3 / x,
        'λ_Å': lambda x: 1e-4 / x,
        'px_px': lambda x: 1,  # beware changing pixels to other units
        'px_m': lambda x: 1,
        'px_mm': lambda x: 1,
        'px_μm': lambda x: 1,
        'px_nm': lambda x: 1,
        'px_Å': lambda x: 1,
        'px_λ': lambda x: 1,
    }

    def __init__(self, x, y, data, xyunit=None, zunit=None, xlabel=None, ylabel=None, zlabel=None):
        """Initialize a new BasicData instance.

        Parameters
        ----------
        x : `numpy.ndarray`
            x unit axis
        y : `numpy.ndarray`
            y unit axis
        data : `numpy.ndarray`
            data
        xyunit : `str`, optional
            unit used for the XY axes
        zunit : `str`, optional
            unit used for the Z (data) axis
        xlabel : `str`, optional
            x label used on plots
        ylabel : `str`, optional
            y label used on plots
        zlabel : `str`, optional
            z label used on plots

        Returns
        -------
        BasicData
            the BasicData instance

        """
        self.x, self.y = x, y
        setattr(self, self._data_attr, data)
        self.interpf_x, self.interpf_y, self.interpf_2d = None, None, None
        self.xlabel, self.ylabel, self.zlabel = xlabel, ylabel, zlabel
        self.xyunit, self.zunit = xyunit, zunit

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

    @property
    def zunit(self):
        """Unit used to describe the optical phase."""
        return self._zunit

    @zunit.setter
    def zunit(self, unit):
        unit = unit.lower()
        if unit == 'å':
            self._zunit = unit.upper()
        else:
            if unit not in self.units:
                raise ValueError(f'{unit} not a valid unit, must be in {set(self.units.keys())}')
            self._zunit = self.units[unit]

    @property
    def xyunit(self):
        """Unit used to describe the spatial phase."""
        return self._xyunit

    @xyunit.setter
    def xyunit(self, unit):
        unit = unit.lower()
        if unit not in self.units:
            raise ValueError(f'{unit} not a valid unit, must be in {set(self.units.keys())}')

        self._xyunit = self.units[unit]

    def change_zunit(self, to, inplace=True):
        """Change the units used to describe the z axis.

        Parameters
        ----------
        to : `str`
            new unit, a member of `BasicData`.units.keys()
        inplace : `bool`, optional
            whether to change (scale) the data, if False, returns updated data, if True, returns self.

        Returns
        -------
        `new_data` : `np.ndarray`
            new data
        OR
        `self` : `BasicData`
            self

        """
        if to not in self.units.keys():
            raise ValueError('unsupported unit')
        if self.zunit == 'a.u.':
            raise ValueError('cannot change arbitrary units to others.')

        wavelength = getattr(self, 'wavelength', None)
        if not wavelength and to.lower() in {'waves', 'lambda', 'λ'}:
            raise ValueError('must have self.wavelength when converting to waves')

        fctr = self.unit_changes['_'.join([self.zunit, self.units[to]])](wavelength)
        new_data = getattr(self, self._data_attr) / fctr
        if inplace:
            setattr(self, self._data_attr, new_data)
            self.zunit = to
            return self
        else:
            return new_data

    def change_xyunit(self, to, inplace=True):
        """Change the x/y used to describe the spatial dimensions.

        Parameters
        ----------
        to : `str`
            new unit, a member of `BasicData`.units.keys()
        inplace : `bool`, optional
            whether to change self.x and self.y.
            If False, returns updated phase, if True, returns self

        Returns
        -------
        `new_x` : `np.ndarray`
            new ordinate x axis
        `new_y` : `np.ndarray`
            new ordinate y axis
        OR
        `self` : `BasicData`
            self

        """
        if to not in self.units.keys():
            raise ValueError('unsupported unit')
        if self.xyunit == 'a.u.':
            raise ValueError('cannot change arbitrary units to others.')

        wavelength = getattr(self, 'wavelength', None)

        if to.lower() != 'px':
            fctr = self.unit_changes['_'.join([self.xyunit, self.units[to]])](wavelength)
            new_ux = self.x / fctr
            new_uy = self.y / fctr
        else:
            sy, sx = self.shape
            new_ux = e.arange(sx, dtype=config.precision)
            new_uy = e.arange(sy, dtype=config.precision)
        if inplace:
            self.x = new_ux
            self.y = new_uy
            self.xyunit = to
            return self
        else:
            return new_ux, new_uy

    def copy(self):
        """Return a (deep) copy of this instance."""
        return copy.deepcopy(self)

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
                      twosided=twosided,
                      xlabel=f'{self.xlabel} [{self.xyunit}]', ylabel=f'{self.zlabel} [{self.zunit}]')

    def _make_interp_function_2d(self):
        """Generate a 2D interpolation function for this instance, used in sampling with exact_xy.

        Returns
        -------
        `scipy.interpolate.RegularGridInterpolator`
            interpolator instance.

        """
        if self.interpf_2d is None:
            self.interpf_2d = interpolate.RegularGridInterpolator((self.x, self.y), self.data)

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
        return self.interpf_2d((x, y), method='linear')

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
        return self.interpf_x(y)

    def plot2d(self, xlim=None, ylim=None, clim=None, cmap=None,
               log=False, power=1, interpolation=config.interpolation,
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

        if xlim is not None and not isinstance(xlim, Iterable):
            xlim = (-xlim, xlim)

        if ylim is None and xlim is not None:
            ylim = xlim
        elif ylim is not None and not isinstance(ylim, Iterable):
            ylim = (-ylim, ylim)

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
            fig.colorbar(im, label=f'{self.zlabel} [{self.zunit}]', ax=ax, fraction=0.046)

        xlab, ylab = None, None
        if show_axlabels:
            xlab = f'{self.xlabel} [{self.xyunit}]'
            ylab = f'{self.ylabel} [{self.xyunit}]'

        ax.set(xlabel=xlab, xlim=xlim, ylabel=ylab, ylim=ylim)

        return fig, ax


class Slices:
    """Slices of data."""
    def __init__(self, data, x, y, twosided=True, xlabel=None, ylabel=None):
        self._source = data
        self._source_polar = None
        self._r = None
        self._p = None
        self._x = x
        self._y = y
        self.center_y, self.center_x = (int(e.ceil(s / 2)) for s in data.shape)
        self.twosided = twosided
        self.xlabel, self.ylabel = xlabel, ylabel

    def check_polar_calculated(self):
        """Ensure that the polar representation of the source data has been calculated."""
        if not self._source_polar:
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
        r, mx = self.azmax()
        r, mn = self.azmin()
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

    def plot(self, slices, lw=config.lw,
             xlim=(None, None), xscale='linear',
             ylim=(None, None), yscale='log',
             fig=None, ax=None):
        fig, ax = share_fig_ax(fig, ax)

        for slice_ in slices:
            ax.plot(*getattr(self, slice_), label=slice_)

        ax.legend(title='Slice')
        ax.set(xscale=xscale, xlim=xlim, xlabel=self.xlabel,
               yscale=yscale, ylim=ylim, ylabel=self.ylabel)


class Units:
    """Units holder for data instances."""
    def __init__(self, x, z, y=None, wavelength=None):
        """Create a new Units instance

        Parameters
        ----------
        x : `astropy.units` subclass or `str`
            unit associated with the x coordinates
        z : `astropy.units` subclass or `str`
            unit associated with the z data
        y : `astropy.units` subclass or `str`, optional
            the same as x, copied from x if not given.
        wavelength : `astropy.units` subclass or `str`, optional
            unit the wavelength is expressed in

        """
        if not y:
            y = x
        self.x, self.y, self.z = x, y, z
        self.wavelength = wavelength


class Labels:
    """Labels holder for data instances."""
    def __init__(self, xybase, z, units, unit_formatter=config.unit_formatter,
                 xy_additions=['X', 'Y'], xy_addition_side='left',
                 addition_joiner=config.xylabel_joiner,
                 unit_prefix=config.unit_prefix,
                 unit_suffix=config.unit_suffix,
                 unit_joiner=config.unit_joiner,
                 show_units=config.show_units):
        """Create a new Labels instance

        Parameters
        ----------
        xybase : `str`
            basic string used to build the X and Y labels
        z : `str`
            z label
        units : `Units`
            units instance
        unit_formatter : `str`, optional
            formatter used by astropy.units.(unit).to_string
        xy_additions : iterable, optional
            text to add to the (x, y) labels
        xy_addition_side : {'left', 'right'. 'l', 'r'}, optional
            side to add the x and y additional text to, left or right
        addition_joiner : `str`, optional
            text used to join the x or y addition
        unit_prefix : `str`, optional
            prefix used to surround the unit text
        unit_suffix : `str`, optional
            suffix used to surround the unit text
        unit_joiner : `str`, optional
            text used to combine the base label and the unit
        show_units : `bool`, optional
            whether to print units
        """
        self.xybase, self.z = xybase, z
        self.units, self.unit_formatter = units, unit_formatter
        self.xy_additions, self.xy_addition_side = xy_additions, xy_addition_side
        self.addition_joiner = addition_joiner
        self.unit_prefix, self.unit_suffix = unit_prefix, unit_suffix
        self.unit_joiner, self.show_units = unit_joiner, show_units

    def _label_factory(self, label):
        """Factory method to produce complex labels.

        Parameters
        ----------
        label : `str`, {'x', 'y', 'z'}
            label to produce

        Returns
        -------
        `str`
            completed label

        """
        if label in ('x', 'y'):
            if label == 'x':
                xy_pos = 0
            else:
                xy_pos = 1
            label_basics = [self.xy_base]
            if self.xy_addition_side.lower() in ('left', 'l'):
                label_basics.insert(0, self.xy_additions[xy_pos])
            else:
                label_basics.append(self.xy_additions[xy_pos])

            label = self.addition_joiner.join(label_basics)
        else:
            label = self.z

        unit_text = ''.join([self.unit_prefix,
                             getattr(self.units, label).to_string(self.unit_formatter),
                             self.unit_suffix])
        label = self.unit_joiner.join([label, unit_text])
        return label

    @property
    def x(self):
        """X label."""
        return self._label_factory('x')

    @property
    def y(self):
        """Y label."""
        return self._label_factory('y')

    @property
    def z(self):
        """Z label."""
        return self._label_factory('z')
