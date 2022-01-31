"""Basic class holding data, used to recycle code."""
import copy
from numbers import Number
from collections.abc import Iterable

from .mathops import np, interpolate
from .coordinates import cart_to_polar, make_xy_grid, uniform_cart_to_polar, polar_to_cart
from .plotting import share_fig_ax


def fix_interp_pair(x, y):
    """Ensure that x, y have the same shape.  If either is scalar, it is broadcast for each value in the other.

    Parameters
    ----------
    x : float or Iterable
        x data
    y : float or Iterable
        y data

    Returns
    -------
    Iterable, Iterable
        x, y

    """
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
    _default_twosided = True

    def __init__(self, data, dx, wavelength):
        """Initialize a new RichData instance.

        Parameters
        ----------
        data : numpy.ndarray
            2D array containing the z data
        dx : float
            inter-sample spacing, mm
        wavelength : float
            wavelength of light, um

        Returns
        -------
        RichData
            the instance

        """
        self.data = data
        self.dx = dx
        self.wavelength = wavelength
        self.interpf_x, self.interpf_y, self.interpf_2d = None, None, None
        self._x, self._y, self._r, self._t = None, None, None, None

    @property
    def shape(self):
        """Proxy to phase or data shape."""
        return self.data.shape

    @property
    def size(self):
        """Proxy to phase or data size."""
        return self.data.size

    @property
    def x(self):
        """X coordinate axis, 1D."""
        if self._x is None:
            self._x, self._y = make_xy_grid(self.data.shape, dx=self.dx)

        return self._x

    @x.setter
    def x(self, x):
        """Set a new value for the X array."""
        self._x = x

    @property
    def y(self):
        """Y coordinate axis, 1D."""
        if self._y is None:
            self._x, self._y = make_xy_grid(self.data.shape, dx=self.dx)

        return self._y

    @y.setter
    def y(self, y):
        """Set a new value for the Y array."""
        self._y = y

    @property
    def r(self):
        """r coordinate axis, 2D."""
        if self._r is None:
            self._r, self._t = cart_to_polar(self.x, self.y)

        return self._r

    @r.setter
    def r(self, r):
        self._r = r

    @property
    def t(self):
        """t coordinate axis, 2D."""
        if self._t is None:
            self._r, self._t = cart_to_polar(self.x, self.y)

        return self._t

    @t.setter
    def t(self, t):
        self._t = t

    @property
    def support_x(self):
        """Width of the domain in X."""
        return float(self.shape[1] * self.dx)

    @property
    def support_y(self):
        """Width of the domain in Y."""
        return float(self.shape[0] * self.dx)

    @property
    def support(self):
        """Width of the domain."""
        return max((self.support_x, self.support_y))

    def copy(self):
        """Return a (deep) copy of this instance."""
        return copy.deepcopy(self)

    def slices(self, twosided=None):
        """Create a Slices instance from this instance.

        Parameters
        ----------
        twosided : bool, optional
            if None, copied from self._default_twosided

        Returns
        -------
        Slices
            a Slices object

        """
        if twosided is None:
            twosided = self._default_twosided

        x, y = self.x, self.y
        x = x[0]
        y = y[..., 0]

        return Slices(data=self.data, x=x, y=y, twosided=twosided)

    def _make_interp_function_2d(self):
        """Generate a 2D interpolation function for this instance, used in sampling with exact_xy.

        Returns
        -------
        scipy.interpolate.RegularGridInterpolator
            interpolator instance.

        """
        x = self.x
        y = self.y
        x = x[0]
        y = y[..., 0]
        if self.interpf_2d is None:
            self.interpf_2d = interpolate.RegularGridInterpolator((y, x), self.data)

        return self.interpf_2d

    def _make_interp_function_xy1d(self):
        """Generate two interpolation functions for the xy slices.

        Returns
        -------
        self.interpf_x : scipy.interpolate.interp1d
            x interpolator
        self.interpf_y : scipy.interpolate.interp1d
            y interpolator

        """
        slc = self.slices()
        if self.interpf_x is None or self.interpf_y is None:
            ux, x = slc.x
            uy, y = slc.y

            self.interpf_x = interpolate.interp1d(ux, x)
            self.interpf_y = interpolate.interp1d(uy, y)

        return self.interpf_x, self.interpf_y

    def exact_polar(self, rho, phi=None):
        """Retrieve data at the specified radial coordinates pairs.

        Parameters
        ----------
        rho : iterable
            radial coordinate(s) to sample
        phi : iterable
            azimuthal coordinate(s) to sample

        Returns
        -------
        numpy.ndarray
            data at the given points

        """
        self._make_interp_function_2d()

        rho, phi = fix_interp_pair(rho, phi)
        x, y = polar_to_cart(rho, phi)
        return self.interpf_2d((y, x), method='linear')

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
        numpy.ndarray
            data at the given points

        """
        self._make_interp_function_2d()

        x, y = fix_interp_pair(x, y)
        return self.interpf_2d((y, x), method='linear')

    def exact_x(self, x):
        """Return data at an exact x coordinate along the y=0 axis.

        Parameters
        ----------
        x : number or numpy.ndarray
            x coordinate(s) to return

        Returns
        -------
        numpy.ndarray
            ndarray of values

        """
        self._make_interp_function_xy1d()
        return self.interpf_x(x)

    def exact_y(self, y):
        """Return data at an exact y coordinate along the x=0 axis.

        Parameters
        ----------
        y : number or numpy.ndarray
            y coordinate(s) to return

        Returns
        -------
        numpy.ndarray
            ndarray of values

        """
        self._make_interp_function_xy1d()
        return self.interpf_y(y)

    def plot2d(self, xlim=None, ylim=None, clim=None, cmap=None,
               log=False, power=1, interpolation=None,
               show_colorbar=True, colorbar_label=None, axis_labels=(None, None),
               fig=None, ax=None):
        """Plot data in 2D.

        Parameters
        ----------
        xlim : float or iterable, optional
            x axis limits.  If not iterable, symmetric version of the single value
        ylim : float or iterable, optional
            y axis limits.  If None and xlim is not None, copied from xlim.
            If not iterable, symmetric version of the single value.
        clim : iterable, optional
            clim passed directly to matplotlib.
            If None, looked up on self._default_clim.
        cmap : str, optional
            colormap to use, passed directly to matplotlib if not None.
            If None, looks up the default cmap for self._data_type on config
        log : bool, optional
            if True, plot on a log color scale
        power : float, optional
            if not 1, plot on a power stretched color scale
        interpolation : str, optional
            interpolation method to use, passed directly to matplotlib
        show_colorbar : bool, optional
            if True, draws the colorbar
        colorbar_label : str, optional
            label for the colorbar
        axis_labels : iterable of str,
            (x, y) axis labels.  If None, not drawn
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
        data = self.data
        x, y = self.x, self.y

        from matplotlib.colors import PowerNorm, LogNorm
        fig, ax = share_fig_ax(fig, ax)

        # sanitize some inputs
        if cmap is None:
            cmap = 'inferno'

        if interpolation is None:
            interpolation = 'lanczos'

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

        im = ax.imshow(data,
                       extent=[x.min(), x.max(), y.max(), y.min()],
                       cmap=cmap,
                       clim=clim,
                       norm=norm,
                       interpolation=interpolation)

        if show_colorbar:
            fig.colorbar(im, label=colorbar_label, ax=ax, fraction=0.046)

        xlab, ylab = axis_labels
        ax.set(xlabel=xlab, xlim=xlim, ylabel=ylab, ylim=ylim)

        return fig, ax


class Slices:
    """Slices of data."""
    def __init__(self, data, x, y, twosided=True):
        """Create a new Slices instance.

        Parameters
        ----------
        data : numpy.ndarray
            2D array of data
        x : numpy.ndarray
            1D array of x points
        y : numpy.ndarray
            1D array of y points
        twosided : bool, optional
            if True, plot slices from (-ext, ext), else from (0,ext)

        """
        self._source = data
        self._source_polar = None
        self._r = None
        self._p = None
        self._x = x
        self._y = y
        self.center_y, self.center_x = np.argmin(abs(y)), np.argmin(abs(x))  # fftrange produced x/y, so argmin=center
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
        x : numpy.ndarray
            coordinates
        slice : numpy.ndarray
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
        y : numpy.ndarray
            coordinates
        slice : numpy.ndarray
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
        rho : numpy.ndarray
            coordinates
        slice : numpy.ndarray
            values of the data array at these coordinates

        """
        self.check_polar_calculated()
        return self._r, np.nanmean(self._source_polar, axis=0)

    @property
    def azmedian(self):
        """Azimuthal median of the data.

        Returns
        -------
        rho : numpy.ndarray
            coordinates
        slice : numpy.ndarray
            values of the data array at these coordinates

        """
        self.check_polar_calculated()
        return self._r, np.nanmedian(self._source_polar, axis=0)

    @property
    def azmin(self):
        """Azimuthal minimum of the data.

        Returns
        -------
        rho : numpy.ndarray
            coordinates
        slice : numpy.ndarray
            values of the data array at these coordinates

        """
        self.check_polar_calculated()
        return self._r, np.nanmin(self._source_polar, axis=0)

    @property
    def azmax(self):
        """Azimuthal maximum of the data.

        Returns
        -------
        rho : numpy.ndarray
            coordinates
        slice : numpy.ndarray
            values of the data array at these coordinates

        """
        self.check_polar_calculated()
        return self._r, np.nanmax(self._source_polar, axis=0)

    @property
    def azpv(self):
        """Azimuthal PV of the data.

        Returns
        -------
        rho : numpy.ndarray
            coordinates
        slice : numpy.ndarray
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
        rho : numpy.ndarray
            coordinates
        slice : numpy.ndarray
            values of the data array at these coordinates

        """
        self.check_polar_calculated()
        return self._r, np.nanvar(self._source_polar, axis=0)

    @property
    def azstd(self):
        """Azimuthal standard deviation of the data.

        Returns
        -------
        rho : numpy.ndarray
            coordinates
        slice : numpy.ndarray
            values of the data array at these coordinates

        """
        self.check_polar_calculated()
        return self._r, np.nanstd(self._source_polar, axis=0)

    def plot(self, slices, lw=None, alpha=None, zorder=None, invert_x=False,
             xlim=(None, None), xscale='linear',
             ylim=(None, None), yscale='linear',
             show_legend=True, axis_labels=(None, None),
             fig=None, ax=None):
        """Plot slice(s).

        Parameters
        ----------
        slices : str or Iterable
            if a string, plots a single slice.  Else, plots several slices.
        lw : float or Iterable, optional
            line width to use for the slice(s).
            If a single value, used for all slice(s).
            If iterable, used pairwise with the slices
        alpha : float or Iterable, optional
            alpha (transparency) to use for the slice(s).
            If a single value, used for all slice(s).
            If iterable, used pairwise with the slices
        zorder : int or Iterable, optional
            zorder (stack height) to use for the slice(s).
            If a single value, used for all slice(s).
            If iterable, used pairwise with the slices
        invert_x : bool, optional
            if True, flip x (i.e., Freq => Period or vice-versa)
        xlim : tuple, optional
            x axis limits
        xscale : str, {'linear', 'log'}, optional
            scale used for the x axis
        ylim : tuple, optional
            y axis limits
        yscale : str, {'linear', 'log'}, optional
            scale used for the y axis
        show_legend : bool, optional
            if True, show the legend
        axis_labels : iterable of str,
            (x, y) axis labels.  If None, not drawn
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
        def safely_invert_x(x, v):
            # these values are unsafe for fp32.  Maybe a bit pressimistic, but that's life
            zeros = abs(x) < 1e-9
            x, v = x.copy(), v.copy()
            x[zeros] = np.nan
            v[zeros] = np.nan
            x = 1 / x
            return x, v

        # error check everything
        if alpha is None:
            alpha = 1

        if lw is None:
            lw = 2

        if zorder is None:
            zorder = 3

        if isinstance(slices, str):
            slices = [slices]

        if isinstance(alpha, Number):
            alpha = [alpha] * len(slices)

        if isinstance(lw, Number):
            lw = [lw] * len(slices)

        if isinstance(zorder, int):
            zorder = [zorder] * len(slices)

        if not hasattr(xlim, '__iter__') and self.twosided:
            xlim = (-xlim, xlim)

        fig, ax = share_fig_ax(fig, ax)

        for slice_, alpha, lw, zorder in zip(slices, alpha, lw, zorder):
            u, v = getattr(self, slice_)
            if invert_x:
                u, v = safely_invert_x(u, v)

            ax.plot(u, v, label=slice_, lw=lw, alpha=alpha, zorder=zorder)

        if show_legend:
            ax.legend(title='Slice')

        xlabel, ylabel = axis_labels

        ax.set(xscale=xscale, xlim=xlim, xlabel=xlabel,
               yscale=yscale, ylim=ylim, ylabel=ylabel)
        if invert_x:
            ax.invert_xaxis()

        return fig, ax
