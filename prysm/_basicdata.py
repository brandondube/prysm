"""Basic class holding data, used to recycle code."""
import copy

from scipy import interpolate

from .mathops import engine as e
from .coordinates import uniform_cart_to_polar, polar_to_cart


def fix_interp_pair(x, y):
    if y is None:
        y = 0

    if x is None:
        x = 0

    if hasattr(x, '__iter__') and not hasattr(y, '__iter__'):
        y = [y] * len(x)
    elif hasattr(y, '__iter__') and not hasattr(x, '__iter__'):
        x = [x] * len(y)

    return x, y


class BasicData:
    """Abstract base class holding some data properties."""
    _data_attr = 'data'

    def __init__(self, x, y, data):
        """Initialize a new BasicData instance.

        Parameters
        ----------
        x : `numpy.ndarray`
            x unit axis
        y : `numpy.ndarray`
            y unit axis
        data : `numpy.ndarray`
            data

        Returns
        -------
        BasicData
            the BasicData instance

        """
        self.x, self.y = x, y
        setattr(self, self._data_attr, data)
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

    @property
    def slice_x(self):
        """Retrieve a slice through the X axis of the phase.

        Returns
        -------
        self.unit : `numpy.ndarray`
            ordinate axis
        slice of self.phase or self.data : `numpy.ndarray`

        """
        return self.x, getattr(self, self._data_attr)[self.center_y, :]

    @property
    def slice_y(self):
        """Retrieve a slice through the Y axis of the phase.

        Returns
        -------
        self.unit : `numpy.ndarray`
            ordinate axis
        slice of self.phase or self.data : `numpy.ndarray`

        """
        return self.y, getattr(self, self._data_attr)[:, self.center_x]

    def copy(self):
        """Return a (deep) copy of this instance."""
        return copy.deepcopy(self)

    def slices(self, twosided=True):
        return Slices(getattr(self, self._data_attr), x=self.x, y=self.y, twosided=twosided)

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


class Slices:
    """Slices of data."""
    def __init__(self, data, x, y, twosided=True):
        self._source = data
        self._source_polar = None
        self._r = None
        self._p = None
        self._x = x
        self._y = y
        self.center_y, self.center_x = (int(e.ceil(s / 2)) for s in data.shape)
        self.twosided = twosided

    def check_polar_calculated(self):
        """Ensure that the polar representation of the source data has been calculated."""
        if not self._source_polar:
            xx, yy = e.meshgrid(self._x, self._y)
            rho, phi, polar = uniform_cart_to_polar(xx, yy, self._source)
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
