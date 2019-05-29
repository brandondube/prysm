"""Basic class holding data, used to recycle code."""
import copy

from .mathops import engine as e
from .coordinates import uniform_cart_to_polar


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
