"""Coordinate conversions."""
from .conf import config
from .mathops import np, interpolate_engine as interpolate
from .fttools import fftrange


def optimize_xy_separable(x, y):
    """Optimize performance for downstream operations.

    Parameters
    ----------
    x : `numpy.ndarray`
        2D or 1D array
    y : `numpy.ndarray`
        2D or 1D array

    Returns
    -------
    x, y
        optimized arrays (x as 1D row, y as 1D column)

    Notes
    -----
    If a calculation is separable in x and y, performing it on a meshgrid of x/y
    takes 2N^2 operations, for N= the linear dimension (the 2 being x and y).
    If the calculation is separable, this can be reduced to 2N by using numpy
    broadcast functionality and two 1D calculations.

    """
    if x.ndim == 2:
        # assume same dimensionality of x and y
        # second indexing converts y to a broadcasted column vector
        x = x[0, :]
        y = y[:, 0][:, np.newaxis]

    return x, y


def cart_to_polar(x, y, vec_to_grid=True):
    """Return the (rho,phi) coordinates of the (x,y) input points.

    Parameters
    ----------
    x : `numpy.ndarray` or number
        x coordinate
    y : `numpy.ndarray` or number
        y coordinate
    vec_to_grid : `bool`, optional
        if True, convert a vector (x,y) input to a grid (r,t) output

    Returns
    -------
    rho : `numpy.ndarray` or number
        radial coordinate
    phi : `numpy.ndarray` or number
        azimuthal coordinate

    """
    # if given x, y as vectors, and the user wants a grid out
    # don't need to check y, let np crash for the user
    # hasattr introduces support for scalars as well as array-likes
    if vec_to_grid and hasattr(x, 'ndim') and x.ndim == 1:
        y = y[:, np.newaxis]
        x = x[np.newaxis, :]

    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return rho, phi


def polar_to_cart(rho, phi):
    """Return the (x,y) coordinates of the (rho,phi) input points.

    Parameters
    ----------
    rho : `numpy.ndarray` or number
        radial coordinate
    phi : `numpy.ndarray` or number
        azimuthal coordinate

    Returns
    -------
    x : `numpy.ndarray` or number
        x coordinate
    y : `numpy.ndarray` or number
        y coordinate

    """
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


def uniform_cart_to_polar(x, y, data):
    """Interpolate data uniformly sampled in cartesian coordinates to polar coordinates.

    Parameters
    ----------
    x : `numpy.ndarray`
        sorted 1D array of x sample pts
    y : `numpy.ndarray`
        sorted 1D array of y sample pts
    data : `numpy.ndarray`
        data sampled over the (x,y) coordinates

    Returns
    -------
    rho : `numpy.ndarray`
        samples for interpolated values
    phi : `numpy.ndarray`
        samples for interpolated values
    f(rho,phi) : `numpy.ndarray`
        data uniformly sampled in (rho,phi)

    """
    # create a set of polar coordinates to interpolate onto
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    _max = max(abs(np.asarray([xmin, xmax, ymin, ymax])))

    rho = np.linspace(0, _max, len(x))
    phi = np.linspace(0, 2 * np.pi, len(y))
    rv, pv = np.meshgrid(rho, phi)

    # map points to x, y and make a grid for the original samples
    xv, yv = polar_to_cart(rv, pv)

    # interpolate the function onto the new points
    f = interpolate.RegularGridInterpolator((y, x), data, bounds_error=False, fill_value=0)
    return rho, phi, f((yv, xv), method='linear')


def resample_2d(array, sample_pts, query_pts, kind='cubic'):
    """Resample 2D array to be sampled along queried points.

    Parameters
    ----------
    array : `numpy.ndarray`
        2D array
    sample_pts : `tuple`
        pair of `numpy.ndarray` objects that contain the x and y sample locations,
        each array should be 1D
    query_pts : `tuple`
        points to interpolate onto, also 1D for each array
    kind : `str`, {'linear', 'cubic', 'quintic'}
        kind / order of spline to use

    Returns
    -------
    `numpy.ndarray`
        array resampled onto query_pts

    """
    interpf = interpolate.interp2d(*sample_pts, array, kind=kind)
    return interpf(*query_pts)


def resample_2d_complex(array, sample_pts, query_pts, kind='linear'):
    """Resample 2D array to be sampled along queried points.

    Parameters
    ----------
    array : `numpy.ndarray`
        2D array
    sample_pts : `tuple`
        pair of `numpy.ndarray` objects that contain the x and y sample locations,
        each array should be 1D
    query_pts : `tuple`
        points to interpolate onto, also 1D for each array
    kind : `str`, {'linear', 'cubic', 'quintic'}
        kind / order of spline to use

    Returns
    -------
    `numpy.ndarray`
        array resampled onto query_pts

    """
    r, c = [resample_2d(a,
                        sample_pts=sample_pts,
                        query_pts=query_pts,
                        kind=kind) for a in (array.real, array.imag)]

    return r + 1j * c


def make_xy_grid(shape, *, dx=0, diameter=0, grid=True):
    """Create an x, y grid from -1, 1 with n number of samples.

    Parameters
    ----------
    shape : `int` or tuple of int
        number of samples per dimension.  If a scalar value, broadcast to
        both dimensions.  Order is numpy axis convention, (row, col)
    dx : `float`
        inter-sample spacing, ignored if diameter is provided
    diameter : `float`
        diameter, clobbers dx if both given
    grid : `bool`, optional
        if True, return meshgrid of x,y; else return 1D vectors (x, y)

    Returns
    -------
    x : `numpy.ndarray`
        x grid
    y : `numpy.ndarray`
        y grid

    """
    if not isinstance(shape, tuple):
        shape = (shape, shape)

    if diameter != 0:
        dx = diameter/shape[0]

    y, x = (fftrange(s, dtype=config.precision) * dx for s in shape)

    if grid:
        x, y = np.meshgrid(x, y)

    return x, y


class Grid:
    """Container for a grid of spatial coordinates."""

    def __init__(self, x=None, y=None, r=None, t=None):
        """Create a new Grid.

        Parameters
        ----------
        x : `numpy.ndarray`
            x coordinates, 2D
        y : `numpy.ndarray`
            y coordinates, 2D
        r : `numpy.ndarray`
            radial coordinates, 2D
        t : `numpy.ndarray`
            azimuthal coordinates, 2D

        Notes
        -----
        x and y may be None if you only require radial variables.  If r and t
        are None, they will be computed once if accessed.

        """
        self.x = x
        self.y = y
        self._r = r
        self._t = t

    @property
    def r(self):
        """Radial variable."""
        if self._r is None:
            self._r, self._t = cart_to_polar(self.x, self.y)

        return self._r

    @property
    def t(self):
        """Azimuthal variable."""
        if self._t is None:
            self._r, self._t = cart_to_polar(self.x, self.y)

        return self._t

    @property
    def dx(self):
        """Inter-sample spacing."""
        return float(self.x[1]-self.x[0])
