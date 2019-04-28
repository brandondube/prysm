"""Coordinate conversions."""
from scipy import interpolate

from .conf import config
from .mathops import engine as e


def cart_to_polar(x, y):
    '''Return the (rho,phi) coordinates of the (x,y) input points.

    Parameters
    ----------
    x : `numpy.ndarray` or number
        x coordinate
    y : `numpy.ndarray` or number
        y coordinate

    Returns
    -------
    rho : `numpy.ndarray` or number
        radial coordinate
    phi : `numpy.ndarray` or number
        azimuthal coordinate

    '''
    rho = e.sqrt(x ** 2 + y ** 2)
    phi = e.arctan2(y, x)
    return rho, phi


def polar_to_cart(rho, phi):
    '''Return the (x,y) coordinates of the (rho,phi) input points.

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

    '''
    x = rho * e.cos(phi)
    y = rho * e.sin(phi)
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
    xmin, xmax = min(x), max(x)
    ymin, ymax = min(y), max(y)

    _max = max(abs(e.asarray([xmin, xmax, ymin, ymax])))

    rho = e.linspace(0, _max, len(x))
    phi = e.linspace(0, 2 * e.pi, len(y))
    rv, pv = e.meshgrid(rho, phi)

    # map points to x, y and make a grid for the original samples
    xv, yv = polar_to_cart(rv, pv)

    # interpolate the function onto the new points
    f = interpolate.RegularGridInterpolator((y, x), data, bounds_error=False, fill_value=0)
    return rho, phi, f((yv, xv), method='linear')


def resample_2d(array, sample_pts, query_pts):
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

    Returns
    -------
    `numpy.ndarray`
        array resampled onto query_pts via bivariate spline

    """
    xq, yq = e.meshgrid(*query_pts)
    interpf = interpolate.RectBivariateSpline(*sample_pts, array)
    return interpf.ev(yq, xq)


def resample_2d_complex(array, sample_pts, query_pts, bounds_error=True, fill_value=0):
    '''Resamples a 2D complex array.

    Works by interpolating the magnitude and phase independently and merging the results into a complex value.

    Parameters
    ----------
    array : `numpy.ndarray`
        complex 2D array
    sample_pts : `tuple`
        pair of `numpy.ndarray` objects that contain the x and y sample locations,
        each array should be 1D
    query_pts : `tuple`
        points to interpolate onto, also 1D for each array
    bounds_error : `bool`, optional
        if True, raise if query point outside of domain of sample points
    fill_value : `float`
        value to fill with in the case of out-of-bound values

    Returns
    -------
    `numpy.ndarray`
        array resampled onto query_pts via bivariate spline

    '''
    xq, yq = e.meshgrid(*query_pts)
    interpf = interpolate.RegularGridInterpolator(sample_pts, array, bounds_error=bounds_error, fill_value=fill_value)
    return interpf((yq, xq))


def make_xy_grid(samples_x, samples_y=None, radius=1):
    """Create an x, y grid from -1, 1 with n number of samples.

    Parameters
    ----------
    samples_x : `int`
        number of samples in x direction
    samples_y : `int`
        number of samples in y direction, if None, copied from sample_x
    radius : `float`
        radius of the output array, will span -radius, radius

    Returns
    -------
    xx : `numpy.ndarray`
        x meshgrid
    yy : `numpy.ndarray`
        y meshgrid

    """
    if samples_y is None:
        samples_y = samples_x
    x = e.linspace(-radius, radius, samples_x, dtype=config.precision)
    y = e.linspace(-radius, radius, samples_y, dtype=config.precision)
    xx, yy = e.meshgrid(x, y)
    return xx, yy


def make_rho_phi_grid(samples_x, samples_y=None, aligned='x', radius=1):
    """Create an rho, phi grid from -1, 1 with n number of samples.

    Parameters
    ----------
    samples_x : `int`
        number of samples in x direction
    samples_y : `int`
        number of samples in y direction, if None, copied from sample_x
    radius : `float`
        radius of the output array

    Returns
    -------
    rho : `numpy.ndarray`
        radial meshgrid
    phi : `numpy.ndarray`
        angular meshgrid

    """
    xx, yy = make_xy_grid(samples_x, samples_y, radius)
    if aligned == 'x':
        rho, phi = cart_to_polar(xx, yy)
    else:
        rho, phi = cart_to_polar(yy, xx)
    return rho, phi
