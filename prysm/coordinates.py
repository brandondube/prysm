"""Coordinate conversions."""
import numpy as np
from scipy import interpolate

from .conf import config
from prysm import mathops as m


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
    rho = m.sqrt(x ** 2 + y ** 2)
    phi = m.atan2(y, x)
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
    x = rho * m.cos(phi)
    y = rho * m.sin(phi)
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

    Notes
    -----
    Assumes data is sampled along x = [-1,1] and y = [-1,1] over a square grid.

    """
    # create a set of polar coordinates to interpolate onto
    xmax = x[-1]
    num_pts = len(x)
    rho = np.linspace(0, xmax, num_pts / 2)
    phi = np.linspace(0, 2 * m.pi, num_pts)
    rv, pv = np.meshgrid(rho, phi)

    # map points to x, y and make a grid for the original samples
    xv, yv = polar_to_cart(rv, pv)

    # interpolate the function onto the new points
    f = interpolate.RegularGridInterpolator((x, y), data)
    return rho, phi, f((xv, yv), method='linear')


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
    xq, yq = np.meshgrid(*query_pts)
    interpf = interpolate.RectBivariateSpline(*sample_pts, array)
    return interpf.ev(yq, xq)


def resample_2d_complex(array, sample_pts, query_pts):
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

    Returns
    -------
    `numpy.ndarray`
        array resampled onto query_pts via bivariate spline

    '''
    xq, yq = np.meshgrid(*query_pts)
    mag = abs(array)
    phase = np.angle(array)

    magfunc = interpolate.RegularGridInterpolator(sample_pts, mag)
    phasefunc = interpolate.RegularGridInterpolator(sample_pts, phase)

    interp_mag = magfunc((yq, xq))
    interp_phase = phasefunc((yq, xq))

    return interp_mag * m.exp(1j * interp_phase)


def make_xy_grid(samples):
    """Create an x, y grid from -1, 1 with n number of samples.

    Parameters
    ----------
    samples : `int`
        number of samples in the output grid, will have shape (samples, samples)

    Returns
    -------
    xx : `numpy.ndarray`
        x meshgrid
    yy : `numpy.ndarray`
        y meshgrid

    """
    x = np.linspace(-1, 1, samples, dtype=config.precision)
    y = np.linspace(-1, 1, samples, dtype=config.precision)
    xx, yy = np.meshgrid(x, y)
    return xx, yy


def make_rho_phi_grid(samples, aligned='x'):
    """Create an rho, phi grid from -1, 1 with n number of samples.

    Parameters
    ----------
    samples : `int`
        number of samples in the output grid, will have shape (samples, samples)

    Returns
    -------
    rho : `numpy.ndarray`
        radial meshgrid
    phi : `numpy.ndarray`
        angular meshgrid

    """
    xx, yy = make_xy_grid(samples)
    if aligned == 'x':
        rho, phi = cart_to_polar(xx, yy)
    else:
        rho, phi = cart_to_polar(yy, xx)
    return rho, phi
