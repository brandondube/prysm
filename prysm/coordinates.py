"""Coordinate conversions."""
import numpy as truenp

from .conf import config
from .mathops import np, interpolate, ndimage
from .fttools import fftrange


def optimize_xy_separable(x, y):
    """Optimize performance for downstream operations.

    Parameters
    ----------
    x : numpy.ndarray
        2D or 1D array
    y : numpy.ndarray
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
    else:
        x = x.reshape(1, -1)
        y = y.reshape(-1, 1)

    return x, y


def broadcast_1d_to_2d(x, y):
    """Broadcast two (x,y) vectors to 2D.

    Parameters
    ----------
    x : numpy.ndarray
        ndarray of shape (n,)
    y : numpy.ndarray
        ndarray of shape (m,)

    Returns
    -------
    xx : numpy.ndarray
        ndarray of shape (m, n)
    yy : numpy.ndarray
        ndarray of shape (m, n)

    """
    shpx = (y.size, x.size)
    shpy = (x.size, y.size)
    xx = np.broadcast_to(x, shpx)
    yy = np.broadcast_to(y, shpy).T
    return xx, yy


def cart_to_polar(x, y, vec_to_grid=True):
    """Return the (rho,phi) coordinates of the (x,y) input points.

    Parameters
    ----------
    x : numpy.ndarray or number
        x coordinate
    y : numpy.ndarray or number
        y coordinate
    vec_to_grid : bool, optional
        if True, convert a vector (x,y) input to a grid (r,t) output

    Returns
    -------
    rho : numpy.ndarray or number
        radial coordinate
    phi : numpy.ndarray or number
        azimuthal coordinate

    """
    # if given x, y as vectors, and the user wants a grid out
    # don't need to check y, let np crash for the user
    # hasattr introduces support for scalars as well as array-likes
    if vec_to_grid and hasattr(x, 'ndim') and x.ndim == 1:
        y = y[:, np.newaxis]
        x = x[np.newaxis, :]

    rho = np.hypot(x, y)
    phi = np.arctan2(y, x)
    return rho, phi


def polar_to_cart(rho, phi):
    """Return the (x,y) coordinates of the (rho,phi) input points.

    Parameters
    ----------
    rho : numpy.ndarray or number
        radial coordinate
    phi : numpy.ndarray or number
        azimuthal coordinate

    Returns
    -------
    x : numpy.ndarray or number
        x coordinate
    y : numpy.ndarray or number
        y coordinate

    """
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


def uniform_cart_to_polar(x, y, data):
    """Interpolate data uniformly sampled in cartesian coordinates to polar coordinates.

    Parameters
    ----------
    x : numpy.ndarray
        sorted 1D array of x sample pts
    y : numpy.ndarray
        sorted 1D array of y sample pts
    data : numpy.ndarray
        data sampled over the (x,y) coordinates

    Returns
    -------
    rho : numpy.ndarray
        samples for interpolated values
    phi : numpy.ndarray
        samples for interpolated values
    f(rho,phi) : numpy.ndarray
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
    array : numpy.ndarray
        2D array
    sample_pts : tuple
        pair of numpy.ndarray objects that contain the x and y sample locations,
        each array should be 1D
    query_pts : tuple
        points to interpolate onto, also 1D for each array
    kind : str, {'linear', 'cubic', 'quintic'}
        kind / order of spline to use

    Returns
    -------
    numpy.ndarray
        array resampled onto query_pts

    """
    interpf = interpolate.interp2d(*sample_pts, array, kind=kind)
    return interpf(*query_pts)


def resample_2d_complex(array, sample_pts, query_pts, kind='linear'):
    """Resample 2D array to be sampled along queried points.

    Parameters
    ----------
    array : numpy.ndarray
        2D array
    sample_pts : tuple
        pair of numpy.ndarray objects that contain the x and y sample locations,
        each array should be 1D
    query_pts : tuple
        points to interpolate onto, also 1D for each array
    kind : str, {'linear', 'cubic', 'quintic'}
        kind / order of spline to use

    Returns
    -------
    numpy.ndarray
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
    shape : int or tuple of int
        number of samples per dimension.  If a scalar value, broadcast to
        both dimensions.  Order is numpy axis convention, (row, col)
    dx : float
        inter-sample spacing, ignored if diameter is provided
    diameter : float
        diameter, clobbers dx if both given
    grid : bool, optional
        if True, return meshgrid of x,y; else return 1D vectors (x, y)

    Returns
    -------
    x : numpy.ndarray
        x grid
    y : numpy.ndarray
        y grid

    """
    if not isinstance(shape, tuple):
        shape = (shape, shape)

    if diameter != 0:
        dx = diameter/max(shape)

    y, x = (fftrange(s, dtype=config.precision) * dx for s in shape)

    if grid:
        x, y = np.meshgrid(x, y)

    return x, y


def make_rotation_matrix(abg, radians=False):
    """Build a rotation matrix.

    The angles are Tait-Bryan angles describing extrinsic rotations about
    Z, Y, X in that order.

    Note that the return is the location of the input points in the output
    space

    For more information, see Wikipedia
    https://en.wikipedia.org/wiki/Euler_angles#Tait%E2%80%93Bryan_angles
    The "Tait-Bryan angles" Z1X2Y3 entry is the rotation matrix
    used in this function.


    Parameters
    ----------
    abg : tuple of float
        the Tait-Bryan angles (α,β,γ)
        units of degrees unless radians=True
        if len < 3, remaining angles are zero
        beta produces horizontal compression and gamma vertical
    radians : bool, optional
        if True, abg are assumed to be radians.  If False, abg are
        assumed to be degrees.

    Returns
    -------
    numpy.ndarray
        3x3 rotation matrix

    """
    ABG = truenp.zeros(3)
    ABG[:len(abg)] = abg
    abg = ABG
    if not radians:
        abg = truenp.radians(abg)

    alpha, beta, gamma = abg
    cos1 = truenp.cos(alpha)
    cos2 = truenp.cos(beta)
    cos3 = truenp.cos(gamma)
    sin1 = truenp.sin(alpha)
    sin2 = truenp.sin(beta)
    sin3 = truenp.sin(gamma)
    # # originally wrote this as a Homomorphic matrix
    # # the m = m[:3,:3] crops it to just the rotation matrix
    # # unclear if may some day want the Homomorphic matrix,
    # # PITA to take it out, so leave it in
    m = truenp.asarray([
        [cos1*cos3 - sin1*sin2*sin3, -cos2*sin1, cos1*sin3 + cos3*sin1*sin2, 0],
        [cos3*sin1 + cos1*sin2*sin3,  cos1*cos2, sin1*sin3 - cos1*cos3*sin2, 0],
        [-cos2*sin3,                  sin2,      cos2*cos3,                  0],
        [0,                           0,         0,                          1],
    ], dtype=config.precision)
    # bit of a weird dance with truenp/np here
    # truenp -- make "m" on CPU, no matter what.
    # np.array on last line will move data from numpy to any other "numpy"
    # (like Cupy/GPU)
    return np.asarray(m[:3, :3])
    # Rx = truenp.asarray([
    #     [1,    0,  0   ],  # NOQA
    #     [0, cos1, -sin1],
    #     [0, sin1,  cos1]
    # ])
    # Ry = truenp.asarray([
    #     [cos2,  0, sin2],
    #     [    0, 1,    0],  # NOQA
    #     [-sin2, 0, cos2],
    # ])
    # Rz = truenp.asarray([
    #     [cos3, -sin3, 0],
    #     [sin3,  cos3, 0],
    #     [0,        0, 1],
    # ])
    # m = Rz@Ry@Rx
    # return m


def make_translation_matrix(tx, ty):
    m = truenp.asarray([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1],
    ], dtype=config.precision)
    return np.asarray(m)


def apply_transformation_matrix(m, x, y, z=None, points=None, return_z=False):
    """Apply the coordinate transformation m to the coordinates (x,y,[z]).

    Parameters
    ----------
    m : numpy.ndarray, optional
        transormation matrix; see make_rotation_matrix, make_translation_matrix
    x : numpy.ndarray
        N dimensional array of x coordinates
    y : numpy.ndarray
        N dimensional array of x coordinates
    z : numpy.ndarray
        N dimensional array of z coordinates
        assumes to be unity if not given
    points : numpy.ndarray, optional
        array of dimension [x.size, 3] containing [x,y,z]
        points will be made by stacking x,y,z if not given.
        passin3 points directly if this is the native storage
        of your coordinates can improve performance.
    return_z : bool, optional
        if True, returns array of shape [3, x.shape]
        if False, returns an array of shape [2, x.shape]
        either return unpacks, such that x, y = rotate(...)

    Returns
    -------
    numpy.ndarray
        ndarray with rotated coordinates

    """
    if z is None:
        z = np.ones_like(x)

    if points is None:
        points = np.stack((x, y, z), axis=2)

    out = np.tensordot(m, points, axes=((1), (2)))
    if return_z:
        return out
    else:
        return out[:2, ...]


def make_3D_rotation_affine(m):
    """Convert a 3D rotation matrix to an affine transform.

    Assumes the rotation is viewed from the birdseye perspective, aka directly
    overhead.

    Parameters
    ----------
    m : numpy.ndarray
        3x3 rotation matrix

    Returns
    -------
    numpy.ndarray
        2x2 affine transform matrix

    """
    return m[:2, :2]


def warp(img, xnew, ynew):
    return ndimage.map_coordinates(img, xnew, ynew)
