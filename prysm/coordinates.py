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

    # would be more efficient to call cos and sine once, but
    # the computation of these variables will be a vanishingly
    # small faction of total runtime for this function if
    # x, y, z are of "reasonable" size

    alpha, beta, gamma = abg
    cosa = truenp.cos(alpha)
    cosb = truenp.cos(beta)
    cosg = truenp.cos(gamma)
    sina = truenp.sin(alpha)
    sinb = truenp.sin(beta)
    sing = truenp.sin(gamma)
    # originally wrote this as a Homomorphic matrix
    # the m = m[:3,:3] crops it to just the rotation matrix
    # unclear if may some day want the Homomorphic matrix,
    # PITA to take it out, so leave it in
    m = truenp.asarray([
        [cosa*cosg - sina*sinb*sing, -cosb*sina, cosa*sing + cosg*sina*sinb, 0],
        [cosg*sina + cosa*sinb*sing,  cosa*cosb, sina*sing - cosa*cosg*sinb, 0],
        [-cosb*sing,                  sinb,      cosb*cosg,                  0],
        [0,                           0,         0,                          1],
    ], dtype=config.precision)
    # bit of a weird dance with truenp/np here
    # truenp -- make "m" on CPU, no matter what.
    # np.array on last line will move data from numpy to any other "numpy"
    # (like Cupy/GPU)
    return np.asarray(m[:3, :3])


def apply_rotation_matrix(m, x, y, z=None, points=None, return_z=False):
    """Rotate the coordinates (x,y,[z]) about the origin by angles (α,β,γ).

    Parameters
    ----------
    m : numpy.ndarray, optional
        rotation matrix; see make_rotation_matrix
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
        passing points directly if this is the native storage
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


def xyXY_to_pixels(xy, XY):
    """Given input points xy and warped points XY, compute pixel indices.

    Lists or tuples work for xy and XY, as do 3D arrays.

    Parameters
    ----------
    xy : numpy.ndarray
        ndarray of shape (2, m, n)
        with [x, y] on the first dimension
        represents the input coordinates
        implicitly rectilinear
    XY : numpy.ndarray
        ndarray of shape (2, m, n)
        with [x, y] on the first dimension
        represents the input coordinates
        not necessarily rectilinear

    Returns
    -------
    numpy.ndarray
        ndarray of shape (2, m, n) with XY linearly projected
        into pixels

    """
    xy = np.array(xy)
    XY = np.array(XY)
    # map coordinates says [0,0] is the upper left corner
    # need to adjust XYZ by xyz origin and sample spacing
    # d = delta; o = origin
    x, y = xy
    ox = x[0, 0]
    oy = y[0, 0]
    dx = x[0, 1] - ox
    dy = y[1, 0] - oy
    XY2 = XY.copy()
    X, Y = XY2
    X -= ox
    Y -= oy
    X /= dx
    Y /= dy
    # ::-1 = reverse X,Y
    # ... = leave other axes as-is
    XY2 = XY2[::-1, ...]
    return XY2


def regularize(xy, XY, z, XY2=None):
    """Regularize the coordinates XY relative to the frame xy.

    This function is used in conjunction with rotate to project
    surface figure errors onto tilted planes or other geometries.

    Parameters
    ----------
    xy : numpy.ndarray
        ndarray of shape (2, m, n)
        with [x, y] on the first dimension
        represents the input coordinates
        implicitly rectilinear
    XY : numpy.ndarray
        ndarray of shape (2, m, n)
        with [x, y] on the first dimension
        represents the input coordinates
        not necessarily rectilinear
    z : numpy.ndarray
        ndarray of shape (m, n)
        flat data to warp
    XY2 : numpy.ndarray, optional
        ndarray of shape (2, m, n)
        XY, after output from xyXY_to_pixels
        compute XY2 once and pass many times
        to optimize models

    Returns
    -------
    Z : numpy.ndarray
        z which exists on the grid XY, looked up at the points xy

    """
    if XY2 is None:
        XY2 = xyXY_to_pixels(xy, XY)

    return ndimage.map_coordinates(z, XY2)
