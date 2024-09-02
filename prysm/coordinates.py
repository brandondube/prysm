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

    data = np.ascontiguousarray(data)

    if not x.flags.owndata:
        x = x.copy()
        x.setflags(write=True)

    if not y.flags.owndata:
        y = y.copy()
        y.setflags(write=True)

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
    interpf = interpolate.RegularGridInterpolator(sample_pts, array, method=kind)
    return interpf(query_pts)


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


def make_rotation_matrix(zyx, radians=False):
    """Build a rotation matrix.

    Parameters
    ----------
    zyx : tuple of float
        Z, Y, X rotation angles in that order
    radians : bool, optional
        if True, abg are assumed to be radians.  If False, abg are
        assumed to be degrees.

    Returns
    -------
    numpy.ndarray
        3x3 rotation matrix

    """
    ZYX = truenp.zeros(3)
    ZYX[:len(zyx)] = zyx
    zyx = ZYX
    if not radians:
        zyx = truenp.radians(zyx)

    # alpha, beta, gamma = abg
    gamma, beta, alpha = zyx
    cos1 = truenp.cos(alpha)
    cos2 = truenp.cos(beta)
    cos3 = truenp.cos(gamma)
    sin1 = truenp.sin(alpha)
    sin2 = truenp.sin(beta)
    sin3 = truenp.sin(gamma)

    Rx = truenp.asarray([
        [1,    0,  0   ],  # NOQA
        [0, cos1, -sin1],
        [0, sin1,  cos1]
    ])
    Ry = truenp.asarray([
        [cos2,  0, sin2],
        [    0, 1,    0],  # NOQA
        [-sin2, 0, cos2],
    ])
    Rz = truenp.asarray([
        [cos3, -sin3, 0],
        [sin3,  cos3, 0],
        [0,        0, 1],
    ])
    m = Rx@Ry@Rz
    return m


def promote_3d_transformation_to_homography(M):
    """Convert a 3D transformation to 4D homography."""
    out = truenp.zeros((4, 4), dtype=config.precision)
    out[:3, :3] = M
    out[3, 3] = 1
    return out


def promote_affine_transformation_to_homography(Maff):
    out = truenp.zeros((3, 3), dtype=config.precision)
    out[:2, :3] = Maff
    out[3, 3] = 1
    return out


def make_homomorphic_translation_matrix(tx=0, ty=0, tz=0):
    out = np.eye(4, dtype=config.precision)
    out[0, -1] = tx
    out[1, -1] = ty
    out[2, -1] = tz
    return out


def drop_z_3d_transformation(M):
    """Drop the Z entries of a 3D homography.

    Drops the third row and third column of 4D transformation matrix M.

    Parameters
    ----------
    M : numpy.ndarray
        4x4 ndarray for (x, y, z, w)

    Returns
    -------
    numpy.ndarray
        3x3 array, (x, y, w)

    """
    mask = [0, 1, 3]
    # first bracket: drop output Z row, second bracket: drop input Z column
    M = M[mask][:, mask]
    return np.ascontiguousarray(M)  # assume this will get used a million times


def pack_xy_to_homographic_points(x, y):
    """Pack (x, y) vectors into a vector of coordinates in homogeneous form.

    Parameters
    ----------
    x : numpy.ndarray
        x points
    y : numpy.ndarray
        y points

    Returns
    -------
    numpy.ndarray
        3xN array (x, y, w)

    """
    out = np.empty((3, x.size), dtype=x.dtype)
    out[0, :] = x.ravel()
    out[1, :] = y.ravel()
    out[2, :] = 1
    return out


def apply_homography(M, x, y):
    points = pack_xy_to_homographic_points(x, y)
    xp, yp, w = M @ points
    xp /= w
    yp /= w
    if x.ndim > 1:
        xp = np.reshape(xp, x.shape)
        yp = np.reshape(yp, x.shape)
    return xp, yp


def solve_for_planar_homography(src, dst):
    """Find the planar homography that transforms src -> dst.

    Parameters
    ----------
    src : numpy.ndarray
        (N, 2) shaped array
    dst : numpy.ndarray
        (N, 2) shaped ndarray

    Returns
    -------
    numpy.ndarray
        3x3 array containing the planar homography such that H * src = dst

    """
    x1, y1 = src.T
    N = len(x1)
    x2, y2 = dst.T
    # TODO: sensitive to numerical precision?
    A = np.zeros((2*N, 9), dtype=config.precision)
    for i in range(N):
        # A[i]   = [-x1,    -y1,    -1, 0, 0, 0, x2x1,        x2y1,        x2   ]
        A[2*i]   = [-x1[i], -y1[i], -1, 0, 0, 0, x2[i]*x1[i], x2[i]*y1[i], x2[i]]  # NOQA
        # A[i+1] = [0, 0, 0, -x1,    -y1,    -1, y2x1,        y2y1,        y2   ]
        A[2*i+1] = [0, 0, 0, -x1[i], -y1[i], -1, y2[i]*x1[i], y2[i]*y1[i], y2[i]]

    ATA = A.T@A
    U, sigma, Vt = np.linalg.svd(ATA)
    return Vt[-1].reshape((3, 3))


def warp(img, xnew, ynew):
    """Warp an image, via "pull" and not "push".

    Parameters
    ----------
    img : numpy.ndarray
        2D ndarray
    xnew : numpy.ndarray
        2D array containing x or column coordinates to look up in img
    ynew : numpy.ndarray
        2D array containing y or row    coordinates to look up in img

    Returns
    -------
    numpy.ndarray
        "pulled" warped image

    Notes
    -----
    The meaning of pull is that the indices of the output array indices
    are the output image coordinates, in other words xnew/ynew specify
    the coordinates in img, at which each output pixel is looked up

    this is a dst->src mapping, aka "pull" in common image processing
    vernacular

    """
    # user provides us (x, y), we provide scipy (row, col) = (y, x)
    return ndimage.map_coordinates(img, (ynew, xnew))


def distort_annular_grid(r, eps):
    """Distort an annular grid, such that an annulus becomes the unit circle.

    This function is used to distort the grid before computing annular Zernike
    or other polynomials

    r and eps should be in the range [0,1]

    Parameters
    ----------
    r : numpy.ndarray
        Undistorted grid of normalized radial coordinates
    eps : float
        linear obscuration fraction, radius, not diameter;
        e.g. for a telescope with 20% diameter linear obscuration, eps=0.1

    Returns
    -------
    numpy.ndarray
        distorted r, to be passed to a polynomial function

    """
    rr = r-eps
    rr = rr * (1/(1-eps))
    return rr
