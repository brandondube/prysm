"""Functions used to generate various geometrical constructs."""
import math

import numpy as truenp

from scipy import spatial

from .conf import config
from .mathops import np
from .coordinates import cart_to_polar, optimize_xy_separable, polar_to_cart


def gaussian(sigma, x, y, center=(0, 0)):
    """Generate a gaussian mask with a given sigma.

    Parameters
    ----------
    sigma : float
        width parameter of the gaussian, expressed in the same units as x and y
    x : numpy.ndarray
        x spatial coordinates, 2D or 1D
    y : numpy.ndarray
        y spatial coordinates, 2D or 1D
    center : tuple of float
        center of the gaussian, (x,y)

    Returns
    -------
    numpy.ndarray
        mask with gaussian shape

    """
    s = sigma

    x, y = optimize_xy_separable(x, y)

    x0, y0 = center
    return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / s ** 2)


def rectangle(width, x, y, height=None, angle=0):
    """Generate a rectangular, with the "width" axis aligned to 'x'.

    Parameters
    ----------
    width : float
        diameter of the rectangle, relative to the width of the array.
        width=1 fills the horizontal extent when angle=0
    height : float
        diameter of the rectangle, relative to the height of the array.
        height=1 fills the vertical extent when angle=0.
        If None, inherited from width to make a square
    angle : float
        angle
    x : numpy.ndarray
        x spatial coordinates, 2D
    y : numpy.ndarray
        y spatial coordinates, 2D

    Returns
    -------
    numpy.ndarray
        array with the rectangle painted at 1 and the background at 0

    """
    if angle != 0:
        if angle == 90:  # for the 90 degree case, just swap x and y
            x, y = y, x
        else:
            r, p = cart_to_polar(x, y)
            p_adj = np.radians(angle)
            p += p_adj
            x, y = polar_to_cart(r, p)
    else:
        x, y = optimize_xy_separable(x, y)

    if height is None:
        height = width
    w_mask = (y <= height) & (y >= -height)
    h_mask = (x <= width) & (x >= -width)
    return w_mask & h_mask


def rotated_ellipse(width_major, width_minor, x, y, major_axis_angle=0):
    """Generate a binary mask for an ellipse, centered at the origin.

    The major axis will notionally extend to the limits of the array, but this
    will not be the case for rotated cases.

    Parameters
    ----------
    width_major : float
        width of the ellipse in its major axis
    width_minor : float
        width of the ellipse in its minor axis
    major_axis_angle : float
        angle of the major axis w.r.t. the x axis, degrees
    x : numpy.ndarray
        x spatial coordinates, 2D
    y : numpy.ndarray
        y spatial coordinates, 2D

    Returns
    -------
    numpy.ndarray
        An ndarray of shape (samples,samples) of value 0 outside the ellipse and value 1 inside the ellipse

    Notes
    -----
    The formula applied is:
         ((x-h)cos(A)+(y-k)sin(A))^2      ((x-h)sin(A)+(y-k)cos(A))^2
        ______________________________ + ______________________________ 1
                     a^2                               b^2

    where x and y are the x and y dimensions, A is the rotation angle of the
    major axis, h and k are the centers of the the ellipse, and a and b are
    the major and minor axis widths.  In this implementation, h=k=0 and the
    formula simplifies to:

            (x*cos(A)+y*sin(A))^2             (x*sin(A)+y*cos(A))^2
        ______________________________ + ______________________________ 1
                     a^2                               b^2

    see SO:
    https://math.stackexchange.com/questions/426150/what-is-the-general-equation-of-the-ellipse-that-is-not-in-the-origin-and-rotate

    Raises
    ------
    ValueError
        if minor axis width is larger than major axis width

    """
    # TODO: can this be optimized with separable x, y?
    if width_minor > width_major:
        raise ValueError('By definition, major axis must be larger than minor.')

    arr = np.ones_like(x)

    A = np.radians(-major_axis_angle)
    a, b = width_major, width_minor
    major_axis_term = ((x * np.cos(A) + y * np.sin(A)) ** 2) / a ** 2
    minor_axis_term = ((x * np.sin(A) - y * np.cos(A)) ** 2) / b ** 2
    arr[major_axis_term + minor_axis_term > 1] = 0
    return arr


def square(x, y):
    """Create a square mask.

    Parameters
    ----------
    samples : int, optional
        number of samples in the square output array
    x : numpy.ndarray
        x spatial coordinates, 2D
    y : numpy.ndarray
        y spatial coordinates, 2D

    Returns
    -------
    numpy.ndarray
        binary ndarray representation of the mask

    """
    return np.ones_like(x)


def truecircle(radius, r):
    """Create a "true" circular mask with anti-aliasing.

    Parameters
    ----------
    samples : int, optional
        number of samples in the square output array
    radius : float, optional
        radius of the shape in the square output array.  radius=1 will fill the
    r : numpy.ndarray
        radial coordinate, 2D

    Returns
    -------
    numpy.ndarray
        nonbinary ndarray representation of the mask

    Notes
    -----
    Based on a more general algorithm by Jim Fienup

    """
    if radius == 0:
        return np.zeros_like(r)
    else:
        samples = r.shape[0]
        one_pixel = 2 / samples
        radius_plus = radius + (one_pixel / 2)
        intermediate = (radius_plus - r) * (samples / 2)
        return np.minimum(np.maximum(intermediate, 0), 1)


def circle(radius, r):
    """Create a circular mask.

    Parameters
    ----------
    radius : float
        radius of the circle, same units as r.  The return is 1 inside the
        radius and 0 outside
    r : numpy.ndarray
        2D array of radial coordinates

    Returns
    -------
    numpy.ndarray
        binary ndarray representation of the mask

    """
    return r <= radius


def regular_polygon(sides, radius, x, y, center=(0, 0), rotation=0):
    """Generate a regular polygon mask with the given number of sides.

    Parameters
    ----------
    sides : int
        number of sides to the polygon
    radius : float, optional
        radius of the regular polygon.  For R=1, will fill the x and y extent
    x : numpy.ndarray
        x spatial coordinates, 2D or 1D
    y : numpy.ndarray
        y spatial coordinates, 2D or 1D
    center : tuple of float
        center of the gaussian, (x,y)
    rotation : float
        rotation of the polygon, degrees

    Returns
    -------
    numpy.ndarray
        mask for regular polygon with radius equal to the array radius

    """
    verts = _generate_vertices(sides, radius, center, rotation)
    return _generate_mask(verts, x, y)


def _generate_mask(vertices, x, y):
    """Create a filled convex polygon mask based on the given vertices.

    Parameters
    ----------
    vertices : iterable
        ensemble of vertice (x,y) coordinates, in array units
    x : numpy.ndarray
        x spatial coordinates, 2D or 1D
    y : numpy.ndarray
        y spatial coordinates, 2D or 1D

    Returns
    -------
    numpy.ndarray
        polygon mask

    """
    vertices = truenp.asarray(vertices)
    if hasattr(x, 'get'):
        xx = x.get()
        yy = y.get()
    else:
        try:
            xx = truenp.array(x)
            yy = truenp.array(y)
        except Exception as e:
            prev = str(e)
            raise Exception('attempted to convert array to genuine numpy array with known methods.  Please make a PR to prysm with a mechanism to convert this data type to real numpy. failed with '+prev)  # NOQA

    xxyy = truenp.stack((xx, yy), axis=2)
    # use delaunay to fill from the vertices and produce a mask
    triangles = spatial.Delaunay(vertices, qhull_options='QJ Qf')
    mask = ~(triangles.find_simplex(xxyy) < 0)
    return mask


def _generate_vertices(sides, radius=1, center=(0, 0), rotation=0):
    """Generate a list of vertices for a convex regular polygon with the given number of sides and radius.

    Parameters
    ----------
    sides : int
        number of sides to the polygon
    radius : float
        radius of the polygon
    center : tuple
        center of the vertices, (x,y)
    rotation : float
        rotation of the vertices, degrees

    Returns
    -------
    numpy.ndarray
        array with first column X points, second column Y points

    """
    angle = 2 * truenp.pi / sides
    rotation = truenp.radians(rotation)
    x0, y0 = center
    points = truenp.arange(sides, dtype=config.precision)
    x = radius * truenp.sin(points * angle + rotation) + x0
    y = radius * truenp.cos(points * angle + rotation) + y0
    return truenp.stack((x, y), axis=1)


def spider(vanes, width, x, y, rotation=0, center=(0, 0), rotation_is_rad=False):
    """Generate the mask for a spider.

    Parameters
    ----------
    vanes : int
        number of spider vanes
    width : float
        width of the vanes in array units, i.e. a width=1/128 spider with
        arydiam=1 and samples=128 will be 1 pixel wide
    x : numpy.ndarray
        x spatial coordinates, 2D or 1D
    y : numpy.ndarray
        y spatial coordinates, 2D or 1D
    rotation : float, optional
        rotational offset of the vanes, clockwise
    center : tuple of float
        point from which the vanes emanate, (x,y)

    Returns
    -------
    numpy.ndarray
        array, 0 inside the spider and 1 outside

    """
    # generate the basic grid
    width = width / 2
    x0, y0 = center
    r, p = cart_to_polar(x-x0, y-y0)

    if rotation != 0:
        if not rotation_is_rad:
            rotation = np.radians(rotation)
        p = p - rotation

    # compute some constants
    rotation = np.radians(360 / vanes)

    # initialize a blank mask
    mask = np.zeros(x.shape, dtype=bool)
    for multiple in range(vanes):
        # iterate through the vanes and generate a mask for each
        # adding it to the initialized mask
        offset = rotation * multiple
        if offset != 0:
            pp = p + offset
        else:
            pp = p

        xxx, yyy = polar_to_cart(r, pp)
        mask_ = (xxx > 0) & (abs(yyy) < width)
        mask |= mask_

    return ~mask


def offset_circle(radius, x, y, center):
    """Rasterize an offset circle.

    Parameters
    ----------
    radius : float
        radius of the circle, same units as x and y
    x : numpy.ndarray
        array of x coordinates
    y : numpy.ndarray
        array of y coordinates
    center : tuple
        tuple of (x, y) centers

    Returns
    -------
    numpy.ndarray
        ndarray containing the boolean mask

    """
    x, y = optimize_xy_separable(x, y)
    # no in-place ops, x, y are shared memory
    x = x - center[0]
    y = y - center[1]
    # not cart to polar; computing theta is waste work
    r = np.hypot(x, y)
    return circle(radius, r)


def _circle_arc(t0, t1, r, N, center=(0, 0)):
    cx, cy = center
    span = t1-t0
    incr = span/N
    pts = []
    for j in range(N):
        theta = t0+(incr*j)
        x = cx + np.cos(theta) * r
        y = cy + np.sin(theta) * r
        pts.append((x, y))

    return pts


def _qhull_points_for_rectangle_with_corner_fillets(width, height, cradius, x, y, center=(0, 0), rotation=0):
    dx = x[0, 1] - x[0, 0]
    # need circumference/4/dx points on the circle
    # 4 = quarter-arc
    # parametric equation of a circle is x=cos(theta)*r, y=sin(theta0*r)
    C = 2*np.pi*cradius
    Ncirc = math.ceil(C/4/dx)

    cx, cy = center

    # extremes of the rectangle
    ledge = -width+cx
    redge = +width+cx
    top = height+cy
    bottom = -height+cy

    all_points = []
    # the basic gist of this algorithm
    #
    #
    # the rectangle is:
    # x----------------------------------x
    # |                                  |
    # |                                  |
    # |                                  |
    # |                                  |
    # |                                  |
    # |                                  |
    # x----------------------------------x
    # find the point at which we transition from the rectangle to the
    # circle, and the center of that circle:
    # x----------------------------------x
    # |     ^                            |
    # |     |                            |
    # | <-  .                            |
    # |                                  |
    # |                                  |
    # |                                  |
    # x----------------------------------x

    # enumerate the points (last_p_rec, p_circ0, p_circ1, ..p_circN, first_p_rec)
    # going around clockwise from top left
    #
    # give those to Qhull and shade the interior from the simplices
    all_points = []

    # top left
    circle_cx = ledge+cradius
    circle_cy = top-cradius
    top_left_leading_extreme_rect = (ledge, circle_cy)
    top_left_trailing_extreme_rect = (circle_cx, top)

    all_points.append(top_left_leading_extreme_rect)
    all_points += _circle_arc(np.pi, np.pi/2, cradius, Ncirc, center=(circle_cx, circle_cy))
    all_points.append(top_left_trailing_extreme_rect)

    # top right
    circle_cx = redge-cradius
    circle_cy = top-cradius
    top_right_leading_extreme_rect = (circle_cx, top)
    top_right_trailing_extreme_rect = (redge, circle_cy)

    all_points.append(top_right_leading_extreme_rect)
    all_points += _circle_arc(np.pi/2, 0, cradius, Ncirc, center=(circle_cx, circle_cy))
    all_points.append(top_right_trailing_extreme_rect)

    # bottom right
    circle_cx = redge-cradius
    circle_cy = bottom+cradius
    bottom_right_leading_extreme_rect = (redge, circle_cy)
    bottom_right_trailing_extreme_rect = (circle_cx, bottom)

    all_points.append(bottom_right_leading_extreme_rect)
    all_points += _circle_arc(0, -np.pi/2, cradius, Ncirc, center=(circle_cx, circle_cy))
    all_points.append(bottom_right_trailing_extreme_rect)

    # bottom left
    circle_cx = ledge+cradius
    circle_cy = bottom+cradius
    bottom_right_leading_extreme_rect = (circle_cx, bottom)
    bottom_right_trailing_extreme_rect = (ledge, circle_cy)

    all_points.append(bottom_right_leading_extreme_rect)
    all_points += _circle_arc(-np.pi/2, -np.pi, cradius, Ncirc, center=(circle_cx, circle_cy))
    all_points.append(bottom_right_trailing_extreme_rect)

    return all_points


def rectangle_with_corner_fillets(width, height, cradius, x, y, center=(0, 0), rotation=0):
    """Shade a rectangle with filleted (circular arc) corners.

    Parameters
    ----------
    width : float
        half-width of the rectangle, same units as x and y
    height : float
        half-height of the rectangle, same units as x and y
    cradius : float
        radius of the corner fillets
    x : numpy.ndarray
        x coordinates
    y : numpy.ndarray
        y coordinates
    center : tuple of float
        (x,y) center of the rectangle
    rotation : float
        degrees of rotation **about coordinate grid center**

    Returns
    -------
    numpy.ndarray
        1 inside "squircle", 0 outside

    """
    points = _qhull_points_for_rectangle_with_corner_fillets(width, height, cradius, x, y, center=center)

    if rotation != 0:
        r, t = cart_to_polar(x, y)
        t += truenp.radians(rotation)
        x, y = polar_to_cart(r, t)

    xxyy = truenp.stack((x, y), axis=2)
    triangles = spatial.Delaunay(points, qhull_options='QJ Qf')
    mask = ~(triangles.find_simplex(xxyy) < 0)
    return mask
