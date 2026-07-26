"""Functions used to generate various geometrical constructs."""

# truenp: polygon vertex generation is host-side scalar work
import numpy as truenp

from .conf import config
from .mathops import np
from .coordinates import cart_to_polar, optimize_xy_separable, polar_to_cart


def antialias(d, dx):
    """Convert signed distance to pixel coverage with a one pixel edge ramp.

    Parameters
    ----------
    d : ndarray
        signed distance, negative inside the shape, same units as dx
    dx : float
        sample spacing

    Returns
    -------
    ndarray
        coverage; 1 inside, 0 outside, fractional within a pixel of the edge

    Notes
    -----
    Generalizes a gray-pixel circle algorithm by Jim Fienup.  Combine shapes
    on distance (union, intersect, subtract) and ramp once; multiplying
    already-ramped masks double counts shared edges.

    """
    coverage = 0.5 - d/dx
    return np.minimum(np.maximum(coverage, 0), 1)


def union(*ds):
    """Signed distance of the union of shapes.

    Parameters
    ----------
    ds : ndarray
        signed distance arrays

    Returns
    -------
    ndarray
        signed distance of the combined shape

    """
    out = ds[0]
    for d in ds[1:]:
        out = np.minimum(out, d)
    return out


def intersect(*ds):
    """Signed distance of the intersection of shapes.

    Parameters
    ----------
    ds : ndarray
        signed distance arrays

    Returns
    -------
    ndarray
        signed distance of the combined shape

    """
    out = ds[0]
    for d in ds[1:]:
        out = np.maximum(out, d)
    return out


def subtract(d1, d2):
    """Signed distance of shape 1 with shape 2 removed.

    Parameters
    ----------
    d1 : ndarray
        signed distance of the shape to keep
    d2 : ndarray
        signed distance of the shape to remove

    Returns
    -------
    ndarray
        signed distance of the combined shape

    """
    return np.maximum(d1, -d2)


def multisample(func, x, y, samples=8):
    """Anti-alias a membership function by multisampling within edge pixels.

    Parameters
    ----------
    func : callable
        func(x, y) -> ndarray, membership of the shape, bool or 0/1 float
    x : ndarray
        x spatial coordinates, 2D or 1D
    y : ndarray
        y spatial coordinates, 2D or 1D
    samples : int
        subsamples per axis within each edge pixel

    Returns
    -------
    ndarray
        coverage; edge pixels carry the mean of samples x samples subsamples

    Notes
    -----
    Fallback for membership functions with no signed distance; prefer
    antialias when one exists.  Edge pixels are those whose 3x3 neighborhood
    is mixed; features containing no pixel centers are invisible.  Subsamples
    are centered within each pixel -- drawing on a finer FFT-aligned grid and
    binning is biased by nearly half a pixel.

    """
    x, y = optimize_xy_separable(x, y)
    xr = x.ravel()
    yr = y.ravel()
    dx = float(xr[1] - xr[0])
    dy = float(yr[1] - yr[0])
    cover = func(x, y).astype(config.precision)
    # edge pixels: any disagreement within the 3x3 neighborhood
    N0, N1 = cover.shape
    p = np.pad(cover, 1, mode='edge')
    mn = None
    mx = None
    for i in range(3):
        for j in range(3):
            window = p[i:i+N0, j:j+N1]
            mn = window if mn is None else np.minimum(mn, window)
            mx = window if mx is None else np.maximum(mx, window)
    edge = mn != mx
    if not edge.any():
        return cover

    xg = np.broadcast_to(x, cover.shape)[edge]
    yg = np.broadcast_to(y, cover.shape)[edge]
    off = (np.arange(samples) + 0.5) / samples - 0.5
    xs = xg[:, None, None] + (off*dx)[None, :, None]
    ys = yg[:, None, None] + (off*dy)[None, None, :]
    vals = func(xs, ys).astype(config.precision)
    cover[edge] = vals.reshape(vals.shape[0], -1).mean(axis=1)
    return cover


def gaussian(sigma, x, y, center=(0, 0)):
    """Generate a gaussian mask with a given sigma.

    Parameters
    ----------
    sigma : float
        width parameter of the gaussian, expressed in the same units as x and y
    x : ndarray
        x spatial coordinates, 2D or 1D
    y : ndarray
        y spatial coordinates, 2D or 1D
    center : tuple of float
        center of the gaussian, (x,y)

    Returns
    -------
    ndarray
        mask with gaussian shape

    """
    s = sigma

    x, y = optimize_xy_separable(x, y)

    x0, y0 = center
    return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / s ** 2)


def rectangle_sdf(width, x, y, height=None, angle=0):
    """Signed distance to a rectangle, negative inside.

    Parameters
    ----------
    width : float
        half-width of the rectangle, x axis when angle=0
    x : ndarray
        x spatial coordinates, 2D or 1D
    y : ndarray
        y spatial coordinates, 2D or 1D
    height : float
        half-height of the rectangle, y axis when angle=0.
        If None, inherited from width to make a square
    angle : float
        angle, degrees

    Returns
    -------
    ndarray
        signed distance

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
    qx = np.abs(x) - width
    qy = np.abs(y) - height
    outside = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0))
    inside = np.minimum(np.maximum(qx, qy), 0)
    return outside + inside


def rectangle(width, x, y, height=None, angle=0):
    """Generate a rectangle, with the "width" axis aligned to 'x'.

    Parameters
    ----------
    width : float
        half-width of the rectangle, x axis when angle=0
    x : ndarray
        x spatial coordinates, 2D or 1D
    y : ndarray
        y spatial coordinates, 2D or 1D
    height : float
        half-height of the rectangle, y axis when angle=0.
        If None, inherited from width to make a square
    angle : float
        angle, degrees

    Returns
    -------
    ndarray
        binary mask, 1 inside the rectangle and 0 outside

    """
    return rectangle_sdf(width, x, y, height=height, angle=angle) <= 0


def rotated_ellipse_sdf(width_major, width_minor, x, y, major_axis_angle=0):
    r"""Signed distance to an ellipse centered at the origin, negative inside.

    Parameters
    ----------
    width_major : float
        semi-major axis length
    width_minor : float
        semi-minor axis length
    x : ndarray
        x spatial coordinates, 2D
    y : ndarray
        y spatial coordinates, 2D
    major_axis_angle : float
        angle of the major axis w.r.t. the x axis, degrees

    Returns
    -------
    ndarray
        signed distance

    Notes
    -----
    First-order (Taubin) estimate :math:`F / \lVert \nabla F \rVert` of the
    implicit equation; exact at the edge to within a pixel, approximate far
    from it.

    """
    if width_minor > width_major:
        raise ValueError('By definition, major axis must be larger than minor.')

    A = np.radians(-major_axis_angle)
    a, b = width_major, width_minor
    xr = x * np.cos(A) + y * np.sin(A)
    yr = x * np.sin(A) - y * np.cos(A)
    F = (xr/a)**2 + (yr/b)**2 - 1
    # grad F vanishes at the center; guard keeps the divide finite, the
    # ramp's clip saturates the interior anyway
    g = np.hypot(2*xr/(a*a), 2*yr/(b*b))
    return F / np.maximum(g, 1e-15)


def rotated_ellipse(width_major, width_minor, x, y, major_axis_angle=0):
    """Generate a mask for an ellipse, centered at the origin.

    Parameters
    ----------
    width_major : float
        semi-major axis length
    width_minor : float
        semi-minor axis length
    x : ndarray
        x spatial coordinates, 2D
    y : ndarray
        y spatial coordinates, 2D
    major_axis_angle : float
        angle of the major axis w.r.t. the x axis, degrees

    Returns
    -------
    ndarray
        binary mask, 1 inside the ellipse and 0 outside

    """
    return rotated_ellipse_sdf(width_major, width_minor, x, y, major_axis_angle=major_axis_angle) <= 0


def square(x, y):
    """Create a square mask.

    Parameters
    ----------
    x : ndarray
        x spatial coordinates, 2D
    y : ndarray
        y spatial coordinates, 2D

    Returns
    -------
    ndarray
        binary ndarray representation of the mask

    """
    return np.ones_like(x)


def circle_sdf(radius, r):
    """Signed distance to a circle, negative inside.

    Parameters
    ----------
    radius : float
        radius of the circle, same units as r
    r : ndarray
        radial coordinates, 2D or 1D

    Returns
    -------
    ndarray
        signed distance

    """
    return r - radius


def circle(radius, r):
    """Create a circular mask.

    Parameters
    ----------
    radius : float
        radius of the circle, same units as r
    r : ndarray
        2D array of radial coordinates

    Returns
    -------
    ndarray
        binary mask, 1 inside the radius and 0 outside

    """
    return circle_sdf(radius, r) <= 0


def annulus_sdf(rin, rout, r):
    """Signed distance to an annulus, negative inside.

    Parameters
    ----------
    rin : float
        inner radius
    rout : float
        outer radius
    r : ndarray
        radial coordinates, 2D or 1D

    Returns
    -------
    ndarray
        signed distance

    """
    center = (rin + rout) / 2
    halfwidth = (rout - rin) / 2
    return np.abs(r - center) - halfwidth


def annulus(rin, rout, r):
    """Create an annular mask.

    Parameters
    ----------
    rin : float
        inner radius
    rout : float
        outer radius
    r : ndarray
        2D array of radial coordinates

    Returns
    -------
    ndarray
        binary mask, 1 between the radii and 0 outside

    """
    return annulus_sdf(rin, rout, r) <= 0


def polygon_sdf(vertices, x, y):
    """Signed distance to a polygon, negative inside.

    Parameters
    ----------
    vertices : iterable
        Nx2 collection of (x, y) vertices, either winding, without a repeated
        closing vertex; need not be convex
    x : ndarray
        x spatial coordinates, 2D or 1D
    y : ndarray
        y spatial coordinates, 2D or 1D

    Returns
    -------
    ndarray
        signed distance

    """
    # segment windows can be empty slabs; element ops tolerate them, x[0, :] cannot
    if x.size and y.size:
        x, y = optimize_xy_separable(x, y)
    n = len(vertices)
    d2 = None
    inside = None
    for i in range(n):
        x0, y0 = (float(v) for v in vertices[i])
        x1, y1 = (float(v) for v in vertices[(i+1) % n])
        ex = x1 - x0
        ey = y1 - y0
        wx = x - x0
        wy = y - y0
        # distance to the edge segment
        t = (wx*ex + wy*ey) / (ex*ex + ey*ey)
        t = np.minimum(np.maximum(t, 0), 1)
        px = wx - t*ex
        py = wy - t*ey
        seg = px*px + py*py
        d2 = seg if d2 is None else np.minimum(d2, seg)
        # even-odd crossing parity for the sign; winding-agnostic
        straddle = (y0 > y) != (y1 > y)
        crosses = straddle & ((wx*ey < ex*wy) == (y1 > y0))
        inside = crosses if inside is None else inside ^ crosses
    d = np.sqrt(d2)
    return np.where(inside, -d, d)


def regular_polygon_sdf(sides, radius, x, y, center=(0, 0), rotation=0):
    """Signed distance to a regular polygon, negative inside.

    Parameters
    ----------
    sides : int
        number of sides to the polygon
    radius : float
        distance from the center to a vertex
    x : ndarray
        x spatial coordinates, 2D or 1D
    y : ndarray
        y spatial coordinates, 2D or 1D
    center : tuple of float
        center of the polygon, (x,y)
    rotation : float
        rotation of the polygon, degrees

    Returns
    -------
    ndarray
        signed distance

    """
    verts = _generate_vertices(sides, radius, center, rotation)
    return polygon_sdf(verts, x, y)


def regular_polygon(sides, radius, x, y, center=(0, 0), rotation=0):
    """Generate a regular polygon mask with the given number of sides.

    Parameters
    ----------
    sides : int
        number of sides to the polygon
    radius : float
        distance from the center to a vertex
    x : ndarray
        x spatial coordinates, 2D or 1D
    y : ndarray
        y spatial coordinates, 2D or 1D
    center : tuple of float
        center of the polygon, (x,y)
    rotation : float
        rotation of the polygon, degrees

    Returns
    -------
    ndarray
        binary mask, 1 inside the polygon and 0 outside

    """
    return regular_polygon_sdf(sides, radius, x, y, center=center, rotation=rotation) <= 0


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
    ndarray
        array with first column X points, second column Y points

    """
    angle = 2 * truenp.pi / sides
    rotation = truenp.radians(rotation)
    x0, y0 = center
    points = truenp.arange(sides, dtype=config.precision)
    x = radius * truenp.sin(points * angle + rotation) + x0
    y = radius * truenp.cos(points * angle + rotation) + y0
    return truenp.stack((x, y), axis=1)


def spider_sdf(vanes, width, x, y, rotation=0, center=(0, 0), rotation_is_rad=False):
    """Signed distance to the vanes of a spider, negative inside the vanes.

    Parameters
    ----------
    vanes : int
        number of spider vanes
    width : float
        full width of the vanes, same units as x and y
    x : ndarray
        x spatial coordinates, 2D or 1D
    y : ndarray
        y spatial coordinates, 2D or 1D
    rotation : float, optional
        rotational offset of the vanes, clockwise
    center : tuple of float
        point from which the vanes emanate, (x,y)
    rotation_is_rad : bool, optional
        if True, the rotation parameter is interpreted to be in radians

    Returns
    -------
    ndarray
        signed distance

    """
    half_width = width / 2
    x0, y0 = center
    x = x - x0
    y = y - y0
    if not rotation_is_rad:
        rotation = np.radians(rotation)

    step = 2 * np.pi / vanes
    d = None
    for multiple in range(vanes):
        angle = step * multiple - rotation
        c = np.cos(angle)
        s = np.sin(angle)
        along = x * c - y * s
        across = x * s + y * c
        # semi-infinite capsule: |across| along the vane, radial at the root
        vane = np.hypot(np.minimum(along, 0), across) - half_width
        d = vane if d is None else np.minimum(d, vane)
    return d


def spider(vanes, width, x, y, rotation=0, center=(0, 0), rotation_is_rad=False):
    """Generate the mask for the vanes of a spider.

    Parameters
    ----------
    vanes : int
        number of spider vanes
    width : float
        full width of the vanes, same units as x and y
    x : ndarray
        x spatial coordinates, 2D or 1D
    y : ndarray
        y spatial coordinates, 2D or 1D
    rotation : float, optional
        rotational offset of the vanes, clockwise
    center : tuple of float
        point from which the vanes emanate, (x,y)
    rotation_is_rad : bool, optional
        if True, the rotation parameter is interpreted to be in radians

    Returns
    -------
    ndarray
        binary mask, 1 inside the vanes and 0 outside

    """
    return spider_sdf(vanes, width, x, y, rotation=rotation, center=center,
                      rotation_is_rad=rotation_is_rad) <= 0


def offset_circle(radius, x, y, center):
    """Rasterize an offset circle.

    Parameters
    ----------
    radius : float
        radius of the circle, same units as x and y
    x : ndarray
        array of x coordinates
    y : ndarray
        array of y coordinates
    center : tuple
        tuple of (x, y) centers

    Returns
    -------
    ndarray
        ndarray containing the boolean mask

    """
    x, y = optimize_xy_separable(x, y)
    # no in-place ops, x, y are shared memory
    x = x - center[0]
    y = y - center[1]
    # not cart to polar; computing theta is waste work
    r = np.hypot(x, y)
    return circle(radius, r)


def rectangle_with_corner_fillets_sdf(width, height, cradius, x, y, center=(0, 0), rotation=0):
    """Signed distance to a rectangle with filleted corners, negative inside.

    Parameters
    ----------
    width : float
        half-width of the rectangle, same units as x and y
    height : float
        half-height of the rectangle, same units as x and y
    cradius : float
        radius of the corner fillets
    x : ndarray
        x coordinates
    y : ndarray
        y coordinates
    center : tuple of float
        (x,y) center of the rectangle
    rotation : float
        degrees of rotation **about coordinate grid center**

    Returns
    -------
    ndarray
        signed distance

    """
    if rotation != 0:
        r, t = cart_to_polar(x, y)
        t += np.radians(rotation)
        x, y = polar_to_cart(r, t)
    else:
        x, y = optimize_xy_separable(x, y)

    x = x - center[0]
    y = y - center[1]
    # rounded box: shrink the box by cradius, inflate the distance field back
    qx = np.abs(x) - (width - cradius)
    qy = np.abs(y) - (height - cradius)
    outside = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0))
    inside = np.minimum(np.maximum(qx, qy), 0)
    return outside + inside - cradius


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
    x : ndarray
        x coordinates
    y : ndarray
        y coordinates
    center : tuple of float
        (x,y) center of the rectangle
    rotation : float
        degrees of rotation **about coordinate grid center**

    Returns
    -------
    ndarray
        binary mask, 1 inside the "squircle" and 0 outside

    """
    return rectangle_with_corner_fillets_sdf(width, height, cradius, x, y,
                                             center=center, rotation=rotation) <= 0
