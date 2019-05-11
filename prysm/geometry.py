"""Functions used to generate various geometrical constructs.
"""

from scipy.spatial import Delaunay

from .conf import config
from .mathops import engine as e
from .coordinates import make_rho_phi_grid


class MaskCache(object):
    """Cache for geometric masks."""
    def __init__(self):
        """Create a new cache instance."""
        self.masks = {}

    def get_mask(self, shape, samples, radius=1):
        """Get a mask with the given number of samples and shape.

        Parameters
        ----------
        shape : `str`
            string of a regular n-sided polygon, e.g. 'square', 'hexagon'.
        samples : `int`
            number of samples, mask is (samples,samples) in shape
        radius : `float`, optional
            normalized radius of the mask.  radius=1 will fill the x, y extent

        Returns
        -------
        `numpy.ndarray`
            ndarray; ones inside the shape, zeros outside

        """
        try:
            mask = self.masks[(shape, samples, radius)]
        except KeyError:
            mask = shapes[shape](samples=samples, radius=radius)
            self.masks[(shape, samples, radius)] = mask.copy()

        return mask

    def __call__(self, shape, samples, radius=1):
        """Get a mask with the given number of samples and shape.

        Parameters
        ----------
        shape : `str`
            string of a regular n-sided polygon, e.g. 'square', 'hexagon'.
        samples : `int`
            number of samples, mask is (samples,samples) in shape
        radius : `float`, optional
            normalized radius of the mask.  radius=1 will fill the x, y extent

        Returns
        -------
        `numpy.ndarray`
            ndarray; ones inside the shape, zeros outside

        """
        return self.get_mask(shape=shape, samples=samples, radius=radius)

    def clear(self, *args):
        """Empty the cache."""
        self.masks = {}


def gaussian(sigma=0.5, samples=128):
    """Generate a gaussian mask with a given sigma.

    Parameters
    ----------
    sigma : `float`
        width parameter of the gaussian, expressed in samples of the output array

    samples : `int`
        number of samples in square array

    Returns
    -------
    `numpy.ndarray`
        mask with gaussian shape

    """
    s = sigma

    x = e.arange(0, samples, 1, dtype=config.precision)
    y = x[:, e.newaxis]

    # // is floor division in python
    x0 = y0 = samples // 2
    return e.exp(-4 * e.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / (s * samples) ** 2)


def rotated_ellipse(width_major, width_minor, major_axis_angle=0, samples=128):
    """Generate a binary mask for an ellipse, centered at the origin.

    The major axis will notionally extend to the limits of the array, but this
    will not be the case for rotated cases.

    Parameters
    ----------
    width_major : `float`
        width of the ellipse in its major axis
    width_minor : `float`
        width of the ellipse in its minor axis
    major_axis_angle : `float`
        angle of the major axis w.r.t. the x axis, degrees
    samples : `int`
        number of samples

    Returns
    -------
    `numpy.ndarray`
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
    if width_minor > width_major:
        raise ValueError('By definition, major axis must be larger than minor.')

    arr = e.ones((samples, samples))
    lim = width_major
    x, y = e.linspace(-lim, lim, samples), e.linspace(-lim, lim, samples)
    xv, yv = e.meshgrid(x, y)
    A = e.radians(-major_axis_angle)
    a, b = width_major, width_minor
    major_axis_term = ((xv * e.cos(A) + yv * e.sin(A)) ** 2) / a ** 2
    minor_axis_term = ((xv * e.sin(A) - yv * e.cos(A)) ** 2) / b ** 2
    arr[major_axis_term + minor_axis_term > 1] = 0
    return arr


def triangle(samples=128, radius=1):
    """Create a square mask.

    Parameters
    ----------
    samples : `int`, optional
        number of samples in the square output array
    radius : `float`, optional
        radius of the shape in the square output array.  radius=1 will fill the
        x

    Returns
    -------
    `numpy.ndarray`
        binary ndarray representation of the mask

    """
    return regular_polygon(3, samples=samples, radius=radius)


def square(samples=128, radius=1):
    """Create a square mask.

    Parameters
    ----------
    samples : `int`, optional
        number of samples in the square output array
    radius : `float`, optional
        radius of the shape in the square output array.  radius=1 will fill the
        x

    Returns
    -------
    `numpy.ndarray`
        binary ndarray representation of the mask

    """
    return e.ones((samples, samples), dtype=bool)


def pentagon(samples=128, radius=1):
    """Create a pentagon mask.

    Parameters
    ----------
    samples : `int`, optional
        number of samples in the square output array
    radius : `float`, optional
        radius of the shape in the square output array.  radius=1 will fill the
        x

    Returns
    -------
    `numpy.ndarray`
        binary ndarray representation of the mask

    """
    return regular_polygon(5, samples=samples, radius=radius)


def hexagon(samples=128, radius=1):
    """Create a hexagon mask.

    Parameters
    ----------
    samples : `int`, optional
        number of samples in the square output array
    radius : `float`, optional
        radius of the shape in the square output array.  radius=1 will fill the
        x

    Returns
    -------
    `numpy.ndarray`
        binary ndarray representation of the mask

    """
    return regular_polygon(6, samples=samples, radius=radius)


def heptagon(samples=128, radius=1):
    """Create a heptagon mask.

    Parameters
    ----------
    samples : `int`, optional
        number of samples in the square output array
    radius : `float`, optional
        radius of the shape in the square output array.  radius=1 will fill the
        x

    Returns
    -------
    `numpy.ndarray`
        binary ndarray representation of the mask

    """
    return regular_polygon(7, samples=samples, radius=radius)


def octagon(samples=128, radius=1):
    """Create a octagon mask.

    Parameters
    ----------
    samples : `int`, optional
        number of samples in the square output array
    radius : `float`, optional
        radius of the shape in the square output array.  radius=1 will fill the
        x

    Returns
    -------
    `numpy.ndarray`
        binary ndarray representation of the mask

    """
    return regular_polygon(8, samples=samples, radius=radius)


def nonagon(samples=128, radius=1):
    """Create a nonagon mask.

    Parameters
    ----------
    samples : `int`, optional
        number of samples in the square output array
    radius : `float`, optional
        radius of the shape in the square output array.  radius=1 will fill the
        x

    Returns
    -------
    `numpy.ndarray`
        binary ndarray representation of the mask

    """
    return regular_polygon(9, samples=samples, radius=radius)


def decagon(samples=128, radius=1):
    """Create a decagon mask.

    Parameters
    ----------
    samples : `int`, optional
        number of samples in the square output array
    radius : `float`, optional
        radius of the shape in the square output array.  radius=1 will fill the
        x

    Returns
    -------
    `numpy.ndarray`
        binary ndarray representation of the mask

    """
    return regular_polygon(10, samples=samples, radius=radius)


def hendecagon(samples=128, radius=1):
    """Create a hendecagon mask.

    Parameters
    ----------
    samples : `int`, optional
        number of samples in the square output array
    radius : `float`, optional
        radius of the shape in the square output array.  radius=1 will fill the
        x

    Returns
    -------
    `numpy.ndarray`
        binary ndarray representation of the mask

    """
    return regular_polygon(11, samples=samples, radius=radius)


def dodecagon(samples=128, radius=1):
    """Create a dodecagon mask.

    Parameters
    ----------
    samples : `int`, optional
        number of samples in the square output array
    radius : `float`, optional
        radius of the shape in the square output array.  radius=1 will fill the
        x

    Returns
    -------
    `numpy.ndarray`
        binary ndarray representation of the mask

    """
    return regular_polygon(12, samples=samples, radius=radius)


def trisdecagon(samples=128, radius=1):
    """Create a trisdecagonal mask.

    Parameters
    ----------
    samples : `int`, optional
        number of samples in the square output array
    radius : `float`, optional
        radius of the shape in the square output array.  radius=1 will fill the
        x

    Returns
    -------
    `numpy.ndarray`
        binary ndarray representation of the mask

    """
    return regular_polygon(13, samples=samples, radius=radius)


def truecircle(samples=128, radius=1):
    """Create a "true" circular mask with anti-aliasing.

    Parameters
    ----------
    samples : `int`, optional
        number of samples in the square output array
    radius : `float`, optional
        radius of the shape in the square output array.  radius=1 will fill the
        x

    Returns
    -------
    `numpy.ndarray`
        nonbinary ndarray representation of the mask

    Notes
    -----
    Based on a more general algorithm by Jim Fienup

    """
    if radius == 0:
        return e.zeros((samples, samples), dtype=config.precision)
    else:
        rho, phi = make_rho_phi_grid(samples, samples)
        one_pixel = 2 / samples
        radius_plus = radius + (one_pixel / 2)
        intermediate = (radius_plus - rho) * (samples / 2)
        return e.minimum(e.maximum(intermediate, 0), 1)


def circle(samples=128, radius=1):
    """Create a circular mask.

    Parameters
    ----------
    samples : `int`, optional
        number of samples in the square output array
    radius : `float`, optional
        radius of the shape in the square output array.  radius=1 will fill the
        x

    Returns
    -------
    `numpy.ndarray`
        binary ndarray representation of the mask

    """
    if radius == 0:
        return e.zeros((samples, samples), dtype=config.precision)
    else:
        rho, phi = make_rho_phi_grid(samples, samples)
        mask = e.ones(rho.shape, dtype=config.precision)
        mask[rho > radius] = 0
        return mask


def inverted_circle(samples=128, radius=1):
    """Create an inverted circular mask (obscuration).

    Parameters
    ----------
    samples : `int`, optional
        number of samples in the square output array
    radius : `float`, optional
        radius of the shape in the square output array.  radius=1 will fill the
        x

    Returns
    -------
    `numpy.ndarray`
        binary ndarray representation of the mask

    """
    if radius == 0:
        return e.zeros((samples, samples), dtype=config.precision)
    else:
        rho, phi = make_rho_phi_grid(samples, samples)
        mask = e.ones(rho.shape, dtype=config.precision)
        mask[rho < radius] = 0
        return mask


def regular_polygon(sides, samples, radius=1):
    """Generate a regular polygon mask with the given number of sides and samples in the mask array.

    Parameters
    ----------
    sides : `int`
        number of sides to the polygon
    samples : `int`
        number of samples in the output polygon
    radius : `float`, optional
        radius of the regular polygon.  For R=1, will fill the x and y extent

    Returns
    -------
    `numpy.ndarray`
        mask for regular polygon with radius equal to the array radius

    """
    verts = generate_vertices(sides, int(e.floor((samples // 2) * radius)))
    verts[:, 0] += samples // 2  # shift y to center
    verts[:, 1] += samples // 2  # shift x to center
    return generate_mask(verts, samples).astype(config.precision)


def generate_mask(vertices, num_samples=128):
    """Create a filled convex polygon mask based on the given vertices.

    Parameters
    ----------
    vertices : `iterable`
        ensemble of vertice (x,y) coordinates, in array units
    num_samples : `int`
        number of points in the output array along each dimension

    Returns
    -------
    `numpy.ndarray`
        polygon mask

    """
    vertices = e.asarray(vertices)
    unit = e.arange(num_samples)
    xxyy = e.stack(e.meshgrid(unit, unit), axis=2)

    # use delaunay to fill from the vertices and produce a mask
    triangles = Delaunay(vertices, qhull_options='QJ Qf')
    mask = ~(triangles.find_simplex(xxyy) < 0)
    return mask


def generate_vertices(sides, radius=1):
    """Generate a list of vertices for a convex regular polygon with the given number of sides and radius.

    Parameters
    ----------
    sides : `int`
        number of sides to the polygon
    radius : `float`
        radius of the polygon

    Returns
    -------
    `numpy.ndarray`
        array with first column X points, second column Y points

    """
    angle = 2 * e.pi / sides
    pts = []
    for point in range(sides):
        x = radius * e.sin(point * angle)
        y = radius * e.cos(point * angle)
        pts.append((int(x), int(y)))

    return e.asarray(pts)


shapes = {
    'invertedcircle': inverted_circle,
    'truecircle': truecircle,
    'circle': circle,
    'triangle': triangle,
    'square': square,
    'pentagon': pentagon,
    'hexagon': hexagon,
    'heptagon': heptagon,
    'octagon': octagon,
    'nonagon': nonagon,
    'decagon': decagon,
    'hendecagon': hendecagon,
    'dodecagon': dodecagon,
    'trisdecagon': trisdecagon,
}

mcache = MaskCache()
config.chbackend_observers.append(mcache.clear)
