"""Tools for working with segmented systems."""
import inspect
from collections import namedtuple

import numpy as truenp

from .mathops import np
from .geometry import regular_polygon
from .coordinates import cart_to_polar
from .polynomials import sum_of_2d_modes

FLAT_TO_FLAT_TO_VERTEX_TO_VERTEX = 2 / truenp.sqrt(3)


Hex = namedtuple('Hex', ['q', 'r', 's'])


def add_hex(h1, h2):
    """Add two hex coordinates together."""
    q = h1.q + h2.q
    r = h1.r + h2.r
    s = h1.s + h2.s
    return Hex(q, r, s)


def sub_hex(h1, h2):
    """Subtract two hex coordinates."""
    q = h1.q - h2.q
    r = h1.r - h2.r
    s = h1.s - h2.s
    return Hex(q, r, s)


def mul_hex(h1, h2):
    """Multiply two hex coordinates."""
    q = h1.q * h2.q
    r = h1.r * h2.r
    s = h1.s * h2.s
    return Hex(q, r, s)


# as given
hex_dirs = [
    Hex(1, 0, -1), Hex(1, -1, 0), Hex(0, -1, 1),
    Hex(-1, 0, 1), Hex(-1, 1, 0), Hex(0, 1, -1)
]


def hex_dir(i):
    """Hex direction associated with a given integer, wrapped at 6."""
    return hex_dirs[i % 6]  # wrap dirs at 6 (there are only 6)


def hex_neighbor(h, direction):
    """Neighboring hex in a given direction."""
    return add_hex(h, hex_dir(direction))


def hex_to_xy(h, radius, rot=90):
    """Convert hexagon coordinate to (x,y), if all hexagons have a given radius and rotation."""
    if rot == 90:
        x = 3/2 * h.q
        y = truenp.sqrt(3)/2 * h.q + truenp.sqrt(3) * h.r
    else:
        x = truenp.sqrt(3) * h.q + truenp.sqrt(3)/2 * h.r
        y = 3/2 * h.r
    return x*radius, y*radius


def scale_hex(h, k):
    """Scale a hex coordinate by some constant factor."""
    return Hex(h.q * k, h.r * k, h.s * k)


def hex_ring(radius):
    """Compute all hex coordinates in a given ring."""
    start = Hex(-radius, radius, 0)
    tile = start
    results = []
    # there are 6*r hexes per ring (the i)
    # the j ensures that we reset the direction we travel every time we reach a
    # 'corner' of the ring.
    for i in range(6):
        for j in range(radius):
            results.append(tile)
            tile = hex_neighbor(tile, i)

    # rotate one so that the first element is 'north'
    for _ in range(radius):
        results.append(results.pop(0))  # roll < radius > elements so that the first element is "north"

    return results


def _local_window(cy, cx, center, dx, samples_per_seg, x, y):
    offset_x = cx + int(center[0]/dx) - samples_per_seg
    offset_y = cy + int(center[1]/dx) - samples_per_seg

    upper_x = offset_x + (2*samples_per_seg)
    upper_y = offset_y + (2*samples_per_seg)

    # clamp the offsets
    if offset_x < 0:
        offset_x = 0
    if offset_x > x.shape[1]:
        offset_x = x.shape[1]
    if offset_y < 0:
        offset_y = 0
    if offset_y > y.shape[0]:
        offset_y = y.shape[0]
    if upper_x < 0:
        upper_x = 0
    if upper_x > x.shape[1]:
        upper_x = x.shape[1]
    if upper_y < 0:
        upper_y = 0
    if upper_y > y.shape[0]:
        upper_y = y.shape[0]

    return slice(offset_y, upper_y), slice(offset_x, upper_x)


class CompositeHexagonalAperture:
    """An aperture composed of several hexagonal segments."""

    def __init__(self, x, y, rings, segment_diameter, segment_separation, segment_angle=90, exclude=()):
        """Create a new CompositeHexagonalAperture.

        Note that __init__ is relatively computationally expensive and hides a lot of work.

        Parameters
        ----------
        x : numpy.ndarray
            array of x sample positions, of shape (m, n)
        y : numpy.ndarray
            array of y sample positions, of shape (m, n)
        rings : int
            number of rings in the structure
        segment_diameter : float
            flat-to-flat diameter of each segment, same units as x
        segment_separation : float
            edge-to-nearest-edge distance between segments, same units as x
        segment_angle : float, optional, {0, 90}
            rotation angle of each segment
        exclude : sequence of int
            which segment numbers to exclude.
            defaults to all segments included.
            The 0th segment is the center of the array.
            Other segments begin from the "up" orientation and count clockwise.

        """
        (
            self.vtov,
            self.all_centers,
            self.windows,
            self.local_coords,
            self.local_masks,
            self.segment_ids,
            self.amp
         ) = _composite_hexagonal_aperture(rings, segment_diameter, segment_separation,
                                           x, y, segment_angle, exclude)

        self.x = x
        self.y = y
        self.segmentd_diameter = segment_diameter
        self.segment_separation = segment_separation
        self.segment_angle = segment_angle
        self.exclude = exclude

    def prepare_opd_bases(self, basis_func, orders, basis_func_kwargs=None, normalization_radius=None):
        """Prepare the polynomial bases for per-segment phase errors.

        Parameters
        ----------
        basis_func : callable
            a function with signature basis_func(orders, [x, y or r, t], **kwargs)
            for example, zernike_nm_sequence from prysm.polyomials fits the bill
        orders : iterable
            sequence of polynomial orders or indices.
            for example, zernike_nm_sequence may be combined with a monoindexing
            function as e.g. orders=[noll_to_nm(j) for j in range(3,12)]
        basis_func_kwargs : dict
            any keyword arguments to pass to basis_func.  The spatial coordinates
            will already be passed based on inspection of the function signature
            and should not be attempted to be included here
        normalization_radius : float
            the normaliation radius to use to convert local surface coordinates
            to normalized coordinates for an orthogonal polynomial.
            if None, defaults to the half segment vertex to vertex distance,
            v to v is 2/sqrt(3) times the segment diameter given in the constructor
            if basis_func does not take arguments (r, t), the radius is assumed
            to be equal in X and Y

        """
        # if the norm radius isn't given, assume it's V to V
        # the conversion to a length two tuple is so that we know it's length
        # two for either the r,t or x,y branch below
        if normalization_radius is None:
            normalization_radius = self.vtov/2

        if not isinstance(normalization_radius, (tuple, list)):
            normalization_radius = (normalization_radius, normalization_radius)

        if basis_func_kwargs is None:
            basis_func_kwargs = {}

        # first thing to do, does basis_func take radial kwargs?
        sig = inspect.signature(basis_func)
        params = sig.parameters
        gridcache = {}
        polycache = {}
        grids = []
        bases = []
        if 'r' in params and 't' in params:
            nr = normalization_radius[0]
            # there is high duplicity in most cases, e.g.
            # a JWST-esque aperture painted on a 6.5m diameter
            # array has 5 unique local grids, which with 18 segments
            # means removing over 66% of the work and memory usage
            # here we have three collections,
            # gridcache is the unique grids
            # grids is the grid for each segment, which may contain many
            # duplicate elements but simplifies access for downstream routines,
            # and polynomials which is the polynomial base for each segment,
            # with the same duplicate note as the grids
            for x, y in self.local_coords:
                corner = float(x[0, 0])  # for Cupy support
                key = (corner, *x.shape)
                if key not in gridcache:
                    r, t = cart_to_polar(x, y)
                    r /= nr
                    basis = list(basis_func(orders, r=r, t=t, **basis_func_kwargs))
                    basis = np.asarray(basis)
                    gridcache[key] = r, t
                    polycache[key] = basis
                else:
                    r, t = gridcache[key]
                    basis = polycache[key]

                grids.append((r, t))
                bases.append(basis)
        else:
            # assume x, y are the kwargs
            for x, y in self.local_coords:
                corner = float(x[0, 0])  # for Cupy support
                key = (corner, *x.shape)
                if key not in gridcache:
                    xx = x / normalization_radius[0]
                    yy = y / normalization_radius[1]
                    basis = list(basis_func(orders, x=xx, y=yy, **basis_func_kwargs))
                    basis = np.asarray(basis)
                    gridcache[key] = xx, yy
                    polycache[key] = basis
                else:
                    xx, yy = gridcache[key]
                    basis = polycache[key]

                grids.append((xx, yy))
                bases.append(basis)

        self.opd_bases = bases
        self.opd_grids = grids
        return grids, bases

    def compose_opd(self, coefs, out=None):
        """Compose per-segment optical path errors using the basis from prepare_opd_bases

        Parameters
        ----------
        coefs : iterable
            an iterable of coefficients for each segment present, i.e. excluding
            those in the exclude list from the constructor
            if an array, must be of shape (len(self.segment_ids), len(orders))
            where orders comes from the proceeding call to prepare_opd_bases
        out : numpy.ndarray
            array to insert OPD into, allocated if None

        Returns
        -------
        numpy.ndarray
            OPD map of real datatype

        """
        if out is None:
            out = np.zeros_like(self.x)
        for win, mask, base, c in zip(self.windows, self.local_masks, self.opd_bases, coefs):
            tile = sum_of_2d_modes(base, c)
            tile *= mask
            out[win] += tile

        return out


def _composite_hexagonal_aperture(rings, segment_diameter, segment_separation, x, y, segment_angle=90, exclude=(0,)):
    if segment_angle not in {0, 90}:
        raise ValueError('can only synthesize composite apertures with hexagons along a cartesian axis')

    segment_vtov = segment_diameter * FLAT_TO_FLAT_TO_VERTEX_TO_VERTEX
    rseg = segment_vtov / 2

    # center segment
    dx = x[0, 1] - x[0, 0]
    samples_per_seg = rseg / dx
    # add 1, must avoid error in the case that non-center segments
    # fall on a different subpixel and have different rounding
    # use rseg since it is what we are directly interested in
    samples_per_seg = int(samples_per_seg+1)

    # compute the center segment over the entire x, y array
    # so that mask covers the entirety of the x/y extent
    # this may look out of place/unused, but the window is used when creating
    # the 'windows' list
    cx = int(np.ceil(x.shape[1]/2))
    cy = int(np.ceil(y.shape[0]/2))
    center_segment_window = _local_window(cy, cx, (0, 0), dx, samples_per_seg, x, y)

    mask = np.zeros(x.shape, dtype=np.bool)

    segment_id = 0
    xx = x[center_segment_window]
    yy = y[center_segment_window]
    center_mask = regular_polygon(6, rseg, xx, yy, center=(0, 0), rotation=segment_angle)
    if 0 not in exclude:
        mask[center_segment_window] |= center_mask
        local_masks = [center_mask]
        segment_ids = [0]
        all_centers = [(0., 0.)]
        windows = [center_segment_window]
        local_coords = [(xx, yy)]
    else:
        local_masks = []
        local_coords = []
        segment_ids = []
        all_centers = []
        windows = []
    for i in range(1, rings+1):
        hexes = hex_ring(i)
        centers = [hex_to_xy(h, rseg+(segment_separation/2), rot=segment_angle) for h in hexes]
        ids = np.arange(segment_id+1, segment_id+1+len(centers), dtype=int)
        id_mask = ~np.isin(ids, exclude, assume_unique=True)
        valid_ids = ids[id_mask]
        centers = truenp.array(centers)
        centers = centers[id_mask]
        all_centers += centers.tolist()
        for segment_id, center in zip(valid_ids, centers):
            # short circuit: if we do not wish to include a segment,
            # do no further work on it
            if segment_id in exclude:
                continue

            segment_ids.append(segment_id)

            local_window = _local_window(cy, cx, center, dx, samples_per_seg, x, y)
            windows.append(local_window)

            xx = x[local_window]
            yy = y[local_window]

            local_coords.append((xx-center[0], yy-center[1]))

            local_mask = regular_polygon(6, rseg, xx, yy, center=center, rotation=segment_angle)
            local_masks.append(local_mask)
            mask[local_window] |= local_mask

        segment_id = ids[-1]

    return segment_vtov, all_centers, windows, local_coords, local_masks, segment_ids, mask
