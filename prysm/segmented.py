"""Tools for working with segmented systems."""
import math
import inspect
from multiprocessing.sharedctypes import Value
import numbers
from collections import namedtuple

import numpy as truenp

from .mathops import np
from .geometry import regular_polygon, circle, spider
from .coordinates import cart_to_polar, polar_to_cart
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
    if isinstance(samples_per_seg, int):
        samples_per_seg = (samples_per_seg, samples_per_seg)

    offset_x = cx + int(center[0]/dx) - samples_per_seg[0]
    offset_y = cy + int(center[1]/dx) - samples_per_seg[1]

    upper_x = offset_x + (2*samples_per_seg[0])
    upper_y = offset_y + (2*samples_per_seg[1])

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
        self.segment_diameter = segment_diameter
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

    mask = np.zeros(x.shape, dtype=bool)

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


class CompositeKeystoneAperture:
    """Composite apertures with keystone shaped segments."""
    def __init__(self, x, y, center_circle_diameter, segment_gap,
                 rings, ring_radius, segments_per_ring, rotation_per_ring=None):
        """Create a new CompositeKeystoneAperture.

        Parameters
        ----------
        x : numpy.ndarray
            array of x sample positions, of shape (m, n)
        y : numpy.ndarray
            array of y sample positions, of shape (m, n)
        center_circle_diameter : float
            diameter of the circular supersegment at the center of the aperture
        segment_gap : float
            segment gap, same units as x and y; has the sense of the full gap,
            not the radius of the gap
        rings : int
            number of rings in the aperture
        ring_radius : float or Iterable
            the radius of each ring, i.e. (OD-ID).  Can be an iterable for
            variable radius of each ring
        segments_per_ring : int or Iterable
            number of segments in a given ring.  Can be an iterable for variable
            segment count in each ring
        rotation_per_ring : float or Iterable, optional
            the rotation of each ring.  Rotation is used to avoid alignment
            of segment gaps into radial lines, when fractal segment divisions
            are used.

            For example, two rings with [8, 16] segments per ring will produce
            a gap in the second ring aligned to the gap in the previous ring.

            None for this argument will shift/rotate/phase the second ring
            by (360/16)=22.5 degrees so that the gaps do not align

        """
        (
            block,
            self.all_centers,
            self.windows,
            self.local_coords,
            self.local_masks,
            self.segment_ids,
            self.amp
         ) = _composite_keystone_aperture(center_circle_diameter, segment_gap,
                                          x, y, rings, ring_radius,
                                          segments_per_ring, rotation_per_ring)
        (
            self.center_xx,
            self.center_yy,
            self.center_rr,
            self.center_tt,
            self.center_mask,
            self.center_win
        ) = block

        self.x = x
        self.y = y
        self.center_circle_diameter = center_circle_diameter
        self.segment_gap = segment_gap
        self.rings = rings
        self.ring_radius = ring_radius
        self.segments_per_ring = segments_per_ring
        self.rotation_per_ring = rotation_per_ring

    def prepare_opd_bases(self, center_basis, center_orders,
                          segment_basis, segment_orders,
                          center_basis_kwargs=None, segment_basis_kwargs=None):
        if center_basis_kwargs is None:
            center_basis_kwargs = {}

        if segment_basis_kwargs is None:
            segment_basis_kwargs = {}

        bases = []
        grids = []

        # take care of the center first
        sig = inspect.signature(center_basis)
        params = sig.params
        if 'r' in params and 't' in params:
            nr = self.center_circle_diameter/2
            rr = self.center_rr
            tt = self.center_tt
            basis = list(center_basis(center_orders, r=r, t=t, **center_basis_kwargs))
            basis = np.asarray(basis)
            grids.append((rr, tt))
            bases.append(basis)

        # now do each segment
        sig = inspect.sinature(segment_basis)
        params = sig.params
        gridcache = {}
        polycache = {}
        if 'r' in params and 't' in params:
            # some grids may end up being identical to others, so we don't
            # do the work twice
            for x, y in self.local_coords:
                corner = float(x[0, 0])  # for Cupy support
                key = (corner, *x.shape)
                if key not in gridcache:
                    xext = float(x[0, -1] - x[0, 0])
                    yext = float(y[-1, 0] - y[0, 0])
                    nr = min(xext, yext) / 2  # /2; diameter -> radius
                    r, t = cart_to_polar(x, y)
                    r /= nr
                    basis = list(segment_basis(segment_orders, r=r, t=t, **segment_basis_kwargs))
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
                    xext = float(x[0, -1] - x[0, 0])
                    yext = float(y[-1, 0] - y[0, 0])
                    xx = x / (xext/2)
                    yy = y / (yext/2)
                    basis = list(segment_basis(segment_orders, r=r, t=t, **segment_basis_kwargs))
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

    def compose_opd(self, center_coefs, segment_coefs, out=None):
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
        # add center mask to returns in class constructor
        # the center basis expansion is done
        # now just need to add segment expansions

        if out is None:
            out = np.zeros_like(self.x)
        tile = sum_of_2d_modes(self.bases[0], center_coefs)
        out[self.center_win] += (tile*self.center_mask)

        for win, mask, base, c in zip(self.windows, self.local_masks, self.opd_bases, segment_coefs):
            tile = sum_of_2d_modes(base, c)
            tile *= mask
            out[win] += tile

        return out


def _composite_keystone_aperture(center_circle_diameter, segment_gap, x, y,
                                 rings, ring_radius, segments_per_ring,
                                 rotation_per_ring=None):
    if isinstance(rotation_per_ring, numbers.Number) or rotation_per_ring is None:
        rotation_per_ring = [rotation_per_ring] * rings

    if isinstance(ring_radius, numbers.Number):
        ring_radius = [ring_radius] * rings

    if isinstance(segments_per_ring, numbers.Number):
        segments_per_ring = [segments_per_ring] * rings

    if isinstance(segment_gap, numbers.Number):
        segment_gap = [segment_gap] * rings

    center_radius = center_circle_diameter / 2

    local_masks = []
    local_coords = []
    segment_ids = []
    all_centers = []
    windows = []
    primary_mask = np.zeros(x.shape, dtype=bool)

    dx = x[0, 1] - x[0, 0]
    r, t = cart_to_polar(x, y)
    # t in [-pi,pi]
    # everything is (much) easier in [0,2pi]
    # numbers positive means all cases are low<t & hi>t
    # t += np.pi
    ccx = int(np.ceil(x.shape[1]/2))
    ccy = int(np.ceil(y.shape[0]/2))

    center_diameter_samples = math.ceil(center_circle_diameter / dx)
    win = _local_window(ccy, ccx, (0, 0), dx, center_diameter_samples, x, y)
    center_xx = x[win]
    center_yy = y[win]
    center_rr = r[win]
    center_tt = t[win]
    center_mask = circle(center_radius, center_rr)
    primary_mask[win] = center_mask
    outer_radius = center_radius

    segment_id = 0
    iterable = (segments_per_ring, ring_radius, segment_gap, rotation_per_ring)
    # for ring in range(len(segments_per_ring)):
    for (nsegments, local_radius, gap, rotation) in zip(*iterable):
        inner_radius = outer_radius + gap
        outer_radius = inner_radius + local_radius

        arc_per_seg = 360 / nsegments
        arc_rad = np.radians(arc_per_seg)

        if rotation is None:
            rotation = arc_per_seg

        segment_angles = np.arange(nsegments, dtype=float) * arc_per_seg + rotation
        segment_angles = np.radians(segment_angles)

        for angle in segment_angles:
            # find the four corners; c = corner
            lo = angle
            hi = angle+arc_rad
            print('before mod, lo, hi', lo, hi)
            while hi > 2*np.pi:
                hi = hi - 2*np.pi
            while lo > 2*np.pi:
                lo = lo - 2*np.pi

            swapped = False
            if hi < lo:
                swapped = True
                lo, hi = hi, lo
            print('after mod, lo, hi', lo, hi)
            # print('-'*80)
            # print(lo, hi)
            # print('-'*80)

            c1 = (inner_radius, lo)
            c2 = (inner_radius, hi)
            c3 = (outer_radius, lo)
            c4 = (outer_radius, hi)
            arr = np.array([c1, c2, c3, c4])
            rr = arr[:, 0]
            tt = arr[:, 1]
            xx, yy = polar_to_cart(rr, tt)
            minx = min(xx)
            maxx = max(xx)
            miny = min(yy)
            maxy = max(yy)
            print(f'x: {minx:.2f} to {maxx:.2f}')
            print(f'y: {miny:.2f} to {maxy:.2f}')
            rangex = maxx - minx
            rangey = maxy - miny
            samples = math.ceil(max((rangex/dx, rangey/dx)))
            cx = minx + rangex/2
            cy = miny + rangey/2

            # make the arc
            center = (cx, cy)
            window = _local_window(ccy, ccx, center, dx, samples, x, y)
            rr = r[window]
            tt = t[window]
            print('t min max', tt.min(), tt.max())
            print('-'*80)
            from matplotlib import pyplot as plt
            # plt.figure()
            # im = plt.imshow(tt)
            # plt.colorbar(im)
            inner_include = circle(inner_radius, rr)
            outer_exclude = circle(outer_radius, rr)
            # if not swapped:
            #     ang_mask = (tt > lo) & (tt < hi)
            # else:
            #     ang_mask = (tt < lo) & (tt > hi)
            ang_mask = (tt > lo) & (tt < hi)
            plt.figure()
            plt.imshow(tt>lo)
            plt.figure()
            plt.imshow(tt<hi)
            plt.figure()

            # print(lo, hi, tt.min(), tt.max())
            # if (lo < np.pi) and (hi <= np.pi):
            #     # basic case
            #     print(lo, hi, 'basic')
            #     ang_mask = (tt > lo) & (tt < hi)
            # elif (lo < np.pi) and (hi > np.pi):
            #     # wrapped around pi
            #     print(lo, hi, 'single wrap')
            #     ang_mask = (tt > lo) | (tt < (hi - 2*np.pi))
            #     # ang_mask |= tt < (hi - 2*np.pi)
            # elif (lo > np.pi) and (hi > np.pi):
            #     # need to phase wwrap
            #     print(lo, hi, 'double wrap')
            #     part_1 = tt > (lo - 2*np.pi)
            #     part_2 = tt < (hi - 2*np.pi)
            #     ang_mask = part_1 & part_2
            # else:
            #     print('STUPID')
            #     print(lo, hi)
            #     raise ValueError('what the fuck')

            # print(rr.shape, tt.shape, inner_include.shape, outer_exclude.shape, ang_mask.shape)
            # print(lo, hi)
            mask = (inner_include ^ outer_exclude) & ang_mask
            # mask = ang_mask
            # print(ang_mask.max(), ang_mask.min())
            primary_mask[window] |= mask

            # below here is the spider, which we don't care about beyond the
            # mask, and we need to store some stuff
            segment_ids.append(segment_id)
            local_masks.append(mask)
            local_coords.append((xx-cx, yy-cy))
            all_centers.append(center)
            windows.append(window)
            segment_id += 1

            # now make the spider between this arc and the next
            # want to cut out a local window at the seam
            # so use c2, c4, which are the "right hand" corners
            minx = min(xx[1], xx[3])
            maxx = max(xx[1], xx[3])
            miny = min(yy[1], yy[3])
            maxy = max(yy[1], yy[3])
            rangex = maxx - minx
            rangey = maxy - miny
            samples = tuple(math.ceil(v) for v in (rangex/dx + gap/dx, rangey/dx + gap/dx))
            cx = minx + rangex/2
            cy = miny + rangey/2

            window = _local_window(ccy, ccx, (cx, cy), dx, samples, x, y)
            xx = x[window]
            yy = y[window]
            rr = r[window]
            # TODO: this can be optimized with fewer bitwise inversions?
            rot = hi
            while rot > (2*np.pi):
                rot = rot - 2*np.pi
            spid = ~spider(1, gap, xx, yy, rotation=rot, rotation_is_rad=True)

            low_cut = ~circle(inner_radius, rr)
            hi_cut = circle(outer_radius, rr)
            spid &= low_cut
            spid &= hi_cut

            primary_mask[window] &= ~spid

    return (center_xx, center_yy, center_rr, center_tt, center_mask, win), \
        all_centers, windows, local_coords, local_masks, segment_ids, primary_mask
