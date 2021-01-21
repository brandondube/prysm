from collections import namedtuple

import numpy as truenp

from .geometry import regular_polygon
from .mathops import np

Hex = namedtuple('Hex', ['q', 'r', 's'])

axial_to_px_0 = truenp.array([
    [truenp.sqrt(3), truenp.sqrt(3)/2],
    [0,          3/2],
])

px_to_axial_0 = truenp.linalg.inv(axial_to_px_0)


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


def composite_hexagonal_aperture(rings, segment_diameter, segment_separation, x, y, segment_angle=90, exclude=(0,)):
    if segment_angle not in {0, 90}:
        raise ValueError('can only synthesize composite apertures with hexagons along a cartesian axis')

    flat_to_flat_to_vertex_vertex = 2 / truenp.sqrt(3)
    segment_vtov = segment_diameter * flat_to_flat_to_vertex_vertex
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
    if 0 in exclude:
        mask = np.logical_xor(mask, mask)

    all_centers = [(0, 0)]
    segment_id = 0
    segment_ids = [segment_id]
    windows = [center_segment_window]
    xx = x[center_segment_window]
    yy = y[center_segment_window]
    local_coords = [
        (xx, yy)
    ]
    center_mask = regular_polygon(6, rseg, xx, yy, center=(0, 0), rotation=segment_angle)
    local_masks = [center_mask]
    for i in range(1, rings+1):
        hexes = hex_ring(i)
        centers = [hex_to_xy(h, rseg+segment_separation, rot=segment_angle) for h in hexes]
        all_centers += centers
        for center in centers:
            segment_id += 1
            segment_ids.append(segment_id)

            local_window = _local_window(cy, cx, center, dx, samples_per_seg, x, y)
            windows.append(local_window)

            xx = x[local_window]
            yy = y[local_window]

            local_coords.append((xx-center[0], yy-center[1]))

            local_mask = regular_polygon(6, rseg, xx, yy, center=center, rotation=segment_angle)
            local_masks.append(local_mask)
            if segment_id in exclude:
                continue
            mask[local_window] |= local_mask

    return segment_vtov, all_centers, windows, local_coords, local_masks, segment_ids, mask
