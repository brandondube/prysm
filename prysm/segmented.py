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
    add_hex(start, scale_hex(hex_dir(0), radius))
    tile = start
    results = []
    # there are 6*r hexes per ring (the i)
    # the j ensures that we reset the direction we travel every time we reach a
    # 'corner' of the ring.
    for i in range(6*radius):
        for j in range(radius):
            results.append(tile)
            tile = hex_neighbor(tile, i)

    return results


# The 18 hexagonal segments are arranged in a large hexagon, with the central
# segment removed to allow the light to reach the instruments. Each segment is
# 1.32 m, measured flat to flat. Beginning with a geometric area of 1.50 m2;
# after cryogenic shrinking and edge removal, the average projected segment area
# is 1.46 m2. With obscuration by the secondary mirror support system of no more
# than 0.86 m2, the total polished area equals 25.37 m2, and vignetting by the
# pupil stops is minimized so that it meets the >25 m2 requirement for the total
# unobscured collecting area for the telescope. The outer diameter, measured
# along the mirror, point to point on the larger hexagon, but flat to flat on
# the individual segments, is 5 times the 1.32 m segment size, or 6.6 m
# (see figure). The minimum diameter from inside point to inside point is 5.50 m.
# The maximum diameter from outside point to outside point is 6.64 m. The average
# distance between the segments is about 7 mm, a distance that is adjustable
# on-orbit. The 25 m2 is equivalent to a filled circle of diameter 5.64 m. The
# telescope has an effective f/# of 20 and an effective focal length of 131.4 m,
# corresponding to an effective diameter of 6.57 m. The secondary mirror is circular,
# 0.74 m in diameter and has a convex aspheric prescription. There are three
# different primary mirror segment prescriptions, with 6 flight segments and 1
# spare segment of each prescription. The telescope is a three-mirror anastigmat,
# so it has primary, secondary and tertiary mirrors, a fine steering mirror, and
# each instrument has one or more pick-off mirrors.
# jwst = 1.32m segments

def composite_hexagonal_aperture(rings, segment_diameter, segment_separation, x, y, segment_angle=90):
    if segment_angle not in {0, 90}:
        raise ValueError('can only synthesize composite apertures with hexagons along a cartesian axis')

    flat_to_flat_to_vertex_vertex = 2 / truenp.sqrt(3)
    segment_vtov = segment_diameter * flat_to_flat_to_vertex_vertex
    rseg = segment_vtov / 2
    mask = regular_polygon(6, rseg, x, y, center=(0, 0), rotation=segment_angle)

    all_centers = [(0, 0)]
    for i in range(1, rings+1):
        hexes = hex_ring(i)
        centers = [hex_to_xy(h, rseg+segment_separation, rot=segment_angle) for h in hexes]
        all_centers += centers
        for center in centers:
            lcl_mask = regular_polygon(6, rseg, x, y, center=center, rotation=segment_angle)
            mask |= lcl_mask
