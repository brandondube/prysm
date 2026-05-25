"""Plotting functions for raytraces.

This is the one module in prysm.x.raytracing that uses import numpy as np
directly: matplotlib's renderer only accepts true numpy arrays, so we cannot
go through the swappable prysm.mathops backend here.  User-supplied arrays
(phist, shist, ...) may be cupy/torch tensors when alternate backends are in
use, so each public function calls array_to_true_numpy at entry.
"""

from collections.abc import Mapping, Sequence

from prysm.plotting import share_fig_ax
from prysm.mathops import array_to_true_numpy

from .spencer_and_murty import (
    RayTraceResult,
    transform_to_global_coords,
    transform_to_local_coords,
)
from .analysis import transverse_ray_aberration
from .surfaces import STYPE_REFLECT, STYPE_REFRACT
from ._meta import lensdata_wavelength

import numpy as np  # see module docstring; do not "fix" to mathops np


def _require_raytrace_result(result):
    if not isinstance(result, RayTraceResult):
        raise TypeError('expected a RayTraceResult')
    return result


def plot_ray_paths(result, *, x='z', y='y', lw=1, ls='-', c='r', alpha=1,
                   zorder=4, fig=None, ax=None):
    """Plot ray paths from a RayTraceResult.

    Parameters
    ----------
    result : RayTraceResult
        Trace result returned by spencer_and_murty.raytrace.
    x : str, {'x', 'y', 'z'}
        Which position to plot on the X axis, defaults to traditional ZY plot.
    y : str, {'x', 'y', 'z'}
        Which position to plot on the Y axis, defaults to traditional ZY plot.
    lw : float, optional
        linewidth
    ls : str, optional
        line style
    c : color
        anything matplotlib interprets as a color, strings, 3-tuples, 4-tuples, ...
    alpha : float
        opacity of the rays, 1=fully opaque, 0=fully transparent
    zorder : int
        stack order in the plot, higher z orders are on top of lower z orders
    fig : matplotlib.figure.Figure
        A figure object
    ax : matplotlib.axes.Axis
        An axis object

    Returns
    -------
    matplotlib.figure.Figure
        A figure object
    matplotlib.axes.Axis
        An axis object

    """
    result = _require_raytrace_result(result)
    fig, ax = share_fig_ax(fig, ax)

    ph = result.P
    ph = np.asarray(array_to_true_numpy(ph))
    xs = ph[..., 0]
    ys = ph[..., 1]
    zs = ph[..., 2]
    sieve = {
        'x': xs,
        'y': ys,
        'z': zs,
    }
    x = sieve[x.lower()]
    y = sieve[y.lower()]
    ax.plot(x, y, c=c, lw=lw, ls=ls, alpha=alpha, zorder=zorder)
    return fig, ax


def _axis_index(axis):
    return {'x': 0, 'y': 1, 'z': 2}[axis]


def _local_axis_coordinates_from_phist(phist, j, y, surf=None):
    p = phist[j + 1]
    if surf is not None:
        dirs = np.zeros_like(p)
        p, _ = transform_to_local_coords(p, surf.P, dirs, surf.R)
    axis = 1 if y == 'y' else 0
    return p[..., axis]


def _axis_extent_from_phist(phist, j, y, surf=None, center=0.0):
    coord = _local_axis_coordinates_from_phist(phist, j, y, surf=surf)
    coord = coord - center
    return max(abs(np.nanmin(coord)), abs(np.nanmax(coord)))


def _global_plot_coordinates(surf, sag, ploty, x, y):
    sag = np.asarray(sag, dtype=float)
    ploty = np.asarray(ploty, dtype=float)
    zeros = np.zeros_like(ploty)
    if y == 'y':
        local = np.stack([zeros, ploty, sag - surf.P[2]], axis=-1)
    else:
        local = np.stack([ploty, zeros, sag - surf.P[2]], axis=-1)
    dirs = np.zeros_like(local)
    rotation = None if surf.R is None else surf.R.T
    global_points, _ = transform_to_global_coords(
        local, surf.P, dirs, rotation,
    )
    return global_points[..., _axis_index(x)], global_points[..., _axis_index(y)]


def _surface_profile(surf, phist, j, points, y, radius=None, clear_radius=None,
                     center=0.0):
    if radius is None:
        if surf.bounding is None:
            p = phist[j + 1]
            dirs = np.zeros_like(p)
            p, _ = transform_to_local_coords(p, surf.P, dirs, surf.R)
            xx = p[..., 0]
            yy = p[..., 1]
            mask = []
            if y == 'y':
                ypt = np.linspace(yy.min(), yy.max(), points)
                ploty = ypt
                xpt = np.zeros_like(ypt)
            else:
                xpt = np.linspace(xx.min(), xx.max(), points)
                ploty = xpt
                ypt = np.zeros_like(xpt)
        else:
            bound = surf.bounding
            local = np.linspace(-bound['outer_radius'], bound['outer_radius'],
                                points)
            mask = abs(local) < bound.get('inner_radius', 0)
            ploty = center + local
            if y == 'y':
                ypt = ploty
                xpt = np.zeros_like(ploty)
            else:
                xpt = ploty
                ypt = np.zeros_like(ploty)
    else:
        local = np.linspace(-radius, radius, points)
        ploty = center + local
        bound = surf.bounding or {}
        mn = bound.get('inner_radius', 0)
        mask = abs(local) < mn
        if y == 'y':
            xpt = np.zeros_like(ploty)
            ypt = ploty
        else:
            xpt = ploty
            ypt = np.zeros_like(ploty)

    sag = surf.sag(xpt, ypt)
    sag = np.asarray(sag, dtype=float) + surf.P[2]
    edge_sag = sag.copy()
    mask = np.asarray(mask)
    if mask.size == 0:
        mask = np.zeros_like(ploty, dtype=bool)
    if clear_radius is not None:
        mask = mask | (abs(ploty - center) > clear_radius)
    sag[mask] = np.nan
    return sag, ploty, edge_sag


def _lens_edge_for(lens_edges, surface_index, pair_index):
    if lens_edges is None:
        return None

    if isinstance(lens_edges, Mapping):
        return lens_edges.get(surface_index, lens_edges.get(pair_index))

    if isinstance(lens_edges, Sequence) and not isinstance(lens_edges, (str, bytes)):
        if pair_index < len(lens_edges):
            return lens_edges[pair_index]
        return None

    return lens_edges


def _mirror_backing_mode(backing):
    if backing is None:
        return 'surface'
    if isinstance(backing, str):
        return backing.lower()
    return backing.get('mode', backing.get('kind', 'parallel')).lower()


def _mirror_clear_radius(backing):
    if backing is None:
        return None
    return backing.get('clear_radius')


def _mirror_backing_center(backing, phist, surface_index, surf, y):
    center = backing.get(f'center_{y}', backing.get(
        'aperture_center', backing.get('center', backing.get('centre', 0.0)),
    ))
    if isinstance(center, str):
        center = center.lower()
        if center in ('rays', 'ray', 'footprint', 'chief'):
            coord = _local_axis_coordinates_from_phist(
                phist, surface_index, y, surf=surf,
            )
            return float(np.nanmean(coord))
        raise ValueError(f'unknown mirror backing center {center!r}')
    return float(center)


def _plot_axis_reference_coordinate(surf, ploty, y, reference):
    if reference is None:
        reference = 'aperture'

    if not isinstance(reference, str):
        return float(reference)

    reference = reference.lower()
    if reference in ('center', 'centre'):
        return float(np.nanmean(ploty))
    if reference in ('local_vertex', 'section_vertex'):
        return 0.0
    if reference in ('parent', 'parent_vertex'):
        params = surf.params or {}
        key = 'dy' if y == 'y' else 'dx'
        return -float(params.get(key, 0.0))
    if reference in ('aperture', 'near_aperture', 'edge', 'near_edge'):
        parent = _plot_axis_reference_coordinate(surf, ploty, y, 'parent')
        return float(np.clip(parent, np.nanmin(ploty), np.nanmax(ploty)))

    raise ValueError(f'unknown mirror backing reference {reference!r}')


def _mirror_surface_outline_from_phist(surf, phist, surface_index, points, x, y,
                                       radius, clear_radius, center=0.0):
    sag, ploty, _ = _surface_profile(
        surf, phist, surface_index, points, y, radius=radius,
        clear_radius=clear_radius, center=center,
    )
    return _global_plot_coordinates(surf, sag, ploty, x, y)


def _mirror_substrate_outline_from_phist(surf, phist, surface_index, backing,
                                         points, x, y, radius):
    if not isinstance(backing, Mapping):
        backing = {'mode': backing}
    center = _mirror_backing_center(backing, phist, surface_index, surf, y)
    if radius is None:
        for key in ('od_radius', 'outer_radius', 'radius'):
            if key in backing:
                radius = backing[key]
                break
        else:
            if surf.bounding is not None and 'outer_radius' in surf.bounding:
                radius = surf.bounding['outer_radius']
            else:
                radius = _axis_extent_from_phist(
                    phist, surface_index, y, surf=surf, center=center,
                )
    clear_radius = _mirror_clear_radius(backing)
    sag, ploty, edge_sag = _surface_profile(
        surf, phist, surface_index, points, y, radius=radius,
        clear_radius=clear_radius, center=center,
    )
    mode = _mirror_backing_mode(backing)
    if mode in ('surface', 'optical', 'optical_surface', 'none'):
        return _global_plot_coordinates(surf, sag, ploty, x, y)

    if 'thickness' not in backing:
        raise ValueError('mirror substrate drawing requires a thickness')

    side = backing.get('side', backing.get('direction', 'auto'))
    if isinstance(side, str):
        side = side.lower()
        if side in ('auto',):
            departure = np.nanmean(edge_sag - edge_sag[len(edge_sag) // 2])
            side = -1.0 if departure > 0 else 1.0
        elif side in ('+', 'positive', 'pos', 'right', 'rear', 'back'):
            side = 1.0
        elif side in ('-', 'negative', 'neg', 'left', 'front'):
            side = -1.0
        else:
            raise ValueError(f'unknown mirror backing side {side!r}')
    else:
        side = float(side)
        if side == 0:
            raise ValueError('mirror backing side must be nonzero')
        side = np.sign(side)

    offset = side * float(backing['thickness'])
    if mode in ('parallel', 'equal', 'equal_thickness'):
        rear_sag = edge_sag + offset
    elif mode in ('flat_parent', 'parent', 'parent_vertex', 'vertex'):
        rear_sag = np.full_like(edge_sag, surf.P[2] + offset)
    elif mode in ('flat_aperture', 'aperture', 'near_uniform',
                  'uniform', 'hockey_puck'):
        reference = backing.get('reference', 'aperture')
        ref = _plot_axis_reference_coordinate(surf, ploty, y, reference)
        zeros = np.asarray([0.0])
        coord = np.asarray([ref])
        if y == 'y':
            xpt = zeros
            ypt = coord
            use_fy = True
        else:
            xpt = coord
            ypt = zeros
            use_fy = False

        z, n_hat = surf.sag_and_normal(xpt, ypt)
        Fx = -n_hat[..., 0] / n_hat[..., 2]
        Fy = -n_hat[..., 1] / n_hat[..., 2]
        slope = (Fy if use_fy else Fx)[0]
        rear_sag = surf.P[2] + z[0] + slope * (ploty - ref) + offset
    else:
        raise ValueError(f'unknown mirror backing mode {mode!r}')

    xx = [*sag, rear_sag[-1], *rear_sag[::-1], sag[0]]
    yy = [*ploty, ploty[-1], *ploty[::-1], ploty[0]]
    return _global_plot_coordinates(surf, xx, yy, x, y)


def mirror_surface_outline(surf, result, surface_index=0, *, points=100,
                           x='z', y='y', radius=None, clear_radius=None,
                           center=0.0):
    """Return X/Y arrays for drawing one mirror optical surface."""
    result = _require_raytrace_result(result)
    phist = np.asarray(array_to_true_numpy(result.P))
    x = x.lower()
    y = y.lower()
    if isinstance(center, str):
        center = _mirror_backing_center(
            {'center': center}, phist, surface_index, surf, y,
        )
    return _mirror_surface_outline_from_phist(
        surf, phist, surface_index, points, x, y, radius, clear_radius, center,
    )


def mirror_substrate_outline(surf, result, surface_index=0, *, backing,
                             points=100, x='z', y='y', radius=None):
    """Return X/Y arrays for drawing one mirror substrate outline.

    Parameters
    ----------
    surf : Surface
        the reflective surface to outline.
    result : RayTraceResult
        a trace whose ray positions bound the drawn extent when radius and
        the backing outer radius are both unset.
    surface_index : int, optional
        index of surf within the traced prescription, used to read the ray
        positions at that surface.
    backing : mapping or str
        substrate geometry.  As a string it names the mode; as a mapping it
        may carry the following keys:

        mode (or kind)
            'surface' / 'optical' / 'none' draws only the optical surface;
            'parallel' offsets the rear face parallel to the optical surface;
            'flat_parent' puts a flat rear face at the parent vertex z;
            'flat_aperture' puts a flat rear face tangent near a reference
            point, giving a near-uniform-thickness (hockey-puck) substrate.
        thickness
            axial substrate thickness; required for every mode except the
            optical-surface-only modes.
        side (or direction)
            which way the substrate is thick: 'auto' (default) picks the side
            that thickens away from the optical surface; 'front'/'rear' (or
            '+'/'-', or a signed number) force it.
        reference
            for flat_aperture only, the axis coordinate the flat is tangent
            to: 'aperture' (default, the parent vertex clipped to the drawn
            aperture), 'parent', 'center', or a numeric coordinate.
        od_radius (or outer_radius / radius)
            outer half-diameter of the substrate.  Falls back to
            surf.bounding['outer_radius'] and then to the traced extent.
        clear_radius
            inner clear-aperture radius punched out of the optical face.
        center (or aperture_center / center_x / center_y)
            local coordinate at the center of the drawn substrate aperture.
            The string 'rays' centers the aperture on the traced footprint in
            the plotted meridian.
    points : int, optional
        number of points sampled along the surface profile.
    y : str, optional
        meridian to draw in, 'y' (default) or 'x'.
    radius : float, optional
        outer half-diameter override; takes precedence over the backing
        radius keys when given.

    Returns
    -------
    xx, yy : list, list
        z (axial) and transverse coordinates of the closed substrate outline.

    """
    result = _require_raytrace_result(result)
    phist = np.asarray(array_to_true_numpy(result.P))
    x = x.lower()
    y = y.lower()
    return _mirror_substrate_outline_from_phist(
        surf, phist, surface_index, backing, points, x, y, radius,
    )


def plot_mirror_surface(surf, result, surface_index=0, *, points=100,
                        x='z', y='y', radius=None, clear_radius=None,
                        center=0.0,
                        lw=1, ls='-', c='k', alpha=1, zorder=3,
                        fig=None, ax=None):
    """Draw one mirror optical surface."""
    fig, ax = share_fig_ax(fig, ax)
    xx, yy = mirror_surface_outline(
        surf, result, surface_index, points=points, y=y, radius=radius,
        x=x, clear_radius=clear_radius, center=center,
    )
    ax.plot(xx, yy, c=c, lw=lw, ls=ls, alpha=alpha, zorder=zorder)
    return fig, ax


def plot_mirror_substrate(surf, result, surface_index=0, *, backing,
                          points=100, x='z', y='y', radius=None,
                          lw=1, ls='-', c='k', alpha=1, zorder=3,
                          fig=None, ax=None):
    """Draw one mirror with an optical surface and substrate.

    See mirror_substrate_outline for the backing schema.

    """
    fig, ax = share_fig_ax(fig, ax)
    xx, yy = mirror_substrate_outline(
        surf, result, surface_index, backing=backing, points=points,
        x=x, y=y, radius=radius,
    )
    ax.plot(xx, yy, c=c, lw=lw, ls=ls, alpha=alpha, zorder=zorder)
    return fig, ax


def lens_groups_from_surfaces(prescription, *, wvl=0.587,
                              ambient_index=1.0, index_atol=1e-9):
    """Group consecutive refracting surfaces into physical lens elements.

    Returns
    -------
    list of tuple
        Each tuple contains the prescription indices of one singlet, cemented
        doublet, triplet, or longer cemented lens group.

    """
    groups = []
    active = []

    for j, surf in enumerate(prescription):
        if surf.typ != STYPE_REFRACT:
            if active:
                raise ValueError(
                    'refracting lens group is interrupted before returning '
                    'to ambient material'
                )
            continue

        if not callable(surf.n):
            raise ValueError('refracting surfaces must define a callable material')
        n_post = surf.n(wvl)
        if np.isscalar(n_post):
            n_post = float(n_post)
        else:
            n_post = np.asarray(array_to_true_numpy(n_post))
            if n_post.size != 1:
                raise ValueError('material evaluation must produce a scalar index')
            n_post = float(n_post.reshape(-1)[0])

        active.append(j)
        if np.isclose(n_post, ambient_index, rtol=0, atol=index_atol):
            if len(active) < 2:
                raise ValueError('lens groups require at least two refracting surfaces')
            groups.append(tuple(active))
            active = []

    if active:
        raise ValueError(
            'cant draw a prescription that terminates before returning to '
            'ambient material'
        )

    return groups


def _surface_clear_radius(lens_edge, which):
    if lens_edge is None:
        return None
    specific_key = f'clear_radius_{which}'
    if specific_key in lens_edge:
        return lens_edge[specific_key]
    return lens_edge.get('clear_radius')


def _append_wall_point(xs, ys, x, y):
    if xs and xs[-1] == x and ys[-1] == y:
        return
    xs.append(x)
    ys.append(y)


def _wall_path(x0, x1, outer_y, features, side, endpoint_names):
    xs = [x0]
    ys = [outer_y]
    direction = np.sign(x1 - x0) or 1
    current = x0
    spans = []

    for feature in features:
        target = feature.get('side', 'both').lower()
        if target not in ('both', side):
            continue
        kind = feature.get('kind', feature.get('type', 'square')).lower()
        if kind in ('square_cut', 'flat', 'chamfer'):
            if 'z_start' not in feature or 'z_end' not in feature:
                raise ValueError(f'{kind} lens edge features require z_start and z_end')
            start = feature['z_start']
            end = feature['z_end']
        elif kind == 'seat':
            if 'width' not in feature:
                raise ValueError('seat lens edge features require width')
            face = feature.get('face', endpoint_names[0]).lower()
            width = feature['width']
            if face == endpoint_names[0]:
                start = x0
                end = x0 + np.sign(x1 - x0) * width
            elif face == endpoint_names[1]:
                start = x1 - np.sign(x1 - x0) * width
                end = x1
            else:
                raise ValueError('seat face must name one wall endpoint')
        elif kind == 'square':
            continue
        else:
            raise ValueError(f'unknown lens edge feature kind {kind!r}')

        if direction < 0:
            start, end = end, start
        lo = min(x0, x1)
        hi = max(x0, x1)
        start = min(max(start, lo), hi)
        end = min(max(end, lo), hi)
        if start == end:
            continue
        depth = feature.get('depth', feature.get('amount', 0))
        kind = feature.get('kind', feature.get('type', 'square')).lower()
        spans.append((start, end, depth, kind))

    spans.sort(key=lambda item: direction * item[0])

    for start, end, depth, kind in spans:
        inset = outer_y + depth if outer_y < 0 else outer_y - depth
        if direction * (start - current) > 0:
            _append_wall_point(xs, ys, start, outer_y)
        if kind == 'chamfer':
            _append_wall_point(xs, ys, end, inset)
        else:
            _append_wall_point(xs, ys, start, inset)
            _append_wall_point(xs, ys, end, inset)
        _append_wall_point(xs, ys, end, outer_y)
        current = end

    _append_wall_point(xs, ys, x1, outer_y)
    return xs, ys


def plot_optics(prescription, result, *, wvl=None, ambient_index=1.0,
                index_atol=1e-9, mirror_backing=None, points=100,
                lw=1, ls='-', c='k', alpha=1, zorder=3,
                x='z', y='y', fig=None, ax=None, lens_edges=None):
    """Draw the optics of a prescription.

    Parameters
    ----------
    prescription : iterable of Surface
        a prescription for an optical layout
    result : RayTraceResult
        Trace result returned by spencer_and_murty.raytrace.
    wvl : float, optional
        Wavelength in microns used to evaluate post-surface material indices.
        Defaults from the LensData reference wavelength, else 0.6328.
    ambient_index : float, optional
        Refractive index that closes a physical lens group.
    index_atol : float, optional
        Absolute tolerance for comparing material index to ambient_index.
    mirror_backing : mapping, sequence, str, optional
        Mechanical substrate geometry for reflective surfaces.  A mapping is
        keyed by surface index or by mirror index.  A sequence is aligned to
        reflective surfaces.  Each entry may define mode, thickness,
        od_radius, clear_radius, side, and reference.  Supported modes are
        surface, parallel, flat_parent, and flat_aperture.  The flat_aperture
        mode uses a flat rear cut tangent near the aperture point closest to
        the parent vertex by default, producing a near-uniform-thickness OAP
        substrate.
    points : int, optional
        the number of points used in making the curve for the surface
    lw : float, optional
        linewidth
    ls : str, optional
        line style
    c : color, optional
        anything matplotlib interprets as a color, strings, 3-tuples, 4-tuples, ...
    alpha : float, optional
        opacity of the rays, 1=fully opaque, 0=fully transparent
    zorder : int
        stack order in the plot, higher z orders are on top of lower z orders
    x : str, {'x', 'y', 'z'}
        which position to plot on the X axis, defaults to traditional ZY plot
    y : str, {'x', 'y', 'z'}
        which position to plot on the X axis, defaults to traditional ZY plot
    fig : matplotlib.figure.Figure
        A figure object
    ax : matplotlib.axes.Axis
        An axis object
    lens_edges : mapping or sequence, optional
        Mechanical edge geometry for refracting lens groups.  A mapping is
        keyed by the first surface index of a group, or by group index.  A
        sequence is aligned to lens groups.  Each entry may define
        od_radius, clear_radius, clear_radius_front, clear_radius_rear, and
        a features list.  Supported feature kinds are square, square_cut,
        seat, chamfer, and flat.

    Returns
    -------
    matplotlib.figure.Figure
        A figure object
    matplotlib.axes.Axis
        An axis object

    """
    wvl = lensdata_wavelength(prescription, wvl)
    x = x.lower()
    y = y.lower()
    fig, ax = share_fig_ax(fig, ax)
    result = _require_raytrace_result(result)
    phist = result.P
    phist = np.asarray(array_to_true_numpy(phist))

    lens_groups = lens_groups_from_surfaces(
        prescription, wvl=wvl, ambient_index=ambient_index,
        index_atol=index_atol,
    )
    groups_by_start = {group[0]: (group_index, group)
                       for group_index, group in enumerate(lens_groups)}

    j = 0
    mirror_index = 0
    jj = len(prescription)
    while j < jj:
        surf = prescription[j]
        if surf.typ == STYPE_REFLECT:
            if isinstance(mirror_backing, Mapping):
                config_keys = {
                    'mode', 'kind', 'thickness', 'od_radius', 'outer_radius',
                    'radius', 'clear_radius', 'side', 'direction', 'reference',
                    'center', 'centre', 'aperture_center', 'center_x',
                    'center_y',
                }
                if config_keys & set(mirror_backing):
                    backing = mirror_backing
                else:
                    backing = _lens_edge_for(mirror_backing, j, mirror_index)
            else:
                backing = _lens_edge_for(mirror_backing, j, mirror_index)

            if backing is None:
                xx, yy = _mirror_surface_outline_from_phist(
                    surf, phist, j, points, x, y, None, None,
                )
            else:
                if not isinstance(backing, Mapping):
                    backing = {'mode': backing}
                if _mirror_backing_mode(backing) in ('surface', 'optical',
                                                     'optical_surface', 'none'):
                    center = _mirror_backing_center(
                        backing, phist, j, surf, y,
                    )
                    xx, yy = _mirror_surface_outline_from_phist(
                        surf, phist, j, points, x, y, None,
                        _mirror_clear_radius(backing), center,
                    )
                else:
                    xx, yy = _mirror_substrate_outline_from_phist(
                        surf, phist, j, backing, points, x, y, None,
                    )

            ax.plot(xx, yy, c=c, lw=lw, ls=ls, alpha=alpha, zorder=zorder)
            mirror_index += 1
            j += 1
        elif surf.typ == STYPE_REFRACT:
            group_index, group = groups_by_start[j]
            lens_edge = _lens_edge_for(lens_edges, j, group_index)

            od_radius = None
            if lens_edge is not None:
                for key in ('od_radius', 'outer_radius', 'radius'):
                    if key in lens_edge:
                        od_radius = lens_edge[key]
                        break
            if od_radius is None:
                radii = []
                for surface_index in group:
                    group_surface = prescription[surface_index]
                    if (group_surface.bounding is not None and
                            'outer_radius' in group_surface.bounding):
                        radii.append(group_surface.bounding['outer_radius'])
                if radii:
                    od_radius = max(radii)
                else:
                    od_radius = max(
                        _axis_extent_from_phist(phist, surface_index, y)
                        for surface_index in group
                    )

            profiles = []
            group_size = len(group)
            for surface_number, surface_index in enumerate(group):
                if surface_number == 0:
                    clear_radius = _surface_clear_radius(lens_edge, 'front')
                elif surface_number == group_size - 1:
                    clear_radius = _surface_clear_radius(lens_edge, 'rear')
                else:
                    clear_radius = _surface_clear_radius(
                        lens_edge, f'surface_{surface_number}',
                    )
                profiles.append(_surface_profile(
                    prescription[surface_index], phist, surface_index, points,
                    y, radius=od_radius, clear_radius=clear_radius,
                ))

            sag1, ploty1, edge_sag1 = profiles[0]
            sag2, ploty2, edge_sag2 = profiles[-1]
            features = []
            if lens_edge is not None:
                features = lens_edge.get('features', [])
                if isinstance(features, Mapping):
                    features = [features]

            top_x, top_y = _wall_path(
                edge_sag1[-1], edge_sag2[-1], od_radius, features, 'upper',
                ('front', 'rear'),
            )
            bottom_x, bottom_y = _wall_path(
                edge_sag2[0], edge_sag1[0], -od_radius, features, 'lower',
                ('rear', 'front'),
            )

            xx = [*sag1, *top_x[1:], *sag2[::-1], *bottom_x[1:]]
            yy = [*ploty1, *top_y[1:], *ploty2[::-1], *bottom_y[1:]]

            for sag, ploty, _ in profiles[1:-1]:
                xx.extend([np.nan, *sag])
                yy.extend([np.nan, *ploty])

            ax.plot(xx, yy, c=c, lw=lw, ls=ls, alpha=alpha, zorder=zorder)
            j = group[-1] + 1
        else:
            j += 1

    return fig, ax


def plot_transverse_ray_aberration(phist, lw=1, ls='-', c='r', alpha=1,
                                   zorder=4, axis='y', fig=None, ax=None,
                                   chief_index=None, status=None):
    """Plot the transverse ray aberration for a single ray fan.

    Parameters
    ----------
    phist : RayTraceResult, list, or ndarray
        Trace result or position history from spencer_and_murty.raytrace.
    lw : float, optional
        linewidth
    ls : str, optional
        line style
    c : color
        anything matplotlib interprets as a color, strings, 3-tuples, 4-tuples, ...
    alpha : float
        opacity of the rays, 1=fully opaque, 0=fully transparent
    zorder : int
        stack order in the plot, higher z orders are on top of lower z orders
    axis : str, {'x', 'y'}
        which ray position to plot, x or y
    fig : matplotlib.figure.Figure
        A figure object
    ax : matplotlib.axes.Axis
        An axis object
    chief_index : int, optional
        row index of the chief ray; default N//2.
    status : ndarray, optional
        per-ray status from raytrace.  Invalid rays are excluded when
        provided.

    Returns
    -------
    matplotlib.figure.Figure
        A figure object
    matplotlib.axes.Axis
        An axis object

    """
    fig, ax = share_fig_ax(fig, ax)

    if isinstance(phist, RayTraceResult):
        result = phist
        phist = result.P
        if status is None:
            status = result.status

    ph = np.asarray(array_to_true_numpy(phist))
    if status is not None:
        status = np.asarray(array_to_true_numpy(status))

    input_rays, output_rays = transverse_ray_aberration(
        ph, axis=axis.lower(), chief_index=chief_index, status=status,
    )
    input_rays = np.asarray(array_to_true_numpy(input_rays))
    output_rays = np.asarray(array_to_true_numpy(output_rays))
    ax.plot(input_rays, output_rays, c=c, lw=lw, ls=ls, alpha=alpha,
            zorder=zorder)
    return fig, ax


def plot_wave_aberration_fan(coord, opd, *, wavelength=None, units='waves',
                             detrend=True, lw=1, ls='-', c='r', alpha=1,
                             zorder=4, axis='y', label=None, fig=None,
                             ax=None):
    """Plot OPD for a single wave-aberration fan.

    Parameters
    ----------
    coord : array_like
        normalized pupil coordinate.
    opd : array_like
        optical path difference in microns.
    wavelength : float, optional
        wavelength in microns. Required for units='waves'.
    units : str, {'waves', 'nm'}
        vertical axis units.
    detrend : bool, optional
        if True, subtract a first-degree fit from the plotted OPD.
    lw : float, optional
        linewidth
    ls : str, optional
        line style
    c : color
        anything matplotlib interprets as a color, strings, 3-tuples, 4-tuples, ...
    alpha : float
        opacity of the rays, 1=fully opaque, 0=fully transparent
    zorder : int
        stack order in the plot, higher z orders are on top of lower z orders
    axis : str, {'x', 'y'}
        pupil axis label.
    label : str, optional
        legend label for this fan.
    fig : matplotlib.figure.Figure
        A figure object
    ax : matplotlib.axes.Axis
        An axis object

    Returns
    -------
    matplotlib.figure.Figure
        A figure object
    matplotlib.axes.Axis
        An axis object

    """
    fig, ax = share_fig_ax(fig, ax)
    coord = np.asarray(array_to_true_numpy(coord), dtype=float)
    opd = np.asarray(array_to_true_numpy(opd), dtype=float)

    units = units.lower()
    if units in ('wave', 'waves'):
        if wavelength is None:
            raise ValueError('wavelength is required when units="waves"')
        opd = opd / float(wavelength)
        ylabel = 'OPD [waves]'
    elif units in ('nm', 'nanometer', 'nanometers'):
        opd = opd * 1e3
        ylabel = 'OPD [nm]'
    else:
        raise ValueError("units must be 'waves' or 'nm'")

    if detrend:
        valid = np.isfinite(coord) & np.isfinite(opd)
        if np.count_nonzero(valid) >= 2:
            slope, intercept = np.polyfit(coord[valid], opd[valid], 1)
            opd = opd - (slope * coord + intercept)

    ax.plot(coord, opd, c=c, lw=lw, ls=ls, alpha=alpha, zorder=zorder,
            label=label)
    ax.set_xlabel(f'normalized pupil {axis}')
    ax.set_ylabel(ylabel)
    return fig, ax


def plot_spot_diagram(phist, marker='+', c='k', alpha=1, zorder=4, s=None, fig=None, ax=None):
    """Plot a spot diagram from a ray trace.

    Parameters
    ----------
    phist : list or ndarray
        the first return from spencer_and_murty.raytrace,
        iterable of arrays of length 3 (X,Y,Z)
    marker : str, optional
        marker style
    c : color
        anything matplotlib interprets as a color, strings, 3-tuples, 4-tuples, ...
    alpha : float
        opacity of the rays, 1=fully opaque, 0=fully transparent
    zorder : int
        stack order in the plot, higher z orders are on top of lower z orders
    s : float
        marker size or variable used for marker size
    axis : str, {'x', 'y'}
        which ray position to plot, x or y
    fig : matplotlib.figure.Figure
        A figure object
    ax : matplotlib.axes.Axis
        An axis object

    Returns
    -------
    matplotlib.figure.Figure
        A figure object
    matplotlib.axes.Axis
        An axis object

    """
    fig, ax = share_fig_ax(fig, ax)
    phist = np.asarray(array_to_true_numpy(phist))
    x = phist[-1, ..., 0]
    y = phist[-1, ..., 1]
    ax.scatter(x, y, c=c, s=s, marker=marker, alpha=alpha, zorder=zorder)
    return fig, ax
