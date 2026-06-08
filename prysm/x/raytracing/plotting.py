"""Plotting functions for raytraces."""

import warnings

from collections.abc import Mapping, Sequence

from prysm.plotting import share_fig_ax
from prysm.mathops import array_to_true_numpy

from .spencer_and_murty import (
    RayTraceResult,
    transform_to_global_coords,
    transform_to_local_coords,
    valid_mask,
)
from .analysis import (
    transverse_ray_aberration,
    chromatic_focal_shift,
    field_curvature,
    distortion,
    _field_curvature_labels,
)
from .surfaces import STYPE_REFLECT, STYPE_REFRACT
from ._meta import system_wavelength

import numpy as np  # see module docstring; do not "fix" to mathops np


def _require_raytrace_result(result):
    if not isinstance(result, RayTraceResult):
        raise TypeError('expected a RayTraceResult')
    return result


def _to_np(x, dtype=None):
    """Bring a possibly-cupy/torch array back to true numpy for matplotlib."""
    return np.asarray(array_to_true_numpy(x), dtype=dtype)


def _axis_pair(coord, y):
    """Return (xpt, ypt) with coord on the drawn meridian and zero off it."""
    zero = np.zeros_like(coord)
    return (zero, coord) if y == 'y' else (coord, zero)


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

    ph = _to_np(result.P)
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
    return p[..., _axis_index(y)]


def _axis_extent_from_phist(phist, j, y, surf=None, center=0.0):
    coord = _local_axis_coordinates_from_phist(phist, j, y, surf=surf)
    coord = coord - center
    return max(abs(np.nanmin(coord)), abs(np.nanmax(coord)))


def _global_plot_coordinates(surf, sag, ploty, x, y):
    sag = np.asarray(sag, dtype=float)
    ploty = np.asarray(ploty, dtype=float)
    xpart, ypart = _axis_pair(ploty, y)
    local = np.stack([xpart, ypart, sag - surf.P[2]], axis=-1)
    dirs = np.zeros_like(local)
    rotation = None if surf.R is None else surf.R.T
    global_points, _ = transform_to_global_coords(
        local, surf.P, dirs, rotation,
    )
    return global_points[..., _axis_index(x)], global_points[..., _axis_index(y)]


def _surface_profile(surf, phist, j, points, y, radius=None, clear_radius=None,
                     center=0.0, max_radius=None):
    if radius is None:
        if surf.bounding is None:
            p = phist[j + 1]
            dirs = np.zeros_like(p)
            p, _ = transform_to_local_coords(p, surf.P, dirs, surf.R)
            src = p[..., 1] if y == 'y' else p[..., 0]
            mask = []
            ploty = np.linspace(src.min(), src.max(), points)
            xpt, ypt = _axis_pair(ploty, y)
        else:
            bound = surf.bounding
            local = np.linspace(-bound['outer_radius'], bound['outer_radius'],
                                points)
            mask = abs(local) < bound.get('inner_radius', 0)
            ploty = center + local
            xpt, ypt = _axis_pair(ploty, y)
    else:
        local = np.linspace(-radius, radius, points)
        ploty = center + local
        bound = surf.bounding or {}
        mn = bound.get('inner_radius', 0)
        mask = abs(local) < mn
        # The plotted points span the full element radius, but the sag is only
        # evaluated out to max_radius (where this surface still exists).  Beyond
        # that the evaluation coordinate is held at the rim, so the sag is flat
        # at the edge value and the profile becomes a constant-z (radial)
        # segment from the surface edge out to the element OD -- the vertical
        # bridge that closes the outline on a surface too steep to reach the OD.
        if max_radius is not None:
            eval_local = np.clip(local, -max_radius, max_radius)
        else:
            eval_local = local
        eval_coord = center + eval_local
        xpt, ypt = _axis_pair(eval_coord, y)

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


def _surface_drawable_radius(surf, radius, axis, center=0.0, samples=512):
    """Largest radius (<= radius) where this surface's sag is still defined.

    A steeply curved surface (small |R|) has no sag beyond its equator — a
    sphere is undefined past r = |R| — so evaluating its profile out at the
    element OD yields NaN and the lens outline breaks open.  Probe the surface
    along the drawn axis and return the largest radius where the sag remains
    finite; the caller draws the optical surface out to there and bridges the
    remaining annulus to the OD with a radial segment.  Returns radius
    unchanged when the surface reaches the OD.

    """
    probe = np.linspace(0.0, radius, samples)
    coord = center + probe
    xpt, ypt = _axis_pair(coord, axis)
    # the probe deliberately reaches past a steep surface's equator, where the
    # sag is undefined; that NaN is the signal we are looking for
    with np.errstate(invalid='ignore'):
        sag = np.asarray(surf.sag(xpt, ypt), dtype=float)
    bad = ~np.isfinite(sag)
    if not bad.any():
        return radius
    first = int(np.argmax(bad))
    return float(probe[first - 1]) if first > 0 else 0.0


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
        coord = np.asarray([ref])
        xpt, ypt = _axis_pair(coord, y)
        use_fy = y == 'y'

        z, n_hat = surf.sag_and_normal(xpt, ypt)
        Fx = -n_hat[..., 0] / n_hat[..., 2]
        Fy = -n_hat[..., 1] / n_hat[..., 2]
        slope = (Fy if use_fy else Fx)[0]
        rear_sag = surf.P[2] + z[0] + slope * (ploty - ref) + offset
    else:
        raise ValueError(f'unknown mirror backing mode {mode!r}')

    # a bored substrate (central hole, set by bounding inner_radius or an
    # explicit backing inner_radius) is open through both faces, so its
    # meridional section is two disjoint closed loops -- one per side of the
    # bore -- not a single outline bridged across a solid back
    inner = 0.0
    if surf.bounding is not None:
        inner = float(surf.bounding.get('inner_radius', 0.0))
    inner = float(backing.get('inner_radius', inner))
    if inner > 0.0:
        ploty_arr = np.asarray(ploty, dtype=float)
        front = np.asarray(sag, dtype=float)
        rear = np.asarray(rear_sag, dtype=float).copy()
        rear[np.abs(ploty_arr - center) < inner] = np.nan
        xx, yy = [], []
        for sel in (ploty_arr >= center + inner, ploty_arr <= center - inner):
            good = sel & np.isfinite(front) & np.isfinite(rear)
            if not good.any():
                continue
            fz, rz, py = front[good], rear[good], ploty_arr[good]
            # front face out, rear face back, closed by the bore and rim walls
            xx += [*fz, *rz[::-1], fz[0], np.nan]
            yy += [*py, *py[::-1], py[0], np.nan]
        return _global_plot_coordinates(surf, xx, yy, x, y)

    xx = [*sag, rear_sag[-1], *rear_sag[::-1], sag[0]]
    yy = [*ploty, ploty[-1], *ploty[::-1], ploty[0]]
    return _global_plot_coordinates(surf, xx, yy, x, y)


def mirror_surface_outline(surf, result, surface_index=0, *, points=100,
                           x='z', y='y', radius=None, clear_radius=None,
                           center=0.0):
    """Return X/Y arrays for drawing one mirror optical surface."""
    result = _require_raytrace_result(result)
    phist = _to_np(result.P)
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
        substrate geometry.
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
    phist = _to_np(result.P)
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

        if surf.material is None:
            raise ValueError('refracting surfaces must define a material')
        n_post = surf.material.n(wvl)
        if np.isscalar(n_post):
            n_post = float(n_post)
        else:
            n_post = _to_np(n_post)
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
        substrate geometry for reflective surfaces.
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
        edge geometry for refracting lens groups.

    Returns
    -------
    matplotlib.figure.Figure
        A figure object
    matplotlib.axes.Axis
        An axis object

    """
    wvl = system_wavelength(prescription, wvl)
    x = x.lower()
    y = y.lower()
    fig, ax = share_fig_ax(fig, ax)
    ax.set(aspect='equal')
    result = _require_raytrace_result(result)
    phist = _to_np(result.P)

    lens_groups = lens_groups_from_surfaces(
        prescription, wvl=wvl, ambient_index=ambient_index,
        index_atol=index_atol,
    )
    groups_by_start = {group[0]: (group_index, group)
                       for group_index, group in enumerate(lens_groups)}

    # when the caller does not pass edge geometry explicitly, pick up whatever
    # the surfaces carry themselves (LensData rows / Surface.edge), keyed by the
    # group's first surface index so _lens_edge_for resolves it per element
    if lens_edges is None:
        derived = {}
        for group in lens_groups:
            edge = getattr(prescription[group[0]], 'edge', None)
            if edge is not None:
                derived[group[0]] = edge
        lens_edges = derived or None

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
                surf_obj = prescription[surface_index]
                sag_radius = _surface_drawable_radius(surf_obj, od_radius, y)
                if sag_radius < od_radius * (1.0 - 1e-9):
                    # a surface that physically cannot reach the OD (its sag is
                    # undefined past the equator) is a layout surprise -- warn
                    warnings.warn(
                        f'surface {surface_index} optical sag only spans radius '
                        f'{sag_radius:.4g}, short of the element outer radius '
                        f'{od_radius:.4g}; drawing a flat edge from the surface '
                        f'rim out to the OD',
                        stacklevel=2,
                    )
                draw_radius = sag_radius
                bound = surf_obj.bounding
                if bound is not None and 'outer_radius' in bound:
                    # the surface's own clear aperture also caps its optical
                    # zone; beyond it the element is drawn as a flat land
                    # (normal to the OD) out to the element outer radius.  This
                    # is an intentional aperture, not a surface that cannot
                    # reach the OD, so it is silent
                    draw_radius = min(draw_radius, float(bound['outer_radius']))
                profiles.append(_surface_profile(
                    surf_obj, phist, surface_index, points,
                    y, radius=od_radius, clear_radius=clear_radius,
                    max_radius=draw_radius,
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
                                   chief_index=None, status=None,
                                   reference='chief'):
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
        matplotlib color.
    alpha : float
        opacity.
    zorder : int
        stack order in the plot, higher z orders are on top of lower z orders
    axis : str, {'x', 'y'}
        which ray position to plot, x or y
    fig : matplotlib.figure.Figure
        figure object.
    ax : matplotlib.axes.Axis
        axis object.
    chief_index : int, optional
        row index of the chief ray; default N//2.
    status : ndarray, optional
        per-ray status from raytrace.  Invalid rays are excluded when
        provided.
    reference : str, optional
        image-plane registration point: 'chief' or 'centroid'.

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

    ph = _to_np(phist)
    if status is not None:
        status = _to_np(status)

    input_rays, output_rays = transverse_ray_aberration(
        ph, axis=axis.lower(), chief_index=chief_index, status=status,
        reference=reference,
    )
    input_rays = _to_np(input_rays)
    output_rays = _to_np(output_rays)
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
        matplotlib color.
    alpha : float
        opacity.
    zorder : int
        stack order in the plot, higher z orders are on top of lower z orders
    axis : str, {'x', 'y'}
        pupil axis label.
    label : str, optional
        legend label for this fan.
    fig : matplotlib.figure.Figure
        figure object.
    ax : matplotlib.axes.Axis
        axis object.

    Returns
    -------
    matplotlib.figure.Figure
        A figure object
    matplotlib.axes.Axis
        An axis object

    """
    fig, ax = share_fig_ax(fig, ax)
    coord = _to_np(coord, dtype=float)
    opd = _to_np(opd, dtype=float)

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


def plot_spot_diagram(phist, marker='+', c='k', alpha=1, zorder=4, s=None,
                      origin=None, status=None, label=None, equal_aspect=True,
                      fig=None, ax=None):
    """Plot a spot diagram from a ray trace.

    Parameters
    ----------
    phist : RayTraceResult, list, or ndarray
        Trace result or position history.
    marker : str, optional
        marker style
    c : color
        matplotlib color.
    alpha : float
        opacity.
    zorder : int
        stack order in the plot, higher z orders are on top of lower z orders
    s : float
        marker size or variable used for marker size
    origin : str or iterable, optional
        center to subtract before plotting.
    status : ndarray, optional
        per-ray status from raytrace.
    label : str, optional
        legend label for this bundle.
    equal_aspect : bool, optional
        if True (default) set equal aspect so the spot is not distorted.
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
    if isinstance(phist, RayTraceResult):
        result = phist
        phist = result.P
        if status is None:
            status = result.status
    phist = _to_np(phist)
    x = phist[-1, ..., 0]
    y = phist[-1, ..., 1]
    if status is not None:
        status = _to_np(status)
        valid = valid_mask(status, phist[-1])
        x = x[valid]
        y = y[valid]
    if origin is not None:
        if isinstance(origin, str):
            if origin.lower() == 'centroid':
                origin = (float(np.nanmean(x)), float(np.nanmean(y)))
            else:
                raise ValueError("origin string must be 'centroid'")
        if isinstance(origin, (list, tuple)):
            origin = np.asarray(origin, dtype=float)
        else:
            origin = _to_np(origin, dtype=float)
        x = x - origin[0]
        y = y - origin[1]
    ax.scatter(x, y, c=c, s=s, marker=marker, alpha=alpha, zorder=zorder,
               label=label)
    if equal_aspect:
        ax.set_aspect('equal')
    return fig, ax


def _field_axis_values(fields):
    """Per-field scalar magnitude for a field-curve y-axis.

    Angle fields report degrees; height fields report the radial object
    height in prescription length units.
    """
    values = []
    for field in fields:
        if field.kind == 'angle':
            ax_rad, ay_rad = field.angle_radians()
            values.append(np.degrees(np.hypot(ax_rad, ay_rad)))
        else:
            values.append(np.hypot(field.hx, field.hy))
    return np.asarray(values, dtype=float)


def plot_field_curvature(prescription, fields, wavelength=None, *, epd=None,
                         marginal_fraction=1e-3,
                         reference='image', c='r', lw=1, alpha=1, zorder=4,
                         label=None, fig=None, ax=None):
    """Plot field-curvature x/y fan curves.

    Parameters
    ----------
    prescription : sequence of Surface or LensData
        the optical system.
    fields : iterable of Field
        field points to evaluate, all kind='angle'.
    wavelength : float, optional
        in microns; defaults from a LensData reference wavelength.
    epd : float, optional
        entrance pupil diameter; defaults from a system aperture spec.
    marginal_fraction : float, optional
        pupil zone for the marginal ray, as a fraction of EPD/2.
    reference : str or float, optional
        zero of the focus-shift axis.
    c : color, optional
        curve color; both fans share it and are distinguished by line style.
    lw : float, optional
        line width.
    alpha : float, optional
        opacity.
    zorder : int, optional
        stack order.
    label : str, optional
        base legend label.
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axis

    Returns
    -------
    matplotlib.figure.Figure
    matplotlib.axes.Axis

    """
    fig, ax = share_fig_ax(fig, ax)
    fields = list(fields)
    x_fan_z, y_fan_z = field_curvature(
        prescription, fields, wavelength, epd=epd,
        marginal_fraction=marginal_fraction,
    )
    x_fan_z = _to_np(x_fan_z, dtype=float)
    y_fan_z = _to_np(y_fan_z, dtype=float)
    if reference is None:
        ref = 0.0
    elif isinstance(reference, str):
        if reference.lower() == 'image':
            ref = float(prescription[-1].P[2])
        else:
            raise ValueError("reference string must be 'image'")
    else:
        ref = float(reference)
    field_values = _field_axis_values(fields)
    suffixes, _ = _field_curvature_labels(prescription, fields)
    x_label = suffixes[0] if label is None else f'{label} {suffixes[0]}'
    y_label = suffixes[1] if label is None else f'{label} {suffixes[1]}'
    ax.plot(x_fan_z - ref, field_values, c=c, ls='-', lw=lw, alpha=alpha,
            zorder=zorder, label=x_label)
    ax.plot(y_fan_z - ref, field_values, c=c, ls='--', lw=lw, alpha=alpha,
            zorder=zorder, label=y_label)
    ax.set_xlabel('focus shift')
    ax.set_ylabel('field')
    return fig, ax


def plot_chromatic_focal_shift(prescription, wavelengths=None, *,
                               reference_wavelength=None,
                               focus='best', epd=None, field=None,
                               sampling=None, samples=101,
                               c='r', marker=None, lw=1, alpha=1,
                               zorder=4, label=None, fig=None, ax=None):
    """Plot focus shift against wavelength.

    Parameters
    ----------
    prescription : sequence of Surface or LensData
        the optical system.
    wavelengths : iterable of float, optional
        wavelengths in microns.
    reference_wavelength : float, optional
        wavelength whose focus is used as zero.
    focus : str, optional
        'best' (default) or 'paraxial'; see analysis.chromatic_focal_shift.
    epd : float, optional
        entrance pupil diameter for focus='best'.
    field : Field, optional
        field for focus='best'.
    sampling : Sampling, optional
        ray bundle used for focus='best'.
    samples : int, optional
        number of wavelength samples used when wavelengths is omitted.
    c : color, optional
        curve color.
    marker : str, optional
        marker style.
    lw : float, optional
        line width.
    alpha : float, optional
        opacity.
    zorder : int, optional
        stack order.
    label : str, optional
        legend label.
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axis

    Returns
    -------
    matplotlib.figure.Figure
    matplotlib.axes.Axis

    """
    fig, ax = share_fig_ax(fig, ax)
    wavelengths, shifts = chromatic_focal_shift(
        prescription, wavelengths, reference_wavelength=reference_wavelength,
        focus=focus, epd=epd, field=field, sampling=sampling,
        samples=samples,
    )
    shifts = _to_np(shifts, dtype=float)
    wavelengths = _to_np(wavelengths, dtype=float)
    ax.plot(wavelengths, shifts, c=c, marker=marker, lw=lw, alpha=alpha,
            zorder=zorder, label=label)
    ax.set_xlabel('wavelength [um]')
    ax.set_ylabel('focus shift')
    return fig, ax


def plot_distortion(prescription, fields, wavelength=None, *, epd=None,
                    distortion_type='f-tan', pupil_z=None,
                    c='r', lw=1, alpha=1, zorder=4, label=None,
                    fig=None, ax=None):
    """Plot percent distortion against field magnitude.

    Parameters
    ----------
    prescription : sequence of Surface or LensData
        the optical system.
    fields : iterable of Field
        field points to evaluate, all kind='angle'.
    wavelength : float, optional
        in microns; defaults from a LensData reference wavelength.
    epd : float, optional
        entrance pupil diameter; defaults from a system aperture spec.
    distortion_type : str, optional
        'f-tan' (default) or 'linear-angle'; see analysis.distortion.
    pupil_z : float, optional
        z of the entrance pupil used for the chief-ray launch.
    c : color, optional
        curve color.
    lw : float, optional
        line width.
    alpha : float, optional
        opacity.
    zorder : int, optional
        stack order.
    label : str, optional
        legend label.
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axis

    Returns
    -------
    matplotlib.figure.Figure
    matplotlib.axes.Axis

    """
    fig, ax = share_fig_ax(fig, ax)
    _, _, percent = distortion(
        prescription, fields, wavelength, epd=epd,
        distortion_type=distortion_type, pupil_z=pupil_z,
    )
    percent = _to_np(percent, dtype=float)
    field_values = _field_axis_values(list(fields))
    ax.plot(percent, field_values, c=c, lw=lw, alpha=alpha, zorder=zorder,
            label=label)
    ax.set_xlabel('distortion [%]')
    ax.set_ylabel('field')
    return fig, ax


# ---------- whole-system grid plotters --------------------------------------
# These consume the stacked grids from analysis (RayFanGrid, OPDFanGrid,
# SpotGrid).  Tracing already happened; these only lay out subplots, so the same
# grid can be drawn many ways or saved instead of drawn.

def _wavelength_label(w):
    return f'{float(w):.4g} µm'


def _field_label(field):
    """Short subplot label for a field point."""
    hx = float(getattr(field, 'hx', 0.0))
    hy = float(getattr(field, 'hy', 0.0))
    if getattr(field, 'kind', 'angle') == 'angle':
        return f'({hx:g}, {hy:g})°'
    return f'({hx:g}, {hy:g})'


def _wavelength_colors(wavelengths, colors):
    """One color per wavelength, shared across every subplot of a grid."""
    if colors is not None:
        return list(colors)
    return [f'C{j % 10}' for j in range(len(wavelengths))]


def _wavelength_legend(ax, nw, legend=True):
    """Wavelength legend policy: drawn on ax only for a multi-wavelength grid."""
    if legend and nw > 1:
        ax.legend(fontsize='small', loc='best')


def _plot_fan_grid(grid, value_label, *, axes, colors, sharey, figsize,
                   legend, fig, axs):
    """Shared layout for ray-aberration and OPD fan grids.

    Rows are fields; columns are the tangential (Y) and/or sagittal (X) fans;
    each subplot overlays one curve per wavelength against the shared normalized
    pupil coordinate.
    """
    import matplotlib.pyplot as plt

    fields = grid.fields
    wavelengths = _to_np(grid.wavelengths, dtype=float)
    pupil = _to_np(grid.pupil, dtype=float)
    grid_x = _to_np(grid.x, dtype=float)
    grid_y = _to_np(grid.y, dtype=float)
    nf = len(fields)
    nw = len(wavelengths)

    if axes == 'both':
        columns = [('y', 'tangential (Y)', grid_y),
                   ('x', 'sagittal (X)', grid_x)]
    elif axes == 'y':
        columns = [('y', 'tangential (Y)', grid_y)]
    elif axes == 'x':
        columns = [('x', 'sagittal (X)', grid_x)]
    else:
        raise ValueError(f"axes must be 'both', 'x', or 'y', got {axes!r}")
    ncol = len(columns)

    if fig is None or axs is None:
        if figsize is None:
            figsize = (4.0 * ncol, 2.4 * nf)
        fig, axs = plt.subplots(nf, ncol, sharex=True, sharey=sharey,
                                figsize=figsize, squeeze=False)
    wl_colors = _wavelength_colors(wavelengths, colors)

    for i in range(nf):
        for ci, (axis, title, data) in enumerate(columns):
            ax = axs[i][ci]
            for j in range(nw):
                first = i == 0 and ci == 0
                ax.plot(pupil, data[i, j], c=wl_colors[j], lw=1,
                        label=_wavelength_label(wavelengths[j]) if first else None)
            ax.axhline(0.0, c='k', lw=0.5, alpha=0.4)
            ax.axvline(0.0, c='k', lw=0.5, alpha=0.4)
            if i == 0:
                ax.set_title(title)
            if i == nf - 1:
                ax.set_xlabel(f'normalized pupil {axis}')
            if ci == 0:
                ax.set_ylabel(f'{_field_label(fields[i])}\n{value_label}')
    _wavelength_legend(axs[0][0], nw, legend)
    fig.tight_layout()
    return fig, axs


def plot_ray_fans(fan_grid, *, axes='both', colors=None, sharey='row',
                  figsize=None, legend=True, fig=None, axs=None):
    """Plot a grid of transverse ray-aberration fans.

    Consumes a RayFanGrid from analysis.ray_aberration_fans and draws one row of
    subplots per field -- the tangential (Y) and sagittal (X) fans -- with one
    curve per wavelength.  Tracing and plotting stay separate: this only draws.

    Parameters
    ----------
    fan_grid : RayFanGrid
        the traced fans from analysis.ray_aberration_fans.
    axes : str, optional
        which fans to draw: 'both' (default), 'x', or 'y'.
    colors : sequence, optional
        one matplotlib color per wavelength; defaults to the property cycle.
    sharey : bool or str, optional
        forwarded to subplots; 'row' (default) shares the error scale within a
        field.
    figsize : tuple, optional
        figure size; a sensible default is derived from the grid shape.
    legend : bool, optional
        draw a wavelength legend on the first subplot when multi-wavelength.
    fig : matplotlib.figure.Figure, optional
    axs : ndarray of matplotlib.axes.Axes, optional
        an existing (n_fields, n_columns) axis grid to draw into.

    Returns
    -------
    matplotlib.figure.Figure
    ndarray of matplotlib.axes.Axes
        the (n_fields, n_columns) axis grid.

    """
    return _plot_fan_grid(fan_grid, 'ray error', axes=axes, colors=colors,
                          sharey=sharey, figsize=figsize, legend=legend,
                          fig=fig, axs=axs)


def plot_opd_fans(opd_grid, *, axes='both', colors=None, sharey='row',
                  figsize=None, legend=True, fig=None, axs=None):
    """Plot a grid of wavefront (OPD) fans.

    Consumes an OPDFanGrid from analysis.opd_fans; otherwise identical in layout
    to plot_ray_fans.  The vertical axis is OPD (waves by default, matching the
    grid's output).

    Returns
    -------
    matplotlib.figure.Figure
    ndarray of matplotlib.axes.Axes

    """
    return _plot_fan_grid(opd_grid, 'OPD [waves]', axes=axes, colors=colors,
                          sharey=sharey, figsize=figsize, legend=legend,
                          fig=fig, axs=axs)


def plot_spot_diagrams(spot_grid, *, ncols=None, colors=None, marker='+',
                       s=None, equal_limits=True, legend=True, figsize=None,
                       fig=None, axs=None):
    """Plot a grid of spot diagrams, one subplot per field.

    Consumes a SpotGrid from analysis.spot_diagrams and scatters every
    wavelength (colored) in each field's subplot, using the centered image
    coordinates the grid already carries.  Tracing and plotting stay separate.

    Parameters
    ----------
    spot_grid : SpotGrid
        the traced spots from analysis.spot_diagrams.
    ncols : int, optional
        number of subplot columns; defaults to all fields in one row.
    colors : sequence, optional
        one matplotlib color per wavelength; defaults to the property cycle.
    marker : str, optional
        scatter marker.
    s : float, optional
        marker size.
    equal_limits : bool, optional
        if True (default) give every subplot the same square limits so spot
        sizes are comparable across fields.
    legend : bool, optional
        draw a wavelength legend on the first subplot when multi-wavelength.
    figsize : tuple, optional
        figure size; a sensible default is derived from the layout.
    fig : matplotlib.figure.Figure, optional
    axs : ndarray of matplotlib.axes.Axes, optional

    Returns
    -------
    matplotlib.figure.Figure
    ndarray of matplotlib.axes.Axes
        the (n_rows, n_cols) axis grid.

    """
    import matplotlib.pyplot as plt

    fields = spot_grid.fields
    wavelengths = _to_np(spot_grid.wavelengths, dtype=float)
    xs = _to_np(spot_grid.x, dtype=float)
    ys = _to_np(spot_grid.y, dtype=float)
    nf = len(fields)
    nw = len(wavelengths)

    if ncols is None:
        ncols = nf
    ncols = max(1, min(int(ncols), nf))
    nrows = -(-nf // ncols)  # ceil

    if fig is None or axs is None:
        if figsize is None:
            figsize = (3.0 * ncols, 3.0 * nrows)
        fig, axs = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    wl_colors = _wavelength_colors(wavelengths, colors)

    lim = None
    if equal_limits:
        rmax = np.nanmax(np.sqrt(xs * xs + ys * ys))
        if np.isfinite(rmax) and rmax > 0:
            lim = 1.1 * float(rmax)

    flat = [axs[r][c] for r in range(nrows) for c in range(ncols)]
    for idx, field in enumerate(fields):
        ax = flat[idx]
        for j in range(nw):
            ax.scatter(xs[idx, j], ys[idx, j], c=wl_colors[j], marker=marker,
                       s=s, label=_wavelength_label(wavelengths[j])
                       if idx == 0 else None)
        ax.set_aspect('equal')
        ax.set_title(_field_label(field))
        if lim is not None:
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
    for idx in range(nf, nrows * ncols):
        flat[idx].axis('off')
    _wavelength_legend(flat[0], nw, legend)
    fig.tight_layout()
    return fig, axs
