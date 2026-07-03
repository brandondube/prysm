"""Plotting functions for raytraces."""

import warnings

from prysm.plotting import share_fig_ax
from prysm.mathops import array_to_true_numpy

from .spencer_and_murty import (
    RayTraceResult,
    transform_to_global_coords,
    transform_to_local_coords,
)
from .analysis import (
    transverse_ray_aberration,
    chromatic_focal_shift,
    field_curvature,
    distortion,
    lateral_color,
    spot_positions,
)
from ._resolve import resolve_wavelength
from .surfaces import STYPE_REFLECT, STYPE_REFRACT
from .aperture import SurfaceSubstrate
from .lensdata import lens_element_groups
from ._trace_grid import (
    _resolve_wavelengths, field_sweep, layout_records,
)

import numpy as np  # see module docstring; do not "fix" to mathops np

_to_np = array_to_true_numpy


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
    fig, ax = share_fig_ax(fig, ax)

    ph = _to_np(result.P)
    status = getattr(result, 'status', None)
    if status is not None:
        # a failed ray's history keeps marching past the surface that killed
        # it; blank everything after the failure so the drawn path ends where
        # the ray did.  imag > 0 (clip / no convergence) means the ray reached
        # surface status.real (1-based), so its intersection there is real;
        # imag < 0 (miss / TIR / evanescent) means it never made it.
        status = _to_np(status)
        real = status.real.astype(int)
        imag = status.imag.astype(int)
        nhist = ph.shape[0]
        last = np.where(imag == 0, nhist - 1, np.where(imag > 0, real, real - 1))
        dead = np.arange(nhist)[:, None] > last[None, :]
        if dead.any():
            ph = ph.copy()
            ph[dead] = np.nan
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
    sag = np.asarray(sag)
    ploty = np.asarray(ploty)
    xpart, ypart = _axis_pair(ploty, y)
    local = np.stack([xpart, ypart, sag - surf.P[2]], axis=-1)
    dirs = np.zeros_like(local)
    rotation = None if surf.R is None else surf.R.T
    global_points, _ = transform_to_global_coords(
        local, surf.P, dirs, rotation,
    )
    return global_points[..., _axis_index(x)], global_points[..., _axis_index(y)]


def _extent_inner_radius(surf):
    """Central bore radius of a surface's drawn extent (0 when none)."""
    extent = surf.aperture.extent
    return 0.0 if extent is None else extent.inner_radius


def _surface_profile(surf, points, y, *, outer_radius, inner_radius=0.0,
                     center=0.0, max_radius=None):
    """Meridional profile (sag, ploty, edge_sag) of one surface face.

    The face is sampled over +/- outer_radius about center; inner_radius masks a
    central bore.  max_radius clamps the sag evaluation (where the surface still
    exists), holding the edge value out to the rim so a steep surface bridges
    flat to the drawn outer radius.
    """
    local = np.linspace(-outer_radius, outer_radius, points)
    ploty = center + local
    mask = np.abs(local) < inner_radius
    if max_radius is not None:
        eval_local = np.clip(local, -max_radius, max_radius)
    else:
        eval_local = local
    xpt, ypt = _axis_pair(center + eval_local, y)
    sag = np.asarray(surf.sag(xpt, ypt)) + surf.P[2]
    edge_sag = sag.copy()
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
        sag = surf.sag(xpt, ypt)
    bad = ~np.isfinite(sag)
    if not bad.any():
        return radius
    first = int(np.argmax(bad))
    return float(probe[first - 1]) if first > 0 else 0.0


def _stop_marker_outline(surf, phist, shist, j, x, y, stem_fraction=0.2):
    """Aperture-stop T marks in global plot coordinates.

    The stop is drawn as one small T per clear-aperture edge on the drawn
    meridian: a stem from the aperture edge pointing radially outward, normal
    to the local optical axis, and a shorter crossbar through the edge
    parallel to the local optical axis.  The local optical axis is the chief
    ray direction at the stop -- the traced ray closest to the stop center --
    falling back to the surface local z axis when the trace carries no usable
    directions.  The clear radius is read from the traced ray extent at the
    stop, since the marginal rays graze the stop edge by definition, and the
    mark size scales with it through stem_fraction.

    Returns (xx, yy) plot coordinates with NaN separators, or None when the
    trace has no finite ray data at the stop surface.

    """
    p = phist[j + 1].reshape(-1, 3)
    dirs = np.zeros_like(p)
    p_loc, _ = transform_to_local_coords(p, surf.P, dirs, surf.R)
    coord = p_loc[..., _axis_index(y)]
    if not np.isfinite(coord).any():
        return None
    a = max(abs(np.nanmin(coord)), abs(np.nanmax(coord)))
    if not (np.isfinite(a) and a > 0):
        return None

    ix = _axis_index(x)
    iy = _axis_index(y)
    rsq = p_loc[..., 0] * p_loc[..., 0] + p_loc[..., 1] * p_loc[..., 1]
    rsq = np.where(np.isfinite(rsq), rsq, np.inf)
    chief = int(np.argmin(rsq))
    s = shist[j + 1].reshape(-1, 3)[chief]
    t = np.asarray([s[ix], s[iy]], dtype=float)
    norm = np.hypot(t[0], t[1])
    if norm == 0 or not np.isfinite(norm):
        # no usable chief direction (synthetic or dead trace); the surface
        # local z axis stands in.  local = R @ (global - P), so the local z
        # axis expressed globally is the third row of R
        axis = np.asarray([0.0, 0.0, 1.0]) if surf.R is None else _to_np(surf.R)[2]
        t = np.asarray([axis[ix], axis[iy]], dtype=float)
        norm = np.hypot(t[0], t[1])
        if norm == 0:
            return None
    t = t / norm
    outward = np.asarray([-t[1], t[0]])

    ploty = np.asarray([-a, a])
    xpt, ypt = _axis_pair(ploty, y)
    sag = np.asarray(surf.sag(xpt, ypt)) + surf.P[2]
    ex, ey = _global_plot_coordinates(surf, sag, ploty, x, y)
    cx = float(np.mean(ex))
    cy = float(np.mean(ey))

    stem = stem_fraction * a
    bar = 0.5 * stem
    xx, yy = [], []
    for k in range(2):
        e0 = float(ex[k])
        e1 = float(ey[k])
        sign = 1.0 if outward[0] * (e0 - cx) + outward[1] * (e1 - cy) >= 0 else -1.0
        out = sign * outward
        xx += [e0 - 0.5 * bar * t[0], e0 + 0.5 * bar * t[0], np.nan,
               e0, e0 + stem * out[0], np.nan]
        yy += [e1 - 0.5 * bar * t[1], e1 + 0.5 * bar * t[1], np.nan,
               e1, e1 + stem * out[1], np.nan]
    return xx, yy


def _warn_unsolved_extent():
    """Warn once that an auto aperture is drawn from the per-call footprint."""
    # Static message so Python's once-per-location dedup holds across the loop.
    warnings.warn(
        'drawing a surface whose auto aperture is unsolved or stale; sizing it '
        'from the per-call ray footprint.  Call sys.solve.apertures() to size '
        'and persist the drawn extents.',
        stacklevel=2,
    )


def _resolve_center(center, phist, surface_index, surf, y):
    """Resolve a drawing center: a number, or 'rays' -> mean ray height."""
    if isinstance(center, str):
        if center.lower() in ('rays', 'ray', 'footprint', 'chief'):
            coord = _local_axis_coordinates_from_phist(
                phist, surface_index, y, surf=surf)
            return float(np.nanmean(coord))
        raise ValueError(f'unknown drawing center {center!r}')
    return float(center)


def _aperture_radius(surf, phist, surface_index, y, center=0.0):
    """Drawn half-diameter from the surface aperture, else the ray footprint."""
    ap = surf.aperture
    if ap.extent is not None:
        return ap.extent.outer_radius
    if ap.clip is not None:
        return ap.drawn_radius()
    # the unsolved footprint fallback is the raw extent (today's behavior); the
    # oversize lives in the solved extent that sys.solve.apertures() persists.
    return _axis_extent_from_phist(phist, surface_index, y, surf=surf,
                                   center=center)


def _drawn_radius(surf, phist, j, y, version, center=0.0):
    """Version-aware drawn half-diameter; a stale/unsolved auto extent warns."""
    ap = surf.aperture
    extent = ap.extent
    if extent is not None and not ap.is_stale(version):
        return extent.outer_radius
    if ap.clip is not None:
        return ap.drawn_radius()
    _warn_unsolved_extent()
    return _axis_extent_from_phist(phist, j, y, surf=surf, center=center)


def _mirror_surface_outline_from_phist(surf, phist, surface_index, points, x, y,
                                       radius, center=0.0):
    if radius is None:
        radius = _aperture_radius(surf, phist, surface_index, y, center)
    sag, ploty, _ = _surface_profile(
        surf, points, y, outer_radius=radius,
        inner_radius=_extent_inner_radius(surf), center=center)
    return _global_plot_coordinates(surf, sag, ploty, x, y)


def _mirror_substrate_outline_from_phist(surf, phist, surface_index, substrate,
                                         points, x, y, radius, center=0.0):
    if radius is None:
        radius = _aperture_radius(surf, phist, surface_index, y, center)
    inner = _extent_inner_radius(surf)
    sag, ploty, edge_sag = _surface_profile(
        surf, points, y, outer_radius=radius, inner_radius=inner, center=center)
    if substrate is None:
        substrate = SurfaceSubstrate()
    bore = max(inner, getattr(substrate, 'bore', 0.0) or 0.0)
    zz, tt = substrate.back_outline(surf, ploty, sag, edge_sag, center,
                                    bore=bore)
    return _global_plot_coordinates(surf, zz, tt, x, y)


def mirror_surface_outline(surf, result, surface_index=0, *, points=100,
                           x='z', y='y', radius=None, center=0.0):
    """Return X/Y arrays for drawing one mirror optical surface.

    Parameters
    ----------
    surf : Surface
        the reflective surface to outline.
    result : RayTraceResult
        a trace whose ray positions bound the drawn extent when radius and the
        surface aperture extent are both unset.
    surface_index : int, optional
        index of surf within the traced system, used to read the ray positions
        at that surface.
    points : int, optional
        number of points sampled along the surface profile.
    x, y : str, optional
        position components mapped to the plot horizontal and vertical axes;
        defaults to the traditional z-y meridional view.
    radius : float, optional
        outer half-diameter of the drawn profile; defaults to the surface
        aperture's drawn radius, else the traced ray extent.
    center : float or str, optional
        transverse center of the drawn extent; the string rays centers on the
        mean ray height at the surface.

    Returns
    -------
    xx, yy : ndarray, ndarray
        plot coordinates of the surface profile in global coordinates.

    """
    phist = _to_np(result.P)
    x = x.lower()
    y = y.lower()
    center = _resolve_center(center, phist, surface_index, surf, y)
    return _mirror_surface_outline_from_phist(
        surf, phist, surface_index, points, x, y, radius, center)


def mirror_substrate_outline(surf, result, surface_index=0, *, substrate,
                             points=100, x='z', y='y', radius=None,
                             center=0.0):
    """Return X/Y arrays for drawing one mirror substrate outline.

    Parameters
    ----------
    surf : Surface
        the reflective surface to outline.
    result : RayTraceResult
        a trace whose ray positions bound the drawn extent when radius and the
        surface aperture extent are both unset.
    surface_index : int, optional
        index of surf within the traced system, used to read the ray positions
        at that surface.
    substrate : Substrate or None
        the substrate to draw (ParallelSubstrate / FlatParentSubstrate /
        FlatBackSubstrate / SurfaceSubstrate); None draws the optical face only.
    points : int, optional
        number of points sampled along the surface profile.
    x, y : str, optional
        plot axes; defaults to the traditional z-y meridional view.
    radius : float, optional
        outer half-diameter override; defaults to the surface aperture's drawn
        radius, else the traced ray extent.
    center : float or str, optional
        transverse center of the drawn extent; rays centers on the mean ray
        height at the surface.

    Returns
    -------
    xx, yy : list, list
        z (axial) and transverse coordinates of the closed substrate outline.

    """
    phist = _to_np(result.P)
    x = x.lower()
    y = y.lower()
    center = _resolve_center(center, phist, surface_index, surf, y)
    return _mirror_substrate_outline_from_phist(
        surf, phist, surface_index, substrate, points, x, y, radius, center)


def plot_mirror_surface(surf, result, surface_index=0, *, points=100,
                        x='z', y='y', radius=None, center=0.0,
                        lw=1, ls='-', c='k', alpha=1, zorder=3,
                        fig=None, ax=None):
    """Draw one mirror optical surface (see mirror_surface_outline)."""
    fig, ax = share_fig_ax(fig, ax)
    xx, yy = mirror_surface_outline(
        surf, result, surface_index, points=points, y=y, radius=radius,
        x=x, center=center)
    ax.plot(xx, yy, c=c, lw=lw, ls=ls, alpha=alpha, zorder=zorder)
    return fig, ax


def plot_mirror_substrate(surf, result, surface_index=0, *, substrate,
                          points=100, x='z', y='y', radius=None, center=0.0,
                          lw=1, ls='-', c='k', alpha=1, zorder=3,
                          fig=None, ax=None):
    """Draw one mirror with an optical surface and substrate."""
    fig, ax = share_fig_ax(fig, ax)
    xx, yy = mirror_substrate_outline(
        surf, result, surface_index, substrate=substrate, points=points,
        x=x, y=y, radius=radius, center=center)
    ax.plot(xx, yy, c=c, lw=lw, ls=ls, alpha=alpha, zorder=zorder)
    return fig, ax


def _append_wall_point(xs, ys, x, y):
    if xs and xs[-1] == x and ys[-1] == y:
        return
    xs.append(x)
    ys.append(y)


def _wall_path(x0, x1, outer_y, features, side, endpoint_names):
    """Rim-wall meridian from x0 to x1, inset by each applicable EdgeFeature."""
    xs = [x0]
    ys = [outer_y]
    direction = np.sign(x1 - x0) or 1
    current = x0
    spans = []

    lo = min(x0, x1)
    hi = max(x0, x1)
    for feature in features:
        if not feature.applies_to(side):
            continue
        start, end, depth = feature.span(x0, x1, endpoint_names)
        if direction < 0:
            start, end = end, start
        start = min(max(start, lo), hi)
        end = min(max(end, lo), hi)
        if start == end:
            continue
        spans.append((start, end, depth, feature.is_chamfer))

    spans.sort(key=lambda item: direction * item[0])

    for start, end, depth, is_chamfer in spans:
        inset = outer_y + depth if outer_y < 0 else outer_y - depth
        if direction * (start - current) > 0:
            _append_wall_point(xs, ys, start, outer_y)
        if is_chamfer:
            _append_wall_point(xs, ys, end, inset)
        else:
            _append_wall_point(xs, ys, start, inset)
            _append_wall_point(xs, ys, end, inset)
        _append_wall_point(xs, ys, end, outer_y)
        current = end

    _append_wall_point(xs, ys, x1, outer_y)
    return xs, ys


def _system_version(system):
    """The owning LensData edit version (None for a bare surface list)."""
    return getattr(getattr(system, 'lens', system), '_version', None)


def plot_optics(system, result, *, wvl=None, ambient_index=1.0,
                index_atol=1e-9, points=100,
                lw=1, ls='-', c='k', alpha=1, zorder=3,
                x='z', y='y', fig=None, ax=None, stop_index=None):
    """Draw the optics of a system.

    Drawing is driven by each surface's Aperture: its drawn extent sizes the
    optical face, its substrate (reflective surfaces) draws the back, and its
    rim features inset the element walls.  A surface whose auto extent is
    unsolved or stale is sized from the per-call ray footprint (and warns once);
    call sys.solve.apertures() to size and persist the extents.

    Parameters
    ----------
    system : iterable of Surface
        a system for an optical layout
    result : RayTraceResult
        Trace result returned by spencer_and_murty.raytrace.
    wvl : float, optional
        Wavelength in microns used to evaluate post-surface material indices.
        Defaults from the LensData reference wavelength, else 0.6328.
    ambient_index : float, optional
        Refractive index that closes a physical lens group.
    index_atol : float, optional
        Absolute tolerance for comparing material index to ambient_index.
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
    stop_index : int, optional
        index of the aperture stop surface; defaults from the system
        (OpticalSystem / LensData stop_index).  A stop that draws nothing
        otherwise (a standalone dummy plane or eval surface) is marked with a
        small T at each clear-aperture edge: the crossbar parallel to the
        local optical axis (the chief ray direction there) and the stem
        pointing radially outward.  A stop on a lens or mirror surface draws
        no extra mark.

    Returns
    -------
    matplotlib.figure.Figure
        A figure object
    matplotlib.axes.Axis
        An axis object

    """
    wvl = resolve_wavelength(system, wvl)
    x = x.lower()
    y = y.lower()
    fig, ax = share_fig_ax(fig, ax)
    ax.set(aspect='equal')
    phist = _to_np(result.P)
    version = _system_version(system)
    if stop_index is None:
        stop_index = getattr(system, 'stop_index', None)

    def draw_stop_marker(j, surf):
        marks = _stop_marker_outline(surf, phist, _to_np(result.S), j, x, y)
        if marks is not None:
            ax.plot(*marks, c=c, lw=lw, ls=ls, alpha=alpha, zorder=zorder)

    lens_groups = lens_element_groups(
        system, wvl=wvl, ambient_index=ambient_index,
        index_atol=index_atol,
    )
    groups_by_start = {group[0]: (group_index, group)
                       for group_index, group in enumerate(lens_groups)}

    j = 0
    jj = len(system)
    while j < jj:
        surf = system[j]
        if surf.typ == STYPE_REFLECT:
            radius = _drawn_radius(surf, phist, j, y, version)
            substrate = surf.aperture.substrate
            if substrate is None:
                xx, yy = _mirror_surface_outline_from_phist(
                    surf, phist, j, points, x, y, radius)
            else:
                xx, yy = _mirror_substrate_outline_from_phist(
                    surf, phist, j, substrate, points, x, y, radius)
            ax.plot(xx, yy, c=c, lw=lw, ls=ls, alpha=alpha, zorder=zorder)
            j += 1
        elif surf.typ == STYPE_REFRACT:
            if j not in groups_by_start:
                # an ambient-to-ambient dummy plane; it belongs to no lens
                # element.  When it is the aperture stop it draws the stop
                # marks; otherwise nothing
                if j == stop_index:
                    draw_stop_marker(j, surf)
                j += 1
                continue
            group_index, group = groups_by_start[j]

            # the element OD is the largest drawn radius in the group
            group_radii = [_drawn_radius(system[si], phist, si, y, version)
                           for si in group]
            od_radius = max(group_radii)

            profiles = []
            for own_radius, surface_index in zip(group_radii, group):
                surf_obj = system[surface_index]
                sag_radius = _surface_drawable_radius(surf_obj, od_radius, y)
                # a surface drawn smaller than the element OD caps its optical
                # zone at its own drawn radius, bridging a flat land out to the
                # OD; that is an intentional aperture and stays silent
                cap = (own_radius if own_radius < od_radius * (1.0 - 1e-9)
                       else None)
                draw_radius = sag_radius if cap is None else min(sag_radius, cap)
                if (sag_radius < od_radius * (1.0 - 1e-9)
                        and (cap is None or sag_radius < cap)):
                    # a surface that physically cannot reach the OD (its sag is
                    # undefined past the equator) is a layout surprise -- warn
                    warnings.warn(
                        f'surface {surface_index} optical sag only spans radius '
                        f'{sag_radius:.4g}, short of the element outer radius '
                        f'{od_radius:.4g}; drawing a flat edge from the surface '
                        f'rim out to the OD',
                        stacklevel=2,
                    )
                profiles.append(_surface_profile(
                    surf_obj, points, y, outer_radius=od_radius,
                    inner_radius=_extent_inner_radius(surf_obj),
                    max_radius=draw_radius))

            sag1, ploty1, edge_sag1 = profiles[0]
            sag2, ploty2, edge_sag2 = profiles[-1]
            # rim features come from the group's first and last surfaces (D2)
            features = (list(system[group[0]].aperture.features)
                        + list(system[group[-1]].aperture.features))

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
            # eval surfaces draw nothing, except the stop marks when one is
            # the aperture stop
            if j == stop_index:
                draw_stop_marker(j, surf)
            j += 1

    return fig, ax


def layout(system, *, fields=None, wavelength=None, sampling=None, axis='y',
           x='z', y='y', colors=None, lw=1, alpha=1, fig=None, ax=None):
    """Draw a 2D layout: the optics plus a ray fan per field.

    A one-call composition of plot_optics (once) and plot_ray_paths (per field)
    for the common show-me-the-system view.  Tracing happens in
    layout_records, so system must carry (or be given) enough metadata to
    launch -- an aperture, fields, and a wavelength.  Solved surface extents
    size the glass; an unsolved auto extent falls back to the per-call ray
    footprint (and warns once).

    Parameters
    ----------
    system : OpticalSystem or sequence of Surface
        the optical system to draw.
    fields : iterable of Field, optional
        fields to trace; defaults to the system field set, else the on-axis
        field.
    wavelength : float, optional
        wavelength in microns; defaults to the system reference wavelength.
    sampling : Sampling or int, optional
        pupil sampling for the drawn fans; an int is shorthand for a fan of
        that many rays along axis.  Defaults to a 3-ray fan along axis.
    axis : str, optional
        fan axis 'y' (default) or 'x', used when sampling is None or an int.
    x, y : str, optional
        position components mapped to the plot horizontal and vertical axes;
        defaults to the traditional z-y meridional view.
    colors : sequence, optional
        one matplotlib color per field; defaults to the property cycle.
    lw, alpha : float, optional
        ray line width and opacity.
    fig : matplotlib.figure.Figure, optional
    ax : matplotlib.axes.Axis, optional

    Returns
    -------
    matplotlib.figure.Figure
        A figure object
    matplotlib.axes.Axis
        An axis object

    """
    records, outline = layout_records(system, fields=fields,
                                      wavelength=wavelength,
                                      sampling=sampling, axis=axis)
    if colors is None:
        colors = [f'C{i % 10}' for i in range(len(records))]
    fig, ax = plot_optics(system, outline, wvl=wavelength, x=x, y=y,
                          fig=fig, ax=ax)
    for record, color in zip(records, colors):
        plot_ray_paths(record.trace, x=x, y=y, c=color, lw=lw, alpha=alpha,
                       fig=fig, ax=ax)
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
    coord = _to_np(coord)
    opd = _to_np(opd)

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
    if status is not None:
        status = _to_np(status)
    if origin is not None and not isinstance(origin, str):
        origin = _to_np(np.asarray(origin))
    x, y = spot_positions(phist[-1], status=status, origin=origin)
    ax.scatter(x, y, c=c, s=s, marker=marker, alpha=alpha, zorder=zorder,
               label=label)
    if equal_aspect:
        ax.set_aspect('equal')
    return fig, ax


def _field_axis_values(fields):
    """Per-field scalar magnitude for a field-curve y-axis.

    Angle fields report degrees; height fields report the radial object
    height in system length units.
    """
    values = []
    for field in fields:
        if field.kind == 'angle':
            ax_rad, ay_rad = field.angle_radians()
            values.append(np.degrees(np.hypot(ax_rad, ay_rad)))
        else:
            values.append(np.hypot(field.hx, field.hy))
    return np.asarray(values)


def plot_field_curvature(system, fields=None, wavelength=None, *,
                         samples=101, result=None,
                         reference='image', c='r', lw=1, alpha=1, zorder=4,
                         label=None, fig=None, ax=None):
    """Plot field-curvature x/y fan curves.

    Parameters
    ----------
    system : sequence of Surface or LensData
        the optical system.
    fields : iterable of Field, optional
        field points to evaluate; defaults to a dense sweep over the system
        FieldSet span (see field_sweep).
    wavelength : float, optional
        in microns; defaults from a LensData reference wavelength.
    samples : int, optional
        number of sweep points when fields is None.
    result : FieldCurvatureResult, optional
        precomputed analysis.field_curvature output; when given no rays are
        traced here and fields and wavelength are unused.
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
    if result is None:
        result = field_curvature(system, fields, wavelength, samples=samples)
    x_fan_z = _to_np(result.x_fan_z)
    y_fan_z = _to_np(result.y_fan_z)
    if reference is None:
        ref = 0.0
    elif isinstance(reference, str):
        if reference.lower() == 'image':
            ref = float(result.image_z)
        else:
            raise ValueError("reference string must be 'image'")
    else:
        ref = float(reference)
    field_values = _field_axis_values(result.fields)
    suffixes = result.labels
    x_label = suffixes[0] if label is None else f'{label} {suffixes[0]}'
    y_label = suffixes[1] if label is None else f'{label} {suffixes[1]}'
    ax.plot(x_fan_z - ref, field_values, c=c, ls='-', lw=lw, alpha=alpha,
            zorder=zorder, label=x_label)
    ax.plot(y_fan_z - ref, field_values, c=c, ls='--', lw=lw, alpha=alpha,
            zorder=zorder, label=y_label)
    ax.set_xlabel('focus shift')
    ax.set_ylabel('field')
    return fig, ax


def plot_chromatic_focal_shift(system, wavelengths=None, *,
                               reference_wavelength=None,
                               focus='best', epd=None, field=None,
                               sampling=None, samples=101, result=None,
                               c='r', marker=None, lw=1, alpha=1,
                               zorder=4, label=None, fig=None, ax=None):
    """Plot focus shift against wavelength.

    Parameters
    ----------
    system : sequence of Surface or LensData
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
    result : tuple of ndarray, optional
        precomputed (wavelengths, shifts) from analysis.chromatic_focal_shift;
        when given no tracing happens here and the data keywords are unused.
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
    if result is None:
        result = chromatic_focal_shift(
            system, wavelengths,
            reference_wavelength=reference_wavelength,
            focus=focus, epd=epd, field=field, sampling=sampling,
            samples=samples,
        )
    wavelengths, shifts = result
    shifts = _to_np(shifts)
    wavelengths = _to_np(wavelengths)
    ax.plot(wavelengths, shifts, c=c, marker=marker, lw=lw, alpha=alpha,
            zorder=zorder, label=label)
    ax.set_xlabel('wavelength [um]')
    ax.set_ylabel('focus shift')
    return fig, ax


def plot_distortion(system, fields=None, wavelength=None, *, epd=None,
                    distortion_type='f-tan', pupil_z=None, samples=101,
                    result=None,
                    c='r', lw=1, alpha=1, zorder=4, label=None,
                    fig=None, ax=None):
    """Plot percent distortion against field magnitude.

    Parameters
    ----------
    system : sequence of Surface or LensData
        the optical system.
    fields : iterable of Field, optional
        field points to evaluate, all kind='angle'; defaults to a dense
        sweep over the system FieldSet span (see field_sweep).
    wavelength : float, optional
        in microns; defaults from a LensData reference wavelength.
    epd : float, optional
        entrance pupil diameter; defaults from a system aperture spec.
    distortion_type : str, optional
        'f-tan' (default) or 'linear-angle'; see analysis.distortion.
    pupil_z : float, optional
        z of the entrance pupil used for the chief-ray launch.
    samples : int, optional
        number of sweep points when fields is None.
    result : DistortionResult, optional
        precomputed analysis.distortion output; when given no rays are traced
        here and fields, wavelength, epd, distortion_type, and pupil_z are
        unused.
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
    if result is None:
        result = distortion(
            system, fields, wavelength, epd=epd,
            distortion_type=distortion_type, pupil_z=pupil_z,
            samples=samples,
        )
    percent = _to_np(result.percent)
    field_values = _field_axis_values(result.fields)
    ax.plot(percent, field_values, c=c, lw=lw, alpha=alpha, zorder=zorder,
            label=label)
    ax.set_xlabel('distortion [%]')
    ax.set_ylabel('field')
    return fig, ax


def plot_lateral_color(system, fields=None, wavelengths=None, *,
                       epd=None, reference_wavelength=None, samples=101,
                       result=None,
                       colors=None, lw=1, alpha=1, zorder=4, legend=True,
                       fig=None, ax=None):
    """Plot lateral color (chief-ray height error vs reference) against field.

    One curve per non-reference wavelength: the signed difference of the
    chief-ray image height from the reference-wavelength height, projected
    onto the reference landing direction so the sign survives (the distortion
    convention).  Fields whose reference landing is on axis read zero.

    Parameters
    ----------
    system : sequence of Surface or LensData
        the optical system.
    fields : iterable of Field, optional
        field points to evaluate, all kind='angle'; defaults to a dense
        sweep over the system FieldSet span (see field_sweep).
    wavelengths : iterable of float, optional
        wavelengths in microns; defaults to the system wavelength set.
    epd : float, optional
        entrance pupil diameter; defaults from a system aperture spec.
    reference_wavelength : float, optional
        wavelength whose chief-ray landing is the zero; the nearest grid
        wavelength is used.  Defaults to the system reference wavelength.
    samples : int, optional
        number of sweep points when fields is None.
    result : ndarray, optional
        precomputed analysis.lateral_color landing grid for fields and
        wavelengths; when given no rays are traced here and epd is unused.
    colors : sequence, optional
        one matplotlib color per wavelength; defaults to the property cycle.
    lw : float, optional
        line width.
    alpha : float, optional
        opacity.
    zorder : int, optional
        stack order.
    legend : bool, optional
        draw a wavelength legend when more than one curve is drawn.
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axis

    Returns
    -------
    matplotlib.figure.Figure
    matplotlib.axes.Axis

    """
    fig, ax = share_fig_ax(fig, ax)
    fields = field_sweep(system, fields, samples)
    wavelengths = _resolve_wavelengths(system, wavelengths)
    if result is None:
        result = lateral_color(system, fields, wavelengths, epd=epd)
    landing = _to_np(result)
    wavelengths = np.asarray(wavelengths, dtype=float)
    if reference_wavelength is None:
        reference_wavelength = resolve_wavelength(system, None)
    j_ref = int(np.argmin(np.abs(wavelengths - float(reference_wavelength))))

    ref = landing[:, j_ref, :]
    href = np.hypot(ref[:, 0], ref[:, 1])
    safe = np.where(href > 0, href, 1.0)
    height = (landing * ref[:, None, :]).sum(axis=-1) / safe[:, None]
    delta = np.where(href[:, None] > 0, height - href[:, None], 0.0)

    field_values = _field_axis_values(fields)
    wl_colors = _wavelength_colors(wavelengths, colors)
    for j in range(len(wavelengths)):
        if j == j_ref:
            continue
        ax.plot(delta[:, j], field_values, c=wl_colors[j], lw=lw, alpha=alpha,
                zorder=zorder, label=_wavelength_label(wavelengths[j]))
    _wavelength_legend(ax, len(wavelengths), legend)
    ax.set_xlabel('lateral color')
    ax.set_ylabel('field')
    return fig, ax


_FULL_FIELD_LABELS = {
    'rms spot': 'RMS spot radius',
    'rms wfe': 'RMS wavefront error [waves]',
    'distortion': 'distortion [%]',
    'lateral color': 'lateral color',
}


def plot_full_field(grid, *, cmap='viridis', clim=None, colorbar=True,
                    fig=None, ax=None):
    """Plot a full-field display (a 2D metric map over the field disc).

    Parameters
    ----------
    grid : FullFieldGrid
        analysis.full_field output; data is NaN outside the field disc and
        those cells are left blank.
    cmap : str or matplotlib colormap, optional
        color map.
    clim : tuple of float, optional
        explicit (lo, hi) color limits; default scales to the data.
    colorbar : bool, optional
        draw a colorbar labeled with the metric.
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axis

    Returns
    -------
    matplotlib.figure.Figure
    matplotlib.axes.Axis

    """
    fig, ax = share_fig_ax(fig, ax)
    hx = _to_np(grid.hx)
    hy = _to_np(grid.hy)
    data = _to_np(grid.data)
    im = ax.pcolormesh(hx, hy, data, cmap=cmap, shading='nearest')
    if clim is not None:
        im.set_clim(*clim)
    ax.set_aspect('equal')
    unit = grid.unit if grid.kind == 'angle' else 'length'
    ax.set_xlabel(f'field x [{unit}]')
    ax.set_ylabel(f'field y [{unit}]')
    if colorbar:
        fig.colorbar(im, ax=ax,
                     label=_FULL_FIELD_LABELS.get(grid.metric, grid.metric))
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
    each subplot overlays one curve per wavelength against that field's
    normalized pupil coordinate (vignetted fans span less than +/-1).
    """
    import matplotlib.pyplot as plt

    fields = grid.fields
    wavelengths = _to_np(grid.wavelengths)
    pupil_x = _to_np(grid.pupil_x)
    pupil_y = _to_np(grid.pupil_y)
    grid_x = _to_np(grid.x)
    grid_y = _to_np(grid.y)
    nf = len(fields)
    nw = len(wavelengths)

    if axes == 'both':
        columns = [('y', 'tangential (Y)', pupil_y, grid_y),
                   ('x', 'sagittal (X)', pupil_x, grid_x)]
    elif axes == 'y':
        columns = [('y', 'tangential (Y)', pupil_y, grid_y)]
    elif axes == 'x':
        columns = [('x', 'sagittal (X)', pupil_x, grid_x)]
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
        for ci, (axis, title, pupil, data) in enumerate(columns):
            ax = axs[i][ci]
            # per-field abscissa: a vignetted fan spans only its transmitted
            # extent of the nominal pupil; it is never stretched back to +/-1
            abscissa = pupil[i] if pupil.ndim == 2 else pupil
            for j in range(nw):
                first = i == 0 and ci == 0
                ax.plot(abscissa, data[i, j], c=wl_colors[j], lw=1,
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
    wavelengths = _to_np(spot_grid.wavelengths)
    xs = _to_np(spot_grid.x)
    ys = _to_np(spot_grid.y)
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
