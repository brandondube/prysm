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

from .spencer_and_murty import RayTraceResult
from .surfaces import STYPE_REFLECT, STYPE_REFRACT

import numpy as np  # see module docstring; do not "fix" to mathops np


def _require_raytrace_result(result):
    if not isinstance(result, RayTraceResult):
        raise TypeError('expected a RayTraceResult')
    return result


def _plot_position_history(phist, *, x='z', y='y', lw=1, ls='-', c='r',
                           alpha=1, zorder=4, fig=None, ax=None):
    fig, ax = share_fig_ax(fig, ax)

    ph = np.asarray(array_to_true_numpy(phist))
    xs = ph[..., 0]
    ys = ph[..., 1]
    zs = ph[..., 2]
    sieve = {
        'x': xs,
        'y': ys,
        'z': zs,
    }
    x = x.lower()
    y = y.lower()
    x = sieve[x]
    y = sieve[y]
    ax.plot(x, y, c=c, lw=lw, ls=ls, alpha=alpha, zorder=zorder)
    return fig, ax


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
    return _plot_position_history(
        result.P, x=x, y=y, lw=lw, ls=ls, c=c, alpha=alpha,
        zorder=zorder, fig=fig, ax=ax,
    )


def plot_rays(result, *args, **kwargs):
    """Deprecated alias for plot_ray_paths."""
    import warnings
    warnings.warn(
        'plot_rays is deprecated; use plot_ray_paths with a RayTraceResult.',
        DeprecationWarning,
        stacklevel=2,
    )
    if isinstance(result, RayTraceResult):
        return plot_ray_paths(result, *args, **kwargs)
    return _plot_position_history(result, *args, **kwargs)


def _phist_from_result_or_history(result):
    if isinstance(result, RayTraceResult):
        return np.asarray(array_to_true_numpy(result.P))

    import warnings
    warnings.warn(
        'passing raw position history to plot_optics is deprecated; pass a '
        'RayTraceResult instead.',
        DeprecationWarning,
        stacklevel=3,
    )
    return np.asarray(array_to_true_numpy(result))


def _gather_inputs_for_surface_sag(surf, phist, j, points, y):
    if surf.bounding is None:
        # need to look at the raytrace to see bounding limits
        p = phist[j + 1]
        xx = p[..., 0]
        yy = p[..., 1]
        mask = []
        if y == 'y':
            ymin = yy.min()
            ymax = yy.max()
            ypt = np.linspace(ymin, ymax, points)
            ploty = ypt
            xpt = np.zeros_like(ypt)
        else:
            xmin = xx.min()
            xmax = xx.max()
            xpt = np.linspace(xmin, xmax, points)
            ploty = xpt
            ypt = np.zeros_like(xpt)
    else:
        bound = surf.bounding
        mx = bound['outer_radius']
        r = np.linspace(-mx, mx, points)
        mn = bound.get('inner_radius', 0)
        ar = abs(r)
        mask = ar < mn
        ploty = r
        if y == 'y':
            ypt = r
            xpt = np.zeros_like(r)
        else:
            xpt = r
            ypt = np.zeros_like(r)

    return xpt, ypt, mask, ploty


def _axis_extent_from_phist(phist, j, y):
    p = phist[j + 1]
    axis = 1 if y == 'y' else 0
    coord = p[..., axis]
    return max(abs(np.nanmin(coord)), abs(np.nanmax(coord)))


def _surface_profile(surf, phist, j, points, y, radius=None, clear_radius=None):
    if radius is None:
        xpt, ypt, mask, ploty = _gather_inputs_for_surface_sag(surf, phist, j, points, y)
    else:
        r = np.linspace(-radius, radius, points)
        ploty = r
        bound = surf.bounding or {}
        mn = bound.get('inner_radius', 0)
        mask = abs(r) < mn
        if y == 'y':
            xpt = np.zeros_like(r)
            ypt = r
        else:
            xpt = r
            ypt = np.zeros_like(r)

    sag = surf.F(xpt, ypt)
    sag = np.asarray(sag, dtype=float) + surf.P[2]
    edge_sag = sag.copy()
    mask = np.asarray(mask)
    if mask.size == 0:
        mask = np.zeros_like(ploty, dtype=bool)
    if clear_radius is not None:
        mask = mask | (abs(ploty) > clear_radius)
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


def _surface_index(surf, wvl):
    if not callable(surf.n):
        raise ValueError('refracting surfaces must define a callable material')
    n = surf.n(wvl)
    if np.isscalar(n):
        return float(n)
    n = np.asarray(array_to_true_numpy(n))
    if n.size != 1:
        raise ValueError('material evaluation must produce a scalar index')
    return float(n.reshape(-1)[0])


def _is_ambient_index(n, ambient_index, index_atol):
    return np.isclose(n, ambient_index, rtol=0, atol=index_atol)


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

        n_post = _surface_index(surf, wvl)
        active.append(j)
        if _is_ambient_index(n_post, ambient_index, index_atol):
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


def _infer_lens_group_od(prescription, group, phist, y, lens_edge=None):
    if lens_edge is not None:
        for key in ('od_radius', 'outer_radius', 'radius'):
            if key in lens_edge:
                return lens_edge[key]

    radii = []
    for j in group:
        surf = prescription[j]
        if surf.bounding is not None and 'outer_radius' in surf.bounding:
            radii.append(surf.bounding['outer_radius'])

    if radii:
        return max(radii)

    return max(_axis_extent_from_phist(phist, j, y) for j in group)


def _lens_edge_features(lens_edge):
    if lens_edge is None:
        return []
    features = lens_edge.get('features', [])
    if isinstance(features, Mapping):
        features = [features]
    return features


def _surface_clear_radius(lens_edge, which):
    if lens_edge is None:
        return None
    specific_key = f'clear_radius_{which}'
    if specific_key in lens_edge:
        return lens_edge[specific_key]
    return lens_edge.get('clear_radius')


def _group_surface_clear_radius(lens_edge, surface_number, group_size):
    if surface_number == 0:
        return _surface_clear_radius(lens_edge, 'front')
    if surface_number == group_size - 1:
        return _surface_clear_radius(lens_edge, 'rear')
    return _surface_clear_radius(lens_edge, f'surface_{surface_number}')


def _feature_applies_to_side(feature, side):
    target = feature.get('side', 'both').lower()
    return target in ('both', side)


def _inset_y(outer_y, depth):
    if outer_y < 0:
        return outer_y + depth
    return outer_y - depth


def _append_wall_point(xs, ys, x, y):
    if xs and xs[-1] == x and ys[-1] == y:
        return
    xs.append(x)
    ys.append(y)


def _feature_interval(feature, x0, x1, endpoint_names):
    kind = feature.get('kind', feature.get('type', 'square')).lower()
    if kind in ('square_cut', 'flat', 'chamfer'):
        if 'z_start' not in feature or 'z_end' not in feature:
            raise ValueError(f'{kind} lens edge features require z_start and z_end')
        return feature['z_start'], feature['z_end']

    if kind == 'seat':
        if 'width' not in feature:
            raise ValueError('seat lens edge features require width')
        face = feature.get('face', endpoint_names[0]).lower()
        width = feature['width']
        if face == endpoint_names[0]:
            return x0, x0 + np.sign(x1 - x0) * width
        if face == endpoint_names[1]:
            return x1 - np.sign(x1 - x0) * width, x1
        raise ValueError('seat face must name one wall endpoint')

    if kind == 'square':
        return None

    raise ValueError(f'unknown lens edge feature kind {kind!r}')


def _wall_path(x0, x1, outer_y, features, side, endpoint_names):
    xs = [x0]
    ys = [outer_y]
    direction = np.sign(x1 - x0) or 1
    current = x0
    spans = []

    for feature in features:
        if not _feature_applies_to_side(feature, side):
            continue
        interval = _feature_interval(feature, x0, x1, endpoint_names)
        if interval is None:
            continue
        start, end = interval
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
        inset = _inset_y(outer_y, depth)
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


def _build_lens_outline(sag1, ploty1, edge_sag1, sag2, ploty2, edge_sag2,
                        od_radius, features):
    front_bottom_z = edge_sag1[0]
    front_top_z = edge_sag1[-1]
    rear_bottom_z = edge_sag2[0]
    rear_top_z = edge_sag2[-1]

    top_x, top_y = _wall_path(
        front_top_z, rear_top_z, od_radius, features, 'upper', ('front', 'rear'))
    bottom_x, bottom_y = _wall_path(
        rear_bottom_z, front_bottom_z, -od_radius, features, 'lower', ('rear', 'front'))

    xx = [*sag1, *top_x[1:], *sag2[::-1], *bottom_x[1:]]
    yy = [*ploty1, *top_y[1:], *ploty2[::-1], *bottom_y[1:]]
    return xx, yy


def _apply_lens_edge_features(sag1, ploty1, edge_sag1, sag2, ploty2, edge_sag2,
                              od_radius, lens_edge):
    features = _lens_edge_features(lens_edge)
    return _build_lens_outline(sag1, ploty1, edge_sag1, sag2, ploty2, edge_sag2, od_radius, features)


def _build_lens_group_outline(profiles, od_radius, lens_edge):
    sag1, ploty1, edge_sag1 = profiles[0]
    sag2, ploty2, edge_sag2 = profiles[-1]

    xx, yy = _apply_lens_edge_features(
        sag1, ploty1, edge_sag1, sag2, ploty2, edge_sag2, od_radius, lens_edge)

    for sag, ploty, _ in profiles[1:-1]:
        xx.extend([np.nan, *sag])
        yy.extend([np.nan, *ploty])

    return xx, yy


def _plot_lens_group(prescription, group, phist, points, y, lens_edge,
                     lw, ls, c, alpha, zorder, ax):
    od_radius = _infer_lens_group_od(prescription, group, phist, y, lens_edge)
    profiles = []
    group_size = len(group)
    for surface_number, j in enumerate(group):
        clear_radius = _group_surface_clear_radius(lens_edge, surface_number, group_size)
        profiles.append(_surface_profile(
            prescription[j], phist, j, points, y, radius=od_radius,
            clear_radius=clear_radius,
        ))

    xx, yy = _build_lens_group_outline(profiles, od_radius, lens_edge)
    ax.plot(xx, yy, c=c, lw=lw, ls=ls, alpha=alpha, zorder=zorder)


def plot_optics(prescription, result, *, wvl=0.587, ambient_index=1.0,
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
    ambient_index : float, optional
        Refractive index that closes a physical lens group.
    index_atol : float, optional
        Absolute tolerance for comparing material index to ambient_index.
    mirror_backing : TODO
        TODO
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
    x = x.lower()
    y = y.lower()
    fig, ax = share_fig_ax(fig, ax)
    phist = _phist_from_result_or_history(result)

    lens_groups = lens_groups_from_surfaces(
        prescription, wvl=wvl, ambient_index=ambient_index,
        index_atol=index_atol,
    )
    groups_by_start = {group[0]: (group_index, group)
                       for group_index, group in enumerate(lens_groups)}

    j = 0
    jj = len(prescription)
    while j < jj:
        surf = prescription[j]
        if surf.typ == STYPE_REFLECT:
            sag, ploty, _ = _surface_profile(surf, phist, j, points, y)
            # TODO: mirror backing
            ax.plot(sag, ploty, c=c, lw=lw, ls=ls, alpha=alpha, zorder=zorder)
            j += 1
        elif surf.typ == STYPE_REFRACT:
            group_index, group = groups_by_start[j]
            lens_edge = _lens_edge_for(lens_edges, j, group_index)
            _plot_lens_group(
                prescription, group, phist, points, y, lens_edge,
                lw, ls, c, alpha, zorder, ax,
            )
            j = group[-1] + 1
        else:
            j += 1

    return fig, ax


def plot_transverse_ray_aberration(phist, lw=1, ls='-', c='r', alpha=1, zorder=4, axis='y', fig=None, ax=None):
    """Plot the transverse ray aberration for a single ray fan.

    Parameters
    ----------
    phist : list or ndarray
        the first return from spencer_and_murty.raytrace,
        iterable of arrays of length 3 (X,Y,Z)
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

    Returns
    -------
    matplotlib.figure.Figure
        A figure object
    matplotlib.axes.Axis
        An axis object

    """
    fig, ax = share_fig_ax(fig, ax)

    ph = np.asarray(array_to_true_numpy(phist))
    sieve = {
        'x': 0,
        'y': 1,
    }
    axis = axis.lower()
    axis = sieve[axis]
    input_rays = ph[0, ..., axis]
    output_rays = ph[-1, ..., axis]
    ax.plot(input_rays, output_rays, c=c, lw=lw, ls=ls, alpha=alpha, zorder=zorder)
    return fig, ax


def _path_length_unit_label(units):
    units = units.lower()
    if units in ('wave', 'waves'):
        return 'waves'
    if units in ('nm', 'nanometer', 'nanometers'):
        return 'nm'
    raise ValueError("units must be 'waves' or 'nm'")


def _convert_path_length_units(opd, wavelength, units):
    units = _path_length_unit_label(units)
    if units == 'waves':
        if wavelength is None:
            raise ValueError('wavelength is required when units="waves"')
        return opd / float(wavelength), 'OPD [waves]'
    return opd * 1e3, 'OPD [nm]'


def _remove_linear_fit(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(valid) < 2:
        return y
    slope, intercept = np.polyfit(x[valid], y[valid], 1)
    return y - (slope * x + intercept)


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
    opd, ylabel = _convert_path_length_units(opd, wavelength, units)
    if detrend:
        opd = _remove_linear_fit(coord, opd)
    ax.plot(coord, opd, c=c, lw=lw, ls=ls, alpha=alpha, zorder=zorder,
            label=label)
    ax.set_xlabel(f'normalized pupil {axis}')
    ax.set_ylabel(ylabel)
    return fig, ax


def plot_wave_aberration(*args, **kwargs):
    """Deprecated alias for plot_transverse_ray_aberration.

    The original implementation never computed wave aberration / OPD; it
    plotted input vs output ray position, identical to
    plot_transverse_ray_aberration.  Use that function directly.  A real
    wave-aberration plot belongs downstream of opt.opd_from_raytrace.

    """
    import warnings
    warnings.warn(
        'plot_wave_aberration is a misnamed alias for '
        'plot_transverse_ray_aberration and will be removed in a future '
        'release; use plot_transverse_ray_aberration directly.',
        DeprecationWarning,
        stacklevel=2,
    )
    return plot_transverse_ray_aberration(*args, **kwargs)


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
