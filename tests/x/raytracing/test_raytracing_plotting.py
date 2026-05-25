"""Tests for raytracing plotting helpers."""
import matplotlib
import numpy as np
import pytest

matplotlib.use('Agg')

from matplotlib import pyplot as plt

from prysm.x.raytracing.plotting import (
    lens_groups_from_surfaces,
    mirror_substrate_outline,
    plot_optics,
    plot_ray_paths,
    plot_wave_aberration_fan,
)
from prysm.x.raytracing.spencer_and_murty import RayTraceResult
from prysm.x.raytracing.surfaces import OffAxisConicSag, PlaneSag, Surface


def _refracting_plane(z, outer_radius=1, inner_radius=None, n=1.0):
    if outer_radius is None:
        bounding = None
    else:
        bounding = {'outer_radius': outer_radius}
    if inner_radius is not None:
        bounding['inner_radius'] = inner_radius

    return Surface(
        shape=PlaneSag(),
        typ='refr',
        P=np.asarray([0., 0., z]),
        n=lambda wvl: n,
        bounding=bounding,
    )


def _reflecting_surface(shape, z=0, outer_radius=1, inner_radius=None):
    if outer_radius is None:
        bounding = None
    else:
        bounding = {'outer_radius': outer_radius}
    if inner_radius is not None:
        bounding['inner_radius'] = inner_radius

    return Surface(
        shape=shape,
        typ='refl',
        P=np.asarray([0., 0., z]),
        bounding=bounding,
    )


def _surface_points(z):
    return np.asarray([
        [0., -1., z],
        [0., 0., z],
        [0., 1., z],
    ])


def _trace_result(prescription):
    z_history = [prescription[0].P[2] - 1]
    z_history.extend(surf.P[2] for surf in prescription)
    P = np.asarray([_surface_points(z) for z in z_history])
    S = np.zeros_like(P)
    OPL = np.zeros(P.shape[:-1])
    status = np.zeros(P.shape[1], dtype=np.complex128)
    return RayTraceResult(P, S, OPL, status)


def _raytrace_result():
    return np.asarray([
        [[0., -1., 0.], [0., 0., 0.], [0., 1., 0.]],
        [[0., -1., 2.], [0., 0., 2.], [0., 1., 2.]],
    ])


def _line_from_plot(prescription, **kwargs):
    fig, ax = plot_optics(prescription, _trace_result(prescription), points=5, **kwargs)
    try:
        line = ax.lines[0]
        return line.get_xdata(), line.get_ydata()
    finally:
        plt.close(fig)


def test_plot_optics_default_lens_od_is_square():
    x, y = _line_from_plot([_refracting_plane(0, n=1.5),
                            _refracting_plane(2, n=1.0)])

    np.testing.assert_allclose(y[:5], np.linspace(-1, 1, 5))
    assert np.any((y[:-1] == 1) & (y[1:] == 1) & (x[:-1] == 0) & (x[1:] == 2))
    assert np.any((y[:-1] == -1) & (y[1:] == -1) & (x[:-1] == 2) & (x[1:] == 0))


def test_plot_optics_infers_larger_paired_surface_od():
    _, y = _line_from_plot([_refracting_plane(0, outer_radius=1, n=1.5),
                            _refracting_plane(2, outer_radius=1.5, n=1.0)])

    assert y.max() == 1.5
    assert y.min() == -1.5


def test_plot_optics_can_mask_clear_aperture_inside_mechanical_od():
    edges = {0: {'od_radius': 1.5, 'clear_radius': 1}}
    x, y = _line_from_plot([_refracting_plane(0, n=1.5),
                            _refracting_plane(2, n=1.0)], lens_edges=edges)

    assert y.max() == 1.5
    assert y.min() == -1.5
    assert np.isnan(x[0])
    assert np.isnan(x[4])
    assert not np.isnan(x[1:4]).any()


def test_plot_optics_keeps_inner_radius_mask_on_lenses():
    x, _ = _line_from_plot([_refracting_plane(0, inner_radius=0.5, n=1.5),
                            _refracting_plane(2, inner_radius=0.5, n=1.0)])

    assert np.isnan(x).any()


def test_plot_optics_square_cut_feature_insets_wall():
    edges = {
        0: {
            'features': [
                {'kind': 'square_cut', 'side': 'upper',
                 'z_start': 0.5, 'z_end': 1.5, 'depth': 0.25},
            ],
        },
    }
    x, y = _line_from_plot([_refracting_plane(0, n=1.5),
                            _refracting_plane(2, n=1.0)], lens_edges=edges)

    np.testing.assert_allclose(x[5:10], [0.5, 0.5, 1.5, 1.5, 2.0])
    np.testing.assert_allclose(y[5:10], [1.0, 0.75, 0.75, 1.0, 1.0])


def test_plot_optics_seat_feature_steps_from_named_face():
    edges = [
        {
            'features': [
                {'kind': 'seat', 'side': 'upper',
                 'face': 'front', 'width': 0.5, 'depth': 0.2},
            ],
        },
    ]
    x, y = _line_from_plot([_refracting_plane(0, n=1.5),
                            _refracting_plane(2, n=1.0)], lens_edges=edges)

    np.testing.assert_allclose(x[5:9], [0.0, 0.5, 0.5, 2.0])
    np.testing.assert_allclose(y[5:9], [0.8, 0.8, 1.0, 1.0])


def test_plot_optics_flat_and_chamfer_features_render_named_segments():
    flat_edges = {0: {'features': [{'kind': 'flat', 'side': 'upper',
                                    'z_start': 0.5, 'z_end': 1.5, 'depth': 0.25}]}}
    chamfer_edges = {0: {'features': [{'kind': 'chamfer', 'side': 'upper',
                                       'z_start': 0.5, 'z_end': 1, 'depth': 0.2}]}}

    prescription = [_refracting_plane(0, n=1.5), _refracting_plane(2, n=1.0)]

    x, y = _line_from_plot(prescription, lens_edges=flat_edges)
    np.testing.assert_allclose(x[5:10], [0.5, 0.5, 1.5, 1.5, 2.0])
    np.testing.assert_allclose(y[5:10], [1.0, 0.75, 0.75, 1.0, 1.0])

    x, y = _line_from_plot(prescription, lens_edges=chamfer_edges)
    np.testing.assert_allclose(x[5:9], [0.5, 1.0, 1.0, 2.0])
    np.testing.assert_allclose(y[5:9], [1.0, 0.8, 1.0, 1.0])


def test_plot_optics_still_rejects_terminal_refracting_surface():
    with pytest.raises(ValueError, match='terminates'):
        _line_from_plot([_refracting_plane(0, n=1.5)])


def test_plot_ray_paths_uses_raytrace_result_positions():
    P = _raytrace_result()
    result = RayTraceResult(P, np.zeros_like(P), np.zeros(P.shape[:-1]),
                            np.zeros(P.shape[1], dtype=np.complex128))

    fig, ax = plot_ray_paths(result)
    try:
        for ray_index, line in enumerate(ax.lines):
            np.testing.assert_allclose(line.get_xdata(), P[:, ray_index, 2])
            np.testing.assert_allclose(line.get_ydata(), P[:, ray_index, 1])
    finally:
        plt.close(fig)


def test_plot_wave_aberration_fan_can_use_nm():
    coord = np.asarray([-1., 0., 1.])
    opd = np.asarray([-0.001, 0., 0.001])

    fig, ax = plot_wave_aberration_fan(coord, opd, units='nm',
                                       detrend=False)
    try:
        line = ax.lines[0]
        np.testing.assert_allclose(line.get_ydata(), [-1., 0., 1.])
        assert ax.get_ylabel() == 'OPD [nm]'
    finally:
        plt.close(fig)


def test_plot_wave_aberration_fan_can_remove_linear_fit():
    coord = np.asarray([-1., 0., 1.])
    opd = 0.5 * coord + 0.125 * coord * coord + 0.25

    fig, ax = plot_wave_aberration_fan(
        coord, opd, wavelength=1, detrend=True,
    )
    try:
        line = ax.lines[0]
        np.testing.assert_allclose(line.get_ydata(), [1 / 24, -1 / 12, 1 / 24])
    finally:
        plt.close(fig)


def test_plot_wave_aberration_fan_detrends_by_default():
    coord = np.asarray([-1., 0., 1.])
    opd = 0.5 * coord + 0.125 * coord * coord + 0.25

    fig, ax = plot_wave_aberration_fan(coord, opd, wavelength=1)
    try:
        line = ax.lines[0]
        np.testing.assert_allclose(line.get_ydata(), [1 / 24, -1 / 12, 1 / 24])
    finally:
        plt.close(fig)


def test_plot_wave_aberration_fan_can_keep_linear_fit():
    coord = np.asarray([-1., 0., 1.])
    opd = 0.5 * coord + 0.125 * coord * coord + 0.25

    fig, ax = plot_wave_aberration_fan(
        coord, opd, wavelength=1, detrend=False,
    )
    try:
        line = ax.lines[0]
        np.testing.assert_allclose(line.get_ydata(), opd)
    finally:
        plt.close(fig)


def test_lens_groups_from_surfaces_groups_singlet():
    prescription = [_refracting_plane(0, n=1.5),
                    _refracting_plane(2, n=1.0)]

    assert lens_groups_from_surfaces(prescription) == [(0, 1)]


def test_lens_groups_from_surfaces_groups_cemented_doublet():
    prescription = [_refracting_plane(0, n=1.5),
                    _refracting_plane(1, n=1.6),
                    _refracting_plane(2, n=1.0)]

    assert lens_groups_from_surfaces(prescription) == [(0, 1, 2)]


def test_lens_groups_from_surfaces_groups_cemented_triplet():
    prescription = [_refracting_plane(0, n=1.5),
                    _refracting_plane(1, n=1.6),
                    _refracting_plane(2, n=1.7),
                    _refracting_plane(3, n=1.0)]

    assert lens_groups_from_surfaces(prescription) == [(0, 1, 2, 3)]


def test_lens_groups_from_surfaces_splits_air_spaced_doublet():
    prescription = [_refracting_plane(0, n=1.5),
                    _refracting_plane(1, n=1.0),
                    _refracting_plane(3, n=1.6),
                    _refracting_plane(4, n=1.0)]

    assert lens_groups_from_surfaces(prescription) == [(0, 1), (2, 3)]


def test_lens_groups_from_surfaces_rejects_terminal_group():
    with pytest.raises(ValueError, match='terminates'):
        lens_groups_from_surfaces([_refracting_plane(0, n=1.5),
                                   _refracting_plane(1, n=1.6)])


def test_plot_optics_group_od_uses_largest_aperture_in_group():
    prescription = [_refracting_plane(0, outer_radius=1.0, n=1.5),
                    _refracting_plane(1, outer_radius=2.0, n=1.6),
                    _refracting_plane(2, outer_radius=1.2, n=1.0)]

    _, y = _line_from_plot(prescription)

    assert np.nanmax(y) == 2.0
    assert np.nanmin(y) == -2.0


def test_plot_optics_draws_mirror_optical_surface_by_default():
    prescription = [_reflecting_surface(PlaneSag(), outer_radius=1)]

    x, y = _line_from_plot(prescription)

    np.testing.assert_allclose(x, np.zeros(5))
    np.testing.assert_allclose(y, np.linspace(-1, 1, 5))


def test_plot_optics_draws_parallel_mirror_substrate():
    prescription = [_reflecting_surface(PlaneSag(), outer_radius=1)]
    backing = {0: {'mode': 'parallel', 'thickness': 2, 'side': 1}}

    x, y = _line_from_plot(prescription, mirror_backing=backing)

    np.testing.assert_allclose(x[:5], np.zeros(5))
    assert np.any((y[:-1] == 1) & (y[1:] == 1) & (x[:-1] == 0) & (x[1:] == 2))
    assert np.any((y[:-1] == -1) & (y[1:] == -1) & (x[:-1] == 2) & (x[1:] == 0))
    np.testing.assert_allclose(x[6:11], np.full(5, 2.0))


def test_plot_optics_accepts_global_mirror_backing_config():
    prescription = [_reflecting_surface(PlaneSag(), outer_radius=1)]
    backing = {'mode': 'parallel', 'thickness': 2, 'side': 1}

    x, _ = _line_from_plot(prescription, mirror_backing=backing)

    np.testing.assert_allclose(x[6:11], np.full(5, 2.0))


def test_mirror_substrate_outline_applies_surface_decenter():
    surf = Surface(
        shape=PlaneSag(),
        typ='refl',
        P=np.asarray([0., 10., 5.]),
        bounding={'outer_radius': 1},
    )
    result = _trace_result([surf])

    x, y = mirror_substrate_outline(
        surf, result, backing={'mode': 'parallel', 'thickness': 2, 'side': 1},
        points=5,
    )

    np.testing.assert_allclose(x[:5], np.full(5, 5.0))
    np.testing.assert_allclose(y[:5], np.linspace(9, 11, 5))
    np.testing.assert_allclose(x[6:11], np.full(5, 7.0))


def test_mirror_substrate_outline_can_center_on_ray_footprint():
    surf = Surface(
        shape=PlaneSag(),
        typ='refl',
        P=np.asarray([0., 0., 0.]),
        bounding={'outer_radius': 1},
    )
    P = np.asarray([
        [[0., 9., -1.], [0., 10., -1.], [0., 11., -1.]],
        [[0., 9., 0.], [0., 10., 0.], [0., 11., 0.]],
    ])
    result = RayTraceResult(
        P, np.zeros_like(P), np.zeros(P.shape[:-1]),
        np.zeros(P.shape[1], dtype=np.complex128),
    )

    x, y = mirror_substrate_outline(
        surf, result, backing={
            'mode': 'parallel',
            'thickness': 2,
            'side': 1,
            'center': 'rays',
        },
        points=5,
    )

    np.testing.assert_allclose(x[:5], np.zeros(5))
    np.testing.assert_allclose(y[:5], np.linspace(9, 11, 5))
    np.testing.assert_allclose(x[6:11], np.full(5, 2.0))


def test_mirror_substrate_outline_applies_surface_tilt_in_xz_projection():
    surf = Surface(
        shape=PlaneSag(),
        typ='refl',
        P=np.asarray([0., 0., 0.]),
        R=(0, -45, 0),
        bounding={'outer_radius': 1},
    )
    result = _trace_result([surf])

    x, y = mirror_substrate_outline(
        surf, result, backing={'mode': 'parallel', 'thickness': 2, 'side': 1},
        points=5, x='z', y='x',
    )

    front_x = np.asarray(x[:5])
    front_y = np.asarray(y[:5])
    assert not np.allclose(front_x, front_x[0])
    assert not np.allclose(front_y, front_y[0])
    np.testing.assert_allclose(np.diff(front_x) / np.diff(front_y),
                               np.full(4, -1.0))


def test_mirror_substrate_can_cut_flat_from_parent_vertex_plane():
    surf = _reflecting_surface(
        OffAxisConicSag(c=1 / 100., k=-1., dy=10),
        outer_radius=5,
    )
    result = _trace_result([surf])

    x, _ = mirror_substrate_outline(
        surf, result, backing={'mode': 'flat_parent', 'thickness': 2, 'side': 1},
        points=5,
    )

    np.testing.assert_allclose(x[6:11], np.full(5, 2.0))


def test_mirror_substrate_can_cut_flat_near_aperture_for_uniform_thickness():
    surf = _reflecting_surface(
        OffAxisConicSag(c=1 / 100., k=-1., dy=10),
        outer_radius=5,
    )
    result = _trace_result([surf])

    x, y = mirror_substrate_outline(
        surf, result, backing={'mode': 'flat_aperture', 'thickness': 2, 'side': 1},
        points=5,
    )

    rear_x = np.asarray(x[6:11])
    rear_y = np.asarray(y[6:11])
    slope = np.diff(rear_x) / np.diff(rear_y)
    assert not np.allclose(rear_x, rear_x[0])
    np.testing.assert_allclose(slope, np.full(4, slope[0]))

    front_lower_edge = surf.sag(np.asarray([0.]), np.asarray([-5.]))[0]
    rear_lower_edge = rear_x[rear_y == -5][0]
    np.testing.assert_allclose(rear_lower_edge - front_lower_edge, 2.0)
