"""Tests for raytracing plotting helpers."""
import matplotlib
import numpy as np
import pytest

matplotlib.use('Agg')

from matplotlib import pyplot as plt

from prysm.x.raytracing.plotting import plot_optics
from prysm.x.raytracing.surfaces import Surface


def _plane_ffp(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    shape = np.broadcast_shapes(x.shape, y.shape)
    zero = np.zeros(shape, dtype=float)
    return zero, zero, zero


def _plane_F(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    shape = np.broadcast_shapes(x.shape, y.shape)
    return np.zeros(shape, dtype=float)


def _refracting_plane(z, outer_radius=1, inner_radius=None):
    bounding = {'outer_radius': outer_radius}
    if inner_radius is not None:
        bounding['inner_radius'] = inner_radius

    return Surface(
        typ='refr',
        P=np.asarray([0., 0., z]),
        n=lambda wvl: 1.5,
        FFp=_plane_ffp,
        F=_plane_F,
        bounding=bounding,
    )


def _phist():
    return np.asarray([
        [[0., -1., 0.], [0., 0., 0.], [0., 1., 0.]],
        [[0., -1., 2.], [0., 0., 2.], [0., 1., 2.]],
    ])


def _line_from_plot(prescription, **kwargs):
    fig, ax = plot_optics(prescription, _phist(), points=5, **kwargs)
    try:
        line = ax.lines[0]
        return line.get_xdata(), line.get_ydata()
    finally:
        plt.close(fig)


def test_plot_optics_default_lens_od_is_square():
    x, y = _line_from_plot([_refracting_plane(0), _refracting_plane(2)])

    np.testing.assert_allclose(y[:5], np.linspace(-1, 1, 5))
    assert np.any((y[:-1] == 1) & (y[1:] == 1) & (x[:-1] == 0) & (x[1:] == 2))
    assert np.any((y[:-1] == -1) & (y[1:] == -1) & (x[:-1] == 2) & (x[1:] == 0))


def test_plot_optics_infers_larger_paired_surface_od():
    _, y = _line_from_plot([_refracting_plane(0, outer_radius=1),
                            _refracting_plane(2, outer_radius=1.5)])

    assert y.max() == 1.5
    assert y.min() == -1.5


def test_plot_optics_can_mask_clear_aperture_inside_mechanical_od():
    edges = {0: {'od_radius': 1.5, 'clear_radius': 1}}
    x, y = _line_from_plot([_refracting_plane(0), _refracting_plane(2)], lens_edges=edges)

    assert y.max() == 1.5
    assert y.min() == -1.5
    assert np.isnan(x[0])
    assert np.isnan(x[4])
    assert not np.isnan(x[1:4]).any()


def test_plot_optics_keeps_inner_radius_mask_on_lenses():
    x, _ = _line_from_plot([_refracting_plane(0, inner_radius=0.5),
                            _refracting_plane(2, inner_radius=0.5)])

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
    x, y = _line_from_plot([_refracting_plane(0), _refracting_plane(2)], lens_edges=edges)

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
    x, y = _line_from_plot([_refracting_plane(0), _refracting_plane(2)], lens_edges=edges)

    np.testing.assert_allclose(x[5:9], [0.0, 0.5, 0.5, 2.0])
    np.testing.assert_allclose(y[5:9], [0.8, 0.8, 1.0, 1.0])


def test_plot_optics_flat_and_chamfer_features_render_named_segments():
    flat_edges = {0: {'features': [{'kind': 'flat', 'side': 'upper',
                                    'z_start': 0.5, 'z_end': 1.5, 'depth': 0.25}]}}
    chamfer_edges = {0: {'features': [{'kind': 'chamfer', 'side': 'upper',
                                       'z_start': 0.5, 'z_end': 1, 'depth': 0.2}]}}

    x, y = _line_from_plot([_refracting_plane(0), _refracting_plane(2)], lens_edges=flat_edges)
    np.testing.assert_allclose(x[5:10], [0.5, 0.5, 1.5, 1.5, 2.0])
    np.testing.assert_allclose(y[5:10], [1.0, 0.75, 0.75, 1.0, 1.0])

    x, y = _line_from_plot([_refracting_plane(0), _refracting_plane(2)], lens_edges=chamfer_edges)
    np.testing.assert_allclose(x[5:9], [0.5, 1.0, 1.0, 2.0])
    np.testing.assert_allclose(y[5:9], [1.0, 0.8, 1.0, 1.0])


def test_plot_optics_still_rejects_terminal_refracting_surface():
    with pytest.raises(ValueError, match='terminates on a refracting surface'):
        _line_from_plot([_refracting_plane(0)])
