"""OpticalSystem convenience plot methods (layout_2d, plot_spots,
plot_ray_fans, plot_opd_fans, plot_field_curvature, plot_distortion,
plot_chromatic_focal_shift, plot_axial_color, plot_lateral_color) and the
live-fingerprint trace cache."""
import matplotlib
import numpy as np
import pytest

matplotlib.use('Agg')
from matplotlib import pyplot as plt

from prysm.x.raytracing import (
    OpticalSystem, ApertureSpec, LensData, Sphere, Plane, Field, Sampling,
)
from prysm.x import materials
from prysm.x.raytracing.analysis import (
    spot_diagrams, ray_aberration_fans, field_curvature, axial_color,
    lateral_color,
)


def _doublet():
    # OBJECT/IMAGE endpoints are implicit (ADR-0006); the first powered surface
    # is row 1, so the stop stays at index 1.
    ld = (LensData()
          .add(Sphere(1 / 61.47), thickness=6.0,
               material=materials.ConstantMaterial(1.5168), semidiameter=12.0)
          .add(Sphere(-1 / 44.64), thickness=2.5,
               material=materials.ConstantMaterial(1.673), semidiameter=12.0)
          .add(Sphere(-1 / 129.94), thickness=0.0,
               material=materials.air, semidiameter=12.0))
    sys = OpticalSystem(ld, aperture=ApertureSpec.epd(22.0),
                        fields=[Field(0, 0), Field(0, 0.7), Field(0, 1.0)],
                        wavelengths=[0.486, 0.587, 0.656], reference=1,
                        stop_index=1)
    sys.solve.image_distance()
    return sys


# ---------- layout_2d -------------------------------------------------------

def test_layout_2d_returns_fig_ax_and_draws_field_fans():
    sys = _doublet()
    fig, ax = sys.plot.layout_2d()
    try:
        assert fig is not None and ax is not None
        # one optics outline plus a fan per field -> more lines than fields
        assert len(ax.lines) > len(sys.fields)
    finally:
        plt.close(fig)


def test_layout_2d_honors_overrides():
    sys = _doublet()
    fig, ax = sys.plot.layout_2d(fields=[Field(0, 0)], sampling=5, axis='y')
    try:
        # a 5-ray single-field fan: 5 ray lines + 1 optics outline
        assert len(ax.lines) == 6
    finally:
        plt.close(fig)


def test_layout_2d_accepts_explicit_sampling_object():
    sys = _doublet()
    fig, ax = sys.plot.layout_2d(sampling=Sampling.fan(n=3, axis='y'))
    try:
        assert len(ax.lines) > 0
    finally:
        plt.close(fig)


# ---------- convenience plot methods ----------------------------------------

def test_plot_spots_returns_fig_axs():
    sys = _doublet()
    fig, axs = sys.plot.spots()
    try:
        assert np.asarray(axs).size == len(sys.fields)
    finally:
        plt.close(fig)


def test_plot_ray_fans_and_opd_fans_return_fig_axs():
    sys = _doublet()
    for method in (sys.plot.ray_fans, sys.plot.opd_fans):
        fig, axs = method()
        try:
            assert np.asarray(axs).shape == (len(sys.fields), 2)
        finally:
            plt.close(fig)


def test_convenience_grid_equals_explicit_two_step():
    sys = _doublet()
    explicit = spot_diagrams(sys)
    cached = sys._cached_grid('spots', spot_diagrams, dict(
        fields=None, wavelengths=None, sampling=None, epd=None,
        reference='centroid'))
    np.testing.assert_allclose(cached.x, explicit.x, equal_nan=True)
    np.testing.assert_allclose(cached.y, explicit.y, equal_nan=True)


# ---------- curve plot methods ----------------------------------------------

def test_plot_field_curvature_defaults_to_dense_field_sweep():
    sys = _doublet()
    fig, ax = sys.plot.field_curvature(samples=33)
    try:
        assert len(ax.lines) == 2  # x and y fans
        y = ax.lines[0].get_ydata()
        assert len(y) == 33
        # the sweep spans the system field magnitudes
        assert y[0] == pytest.approx(0.0)
        assert y[-1] == pytest.approx(1.0)
    finally:
        plt.close(fig)


def test_plot_field_curvature_explicit_fields_verbatim():
    sys = _doublet()
    fig, ax = sys.plot.field_curvature(fields=list(sys.fields))
    try:
        assert len(ax.lines[0].get_ydata()) == len(sys.fields)
    finally:
        plt.close(fig)


def test_plot_distortion_defaults_to_dense_field_sweep():
    sys = _doublet()
    fig, ax = sys.plot.distortion(samples=33)
    try:
        assert len(ax.lines) == 1
        assert len(ax.lines[0].get_xdata()) == 33
    finally:
        plt.close(fig)


def test_plot_chromatic_focal_shift_spans_system_wavelengths():
    sys = _doublet()
    fig, ax = sys.plot.chromatic_focal_shift(focus='paraxial', samples=7)
    try:
        x = ax.lines[0].get_xdata()
        assert len(x) == 7
        assert x[0] == pytest.approx(min(sys.wavelengths))
        assert x[-1] == pytest.approx(max(sys.wavelengths))
    finally:
        plt.close(fig)


def test_plot_axial_color_zero_at_reference_wavelength():
    sys = _doublet()
    fig, ax = sys.plot.axial_color()
    try:
        line = ax.lines[0]
        x = line.get_xdata()
        y = line.get_ydata()
        assert len(x) == len(sys.wavelengths)
        assert y[sys.reference] == pytest.approx(0.0)
        # crown-flint doublet: the shift matches the paraxial BFD differences
        bfd = axial_color(sys)
        np.testing.assert_allclose(y, bfd - bfd[sys.reference])
    finally:
        plt.close(fig)


def test_plot_lateral_color_one_curve_per_nonreference_wavelength():
    sys = _doublet()
    fig, ax = sys.plot.lateral_color()
    try:
        assert len(ax.lines) == len(sys.wavelengths) - 1
        # default dense field sweep, matching the analysis default
        landing = lateral_color(sys)
        assert len(ax.lines[0].get_ydata()) == landing.shape[0]
        # pure +y fields: the signed projection is the chief-ray y difference
        expected = landing[:, 0, 1] - landing[:, sys.reference, 1]
        np.testing.assert_allclose(ax.lines[0].get_xdata(), expected,
                                   atol=1e-12)
        # the on-axis field has zero lateral color
        assert ax.lines[0].get_xdata()[0] == pytest.approx(0.0)
    finally:
        plt.close(fig)


def test_curve_convenience_data_cached_and_matches_explicit():
    sys = _doublet()
    explicit = field_curvature(sys)
    kw = dict(fields=None, wavelength=None, samples=101)
    cached = sys._cached_grid('field_curvature', field_curvature, kw)
    np.testing.assert_allclose(cached.x_fan_z, explicit.x_fan_z)
    np.testing.assert_allclose(cached.y_fan_z, explicit.y_fan_z)
    assert sys._cached_grid('field_curvature', field_curvature, kw) is cached


# ---------- the fingerprint trace cache -------------------------------------

def test_reset_raytrace_cache_clears_caches_and_resets_version():
    sys = _doublet()
    wvl = sys.wavelength()
    grid_kw = dict(fields=None, wavelengths=None, nrays=11, epd=None,
                   distribution='uniform', reference='chief')

    P_xp = sys.exit_pupil(wvl)
    grid = sys._cached_grid('ray_fans', ray_aberration_fans, grid_kw)
    sys.lens.to_surfaces()
    assert sys._derived
    assert sys._trace_cache
    assert sys.lens._surfaces_cache is not None

    sys.lens.rows[1].thickness = 6.5
    assert sys.lens._version > 0

    out = sys.reset_raytrace_cache()
    assert out is sys
    assert sys.lens._version == 0
    assert sys.lens._surfaces_cache is None
    assert sys._derived == {}
    assert sys._trace_cache == {}

    assert sys.exit_pupil(wvl) is not P_xp
    assert sys._cached_grid('ray_fans', ray_aberration_fans, grid_kw) is not grid


def test_trace_cache_hits_then_invalidates_on_every_change():
    sys = _doublet()
    kw = dict(fields=None, wavelengths=None, nrays=11, epd=None,
              distribution='uniform', reference='chief')

    g1 = sys._cached_grid('ray_fans', ray_aberration_fans, kw)
    g2 = sys._cached_grid('ray_fans', ray_aberration_fans, kw)
    assert g1 is g2  # unchanged system -> cache hit (same object)

    # a lens edit bumps lens._version -> miss
    sys.lens.rows[1].thickness = 6.5
    g3 = sys._cached_grid('ray_fans', ray_aberration_fans, kw)
    assert g3 is not g2

    # a stop_index reassignment (no version bump) -> miss via live fingerprint
    sys.stop_index = 0
    g4 = sys._cached_grid('ray_fans', ray_aberration_fans, kw)
    assert g4 is not g3
    sys.stop_index = 1

    # an in-place fields mutation -> miss (a version counter would not catch it)
    g5 = sys._cached_grid('ray_fans', ray_aberration_fans, kw)
    sys.fields.fields.append(Field(0, 1.4))
    g6 = sys._cached_grid('ray_fans', ray_aberration_fans, kw)
    assert g6 is not g5

    # a wavelength-set reassignment -> miss
    sys.wavelengths = np.asarray([0.55])
    g7 = sys._cached_grid('ray_fans', ray_aberration_fans, kw)
    assert g7 is not g6

    # an in-place field vignetting mutation (set_vignetting) -> miss
    sys.fields.fields[0].vignetting = {'vux': 0.0, 'vlx': 0.0,
                                       'vuy': 0.1, 'vly': 0.1}
    g8 = sys._cached_grid('ray_fans', ray_aberration_fans, kw)
    assert g8 is not g7


def test_trace_cache_distinguishes_call_arguments():
    sys = _doublet()
    a = sys._cached_grid('ray_fans', ray_aberration_fans, dict(nrays=11))
    b = sys._cached_grid('ray_fans', ray_aberration_fans, dict(nrays=21))
    assert a is not b
    assert a.x.shape[-1] == 11 and b.x.shape[-1] == 21


def test_plot_full_field_draws_metric_map():
    sys = _doublet()
    fig, ax = sys.plot.full_field(samples=5)
    try:
        assert len(ax.collections) == 1  # the pcolormesh
        data = ax.collections[0].get_array()
        assert np.isfinite(np.asarray(data)).any()
        assert ax.get_xlabel() == 'field x [deg]'
        # the grid is cached on the system fingerprint
        from prysm.x.raytracing.analysis import full_field
        kw = dict(metric='rms spot', samples=5, max_field=None,
                  wavelengths=None, sampling=None, epd=None, stop_index=None)
        g1 = sys._cached_grid('full_field', full_field, kw)
        g2 = sys._cached_grid('full_field', full_field, kw)
        assert g1 is g2
    finally:
        plt.close(fig)
