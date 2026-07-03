"""Tests for raytracing plotting helpers."""
import matplotlib
import numpy as np
import pytest
from prysm.x import materials
from prysm.x.raytracing import OpticalSystem

matplotlib.use('Agg')

from matplotlib import pyplot as plt

from prysm.x.raytracing.plotting import (
    mirror_substrate_outline,
    plot_chromatic_focal_shift,
    plot_distortion,
    plot_field_curvature,
    plot_optics,
    plot_ray_paths,
    plot_spot_diagram,
    plot_transverse_ray_aberration,
    plot_wave_aberration_fan,
)
from prysm.x.raytracing.lensdata import LensData, lens_element_groups
from prysm.x.raytracing.launch import Field
from prysm.x.raytracing.spencer_and_murty import RayTraceResult
from prysm.x.raytracing.surfaces import Conic, OffAxisConic, Plane, Surface
from prysm.x.raytracing.aperture import (
    Aperture,
    Chamfer,
    CircularExtent,
    Flat,
    FlatBackSubstrate,
    FlatParentSubstrate,
    ParallelSubstrate,
    Seat,
    SquareCut,
)


def _extent(outer_radius, inner_radius=None):
    """A drawn-only Aperture (no clip), mirroring the old bounding dict."""
    if outer_radius is None:
        return None
    inner = 0.0 if inner_radius is None else inner_radius
    return Aperture(extent=CircularExtent(outer_radius, inner_radius=inner))


def _singlet_lensdata():
    """A simple constant-index biconvex singlet with a 3-point field set."""
    n15 = materials.ConstantMaterial(1.5)
    air = materials.air
    lens = LensData()
    (lens.add(Conic(1 / 60.0, 0.0), thickness=4.0, material=n15,
              aperture=8.0)
         .add(Conic(-1 / 60.0, 0.0), thickness=95.0, material=air,
              aperture=8.0))
    ld = OpticalSystem(lens, aperture=10.0, fields=[0.0, 3.0, 5.0],
                       wavelengths=[0.5876], reference=0)
    ld.solve.image_distance()
    return ld


def _refracting_plane(z, outer_radius=1, inner_radius=None, n=1.0):
    return Surface(
        shape=Plane(),
        interaction='refr',
        P=np.asarray([0., 0., z]),
        material=materials.ConstantMaterial(n),
        aperture=_extent(outer_radius, inner_radius),
    )


def _reflecting_surface(shape, z=0, outer_radius=1, inner_radius=None):
    return Surface(
        shape=shape,
        interaction='refl',
        P=np.asarray([0., 0., z]),
        aperture=_extent(outer_radius, inner_radius),
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
    kwargs.setdefault('wvl', 0.55)
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


def _featured_front(*features, z=0, n=1.5, outer_radius=1):
    """A refracting plane whose aperture carries rim features at the given OD."""
    surf = _refracting_plane(z, outer_radius=outer_radius, n=n)
    surf.aperture = Aperture(extent=CircularExtent(outer_radius),
                             features=features)
    return surf


def test_plot_optics_keeps_inner_radius_mask_on_lenses():
    x, _ = _line_from_plot([_refracting_plane(0, inner_radius=0.5, n=1.5),
                            _refracting_plane(2, inner_radius=0.5, n=1.0)])

    assert np.isnan(x).any()


def test_plot_optics_square_cut_feature_insets_wall():
    front = _featured_front(SquareCut(0.5, 1.5, 0.25, side='upper'))
    x, y = _line_from_plot([front, _refracting_plane(2, n=1.0)])

    np.testing.assert_allclose(x[5:10], [0.5, 0.5, 1.5, 1.5, 2.0])
    np.testing.assert_allclose(y[5:10], [1.0, 0.75, 0.75, 1.0, 1.0])


def test_plot_optics_seat_feature_steps_from_named_face():
    front = _featured_front(Seat('front', 0.5, 0.2, side='upper'))
    x, y = _line_from_plot([front, _refracting_plane(2, n=1.0)])

    np.testing.assert_allclose(x[5:9], [0.0, 0.5, 0.5, 2.0])
    np.testing.assert_allclose(y[5:9], [0.8, 0.8, 1.0, 1.0])


def test_plot_optics_flat_and_chamfer_features_render_named_segments():
    front_flat = _featured_front(Flat(0.5, 1.5, 0.25, side='upper'))
    x, y = _line_from_plot([front_flat, _refracting_plane(2, n=1.0)])
    np.testing.assert_allclose(x[5:10], [0.5, 0.5, 1.5, 1.5, 2.0])
    np.testing.assert_allclose(y[5:10], [1.0, 0.75, 0.75, 1.0, 1.0])

    front_chamfer = _featured_front(Chamfer(0.5, 1.0, 0.2, side='upper'))
    x, y = _line_from_plot([front_chamfer, _refracting_plane(2, n=1.0)])
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


def test_plot_ray_paths_truncates_failed_rays_at_failure_surface():
    # the kernel keeps marching a failed ray's position history past the
    # surface that killed it; the drawn path must stop where the ray did.
    # ray 0: valid through both surfaces.  ray 1: clipped at surface 1 (1-based,
    # imag=+2) -- reached it, so its intersection there is drawn, but nothing
    # after.  ray 2: missed surface 1 (imag=-1) -- never arrived, draw only the
    # launch point.
    P = np.asarray([
        [[0., 0., 0.], [0., 1., 0.], [0., 2., 0.]],
        [[0., 0., 1.], [0., 1., 1.], [0., 2., 1.]],
        [[0., 0., 2.], [0., 1., 2.], [0., 2., 2.]],
    ])
    status = np.asarray([2 + 0j, 1 + 2j, 1 - 1j])
    result = RayTraceResult(P, np.zeros_like(P), np.zeros(P.shape[:-1]), status)

    fig, ax = plot_ray_paths(result)
    try:
        valid, clipped, missed = (np.asarray(line.get_ydata())
                                  for line in ax.lines)
        np.testing.assert_allclose(valid, [0., 0., 0.])
        np.testing.assert_allclose(clipped[:2], [1., 1.])
        assert np.isnan(clipped[2])
        np.testing.assert_allclose(missed[:1], [2.])
        assert np.isnan(missed[1:]).all()
    finally:
        plt.close(fig)


def test_plot_transverse_ray_aberration_plots_chief_relative_fan():
    P = np.asarray([
        [[0., -1., 0.], [0., 0., 0.], [0., 1., 0.]],
        [[0., 9., 1.], [0., 10., 1.], [0., 12., 1.]],
    ])

    fig, ax = plot_transverse_ray_aberration(P, axis='y')
    try:
        line = ax.lines[0]
        np.testing.assert_allclose(line.get_xdata(), [-1., 0., 1.])
        np.testing.assert_allclose(line.get_ydata(), [-1., 0., 2.])
    finally:
        plt.close(fig)


def test_plot_transverse_ray_aberration_accepts_raytrace_result_status():
    P = np.asarray([
        [[0., -1., 0.], [0., 0., 0.], [0., 1., 0.]],
        [[0., 9., 1.], [0., 10., 1.], [0., 12., 1.]],
    ])
    result = RayTraceResult(P, np.zeros_like(P), np.zeros(P.shape[:-1]),
                            np.asarray([1 + 2j, 0 + 0j, 0 + 0j]))

    fig, ax = plot_transverse_ray_aberration(result, axis='y')
    try:
        line = ax.lines[0]
        np.testing.assert_allclose(line.get_xdata(), [0., 1.])
        np.testing.assert_allclose(line.get_ydata(), [0., 2.])
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


def test_plot_wave_aberration_fan_detrend():
    coord = np.asarray([-1., 0., 1.])
    opd = 0.5 * coord + 0.125 * coord * coord + 0.25
    detrended = [1 / 24, -1 / 12, 1 / 24]
    # detrend removes the piston+tilt fit; on by default and via detrend=True
    for kw in (dict(wavelength=1), dict(wavelength=1, detrend=True)):
        fig, ax = plot_wave_aberration_fan(coord, opd, **kw)
        try:
            np.testing.assert_allclose(ax.lines[0].get_ydata(), detrended)
        finally:
            plt.close(fig)
    # detrend=False keeps the raw OPD
    fig, ax = plot_wave_aberration_fan(coord, opd, wavelength=1, detrend=False)
    try:
        np.testing.assert_allclose(ax.lines[0].get_ydata(), opd)
    finally:
        plt.close(fig)


def test_lens_element_groups_groups_singlet():
    prescription = [_refracting_plane(0, n=1.5),
                    _refracting_plane(2, n=1.0)]

    assert lens_element_groups(prescription) == [(0, 1)]


def test_lensdata_element_groups_method_queries_the_spine():
    # two refractors form one singlet (compiled indices 1, 2; index 0 is OBJECT)
    sys_ = _singlet_lensdata()
    assert sys_.lens.element_groups(wvl=0.5876) == [(1, 2)]


def test_lens_element_groups_groups_cemented_doublet():
    prescription = [_refracting_plane(0, n=1.5),
                    _refracting_plane(1, n=1.6),
                    _refracting_plane(2, n=1.0)]

    assert lens_element_groups(prescription) == [(0, 1, 2)]


def test_lens_element_groups_groups_cemented_triplet():
    prescription = [_refracting_plane(0, n=1.5),
                    _refracting_plane(1, n=1.6),
                    _refracting_plane(2, n=1.7),
                    _refracting_plane(3, n=1.0)]

    assert lens_element_groups(prescription) == [(0, 1, 2, 3)]


def test_lens_element_groups_splits_air_spaced_doublet():
    prescription = [_refracting_plane(0, n=1.5),
                    _refracting_plane(1, n=1.0),
                    _refracting_plane(3, n=1.6),
                    _refracting_plane(4, n=1.0)]

    assert lens_element_groups(prescription) == [(0, 1), (2, 3)]


def test_lens_element_groups_skips_lone_dummy_plane():
    # air-to-air dummy planes are not lens elements
    prescription = [_refracting_plane(0, n=1.5),
                    _refracting_plane(1, n=1.0),
                    _refracting_plane(2, n=1.0),
                    _refracting_plane(3, n=1.6),
                    _refracting_plane(4, n=1.0)]

    assert lens_element_groups(prescription) == [(0, 1), (3, 4)]


def test_plot_optics_skips_lone_dummy_plane():
    prescription = [_refracting_plane(0, n=1.5),
                    _refracting_plane(1, n=1.0),
                    _refracting_plane(2, n=1.0),
                    _refracting_plane(3, n=1.6),
                    _refracting_plane(4, n=1.0)]

    fig, ax = plot_optics(prescription, _trace_result(prescription),
                          points=5, wvl=0.55)
    plt.close(fig)


def test_plot_optics_draws_stop_marker_on_dummy_plane():
    prescription = [_refracting_plane(0, n=1.5),
                    _refracting_plane(1, n=1.0),
                    _refracting_plane(2, n=1.0),
                    _refracting_plane(3, n=1.6),
                    _refracting_plane(4, n=1.0)]

    fig, ax = plot_optics(prescription, _trace_result(prescription),
                          points=5, wvl=0.55, stop_index=2)
    try:
        # two lens groups plus the stop marker, drawn in surface order
        assert len(ax.lines) == 3
        marker = ax.lines[1]
        xd = np.asarray(marker.get_xdata(), dtype=float)
        yd = np.asarray(marker.get_ydata(), dtype=float)
        # rays span y=+/-1 at the stop -> semidiameter 1, stem 0.2, bar 0.1;
        # the synthetic trace has no directions, so the local optical axis
        # falls back to the surface axis (+z).  Bottom edge first: crossbar
        # along z through (2, -1), then the stem outward to (2, -1.2)
        assert xd[0] == pytest.approx(1.95)
        assert xd[1] == pytest.approx(2.05)
        assert yd[0] == -1 and yd[1] == -1
        assert xd[3] == 2 and yd[3] == -1
        assert xd[4] == 2 and yd[4] == pytest.approx(-1.2)
        # top edge mirrors it, stem pointing up and out
        assert xd[9] == 2 and yd[9] == 1
        assert xd[10] == 2 and yd[10] == pytest.approx(1.2)
    finally:
        plt.close(fig)


def test_plot_optics_marks_stop_from_system_metadata():
    # a standalone stop ahead of a singlet; stop_index lives on the system and
    # plot_optics picks it up without an explicit kwarg
    n15 = materials.ConstantMaterial(1.5)
    air = materials.air
    lens = LensData()
    (lens.add(Plane(), thickness=5.0, material=air, aperture=5.0)
         .add(Conic(1 / 60.0, 0.0), thickness=4.0, material=n15,
              aperture=8.0)
         .add(Conic(-1 / 60.0, 0.0), thickness=95.0, material=air,
              aperture=8.0)
         .add(Plane(), typ='eval', material=air, aperture=20.0))
    sys = OpticalSystem(lens, aperture=8.0, fields=[0.0],
                        wavelengths=[0.5876], reference=0, stop_index=0)

    fig, ax = sys.plot.layout_2d()
    try:
        markers = [ln for ln in ax.lines if len(ln.get_xdata()) == 12]
        assert len(markers) == 1
        xd = np.asarray(markers[0].get_xdata(), dtype=float)
        yd = np.asarray(markers[0].get_ydata(), dtype=float)
        # the marks cluster around the stop plane at z=0, at +/- the stop
        # semidiameter (EPD/2 = 4, the stop is the entrance pupil here)
        assert np.nanmax(np.abs(xd)) < 1.0
        assert np.nanmax(np.abs(yd)) == pytest.approx(4 * 1.2, rel=1e-6)
    finally:
        plt.close(fig)


def test_lens_element_groups_rejects_terminal_group():
    with pytest.raises(ValueError, match='terminates'):
        lens_element_groups([_refracting_plane(0, n=1.5),
                                   _refracting_plane(1, n=1.6)])


def test_plot_optics_group_od_uses_largest_aperture_in_group():
    prescription = [_refracting_plane(0, outer_radius=1.0, n=1.5),
                    _refracting_plane(1, outer_radius=2.0, n=1.6),
                    _refracting_plane(2, outer_radius=1.2, n=1.0)]

    _, y = _line_from_plot(prescription)

    assert np.nanmax(y) == 2.0
    assert np.nanmin(y) == -2.0


def test_plot_optics_bridges_steep_surface_to_od_with_normal_segment():
    # steep surfaces stop at the equator and bridge to the element OD
    gentle = Surface(shape=Conic(1 / 5.0, 0.0), interaction='refr',
                     P=np.asarray([0., 0., 0.]), material=materials.ConstantMaterial(1.5))
    steep = Surface(shape=Conic(1 / 0.5, 0.0), interaction='refr',
                    P=np.asarray([0., 0., 1.0]), material=materials.air)
    prescription = [gentle, steep]

    with pytest.warns(UserWarning, match='flat edge'):
        fig, ax = plot_optics(prescription, _trace_result(prescription),
                              points=41, wvl=0.55)
    try:
        x = np.asarray(ax.lines[0].get_xdata())
        y = np.asarray(ax.lines[0].get_ydata())
    finally:
        plt.close(fig)

    assert np.isfinite(x).all()
    assert np.isfinite(y).all()
    # element OD is preserved
    np.testing.assert_allclose(np.max(np.abs(y)), 1.0)
    # constant-z bridge reaches the OD
    ridge = np.isclose(x, np.max(x))
    assert ridge.sum() >= 2
    np.testing.assert_allclose(np.max(np.abs(y[ridge])), 1.0)
    assert np.min(np.abs(y[ridge])) <= 0.55


def test_plot_optics_draws_clear_aperture_land_to_od_silently():
    # intentional smaller drawn extents bridge silently
    front = Surface(shape=Conic(1 / 50.0, 0.0), interaction='refr',
                    P=np.asarray([0., 0., 0.]), material=materials.ConstantMaterial(1.5),
                    aperture=_extent(1.0))
    rear = Surface(shape=Conic(-1 / 50.0, 0.0), interaction='refr',
                   P=np.asarray([0., 0., 1.0]), material=materials.air,
                   aperture=_extent(3.0))
    prescription = [front, rear]

    import warnings as _warnings
    with _warnings.catch_warnings():
        _warnings.simplefilter('error')  # an intentional aperture must not warn
        fig, ax = plot_optics(prescription, _trace_result(prescription),
                              points=41, wvl=0.55)
    try:
        x = np.asarray(ax.lines[0].get_xdata())
        y = np.asarray(ax.lines[0].get_ydata())
    finally:
        plt.close(fig)

    assert np.isfinite(x).all()
    assert np.isfinite(y).all()
    # the element OD is the larger surface's drawn extent
    np.testing.assert_allclose(np.max(np.abs(y)), 3.0)
    # small front surface bridges to the larger OD
    rim_sag = float(front.sag(np.asarray([0.]), np.asarray([1.0]))[0])
    land = np.isclose(x, rim_sag) & (np.abs(y) > 1.0 + 1e-9)
    assert land.sum() >= 2
    np.testing.assert_allclose(np.max(np.abs(y[land])), 3.0)


def test_plot_optics_steep_surface_capped_by_own_aperture_is_silent():
    # extent-limited steep surfaces do not warn
    gentle = Surface(shape=Conic(1 / 5.0, 0.0), interaction='refr',
                     P=np.asarray([0., 0., 0.]), material=materials.ConstantMaterial(1.5),
                     aperture=_extent(1.0))
    steep = Surface(shape=Conic(1 / 0.5, 0.0), interaction='refr',
                    P=np.asarray([0., 0., 1.0]), material=materials.air,
                    aperture=_extent(0.4))
    prescription = [gentle, steep]

    import warnings as _warnings
    with _warnings.catch_warnings():
        _warnings.simplefilter('error')
        fig, ax = plot_optics(prescription, _trace_result(prescription),
                              points=41, wvl=0.55)
    plt.close(fig)


def test_plot_optics_reads_edge_features_from_surface_aperture():
    front = _featured_front(SquareCut(0.5, 1.5, 0.25, side='upper'))
    rear = _refracting_plane(2, n=1.0)

    x, y = _line_from_plot([front, rear])

    np.testing.assert_allclose(x[5:10], [0.5, 0.5, 1.5, 1.5, 2.0])
    np.testing.assert_allclose(y[5:10], [1.0, 0.75, 0.75, 1.0, 1.0])


def test_lensdata_add_aperture_features_propagate_to_compiled_surface():
    ap = Aperture(extent=CircularExtent(9.0),
                  features=(Chamfer(0.0, 0.5, 0.3),))
    lens = LensData()
    (lens.add(Conic(1 / 60.0, 0.0), thickness=4.0, material=materials.ConstantMaterial(1.5),
              aperture=ap)
         .add(Conic(-1 / 60.0, 0.0), thickness=95.0, material=materials.air,
              aperture=8.0))
    ld = OpticalSystem(lens, aperture=10.0, wavelengths=[0.5876],
                       reference=0)

    surfaces = ld.to_surfaces()
    assert surfaces[1].aperture.features == ap.features  # [0] is OBJECT
    assert surfaces[2].aperture.features == ()
    # the features survive a copy of the LensData
    assert ld.copy().to_surfaces()[1].aperture.features == ap.features


def test_plot_optics_draws_mirror_optical_surface_by_default():
    prescription = [_reflecting_surface(Plane(), outer_radius=1)]

    x, y = _line_from_plot(prescription)

    np.testing.assert_allclose(x, np.zeros(5))
    np.testing.assert_allclose(y, np.linspace(-1, 1, 5))


def _mirror_with_substrate(substrate, outer_radius=1, inner_radius=None,
                           shape=None, **surf_kwargs):
    """A reflective surface carrying a drawn extent and a substrate."""
    surf = Surface(
        shape=shape if shape is not None else Plane(),
        interaction='refl',
        aperture=Aperture(
            extent=CircularExtent(
                outer_radius,
                inner_radius=0.0 if inner_radius is None else inner_radius),
            substrate=substrate),
        **surf_kwargs,
    )
    return surf


def test_plot_optics_draws_parallel_mirror_substrate():
    surf = _mirror_with_substrate(ParallelSubstrate(thickness=2, side=1),
                                  P=np.asarray([0., 0., 0.]))
    x, y = _line_from_plot([surf])

    np.testing.assert_allclose(x[:5], np.zeros(5))
    assert np.any((y[:-1] == 1) & (y[1:] == 1) & (x[:-1] == 0) & (x[1:] == 2))
    assert np.any((y[:-1] == -1) & (y[1:] == -1) & (x[:-1] == 2) & (x[1:] == 0))
    np.testing.assert_allclose(x[6:11], np.full(5, 2.0))


def test_mirror_substrate_outline_applies_surface_decenter():
    surf = _mirror_with_substrate(ParallelSubstrate(thickness=2, side=1),
                                  P=np.asarray([0., 10., 5.]))
    result = _trace_result([surf])

    x, y = mirror_substrate_outline(
        surf, result, substrate=surf.aperture.substrate, points=5)

    np.testing.assert_allclose(x[:5], np.full(5, 5.0))
    np.testing.assert_allclose(y[:5], np.linspace(9, 11, 5))
    np.testing.assert_allclose(x[6:11], np.full(5, 7.0))


def test_mirror_substrate_outline_bores_a_through_hole():
    # an annular drawn extent draws two loops with an open bore
    surf = _mirror_with_substrate(
        FlatParentSubstrate(thickness=5.0, side=1),
        shape=Conic(1 / 200.0, 0.0), outer_radius=10.0, inner_radius=3.0,
        P=np.asarray([0., 0., 0.]))
    result = _trace_result([surf])
    x, y = mirror_substrate_outline(
        surf, result, substrate=surf.aperture.substrate, points=41)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    # one separator between loops, plus the trailing separator
    assert np.isnan(x).sum() == 2
    # the bore is open
    finite = np.isfinite(x) & np.isfinite(y)
    assert np.all(np.abs(y[finite]) >= 3.0 - 1e-9)
    # both faces are present out at the rim
    assert np.isclose(x[finite].max(), 5.0)


def test_mirror_substrate_outline_can_center_on_ray_footprint():
    surf = _mirror_with_substrate(ParallelSubstrate(thickness=2, side=1),
                                  P=np.asarray([0., 0., 0.]))
    P = np.asarray([
        [[0., 9., -1.], [0., 10., -1.], [0., 11., -1.]],
        [[0., 9., 0.], [0., 10., 0.], [0., 11., 0.]],
    ])
    result = RayTraceResult(
        P, np.zeros_like(P), np.zeros(P.shape[:-1]),
        np.zeros(P.shape[1], dtype=np.complex128),
    )

    x, y = mirror_substrate_outline(
        surf, result, substrate=surf.aperture.substrate, center='rays',
        points=5)

    np.testing.assert_allclose(x[:5], np.zeros(5))
    np.testing.assert_allclose(y[:5], np.linspace(9, 11, 5))
    np.testing.assert_allclose(x[6:11], np.full(5, 2.0))


def test_mirror_substrate_outline_applies_surface_tilt_in_xz_projection():
    surf = _mirror_with_substrate(ParallelSubstrate(thickness=2, side=1),
                                  P=np.asarray([0., 0., 0.]), R=(0, -45, 0))
    result = _trace_result([surf])

    x, y = mirror_substrate_outline(
        surf, result, substrate=surf.aperture.substrate, points=5,
        x='z', y='x')

    front_x = np.asarray(x[:5])
    front_y = np.asarray(y[:5])
    assert not np.allclose(front_x, front_x[0])
    assert not np.allclose(front_y, front_y[0])
    np.testing.assert_allclose(np.diff(front_x) / np.diff(front_y),
                               np.full(4, -1.0))


def test_mirror_substrate_can_cut_flat_from_parent_vertex_plane():
    surf = _mirror_with_substrate(
        FlatParentSubstrate(thickness=2, side=1),
        shape=OffAxisConic(c=1 / 100., k=-1., dy=10), outer_radius=5,
        P=np.asarray([0., 0., 0.]))
    result = _trace_result([surf])

    x, _ = mirror_substrate_outline(
        surf, result, substrate=surf.aperture.substrate, points=5)

    np.testing.assert_allclose(x[6:11], np.full(5, 2.0))


def test_mirror_substrate_can_cut_flat_near_aperture_for_uniform_thickness():
    surf = _mirror_with_substrate(
        FlatBackSubstrate(thickness=2, side=1),
        shape=OffAxisConic(c=1 / 100., k=-1., dy=10), outer_radius=5,
        P=np.asarray([0., 0., 0.]))
    result = _trace_result([surf])

    x, y = mirror_substrate_outline(
        surf, result, substrate=surf.aperture.substrate, points=5)

    rear_x = np.asarray(x[6:11])
    rear_y = np.asarray(y[6:11])
    slope = np.diff(rear_x) / np.diff(rear_y)
    assert not np.allclose(rear_x, rear_x[0])
    np.testing.assert_allclose(slope, np.full(4, slope[0]))

    front_lower_edge = surf.sag(np.asarray([0.]), np.asarray([-5.]))[0]
    rear_lower_edge = rear_x[rear_y == -5][0]
    np.testing.assert_allclose(rear_lower_edge - front_lower_edge, 2.0)


def test_plot_spot_diagram_accepts_result_and_masks_invalid():
    P = np.asarray([
        [[0., -1., 0.], [0., 0., 0.], [0., 1., 0.]],
        [[2., 3., 5.], [0., 1., 5.], [-2., -1., 5.]],
    ])
    result = RayTraceResult(P, np.zeros_like(P), np.zeros(P.shape[:-1]),
                            np.asarray([0 + 0j, 0 + 0j, 1 + 2j]))

    fig, ax = plot_spot_diagram(result)
    try:
        offsets = ax.collections[0].get_offsets()
        # invalid third ray dropped; first two surviving image points kept
        np.testing.assert_allclose(offsets, [[2., 3.], [0., 1.]])
        assert ax.get_aspect() == 1.0
    finally:
        plt.close(fig)


def test_plot_spot_diagram_subtracts_centroid_origin():
    P = np.asarray([
        [[0., 0., 0.], [0., 0., 0.]],
        [[1., 3., 5.], [3., 5., 5.]],
    ])

    fig, ax = plot_spot_diagram(P, origin='centroid')
    try:
        offsets = ax.collections[0].get_offsets()
        # centroid (2, 4) removed
        np.testing.assert_allclose(offsets, [[-1., -1.], [1., 1.]])
    finally:
        plt.close(fig)


def test_plot_spot_diagram_subtracts_explicit_origin():
    P = np.asarray([
        [[0., 0., 0.], [0., 0., 0.]],
        [[1., 3., 5.], [3., 5., 5.]],
    ])

    fig, ax = plot_spot_diagram(P, origin=(1., 3.))
    try:
        offsets = ax.collections[0].get_offsets()
        np.testing.assert_allclose(offsets, [[0., 0.], [2., 2.]])
    finally:
        plt.close(fig)


def test_plot_field_curvature_plots_s_and_t_vs_field():
    ld = _singlet_lensdata()

    fig, ax = plot_field_curvature(ld, ld.fields, label='d')
    try:
        assert [line.get_label() for line in ax.lines] == ['d S', 'd T']
        for line in ax.lines:
            np.testing.assert_allclose(line.get_ydata(), [0., 3., 5.])
        # the focus shift is referenced to the image vertex, so the plotted
        # x-values differ from the raw lab-frame foci by that vertex z
        image_z = float(ld[-1].P[2])
        from prysm.x.raytracing.analysis import field_curvature
        result = field_curvature(ld, ld.fields, ld.wavelength())
        np.testing.assert_allclose(ax.lines[0].get_xdata(),
                                   np.asarray(result.x_fan_z) - image_z)
        np.testing.assert_allclose(ax.lines[1].get_xdata(),
                                   np.asarray(result.y_fan_z) - image_z)
        # on-axis sagittal and tangential foci coincide
        np.testing.assert_allclose(ax.lines[0].get_xdata()[0],
                                   ax.lines[1].get_xdata()[0])
    finally:
        plt.close(fig)


def test_plot_field_curvature_uses_xy_labels_for_non_pure_y_fields():
    ld = _singlet_lensdata()
    fields = [Field(1.0, 1.0, unit='deg'), Field(2.0, 3.0, unit='deg')]

    fig, ax = plot_field_curvature(ld, fields, label='d')
    try:
        assert [line.get_label() for line in ax.lines] == ['d X', 'd Y']
    finally:
        plt.close(fig)


def test_plot_chromatic_focal_shift_plots_shift_vs_wavelength():
    ld = _singlet_lensdata()

    fig, ax = plot_chromatic_focal_shift(
        ld, focus='paraxial', samples=9, label='paraxial',
    )
    try:
        line = ax.lines[0]
        from prysm.x.raytracing.analysis import chromatic_focal_shift
        wavelengths, shifts = chromatic_focal_shift(
            ld, focus='paraxial', samples=9,
        )
        np.testing.assert_allclose(line.get_xdata(), wavelengths)
        np.testing.assert_allclose(line.get_ydata(), shifts)
        assert len(line.get_xdata()) == 9
        assert line.get_label() == 'paraxial'
        assert ax.get_xlabel() == 'wavelength [um]'
        assert ax.get_ylabel() == 'focus shift'
    finally:
        plt.close(fig)


def test_plot_distortion_plots_percent_vs_field():
    ld = _singlet_lensdata()

    fig, ax = plot_distortion(ld, ld.fields)
    try:
        line = ax.lines[0]
        np.testing.assert_allclose(line.get_ydata(), [0., 3., 5.])
        from prysm.x.raytracing.analysis import distortion
        result = distortion(ld, ld.fields, ld.wavelength())
        np.testing.assert_allclose(line.get_xdata(), result.percent)
        assert line.get_xdata()[0] == 0.0  # no distortion on axis
        assert ax.get_xlabel() == 'distortion [%]'
    finally:
        plt.close(fig)
