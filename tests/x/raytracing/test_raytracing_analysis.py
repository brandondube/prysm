"""Tests for the analysis primitives."""
import numpy as np
import pytest

from tests.x.raytracing.surface_helpers import (
    plane, sphere, conic, off_axis_conic, even_asphere, q2d, zernike, xy,
    chebyshev, jacobi, toroid, biconic,
)

from prysm.x.raytracing.surfaces import (
    Surface, circular_aperture, annular_aperture,
)
from prysm.x.raytracing.spencer_and_murty import STATUS_CLIP
from prysm.x.raytracing.spencer_and_murty import raytrace
from prysm.x.raytracing.launch import Field, Sampling, launch
from prysm.x.raytracing.analysis import (
    transverse_ray_aberration,
    wavefront,
    wavefront_zernike_fit,
    distortion,
    field_curvature,
    axial_color,
    lateral_color,
)
from prysm.x.raytracing.opt import opd_from_raytrace, xp_reference_sphere


# ---------- helpers ---------------------------------------------------------

def _concave_parabola():
    """Parabolic mirror at z=0, image at parabolic focus z=-40."""
    c = -1 / 80.0
    f = 1.0 / (2.0 * c)  # = -40
    s = conic(c=c, k=-1.0, interaction='refl', P=[0, 0, 0])
    img = plane(interaction='eval', P=[0, 0, f])
    return [s, img]


def _spherical_singlet():
    """Two-surface BK7-like singlet, f ~= 50 mm."""
    n_glass = lambda w: 1.5
    s1 = conic(c=1 / 50.0, k=0.0, interaction='refr',
                       P=[0, 0, 0], material=n_glass)
    s2 = conic(c=-1 / 50.0, k=0.0, interaction='refr',
                       P=[0, 0, 5.0], material=lambda w: 1.0)
    img = plane(interaction='eval', P=[0, 0, 100.0])
    return [s1, s2, img]


# ---------- transverse_ray_aberration --------------------------------------

def test_tra_axes_pick_correct_column():
    presc = _concave_parabola()
    P, S = launch(presc, Field(0., 0.), 0.55e-3,
                  Sampling.fan(n=11), epd=10.0, pupil_z=-50.0)
    trace = raytrace(presc, P, S, 0.55e-3)
    pupil_y, dy = transverse_ray_aberration(trace.P, axis='y')
    pupil_x, dx = transverse_ray_aberration(trace.P, axis='x')
    np.testing.assert_array_equal(pupil_y, P[:, 1])
    np.testing.assert_array_equal(pupil_x, P[:, 0])


def test_tra_chief_offset_is_zero():
    """The chief ray's contribution to TRA must be exactly zero."""
    presc = _concave_parabola()
    P, S = launch(presc, Field(0., 0.), 0.55e-3,
                  Sampling.fan(n=11), epd=10.0, pupil_z=-50.0)
    trace = raytrace(presc, P, S, 0.55e-3)
    _, dy = transverse_ray_aberration(trace.P, axis='y')
    assert dy[len(dy) // 2] == 0.0


def test_tra_parabola_on_axis_dy_is_small():
    """For an on-axis collimated bundle on a perfect parabola, TRA is ~0."""
    presc = _concave_parabola()
    P, S = launch(presc, Field(0., 0.), 0.55e-3,
                  Sampling.fan(n=11), epd=10.0, pupil_z=-50.0)
    trace = raytrace(presc, P, S, 0.55e-3)
    _, dy = transverse_ray_aberration(trace.P, axis='y')
    assert float(np.max(np.abs(dy))) < 1e-9


def test_tra_rejects_bad_axis():
    presc = _concave_parabola()
    P, S = launch(presc, Field(0., 0.), 0.55e-3,
                  Sampling.fan(n=5), epd=4.0, pupil_z=-10.0)
    trace = raytrace(presc, P, S, 0.55e-3)
    with pytest.raises(ValueError):
        transverse_ray_aberration(trace.P, axis='z')


def test_tra_filters_invalid_rays_from_status():
    P_hist = np.array([
        [[0., -1., 0.], [0., 0., 0.], [0., 1., 0.]],
        [[0., 100., 1.], [0., 0., 1.], [0., 1., 1.]],
    ])
    status = np.array([1 + STATUS_CLIP * 1j, 0 + 0j, 0 + 0j])
    pupil_y, dy = transverse_ray_aberration(P_hist, axis='y',
                                            chief_index=1,
                                            status=status)
    np.testing.assert_array_equal(pupil_y, [0., 1.])
    np.testing.assert_array_equal(dy, [0., 1.])


def test_tra_filters_nonfinite_rays_without_status():
    P_hist = np.array([
        [[0., -1., 0.], [0., 0., 0.], [0., 1., 0.]],
        [[0., np.nan, 1.], [0., 0., 1.], [0., 1., 1.]],
    ])
    pupil_y, dy = transverse_ray_aberration(P_hist, axis='y',
                                            chief_index=1)
    np.testing.assert_array_equal(pupil_y, [0., 1.])
    np.testing.assert_array_equal(dy, [0., 1.])


def test_tra_centroid_reference_works_without_a_chief():
    # chief (index 1) lands at NaN -- e.g. clipped by a central obscuration;
    # the centroid reference registers on the surviving rays instead
    P_hist = np.array([
        [[0., -1., 0.], [0., 0., 0.], [0., 1., 0.]],
        [[0., 2., 10.], [0., np.nan, 10.], [0., 4., 10.]],
    ])
    status = np.array([0 + 0j, 1 + STATUS_CLIP * 1j, 0 + 0j])
    pupil_y, dy = transverse_ray_aberration(P_hist, axis='y', chief_index=1,
                                            status=status, reference='centroid')
    np.testing.assert_array_equal(pupil_y, [-1., 1.])
    np.testing.assert_allclose(dy, [-1., 1.])  # mean of {2,4}=3 subtracted


def test_tra_chief_reference_raises_when_chief_clipped():
    P_hist = np.array([
        [[0., -1., 0.], [0., 0., 0.], [0., 1., 0.]],
        [[0., 2., 10.], [0., np.nan, 10.], [0., 4., 10.]],
    ])
    with pytest.raises(ValueError, match='centroid'):
        transverse_ray_aberration(P_hist, axis='y', chief_index=1)


def test_tra_pupil_is_chief_relative_under_launch_shift():
    """Pupil coordinate must re-center on the chief, not the launch axis.

    Routing an off-axis bundle through the entrance pupil rigidly shifts the
    whole fan laterally at the launch plane, so the chief ray no longer
    launches on axis.  The reported pupil coordinate is the launch offset
    from the chief, so the fan stays symmetric about the pupil center
    regardless of that shift (matches wavefront()).
    """
    shift = 5.0
    launch_y = np.array([-1., 0., 1.]) + shift   # chief (index 1) at +shift
    P_hist = np.array([
        [[0., launch_y[0], 0.], [0., launch_y[1], 0.], [0., launch_y[2], 0.]],
        [[0., 0.3, 10.], [0., 0.0, 10.], [0., -0.3, 10.]],
    ])
    pupil_y, dy = transverse_ray_aberration(P_hist, axis='y', chief_index=1)
    np.testing.assert_allclose(pupil_y, [-1., 0., 1.])
    np.testing.assert_allclose(dy, [0.3, 0.0, -0.3])


# ---------- wavefront -------------------------------------------------------

def test_wavefront_chief_opd_is_zero():
    presc = _spherical_singlet()
    P, S = launch(presc, Field(0., 0.), 0.55,
                  Sampling.fan(n=9), epd=4.0, pupil_z=-5.0)
    opd, x_pup, y_pup = wavefront(presc, P, S, 0.55)
    chief = len(opd) // 2
    np.testing.assert_allclose(opd[chief], 0.0, atol=1e-12)
    np.testing.assert_array_equal(x_pup, P[:, 0])
    np.testing.assert_array_equal(y_pup, P[:, 1])


def test_wavefront_uses_penultimate_surface_image_medium():
    presc = _spherical_singlet()
    presc[-2].n = lambda w: 1.25
    wvl = 0.55
    P, S = launch(presc, Field(0., 0.), wvl,
                  Sampling.fan(n=9), epd=4.0, pupil_z=-5.0)
    opd, _, _ = wavefront(presc, P, S, wvl)

    trace = raytrace(presc, P, S, wvl)
    chief = len(P) // 2
    C, _, P_xp = xp_reference_sphere(trace.P[-1, chief], trace.S[-1, chief])
    expected = opd_from_raytrace(trace.P, trace.S, trace.OPL, C, P_xp,
                                 n_image=1.25, chief_index=chief)
    wrong_air = opd_from_raytrace(trace.P, trace.S, trace.OPL, C, P_xp,
                                  n_image=1.0, chief_index=chief)

    np.testing.assert_allclose(opd, expected, atol=1e-12)
    assert np.max(np.abs(expected - wrong_air)) > 1e-8


def test_wavefront_uses_surface_zero_object_medium_when_present():
    object_surface = plane(interaction='eval', P=[0, 0, -10.0],
                           material=lambda w: 1.2)
    presc = [object_surface] + _spherical_singlet()
    wvl = 0.55
    P, S = launch(presc, Field(0., 0.), wvl,
                  Sampling.fan(n=9), epd=4.0, pupil_z=-20.0)
    opd, _, _ = wavefront(presc, P, S, wvl)

    trace = raytrace(presc, P, S, wvl)
    chief = len(P) // 2
    C, _, P_xp = xp_reference_sphere(trace.P[-1, chief], trace.S[-1, chief])
    expected = opd_from_raytrace(trace.P, trace.S, trace.OPL, C, P_xp,
                                 n_image=1.0, chief_index=chief)

    np.testing.assert_allclose(opd, expected, atol=1e-12)


def test_wavefront_parabola_is_diffraction_limited():
    """Perfect parabolic mirror, on-axis collimated input → OPD ~ 0."""
    presc = _concave_parabola()
    P, S = launch(presc, Field(0., 0.), 0.55e-3,
                  Sampling.fan(n=11), epd=10.0, pupil_z=-50.0)
    opd, _, _ = wavefront(presc, P, S, 0.55e-3)
    assert float(np.max(np.abs(opd))) < 1e-9


def test_wavefront_filters_vignetted_rays():
    presc = _spherical_singlet()
    presc[0].aperture = circular_aperture(1.5)
    P, S = launch(presc, Field(0., 0.), 0.55,
                  Sampling.fan(n=9), epd=4.0, pupil_z=-5.0)
    trace = raytrace(presc, P, S, 0.55)
    valid = trace.status.imag == 0
    assert valid.sum() < valid.size

    opd, x_pup, y_pup = wavefront(presc, P, S, 0.55)
    assert opd.shape == (valid.sum(),)
    assert np.isfinite(opd).all()
    np.testing.assert_array_equal(x_pup, P[valid, 0])
    np.testing.assert_array_equal(y_pup, P[valid, 1])


# ---------- wavefront_zernike_fit ------------------------------------------

def test_zernike_fit_recovers_known_piston():
    """If OPD is a constant, the piston Z(0,0) coefficient recovers it."""
    n = 256
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, n)
    y = rng.uniform(-1, 1, n)
    # restrict to unit disk
    inside = x * x + y * y <= 1.0
    x = x[inside]
    y = y[inside]
    opd = 0.123 * np.ones_like(x)
    nms = [(0, 0), (1, 1), (1, -1), (2, 0)]
    coefs, rms = wavefront_zernike_fit(opd, x, y, nms,
                                       normalization_radius=1.0,
                                       norm=False)
    np.testing.assert_allclose(coefs[0], 0.123, atol=1e-12)
    assert rms < 1e-12


def test_zernike_fit_residual_is_zero_for_basis_term():
    """A pure Zernike Z(2, 0) input should fit exactly with residual 0."""
    n = 1024
    rng = np.random.default_rng(7)
    x = rng.uniform(-1, 1, n)
    y = rng.uniform(-1, 1, n)
    inside = x * x + y * y <= 1.0
    x = x[inside]
    y = y[inside]
    rsq = x * x + y * y
    # zero-to-peak Z(2,0) = 2 r^2 - 1
    opd_input = 0.5 * (2.0 * rsq - 1.0)
    nms = [(0, 0), (2, 0)]
    coefs, rms = wavefront_zernike_fit(opd_input, x, y, nms,
                                       normalization_radius=1.0, norm=False)
    np.testing.assert_allclose(coefs[1], 0.5, atol=1e-12)
    assert rms < 1e-12


def test_zernike_fit_norm_radius_must_be_positive():
    with pytest.raises(ValueError):
        wavefront_zernike_fit(np.zeros(10), np.zeros(10), np.zeros(10),
                              [(0, 0)], normalization_radius=0.0)


# ---------- distortion ------------------------------------------------------

def test_distortion_zero_for_on_axis_field():
    presc = _spherical_singlet()
    real_xy, paraxial_xy, percent = distortion(
        presc, [Field(0., 0., unit='deg')], 0.55, epd=4.0,
    )
    np.testing.assert_allclose(real_xy[0], 0.0, atol=1e-12)
    assert percent[0] == 0.0


def test_distortion_returns_per_field_arrays():
    presc = _spherical_singlet()
    fields = [Field(0., 0., unit='deg'),
              Field(0., 1., unit='deg'),
              Field(0., 2., unit='deg')]
    real_xy, paraxial_xy, percent = distortion(
        presc, fields, 0.55, epd=4.0,
    )
    assert real_xy.shape == (3, 2)
    assert paraxial_xy.shape == (3, 2)
    assert percent.shape == (3,)


def test_distortion_paraxial_proxy_scales_linearly():
    """For a small field, the paraxial-proxy landing should match the real
    landing (no third-order distortion at small angles)."""
    presc = _spherical_singlet()
    real_xy, paraxial_xy, percent = distortion(
        presc, [Field(0., 0.05, unit='deg')], 0.55, epd=4.0,
    )
    # at 0.05 deg a paraxial-quality system should have <0.1% distortion
    assert abs(percent[0]) < 0.1


def test_distortion_sign_distinguishes_barrel_and_pincushion():
    # a positive singlet with the stop in front of the lens gives barrel
    # (negative) distortion; with the stop behind it gives pincushion
    # (positive).  The signed percent must tell them apart -- a magnitude
    # would report both as the same positive number.
    presc = _spherical_singlet()
    field = [Field(0., 8., unit='deg')]
    _, _, barrel = distortion(presc, field, 0.55, epd=4.0, pupil_z=-30.0)
    _, _, pincushion = distortion(presc, field, 0.55, epd=4.0, pupil_z=30.0)
    assert barrel[0] < 0.0
    assert pincushion[0] > 0.0


# ---------- field_curvature -------------------------------------------------

def test_field_curvature_on_axis_sag_equals_tan():
    """For an on-axis field of an axisymmetric system, sag and tan focus
    coincide (within FP)."""
    presc = _spherical_singlet()
    sag, tan = field_curvature(presc, [Field(0., 0., unit='deg')],
                               0.55, epd=4.0)
    np.testing.assert_allclose(sag, tan, atol=1e-9)


def test_field_curvature_returns_arrays_of_correct_shape():
    presc = _spherical_singlet()
    fields = [Field(0., h, unit='deg') for h in (0., 1., 2.)]
    sag, tan = field_curvature(presc, fields, 0.55, epd=4.0)
    assert sag.shape == (3,)
    assert tan.shape == (3,)


def test_field_curvature_default_is_differential():
    """The default reports the differential (Coddington) foci.

    The default marginal_fraction must sample near the chief, so the result
    matches the marginal_fraction -> 0 limit and is distinct from a finite
    zonal focus (which folds in coma / oblique spherical aberration).
    """
    presc = _spherical_singlet()
    fields = [Field(0., 8., unit='deg')]
    sag_d, tan_d = field_curvature(presc, fields, 0.55, epd=4.0)
    sag_lim, tan_lim = field_curvature(presc, fields, 0.55, epd=4.0,
                                       marginal_fraction=1e-4)
    sag_zone, tan_zone = field_curvature(presc, fields, 0.55, epd=4.0,
                                         marginal_fraction=0.7)
    # default agrees with the differential limit ...
    np.testing.assert_allclose(sag_d, sag_lim, atol=5e-3)
    np.testing.assert_allclose(tan_d, tan_lim, atol=5e-3)
    # ... and is meaningfully distinct from the 0.7-zone focus
    assert abs(float(tan_d[0] - tan_zone[0])) > 0.1


# ---------- axial / lateral color ------------------------------------------

def test_axial_color_constant_index_returns_constant_bfd():
    """When n_glass(w) is constant the paraxial image distance must not
    depend on wavelength."""
    presc = _spherical_singlet()
    bfd = axial_color(presc, [0.45, 0.55, 0.65])
    np.testing.assert_allclose(bfd, bfd[0], rtol=1e-12)


def test_axial_color_varying_index_changes_bfd():
    """A toy dispersion (n decreasing with wvl) should monotonically shift
    the paraxial image."""
    n_glass = lambda w: 1.6 - 0.1 * (w - 0.45) / 0.2
    s1 = conic(c=1 / 50.0, k=0.0, interaction='refr',
                       P=[0, 0, 0], material=n_glass)
    s2 = conic(c=-1 / 50.0, k=0.0, interaction='refr',
                       P=[0, 0, 5.0], material=lambda w: 1.0)
    img = plane(interaction='eval', P=[0, 0, 100.0])
    presc = [s1, s2, img]
    bfd = axial_color(presc, [0.45, 0.55, 0.65])
    # n decreasing → focal length increasing (weaker glass) → bfd grows
    assert bfd[0] != bfd[2]


def test_lateral_color_shape():
    presc = _spherical_singlet()
    fields = [Field(0., h, unit='deg') for h in (0., 1.)]
    landing = lateral_color(presc, fields, [0.45, 0.55], epd=4.0)
    assert landing.shape == (2, 2, 2)


def test_lateral_color_constant_index_constant_landing():
    """Achromatic glass → chief landing identical across wavelengths."""
    presc = _spherical_singlet()
    fields = [Field(0., 1., unit='deg')]
    landing = lateral_color(presc, fields, [0.45, 0.55, 0.65], epd=4.0)
    np.testing.assert_allclose(landing[0, 0], landing[0, 1], atol=1e-12)
    np.testing.assert_allclose(landing[0, 1], landing[0, 2], atol=1e-12)


# ---------- chief-ray reference (obscured bundles) --------------------------

def test_wavefront_centroid_reference_for_obscured_chief():
    """wavefront(reference='centroid') works when the chief is obscured.

    A central obstruction on the first surface clips the geometric chief ray.
    reference='chief' cannot define a reference sphere and raises; the new
    reference='centroid' anchors on the surviving ray nearest the pupil center.
    """
    presc = _spherical_singlet()
    presc[0].aperture = annular_aperture(1.5, 6.0)  # central obstruction
    P, S = launch(presc, Field(0., 0.), 0.55e-3, Sampling.hex(nrings=4),
                  epd=8.0, pupil_z=-5.0)
    with pytest.raises(ValueError):
        wavefront(presc, P, S, 0.55e-3, reference='chief')
    opd, xp, yp = wavefront(presc, P, S, 0.55e-3, reference='centroid')
    assert np.all(np.isfinite(np.asarray(opd, dtype=float)))
    assert opd.shape[0] > 0


def test_wavefront_centroid_matches_chief_when_chief_valid():
    """With a valid chief, reference='centroid' equals reference='chief'."""
    presc = _spherical_singlet()
    P, S = launch(presc, Field(0., 0.), 0.55e-3, Sampling.hex(nrings=3),
                  epd=8.0, pupil_z=-5.0)
    opd_chief, _, _ = wavefront(presc, P, S, 0.55e-3, reference='chief')
    opd_cent, _, _ = wavefront(presc, P, S, 0.55e-3, reference='centroid')
    np.testing.assert_allclose(np.asarray(opd_chief, dtype=float),
                               np.asarray(opd_cent, dtype=float), atol=1e-12)
