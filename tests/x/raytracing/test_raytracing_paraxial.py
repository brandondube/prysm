"""Tests for prysm.x.raytracing.paraxial — ABCD matrix solver."""
import numpy as np
import pytest

from tests.x.raytracing.surface_helpers import (
    plane, sphere, conic, off_axis_conic, even_asphere, q2d, zernike, xy,
    chebyshev, jacobi, toroid, biconic,
)

from prysm.x.raytracing.surfaces import Surface
from prysm.x.raytracing import LensData
from prysm.x.raytracing.surfaces import ConicSag
from prysm.x.raytracing.paraxial import (
    system_matrix,
    paraxial_image_distance,
    effective_focal_length,
    back_focal_length,
    front_focal_length,
    first_order,
    FirstOrderProperties,
)
from prysm.x.raytracing.opt import paraxial_image_solve
from prysm.x.raytracing.auto import rc_prescription_from_efl_bfl_sep


# ---------- system_matrix sanity ----------

def test_system_matrix_single_plane_is_identity():
    """A single plane carries no power and no gap, so M should be I."""
    rx = [plane(typ='eval', P=np.array([0., 0., 0.]))]
    M, n = system_matrix(rx, wvl=0.55)
    np.testing.assert_allclose(M, np.eye(2), atol=1e-12)
    assert n == 1.0


def test_system_matrix_translation_only():
    """Two planes 10 mm apart in n=1: M = [[1, 10], [0, 1]]."""
    rx = [
        plane(typ='eval', P=np.array([0., 0., 0.])),
        plane(typ='eval', P=np.array([0., 0., 10.])),
    ]
    M, n = system_matrix(rx, wvl=0.55)
    np.testing.assert_allclose(M, [[1.0, 10.0], [0.0, 1.0]], atol=1e-12)
    assert n == 1.0


def test_system_matrix_thin_lens_efl():
    """A thin lens of focal length f has system matrix [[1,0],[-1/f,1]]."""
    R1, R2 = 100.0, -100.0
    n_glass = 1.5
    # thin lens: thickness ~ 0
    f_lens = 1.0 / ((n_glass - 1) * (1.0 / R1 - 1.0 / R2))
    rx = [
        sphere(c=1.0 / R1, typ='refr', P=np.array([0., 0., 0.]),
                       n=lambda wvl: n_glass),
        sphere(c=1.0 / R2, typ='refr', P=np.array([0., 0., 1e-9]),
                       n=lambda wvl: 1.0),
    ]
    M, n = system_matrix(rx, wvl=0.55)
    np.testing.assert_allclose(M[0, 0], 1.0, atol=1e-9)
    np.testing.assert_allclose(M[1, 0], -1.0 / f_lens, rtol=1e-6)
    assert n == pytest.approx(1.0)


def test_system_matrix_mirror_flips_sign_of_n():
    """A single mirror: image-space index goes to -1."""
    rx = [
        conic(c=1 / 200.0, k=-1.0, typ='refl', P=np.array([0., 0., 0.])),
    ]
    _, n = system_matrix(rx, wvl=0.55)
    assert n == pytest.approx(-1.0)


# ---------- paraxial_image_distance ----------

def test_image_distance_single_refracting_sphere():
    """Single refracting sphere: image at z = n_after * R / (n_after - n_before)."""
    R = 50.0
    n_glass = 1.5
    expected = n_glass * R / (n_glass - 1.0)
    rx = [
        sphere(c=1.0 / R, typ='refr', P=np.array([0., 0., 0.]),
                       n=lambda wvl: n_glass),
    ]
    bfd = paraxial_image_distance(rx)
    np.testing.assert_allclose(bfd, expected, rtol=1e-12)


def test_image_distance_unchanged_by_eval_plane_after_last_surface():
    """Inserting a downstream eval plane must not move the image position."""
    R = 50.0
    n_glass = 1.5
    rx_base = [
        sphere(c=1.0 / R, typ='refr', P=np.array([0., 0., 0.]),
                       n=lambda wvl: n_glass),
    ]
    rx_with_eval = rx_base + [
        plane(typ='eval', P=np.array([0., 0., 100.])),
    ]
    img_z_base = 0.0 + paraxial_image_distance(rx_base)
    img_z_eval = 100.0 + paraxial_image_distance(rx_with_eval)
    np.testing.assert_allclose(img_z_eval, img_z_base, rtol=1e-12)


def test_image_distance_no_power_raises():
    rx = [plane(typ='eval', P=np.array([0., 0., 0.]))]
    with pytest.raises(ValueError, match='no net power'):
        paraxial_image_distance(rx)


# ---------- agreement with the numerical solver ----------

def test_matrix_vs_numerical_single_sphere():
    rx = [
        sphere(c=1 / 50.0, typ='refr', P=np.array([0., 0., 0.]),
                       n=lambda wvl: 1.5),
        plane(typ='eval', P=np.array([0., 0., 100.])),
    ]
    img_num = paraxial_image_solve(rx, z=0, epd=10.0, method='numerical')
    img_mat = paraxial_image_solve(rx, z=0, method='matrix')
    np.testing.assert_allclose(img_mat[2], img_num[2], rtol=1e-3)


def test_matrix_vs_numerical_rc_telescope():
    """ABCD must agree with the 4-ray solve on an RC to ~1e-3 (relative)."""
    efl, bfl, sep = 1500.0, 250.0, 400.0
    c1, c2, k1, k2 = rc_prescription_from_efl_bfl_sep(efl, bfl, sep)
    P_pm = np.array([0.0, 0.0, 0.0])
    P_sm = np.array([0.0, 0.0, -sep])
    P_img = np.array([0.0, 0.0, bfl - sep])
    rx = [
        conic(c1, k1, 'refl', P_pm),
        conic(c2, k2, 'refl', P_sm),
        plane('eval', P_img),
    ]
    img_num = paraxial_image_solve(rx, z=0, epd=200.0, method='numerical')
    img_mat = paraxial_image_solve(rx, z=0, method='matrix')
    np.testing.assert_allclose(img_mat[2], img_num[2], rtol=5e-3)
    # the matrix answer is the truth here; should land within 1e-9 of the
    # design BFL
    np.testing.assert_allclose(img_mat[2], P_img[2], rtol=1e-9)


def test_matrix_method_unknown_raises():
    rx = [
        sphere(c=1 / 50.0, typ='refr', P=np.array([0., 0., 0.]),
                       n=lambda wvl: 1.5),
    ]
    with pytest.raises(ValueError, match="method must be"):
        paraxial_image_solve(rx, z=0, method='not_a_method')


# ---------- effective focal length ----------

def test_efl_thin_lens_matches_lensmakers():
    R1, R2 = 100.0, -100.0
    n_glass = 1.5
    f_lens = 1.0 / ((n_glass - 1) * (1.0 / R1 - 1.0 / R2))
    rx = [
        sphere(c=1.0 / R1, typ='refr', P=np.array([0., 0., 0.]),
                       n=lambda wvl: n_glass),
        sphere(c=1.0 / R2, typ='refr', P=np.array([0., 0., 1e-9]),
                       n=lambda wvl: 1.0),
    ]
    efl = effective_focal_length(rx)
    np.testing.assert_allclose(efl, f_lens, rtol=1e-6)


def test_lensdata_without_wavelengths_uses_default_wavelength():
    ld = LensData().add(ConicSag(1 / 50.0, 0.0),
                        typ='refr', material=lambda wvl: 1.5)
    np.testing.assert_allclose(effective_focal_length(ld), 100.0)


def test_lensdata_n_ambient_defaults_into_paraxial_calculations():
    ld = LensData(n_ambient=1.33).add(
        ConicSag(1 / 50.0, 0.0), typ='refr', material=lambda wvl: 1.5)
    np.testing.assert_allclose(
        effective_focal_length(ld),
        effective_focal_length(ld, n_ambient=1.33),
    )


def test_efl_rc_telescope_matches_design():
    """RC EFL must match the value used to derive the prescription."""
    efl_design, bfl, sep = 1500.0, 250.0, 400.0
    c1, c2, k1, k2 = rc_prescription_from_efl_bfl_sep(efl_design, bfl, sep)
    rx = [
        conic(c1, k1, 'refl', np.array([0., 0., 0.])),
        conic(c2, k2, 'refl', np.array([0., 0., -sep])),
    ]
    efl = effective_focal_length(rx)
    # signed: the RC design convention may yield negative depending on
    # mirror sign conventions; magnitude is what matches the design value.
    np.testing.assert_allclose(abs(efl), efl_design, rtol=1e-9)


# ---------- back focal length ----------

def test_bfl_matches_image_distance_when_last_surface_is_powered():
    rx = [
        sphere(c=1 / 50.0, typ='refr', P=np.array([0., 0., 0.]),
                       n=lambda wvl: 1.5),
    ]
    bfl = back_focal_length(rx)
    bfd = paraxial_image_distance(rx)
    np.testing.assert_allclose(bfl, bfd, rtol=1e-12)


def test_bfl_unchanged_by_trailing_eval_planes():
    """BFL is from the last *powered* surface; adding eval planes after
    must not change it."""
    rx_base = [
        sphere(c=1 / 50.0, typ='refr', P=np.array([0., 0., 0.]),
                       n=lambda wvl: 1.5),
    ]
    rx_eval = rx_base + [
        plane(typ='eval', P=np.array([0., 0., 50.])),
        plane(typ='eval', P=np.array([0., 0., 75.])),
    ]
    np.testing.assert_allclose(back_focal_length(rx_base),
                               back_focal_length(rx_eval), rtol=1e-12)


# ---------- front focal length ----------

def test_ffl_thin_lens_matches_lensmakers():
    """Thin lens in air: |FFL| == |BFL| == f."""
    R1, R2 = 100.0, -100.0
    n_glass = 1.5
    f_lens = 1.0 / ((n_glass - 1) * (1.0 / R1 - 1.0 / R2))
    rx = [
        sphere(c=1.0 / R1, typ='refr', P=np.array([0., 0., 0.]),
                       n=lambda wvl: n_glass),
        sphere(c=1.0 / R2, typ='refr', P=np.array([0., 0., 1e-9]),
                       n=lambda wvl: 1.0),
    ]
    ffl = front_focal_length(rx)
    np.testing.assert_allclose(abs(ffl), f_lens, rtol=1e-6)


def test_ffl_unchanged_by_leading_eval_planes():
    """FFL is from the first powered surface; a leading eval plane must
    not change it."""
    rx_base = [
        sphere(c=1 / 50.0, typ='refr', P=np.array([0., 0., 10.]),
                       n=lambda wvl: 1.5),
    ]
    rx_eval = [
        plane(typ='eval', P=np.array([0., 0., 0.])),
    ] + rx_base
    np.testing.assert_allclose(front_focal_length(rx_base),
                               front_focal_length(rx_eval), rtol=1e-12)


def test_ffl_no_power_raises():
    rx = [plane(typ='eval', P=np.array([0., 0., 0.]))]
    with pytest.raises(ValueError, match='no powered surfaces'):
        front_focal_length(rx)


# ---------- first_order report ----------

def _thin_lens_prescription(R1=100.0, R2=-100.0, n_glass=1.5, z0=0.0):
    return [
        sphere(c=1.0 / R1, typ='refr', P=np.array([0., 0., z0]),
                       n=lambda wvl: n_glass),
        sphere(c=1.0 / R2, typ='refr',
                       P=np.array([0., 0., z0 + 1e-9]),
                       n=lambda wvl: 1.0),
    ]


def test_first_order_returns_carrier_with_basics():
    rx = _thin_lens_prescription()
    fo = first_order(rx, wvl=0.55)
    assert isinstance(fo, FirstOrderProperties)
    assert fo.n_surfaces == 2
    assert fo.n_refractive == 2
    assert fo.n_reflective == 0
    assert fo.n_eval == 0
    assert fo.n_image == pytest.approx(1.0)
    np.testing.assert_allclose(fo.efl, effective_focal_length(rx, wvl=0.55),
                               rtol=1e-12)
    np.testing.assert_allclose(fo.bfl, back_focal_length(rx, wvl=0.55),
                               rtol=1e-12)
    np.testing.assert_allclose(fo.ffl, front_focal_length(rx, wvl=0.55),
                               rtol=1e-12)
    np.testing.assert_allclose(fo.paraxial_image_distance,
                               paraxial_image_distance(rx, wvl=0.55),
                               rtol=1e-12)
    # no pupil info without stop_index/epd
    assert fo.fno is None
    assert fo.ep_z is None
    assert fo.xp_z is None
    assert fo.stop_diameter is None


def test_first_order_fno_and_na():
    rx = _thin_lens_prescription()
    epd = 25.0
    fo = first_order(rx, wvl=0.55, epd=epd)
    assert fo.epd == pytest.approx(epd)
    assert fo.fno == pytest.approx(abs(fo.efl) / epd)
    # NA = sin(theta) ~ (epd/2) / |bfl| for a thin lens with object at infinity
    expected_na = (epd / 2.0) / abs(fo.efl)
    np.testing.assert_allclose(fo.na_image, expected_na, rtol=1e-6)


def test_first_order_stop_at_thin_lens_places_pupils_at_lens():
    rx = _thin_lens_prescription()
    fo = first_order(rx, wvl=0.55, epd=20.0, stop_index=0)
    # stop coincident with the first (and effectively only) lens vertex,
    # so the EP and XP are both at the lens.
    assert fo.ep_z == pytest.approx(0.0, abs=1e-9)
    assert fo.xp_z == pytest.approx(0.0, abs=1e-6)
    assert fo.ep_diameter == pytest.approx(20.0)
    assert fo.stop_diameter == pytest.approx(20.0, rel=1e-6)
    assert fo.xp_diameter == pytest.approx(20.0, rel=1e-6)


def test_first_order_stop_behind_single_lens_places_ep_in_front():
    """Aperture stop placed behind a single positive lens at t < f gives a
    virtual EP on the image side of the lens (z_EP > 0) and the EP is
    larger than the stop."""
    R1, R2 = 100.0, -100.0
    n_glass = 1.5
    f = 1.0 / ((n_glass - 1) * (1.0 / R1 - 1.0 / R2))
    t = 0.25 * f
    rx = _thin_lens_prescription(R1=R1, R2=R2, n_glass=n_glass) + [
        plane(typ='eval', P=np.array([0., 0., t])),
    ]
    stop_diameter = 10.0
    # Pretend EPD is the value that would put a stop_diameter=10 aperture
    # at the stop plane; A_b for this geometry = 1 - t/f.
    A_b = 1.0 - t / f
    epd = stop_diameter / abs(A_b)
    fo = first_order(rx, wvl=0.55, epd=epd, stop_index=2)
    # M_to_stop = T(t) @ thin_lens(P=1/f) → EP at z = +t/(1 - t/f) measured
    # from S1. For t = f/4: EP at z = (f/4)/(3/4) = f/3 (downstream of S1).
    np.testing.assert_allclose(fo.ep_z, f / 3.0, rtol=1e-4)
    np.testing.assert_allclose(fo.stop_diameter, stop_diameter, rtol=1e-6)


def test_first_order_repr_lists_populated_rows_only():
    rx = _thin_lens_prescription()
    fo = first_order(rx, wvl=0.55)
    s = repr(fo)
    assert 'EFL' in s
    assert 'BFL' in s
    # epd-only rows must be absent when epd was not supplied
    assert 'F/#' not in s
    assert 'EPD' not in s
    assert 'stop diameter' not in s


def test_first_order_afocal_returns_none_for_power_dependent_fields():
    rx = [plane(typ='eval', P=np.array([0., 0., 0.])),
          plane(typ='eval', P=np.array([0., 0., 10.]))]
    fo = first_order(rx)
    assert fo.efl is None
    assert fo.bfl is None
    assert fo.ffl is None
    assert fo.paraxial_image_distance is None
    assert fo.total_track == pytest.approx(10.0)


def test_first_order_stop_index_out_of_range_raises():
    rx = _thin_lens_prescription()
    with pytest.raises(IndexError):
        first_order(rx, stop_index=5)
