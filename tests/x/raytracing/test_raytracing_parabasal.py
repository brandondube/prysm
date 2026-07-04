"""Tests for prysm.x.raytracing.parabasal — firABCD-style first order."""
import warnings

import numpy as np
import pytest

from prysm.x import materials
from prysm.x.raytracing import (
    LensData, OpticalSystem, Field, Sampling, launch, raytrace,
)
from prysm.x.raytracing.surfaces import Sphere, Conic, Plane
from prysm.x.raytracing.launch import _perp_basis
from prysm.x.raytracing.paraxial import ynu_first_order
from prysm.x.raytracing.parabasal import (
    ParabasalFirstOrder, first_order, parabasal_foci, _PAIR_SLOTS,
)
from prysm.x.raytracing._diff_raytrace import DiffSeed, raytrace_with_tangents


# ---------- builders ---------------------------------------------------------

def _singlet_system(aperture_radius=None):
    ld = LensData()
    ld.add(Sphere(1 / 100.0), thickness=4,
           material=materials.ConstantMaterial(1.52),
           aperture=(aperture_radius if aperture_radius is not None else 12))
    ld.add(Sphere(-1 / 100.0), thickness=92, material=materials.air,
           aperture=12)
    ld.add(Plane(), typ='eval', aperture=30)
    return OpticalSystem(ld, stop_index=1, wavelengths=[0.55])


def _parabola_system():
    ld = LensData()
    ld.add(Conic(-1 / 400.0, -1.0), thickness=-200, typ='refl',
           aperture=30)
    ld.add(Plane(), typ='eval', aperture=5)
    return OpticalSystem(ld, stop_index=0, wavelengths=[0.55])


def _two_mirror_system():
    ld = LensData()
    ld.add(Conic(-1 / 400.0, -1.0), thickness=-80, typ='refl',
           aperture=30)
    ld.add(Conic(-1 / 150.0, -3.0), thickness=200, typ='refl',
           aperture=8)
    ld.add(Plane(), typ='eval', aperture=5)
    return OpticalSystem(ld, stop_index=0, wavelengths=[0.55])


def _finite_conjugate_system():
    ld = LensData()
    ld.add(Plane(), thickness=300, typ='eval', aperture=1)
    ld.add(Sphere(1 / 100.0), thickness=4,
           material=materials.ConstantMaterial(1.52), aperture=12)
    ld.add(Sphere(-1 / 100.0), thickness=140, material=materials.air,
           aperture=12)
    ld.add(Plane(), typ='eval', aperture=30)
    return OpticalSystem(ld, stop_index=2, wavelengths=[0.55],
                         fields=[Field(0, 0, kind='height', object_z=0.0)])


def _decentered_singlet_system(dy=0.4):
    ld = LensData()
    ld.add(Sphere(1 / 100.0), thickness=4,
           material=materials.ConstantMaterial(1.52), aperture=12)
    ld.add_coordbreak(decenter=(0.0, dy, 0.0))
    ld.add(Sphere(-1 / 100.0), thickness=92, material=materials.air,
           aperture=12)
    ld.add(Plane(), typ='eval', aperture=30)
    return OpticalSystem(ld, stop_index=1, wavelengths=[0.55])


def _assert_pairs_match_ynu(fo_p, fo_y, rtol=1e-9, skip=()):
    assert fo_p.backend == 'parabasal'
    for name in _PAIR_SLOTS:
        if name in skip:
            continue
        vy = getattr(fo_y, name)
        vp = getattr(fo_p, name)
        if vy is None:
            assert vp is None, name
            continue
        assert vp is not None, name
        np.testing.assert_allclose(vp[0], vy, rtol=rtol, atol=1e-9,
                                   err_msg=f'{name} x section')
        np.testing.assert_allclose(vp[1], vy, rtol=rtol, atol=1e-9,
                                   err_msg=f'{name} y section')


# ---------- _perp_basis: meridional T/S pinning ------------------------------

def test_perp_basis_axial_returns_lab_axes():
    w = np.array([0.0, 0.0, 1.0])
    e1, e2 = _perp_basis(w)
    np.testing.assert_allclose(e1, [1.0, 0.0, 0.0])
    np.testing.assert_allclose(e2, [0.0, 1.0, 0.0])


def test_perp_basis_backward_axial_is_right_handed():
    w = np.array([0.0, 0.0, -1.0])
    e1, e2 = _perp_basis(w)
    np.testing.assert_allclose(np.cross(e1, e2), w, atol=1e-12)


def test_perp_basis_y_meridian_continuous_both_signs():
    for s in (0.3, -0.3):
        w = np.array([0.0, s, np.sqrt(1 - s * s)])
        e1, e2 = _perp_basis(w)
        # sagittal axis is +x for the whole y-z meridian
        np.testing.assert_allclose(e1, [1.0, 0.0, 0.0], atol=1e-12)
        # tangential axis lies in the meridional plane
        assert abs(float(e2[0])) < 1e-12
        np.testing.assert_allclose(np.cross(e1, e2), w, atol=1e-12)


def test_perp_basis_skew_chief_is_ts_pure():
    # sagittal vector has no z component for any skew chief
    w = np.array([0.25, 0.35, 0.0])
    w[2] = np.sqrt(1 - np.sum(w * w))
    e1, e2 = _perp_basis(w)
    assert abs(float(e1[2])) < 1e-12
    assert abs(float(e1 @ w)) < 1e-12
    assert abs(float(e2 @ w)) < 1e-12
    np.testing.assert_allclose(np.cross(e1, e2), w, atol=1e-12)


# ---------- launch tangent seeds vs finite differences ------------------------

def test_launch_tangent_seeds_match_central_differences():
    sys = _singlet_system()
    surfs = sys.to_surfaces()
    fld = Field(0, 7.0)
    P0, S0 = launch(sys, fld, 0.55, Sampling.chief())
    e1, e2 = _perp_basis(S0[0])
    zero = np.zeros(3)
    Pdot0 = np.stack([e1, e2, zero, zero], axis=-1)[None]
    Sdot0 = np.stack([zero, zero, e1, e2], axis=-1)[None]
    seeds = [DiffSeed(name=n) for n in ('dx', 'dy', 'du', 'dv')]
    res = raytrace_with_tangents(surfs, P0, S0, 0.55, seeds,
                                 Pdot0=Pdot0, Sdot0=Sdot0)
    h = 1e-6
    for col, (dP, dS) in enumerate([(e1, None), (e2, None),
                                    (None, e1), (None, e2)]):
        if dP is not None:
            Pp, Sp, Pm, Sm = P0 + h * dP, S0, P0 - h * dP, S0
        else:
            Sp = np.cos(h) * S0 + np.sin(h) * dS
            Sm = np.cos(h) * S0 - np.sin(h) * dS
            Pp = Pm = P0
        tp = raytrace(surfs, Pp, Sp, 0.55)
        tm = raytrace(surfs, Pm, Sm, 0.55)
        fd_P = (tp.P[-1, 0] - tm.P[-1, 0]) / (2 * h)
        fd_S = (tp.S[-1, 0] - tm.S[-1, 0]) / (2 * h)
        np.testing.assert_allclose(res.Pdot[-1, 0, :, col], fd_P, atol=1e-7)
        np.testing.assert_allclose(res.Sdot[-1, 0, :, col], fd_S, atol=1e-7)


def test_launch_tangent_seeds_shape_validated():
    sys = _singlet_system()
    surfs = sys.to_surfaces()
    P0, S0 = launch(sys, Field(0, 0), 0.55, Sampling.chief())
    seeds = [DiffSeed(name='dx')]
    with pytest.raises(ValueError, match='shape'):
        raytrace_with_tangents(surfs, P0, S0, 0.55, seeds,
                               Pdot0=np.zeros((1, 3, 2)))


# ---------- parity with the YNU walk -----------------------------------------

def test_parabasal_matches_ynu_singlet():
    sys = _singlet_system()
    fo_y = ynu_first_order(sys.to_surfaces(), wvl=0.55, epd=20, stop_index=1)
    fo_p = first_order(sys, wavelength=0.55, epd=20, stop_index=1)
    assert isinstance(fo_p, ParabasalFirstOrder)
    _assert_pairs_match_ynu(fo_p, fo_y)
    assert fo_p.n_object == pytest.approx(fo_y.n_object)
    assert fo_p.n_image == pytest.approx(fo_y.n_image)
    assert fo_p.abcd.shape == (4, 4)


def test_parabasal_matches_ynu_single_mirror_signs():
    sys = _parabola_system()
    fo_y = ynu_first_order(sys.to_surfaces(), wvl=0.55, epd=50, stop_index=0)
    fo_p = first_order(sys, wavelength=0.55, epd=50, stop_index=0)
    # signs included: efl positive for the converging mirror in both sections
    _assert_pairs_match_ynu(fo_p, fo_y)
    assert fo_p.efl[0] > 0 and fo_p.efl[1] > 0
    assert fo_p.n_image == pytest.approx(-1.0)


def test_parabasal_matches_ynu_two_mirror():
    sys = _two_mirror_system()
    fo_y = ynu_first_order(sys.to_surfaces(), wvl=0.55, epd=50, stop_index=0)
    fo_p = first_order(sys, wavelength=0.55, epd=50, stop_index=0)
    _assert_pairs_match_ynu(fo_p, fo_y)


def test_parabasal_finite_conjugate_image_is_conjugate_correct():
    # the YNU walk always reports the collimated-input (rear focal) image;
    # the parabasal reports the image of the actual finite object
    sys = _finite_conjugate_system()
    fo_y = ynu_first_order(sys.to_surfaces(), wvl=0.55, epd=20, stop_index=2)
    fo_p = first_order(sys, wavelength=0.55, epd=20, stop_index=2)
    _assert_pairs_match_ynu(
        fo_p, fo_y, skip=('paraxial_image_z', 'paraxial_image_distance'))
    # thin-lens conjugate from the principal planes: 1/i = 1/f - 1/o
    f = fo_p.efl[1]
    o = 300.0 + (f - fo_y.ffl)               # object to front principal plane
    i = 1.0 / (1.0 / f - 1.0 / o)
    z_h_rear = (304.0 + fo_y.bfl) - f        # rear principal plane z
    np.testing.assert_allclose(fo_p.paraxial_image_z[1], z_h_rear + i,
                               rtol=1e-9)


def test_parabasal_force_sym_scalars():
    sys = _singlet_system()
    fo = first_order(sys, wavelength=0.55, epd=20, force_sym=True)
    fo_y = ynu_first_order(sys.to_surfaces(), wvl=0.55, epd=20, stop_index=1)
    assert isinstance(fo.efl, float)
    np.testing.assert_allclose(fo.efl, fo_y.efl, rtol=1e-9)
    np.testing.assert_allclose(fo.xp_z, fo_y.xp_z, rtol=1e-9)


def test_parabasal_stop_index_out_of_range_raises():
    sys = _singlet_system()
    with pytest.raises(IndexError):
        first_order(sys, wavelength=0.55, stop_index=7)


def test_first_order_bare_surfaces_defaults_to_on_axis():
    # Bare compiled surfaces with no explicit field default to on-axis.
    sys = _singlet_system()
    surfs = sys.to_surfaces()
    fo = first_order(surfs, wavelength=0.55, epd=20, stop_index=1)
    assert fo.field.hx == pytest.approx(0.0)
    assert fo.field.hy == pytest.approx(0.0)
    fo_y = ynu_first_order(surfs, wvl=0.55, epd=20, stop_index=1)
    _assert_pairs_match_ynu(fo, fo_y)


def test_system_field_indices_are_authoritative():
    sys = _singlet_system()
    sys.fields.fields = [Field(0, 0), Field(0, 7.0)]

    fo_index = first_order(sys, field=1, wavelength=0.55, epd=20)
    assert fo_index.field is sys.field(1)
    assert fo_index.field.hy == pytest.approx(7.0)

    fo_literal = first_order(sys, field=(0.0, 1.0), wavelength=0.55, epd=20)
    assert fo_literal.field.hy == pytest.approx(1.0)

    # A bare float is neither an index nor a literal field.
    with pytest.raises(TypeError):
        first_order(sys, field=1.0, wavelength=0.55, epd=20)

    with pytest.raises(IndexError):
        first_order(sys, field=7, wavelength=0.55, epd=20)
    with pytest.raises(IndexError):
        parabasal_foci(sys, 7, 0.55)


def test_raw_prescription_accepts_tuple_field_literals():
    sys = _singlet_system()
    surfaces = sys.to_surfaces()

    fo = first_order(surfaces, field=(1.0, 2.0), wavelength=0.55,
                     epd=20, stop_index=1)
    assert fo.field.hx == pytest.approx(1.0)
    assert fo.field.hy == pytest.approx(2.0)

    x_z, y_z = parabasal_foci(surfaces, (1.0, 2.0), 0.55)
    assert np.isfinite(x_z)
    assert np.isfinite(y_z)


# ---------- where the YNU walk cannot go --------------------------------------

def test_parabasal_handles_decentered_geometry():
    sys = _decentered_singlet_system()
    with pytest.raises(ValueError, match='centered axial geometry'):
        ynu_first_order(sys.to_surfaces(), wvl=0.55, epd=20, stop_index=1)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # paraxial-aiming launch warning
        fo = first_order(sys, wavelength=0.55, epd=20, stop_index=1)
    assert fo.backend == 'parabasal'
    # a 0.4 mm element decenter perturbs but does not destroy the design
    np.testing.assert_allclose(fo.efl[0], 96.8163, rtol=1e-3)
    np.testing.assert_allclose(fo.efl[1], 96.8163, rtol=1e-3)


def test_parabasal_off_axis_field_splits_ts_foci():
    sys = _singlet_system()
    fo0 = first_order(sys, field=Field(0, 0), wavelength=0.55, epd=20)
    fo7 = first_order(sys, field=Field(0, 7.0), wavelength=0.55, epd=20)
    # on axis the sections coincide; off axis astigmatism splits them
    np.testing.assert_allclose(fo0.paraxial_image_z[0],
                               fo0.paraxial_image_z[1], rtol=1e-12)
    assert abs(fo7.paraxial_image_z[0] - fo7.paraxial_image_z[1]) > 0.1


# ---------- the YNU fallback ---------------------------------------------------

def test_parabasal_falls_back_to_ynu_when_chief_clipped():
    sys = _singlet_system(aperture_radius=2.0)
    fo = first_order(sys, field=Field(0, 60.0), wavelength=0.55, epd=20)
    fo_y = ynu_first_order(sys.to_surfaces(), wvl=0.55, epd=20, stop_index=1)
    assert fo.backend == 'ynu'
    assert fo.efl == (fo_y.efl, fo_y.efl)
    assert fo.abcd is None


def test_90_degree_field_corridor_known_limitation():
    # exactly 90 deg leaves the chief non-finite, so first_order falls back
    sys = _singlet_system()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        fo = first_order(sys, field=Field(0, 90.0), wavelength=0.55, epd=20)
    assert fo.backend == 'ynu'


# ---------- parabasal_foci -----------------------------------------------------

def test_parabasal_foci_on_axis_match_paraxial_image():
    sys = _singlet_system()
    fo_y = ynu_first_order(sys.to_surfaces(), wvl=0.55, epd=20, stop_index=1)
    x_z, y_z = parabasal_foci(sys, Field(0, 0), 0.55)
    np.testing.assert_allclose(x_z, fo_y.paraxial_image_z, rtol=1e-9)
    np.testing.assert_allclose(y_z, fo_y.paraxial_image_z, rtol=1e-9)


# ---------- launch warning ------------------------------------------------------

def test_launch_warns_on_decentered_with_paraxial_aiming():
    sys = _decentered_singlet_system()
    with pytest.warns(UserWarning, match='tilts/decenters'):
        launch(sys, Field(0, 1.0), 0.55, Sampling.fan(5), epd=10)


def test_launch_does_not_warn_on_centered_system():
    sys = _singlet_system()
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        launch(sys, Field(0, 1.0), 0.55, Sampling.fan(5), epd=10)


def test_launch_does_not_warn_with_real_aiming():
    sys = _decentered_singlet_system()
    sys.ray_aiming = 'real'
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        launch(sys, Field(0, 1.0), 0.55, Sampling.fan(5), epd=10)
