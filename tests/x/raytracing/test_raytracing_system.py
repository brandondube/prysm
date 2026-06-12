"""OpticalSystem / ApertureSpec / FieldSet and the tabular listings.

Covers the two-type split: the aperture spec mode conversions and validation,
media-from-surfaces, the surface/aperture/decenter listings, and the IO
vignetting-ignored warning.
"""
import numpy as np
import pytest

from prysm.x.raytracing import (
    LensData, OpticalSystem, ApertureSpec, FieldSet, Field,
    FRAUNHOFER_LINES_UM,
)
from prysm.x import materials
from prysm.x.raytracing.system import (
    EPD, FNO_IMAGE, NA_IMAGE, NA_OBJECT, FNO_OBJECT,
)
from prysm.x.raytracing.surfaces import Conic, Plane
from prysm.x.raytracing import launch, Sampling, raytrace
from prysm.x.raytracing.paraxial import entrance_pupil_z
from prysm.x.raytracing.io._common import warn_vignetting_ignored


def _singlet(aperture=ApertureSpec.epd(20.0), with_object=None):
    ld = LensData()
    if with_object is not None:
        ld.add(Plane(), typ='eval', material=materials.ConstantMaterial(with_object),
               thickness=200.0)
    (ld.add(Conic(1 / 102.0, 0.0), thickness=6.0, material=materials.ConstantMaterial(1.5168),
            semidiameter=12.0)
       .add(Conic(-1 / 102.0, 0.0), thickness=95.0, material=materials.air,
            semidiameter=12.0)
       .add(Plane(), typ='eval', material=materials.air, semidiameter=12.0))
    stop = 1 if with_object is not None else 0
    return OpticalSystem(ld, aperture=aperture,
                         wavelengths=list(FRAUNHOFER_LINES_UM.values()),
                         reference=1, stop_index=stop, unit='mm')


def _afocal(aperture):
    ld = LensData()
    (ld.add(Plane(), typ='eval', material=materials.air, thickness=10.0)
       .add(Plane(), typ='refr', material=materials.air, thickness=10.0)
       .add(Plane(), typ='eval', material=materials.air))
    return OpticalSystem(ld, aperture=aperture,
                         wavelengths=list(FRAUNHOFER_LINES_UM.values()),
                         reference=1, stop_index=1, unit='mm')


# ---------- ApertureSpec ----------------------------------------------------

def test_aperture_spec_modes_and_factories():
    assert ApertureSpec.epd(10).mode == EPD
    assert ApertureSpec.fno(4.0).mode == FNO_IMAGE
    assert ApertureSpec.fno(4.0, object_space=True).mode == FNO_OBJECT
    assert ApertureSpec.na(0.1).mode == NA_IMAGE
    assert ApertureSpec.na(0.1, object_space=True).mode == NA_OBJECT
    with pytest.raises(ValueError, match='aperture mode'):
        ApertureSpec(1.0, mode='nonsense')


def test_aperture_epd_resolves_directly():
    sys = _singlet(ApertureSpec.epd(20.0))
    # resolve returns the tagged definition; the first-order EPD readout is 20.
    assert sys.aperture.resolve(sys) == (EPD, 20.0)
    assert sys.aperture.entrance_pupil_diameter(sys) == pytest.approx(20.0)
    assert sys.epd == pytest.approx(20.0)


def test_aperture_fno_and_na_image_round_trip_against_first_order():
    # read an EPD, read back F/# and NA via first_order, re-derive the EPD via
    # the first-order readout: identity.
    sys = _singlet(ApertureSpec.epd(20.0))
    fo = sys.first_order()
    epd_from_fno = ApertureSpec.fno(fo.fno).entrance_pupil_diameter(sys)
    epd_from_na = ApertureSpec.na(fo.na_image).entrance_pupil_diameter(sys)
    np.testing.assert_allclose(epd_from_fno, 20.0, rtol=1e-9)
    np.testing.assert_allclose(epd_from_na, 20.0, rtol=1e-9)


def test_object_space_aperture_illegal_at_infinity():
    spec = ApertureSpec.na(0.1, object_space=True)
    with pytest.raises(ValueError, match='object-space'):
        spec.validate(object_at_infinity=True)
    # the same spec is legal for a finite object
    spec.validate(object_at_infinity=False)


def test_object_space_aperture_validation_is_enforced_at_infinity():
    sys = _singlet(ApertureSpec.na(0.1, object_space=True))
    assert sys.object_at_infinity is True
    with pytest.raises(ValueError, match='object-space'):
        sys.aperture.resolve(sys)
    with pytest.raises(ValueError, match='object-space'):
        _ = sys.epd
    fld = Field(0.0, 1.0, kind='height', object_z=-10.0)
    with pytest.raises(ValueError, match='object-space'):
        launch(sys, fld, sys.wavelength(), Sampling.fan(n=3))


def test_focusing_aperture_illegal_for_afocal():
    spec = ApertureSpec.fno(4.0)
    with pytest.raises(ValueError, match='no net power'):
        spec.validate(object_at_infinity=True, has_power=False)


def test_focusing_apertures_raise_for_afocal_system():
    specs = [
        ApertureSpec.fno(4.0),
        ApertureSpec.fno(4.0, object_space=True),
        ApertureSpec.na(0.1),
        ApertureSpec.na(0.1, object_space=True),
    ]
    for spec in specs:
        sys = _afocal(spec)
        with pytest.raises(ValueError, match='no net power'):
            spec.resolve(sys)
        with pytest.raises(ValueError, match='no net power'):
            _ = sys.epd


def test_object_space_na_resolves_to_positive_epd_finite_conjugate():
    sys = _singlet(ApertureSpec.na(0.05, object_space=True), with_object=1.0)
    assert sys.aperture.resolve(sys) == (NA_OBJECT, 0.05)
    epd = sys.aperture.entrance_pupil_diameter(sys)
    assert epd > 0.0


# ---------- object-space NA real-ray launch (Phase 9) -----------------------

def test_object_space_na_launch_honors_sine_condition():
    """The launched object cone's marginal ray obeys n_object*sin(U)=NA."""
    na = 0.1
    sys = _singlet(ApertureSpec.na(na, object_space=True), with_object=1.0)
    z_obj = float(sys[0].P[2])
    fld = Field(0.0, 0.0, kind='height', object_z=z_obj)
    P, S = launch(sys, fld, sys.wavelength(), Sampling.fan(n=11, axis='y'))
    # on-axis: every ray emanates from the one object point
    np.testing.assert_allclose(P[:, 2], z_obj)
    np.testing.assert_allclose(P[:, :2], 0.0, atol=1e-12)
    # the marginal (rim) ray's object-space direction sine equals NA / n_object
    sin_marg = float(np.max(np.hypot(S[:, 0], S[:, 1])))
    np.testing.assert_allclose(1.0 * sin_marg, na, rtol=1e-6)


def test_object_space_na_marginal_fills_stop_at_na_radius():
    """The aimed cone threads the stop: the chief lands at the stop center and
    the marginal lands at the NA-implied (paraxial) radius, not the stop's full
    clear aperture."""
    na = 0.05
    sys = _singlet(ApertureSpec.na(na, object_space=True), with_object=1.0)
    z_obj = float(sys[0].P[2])
    fld = Field(0.0, 0.0, kind='height', object_z=z_obj)
    P, S = launch(sys, fld, sys.wavelength(), Sampling.fan(n=11, axis='y'))
    tr = raytrace(sys, P, S, sys.wavelength())
    y_stop = tr.P[sys.stop_index + 1, :, 1]  # +1: history offset for launch row
    # chief (center sample) crosses the stop on axis
    np.testing.assert_allclose(y_stop[len(y_stop) // 2], 0.0, atol=1e-9)
    # marginal height at the stop sits at the NA-implied (paraxial) radius to
    # within the real-ray/paraxial (sine-vs-tangent) deviation, far short of
    # the 12 mm clear aperture -- the NA sets the cone, not the stop size.
    epd = sys.aperture.entrance_pupil_diameter(sys)
    np.testing.assert_allclose(np.max(np.abs(y_stop)), epd / 2.0, rtol=1e-2)


def test_object_space_na_low_na_matches_paraxial_epd():
    """At low NA the real-ray entrance-pupil footprint matches the paraxial
    entrance-pupil diameter (sine and tangent of U agree as U -> 0)."""
    na = 0.005
    sys = _singlet(ApertureSpec.na(na, object_space=True), with_object=1.0)
    z_obj = float(sys[0].P[2])
    z_ep = entrance_pupil_z(sys, sys.wavelength())
    fld = Field(0.0, 0.0, kind='height', object_z=z_obj)
    P, S = launch(sys, fld, sys.wavelength(), Sampling.fan(n=5, axis='y'))
    # propagate the +y rim ray in object space to the entrance-pupil plane
    rim = int(np.argmax(S[:, 1]))
    y_ep = P[rim, 1] + (z_ep - z_obj) * S[rim, 1] / S[rim, 2]
    real_epd = 2.0 * y_ep
    np.testing.assert_allclose(real_epd,
                               sys.aperture.entrance_pupil_diameter(sys),
                               rtol=1e-4)


def test_object_space_na_requires_finite_conjugate_field():
    """An object-space aperture cannot launch a collimated (angle) field."""
    sys = _singlet(ApertureSpec.na(0.1, object_space=True), with_object=1.0)
    with pytest.raises(ValueError, match='finite-'):
        launch(sys, Field(0.0, 0.0, kind='angle'), sys.wavelength(),
               Sampling.fan(n=5))


# ---------- media from surfaces --------------------------------------------

def test_object_index_from_object_surface_material():
    air = _singlet(with_object=None).first_order()
    water = _singlet(with_object=1.33).first_order()
    assert air.n_object == pytest.approx(1.0)
    assert water.n_object == pytest.approx(1.33)
    assert water.n_image == pytest.approx(1.0)   # air image space


# ---------- OpticalSystem behavior -----------------------------------------

def test_optical_system_sequence_delegation():
    sys = _singlet()
    assert len(sys) == len(sys.lens)
    assert list(sys)[0] is sys.lens[0]
    assert sys.to_surfaces() is sys.lens.to_surfaces()


def test_system_copy_is_independent():
    sys = _singlet()
    clone = sys.copy()
    clone.lens.rows[0].thickness = 999.0
    assert sys.lens.rows[0].thickness != 999.0
    assert clone.stop_index == sys.stop_index


def test_fieldset_repr_lists_fields():
    fs = FieldSet([0.0, 1.0, (0.5, 2.0)])
    text = repr(fs)
    assert 'FieldSet' in text
    assert len(fs) == 3


# ---------- listings --------------------------------------------------------

def test_surface_table_marks_stop_and_formats_radius():
    sys = _singlet()
    table = sys.list_surfaces()
    text = repr(table)
    assert 'SurfaceTable' in text
    assert '[mm]' in text
    assert 'inf' in text          # the flat image plane
    # the stop row carries a marker
    stop_record = table.records[sys.stop_index]
    assert stop_record['stop'] is True


def test_surface_table_marks_compiled_stop_after_coordbreak():
    ld = LensData()
    ld.add_coordbreak(decenter=(1.0, 0.0, 0.0))
    ld.add(Plane(), typ='eval')
    sys = OpticalSystem(ld, stop_index=0)
    table = sys.list_surfaces()
    assert table.records[0]['stop'] is False
    assert table.records[0]['surface_index'] is None
    assert table.records[1]['stop'] is True
    assert table.records[1]['surface_index'] == 0


def test_aperture_table_reports_semidiameters():
    sys = _singlet()
    table = sys.list_apertures()
    assert repr(table).startswith('ApertureTable')
    assert table.records[0]['semidiameter'] == pytest.approx(12.0)


def test_decenter_table_lists_coordinate_breaks():
    ld = LensData()
    ld.add_coordbreak(decenter=(1.0, 2.0, 0.0), tilt=(0.0, 3.0, 0.0),
                      kind='basic')
    ld.add(Plane(), typ='eval')
    sys = OpticalSystem(ld)
    table = sys.list_decenters()
    assert len(table.records) == 1
    rec = table.records[0]
    assert rec['dx'] == pytest.approx(1.0)
    assert rec['ry'] == pytest.approx(3.0)
    assert rec['kind'] == 'basic'
    # a system without coordinate breaks reports the empty table
    assert 'no coordinate breaks' in repr(_singlet().list_decenters())


# ---------- IO vignetting warning ------------------------------------------

def test_vignetting_warning_fires_only_when_present():
    with pytest.warns(UserWarning, match='vignetting'):
        warn_vignetting_ignored('S 0.02 5.0\nVUY 0.1 0.2\n', 'Code V')
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('error')  # any warning becomes an error
        warn_vignetting_ignored('S 0.02 5.0\nTHI 3.0\n', 'Code V')  # no warn


# ---------- exit-pupil resolution + version-stamped cache -------------------

def test_exit_pupil_matches_first_order_and_caches():
    sys = _singlet()
    wvl = sys.wavelength()
    P_xp = sys.exit_pupil(wvl)
    np.testing.assert_allclose(P_xp[2], sys.first_order(wvl).xp_z)
    np.testing.assert_allclose(np.asarray(P_xp[:2], dtype=float), 0.0)
    # repeat call is a cache hit -> the very same array object
    assert sys.exit_pupil(wvl) is P_xp


def test_exit_pupil_cache_invalidated_by_lens_edit():
    sys = _singlet()
    wvl = sys.wavelength()
    first = sys.exit_pupil(wvl)
    v0 = sys.lens._version
    # an edit through the lens bumps the version and so misses the cache
    sys.lens.rows[0].thickness = float(sys.lens.rows[0].thickness) + 1.0
    assert sys.lens._version > v0
    second = sys.exit_pupil(wvl)
    assert second is not first


def test_exit_pupil_cache_keyed_by_stop_index():
    sys = _singlet()
    wvl = sys.wavelength()
    xp0 = sys.exit_pupil(wvl)
    # stop_index is a plain slot write (no hook), so it is part of the cache
    # key: reassigning it must not return the stale pupil.
    sys.stop_index = 1
    xp1 = sys.exit_pupil(wvl)
    assert xp1 is not xp0


def test_exit_pupil_cache_key_includes_geometric_field_metadata():
    from prysm.x.raytracing.analysis import resolve_exit_pupil
    sys = _singlet()
    sys.stop_index = None
    wvl = sys.wavelength()
    fdeg = Field(0.1, 0.0, unit='deg')
    frad = Field(0.1, 0.0, unit='rad')

    direct_deg = resolve_exit_pupil(sys, wvl, field=fdeg)
    direct_rad = resolve_exit_pupil(sys, wvl, field=frad)
    assert not np.allclose(direct_deg, direct_rad)

    cached_deg = sys.exit_pupil(wvl, field=fdeg)
    cached_rad = sys.exit_pupil(wvl, field=frad)
    np.testing.assert_allclose(cached_deg, direct_deg)
    np.testing.assert_allclose(cached_rad, direct_rad)
    assert cached_rad is not cached_deg


def test_resolve_exit_pupil_paraxial_branch_field_independent():
    from prysm.x.raytracing.analysis import resolve_exit_pupil
    sys = _singlet()
    wvl = sys.wavelength()
    on_axis = resolve_exit_pupil(sys, wvl, field=Field(0.0, 0.0))
    off_axis = resolve_exit_pupil(sys, wvl, field=Field(0.0, 5.0))
    # the paraxial exit pupil depends only on (lens, wavelength), not field
    np.testing.assert_allclose(np.asarray(on_axis, dtype=float),
                               np.asarray(off_axis, dtype=float), atol=1e-12)


# ---------- wavelength model ------------------------------------------------

def test_wavelengths_are_a_flat_micron_array():
    sys = OpticalSystem(LensData(), wavelengths=[0.486, 0.587, 0.656])
    assert isinstance(sys.wavelengths, np.ndarray)
    np.testing.assert_allclose(sys.wavelengths, [0.486, 0.587, 0.656])


def test_wavelength_resolver_two_branches():
    sys = OpticalSystem(LensData(), wavelengths=[0.486, 0.587, 0.656],
                        reference=1)
    # None -> the reference-index wavelength
    assert sys.wavelength() == pytest.approx(0.587)
    assert sys.wavelength(None) == pytest.approx(0.587)
    # any value -> a literal micron value
    assert sys.wavelength(0.5) == pytest.approx(0.5)


def test_reference_defaults_to_zero_and_property_reads_microns():
    sys = OpticalSystem(LensData(), wavelengths=[0.4, 0.6])
    assert sys.reference == 0
    assert sys.reference_wavelength == pytest.approx(0.4)
    with pytest.raises(AttributeError):
        sys.reference_wavelength = 0.5  # read-only


def test_wavelength_default_when_no_wavelengths():
    sys = OpticalSystem(LensData())
    assert len(sys.wavelengths) == 0
    assert sys.reference_wavelength is None
    assert sys.wavelength() == pytest.approx(0.6328)


def test_weights_default_to_ones_and_validate_length():
    sys = OpticalSystem(LensData(), wavelengths=[0.4, 0.6])
    np.testing.assert_allclose(sys.weights, [1.0, 1.0])
    sys2 = OpticalSystem(LensData(), wavelengths=[0.4, 0.6], weights=[2.0, 3.0])
    np.testing.assert_allclose(sys2.weights, [2.0, 3.0])
    with pytest.raises(ValueError, match='weights length'):
        OpticalSystem(LensData(), wavelengths=[0.4, 0.6], weights=[1.0])


def test_wavelengths_mapping_is_rejected_with_a_clear_message():
    with pytest.raises(TypeError, match='not a mapping'):
        OpticalSystem(LensData(), wavelengths={'d': 0.587})


def test_nanometer_wavelength_guard():
    with pytest.warns(UserWarning, match='nanometers'):
        OpticalSystem(LensData(), wavelengths=[486.0, 587.0])
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('error')  # no warning for micron values
        OpticalSystem(LensData(), wavelengths=[0.486, 0.587])


def test_solve_image_distance_lands_d_line_paraxial_image():
    from prysm.x.raytracing.paraxial import paraxial_image_distance
    sys = _singlet()  # FRAUNHOFER wavelengths, reference index 1 (d-line)
    wvl = sys.reference_wavelength
    sys.solve_image_distance()  # defaults to the reference (d) wavelength
    surfaces = sys.to_surfaces()
    # the image plane sits at the d-line paraxial image of the last powered surface
    expected = (float(surfaces[-2].P[2])
                + float(paraxial_image_distance(surfaces[:-1], wvl=wvl)))
    assert float(surfaces[-1].P[2]) == pytest.approx(expected, abs=1e-9)


# ---------- set_vignetting / solve_vignetting -------------------------------

def _vignetted_singlet(rear_semidiameter=4.0, field=0.0):
    # the rear aperture clips the nominal marginal rays (which arrive at
    # r ~ 4.8 for the default semidiameter of 4); the front surface and the
    # image plane are oversized so the rear surface is the limiting aperture
    n15 = materials.ConstantMaterial(1.5)
    ld = LensData()
    (ld.add(Conic(1 / 30.0, 0.0), thickness=4.0, material=n15,
            semidiameter=6.0)
       .add(Plane(), thickness=50.0, material=materials.air,
            semidiameter=rear_semidiameter)
       .add(Plane(), typ='eval', material=materials.air, semidiameter=20.0))
    sys = OpticalSystem(ld, aperture=10.0, fields=[field],
                        wavelengths=[0.5876], reference=0)
    sys.solve_image_distance()
    return sys


def test_solve_vignetting_factors_are_symmetric_on_axis():
    from prysm.x.raytracing.launch import solve_vignetting

    sys = _vignetted_singlet()
    factors = solve_vignetting(sys, sys.field(0), sys.wavelength())
    vals = [factors[k] for k in ('vux', 'vlx', 'vuy', 'vly')]
    # rotationally symmetric system, on-axis field: all four sides agree
    assert max(vals) - min(vals) < 1e-9
    assert 0.05 < vals[0] < 0.5


def test_set_vignetting_rim_rays_transmit_inside_limiting_aperture():
    from prysm.x.raytracing.spencer_and_murty import valid_mask

    sys = _vignetted_singlet()
    out = sys.set_vignetting()
    assert out is sys
    assert sys.field(0).vignetting is not None

    # with the factors stored, an ordinary rim-sampled launch transmits in
    # full and its marginal rays ride just inside the limiting aperture
    P, S = launch(sys, sys.field(0), sys.wavelength(), Sampling.cross(n=11))
    tr = raytrace(sys, P, S, sys.wavelength())
    assert valid_mask(tr.status).all()
    r_rear = np.hypot(tr.P[2, :, 0], tr.P[2, :, 1])
    assert float(r_rear.max()) <= 4.0
    assert float(r_rear.max()) > 4.0 * 0.98


def test_set_vignetting_overwrites_and_collapses_unvignetted_to_none():
    sys = _vignetted_singlet(rear_semidiameter=8.0)
    # stale hand-typed factors are referenced to the nominal pupil and
    # overwritten, not composed with; an unvignetted field stores None
    sys.field(0).vignetting = {'vux': 0.1, 'vlx': 0.1, 'vuy': 0.1, 'vly': 0.1}
    sys.set_vignetting()
    assert sys.field(0).vignetting is None


def test_solve_vignetting_blocked_chief_raises():
    from prysm.x.raytracing.launch import solve_vignetting

    # at 10 degrees the chief crosses the rear surface at y ~ 0.47, outside
    # the 0.3 semidiameter; factors are chief-referenced, so this must raise
    sys = _vignetted_singlet(rear_semidiameter=0.3, field=10.0)
    with pytest.raises(ValueError, match='chief'):
        solve_vignetting(sys, sys.field(0), sys.wavelength())
