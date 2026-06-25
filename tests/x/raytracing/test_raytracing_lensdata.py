"""Tests for the LensData spine: axial layout parity, thickness slide,
mirror fold, the optimizer surface (pack/update/bounds), variable selection,
metadata, and shape round-trip."""

import numpy as np
import pytest

from prysm.x.raytracing import (
    OpticalSystem,
    FRAUNHOFER_LINES_UM,
    CoordBreak,
    Field,
    LensData,
    raytrace,
)
from prysm.x import materials
from prysm.x.raytracing.raygen import generate_collimated_ray_fan
from prysm.x.raytracing.surfaces import (
    Conic,
    EvenAsphere,
    Plane,
    Surface,
    Zernike,
    circular_aperture,
)


n_bk7 = materials.ConstantMaterial(1.5168, name='N-BK7')


def make_singlet_lensdata(image_gap=95.0):
    ld = LensData()
    (ld.add(Conic(1 / 102.0, 0.0), thickness=6.0, material=n_bk7,
            semidiameter=10.0)
       .add(Conic(-1 / 102.0, 0.0), thickness=image_gap,
            material=materials.air, semidiameter=10.0))
    return ld


def make_singlet_hand(image_gap=95.0):
    # OBJECT, ...powered, IMAGE (ADR-0006): OBJECT sits coincident with the
    # first powered surface at z=0 for an infinite conjugate.
    return [
        Surface(shape=Plane(), interaction='object', P=[0, 0, 0.0],
                material=materials.air),
        Surface(shape=Conic(1 / 102.0, 0.0), interaction='refr',
                P=[0, 0, 0.0], material=n_bk7, bounding={'outer_radius': 10.0},
                aperture=circular_aperture(10.0)),
        Surface(shape=Conic(-1 / 102.0, 0.0), interaction='refr',
                P=[0, 0, 6.0], material=materials.air,
                bounding={'outer_radius': 10.0},
                aperture=circular_aperture(10.0)),
        Surface(shape=Plane(), interaction='image', P=[0, 0, 6.0 + image_gap],
                material=materials.air),
    ]


def assert_surfaces_equivalent(a, b):
    assert len(a) == len(b)
    for sa, sb in zip(a, b):
        assert type(sa.shape) is type(sb.shape)
        assert sa.typ == sb.typ
        np.testing.assert_array_equal(np.asarray(sa.P), np.asarray(sb.P))
        assert (sa.R is None) == (sb.R is None)
        pa = sa.params or {}
        pb = sb.params or {}
        assert set(pa) == set(pb)
        for key in pa:
            np.testing.assert_array_equal(np.asarray(pa[key]),
                                          np.asarray(pb[key]))


# ---------------------------------------------------------------------------
# axial layout parity
# ---------------------------------------------------------------------------

def test_axial_layout_matches_hand_built_surfaces():
    ld = make_singlet_lensdata()
    assert_surfaces_equivalent(ld.to_surfaces(), make_singlet_hand())


def test_axial_trace_bit_identical_to_hand_built():
    ld = make_singlet_lensdata()
    hand = make_singlet_hand()
    P0, S0 = generate_collimated_ray_fan(21, maxr=9.0, z=-50.0)
    wvl = FRAUNHOFER_LINES_UM['d']
    ra = raytrace(ld, P0, S0, wvl=wvl)
    rb = raytrace(hand, P0, S0, wvl=wvl)
    np.testing.assert_array_equal(ra.P, rb.P)
    np.testing.assert_array_equal(ra.S, rb.S)
    np.testing.assert_array_equal(ra.status, rb.status)


def test_lensdata_duck_types_as_surface_sequence():
    ld = make_singlet_lensdata()
    assert len(ld) == 4  # OBJECT, two conics, IMAGE
    assert list(ld) == ld.surfaces
    assert ld[0] is ld.surfaces[0]


def test_direct_surface_row_thickness_edit_invalidates_cache():
    ld = make_singlet_lensdata()
    before = ld.surfaces
    ld.rows[1].thickness = 12.0           # first powered gap (rows[0] is OBJECT)
    after = ld.surfaces
    assert after is not before
    z = [float(np.asarray(s.P)[2]) for s in after]
    assert z == pytest.approx([0.0, 0.0, 12.0, 107.0])


def test_direct_surface_row_array_and_metadata_edits_invalidate_cache():
    ld = LensData().add(Conic(1 / 100.0, 0.0), typ='refr',
                        material=materials.ConstantMaterial(1.5),
                        bounding={'outer_radius': 4.0})
    before = ld.surfaces

    ld.rows[1].params[0] = 1 / 50.0       # rows[0] is OBJECT; the conic is rows[1]
    assert ld.surfaces is not before
    assert ld.surfaces[1].shape.params['c'] == pytest.approx(1 / 50.0)

    ld.rows[1].material = materials.ConstantMaterial(1.6)
    assert ld.surfaces[1].material.n(0.55) == pytest.approx(1.6)

    ld.rows[1].aperture = circular_aperture(2.0)
    blocked = ld.surfaces[1].aperture(np.array([3.0]), np.array([0.0]))
    assert bool(blocked[0]) is False

    ld.rows[1].bounding['outer_radius'] = 6.0
    assert ld.surfaces[1].bounding['outer_radius'] == pytest.approx(6.0)


def test_direct_coordbreak_array_edit_invalidates_cache():
    ld = LensData()
    ld.add_coordbreak()                   # inserted before IMAGE -> rows[1]
    before = ld.surfaces
    ld.rows[1].decenter[0] = 3.0
    after = ld.surfaces
    assert after is not before
    # OBJECT stays at the origin; the break decenters the downstream IMAGE plane
    np.testing.assert_allclose(after[1].P, [3.0, 0.0, 0.0])


def test_material_object_identity_is_preserved():
    # a MaterialProtocol object is carried to the compiled surface verbatim --
    # no wrapping, so identity holds.
    ld = make_singlet_lensdata()
    assert ld.surfaces[1].material is n_bk7   # surfaces[0] is the OBJECT plane


def test_add_rejects_bare_material_forms():
    # ADR-0002: materials are MaterialProtocol objects -- a bare number, string,
    # or lambda raises at add() rather than detonating mid-trace.
    for bad in (1.5, 'N-BK7', lambda wvl: 1.5):
        with pytest.raises(TypeError, match='MaterialProtocol'):
            LensData().add(Conic(0.0, 0.0), thickness=1.0, material=bad)


# ---------------------------------------------------------------------------
# thickness slide
# ---------------------------------------------------------------------------

def test_thickness_dof_slides_downstream_surfaces():
    ld = OpticalSystem(make_singlet_lensdata())
    z_before = [float(np.asarray(s.P)[2]) for s in ld.surfaces]

    ld.opt.vary('thickness', surfaces=1)      # first powered gap
    x = ld.opt.pack()
    assert x == pytest.approx([6.0])
    ld.opt.update(x + 5.0)

    z_after = [float(np.asarray(s.P)[2]) for s in ld.surfaces]
    # the gap grew by 5; this surface and ALL downstream slide by 5
    assert z_after == pytest.approx([z_before[0],
                                     z_before[1],
                                     z_before[2] + 5.0,
                                     z_before[3] + 5.0])


def test_thickness_slide_matches_manual_relayout():
    ld = OpticalSystem(make_singlet_lensdata(image_gap=95.0))
    ld.opt.vary('thickness', surfaces=1)      # first powered gap
    ld.opt.update(np.array([9.0]))  # first gap 6 -> 9
    manual = make_singlet_hand(image_gap=95.0)
    # manual relayout: second conic at 9, image at 9 + 95 = 104
    manual[2].P[2] = 9.0
    manual[3].P[2] = 104.0
    assert_surfaces_equivalent(ld.to_surfaces(), manual)


# ---------------------------------------------------------------------------
# mirror fold (untilted, on-axis)
# ---------------------------------------------------------------------------

def test_mirror_fold_steps_downstream_to_negative_z():
    ld = LensData().add(Conic(1 / 200.0, -1.0), typ='refl', thickness=50.0)
    surfs = ld.to_surfaces()
    assert [float(np.asarray(s.P)[2]) for s in surfs] == pytest.approx(
        [0.0, 0.0, -50.0])


def test_mirror_fold_trace_matches_hand_built():
    ld = LensData().add(Conic(1 / 200.0, -1.0), typ='refl', thickness=50.0)
    hand = [
        Surface(shape=Plane(), interaction='object', P=[0, 0, 0.0],
                material=materials.air),
        Surface(shape=Conic(1 / 200.0, -1.0), interaction='refl',
                P=[0, 0, 0.0]),
        Surface(shape=Plane(), interaction='image', P=[0, 0, -50.0],
                material=materials.air),
    ]
    P0, S0 = generate_collimated_ray_fan(11, maxr=20.0, z=-200.0)
    ra = raytrace(ld, P0, S0, wvl=0.55)
    rb = raytrace(hand, P0, S0, wvl=0.55)
    np.testing.assert_array_equal(ra.P, rb.P)
    np.testing.assert_array_equal(ra.status, rb.status)


def test_two_mirror_fold_returns_to_increasing_z():
    # two reflections -> net forward direction again
    ld = (LensData()
          .add(Plane(), typ='refl', thickness=10.0)
          .add(Plane(), typ='refl', thickness=4.0))
    z = [float(np.asarray(s.P)[2]) for s in ld.to_surfaces()]
    assert z == pytest.approx([0.0, 0.0, -10.0, -6.0])


# ---------------------------------------------------------------------------
# variable selection + optimizer surface
# ---------------------------------------------------------------------------

def test_vary_curvature_packs_curvatures_in_order():
    ld = OpticalSystem(make_singlet_lensdata())
    ld.opt.vary('curvature', surfaces=[1, 2])     # the two conics (rows[0] is OBJECT)
    np.testing.assert_allclose(ld.opt.pack(), [1 / 102.0, -1 / 102.0])


def test_vary_all_then_freeze_all():
    ld = OpticalSystem(make_singlet_lensdata())
    ld.opt.vary_all()
    # OBJECT thickness (1) + two conics (c, k, thickness = 3 each) +
    # IMAGE thickness (1) = 1 + 2*3 + 1 = 8
    assert len(ld.opt.pack()) == 8
    ld.opt.freeze_all()
    assert len(ld.opt.pack()) == 0


def test_freeze_is_inverse_of_vary():
    ld = OpticalSystem(make_singlet_lensdata())
    ld.opt.vary('conic', surfaces='all')
    assert len(ld.opt.pack()) == 2
    ld.opt.freeze('conic', surfaces=1)        # first conic (rows[0] is OBJECT)
    assert len(ld.opt.pack()) == 1


def test_vary_missing_category_is_silent_no_op():
    ld = OpticalSystem(make_singlet_lensdata())
    # the eval plane (surface 2) has no conic; selecting all should still work
    ld.opt.vary('conic', surfaces='all')
    assert len(ld.opt.pack()) == 2  # only the two conics, plane skipped


def test_update_round_trips_pack():
    ld = OpticalSystem(make_singlet_lensdata())
    ld.opt.vary('curvature', surfaces=[1, 2]).vary('conic', surfaces=[1])
    x = ld.opt.pack()
    x2 = x + np.array([1e-3, -2e-3, 0.05])
    ld.opt.update(x2)
    np.testing.assert_allclose(ld.opt.pack(), x2)


def test_update_rejects_wrong_length():
    ld = OpticalSystem(make_singlet_lensdata())
    ld.opt.vary('curvature', surfaces=0)
    with pytest.raises(ValueError):
        ld.opt.update(np.array([1.0, 2.0]))


def test_bounds_default_to_infinite():
    ld = OpticalSystem(make_singlet_lensdata())
    ld.opt.vary('curvature', surfaces=[0, 1])
    lo, hi = ld.opt.bounds()
    assert np.all(lo == -np.inf)
    assert np.all(hi == np.inf)


# ---------------------------------------------------------------------------
# metadata absorption
# ---------------------------------------------------------------------------

def test_metadata_absorbed_and_resolved():
    sys = OpticalSystem(make_singlet_lensdata(), aperture=20.0, fields=[0],
                        wavelengths=list(FRAUNHOFER_LINES_UM.values()),
                        reference=1)
    assert sys.epd == 20.0
    assert sys.reference_wavelength == pytest.approx(FRAUNHOFER_LINES_UM['d'])
    assert sys.wavelength() == pytest.approx(FRAUNHOFER_LINES_UM['d'])
    assert sys.wavelength(0.5) == pytest.approx(0.5)
    assert isinstance(sys.field(0), Field)
    assert sys.field(0).hy == pytest.approx(0.0)


def test_extras_and_provenance_fields_round_trip():
    ld = OpticalSystem(LensData(), source_path='/tmp/x.seq', source_format='codev',
                  stop_index=2, extras={'note': 'hi'})
    assert ld.source_path == '/tmp/x.seq'
    assert ld.source_format == 'codev'
    assert ld.stop_index == 2
    assert ld.extras['note'] == 'hi'


# ---------------------------------------------------------------------------
# shape round-trip (rebuild parity for non-trivial shapes)
# ---------------------------------------------------------------------------

def test_even_asphere_round_trips_through_lensdata():
    coefs = (1e-4, -2e-6, 3e-9)
    shape = EvenAsphere(1 / 50.0, -0.5, coefs)
    ld = LensData().add(shape, thickness=2.0, material=n_bk7, semidiameter=8.0)
    rebuilt = ld.surfaces[1].shape    # surfaces[0] is the OBJECT plane
    x = np.linspace(-7, 7, 13)
    y = np.linspace(-7, 7, 13)
    np.testing.assert_allclose(rebuilt.sag(x, y), shape.sag(x, y))
    assert rebuilt.params['c'] == pytest.approx(1 / 50.0)
    np.testing.assert_allclose(np.asarray(rebuilt.params['coefs']), coefs)


def test_coating_round_trips_onto_compiled_surface_and_listing():
    from prysm.x.coatings.stack import Stack
    from prysm.x.raytracing import surface_table
    ar = Stack([1.38], [0.1], substrate_index=1.5, ambient_index=1.0)
    ld = LensData().add(Conic(1 / 100.0, 0.0), thickness=5.0, material=n_bk7,
                        semidiameter=8.0, coating=ar)
    assert ld.rows[1].coating is ar
    assert ld.surfaces[1].coating is ar
    assert ld.surfaces[2].coating is None            # bare image plane
    assert ld.copy().rows[1].coating is ar
    records = surface_table(ld).records
    assert records[1]['coating'] is True
    assert records[2]['coating'] is False


def test_zernike_round_trips_with_static_metadata():
    nms = [(2, 0), (4, 0)]
    coefs = (0.3, -0.1)
    shape = Zernike(0.0, 0.0, 10.0, nms, coefs, norm=True)
    ld = LensData().add(shape, thickness=1.0, typ='eval', semidiameter=10.0)
    rebuilt = ld.surfaces[1].shape    # surfaces[0] is the OBJECT plane
    assert rebuilt.params['nms'] == tuple(nms)
    assert rebuilt.params['norm'] is True
    x = np.linspace(-9, 9, 11)
    y = np.linspace(-9, 9, 11)
    np.testing.assert_allclose(rebuilt.sag(x, y), shape.sag(x, y))


def test_varying_a_coef_changes_only_that_coef():
    coefs = (1e-4, -2e-6, 3e-9)
    ld = OpticalSystem(LensData().add(EvenAsphere(1 / 50.0, -0.5, coefs),
                                      thickness=2.0, material=n_bk7,
                                      semidiameter=8.0))
    ld.opt.vary('coefs', surfaces=1)          # rows[0] is OBJECT
    x = ld.opt.pack()
    assert len(x) == 3
    x[1] = -5e-6
    ld.opt.update(x)
    np.testing.assert_allclose(np.asarray(ld.surfaces[1].params['coefs']),
                               [1e-4, -5e-6, 3e-9])


# ---------------------------------------------------------------------------
# copy independence
# ---------------------------------------------------------------------------

def test_copy_is_independent():
    ld = OpticalSystem(make_singlet_lensdata())
    ld.opt.vary('curvature', surfaces=1)      # first conic (rows[0] is OBJECT)
    clone = ld.copy()
    clone.opt.update(np.array([1 / 80.0]))
    # original unchanged
    assert ld.surfaces[1].params['c'] == pytest.approx(1 / 102.0)
    assert clone.surfaces[1].params['c'] == pytest.approx(1 / 80.0)
    # free selection preserved
    assert len(clone.opt.pack()) == 1


# ---------------------------------------------------------------------------
# coordinate-break declaration + DOFs (layout itself is exercised in the
# coordinate-break test module)
# ---------------------------------------------------------------------------

def test_coordbreak_declares_dofs_and_lays_out():
    ld = OpticalSystem(LensData()
          .add(Conic(1 / 100.0, 0.0), thickness=5.0, material=n_bk7,
               semidiameter=5.0)
          .add_coordbreak(tilt=(0.0, 0.0, 5.0), thickness=2.0))
    assert isinstance(ld.rows[2], CoordBreak)   # rows[0]=OBJECT, rows[1]=conic
    ld.opt.vary('tilt', surfaces=2)
    np.testing.assert_allclose(ld.opt.pack(), [0.0, 0.0, 5.0])
    # the break now lays out: the downstream surface is tilted
    surfs = ld.to_surfaces()
    assert len(surfs) == 3                       # OBJECT, conic, IMAGE
    assert surfs[2].R is not None
