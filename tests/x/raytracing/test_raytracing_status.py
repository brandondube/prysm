"""Tests for the RayTraceResult / status plumbing."""
import numpy as np
import pytest

from tests.x.raytracing.surface_helpers import (
    plane, sphere, conic, off_axis_conic, even_asphere, q2d, zernike, xy,
    chebyshev, jacobi, toroid, biconic,
)

from prysm.x.raytracing.surfaces import Surface, annular_aperture, circular_aperture
from prysm.x.raytracing.spencer_and_murty import (
    raytrace,
    RayTraceResult,
    STATUS_OK,
    STATUS_NEWTON,
    STATUS_CLIP,
    STATUS_MISS,
    STATUS_TIR,
    decode_status,
    valid_mask,
)
from prysm.x.raytracing.raygen import generate_collimated_ray_fan


def _simple_prescription():
    return [
        conic(c=1 / 200., k=-1.0, interaction='refl', P=np.array([0., 0., 0.])),
        plane('eval', P=np.array([0., 0., -50.])),
    ]


def test_raytrace_result_has_named_attributes():
    pres = _simple_prescription()
    P0, S0 = generate_collimated_ray_fan(7, maxr=10.0, z=-100.0)
    result = raytrace(pres, P0, S0, wvl=0.55)
    assert isinstance(result, RayTraceResult)
    np.testing.assert_array_equal(result.P[0], P0)
    np.testing.assert_array_equal(result.S[0], S0)
    assert result.status.shape == (7,)
    assert result.status.dtype == np.complex128
    np.testing.assert_array_equal(result.status_record.surface,
                                  result.status.real.astype(int))
    np.testing.assert_array_equal(result.status_record.code,
                                  result.status.imag.astype(int))


# ---------- valid ray status ----------

def test_collimated_through_parabola_all_valid():
    """Collimated rays well within a parabolic mirror's aperture should all
    finish status==jj + 0j."""
    pres = _simple_prescription()
    P0, S0 = generate_collimated_ray_fan(11, maxr=20.0, z=-200.0)
    result = raytrace(pres, P0, S0, wvl=0.55)
    assert valid_mask(result.status, result.P[-1]).all()
    # status.real records the surface count for valid rays
    np.testing.assert_array_equal(result.status.real, len(pres))


def test_single_ray_1d_input_returns_length1_status():
    pres = _simple_prescription()
    P0 = np.array([0.0, 0.0, -100.0])
    S0 = np.array([0.0, 0.0, 1.0])
    result = raytrace(pres, P0, S0, wvl=0.55)
    assert result.status.shape == (1,)
    assert valid_mask(result.status, result.P[-1])[0]


def test_decode_status_handles_scalar_and_arrays():
    assert decode_status(1 + STATUS_MISS * 1j) == 'MISS at surface 1'
    status = np.array([
        4 + STATUS_OK * 1j,
        2 + STATUS_NEWTON * 1j,
        3 + STATUS_CLIP * 1j,
        1 + STATUS_MISS * 1j,
        5 + STATUS_TIR * 1j,
    ], dtype=np.complex128)
    labels = decode_status(status)
    assert labels.shape == status.shape
    assert labels.tolist() == [
        'OK',
        'NEWTON at surface 2',
        'CLIPPED at surface 3',
        'MISS at surface 1',
        'TIR at surface 5',
    ]


def test_decode_status_all_valid_majority_case():
    status = np.full((2, 3), 7 + STATUS_OK * 1j, dtype=np.complex128)
    labels = decode_status(status)
    assert labels.tolist() == [['OK', 'OK', 'OK'], ['OK', 'OK', 'OK']]


def test_valid_mask_handles_status_and_finite_positions():
    status = np.array([
        2 + STATUS_OK * 1j,
        1 + STATUS_CLIP * 1j,
        2 + STATUS_OK * 1j,
    ], dtype=np.complex128)
    P = np.array([
        [0.0, 0.0, 0.0],
        [np.nan, np.nan, np.nan],
        [1.0, np.nan, 0.0],
    ])
    np.testing.assert_array_equal(valid_mask(status), [True, False, True])
    np.testing.assert_array_equal(valid_mask(status, P), [True, False, False])
    np.testing.assert_array_equal(valid_mask(None, P), [True, False, False])
    assert valid_mask(None, None) is None


# ---------- aperture clipping (STATUS_CLIP) ----------

def test_aperture_clipping_marks_outside_rays():
    """An off-axis ray that misses a circular aperture is marked CLIP."""
    # circular aperture of radius 5 on a flat eval surface
    aperture = lambda x, y: (x * x + y * y) <= 25.0
    pres = [
        plane(interaction='eval', P=np.array([0., 0., 0.]), aperture=aperture),
    ]
    # 7 collimated rays from y=-9 to y=+9: outer rays clipped, center inside
    P0, S0 = generate_collimated_ray_fan(7, maxr=9.0, z=-50.0)
    result = raytrace(pres, P0, S0, wvl=0.55)
    launch_radii = np.sqrt(P0[:, 0] ** 2 + P0[:, 1] ** 2)
    # collimated +z rays outside r=5 should be clipped at the plane
    expected_clipped = launch_radii > 5.0
    actual_clipped = result.status.imag == STATUS_CLIP
    np.testing.assert_array_equal(actual_clipped, expected_clipped)
    assert np.isnan(result.P[1, actual_clipped]).all()
    assert np.isnan(result.S[1, actual_clipped]).all()
    assert np.isnan(result.OPL[1, actual_clipped]).all()
    # the surface index is 1-based; the only surface is index 1
    np.testing.assert_array_equal(
        result.status.real[actual_clipped],
        np.full(actual_clipped.sum(), 1.0),
    )


def test_circular_aperture_helper_marks_inside_circle():
    aperture = circular_aperture(2.0)
    x = np.array([0.0, 2.0, 2.1])
    y = np.array([0.0, 0.0, 0.0])
    np.testing.assert_array_equal(aperture(x, y), [True, True, False])


def test_annular_aperture_blocks_central_obstruction():
    aperture = annular_aperture(1.0, 2.0)
    x = np.array([0.0, 0.5, 1.0, 2.0, 2.1])
    y = np.zeros_like(x)
    np.testing.assert_array_equal(aperture(x, y), [False, False, True, True, False])


def test_clip_persists_through_subsequent_surfaces():
    """A ray clipped at the first surface stays invalid through downstream surfaces."""
    aperture = lambda x, y: x * x + y * y <= 1.0
    pres = [
        plane(interaction='eval', P=np.array([0., 0., 0.]), aperture=aperture),
        plane(interaction='eval', P=np.array([0., 0., 5.])),  # downstream, no aperture
    ]
    P0, S0 = generate_collimated_ray_fan(5, maxr=2.0, z=-10.0)
    result = raytrace(pres, P0, S0, wvl=0.55)
    # the outer rays were clipped at surface 1; status.real should remain 1
    clipped = result.status.imag == STATUS_CLIP
    assert clipped.any()
    # verify status.real for clipped rays is 1, not 2 (didn't get re-marked)
    np.testing.assert_array_equal(
        result.status.real[clipped],
        np.full(clipped.sum(), 1.0),
    )
    # clipped rays are not propagated through later surfaces with plausible
    # coordinates; their histories are NaN from the clipping surface onward.
    assert np.isnan(result.P[1:, clipped]).all()
    assert np.isnan(result.S[1:, clipped]).all()
    assert np.isnan(result.OPL[1:, clipped]).all()


# ---------- analytic miss (STATUS_MISS) ----------

def test_analytic_miss_marked_as_miss():
    """A ray geometrically incapable of intersecting the sphere should be
    STATUS_MISS at that surface index."""
    # small sphere; rays well outside its support
    pres = [
        sphere(c=1 / 5.0, interaction='refl', P=np.array([0., 0., 0.]), material=None),
    ]
    P0 = np.array([[0., 0., -10.],   # axial, hits the sphere
                   [50., 0., -10.]])  # 50mm off-axis, sphere R=5 → can't reach
    S0 = np.array([[0., 0., 1.],
                   [0., 0., 1.]])
    result = raytrace(pres, P0, S0, wvl=0.55)
    assert valid_mask(result.status, result.P[-1])[0]  # axial valid
    assert result.status[1].imag == STATUS_MISS  # off-axis missed
    assert result.status[1].real == 1.0


# ---------- TIR detection (STATUS_TIR) ----------

def test_total_internal_reflection_marked_as_tir():
    """A ray going from glass (n=1.5) to air past the critical angle should TIR."""
    # critical angle for n=1.5 → 1.0 is arcsin(1/1.5) ≈ 41.81°.
    # build a refracting plane at z=0; ray comes from -z in glass at 50° to normal.
    # the launch medium (n=1.5) is carried by a leading eval object surface
    pres = [
        plane(interaction='eval', P=np.array([0., 0., -10.]),
                      material=lambda wvl: 1.5),  # object immersed in glass
        plane(interaction='refr', P=np.array([0., 0., 0.]),
                      material=lambda wvl: 1.0),  # the medium AFTER the surface
    ]
    angle = np.radians(50.0)  # > critical
    # ray in n=1.5 medium hitting the surface at 50° to z (normal)
    P0 = np.array([[0., -10., -10.]])
    S0 = np.array([[0., np.sin(angle), np.cos(angle)]])
    result = raytrace(pres, P0, S0, wvl=0.55)
    assert result.status[0].imag == STATUS_TIR
    assert result.status[0].real == 2.0  # TIR at the refr surface (index 2)


# ---------- end-to-end: aperture + valid + clipped mixed batch ----------

def test_mixed_batch_status_codes_distinct():
    """A batch with valid rays, clipped rays, and missed rays should produce
    three distinct status outcomes."""
    aperture = lambda x, y: (x * x + y * y) <= 4.0
    pres = [
        sphere(c=1 / 100.0, interaction='refl', P=np.array([0., 0., 0.]), material=None,
                       aperture=aperture),
        plane(interaction='eval', P=np.array([0., 0., -10.])),
    ]
    P0 = np.array([
        [0., 0., -50.],     # axial: valid through both surfaces
        [3., 0., -50.],     # outside aperture (r=3 > 2): clipped
        [200., 0., -50.],   # outside the sphere's reach: missed
    ])
    S0 = np.array([[0., 0., 1.]] * 3)
    result = raytrace(pres, P0, S0, wvl=0.55)
    assert valid_mask(result.status, result.P[-1])[0]
    assert result.status[1].imag == STATUS_CLIP
    assert result.status[2].imag == STATUS_MISS
