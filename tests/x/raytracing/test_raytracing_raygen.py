"""Tests for prysm.x.raytracing.raygen and the valid-mask plumbing."""
import numpy as np
import pytest

from tests.x.raytracing.surface_helpers import (
    plane, sphere, conic, off_axis_conic, even_asphere, q2d, zernike, xy,
    chebyshev, jacobi, toroid, biconic,
)

from prysm.x.raytracing.raygen import (
    concat_rayfans,
    split_rayfans,
    generate_collimated_ray_fan,
    generate_collimated_rect_ray_grid,
    generate_finite_ray_fan,
    generate_collimated_hex_ray_grid,
    generate_collimated_radial_spiral_ray_grid,
    clip_to_aperture,
)
from prysm.x.raytracing.surfaces import CallableShape, Surface
from prysm.x.raytracing.sags import gradient_to_unit_normal
from prysm.x.raytracing.spencer_and_murty import (
    intersect as newton_intersect,
    newton_raphson_solve_s,
)


# ---------- raygen ----------

def _sag_and_normal_from_derivatives(sag_derivatives):
    def sag_and_normal(x, y):
        z, Fx, Fy = sag_derivatives(x, y)
        return z, gradient_to_unit_normal(Fx, Fy)
    return sag_and_normal

def test_generate_collimated_ray_fan_uniform():
    P, S = generate_collimated_ray_fan(11, maxr=10.0, z=-50.0, azimuth=90)
    assert P.shape == (11, 3)
    assert S.shape == (11, 3)
    # default azimuth=90 → ray fan in y; x positions are zero
    np.testing.assert_allclose(P[:, 0], 0.0, atol=1e-12)
    # y monotonic from -10 to +10
    np.testing.assert_allclose(P[:, 1], np.linspace(-10, 10, 11))
    # z is the supplied plane
    np.testing.assert_allclose(P[:, 2], -50.0)
    # default direction is +z, no tilt
    np.testing.assert_allclose(S, np.tile([0, 0, 1], (11, 1)), atol=1e-15)


def test_generate_collimated_ray_fan_with_yangle():
    """yangle must produce unit-norm direction cosines tilted off pure +z.

    (yangle here parameterizes a rotation about the Y axis; the direction
    cosines pick up an x component, not a y component — see make_rotation_matrix.)
    """
    P, S = generate_collimated_ray_fan(5, maxr=2.0, z=0, yangle=10.0)
    norms = np.linalg.norm(S, axis=-1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-12)
    # not pure +z anymore
    assert not np.allclose(S, [0, 0, 1])


def test_generate_collimated_rect_ray_grid_shape_and_unitnorm():
    P, S = generate_collimated_rect_ray_grid(5, maxx=1.0, z=0)
    assert P.shape == (25, 3)
    assert S.shape == (25, 3)
    norms = np.linalg.norm(S, axis=-1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-12)


def test_generate_finite_ray_fan_shape_and_unitnorm():
    P, S = generate_finite_ray_fan(7, na=0.1, P=[0, 0, -50.0])
    assert P.shape == (7, 3)
    assert S.shape == (7, 3)
    np.testing.assert_allclose(np.linalg.norm(S, axis=-1), 1.0, atol=1e-12)
    # all rays start at the same P
    np.testing.assert_allclose(P, np.tile([0, 0, -50.0], (7, 1)))


def test_concat_rayfans_combines_two_fans():
    P1, S1 = generate_collimated_ray_fan(3, maxr=1.0, z=0)
    P2, S2 = generate_collimated_ray_fan(5, maxr=1.0, z=0)
    P, S = concat_rayfans((P1, S1), (P2, S2))
    assert P.shape == (8, 3)
    assert S.shape == (8, 3)
    np.testing.assert_array_equal(P[:3], P1)
    np.testing.assert_array_equal(P[3:], P2)


def test_split_rayfans_round_trips_through_concat():
    P1, S1 = generate_collimated_ray_fan(3, maxr=1.0, z=0)
    P2, S2 = generate_collimated_ray_fan(5, maxr=1.0, z=0)
    P, S = concat_rayfans((P1, S1), (P2, S2))
    chunks_P, chunks_S = split_rayfans(P, [3, 5], S=S)
    assert len(chunks_P) == 2 and len(chunks_S) == 2
    np.testing.assert_array_equal(chunks_P[0], P1)
    np.testing.assert_array_equal(chunks_P[1], P2)
    np.testing.assert_array_equal(chunks_S[0], S1)
    np.testing.assert_array_equal(chunks_S[1], S2)


def test_split_rayfans_without_S():
    P1, _ = generate_collimated_ray_fan(2, maxr=1.0, z=0)
    P2, _ = generate_collimated_ray_fan(4, maxr=1.0, z=0)
    P, _ = concat_rayfans((P1, np.zeros_like(P1)), (P2, np.zeros_like(P2)))
    chunks = split_rayfans(P, [2, 4])
    assert len(chunks) == 2
    np.testing.assert_array_equal(chunks[0], P1)
    np.testing.assert_array_equal(chunks[1], P2)


def test_split_rayfans_length_mismatch_raises():
    """Regression: previously this branch was unreachable (used .size[0]
    on a scalar) and even if reached returned the exception object instead
    of raising it."""
    P = np.zeros((10, 3))
    with pytest.raises(ValueError, match='sum.*chunksizes'):
        split_rayfans(P, [3, 4])  # sums to 7, P has 10


# ---------- valid-mask plumbing ----------

def _ray_batch(seed=0, span=4.0, n=11):
    rng = np.random.default_rng(seed)
    xs = np.linspace(-span, span, n)
    X, Y = np.meshgrid(xs, xs, indexing='xy')
    P = np.stack([X.ravel(), Y.ravel(), np.full(X.size, -50.0)], axis=-1)
    Sx = rng.normal(scale=0.02, size=X.size)
    Sy = rng.normal(scale=0.02, size=X.size)
    Sz = np.sqrt(1 - Sx * Sx - Sy * Sy)
    S = np.stack([Sx, Sy, Sz], axis=-1)
    return P.astype(np.float64), S.astype(np.float64)


def test_newton_solver_valid_mask_all_true_for_simple_sphere():
    def sag(x, y):
        return (
            (1 / 100. * (x * x + y * y))
            / (1 + np.sqrt(1 - (1 / 100.) ** 2 * (x * x + y * y)))
        )

    def sag_derivatives(x, y):
        return (
            sag(x, y),
            (1 / 100.) * x / np.sqrt(1 - (1 / 100.) ** 2 * (x * x + y * y)),
            (1 / 100.) * y / np.sqrt(1 - (1 / 100.) ** 2 * (x * x + y * y)),
        )

    shape = CallableShape(
        sag,
        _sag_and_normal_from_derivatives(sag_derivatives),
    )
    surf = Surface(shape=shape, interaction='refl', P=np.array([0., 0., 0.]))
    P, S = _ray_batch(span=3.0)
    Q, n, valid = surf.intersect(P, S)
    assert valid.shape == (P.shape[0],)
    assert valid.dtype == bool
    assert valid.all()
    assert np.all(np.isfinite(Q))


def test_newton_solver_valid_mask_flags_nonconvergence():
    """A ray that won't reach the surface within maxiter should be flagged invalid."""
    # use a very low maxiter and a steep curvature so the iteration doesn't
    # have time to converge for off-axis rays
    surf = conic(c=1 / 5.0, k=-2.0, interaction='refl', P=np.array([0., 0., 0.]))
    # build via a generic Surface (forces Newton, not the analytic Conic path)
    bare = Surface(shape=CallableShape(surf.sag, surf.sag_and_normal,
                                       params=dict(surf.params)),
                   interaction='refl', P=np.array([0., 0., 0.]))
    # a ray nearly parallel to the surface in the steep region won't converge in 1 iter
    P = np.array([[3.5, 0., -50.], [0., 0., -50.]])
    S = np.array([[0.05, 0., np.sqrt(1 - 0.0025)], [0., 0., 1.]])
    Q, n, valid = bare.intersect(P, S, maxiter=1)
    # the on-axis ray should converge in 1 iteration; the off-axis ray should not
    assert valid[1]  # on-axis converged
    assert not valid[0]  # off-axis did not
    # the failed ray gets NaN'd in Q and n
    assert np.all(np.isnan(Q[0]))
    assert np.all(np.isnan(n[0]))


def test_analytic_intersect_valid_mask_flags_no_intersection():
    """An off-axis ray that lies outside the sphere's extent should be
    reported invalid (negative discriminant) by the analytic conic path."""
    # sphere of radius 50 centered at (0,0,50), vertex at origin, axis +z
    surf = sphere(c=1 / 50.0, interaction='refl', P=np.array([0., 0., 0.]), material=None)
    # axial ray (hits the vertex) and an off-axis ray with x=60 > R=50 going +z
    # (the implicit sphere equation has no real root for that ray)
    P = np.array([[0., 0., -10.],
                  [60., 0., -10.]])
    S = np.array([[0., 0., 1.],
                  [0., 0., 1.]])
    Q, n, valid = surf.intersect(P, S)
    assert valid[0]
    assert not valid[1]


# ---------- cheby distribution + hex / spiral grids ----------

def test_cheby_distribution_includes_endpoints_and_clusters():
    """Chebyshev-Gauss-Lobatto nodes must include both endpoints and cluster
    near them (consecutive interior gaps are larger than edge gaps)."""
    P, _ = generate_collimated_ray_fan(7, maxr=10.0, distribution='cheby')
    ys = P[:, 1]
    # endpoints exact
    np.testing.assert_allclose(ys[0], -10.0, atol=1e-12)
    np.testing.assert_allclose(ys[-1], 10.0, atol=1e-12)
    # gaps near the edges must be smaller than the central gap
    gaps = np.diff(ys)
    assert gaps[0] < gaps[len(gaps) // 2]
    assert gaps[-1] < gaps[len(gaps) // 2]


def test_cheby_rect_grid_uses_cheby_on_both_axes():
    P, _ = generate_collimated_rect_ray_grid(5, maxx=2.0, distribution='cheby')
    # there should be 25 rays
    assert P.shape == (25, 3)
    # extreme x and y values must be ±maxx
    assert np.isclose(P[:, 0].min(), -2.0, atol=1e-12)
    assert np.isclose(P[:, 0].max(), 2.0, atol=1e-12)
    assert np.isclose(P[:, 1].min(), -2.0, atol=1e-12)
    assert np.isclose(P[:, 1].max(), 2.0, atol=1e-12)


def test_cheby_unknown_distribution_raises():
    with pytest.raises(ValueError, match='unknown distribution'):
        generate_collimated_ray_fan(5, maxr=1.0, distribution='nonsense')


@pytest.mark.parametrize('nrings', [0, 1, 2, 3, 5])
def test_hex_grid_count_matches_formula(nrings):
    """Hexapolar count is 1 + 3*N*(N+1)."""
    P, S = generate_collimated_hex_ray_grid(nrings, spacing=1.0)
    expected = 1 + 3 * nrings * (nrings + 1)
    assert P.shape == (expected, 3)
    assert S.shape == (expected, 3)
    # all S unit-norm
    np.testing.assert_allclose(np.linalg.norm(S, axis=-1), 1.0, atol=1e-12)


def test_hex_grid_radii_match_ring_index():
    """Each ring k contributes 6k rays at radius k*spacing."""
    spacing = 2.5
    P, _ = generate_collimated_hex_ray_grid(3, spacing=spacing)
    radii = np.sqrt(P[:, 0] ** 2 + P[:, 1] ** 2)
    # rounded so we can group by ring
    rounded = np.round(radii / spacing).astype(int)
    counts = np.bincount(rounded)
    np.testing.assert_array_equal(counts, [1, 6, 12, 18])  # 1+6+12+18 = 37


def test_hex_grid_negative_nrings_raises():
    with pytest.raises(ValueError, match='nrings'):
        generate_collimated_hex_ray_grid(-1, spacing=1.0)


def test_radial_spiral_grid_default_density_matches_hex():
    """Default samples_per_ring=6k yields the same total count as the hex grid."""
    P_spiral, _ = generate_collimated_radial_spiral_ray_grid(3, maxr=10.0)
    P_hex, _ = generate_collimated_hex_ray_grid(3, spacing=10.0 / 3)
    assert P_spiral.shape == P_hex.shape  # same total ray count


def test_radial_spiral_cheby_radii_cluster_at_rim():
    """With cheby radial spacing, ring radii cluster near maxr."""
    P, _ = generate_collimated_radial_spiral_ray_grid(4, maxr=10.0,
                                                      radial_distribution='cheby')
    radii = np.sqrt(P[:, 0] ** 2 + P[:, 1] ** 2)
    unique_radii = np.unique(np.round(radii, 4))
    # exclude 0 (center) — non-zero ring radii sorted ascending
    rings = np.sort([r for r in unique_radii if r > 0])
    # the gap from outer ring to maxr should be smaller than the gap between
    # the innermost two non-center rings
    assert (10.0 - rings[-1]) < (rings[1] - rings[0])


def test_radial_spiral_no_center_option():
    """include_center=False omits the (0, 0) sample."""
    P, _ = generate_collimated_radial_spiral_ray_grid(2, maxr=5.0,
                                                      include_center=False)
    radii = np.sqrt(P[:, 0] ** 2 + P[:, 1] ** 2)
    assert radii.min() > 0  # no center


def test_radial_spiral_custom_samples_per_ring():
    """User-supplied callable controls azimuthal density."""
    samples = lambda k: 4  # constant 4 per ring
    P, _ = generate_collimated_radial_spiral_ray_grid(3, maxr=5.0,
                                                      samples_per_ring=samples)
    # 1 (center) + 3 * 4 = 13
    assert P.shape == (13, 3)


def test_radial_spiral_negative_nrings_raises():
    with pytest.raises(ValueError, match='nrings'):
        generate_collimated_radial_spiral_ray_grid(0, maxr=1.0)


# ---------- clip_to_aperture ----------

def test_clip_to_aperture_drops_outside_rays():
    """Circular aperture filter on a rect grid keeps only inside-disk rays."""
    rayfan = generate_collimated_rect_ray_grid(11, maxx=1.0)
    aperture = lambda x, y: x * x + y * y <= 0.25  # r <= 0.5
    P_kept, S_kept = clip_to_aperture(rayfan, aperture)
    # all kept rays must satisfy the aperture
    radii = np.sqrt(P_kept[:, 0] ** 2 + P_kept[:, 1] ** 2)
    assert (radii <= 0.5 + 1e-12).all()
    # at least one ray was dropped (corners of the rect grid are outside r=0.5)
    P_orig, _ = rayfan
    assert P_kept.shape[0] < P_orig.shape[0]
    # S stays unit-norm
    np.testing.assert_allclose(np.linalg.norm(S_kept, axis=-1), 1.0, atol=1e-12)


def test_clip_to_aperture_no_drop_when_aperture_covers_all():
    """If every ray is inside, the output equals the input."""
    rayfan = generate_collimated_ray_fan(5, maxr=0.5)
    P_kept, S_kept = clip_to_aperture(rayfan, lambda x, y: x * x + y * y <= 100.0)
    P_orig, S_orig = rayfan
    np.testing.assert_array_equal(P_kept, P_orig)
    np.testing.assert_array_equal(S_kept, S_orig)


def test_clip_to_aperture_empty_result_when_all_outside():
    """Aperture excluding everything yields an empty (0, 3) ray fan."""
    rayfan = generate_collimated_ray_fan(7, maxr=10.0)
    P_kept, S_kept = clip_to_aperture(rayfan, lambda x, y: np.zeros_like(x, dtype=bool))
    assert P_kept.shape == (0, 3)
    assert S_kept.shape == (0, 3)
