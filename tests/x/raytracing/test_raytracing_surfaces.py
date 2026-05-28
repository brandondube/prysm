"""Tests for prysm.x.raytracing.surfaces."""
import numpy as np
import pytest

from tests.x.raytracing.surface_helpers import (
    plane, sphere, conic, off_axis_conic, even_asphere, q2d, zernike, xy,
    chebyshev, jacobi, toroid, biconic,
)

from prysm.x.raytracing.surfaces import (
    CallableShape,
    Shape,
    Surface,
    Q2D,
    conic_sag,
    conic_sag_and_normal,
    conic_sag_der_xy,
    even_asphere_sag,
    even_asphere_sag_der_xy,
    ray_plane_intersect,
    ray_sphere_intersect,
    ray_conic_intersect,
)
from prysm.x.raytracing.sags import _q2d_sigma_inv_der, gradient_to_unit_normal
from prysm.x.raytracing.spencer_and_murty import (
    intersect as newton_intersect,
    raytrace,
    transform_to_local_coords,
)
from prysm.x.raytracing.raygen import generate_collimated_ray_fan


def _polar_grid(rmax=5.0, n=21):
    r = np.linspace(0.5, rmax, n)
    t = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    R, T = np.meshgrid(r, t, indexing='ij')
    return R, T


def _sag_and_normal_from_derivatives(sag_derivatives):
    def sag_and_normal(x, y):
        z, Fx, Fy = sag_derivatives(x, y)
        return z, gradient_to_unit_normal(Fx, Fy)
    return sag_and_normal


def _sag_derivs(shape, x, y):
    """Recover (z, dz/dx, dz/dy) from a shape's sag_and_normal unit normal."""
    z, n_hat = shape.sag_and_normal(x, y)
    nz = n_hat[..., 2]
    return z, -n_hat[..., 0] / nz, -n_hat[..., 1] / nz


@pytest.mark.parametrize('dx,dy', [(3.0, 4.0), (-2.0, 5.0)])
def test_q2d_sigma_inv_der_matches_numerical(dx, dy):
    c = 1 / 80.0
    kappa = -1.0
    r, t = _polar_grid()
    x = r * np.cos(t)
    y = r * np.sin(t)
    h = 1e-5

    def inv_sigma(rr, tt):
        xx = rr * np.cos(tt)
        yy = rr * np.sin(tt)
        _, n_hat = conic_sag_and_normal(c, kappa, xx + dx, yy + dy)
        return 1.0 / n_hat[..., 2]

    dr_an, dt_an = _q2d_sigma_inv_der(c, kappa, x, y, r, t, dx, dy)
    dr_num = (inv_sigma(r + h, t) - inv_sigma(r - h, t)) / (2 * h)
    dt_num = (inv_sigma(r, t + h) - inv_sigma(r, t - h)) / (2 * h)

    np.testing.assert_allclose(dr_an, dr_num, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(dt_an, dt_num, rtol=1e-5, atol=1e-7)


def _xy_grid(span=4.0, n=15):
    xs = np.linspace(-span, span, n)
    X, Y = np.meshgrid(xs, xs, indexing='xy')
    return X, Y


@pytest.mark.parametrize('kappa', [0.0, -1.0, -0.5, 0.7])
def test_conic_sag_der_xy_matches_numerical(kappa):
    c = 1 / 80.0
    x, y = _xy_grid()
    h = 1e-5

    def sag(xx, yy):
        return conic_sag(c, kappa, xx * xx + yy * yy)

    dx_an, dy_an = conic_sag_der_xy(c, kappa, x, y)
    dx_num = (sag(x + h, y) - sag(x - h, y)) / (2 * h)
    dy_num = (sag(x, y + h) - sag(x, y - h)) / (2 * h)
    np.testing.assert_allclose(dx_an, dx_num, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(dy_an, dy_num, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize('dx,dy', [(0.0, 0.0), (3.0, 0.0), (0.0, 4.0), (3.0, 4.0), (-2.0, 5.0)])
@pytest.mark.parametrize('kappa', [0.0, -1.0, -0.5, 0.7])
def test_shifted_conic_sag_der_xy_matches_numerical(dx, dy, kappa):
    c = 1 / 80.0
    x, y = _xy_grid()
    h = 1e-5

    def sag(xx, yy):
        z, _ = conic_sag_and_normal(c, kappa, xx + dx, yy + dy)
        return z

    _, n_hat = conic_sag_and_normal(c, kappa, x + dx, y + dy)
    ddx_an = -n_hat[..., 0] / n_hat[..., 2]
    ddy_an = -n_hat[..., 1] / n_hat[..., 2]
    ddx_num = (sag(x + h, y) - sag(x - h, y)) / (2 * h)
    ddy_num = (sag(x, y + h) - sag(x, y - h)) / (2 * h)
    np.testing.assert_allclose(ddx_an, ddx_num, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(ddy_an, ddy_num, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize('dx,dy', [(0.0, 0.0), (3.0, 4.0), (-2.0, 5.0)])
def test_xy_derivatives_finite_at_origin(dx, dy):
    """No singularity at x=y=0 for conic / off-axis-conic Cartesian derivatives."""
    c = 1 / 80.0
    kappa = -1.0
    zero = np.array(0.0)
    _, n_hat = conic_sag_and_normal(c, kappa, zero + dx, zero + dy)
    ddx = -n_hat[..., 0] / n_hat[..., 2]
    ddy = -n_hat[..., 1] / n_hat[..., 2]
    assert np.isfinite(ddx) and np.isfinite(ddy)
    if dx == 0.0 and dy == 0.0:
        assert ddx == 0.0 and ddy == 0.0


def test_surface_conic_derivatives_finite_at_origin():
    """Conic-like sag_derivatives paths must not produce NaN/Inf at origin."""
    P = np.array([0.0, 0.0, 0.0])
    s_conic = conic(c=1 / 80.0, k=-1.0, typ='refl', P=P)
    s_oac = off_axis_conic(c=1 / 80.0, k=-1.0, typ='refl', P=P, dx=3.0, dy=4.0)
    s_sphere = sphere(c=1 / 80.0, typ='refl', P=P, n=None)

    x = np.array([0.0, 1.0, -2.0])
    y = np.array([0.0, -1.0, 3.0])
    for s in (s_conic, s_oac, s_sphere):
        z, ddx, ddy = _sag_derivs(s.shape, x, y)
        assert np.all(np.isfinite(z))
        assert np.all(np.isfinite(ddx))
        assert np.all(np.isfinite(ddy))


# ---------- Q2d_and_der prelude (Phase 3.0b) ----------

def _q2d_sag_only(cm0, ams, bms, x, y, normalization_radius, c, k, dx=0, dy=0):
    """Wrapper that returns just the sag from Q2d_and_der, for FD use."""
    from prysm.x.raytracing.surfaces import Q2d_and_der
    z, _, _ = Q2d_and_der(cm0, ams, bms, x, y, normalization_radius, c, k, dx=dx, dy=dy)
    return z


@pytest.mark.parametrize('cm0,ams,bms', [
    # pure base conic, no Q expansion contribution at all
    ([0.0], [[0.0]], [[0.0]]),
    # axisymmetric Q correction (m=0 only, n=0 nonzero)
    ([1e-3], [[0.0]], [[0.0]]),
    # m=1 cosine term
    ([0.0], [[1e-3]], [[0.0]]),
    # m=2 sine term (single radial order)
    ([0.0], [[0.0], [0.0]], [[0.0], [1e-3]]),
])
def test_Q2d_and_der_polar_derivatives_match_finite_diff(cm0, ams, bms):
    """Pre-flight check on the dormant Q2d_and_der function.

    The function returns (z, dz/drho, dz/dtheta) in the surface's polar frame.
    Verify both polar derivatives match central differences in the polar frame.

    Note: ``Q2d_and_der`` calls ``cart_to_polar`` internally, which by default
    auto-meshgrids 1D inputs.  We pass 2D-shape (N, 1) coordinates here to
    bypass that behaviour and keep the per-point semantics the FD assumes.
    """
    from prysm.coordinates import polar_to_cart
    c = 1 / 80.0
    k = -1.0
    norm_r = 10.0

    rng = np.random.default_rng(0)
    # well inside the unit disk in normalized coordinates so phi stays real
    r = rng.uniform(1.0, 5.0, size=20).reshape(-1, 1)
    t = rng.uniform(0.0, 2 * np.pi, size=20).reshape(-1, 1)
    x, y = polar_to_cart(r, t)
    h = 1e-5

    from prysm.x.raytracing.surfaces import Q2d_and_der
    _, dr_an, dt_an = Q2d_and_der(cm0, ams, bms, x, y, norm_r, c, k)

    # FD in the polar frame: perturb r (or t), recompute via polar_to_cart
    xp, yp = polar_to_cart(r + h, t)
    xm, ym = polar_to_cart(r - h, t)
    z_rp = _q2d_sag_only(cm0, ams, bms, xp, yp, norm_r, c, k)
    z_rm = _q2d_sag_only(cm0, ams, bms, xm, ym, norm_r, c, k)
    dr_num = (z_rp - z_rm) / (2 * h)

    xp, yp = polar_to_cart(r, t + h)
    xm, ym = polar_to_cart(r, t - h)
    z_tp = _q2d_sag_only(cm0, ams, bms, xp, yp, norm_r, c, k)
    z_tm = _q2d_sag_only(cm0, ams, bms, xm, ym, norm_r, c, k)
    dt_num = (z_tp - z_tm) / (2 * h)

    np.testing.assert_allclose(dr_an, dr_num, rtol=1e-4, atol=1e-7)
    np.testing.assert_allclose(dt_an, dt_num, rtol=1e-4, atol=1e-7)


def test_Q2d_and_der_treats_1d_inputs_per_point():
    """Regression: cart_to_polar's vec_to_grid=True default would meshgrid 1D
    (x, y) into a 2D output.  Q2d_and_der must opt out of that."""
    from prysm.x.raytracing.surfaces import Q2d_and_der
    rng = np.random.default_rng(123)
    x = rng.uniform(-3, 3, size=10)
    y = rng.uniform(-3, 3, size=10)
    z, dr, dt = Q2d_and_der([1e-3], [[0.0]], [[0.0]], x, y,
                            normalization_radius=10.0, c=1 / 80.0, k=-1.0)
    # output shapes match input shapes (no auto-meshgrid)
    assert z.shape == (10,)
    assert dr.shape == (10,)
    assert dt.shape == (10,)


def test_Q2d_and_der_zero_coefficients_matches_conic_sag_and_normal():
    """When all Q coefficients are zero, Q2d_and_der's sag must equal the bare
    conic sag.  See note above re: 2D-shape inputs to bypass cart_to_polar's
    vec_to_grid auto-meshgrid."""
    from prysm.coordinates import polar_to_cart
    from prysm.x.raytracing.surfaces import Q2d_and_der
    c = 1 / 80.0
    k = -1.0
    norm_r = 10.0
    rng = np.random.default_rng(1)
    r = rng.uniform(0.5, 4.0, size=15).reshape(-1, 1)
    t = rng.uniform(0.0, 2 * np.pi, size=15).reshape(-1, 1)
    x, y = polar_to_cart(r, t)
    z_q, _, _ = Q2d_and_der([0.0], [[0.0]], [[0.0]], x, y, norm_r, c, k)
    # bare on-axis conic sag at the same (x, y)
    z_conic, _ = conic_sag_and_normal(c, k, x, y)
    np.testing.assert_allclose(z_q, z_conic, rtol=1e-12, atol=1e-12)


# ---------- Analytic ray-surface intersection ----------

def _ray_batch(seed=0, span=8.0, n=15):
    """Build a batch of rays: origins on z=-50 plane, mostly +z direction."""
    rng = np.random.default_rng(seed)
    xs = np.linspace(-span, span, n)
    X, Y = np.meshgrid(xs, xs, indexing='xy')
    P = np.stack([X.ravel(), Y.ravel(), np.full(X.size, -50.0)], axis=-1)
    # mostly +z, with small lateral wobble so we exercise non-axial paths
    Sx = rng.normal(scale=0.02, size=X.size)
    Sy = rng.normal(scale=0.02, size=X.size)
    Sz = np.sqrt(1 - Sx * Sx - Sy * Sy)
    S = np.stack([Sx, Sy, Sz], axis=-1)
    return P.astype(np.float64), S.astype(np.float64)


def test_ray_plane_intersect_matches_newton():
    P, S = _ray_batch()
    surf = plane('refl', np.array([0.0, 0.0, 0.0]))
    Q_an, n_an = surf.intersect(P, S)              # analytic via Plane subclass
    Q_nw, n_nw = newton_intersect(P, S, surf.sag_and_normal)
    np.testing.assert_allclose(Q_an, Q_nw, atol=1e-10)
    np.testing.assert_allclose(n_an, n_nw, atol=1e-12)


@pytest.mark.parametrize('c', [1 / 50.0, -1 / 50.0])
def test_ray_sphere_intersect_matches_newton(c):
    P, S = _ray_batch(span=4.0)
    surf = sphere(c, 'refl', np.array([0.0, 0.0, 0.0]), n=None)
    Q_an, n_an = surf.intersect(P, S)
    Q_nw, n_nw = newton_intersect(P, S, surf.sag_and_normal)
    np.testing.assert_allclose(Q_an, Q_nw, atol=1e-9)
    np.testing.assert_allclose(n_an, n_nw, atol=1e-9)


@pytest.mark.parametrize('c, k', [(1 / 50.0, -1.0), (1 / 50.0, -0.5),
                                  (1 / 50.0, 0.7), (-1 / 60.0, -1.0)])
def test_ray_conic_intersect_matches_newton(c, k):
    P, S = _ray_batch(span=4.0)
    surf = conic(c, k, 'refl', np.array([0.0, 0.0, 0.0]))
    Q_an, n_an = surf.intersect(P, S)
    Q_nw, n_nw = newton_intersect(P, S, surf.sag_and_normal)
    np.testing.assert_allclose(Q_an, Q_nw, atol=1e-9)
    np.testing.assert_allclose(n_an, n_nw, atol=1e-9)


@pytest.mark.parametrize('dx, dy', [(3.0, 0.0), (0.0, 4.0), (3.0, 4.0)])
def test_ray_off_axis_conic_intersect_matches_newton(dx, dy):
    c = 1 / 80.0
    k = -1.0
    surf = off_axis_conic(c, k, 'refl', np.array([0.0, 0.0, 0.0]),
                                  dx=dx, dy=dy)
    P, S = _ray_batch(span=3.0)
    Q_an, n_an = surf.intersect(P, S)
    Q_nw, n_nw = newton_intersect(P, S, surf.sag_and_normal)
    # Newton's per-ray convergence tolerance ~100*eps; with off-axis offsets
    # the absolute values grow, so allow a slightly looser comparison
    np.testing.assert_allclose(Q_an, Q_nw, atol=1e-8)
    np.testing.assert_allclose(n_an, n_nw, atol=1e-8)


def test_paraboloid_axial_ray_returns_vertex():
    """Special case: A_ = 1 + k Sz^2 = 0 for paraboloid + axial ray."""
    surf = conic(1.0, -1.0, 'refl', np.array([0.0, 0.0, 0.0]))
    P = np.array([[0.0, 0.0, -5.0]])
    S = np.array([[0.0, 0.0, 1.0]])
    Q, n = surf.intersect(P, S)
    np.testing.assert_allclose(Q, [[0.0, 0.0, 0.0]], atol=1e-12)
    np.testing.assert_allclose(n, [[0.0, 0.0, 1.0]], atol=1e-12)


def test_generic_surface_falls_back_to_newton():
    """A bare callable Surface must still work."""
    # build a plain Surface with the same sag and normal as Sphere, but without
    # a shape-level analytic intersect
    c = 1 / 100.0

    def sag_derivatives(x, y):
        rsq = x * x + y * y
        phi = np.sqrt(1 - c * c * rsq)
        return (c * rsq) / (1 + phi), (c * x) / phi, (c * y) / phi

    def sag(x, y):
        rsq = x * x + y * y
        return (c * rsq) / (1 + np.sqrt(1 - c * c * rsq))

    bare = Surface(
        shape=CallableShape(
            sag,
            _sag_and_normal_from_derivatives(sag_derivatives),
            params={'c': c},
        ),
        typ='refl', P=np.array([0.0, 0.0, 0.0]), n=None,
    )
    sph = sphere(c, 'refl', np.array([0.0, 0.0, 0.0]), n=None)
    P, S = _ray_batch(span=5.0)
    Q_bare, _ = bare.intersect(P, S)
    Q_sph, _ = sph.intersect(P, S)
    # the analytic Sphere result and Newton on a generic Surface should agree
    np.testing.assert_allclose(Q_bare, Q_sph, atol=1e-9)


def test_shape_with_only_sag_uses_finite_difference_normals():
    class Paraboloid(Shape):
        def sag(self, x, y):
            return 0.01 * (x * x + y * y)

    surf = Surface(shape=Paraboloid(), typ='refl',
                   P=np.array([0.0, 0.0, 0.0]))
    x = np.array([1.0, 0.0])
    y = np.array([0.0, -2.0])
    z, n_hat = surf.sag_and_normal(x, y)
    expected_z = 0.01 * (x * x + y * y)
    expected_n = np.stack([-0.02 * x, -0.02 * y, np.ones_like(x)], axis=-1)
    expected_n = expected_n / np.linalg.norm(expected_n, axis=-1,
                                             keepdims=True)
    np.testing.assert_allclose(z, expected_z, atol=1e-14)
    np.testing.assert_allclose(n_hat, expected_n, atol=1e-8)


# ---------- Even asphere ----------

@pytest.mark.parametrize('coefs', [
    (),                      # pure conic (degenerate case)
    (1e-3,),                 # r^4 only
    (1e-3, 1e-5, 1e-7),      # r^4 + r^6 + r^8
])
def test_even_asphere_sag_matches_explicit(coefs):
    c = 1 / 5.0
    k = -0.5
    rng = np.random.default_rng(0)
    rsq = rng.uniform(0.0, 4.0, size=20)
    z = even_asphere_sag(c, k, coefs, rsq)
    z_conic = conic_sag(c, k, rsq)
    poly = sum(a * rsq ** (i + 2) for i, a in enumerate(coefs))
    np.testing.assert_allclose(z, z_conic + poly, rtol=0, atol=1e-14)


@pytest.mark.parametrize('coefs', [(), (1e-3,), (5e-4, -1e-6, 2e-8)])
def test_even_asphere_der_xy_matches_numerical(coefs):
    c = 1 / 5.0
    k = -0.5
    xs = np.linspace(-1.5, 1.5, 17)
    X, Y = np.meshgrid(xs, xs, indexing='xy')
    h = 1e-6

    def sag(xx, yy):
        return even_asphere_sag(c, k, coefs, xx * xx + yy * yy)

    dx_an, dy_an = even_asphere_sag_der_xy(c, k, coefs, X, Y)
    dx_num = (sag(X + h, Y) - sag(X - h, Y)) / (2 * h)
    dy_num = (sag(X, Y + h) - sag(X, Y - h)) / (2 * h)
    np.testing.assert_allclose(dx_an, dx_num, atol=1e-7, rtol=1e-6)
    np.testing.assert_allclose(dy_an, dy_num, atol=1e-7, rtol=1e-6)


def test_even_asphere_with_empty_coefs_equals_conic():
    """Empty coefficient list ⇒ EvenAsphere acts as a Conic."""
    c, k = 1 / 80.0, -1.0
    P0 = np.array([0.0, 0.0, 0.0])
    s_asph = even_asphere(c, k, (), 'refl', P0)
    s_conic = conic(c, k, 'refl', P0)
    P, S = _ray_batch(span=4.0)
    Q_a, n_a = s_asph.intersect(P, S)
    Q_c, n_c = s_conic.intersect(P, S)
    np.testing.assert_allclose(Q_a, Q_c, atol=1e-12)
    np.testing.assert_allclose(n_a, n_c, atol=1e-12)


def test_even_asphere_intersect_matches_naive_newton():
    """EvenAsphere.intersect (conic-seeded Newton) must match the
    bare-Surface Newton starting from t=0, on the same sag and normal."""
    c, k = 1 / 5.0, -0.5
    coefs = (1e-3, 1e-5, 1e-7)
    P0 = np.array([0.0, 0.0, 0.0])
    s_asph = even_asphere(c, k, coefs, 'refr', P0, n=lambda w: 1.5)
    # bare Surface using the same sag and normal
    bare = Surface(shape=CallableShape(s_asph.sag, s_asph.sag_and_normal,
                                       params=dict(s_asph.params)),
                   typ='refr', P=P0, n=lambda w: 1.5)
    P, S = _ray_batch(span=1.5)
    Q_a, n_a = s_asph.intersect(P, S)
    Q_b, n_b = bare.intersect(P, S)
    np.testing.assert_allclose(Q_a, Q_b, atol=1e-9)
    np.testing.assert_allclose(n_a, n_b, atol=1e-9)


def test_even_asphere_intersect_converges_with_low_maxiter():
    """The conic-seeded Newton should converge in very few iterations."""
    c, k = 1 / 10.0, -1.0
    coefs = (1e-4, 1e-6)
    s = even_asphere(c, k, coefs, 'refr',
                             np.array([0.0, 0.0, 0.0]), n=lambda w: 1.5)
    P, S = _ray_batch(span=2.0)
    # 5 iterations is comfortably enough when seeded from the conic root;
    # the same call on a bare Surface starting from t=0 would need many more
    Q_fast, _ = s.intersect(P, S, maxiter=5)
    Q_ref, _ = s.intersect(P, S)  # default maxiter
    np.testing.assert_allclose(Q_fast, Q_ref, atol=1e-12)
    assert np.all(np.isfinite(Q_fast))


def test_raytrace_end_to_end_analytic_vs_newton():
    """Full-system raytrace: Newton (via callable Surface) and analytic (via shapes)
    must agree on every (P, S) and OPL up to Newton tolerance."""
    c1, c2, k1, k2 = 1 / -200.0, 1 / -67.0, -1.0, -2.5
    P_pm = np.array([0.0, 0.0, 0.0])
    P_sm = np.array([0.0, 0.0, -80.0])
    P_img = np.array([0.0, 0.0, 50.0])

    # analytic path: shape objects provide analytic intersection
    surfs_an = [
        conic(c1, k1, 'refl', P_pm),
        conic(c2, k2, 'refl', P_sm),
        plane('eval', P_img),
    ]

    # newton path: build bare Surface with the same sag and normal
    def make_conic(cc, kk):
        def sag_derivatives(x, y):
            rsq = x * x + y * y
            phi = np.sqrt(1 - (1 + kk) * cc * cc * rsq)
            return (cc * rsq) / (1 + phi), (cc * x) / phi, (cc * y) / phi

        def sag(x, y):
            rsq = x * x + y * y
            return (cc * rsq) / (1 + np.sqrt(1 - (1 + kk) * cc * cc * rsq))

        return sag, _sag_and_normal_from_derivatives(sag_derivatives)

    def plane_sag_and_normal(x, y):
        zero = np.broadcast_to(np.array([0.0], dtype=x.dtype), x.shape)
        normal = np.stack([zero, zero, np.ones_like(zero)], axis=-1)
        return zero, normal

    def plane_sag(x, y):
        return np.broadcast_to(np.array([0.0], dtype=x.dtype), x.shape)

    f1, n1 = make_conic(c1, k1)
    f2, n2 = make_conic(c2, k2)
    surfs_nw = [
        Surface(shape=CallableShape(f1, n1), typ='refl', P=P_pm, n=None),
        Surface(shape=CallableShape(f2, n2), typ='refl', P=P_sm, n=None),
        Surface(shape=CallableShape(plane_sag, plane_sag_and_normal),
                typ='eval', P=P_img, n=None),
    ]

    P0, S0 = generate_collimated_ray_fan(11, maxr=20.0, z=-1e3, azimuth=90)
    trace_an = raytrace(surfs_an, P0, S0, wvl=0.55)
    trace_nw = raytrace(surfs_nw, P0, S0, wvl=0.55)
    np.testing.assert_allclose(trace_an.P, trace_nw.P, atol=1e-7)
    np.testing.assert_allclose(trace_an.S, trace_nw.S, atol=1e-9)
    np.testing.assert_allclose(trace_an.OPL, trace_nw.OPL, atol=1e-7)


# ---------- Q2D ----------

def test_q2d_zero_coefficients_matches_conic():
    """Q2D with all-zero Q coefs is exactly a Conic."""
    c, k = 1 / 200., -1.0
    P_at = np.array([0., 0., 0.])
    s_q2d = q2d(c=c, k=k, normalization_radius=20.0,
                        cm0=[0.0], ams=[[0.0]], bms=[[0.0]],
                        typ='refl', P=P_at)
    s_conic = conic(c=c, k=k, typ='refl', P=P_at)
    P, S = _ray_batch(span=10.0)
    Q_q, n_q = s_q2d.intersect(P, S)
    Q_c, n_c = s_conic.intersect(P, S)
    np.testing.assert_allclose(Q_q, Q_c, atol=1e-9)
    np.testing.assert_allclose(n_q, n_c, atol=1e-9)


def test_q2d_factory_returns_surface_with_q2d_shape():
    s = q2d(c=1 / 100., k=0.0, normalization_radius=10.0,
                    cm0=[0.0], ams=[[0.0]], bms=[[0.0]],
                    typ='refl', P=np.array([0., 0., 0.]))
    assert isinstance(s.shape, Q2D)
    # params dict is preserved
    assert s.params['normalization_radius'] == 10.0


def test_q2d_derivative_sag_matches_Q2d_and_der_directly():
    """Q2D.shape.sag must return the same z as a direct Q2d_and_der call."""
    from prysm.x.raytracing.surfaces import Q2d_and_der
    c, k = 1 / 200., -1.0
    norm_r = 15.0
    cm0 = (1e-3, 5e-5)
    ams = ((1e-4, 0.0),)
    bms = ((0.0, 0.0),)
    s = q2d(c=c, k=k, normalization_radius=norm_r,
                    cm0=cm0, ams=ams, bms=bms,
                    typ='refl', P=np.array([0., 0., 0.]))
    rng = np.random.default_rng(7)
    x = rng.uniform(-3, 3, size=10)
    y = rng.uniform(-3, 3, size=10)
    z_der = s.shape.sag(x, y)
    z_direct, _, _ = Q2d_and_der(cm0, ams, bms, x, y, norm_r, c, k)
    np.testing.assert_allclose(z_der, z_direct, atol=1e-12)


def test_q2d_xy_derivatives_match_finite_diff():
    """The chain-rule polar-to-Cartesian conversion in Q2D must agree
    with central differences taken on the sag.  Skip points near r=0 (where
    the function approximates the gradient as zero)."""
    c, k = 1 / 200., -1.0
    norm_r = 15.0
    cm0 = (1e-3, 5e-5)
    ams = ((1e-4,),)
    bms = ((5e-5,),)
    s = q2d(c=c, k=k, normalization_radius=norm_r,
                    cm0=cm0, ams=ams, bms=bms,
                    typ='refl', P=np.array([0., 0., 0.]))
    rng = np.random.default_rng(9)
    # stay away from the origin so the r=0 patch isn't exercised
    r_min = 0.5
    angles = rng.uniform(0, 2 * np.pi, size=15)
    radii = rng.uniform(r_min, 5.0, size=15)
    x = (radii * np.cos(angles))
    y = (radii * np.sin(angles))
    h = 1e-5

    z, ddx_an, ddy_an = _sag_derivs(s.shape, x, y)
    z_xp = s.shape.sag(x + h, y); z_xm = s.shape.sag(x - h, y)
    z_yp = s.shape.sag(x, y + h); z_ym = s.shape.sag(x, y - h)
    ddx_num = (z_xp - z_xm) / (2 * h)
    ddy_num = (z_yp - z_ym) / (2 * h)
    np.testing.assert_allclose(ddx_an, ddx_num, atol=1e-5, rtol=1e-4)
    np.testing.assert_allclose(ddy_an, ddy_num, atol=1e-5, rtol=1e-4)


def test_q2d_intersect_finite_at_origin():
    """An axial ray intersecting a Q2D surface must return a finite (Q, normal)."""
    s = q2d(c=1 / 200., k=-1.0, normalization_radius=20.0,
                    cm0=[1e-3], ams=[[0.0]], bms=[[0.0]],
                    typ='refl', P=np.array([0., 0., 0.]))
    P = np.array([[0., 0., -100.]])
    S = np.array([[0., 0., 1.]])
    Q, n = s.intersect(P, S)
    assert np.all(np.isfinite(Q))
    assert np.all(np.isfinite(n))


def test_q2d_intersect_seeded_newton_converges_quickly():
    """The conic-seeded Newton should reach machine precision in just a few
    iterations because the Q correction is small."""
    s = q2d(c=1 / 200., k=-1.0, normalization_radius=20.0,
                    cm0=[5e-4], ams=[[0.0]], bms=[[0.0]],
                    typ='refl', P=np.array([0., 0., 0.]))
    P, S = _ray_batch(span=8.0)
    Q_fast, _ = s.intersect(P, S, maxiter=5)
    Q_ref, _ = s.intersect(P, S)  # default maxiter=100
    np.testing.assert_allclose(Q_fast, Q_ref, atol=1e-12)


def test_q2d_with_off_axis_conic_base():
    """Q2D supports a dx/dy-shifted base conic (off-axis paraboloid + Q)."""
    s = q2d(c=1 / 200., k=-1.0, normalization_radius=20.0,
                    cm0=[0.0], ams=[[0.0]], bms=[[0.0]],
                    typ='refl', P=np.array([0., 0., 0.]),
                    dx=0.0, dy=20.0)
    s_oac = off_axis_conic(c=1 / 200., k=-1.0, typ='refl',
                                   P=np.array([0., 0., 0.]),
                                   dy=20.0)
    P, S = _ray_batch(span=8.0)
    Q_q, _ = s.intersect(P, S)
    Q_oac, _ = s_oac.intersect(P, S)
    np.testing.assert_allclose(Q_q, Q_oac, atol=1e-9)
