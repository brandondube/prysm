"""Tests for the polynomial-sag surface classes (Phase 5)."""
import numpy as np
import pytest

from tests.x.raytracing.surface_helpers import (
    plane, sphere, conic, off_axis_conic, even_asphere, q2d, zernike, xy,
    chebyshev, jacobi, toroid, biconic,
)

from prysm.x.raytracing.surfaces import (
    Surface,
    Zernike,
    XY,
    Chebyshev,
    Jacobi,
)
from prysm.x.raytracing.intersections import ConicSeedMixin
from prysm.x.raytracing.spencer_and_murty import raytrace
from prysm.x.raytracing.raygen import generate_collimated_rect_ray_grid
from prysm.polynomials import (
    zernike_sum_der_xy,
    cheby1_seq,
    jacobi_seq,
)


def _xy_grid(rmax=4.0, n=9):
    x1 = np.linspace(-rmax, rmax, n)
    x, y = np.meshgrid(x1, x1, indexing='xy')
    return x, y


def _sag_derivs(shape, x, y):
    """Recover (z, dz/dx, dz/dy) from a shape's sag_and_normal unit normal."""
    z, n_hat = shape.sag_and_normal(x, y)
    nz = n_hat[..., 2]
    return z, -n_hat[..., 0] / nz, -n_hat[..., 1] / nz


def _central_difference_xy(sag, x, y, h=1e-6):
    z_xp = sag(x + h, y)
    z_xm = sag(x - h, y)
    z_yp = sag(x, y + h)
    z_ym = sag(x, y - h)
    return (z_xp - z_xm) / (2 * h), (z_yp - z_ym) / (2 * h)


# ---------- shared base / inheritance ----------------------------------------

def test_polynomial_shapes_use_conic_seeded_newton():
    """All polynomial sag shapes share the conic-seeded Newton intersect."""
    for cls in (Zernike, XY, Chebyshev, Jacobi):
        assert issubclass(cls, ConicSeedMixin), cls.__name__


# ---------- Zernike ----------------------------------------------------------

def test_zernike_zero_coefs_matches_conic():
    c, k = 1 / 80.0, -1.0
    s_zern = zernike(c=c, k=k, normalization_radius=10.0,
                             nms=[], coefs=[], typ='refl', P=[0, 0, 0])
    s_conic = conic(c=c, k=k, typ='refl', P=[0, 0, 0])
    x, y = _xy_grid()
    z_z, dx_z, dy_z = _sag_derivs(s_zern.shape, x, y)
    z_c, dx_c, dy_c = _sag_derivs(s_conic.shape, x, y)
    np.testing.assert_allclose(z_z, z_c, atol=1e-12)
    np.testing.assert_allclose(dx_z, dx_c, atol=1e-12)
    np.testing.assert_allclose(dy_z, dy_c, atol=1e-12)


def test_zernike_sag_matches_library():
    R_n = 8.0
    nms = [(2, 0), (3, 1), (4, 0), (3, -1)]
    coefs = [0.05, -0.02, 0.03, 0.01]
    s = zernike(c=0.0, k=0.0, normalization_radius=R_n,
                        nms=nms, coefs=coefs, typ='refl', P=[0, 0, 0])
    x, y = _xy_grid()
    z_s = s.shape.sag(x, y)
    z_lib, _, _ = zernike_sum_der_xy(coefs, nms, x / R_n, y / R_n, norm=True)
    np.testing.assert_allclose(z_s, z_lib, atol=1e-12)


def test_zernike_derivatives_central_diff():
    s = zernike(c=1 / 80.0, k=0.0, normalization_radius=10.0,
                        nms=[(2, 0), (4, 0), (3, 1), (3, -1)],
                        coefs=[0.05, 0.02, -0.03, 0.04],
                        typ='refl', P=[0, 0, 0])
    x, y = _xy_grid()
    _, dx_an, dy_an = _sag_derivs(s.shape, x, y)
    dx_num, dy_num = _central_difference_xy(s.shape.sag, x, y)
    np.testing.assert_allclose(dx_an, dx_num, rtol=2e-5, atol=1e-7)
    np.testing.assert_allclose(dy_an, dy_num, rtol=2e-5, atol=1e-7)


# ---------- XY ----------------------------------------------------------------

def test_xy_zero_coefs_matches_conic():
    c, k = 1 / 50.0, 0.0
    s_xy = xy(c=c, k=k, normalization_radius=1.0,
                      mns=[], coefs=[], typ='refl', P=[0, 0, 0])
    s_conic = conic(c=c, k=k, typ='refl', P=[0, 0, 0])
    x, y = _xy_grid()
    z_xy, dx_xy, dy_xy = _sag_derivs(s_xy.shape, x, y)
    z_c, dx_c, dy_c = _sag_derivs(s_conic.shape, x, y)
    np.testing.assert_allclose(z_xy, z_c, atol=1e-12)
    np.testing.assert_allclose(dx_xy, dx_c, atol=1e-12)
    np.testing.assert_allclose(dy_xy, dy_c, atol=1e-12)


def test_xy_sag_matches_direct_polynomial():
    R_n = 5.0
    mns = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2), (3, 1)]
    coefs = [0.1, 0.05, -0.04, 0.02, 0.01, -0.015, 0.003]
    s = xy(c=0.0, k=0.0, normalization_radius=R_n,
                   mns=mns, coefs=coefs, typ='refl', P=[0, 0, 0])
    x, y = _xy_grid(rmax=2.0, n=7)
    z_s = s.shape.sag(x, y)
    xn = x / R_n
    yn = y / R_n
    z_ref = sum(c * xn ** m * yn ** n for c, (m, n) in zip(coefs, mns))
    np.testing.assert_allclose(z_s, z_ref, atol=1e-12)


def test_xy_derivatives_central_diff():
    s = xy(c=1 / 80.0, k=0.0, normalization_radius=10.0,
                   mns=[(0, 0), (2, 0), (0, 2), (1, 1), (3, 1), (2, 2)],
                   coefs=[0.0, 0.05, 0.04, 0.02, 0.005, 0.003],
                   typ='refl', P=[0, 0, 0])
    x, y = _xy_grid()
    _, dx_an, dy_an = _sag_derivs(s.shape, x, y)
    dx_num, dy_num = _central_difference_xy(s.shape.sag, x, y)
    np.testing.assert_allclose(dx_an, dx_num, rtol=2e-5, atol=1e-7)
    np.testing.assert_allclose(dy_an, dy_num, rtol=2e-5, atol=1e-7)


# ---------- Chebyshev --------------------------------------------------------

def test_chebyshev_zero_coefs_matches_conic():
    c, k = 1 / 50.0, 0.0
    s_cb = chebyshev(c=c, k=k, x_norm=10.0, y_norm=10.0,
                             mns=[], coefs=[], typ='refl', P=[0, 0, 0])
    s_conic = conic(c=c, k=k, typ='refl', P=[0, 0, 0])
    x, y = _xy_grid()
    z_cb, dx_cb, dy_cb = _sag_derivs(s_cb.shape, x, y)
    z_c, dx_c, dy_c = _sag_derivs(s_conic.shape, x, y)
    np.testing.assert_allclose(z_cb, z_c, atol=1e-12)
    np.testing.assert_allclose(dx_cb, dx_c, atol=1e-12)
    np.testing.assert_allclose(dy_cb, dy_c, atol=1e-12)


def test_chebyshev_sag_matches_library():
    x_norm, y_norm = 8.0, 6.0
    mns = [(0, 0), (2, 0), (0, 2), (1, 1), (4, 0), (2, 2), (3, 1)]
    coefs = [0.02, 0.05, 0.04, -0.03, 0.01, 0.005, 0.003]
    s = chebyshev(c=0.0, k=0.0, x_norm=x_norm, y_norm=y_norm,
                          mns=mns, coefs=coefs, typ='refl', P=[0, 0, 0])
    x, y = _xy_grid()
    z_s = s.shape.sag(x, y)
    Tx = cheby1_seq(range(max(m for m, _ in mns) + 1), x / x_norm)
    Ty = cheby1_seq(range(max(n for _, n in mns) + 1), y / y_norm)
    z_ref = np.zeros_like(x)
    for c, (m, n) in zip(coefs, mns):
        z_ref = z_ref + c * Tx[m] * Ty[n]
    np.testing.assert_allclose(z_s, z_ref, atol=1e-12)


def test_chebyshev_derivatives_central_diff():
    s = chebyshev(c=1 / 80.0, k=0.0, x_norm=10.0, y_norm=10.0,
                          mns=[(0, 0), (2, 0), (0, 2), (1, 1), (4, 0)],
                          coefs=[0.01, 0.05, 0.04, -0.02, 0.01],
                          typ='refl', P=[0, 0, 0])
    x, y = _xy_grid()
    _, dx_an, dy_an = _sag_derivs(s.shape, x, y)
    dx_num, dy_num = _central_difference_xy(s.shape.sag, x, y)
    np.testing.assert_allclose(dx_an, dx_num, rtol=2e-5, atol=1e-7)
    np.testing.assert_allclose(dy_an, dy_num, rtol=2e-5, atol=1e-7)


# ---------- Jacobi (axisymmetric radial) -------------------------------------

def test_jacobi_zero_coefs_matches_conic():
    c, k = 1 / 50.0, 0.0
    s_j = jacobi(c=c, k=k, normalization_radius=10.0,
                         alpha=0.0, beta=0.0, ns=[], coefs=[],
                         typ='refl', P=[0, 0, 0])
    s_conic = conic(c=c, k=k, typ='refl', P=[0, 0, 0])
    x, y = _xy_grid()
    z_j, dx_j, dy_j = _sag_derivs(s_j.shape, x, y)
    z_c, dx_c, dy_c = _sag_derivs(s_conic.shape, x, y)
    np.testing.assert_allclose(z_j, z_c, atol=1e-12)
    np.testing.assert_allclose(dx_j, dx_c, atol=1e-12)
    np.testing.assert_allclose(dy_j, dy_c, atol=1e-12)


def test_jacobi_sag_matches_library():
    R_n = 8.0
    alpha, beta = 0.5, 0.5
    ns = [0, 1, 2, 3]
    coefs = [0.01, 0.02, 0.03, -0.01]
    s = jacobi(c=0.0, k=0.0, normalization_radius=R_n,
                       alpha=alpha, beta=beta, ns=ns, coefs=coefs,
                       typ='refl', P=[0, 0, 0])
    x, y = _xy_grid()
    z_s = s.shape.sag(x, y)
    rsq = x * x + y * y
    u = 2 * rsq / (R_n * R_n) - 1
    Pn = jacobi_seq(ns, alpha, beta, u)
    z_ref = sum(c * Pn[i] for i, c in enumerate(coefs))
    np.testing.assert_allclose(z_s, z_ref, atol=1e-12)


@pytest.mark.parametrize('alpha,beta', [(0.0, 0.0), (-0.5, -0.5),
                                        (0.5, 0.5), (1.0, 0.0)])
def test_jacobi_derivatives_central_diff(alpha, beta):
    s = jacobi(c=1 / 80.0, k=0.0, normalization_radius=10.0,
                       alpha=alpha, beta=beta, ns=[0, 1, 2, 3],
                       coefs=[0.0, 0.05, 0.02, -0.01],
                       typ='refl', P=[0, 0, 0])
    x, y = _xy_grid()
    _, dx_an, dy_an = _sag_derivs(s.shape, x, y)
    dx_num, dy_num = _central_difference_xy(s.shape.sag, x, y)
    np.testing.assert_allclose(dx_an, dx_num, rtol=2e-5, atol=1e-7)
    np.testing.assert_allclose(dy_an, dy_num, rtol=2e-5, atol=1e-7)


def test_jacobi_no_origin_singularity():
    """sag and derivatives finite at r=0 even for high orders / near-zero r."""
    s = jacobi(c=1 / 80.0, k=0.0, normalization_radius=10.0,
                       alpha=0.0, beta=0.0, ns=[0, 1, 2, 3, 4, 5],
                       coefs=[0.01, 0.05, -0.03, 0.02, -0.01, 0.005],
                       typ='refl', P=[0, 0, 0])
    x = np.array([0.0, 1e-12, 1.0])
    y = np.array([0.0, 1e-12, 0.5])
    z, dx, dy = _sag_derivs(s.shape, x, y)
    assert np.isfinite(z).all()
    assert np.isfinite(dx).all()
    assert np.isfinite(dy).all()


# ---------- intersect convergence + raytrace round-trip ---------------------

@pytest.fixture
def _polynomial_surfaces():
    """Each polynomial-sag surface with a small perturbation; same conic base."""
    c, k = 1 / 80.0, 0.0
    return [
        zernike(c=c, k=k, normalization_radius=10.0,
                        nms=[(2, 0), (3, 1)], coefs=[0.05, 0.02],
                        typ='refl', P=[0, 0, 0]),
        xy(c=c, k=k, normalization_radius=10.0,
                   mns=[(2, 0), (1, 1)], coefs=[0.05, 0.02],
                   typ='refl', P=[0, 0, 0]),
        chebyshev(c=c, k=k, x_norm=10.0, y_norm=10.0,
                          mns=[(2, 0), (0, 2)], coefs=[0.05, 0.04],
                          typ='refl', P=[0, 0, 0]),
        jacobi(c=c, k=k, normalization_radius=10.0,
                       alpha=0.0, beta=0.0, ns=[1, 2], coefs=[0.05, 0.02],
                       typ='refl', P=[0, 0, 0]),
    ]


def test_polynomial_surfaces_intersect_lands_on_surface(_polynomial_surfaces):
    """Newton intersect lands rays on each surface (Q.z == sag(Q.xy))."""
    P = np.array([[1.0, 0.5, -50.0],
                  [-1.0, 0.0, -50.0],
                  [0.0, 0.0, -50.0],
                  [3.0, -2.0, -50.0]])
    S = np.array([[0.0, 0.0, 1.0]] * 4)
    for surf in _polynomial_surfaces:
        Q, _, valid = surf.intersect(P, S, return_valid=True)
        assert valid.all(), f'{type(surf).__name__} intersect failed'
        z = surf.shape.sag(Q[..., 0], Q[..., 1])
        np.testing.assert_allclose(Q[..., 2], z, atol=1e-9,
                                   err_msg=type(surf).__name__)


def test_polynomial_surfaces_zero_pert_matches_conic_image_spot():
    """A zero-perturbation polynomial mirror gives the same image-plane spot
    as a pure conic with the same base parameters."""
    c, k = 1 / 80.0, -1.0  # parabolic
    f = -1.0 / (2.0 * c)  # paraxial focus, negative side after reflection
    P, S = generate_collimated_rect_ray_grid(nrays=5, maxx=5, miny=-5, maxy=5)
    s_image = plane(typ='eval', P=[0, 0, f])

    s_conic = conic(c=c, k=k, typ='refl', P=[0, 0, 0])
    res_conic = raytrace([s_conic, s_image], P, S, wvl=0.55)
    spot_c = res_conic.P[-1, ..., :2]

    polys = [
        zernike(c=c, k=k, normalization_radius=10.0,
                        nms=[], coefs=[], typ='refl', P=[0, 0, 0]),
        xy(c=c, k=k, normalization_radius=10.0,
                   mns=[], coefs=[], typ='refl', P=[0, 0, 0]),
        chebyshev(c=c, k=k, x_norm=10.0, y_norm=10.0,
                          mns=[], coefs=[], typ='refl', P=[0, 0, 0]),
        jacobi(c=c, k=k, normalization_radius=10.0,
                       alpha=0.0, beta=0.0, ns=[], coefs=[],
                       typ='refl', P=[0, 0, 0]),
    ]
    for surf in polys:
        res = raytrace([surf, s_image], P, S, wvl=0.55)
        spot = res.P[-1, ..., :2]
        np.testing.assert_allclose(spot, spot_c, atol=1e-9,
                                   err_msg=type(surf).__name__)
