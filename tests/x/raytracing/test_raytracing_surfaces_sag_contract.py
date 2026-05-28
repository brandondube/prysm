"""Every Surface subclass must expose sag(x, y) that agrees with sag_and_normal.

This file pins the new surface contract: for one constructed instance of each
surface type, sag(x, y) must equal sag_and_normal(x, y)[0] to f64 tolerance on
a fixed grid.
"""
import numpy as np
import pytest

from tests.x.raytracing.surface_helpers import (
    plane, sphere, conic, off_axis_conic, even_asphere, q2d, zernike, xy,
    chebyshev, jacobi, toroid, biconic,
)

from prysm.x.raytracing.surfaces import Conic, Surface


P0 = np.array([0.0, 0.0, 0.0])


def _grid():
    x = np.linspace(-2.0, 2.0, 11)
    y = np.linspace(-2.0, 2.0, 11)
    return x, y


def _check(surf):
    x, y = _grid()
    z_sag = surf.sag(x, y)
    z_normal, *_ = surf.sag_and_normal(x, y)
    np.testing.assert_allclose(np.asarray(z_sag), np.asarray(z_normal),
                               rtol=0, atol=1e-12)


def test_plane_sag_matches_sag_and_normal():
    _check(plane('refl', P0))


def test_sphere_sag_matches_sag_and_normal():
    _check(sphere(1 / 50.0, 'refl', P0, n=None))


def test_conic_sag_matches_sag_and_normal():
    _check(conic(1 / 50.0, -0.5, 'refl', P0))


def test_shifted_conic_shape_sag_matches_sag_and_normal():
    _check(off_axis_conic(1 / 50.0, -0.5, 'refl', P0,
                                  dx=10.0, dy=5.0))


def test_even_asphere_sag_matches_sag_and_normal():
    _check(even_asphere(1 / 50.0, -0.5, (1e-4, 1e-6),
                                'refl', P0))


def test_q2d_sag_matches_sag_and_normal():
    # axisymmetric m=0 plus a small (m=1, n=0) cosine term
    _check(q2d(c=1 / 50.0, k=-0.5,
                       normalization_radius=10.0,
                       cm0=(0.0, 1e-3),
                       ams=((1e-4,),),
                       bms=((0.0,),),
                       typ='refl', P=P0))


def test_zernike_sag_matches_sag_and_normal():
    _check(zernike(c=1 / 50.0, k=-0.5,
                           normalization_radius=10.0,
                           nms=[(2, 0), (4, 0), (3, 1)],
                           coefs=[1e-3, 5e-4, 2e-4],
                           typ='refl', P=P0))


def test_xy_sag_matches_sag_and_normal():
    _check(xy(c=1 / 50.0, k=-0.5,
                      normalization_radius=10.0,
                      mns=[(2, 0), (0, 2), (1, 1)],
                      coefs=[1e-3, 1e-3, 5e-4],
                      typ='refl', P=P0))


def test_chebyshev_sag_matches_sag_and_normal():
    _check(chebyshev(c=1 / 50.0, k=-0.5,
                             x_norm=10.0, y_norm=10.0,
                             mns=[(2, 0), (0, 2), (1, 1)],
                             coefs=[1e-3, 1e-3, 5e-4],
                             typ='refl', P=P0))


def test_jacobi_sag_matches_sag_and_normal():
    _check(jacobi(c=1 / 50.0, k=-0.5,
                          normalization_radius=10.0,
                          alpha=0.0, beta=0.0,
                          ns=[1, 2, 3],
                          coefs=[1e-3, 5e-4, 2e-4],
                          typ='refl', P=P0))


def test_toroid_sag_matches_sag_and_normal():
    _check(toroid(c_x=1 / 80.0, c_y=1 / 50.0, k_y=-0.5,
                          coefs_y=(1e-4,),
                          typ='refl', P=P0))


def test_biconic_sag_matches_sag_and_normal():
    _check(biconic(c_x=1 / 80.0, c_y=1 / 50.0,
                           k_x=-0.2, k_y=-0.5,
                           typ='refl', P=P0))


def test_shape_required_by_init():
    with pytest.raises(TypeError):
        Surface(typ='refl', P=P0, n=None)


def test_explicit_shape_constructor_uses_mutable_shape_params():
    surf = Surface(shape=Conic(c=1 / 50.0, k=0.0),
                   interaction='refl', P=P0)
    assert surf.params is surf.shape.params
    z1 = surf.sag(np.array([1.0]), np.array([0.0]))
    surf.shape.params['c'] = 1 / 25.0
    z2 = surf.sag(np.array([1.0]), np.array([0.0]))
    assert float(z2[0]) > float(z1[0])
