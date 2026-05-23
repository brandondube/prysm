"""Tests for coordinate conversion."""
import pytest

import numpy as np

from prysm import coordinates

TEST_SAMPLES = 32


@pytest.fixture
def data_2d():
    x, y = np.linspace(-1, 1, TEST_SAMPLES), np.linspace(-1, 1, TEST_SAMPLES)
    xx, yy = np.meshgrid(x, y)
    dat = xx + yy
    return x, y, dat


@pytest.fixture
def data_2d_complex():
    x, y = np.linspace(-1, 1, TEST_SAMPLES), np.linspace(-1, 1, TEST_SAMPLES)
    xx, yy = np.meshgrid(x, y)
    dat = xx + 1j * yy
    return x, y, dat


@pytest.mark.parametrize('x, y', [
    [1, 0],
    [0, 1],
    [1, 1],
    [-1, 0],
    [0, -1],
    [-1, -1],
    [np.linspace(-1, 1, TEST_SAMPLES), np.linspace(-1, 1, TEST_SAMPLES)]])
def test_cart_to_polar(x, y):
    rho, phi = coordinates.cart_to_polar(x, y, vec_to_grid=False)
    assert np.allclose(rho, np.sqrt(x**2 + y**2))
    assert np.allclose(phi, np.arctan2(y, x))


@pytest.mark.parametrize('rho, phi', [
    [1, 0],
    [0, 90],
    [0, 180],
    [-1, 90],
    [np.linspace(0, 1, TEST_SAMPLES), np.linspace(0, 2 * np.pi, TEST_SAMPLES)]])
def test_polar_to_cart(rho, phi):
    x, y = coordinates.polar_to_cart(rho, phi)
    assert np.allclose(x, rho * np.cos(phi))
    assert np.allclose(y, rho * np.sin(phi))


def test_sample_axis_cheby_matches_lobatto_endpoints():
    x = coordinates.sample_axis('cheby', -2, 2, 5)
    np.testing.assert_allclose(x[[0, -1]], [-2, 2])
    assert abs(x[2]) < 1e-15


def test_promote_3d_point_scalar_and_trailing_values():
    np.testing.assert_allclose(coordinates.promote_3d_point(5), [0, 0, 5])
    np.testing.assert_allclose(coordinates.promote_3d_point([2, 5]), [0, 2, 5])
    np.testing.assert_allclose(coordinates.promote_3d_point([1, 2, 5]), [1, 2, 5])


def test_uniform_cart_to_polar_preserves_constant_field():
    x = np.linspace(-1, 1, TEST_SAMPLES)
    y = np.linspace(-1, 1, TEST_SAMPLES)
    dat = np.ones((TEST_SAMPLES, TEST_SAMPLES))

    rho, phi, result = coordinates.uniform_cart_to_polar(x, y, dat)

    assert rho[0] == 0
    assert phi[0] == 0
    np.testing.assert_allclose(result, 1)


@pytest.mark.skip('changed rotation order, need to re-do scipy match')
def test_make_rotation_matrix_matches_scipy():
    from scipy.spatial.transform import Rotation as R

    angles = (1, 2, 3)
    sp = R.from_euler('ZYX', angles, degrees=True).as_matrix()
    pry = coordinates.make_rotation_matrix(angles)
    assert np.allclose(sp, pry)


def test_warp_identity_coordinates_returns_input():
    z = np.arange(16, dtype=float).reshape(4, 4)
    yy, xx = np.indices(z.shape)

    out = coordinates.warp(z, xx, yy)

    np.testing.assert_allclose(out, z, atol=1e-14)


def test_distort_annular_grid_maps_obscuration_to_zero_and_outer_radius_to_one():
    r = np.asarray([0.2, 0.6, 1.0])

    out = coordinates.distort_annular_grid(r, eps=0.2)

    np.testing.assert_allclose(out, [0.0, 0.5, 1.0])


def test_chebygauss_quadrature_xy_honors_radius_and_center():
    x, y = coordinates.chebygauss_quadrature_xy(3, radius=2, center=(10, -5))
    radii = np.hypot(x - 10, y + 5)

    assert x.shape == y.shape == (21,)
    assert radii.max() < 2
    assert radii.min() > 0
