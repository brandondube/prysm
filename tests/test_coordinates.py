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


# TODO: tests below here are for function, not accuracy


def test_uniform_cart_polar_functions(data_2d):
    x, y, dat = data_2d
    rho, phi, result = coordinates.uniform_cart_to_polar(x, y, dat)
    assert type(rho) is np.ndarray
    assert type(phi) is np.ndarray
    assert type(result) is np.ndarray


# TODO: add a test that this returns expected points for a known function
def test_resample_2d_does_not_distort(data_2d):
    x, y, dat = data_2d
    xx, yy = np.meshgrid(x, y)
    resampled = coordinates.resample_2d(dat, (x, y), (xx, yy))
    assert np.allclose(dat, resampled)


# def test_resample_2d_complex_does_not_distort(data_2d_complex):
#     x, y, dat = data_2d_complex
#     xx, yy = np.meshgrid(x, y)
#     resampled = coordinates.resample_2d_complex(dat, (x, y), (xx, yy))
#     assert np.allclose(dat, resampled)


def test_make_rotation_matrix_matches_scipy():
    from scipy.spatial.transform import Rotation as R

    angles = (1, 2, 3)
    sp = R.from_euler('ZYX', angles, degrees=True).as_matrix()
    pry = coordinates.make_rotation_matrix(angles)
    assert np.allclose(sp, pry)


def test_plane_warping_pipeline_functions(data_2d):
    x, y, z = data_2d
    x, y = np.meshgrid(x, y)
    shape = x.shape
    R = coordinates.make_rotation_matrix((1, 2, 3))
    oy, ox = [(s-1)/2 for s in shape]
    y, x = [np.arange(s) for s in shape]
    y, x = np.meshgrid(y, x)
    Tin = coordinates.make_homomorphic_translation_matrix(-ox, -oy)
    Tout = coordinates.make_homomorphic_translation_matrix(ox, oy)
    R = coordinates.promote_3d_transformation_to_homography(R)
    Mfwd = Tout@(R@Tin)
    Mfwd = coordinates.drop_z_3d_transformation(Mfwd)
    Mifwd = np.linalg.inv(Mfwd)
    xfwd, yfwd = coordinates.apply_homography(Mifwd, x, y)
    zp = coordinates.warp(z, xfwd, yfwd)
    assert zp.any()
