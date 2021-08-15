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
    resampled = coordinates.resample_2d(dat, (x, y), (x, y))
    assert np.allclose(dat, resampled)


def test_resample_2d_complex_does_not_distort(data_2d_complex):
    x, y, dat = data_2d_complex
    resampled = coordinates.resample_2d_complex(dat, (x, y), (x, y))
    assert np.allclose(dat, resampled)


def test_make_rotation_matrix_matches_scipy():
    from scipy.spatial.transform import Rotation as R

    angles = (0, 30, 0)
    sp = R.from_euler('ZXZ', angles, degrees=True).as_matrix()
    pry = coordinates.make_rotation_matrix(angles)
    assert np.allclose(sp, pry)


def test_plane_warping_pipeline_functions(data_2d):
    x, y, z = data_2d
    x, y = np.meshgrid(x, y)
    m = coordinates.make_rotation_matrix((0, 30, 0))
    x2, y2 = coordinates.apply_rotation_matrix(m, x, y)
    regular = coordinates.regularize([x, y], [x2, y2], z)
    assert regular.any()
