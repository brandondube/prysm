"""Tests for convolution code."""
import numpy as np

import pytest

from prysm import PixelAperture, Pupil, PSF

import matplotlib
matplotlib.use('TkAgg')


@pytest.fixture
def sample_psf():
    p = Pupil()
    return PSF.from_pupil(p, 10)


@pytest.fixture
def sample_psf_bigger():
    p = Pupil()
    return PSF.from_pupil(p, 20)


@pytest.fixture
def sample_pixel():
    return PixelAperture(5)


@pytest.fixture
def sample_pixel_gridded():
    return PixelAperture(5, sample_spacing=0.1, samples_x=256)


def test_double_analyical_convolution_functions(sample_pixel, sample_pixel_gridded):
    assert sample_pixel.conv(sample_pixel_gridded)


def test_single_analytical_convolution_functions(sample_pixel, sample_psf):
    assert sample_pixel.conv(sample_psf)


def test_numerical_convolution_equal_functions(sample_psf):
    assert sample_psf.conv(sample_psf)


def test_numerical_convolution_unequal_functions(sample_psf, sample_psf_bigger):
    assert sample_psf.conv(sample_psf_bigger)


def test_show_functions(sample_psf):
    fig, ax = sample_psf.show(show_colorbar=True)
    assert fig
    assert ax


def test_show_fourier_functions_numerical(sample_psf):
    fig, ax = sample_psf.show_fourier()
    assert fig
    assert ax


def test_show_fourier_functions_analytic(sample_pixel):
    x = y = np.linspace(-10, 10, 10)
    fig, ax = sample_pixel.show_fourier(freq_x=x, freq_y=y)
    assert fig
    assert ax


def test_show_fourier_raises_analytic_with_no_xy(sample_pixel):
    with pytest.raises(ValueError):
        sample_pixel.show_fourier()


def test_sensical_attributes_dataless_convolvable(sample_pixel):
        assert sample_pixel.shape == (0, 0)
        assert sample_pixel.size == 0
        assert sample_pixel.samples_x == 0
        assert sample_pixel.samples_y == 0
