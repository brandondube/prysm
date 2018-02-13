'''  Unit tests for unit conversion factors
'''
import pytest

from prysm import units


@pytest.fixture(scope='function', params=[0.1, 1])
def test_wavelengths(request):
    return request.param


def test_wavelength_to_microns(test_wavelengths):
    assert units.waves_to_microns(test_wavelengths) == 1 / test_wavelengths


def test_wavelength_to_nanometers(test_wavelengths):
    assert units.waves_to_nanometers(test_wavelengths) == 1 / (test_wavelengths * 1e3)


def test_microns_to_waves(test_wavelengths):
    assert units.microns_to_waves(test_wavelengths) == test_wavelengths


def test_nanometers_to_waves(test_wavelengths):
    assert units.nanometers_to_waves(test_wavelengths) == test_wavelengths * 1e3
