"""Tests for public wavelength constants."""

from prysm import wavelengths


def test_laser_wavelength_constants_are_microns():
    assert wavelengths.HeNe == 0.6328
    assert wavelengths.NdYAG == 1.064
    assert wavelengths.CO2 == 10.6


def test_wavelength_families_are_ordered_long_to_short():
    assert wavelengths.CO2 > wavelengths.NdYAP > wavelengths.NdYAG > wavelengths.InGaAs
    assert wavelengths.Ruby > wavelengths.HeNe > wavelengths.Cu
    assert wavelengths.XeF > wavelengths.XeCl > wavelengths.KrF > wavelengths.KrCl > wavelengths.ArF
