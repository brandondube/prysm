"""Tests for thinlens functions."""
import pytest

import numpy as np

from prysm import thinlens


@pytest.fixture(params=[1, 1.25, 2])
def inf_fno(request):
    return request.param


@pytest.mark.parametrize('objdist', [1.25, 2, 3, -1, -2])
def test_magnification_matches_thin_lens_formula(objdist):
    efl = 1

    result = thinlens.object_dist_to_mag(efl, objdist)

    assert result == pytest.approx(efl / (efl - objdist))


@pytest.mark.parametrize('mag', [0, 1, 2, 3, 3.05])
def test_lin_to_long_mag(mag):
    assert thinlens.linear_to_long_mag(mag) == mag ** 2


@pytest.mark.parametrize('mag', [0, 1, -2, 3.05])
def test_mag_to_fno_matches_working_f_number_formula(mag, inf_fno):
    pupil_mag = 0.5

    result = thinlens.mag_to_fno(mag, inf_fno, pupil_mag)

    assert result == pytest.approx((1 + abs(mag) / pupil_mag) * inf_fno)


@pytest.mark.parametrize('fno', [1, 1.4, 2, 2.8, 4, 5.6, 8, 11, 16, 22])
def test_fno_to_na_and_na_to_fno_invert(fno):
    na = thinlens.fno_to_na(fno)
    assert thinlens.na_to_fno(na) == pytest.approx(fno, rel=0.05, abs=0.01)


def test_object_to_image_distance_unity_case():
    efl = 1
    obj = -2
    assert thinlens.object_to_image_dist(efl, obj) == -obj


def test_object_image_to_efl_inverts_object_to_image_dist():
    efl = 50
    obj = np.array([-75, -100, -200])
    img = thinlens.object_to_image_dist(efl, obj)

    result = thinlens.object_image_to_efl(obj, img)

    assert np.allclose(result, efl)


def test_power_and_efl_invert():
    efl = np.array([50, 100, -200])
    n = 1.33

    power = thinlens.efl_to_power(efl, n)

    assert np.allclose(thinlens.power_to_efl(power, n), efl)


def test_efl_fno_epd_conversions():
    efl = -100
    epd = 25

    fno = thinlens.efl_to_fno(efl, epd)

    assert fno == 4
    assert thinlens.fno_to_efl(fno, epd) == abs(efl)
    assert thinlens.fno_to_epd(fno, efl) == epd


def test_image_distance_epd_to_fno_matches_na_conversion():
    image_distance = 10
    epd = 5

    fno = thinlens.image_dist_epd_to_fno(image_distance, epd)
    na = thinlens.image_dist_epd_to_na(image_distance, epd)

    assert fno == pytest.approx(thinlens.na_to_fno(na))


def test_image_displacement_to_defocus_all_cases():
    displacement = np.array([-50, 0, 5, 50])
    fno, wvl = 4, 0.55
    result_wvs = thinlens.image_displacement_to_defocus(displacement, fno, wvl)
    result_um = thinlens.image_displacement_to_defocus(displacement, fno)
    true_wvs = displacement / (8 * fno ** 2 * wvl)
    true_um = displacement / (8 * fno ** 2)
    assert np.allclose(result_wvs, true_wvs)
    assert np.allclose(result_um, true_um)


def test_defocus_to_image_displacement_all_cases():
    defocus = np.array([-2, 0.0005, 2])
    fno, wvl = 4, 0.55
    result_wvs = thinlens.defocus_to_image_displacement(defocus, fno, wvl)
    result_um = thinlens.defocus_to_image_displacement(defocus, fno)
    true_wvs = 8 * fno ** 2 * wvl * defocus
    true_um = 8 * fno ** 2 * defocus

    assert np.allclose(result_wvs, true_wvs)
    assert np.allclose(result_um, true_um)


@pytest.mark.parametrize('mag', [-2, -1, -0.5, 0.5, 2])
def test_mag_to_object_dist_inverts_object_dist_to_mag(mag):
    efl = 10

    object_dist = thinlens.mag_to_object_dist(efl, mag)

    assert thinlens.object_dist_to_mag(efl, object_dist) == pytest.approx(mag)


@pytest.mark.parametrize('mag', [-2, -1, -0.5, 0.5, 2])
def test_mag_to_image_dist_matches_thin_lens_conjugate(mag):
    efl = 10
    object_dist = thinlens.mag_to_object_dist(efl, mag)
    image_dist = thinlens.mag_to_image_dist(efl, mag)

    result = thinlens.object_to_image_dist(efl, -object_dist)

    assert image_dist == pytest.approx(result)


def test_twolens_efl_matches_in_contact():
    efl1, efl2 = 2.0, 2.0
    assert thinlens.twolens_efl(efl1, efl2, 0) == efl1 / 2


def test_twolens_bfl_matches_efl_in_contact():
    efl1, efl2 = 2.0, 2.0
    assert thinlens.twolens_bfl(efl1, efl2, 0) == efl1 / 2


def test_twolens_bfl_matches_first_order_formula():
    efl1, efl2, t = 100, 100, 20
    efl = thinlens.twolens_efl(efl1, efl2, t)

    bfl = thinlens.twolens_bfl(efl1, efl2, t)

    assert bfl == pytest.approx(efl * (1 - t / efl1))


def test_twolens_ffl_matches_first_order_formula():
    efl1, efl2, t = 100, 50, 10
    efl = thinlens.twolens_efl(efl1, efl2, t)

    ffl = thinlens.twolens_ffl(efl1, efl2, t)

    assert ffl == pytest.approx(-efl * (1 - t / efl2))


def test_twolens_power_and_separation_invert():
    efl1, efl2, t = 75, 125, 12
    efl = thinlens.twolens_efl(efl1, efl2, t)

    assert thinlens.twolens_power(efl1, efl2, t) == pytest.approx(1 / efl)
    assert thinlens.twolens_separation(efl1, efl2, efl) == pytest.approx(t)


def test_singlet_efl():
    R = 200
    c = 1/R
    efl = thinlens.singlet_efl(c, -c, 0, 1.55)
    assert efl == pytest.approx(181.8181818181818)


def test_singlet_power_matches_efl():
    R1, R2 = 100, -100
    c1, c2 = 1/R1, 1/R2
    t, n = 8, 1.5

    power = thinlens.singlet_power(c1, c2, t, n)
    efl = thinlens.singlet_efl(c1, c2, t, n)

    assert efl == pytest.approx(1 / power)


def test_singlet_efl_uses_ambient_index():
    R = 100
    n_ambient = 1.33
    c = 1/R

    result = thinlens.singlet_efl(c, -c, 0, 1.5, n_ambient)

    expected_power = 2 * (1.5 - n_ambient) * c
    assert result == pytest.approx(n_ambient / expected_power)


def test_singlet_bfl_and_ffl_match_abcd_formula():
    R1, R2 = 100, -50
    c1, c2 = 1/R1, 1/R2
    t, n = 8, 1.5
    phi1 = (n - 1) * c1
    phi2 = (1 - n) * c2
    efl = thinlens.singlet_efl(c1, c2, t, n)

    bfl = thinlens.singlet_bfl(c1, c2, t, n)
    ffl = thinlens.singlet_ffl(c1, c2, t, n)

    assert bfl == pytest.approx(efl * (1 - t/n * phi1))
    assert ffl == pytest.approx(-efl * (1 - t/n * phi2))
