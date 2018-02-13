''' Unit tests for thinlens functions.
'''
import pytest

import numpy as np

from prysm import thinlens


@pytest.fixture(params=[1, 1.25, 2])
def inf_fno(request):
    return request.param


def test_magnification_unity_case():
    efl = 1
    objdist = 2
    assert thinlens.object_dist_to_mag(efl, objdist) == -1


@pytest.mark.parametrize('objdist', [1.01, 3, -1, -2])
def test_magnification_general(objdist):
    efl = 1
    result = thinlens.object_dist_to_mag(efl, objdist)
    assert result != np.nan
    assert result != np.inf


@pytest.mark.parametrize('mag', [0, 1, 2, 3, 3.05])
def test_lin_to_long_mag(mag):
    assert thinlens.linear_to_long_mag(mag) == mag ** 2


def test_mag_to_fno_inf_case(inf_fno):
    m = 0
    assert thinlens.mag_to_fno(m, inf_fno) == inf_fno


@pytest.mark.parametrize('mag', [0, 1, 2, 3, 1.05, -1, -2, -3, -1.05])
def test_mag_to_fno_general(mag, inf_fno):  # TODO: not purely functional
    assert thinlens.mag_to_fno(mag, inf_fno)


def test_mag_to_fno_reacts_to_pupil_mag():
    m, inf_fno = 1.17, 10.85
    assert thinlens.mag_to_fno(m, inf_fno, 1) != thinlens.mag_to_fno(m, inf_fno, 0.5)


@pytest.mark.parametrize('fno', [1, 1.4, 2, 2.8, 4, 5.6, 8, 11, 16, 22])
def test_fno_to_na_and_na_to_fno_invert(fno):
    na = thinlens.fno_to_na(fno)
    assert thinlens.na_to_fno(na) == pytest.approx(fno, rel=0.05, abs=0.01)


def test_object_to_image_distance_unity_case():
    efl = 1
    obj = -2
    assert thinlens.object_to_image_dist(efl, obj) == -obj


def test_imagedist_epd_to_fno():  # purely functional test
    assert thinlens.image_dist_epd_to_fno(10, 50)


def test_image_displacement_to_defocus_all_cases():
    displacement = [-50, 5, 50]
    fno, wvl = 4, 0.55
    result_nonzern = thinlens.image_displacement_to_defocus(displacement, fno, wvl, zernike=False)
    result_zern = thinlens.image_displacement_to_defocus(displacement, fno, wvl, zernike=True)
    result_zern_rms = thinlens.image_displacement_to_defocus(displacement, fno, wvl, zernike=True, rms_norm=True)

    assert result_nonzern.all()
    assert ~np.allclose(result_nonzern, result_zern)
    assert ~np.allclose(result_zern, result_zern_rms)
    # TODO: assertion that rms_norm applies correct scale factor


def test_defocus_to_image_displacement_all_cases():
    defocus = [-2, 0.0005, 2]
    fno, wvl = 4, 0.55
    result_nonzern = thinlens.defocus_to_image_displacement(defocus, fno, wvl, zernike=False)
    result_zern = thinlens.defocus_to_image_displacement(defocus, fno, wvl, zernike=True)
    result_zern_rms = thinlens.defocus_to_image_displacement(defocus, fno, wvl, zernike=True, rms_norm=True)

    assert result_nonzern.all()
    assert ~np.allclose(result_nonzern, result_zern)
    assert ~np.allclose(result_zern, result_zern_rms)
    # TODO: assertion that rms_norm applies correct scale factor


def test_twolens_efl_matches_in_contact():
    efl1, efl2 = 2.0, 2.0
    assert thinlens.twolens_efl(efl1, efl2, 0) == efl1 / 2


def test_twolens_bfl_matches_efl_in_contact():
    efl1, efl2 = 2.0, 2.0
    assert thinlens.twolens_bfl(efl1, efl2, 0) == efl1 / 2


@pytest.fixture(params=[[1, 1, 0], [-1, 1, 1], [1, 1, 50]])
def twolens_params(request):
    return request.param


def test_twolens_efl_general(twolens_params):
    efl1, efl2, t = twolens_params
    assert thinlens.twolens_efl(efl1, efl2, t)


def test_twolens_bfl_general(twolens_params):
    efl1, efl2, t = twolens_params
    assert thinlens.twolens_bfl(efl1, efl2, t)
