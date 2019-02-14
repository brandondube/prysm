''' Unit tests for the physics of prysm.
'''
import os
from itertools import product

import numpy as np

import pytest

from prysm import Pupil, PSF, MTF, FringeZernike
from prysm.psf import airydisk
from prysm.otf import diffraction_limited_mtf

PRECISION = 1e-3  # ~0.1%

TEST_PARAMETERS = [
    (10.0, 1.000, 0.5),  # f/10, visible light
    (10.0, 1.000, 1.0),  # f/10, SWIR light
    (3.00, 1.125, 3.0)]  # f/2.66666, MWIR light


@pytest.mark.parametrize('efl, epd, wvl', TEST_PARAMETERS)
def test_diffprop_matches_airydisk(efl, epd, wvl):
    fno = efl / epd

    p = Pupil(wavelength=wvl, dia=epd)
    psf = PSF.from_pupil(p, efl, Q=3)  # use Q=3 not Q=4 for improved accuracy
    u, sx = psf.slice_x
    u, sy = psf.slice_y
    analytic = airydisk(u, fno, wvl)
    assert np.allclose(sx, analytic, atol=PRECISION)
    assert np.allclose(sy, analytic, atol=PRECISION)


@pytest.mark.skipif('TRAVIS' in os.environ and os.environ['TRAVIS'] == 'true', reason='lapack error on travis')
@pytest.mark.parametrize('efl, epd, wvl', TEST_PARAMETERS)
def test_diffprop_matches_analyticmtf(efl, epd, wvl):
    fno = efl / epd
    p = Pupil(wavelength=wvl, dia=epd)
    psf = PSF.from_pupil(p, efl)
    mtf = MTF.from_psf(psf)
    u, t = mtf.tan
    uu, s = mtf.sag

    analytic_1 = diffraction_limited_mtf(fno, wvl, frequencies=u)
    analytic_2 = diffraction_limited_mtf(fno, wvl, frequencies=uu)
    assert np.allclose(analytic_1, t, atol=PRECISION)
    assert np.allclose(analytic_2, s, atol=PRECISION)


def test_array_orientation_consistency_tilt():
    ''' The pupil array should be shaped as arr[y,x], as should the psf and MTF.
        A linear phase error in the pupil along y should cause a motion of the
        PSF in y.  Specifically, for a positive-signed phase, that should cause
        a shift in the +y direction.
    '''
    samples = 128
    p = FringeZernike(Z2=1, base=1, samples=samples)
    ps = PSF.from_pupil(p, 1)
    idx_y, idx_x = np.unravel_index(ps.data.argmax(), ps.data.shape)  # row-major y, x
    assert idx_x == ps.center_x
    assert idx_y > ps.center_y


FNOS = [1, 1.4, 2, 2.8, 4, 5.6, 8]
WVLS = [.5, .55, 1, 10]


@pytest.mark.parametrize('fno, wvl', product(FNOS, WVLS))
def test_airydisk_has_unit_peak(fno, wvl):
    assert airydisk(0, fno=fno, wavelength=wvl) == 1
