"""Unit tests for the physics of prysm."""
from itertools import product

import numpy as np

import pytest

from prysm.coordinates import make_xy_grid, cart_to_polar
from prysm.geometry import circle
from prysm.propagation import Wavefront
from prysm.psf import airydisk
from prysm.otf import diffraction_limited_mtf, mtf_from_psf

PRECISION = 1e-3  # ~0.1%

TEST_PARAMETERS = [
    (10.0, 1.000, 0.5),  # f/10, visible light
    (10.0, 1.000, 1.0),  # f/10, SWIR light
    (3.00, 1.125, 3.0)]  # f/2.66666, MWIR light


@pytest.mark.parametrize('efl, epd, wvl', TEST_PARAMETERS)
def test_diffprop_matches_airydisk(efl, epd, wvl):
    fno = efl / epd
    x, y = make_xy_grid(128, diameter=epd)
    r, t = cart_to_polar(x, y)
    amp = circle(epd/2, r)
    wf = Wavefront.from_amp_and_phase(amp.astype(float), None, wvl, x[0, 1] - x[0, 0]).pad2d(Q=3)
    wf.data *= 3*np.sqrt(amp.size)/amp.sum()
    psf = wf.focus(efl, Q=1)
    s = psf.intensity.slices()
    u_, sx = s.x
    u_, sy = s.y
    analytic = airydisk(u_, fno, wvl)
    assert np.allclose(sx, analytic, atol=PRECISION)
    assert np.allclose(sy, analytic, atol=PRECISION)


@pytest.mark.parametrize('efl, epd, wvl', TEST_PARAMETERS)
def test_diffprop_matches_analyticmtf(efl, epd, wvl):
    fno = efl / epd
    x, y = make_xy_grid(128, diameter=epd)
    r, t = cart_to_polar(x, y)
    amp = circle(epd/2, r)
    wf = Wavefront.from_amp_and_phase(amp, None, wvl, x[0, 1] - x[0, 0])
    psf = wf.focus(efl, Q=3).intensity
    mtf = mtf_from_psf(psf.data, psf.dx)
    s = mtf.slices()
    u_, sx = s.x
    u_, sy = s.y

    analytic_1 = diffraction_limited_mtf(fno, wvl, frequencies=u_)
    analytic_2 = diffraction_limited_mtf(fno, wvl, frequencies=u_)
    assert np.allclose(analytic_1, sx, atol=PRECISION)
    assert np.allclose(analytic_2, sy, atol=PRECISION)


def test_array_orientation_consistency_tilt():
    """The pupil array should be shaped as arr[y,x], as should the psf and MTF.

        A linear phase error in the pupil along y should cause a motion of the
        PSF in y.  Specifically, for a positive-signed phase, that should cause
        a shift in the +y direction.
    """
    N = 128
    wvl = .5
    Q = 3
    x, y = make_xy_grid(N, diameter=2.1)
    r, t = cart_to_polar(x, y)
    amp = circle(1, r)
    wf = Wavefront.from_amp_and_phase(amp, None, wvl, x[0, 1] - x[0, 0])
    psf = wf.focus(1, Q=Q).intensity
    idx_y, idx_x = np.unravel_index(psf.data.argmax(), psf.data.shape)  # row-major y, x
    assert idx_x == (N*Q) // 2
    assert idx_y > N // 2


FNOS = [1, 1.4, 2, 2.8, 4, 5.6, 8]
WVLS = [.5, .55, 1, 10]


@pytest.mark.parametrize('fno, wvl', product(FNOS, WVLS))
def test_airydisk_has_unit_peak(fno, wvl):
    assert airydisk(0, fno=fno, wavelength=wvl) == 1
