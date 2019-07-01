"""Tests for pupil objects."""
import pytest

import numpy as np

from prysm import Pupil, FringeZernike


@pytest.fixture
def p():
    return Pupil()


@pytest.fixture
def p_tlt():
    return FringeZernike(Z2=1, base=1, samples=64)


def test_create_pupil():
    p = Pupil()
    assert hasattr(p, 'wavelength')
    assert hasattr(p, 'diameter')
    assert hasattr(p, 'sample_spacing')
    assert hasattr(p, 'samples')
    assert hasattr(p, 'zunit')
    assert hasattr(p, 'xyunit')
    assert hasattr(p, 'phase')
    assert hasattr(p, 'fcn')
    assert hasattr(p, 'x')
    assert hasattr(p, 'y')
    assert hasattr(p, 'rho')
    assert hasattr(p, 'phi')
    assert hasattr(p, 'center_x')
    assert hasattr(p, 'center_y')


def test_pupil_passes_valid_params():
    parameters = {
        'samples': 16,
        'dia': 128.2,
        'wavelength': 0.6328,
        'opd_unit': 'nm'}
    p = Pupil(**parameters)
    assert p.samples == parameters['samples']
    assert p.dia == parameters['dia']
    assert p.wavelength == parameters['wavelength']
    assert p.zunit == 'nm'


def test_pupil_rejects_bad_opd_unit():
    with pytest.raises(ValueError):
        Pupil(opd_unit='foo')


def test_pupil_has_zero_pv(p):
    assert p.pv == pytest.approx(0)


def test_pupil_has_zero_rms(p):
    assert p.rms == pytest.approx(0)


def test_tilt_pupil_axis_is_x(p_tlt):
    u, x = p_tlt.slices().x
    x = x[1:-1]
    zeros = np.zeros(x.shape)
    assert np.allclose(x, zeros, atol=1e-1)


def test_pupil_plot2d_functions(p):
    fig, ax = p.plot2d()
    assert fig
    assert ax


def test_pupil_interferogram_functions(p):
    fig, ax = p.interferogram()
    assert fig
    assert ax


def test_pupil_add_functions(p):
    assert p + p


def test_pupil_sub_functions(p):
    assert p - p
