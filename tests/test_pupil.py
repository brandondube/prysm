''' Unit tests for pupil objects
'''
import pytest

import numpy as np

from prysm import Pupil, Seidel


@pytest.fixture
def p():
    return Pupil()


@pytest.fixture
def p_tlt():
    return Seidel(W111=1, samples=65)


def test_create_pupil():
    p = Pupil()
    assert hasattr(p, 'wavelength')
    assert hasattr(p, 'epd')
    assert hasattr(p, 'sample_spacing')
    assert hasattr(p, 'samples')
    assert hasattr(p, 'opd_unit')
    assert hasattr(p, '_opd_unit')
    assert hasattr(p, '_opd_str')
    assert hasattr(p, 'phase')
    assert hasattr(p, 'fcn')
    assert hasattr(p, 'unit')
    assert hasattr(p, 'rho')
    assert hasattr(p, 'phi')
    assert hasattr(p, 'center')


def test_pupil_passes_valid_params():
    parameters = {
        'samples': 16,
        'epd': 128.2,
        'wavelength': 0.6328,
        'opd_unit': 'nm'}
    p = Pupil(**parameters)
    assert p.samples == parameters['samples']
    assert p.epd == parameters['epd']
    assert p.wavelength == parameters['wavelength']
    assert p._opd_str == parameters['opd_unit']
    assert p._opd_unit == 'nanometers'  # make sure this is updated if the test is changed to a different unit


def test_pupil_rejects_bad_opd_unit():
    with pytest.raises(ValueError):
        Pupil(opd_unit='foo')


def test_pupil_has_zero_pv(p):
    assert p.pv == pytest.approx(0)


def test_pupil_has_zero_rms(p):
    assert p.rms == pytest.approx(0)


def test_tilt_pupil_axis_is_not_x(p_tlt):
    u, x = p_tlt.slice_x
    idxs = np.isfinite(x)
    zeros = np.zeros(x.shape)
    assert np.allclose(x[idxs], zeros[idxs])


def test_pupil_plot2d_functions(p):
    fig, ax = p.plot2d()
    assert fig
    assert ax


def test_pupil_interferogram_functions(p):
    fig, ax = p.interferogram()
    assert fig
    assert ax


def test_pupil_plot_slice_xy_functions(p):
    fig, ax = p.plot_slice_xy()
    assert fig
    assert ax
