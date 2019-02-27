"""Tests for MTF utils."""
import numpy as np

import pytest

from prysm import sample_files
from prysm import mtf_utils


@pytest.fixture
def sample_data():
    return mtf_utils.MTFvFvF.from_trioptics_file(sample_files('mtfvfvf'))


def test_can_load_MTFVFVF_file(sample_data):
    assert sample_data


def test_mtfvfvf_addition_works(sample_data):
    out = sample_data + sample_data
    twox = sample_data.data * 2
    assert np.allclose(twox, out.data)


def test_mtfvfvf_subtraction_works(sample_data):
    out = sample_data - sample_data
    zero = np.zeros(out.data.shape)
    assert np.allclose(zero, out.data)


def test_mtfvfvf_mul_works(sample_data):
    out = sample_data * 2
    twox = sample_data.data * 2
    assert np.allclose(twox, out.data)


def test_mtfvfvf_div_works(sample_data):
    out = sample_data / 2
    halfx = sample_data.data / 2
    assert np.allclose(halfx, out.data)


def test_mtfvfvf_iops_reverse(sample_data):
    dat = sample_data.data.copy()
    sample_data /= 2
    sample_data *= 2
    assert np.allclose(dat, sample_data.data)


@pytest.mark.parametrize('sym, contour', [[True, True], [False, False], [True, False], [False, True]])
def test_mtfvfvf_plot2d_functions(sample_data, sym, contour):
    fig, ax = sample_data.plot2d(30, symmetric=sym, contours=contour)
    assert fig
    assert ax


def test_mtfvfvf_plot_singlefield_throughfocus_functions(sample_data):
    fig, ax = sample_data.plot_thrufocus_singlefield(14)
    assert fig
    assert ax


@pytest.mark.parametrize('algo', ['0.5', 'avg'])
def mtfvfvf_trace_focus_functions(sample_data, algo):
    focuses, fields = sample_data.trace_focus(algo)
    assert focuses
    assert fields
