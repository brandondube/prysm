"""Tests for MTF utils."""
import random

import numpy as np

import pytest

import matplotlib as mpl

from prysm.sample_data import sample_files
from prysm import mtf_utils

mpl.use('Agg')


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
def test_mtfvfvf_trace_focus_functions(sample_data, algo):
    fields, focuses = sample_data.trace_focus(algo)
    assert any(focuses)
    assert any(fields)


def test_mtfvfvf_from_dataframe_correct_data_order():
    from itertools import product
    import pandas as pd
    fields = np.arange(21) - 11
    focus = np.arange(21) - 11
    focus *= 10  # +/- 100 microns
    freq = np.arange(21) * 10  # 0..10..210 cy/mm
    fff = product(fields, focus, freq)
    data = []
    for field, focus, freq in fff:
        data_base = {
            'Field': field,
            'Focus': focus,
            'Freq': freq,
        }
        data.append({
            **data_base,
            'Azimuth': 'Tan',
            'MTF': random.random(),
        })
        data.append({
            **data_base,
            'Azimuth': 'Sag',
            'MTF': random.random(),
        })
    df = pd.DataFrame(data=data)
    t, s = mtf_utils.MTFvFvF.from_dataframe(df)
    assert t.data.shape == (21, 21, 21)
    assert s.data.shape == (21, 21, 21)
