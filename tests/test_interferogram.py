"""Interferogram tests."""
import pytest

import numpy as np

from prysm import sample_files
from prysm.interferogram import Interferogram

import matplotlib
matplotlib.use('TkAgg')


@pytest.fixture
def sample_i():
    i = Interferogram.from_zygo_dat(sample_files('dat'))
    return i.mask('circle', 40).crop().remove_piston_tiptilt()


@pytest.fixture
def sample_i_mutate():
    i = Interferogram.from_zygo_dat(sample_files('dat'))
    return i.mask('circle', 40).crop().remove_piston_tiptilt_power().fill()


def test_dropout_is_correct(sample_i):
    assert pytest.approx(sample_i.dropout_percentage, 21.67, abs=1e-2)


def test_pvr_is_correct(sample_i):
    assert pytest.approx(sample_i.pvr, 118.998, abs=1e-3)


def test_pv_is_correct(sample_i):
    assert pytest.approx(sample_i.pv, 96.8079, abs=1e-3)


def test_rms_is_correct(sample_i):
    assert pytest.approx(sample_i.rms, 17.736, abs=1e-3)


def test_std_is_correct(sample_i):
    assert pytest.approx(sample_i.std, 15.696, abs=1e-3)


def test_bandlimited_rms_is_correct(sample_i_mutate):
    assert pytest.approx(sample_i_mutate.bandlimited_rms(1, 10), 10.6, abs=1e-3)


def test_plot_psd_slices_functions(sample_i_mutate):
    fig, ax = sample_i_mutate.plot_psd_slices(x=True, y=True, azavg=True, azmin=True, azmax=True)
    assert fig
    assert ax


def test_plot_psd_2d_functions(sample_i_mutate):
    fig, ax = sample_i_mutate.plot_psd2d()
    assert fig
    assert ax
