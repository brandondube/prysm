"""Interferogram tests."""
import pytest

import numpy as np

from prysm import sample_files
from prysm.interferogram import Interferogram, make_window, fit_psd

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


def test_spike_clip_functions(sample_i_mutate):
    sample_i_mutate.spike_clip(3)
    assert sample_i_mutate


def test_tis_functions(sample_i_mutate):
    sample_i_mutate.change_xy_unit('um')
    sample_i_mutate.fill()
    assert sample_i_mutate.total_integrated_scatter(0.4, 0)


def test_save_ascii_functions(sample_i, tmpdir):
    sample_i.save_zygo_ascii(tmpdir / 'z.asc')


def test_doublecrop_has_no_effect(sample_i_mutate):
    sample_i_mutate.crop()
    shape = sample_i_mutate.shape
    sample_i_mutate.crop()
    shape2 = sample_i_mutate.shape
    assert shape == shape2


def test_descale_latcal_ok(sample_i_mutate):
    plate_scale = sample_i_mutate.sample_spacing
    sample_i_mutate.strip_latcal()
    assert pytest.approx(sample_i_mutate.sample_spacing, 1, abs=1e-8)
    sample_i_mutate.latcal(plate_scale, 'mm')
    assert pytest.approx(plate_scale, sample_i_mutate.sample_spacing, abs=1e-8)


def test_make_window_passes_array():
    win = signal = np.empty((2, 2))
    win2 = make_window(signal, 1, win)
    assert (win == win2).all()


@pytest.mark.parametrize('win', ['welch', 'hanning'])
def test_make_window_functions_for_known_geometries(win):
    signal = np.empty((10, 10))
    window = make_window(signal, 1, win)
    assert window.any()


def test_synthesize_from_psd_functions():
    assert Interferogram.render_from_psd(100, 64, rms=5, a=1e4, b=1/100, c=2)


@pytest.mark.parametrize('freq, period', [
    [None, 10],
    [None, (25, 10)],
    [1, None],
    [(0.1, 1), None]
])
def test_filter_functions(sample_i_mutate, freq, period):
    sample_i_mutate.fill()
    sample_i_mutate.filter(freq, period)
    assert sample_i_mutate


def test_pad_functions(sample_i_mutate):
    assert sample_i_mutate.pad(5)


def test_recenter_functions(sample_i_mutate):
    assert sample_i_mutate.recenter()


def test_fit_psd(sample_i_mutate):
    a, b, c = fit_psd(*sample_i_mutate.psd().slices().azavg)
    assert a
    assert b
    assert c


def test_print_does_not_throw(sample_i):
    import contextlib
    import io

    s = io.StringIO()
    with contextlib.redirect_stdout(s):
        print(sample_i)

    assert sample_i
