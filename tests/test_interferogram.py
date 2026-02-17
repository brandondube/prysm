"""Interferogram tests."""
import pytest

import numpy as np

from prysm.sample_data import sample_files
from prysm.interferogram import Interferogram, make_random_subaperture_mask, make_window, fit_psd
from prysm.geometry import circle

import matplotlib
matplotlib.use('Agg')


@pytest.fixture(scope='function')
def sample_i():
    i = Interferogram.from_zygo_dat(sample_files('dat'))
    return i.mask(circle(40, i.r)).crop().remove_piston().remove_tiptilt()


@pytest.fixture(scope='function')
def sample_i_mutate():
    i = Interferogram.from_zygo_dat(sample_files('dat'))
    return i.mask(circle(40, i.r)).crop().remove_piston().remove_tiptilt().remove_power().fill()


def test_dropout_is_correct(sample_i):
    assert 25.73 == pytest.approx(sample_i.dropout_percentage, abs=1e-2)


def test_pv_is_correct(sample_i):
    assert 330.7 == pytest.approx(sample_i.pv, abs=1e-2)


def test_rms_is_correct(sample_i):
    assert 44.591 == pytest.approx(sample_i.rms, abs=1e-2)


def test_std_is_correct(sample_i):
    assert 44.591 == pytest.approx(sample_i.std, abs=1e-2)


def test_pvr_is_correct(sample_i):
    assert 299.814 == pytest.approx(sample_i.pvr(24), abs=1e-2)


def test_sa_is_correct(sample_i):
    assert 29.552 == pytest.approx(sample_i.Sa, abs=1e3)


def test_strehl_is_correct(sample_i):
    assert 0.938 == pytest.approx(sample_i.strehl, abs=1e3)


def test_bandlimited_rms_is_correct(sample_i_mutate):
    assert 11.527 == pytest.approx(sample_i_mutate.bandlimited_rms(1, 10), abs=1e-2)


def test_spike_clip_functions(sample_i_mutate):
    sample_i_mutate.spike_clip(3)
    assert sample_i_mutate


def test_tis_functions(sample_i_mutate):
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
    plate_scale = sample_i_mutate.dx
    sample_i_mutate.strip_latcal()
    assert 1 == pytest.approx(sample_i_mutate.dx, abs=1e-8)
    sample_i_mutate.latcal(plate_scale)
    assert plate_scale == pytest.approx(sample_i_mutate.dx, abs=1e-8)


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


def test_pad_functions(sample_i_mutate):
    assert sample_i_mutate.pad(np.nan, samples=5)


def test_recenter_functions(sample_i_mutate):
    assert sample_i_mutate.recenter()


def test_fit_psd(sample_i_mutate):
    with np.testing.suppress_warnings() as sup:
        sup.filter(RuntimeWarning)
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


def test_constructor_accepts_no_dx():
    z = np.random.rand(128, 128)
    i = Interferogram(z)
    assert i


def test_bandlimited_rms_works_with_frequency_specs(sample_i):
    assert sample_i.bandlimited_rms(flow=1, fhigh=10)


def test_can_make_with_meta_wavelength_dict():
    # this basically tests that getting the wavelength property
    # from a dat or datx file works
    meta = {'Wavelength': 1.}
    z = np.random.rand(2, 2)
    i = Interferogram(z, meta=meta)
    assert i


def test_crop_mask_works():
    z = np.random.rand(32, 32)
    i = Interferogram(z, dx=1)
    i.mask(circle(10, i.r))
    i.crop()
    assert i


def test_random_subaperture_mask_works():
    mask = np.zeros((10, 10), dtype=bool)
    mask[5, 5] = 1
    shp = (100, 100)
    out = make_random_subaperture_mask(shp, mask)
    assert out.sum() == 1


@pytest.mark.parametrize('fc, typ', [
    (0.5, 'lp'),
    (0.5, 'hp'),
    ((0.1, 0.2), 'bp'),
    ((0.1, 0.2), 'br')
])
def test_filter_functions(sample_i_mutate, fc, typ):
    sample_i_mutate.filter(fc, typ)
    assert sample_i_mutate


def test_interferogram_functions(sample_i_mutate):
    fig, ax = sample_i_mutate.interferogram()
    assert fig
    assert ax
