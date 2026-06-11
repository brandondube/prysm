"""Interferogram tests."""
import pytest

import numpy as np

from prysm.sample_data import sample_files
from prysm.interferogram import Interferogram, make_random_subaperture_mask, make_window, fit_psd
from prysm.geometry import circle

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


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


def test_sa_is_correct(sample_i):
    assert 29.552 == pytest.approx(sample_i.Sa, abs=1e3)


def test_strehl_is_correct(sample_i):
    assert 0.938 == pytest.approx(sample_i.strehl, abs=1e3)

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
def test_make_window_known_geometries_are_center_weighted(win):
    signal = np.empty((11, 11))

    window = make_window(signal, 1, win)

    assert window[5, 5] == pytest.approx(1)
    assert window[0, 0] < window[5, 5]


def test_synthesize_from_psd_sets_requested_rms():
    i = Interferogram.render_from_psd(100, 64, rms=5, a=1e4, b=1/100, c=2)

    assert i.rms == pytest.approx(5, rel=0.15)


def test_pad_and_recenter_preserve_valid_data(sample_i_mutate):
    original_shape = sample_i_mutate.shape

    padded = sample_i_mutate.pad(np.nan, samples=5)
    recentered = padded.recenter()

    assert padded.shape[0] == original_shape[0] + 5
    assert recentered.shape == padded.shape
    assert np.isfinite(recentered.data).any()


def test_spike_clip_replaces_outlier(sample_i_mutate):
    sample_i_mutate.data[0, 0] = 1e9

    sample_i_mutate.spike_clip(3)

    assert sample_i_mutate.data[0, 0] != 1e9


def test_total_integrated_scatter_is_positive_for_filled_sample(sample_i_mutate):
    sample_i_mutate.fill()

    tis = sample_i_mutate.total_integrated_scatter(0.4, 0)

    assert tis > 0


def test_save_ascii_writes_nonempty_file(sample_i, tmpdir):
    path = tmpdir / 'z.asc'

    sample_i.save_zygo_ascii(path)

    assert path.size() > 0


def test_fit_psd_returns_positive_model_parameters(sample_i_mutate):
    with np.testing.suppress_warnings() as sup:
        sup.filter(RuntimeWarning)
        a, b, c = fit_psd(*sample_i_mutate.psd().slices().azavg)

    assert a > 0
    assert b > 0
    assert c > 0


def test_fit_psd_recovers_abc_parameters():
    from prysm.interferogram import abc_psd

    f = np.logspace(-2, 1, 200)
    true = (100.0, 0.5, 3.5)
    psd = abc_psd(f, *true)
    coefs = fit_psd(f, psd)
    np.testing.assert_allclose(coefs, true, rtol=1e-4)


def test_fit_psd_recovers_ab_parameters_closed_form():
    from prysm.interferogram import ab_psd

    f = np.logspace(-2, 1, 200)
    true = (10.0, 2.5)
    psd = ab_psd(f, *true)
    coefs = fit_psd(f, psd, callable=ab_psd)
    np.testing.assert_allclose(coefs, true, rtol=1e-10)
    # optres path reports the closed-form solve
    res = fit_psd(f, psd, callable=ab_psd, return_='optres')
    assert res.success
    np.testing.assert_allclose(res.x, true, rtol=1e-10)


def test_constructor_default_dx_and_meta_wavelength():
    i = Interferogram(np.zeros((2, 2)), meta={'Wavelength': 1.23})

    assert i.dx == 0
    assert i.meta['Wavelength'] == 1.23


def test_bandlimited_rms_works_with_frequency_specs(sample_i_mutate):
    assert sample_i_mutate.bandlimited_rms(flow=1, fhigh=10) > 0


def test_crop_mask_reduces_array_size():
    z = np.ones((32, 32))
    i = Interferogram(z, dx=1)

    i.mask(circle(5, i.r))
    i.crop()

    assert i.shape[0] < z.shape[0]
    assert i.shape[1] < z.shape[1]


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
def test_filter_reduces_or_preserves_shape(sample_i_mutate, fc, typ):
    shape = sample_i_mutate.shape

    sample_i_mutate.filter(fc, typ)

    assert sample_i_mutate.shape == shape
    assert np.isfinite(sample_i_mutate.data).all()


def test_interferogram_plot_labels_axes(sample_i_mutate):
    fig, ax = sample_i_mutate.interferogram()

    assert len(ax.images) == 1
    plt.close(fig)
