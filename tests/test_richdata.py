"""Tests for rich data"""
import pytest

from prysm import _richdata as rdata

import numpy as np

from matplotlib import pyplot as plt


def test_general_properties_and_copy():
    data = np.random.rand(100, 100)
    rd = rdata.RichData(data, 1., 1.)
    assert rd.shape == rd.data.shape
    assert rd.size == rd.data.size
    assert rd.support == 100.
    cpy = rd.copy()
    # TODO: this relies on the unspecified behavior of CPython
    # ~= id == data ptr of variable.
    # since numpy is pretty heavily tied to CPython, this
    # is probably completely harmless.
    assert id(cpy.data) != id(rd.data)


def test_exact_functional():
    data = np.random.rand(100, 100)
    rd = rdata.RichData(data, 1., 1.)
    pt = rd.exact_x(3)
    assert np.isfinite(pt)
    pt = rd.exact_y(3)
    assert np.isfinite(pt)

    pt = rd.exact_xy(2, 2)
    assert np.isfinite(pt)
    if hasattr(pt, 'ndim'):  # backend agnosticism means we could get a scalar or an array
        assert pt.ndim == 0

    pt = rd.exact_polar(2, 0)
    assert np.isfinite(pt)
    if hasattr(pt, 'ndim'):  # backend agnosticism means we could get a scalar or an array
        assert pt.ndim == 0


def test_xyrt_synthesis_for_no_xytr_as_expected():
    data = np.random.rand(10, 10)
    dx = 1.234
    rd = rdata.RichData(data, dx, None)
    x, y = rd.x, rd.y
    r, t = rd.r, rd.t
    assert (x[0, 1] - x[0, 0]) == pytest.approx(dx, 0.001)
    assert y.shape == data.shape
    assert r.shape == data.shape
    assert t.shape == data.shape


def test_slices_does_not_alter_twosided():
    data = np.random.rand(11, 11)
    dx = 1.234
    rd = rdata.RichData(data, dx, None)
    slc = rd.slices(twosided=True)
    _, y = slc.y
    _, x = slc.x
    assert (y == data[:, 5]).all()
    assert (x == data[5, :]).all()


def test_slices_various_interped_profiles_function():
    data = np.random.rand(11, 11)
    dx = 1.234
    rd = rdata.RichData(data, dx, None)
    slc = rd.slices(twosided=True)
    u, azavg = slc.azavg
    assert np.isfinite(u).all()
    assert np.isfinite(azavg).all()

    u, azmin = slc.azmin
    assert np.isfinite(u).all()
    assert np.isfinite(azmin).all()

    u, azmax = slc.azmax
    assert np.isfinite(u).all()
    assert np.isfinite(azmax).all()

    u, azpv = slc.azpv
    assert np.isfinite(u).all()
    assert np.isfinite(azpv).all()

    u, azvar = slc.azvar
    assert np.isfinite(u).all()
    assert np.isfinite(azvar).all()

    u, azstd = slc.azstd
    assert np.isfinite(u).all()
    assert np.isfinite(azstd).all()


def test_plot2d_applies_limits_and_color_limits():
    data = np.arange(100, dtype=float).reshape(10, 10)
    rd = rdata.RichData(data, 0.5, 1.0)

    fig, ax = rd.plot2d(xlim=1, ylim=1, clim=(10, 90))

    assert ax.get_xlim() == pytest.approx((-1, 1))
    assert ax.get_ylim() == pytest.approx((-1, 1))
    assert ax.images[0].get_clim() == (10, 90)
    plt.close(fig)


def test_plot2d_log_uses_log_normalization():
    data = np.arange(1, 101, dtype=float).reshape(10, 10)
    rd = rdata.RichData(data, 1.0, 1.0)

    fig, ax = rd.plot2d(log=True)

    assert ax.images[0].norm.__class__.__name__ == 'LogNorm'
    plt.close(fig)


def test_slice_plot_selects_requested_slice_and_inverts_x():
    data = np.arange(121, dtype=float).reshape(11, 11)
    rd = rdata.RichData(data, 1.0, None)
    slc = rd.slices(twosided=True)

    fig, ax = slc.plot(slices='x', show_legend=True, invert_x=True)

    assert len(ax.lines) == 1
    assert ax.xaxis_inverted()
    assert ax.get_legend() is not None
    plt.close(fig)
