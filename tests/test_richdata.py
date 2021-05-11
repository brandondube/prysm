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


def test_plot2d_all_none():
    data = np.random.rand(100, 100)
    rd = rdata.RichData(data, 1., 1.)
    fig, ax = rd.plot2d()
    assert fig
    plt.close(fig)


def test_plot2d_given_xlim():
    data = np.random.rand(100, 100)
    rd = rdata.RichData(data, 1., 1.)
    fig, ax = rd.plot2d(xlim=1)
    assert fig
    plt.close(fig)


def test_plot2d_given_ylim():
    data = np.random.rand(100, 100)
    rd = rdata.RichData(data, 1., 1.)
    fig, ax = rd.plot2d(ylim=1)
    assert fig
    plt.close(fig)


def test_plot2d_given_clim():
    data = np.random.rand(100, 100)
    rd = rdata.RichData(data, 1., 1.)
    fig, ax = rd.plot2d(clim=10)
    assert fig
    plt.close(fig)


def test_plot2d_given_power():
    data = np.random.rand(100, 100)
    rd = rdata.RichData(data, 1., 1.)
    fig, ax = rd.plot2d(power=1/4)
    assert fig
    plt.close(fig)


def test_plot2d_log():
    data = np.random.rand(100, 100)
    rd = rdata.RichData(data, 1., 1.)
    fig, ax = rd.plot2d(log=True)
    assert fig
    plt.close(fig)


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
    assert (y == data[:, 6]).all()
    assert (x == data[6, :]).all()


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


def test_slice_plot_all_flavors():
    data = np.random.rand(11, 11)
    dx = 1.234
    rd = rdata.RichData(data, dx, None)
    slc = rd.slices(twosided=True)
    fig, ax = slc.plot(alpha=None, lw=None, zorder=None, slices='x', show_legend=True, invert_x=True)
    assert fig
    assert ax
    plt.close(fig)
