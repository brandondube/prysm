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
