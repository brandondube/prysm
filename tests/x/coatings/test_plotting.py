"""Smoke tests for the coatings plotting module.

These assert that each plotter runs end to end and returns a (figure, axis)
without error, using the non-interactive Agg backend.
"""
import numpy as np
import pytest

import matplotlib
matplotlib.use('Agg')

from prysm.x.coatings import Stack
from prysm.x.coatings import plotting as cp

WVL = 0.55
SUB = 1.52


@pytest.fixture
def stack():
    return Stack([1.38, 2.05, 1.38, 2.05], [0.10, 0.067, 0.10, 0.067], SUB)


def test_plot_spectrum(stack):
    wvls = np.linspace(0.45, 0.65, 50)
    fig, ax = cp.plot_spectrum(stack, wvls, quantities=('R', 'T', 'A'))
    assert ax.lines


def test_plot_spectrum_single_pol(stack):
    wvls = np.linspace(0.45, 0.65, 30)
    fig, ax = cp.plot_spectrum(stack, wvls, pol='s', quantities=('R',))
    assert ax.lines


def test_plot_index_profile(stack):
    fig, ax = cp.plot_index_profile(stack)
    assert ax.lines


def test_plot_field_intensity(stack):
    fig, ax = cp.plot_field_intensity(stack, WVL, pol='s')
    assert ax.lines


def test_plot_admittance(stack):
    fig, ax = cp.plot_admittance(stack, WVL, pol='s')
    assert ax.lines


def test_plot_monitoring_trace(stack):
    fig, ax = cp.plot_monitoring_trace(stack, 1, WVL, max_factor=2.0)
    assert ax.lines
