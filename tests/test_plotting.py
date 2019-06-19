"""Unit tests for plotting functions."""
import matplotlib as mpl

mpl.use('TkAgg')

from matplotlib import pyplot as plt  # NOQA

from prysm import util  # NOQA


def test_share_fig_ax_figure_number_remains_unchanged():
    fig, ax = plt.subplots()
    fig2, ax2 = util.share_fig_ax(fig, ax)
    assert fig.number == fig2.number


def test_share_fig_ax_produces_figure_and_axis():
    fig, ax = util.share_fig_ax()
    assert fig
    assert ax


def test_share_fig_ax_produces_an_axis():
    fig = plt.figure()
    fig, ax = util.share_fig_ax(fig)
    assert ax is not None
