"""Unit tests for plotting functions."""
import matplotlib as mpl

mpl.use('Agg')

from matplotlib import pyplot as plt  # NOQA

from prysm import plotting  # NOQA


def test_share_fig_ax_figure_number_remains_unchanged():
    fig, ax = plt.subplots()
    fig2, ax2 = plotting.share_fig_ax(fig, ax)
    assert fig.number == fig2.number
    assert ax2 is ax


def test_share_fig_ax_creates_requested_shared_axes():
    fig, axes = plotting.share_fig_ax(numax=3, sharex=True, sharey=True)

    assert len(axes) == 3
    assert axes[1].get_shared_x_axes().joined(axes[0], axes[1])
    assert axes[2].get_shared_y_axes().joined(axes[0], axes[2])


def test_share_fig_ax_produces_an_axis():
    fig = plt.figure()
    fig, ax = plotting.share_fig_ax(fig)
    assert ax is not None
