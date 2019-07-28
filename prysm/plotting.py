"""Plotting-related functions."""


def share_fig_ax(fig=None, ax=None, numax=1, sharex=False, sharey=False):
    """Reurns the given figure and/or axis if given one.  If they are None, creates a new fig/ax.

    Parameters
    ----------
    fig : `matplotlib.figure.Figure`, optional
        figure
    ax : `matplotlib.axes.Axis`
        axis or array of axes
    numax : `int`
        number of axes in the desired figure, 1 for most plots, 3 for plot_fourier_chain
    sharex : `bool`, optional
        whether to share the x axis
    sharey : `bool`, optional
        whether to share the y axis

    Returns
    -------
    `matplotlib.figure.Figure`
        A figure object
    `matplotlib.axes.Axis`
        An axis object

    """
    from matplotlib import pyplot as plt

    if fig is None and ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=numax, sharex=sharex, sharey=sharey)
    elif ax is None:
        ax = fig.gca()

    return fig, ax
