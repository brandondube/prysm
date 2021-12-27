"""Plotting-related functions."""

from .conf import config


def share_fig_ax(fig=None, ax=None, numax=1, sharex=False, sharey=False):
    """Reurns the given figure and/or axis if given one.  If they are None, creates a new fig/ax.

    Parameters
    ----------
    fig : matplotlib.figure.Figure, optional
        figure
    ax : matplotlib.axes.Axis
        axis or array of axes
    numax : int
        number of axes in the desired figure, 1 for most plots, 3 for plot_fourier_chain
    sharex : bool, optional
        whether to share the x axis
    sharey : bool, optional
        whether to share the y axis

    Returns
    -------
    matplotlib.figure.Figure
        A figure object
    matplotlib.axes.Axis
        An axis object

    """
    from matplotlib import pyplot as plt

    if fig is None and ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=numax, sharex=sharex, sharey=sharey)
    elif ax is None:
        ax = fig.gca()

    return fig, ax


def add_psd_model(psd, fig=None, ax=None, invert_x=False,
                  lw=None, ls='--', color='k', alpha=1, zorder=None,
                  psd_fcn=None, **psd_fcn_kwargs):
    """Add a PSD model to a line plot.

    Parameters
    ----------
    psd : prysm.interferogram.PSD
        a PSD object
    fig : matplotlib.figure.Figure
        Figure containing the plot
    ax : matplotlib.axes.Axis
        Axis containing the plot
    invert_x : bool, optional
        if True, plot with x axis of spatial period
    lw : float, optional
        line width
    ls : str, optional
        line style
    color : str, optional
        something matplotlib understands as a color
    alpha : float, optional
        alpha (transparency) parameter for matplotlib
    zorder : int, optional
        z order (height in the stack)
    psd_fcn : callable, optional
        a callable function.  If None, inferred between ab_psd and abc_psd based on if c is in psd_fcn_kwargs
    **psd_fcn_kwargs
        keyword arguments arguments passed to psd_fcn after the spatial frequency variable

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the plot
    ax : matplotlib.axes.Axis
        Axis containing the plot

    """
    if lw is None:
        lw = config.lw * 1.5

    if zorder is None:
        zorder = config.zorder

    u = psd.slices().x[0]
    if invert_x:
        u2 = u.copy()
        u2[0] = u2[1] / 4
        u = 1 / u2

    fig, ax = share_fig_ax(fig, ax)

    if psd_fcn is None:
        from .interferogram import abc_psd, ab_psd
        if 'c' in psd_fcn_kwargs:
            psd_fcn = abc_psd
        else:
            psd_fcn = ab_psd

    line = psd_fcn(u2, **psd_fcn_kwargs)

    ax.plot(u, line, lw=lw, color=color, alpha=alpha, zorder=zorder)

    return fig, ax
