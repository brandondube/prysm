"""Plotting functions for raytraces."""

from prysm.plotting import share_fig_ax

from .surfaces import STYPE_REFLECT, STYPE_REFRACT

import numpy as np  # always numpy, matplotlib only understands numpy


def plot_rays(phist, lw=1, ls='-', c='r', alpha=1, zorder=4, x='z', y='y', fig=None, ax=None):
    """Plot rays in 2D.

    Parameters
    ----------
    phist : list or numpy.ndarray
        the first return from spencer_and_murty.raytrace,
        iterable of arrays of length 3 (X,Y,Z)
    lw : float, optional
        linewidth
    ls : str, optional
        line style
    c : color
        anything matplotlib interprets as a color, strings, 3-tuples, 4-tuples, ...
    alpha : float
        opacity of the rays, 1=fully opaque, 0=fully transparent
    zorder : int
        stack order in the plot, higher z orders are on top of lower z orders
    x : str, {'x', 'y', 'z'}
        which position to plot on the X axis, defaults to traditional ZY plot
    y : str, {'x', 'y', 'z'}
        which position to plot on the X axis, defaults to traditional ZY plot
    fig : matplotlib.figure.Figure
        A figure object
    ax : matplotlib.axes.Axis
        An axis object

    Returns
    -------
    matplotlib.figure.Figure
        A figure object
    matplotlib.axes.Axis
        An axis object

    """
    fig, ax = share_fig_ax(fig, ax)

    ph = np.asarray(phist)
    xs = ph[..., 0]
    ys = ph[..., 1]
    zs = ph[..., 2]
    sieve = {
        'x': xs,
        'y': ys,
        'z': zs,
    }
    x = x.lower()
    y = y.lower()
    x = sieve[x]
    y = sieve[y]
    ax.plot(x, y, c=c, lw=lw, ls=ls, alpha=alpha, zorder=zorder)
    return fig, ax


def _gather_inputs_for_surface_sag(surf, phist, j, points, y):
    if surf.bounding is None:
        # need to look at the raytrace to see bounding limits
        p = phist[j+1]  # j+1, first element of phist is the start of the raytrace
        xx = p[..., 0]
        yy = p[..., 1]
        mask = []
        if y == 'y':
            ymin = yy.min()
            ymax = yy.max()
            ypt = np.linspace(ymin, ymax, points)
            ploty = ypt
            xpt = 0
        else:
            xmin = xx.min()
            xmax = xx.max()
            xpt = np.linspace(xmin, xmax, points)
            ploty = xpt
            ypt = 0
    else:
        bound = surf.bounding
        mx = bound['outer_radius']
        r = np.linspace(-mx, mx, points)
        mn = bound.get('inner_radius', 0)
        ar = abs(r)
        mask = ar < mn
        ploty = r
        if y == 'y':
            ypt = r
            xpt = 0
        else:
            xpt = r
            ypt = 0

    return xpt, ypt, mask, ploty


def plot_optics(prescription, phist, mirror_backing=None, points=100,
                lw=1, ls='-', c='k', alpha=1, zorder=3,
                x='z', y='y', fig=None, ax=None):
    """Draw the optics of a prescription.

    Parameters
    ----------
    prescription : iterable of Surface
        a prescription for an optical layout
    phist : iterable of numpy.ndarray
        the first return of spencer_and_murty.raytrace, the history of positions
        through a raytrace
    mirror_backing : TODO
        TODO
    points : int, optional
        the number of points used in making the curve for the surface
    lw : float, optional
        linewidth
    ls : str, optional
        line style
    c : color, optional
        anything matplotlib interprets as a color, strings, 3-tuples, 4-tuples, ...
    alpha : float, optional
        opacity of the rays, 1=fully opaque, 0=fully transparent
    zorder : int
        stack order in the plot, higher z orders are on top of lower z orders
    x : str, {'x', 'y', 'z'}
        which position to plot on the X axis, defaults to traditional ZY plot
    y : str, {'x', 'y', 'z'}
        which position to plot on the X axis, defaults to traditional ZY plot
    fig : matplotlib.figure.Figure
        A figure object
    ax : matplotlib.axes.Axis
        An axis object

    Returns
    -------
    matplotlib.figure.Figure
        A figure object
    matplotlib.axes.Axis
        An axis object

    """
    x = x.lower()
    y = y.lower()
    fig, ax = share_fig_ax(fig, ax)

    # manual iteration due to how lenses are drawn, start from -1 so the
    # increment can be at the top of a large loop
    j = -1
    jj = len(prescription)
    while True:
        j += 1
        if j == jj:
            break
        surf = prescription[j]
        if surf.typ == STYPE_REFLECT:
            z = surf.P[2]
            xpt, ypt, mask, ploty = _gather_inputs_for_surface_sag(surf, phist, j, points, y)
            sag = surf.F(xpt, ypt)
            sag += z
            sag[mask] = np.nan
            # TODO: mirror backing
            ax.plot(sag, ploty, c=c, lw=lw, ls=ls, alpha=alpha, zorder=zorder)
        elif surf.typ == STYPE_REFRACT:
            if (j + 1) == jj:
                raise ValueError('cant draw a prescription that terminates on a refracting surface')

            z = surf.P[2]
            xpt, ypt, mask, ploty = _gather_inputs_for_surface_sag(surf, phist, j, points, y)
            sag = surf.F(xpt, ypt)
            sag += z
            sag[mask] = np.nan

            # now get the points for the second surface of the lens
            j += 1
            surf = prescription[j]
            z = surf.P[2]
            xpt2, ypt2, mask2, ploty2 = _gather_inputs_for_surface_sag(surf, phist, j, points, y)
            sag2 = surf.F(xpt2, ypt2)
            sag2 += z
            sag2[mask2] = np.nan

            # now bundle the two surfaces together so one line is drawn for the
            # whole lens
            first_x = sag[0]
            first_y = ploty[0]
            # the ::-1 are because we need to reverse the order of the second
            # surface's points, so that matplotlib doesn't draw an X through the lens
            xx = [*sag, *sag2[::-1], first_x]
            yy = [*ploty, *ploty2[::-1], first_y]
            ax.plot(xx, yy, c=c, lw=lw, ls=ls, alpha=alpha, zorder=zorder)

    return fig, ax


def plot_transverse_ray_aberration(phist, lw=1, ls='-', c='r', alpha=1, zorder=4, axis='y', fig=None, ax=None):
    """Plot the transverse ray aberration for a single ray fan.

    Parameters
    ----------
    phist : list or numpy.ndarray
        the first return from spencer_and_murty.raytrace,
        iterable of arrays of length 3 (X,Y,Z)
    lw : float, optional
        linewidth
    ls : str, optional
        line style
    c : color
        anything matplotlib interprets as a color, strings, 3-tuples, 4-tuples, ...
    alpha : float
        opacity of the rays, 1=fully opaque, 0=fully transparent
    zorder : int
        stack order in the plot, higher z orders are on top of lower z orders
    axis : str, {'x', 'y'}
        which ray position to plot, x or y
    fig : matplotlib.figure.Figure
        A figure object
    ax : matplotlib.axes.Axis
        An axis object

    Returns
    -------
    matplotlib.figure.Figure
        A figure object
    matplotlib.axes.Axis
        An axis object

    """
    fig, ax = share_fig_ax(fig, ax)

    ph = np.asarray(phist)
    sieve = {
        'x': 0,
        'y': 1,
    }
    axis = axis.lower()
    axis = sieve[axis]
    input_rays = ph[0, ..., axis]
    output_rays = ph[-1, ..., axis]
    ax.plot(input_rays, output_rays, c=c, lw=lw, ls=ls, alpha=alpha, zorder=zorder)
    return fig, ax


def plot_wave_aberration(phist, lw=1, ls='-', c='r', alpha=1, zorder=4, axis='y', fig=None, ax=None):
    """Plot the transverse ray aberration for a single ray fan.

    Parameters
    ----------
    phist : list or numpy.ndarray
        the first return from spencer_and_murty.raytrace,
        iterable of arrays of length 3 (X,Y,Z)
    lw : float, optional
        linewidth
    ls : str, optional
        line style
    c : color
        anything matplotlib interprets as a color, strings, 3-tuples, 4-tuples, ...
    alpha : float
        opacity of the rays, 1=fully opaque, 0=fully transparent
    zorder : int
        stack order in the plot, higher z orders are on top of lower z orders
    axis : str, {'x', 'y'}
        which ray position to plot, x or y
    fig : matplotlib.figure.Figure
        A figure object
    ax : matplotlib.axes.Axis
        An axis object

    Returns
    -------
    matplotlib.figure.Figure
        A figure object
    matplotlib.axes.Axis
        An axis object

    """
    fig, ax = share_fig_ax(fig, ax)

    sieve = {
        'x': 0,
        'y': 1,
    }
    axis = axis.lower()
    axis = sieve[axis]
    input_rays = phist[0, ..., axis]
    output_rays = phist[-1, ..., axis]
    ax.plot(input_rays, output_rays, c=c, lw=lw, alpha=alpha, zorder=zorder)
    return fig, ax


def plot_spot_diagram(phist, marker='+', c='k', alpha=1, zorder=4, s=None, fig=None, ax=None):
    """Plot a spot diagram from a ray trace.

    Parameters
    ----------
    phist : list or numpy.ndarray
        the first return from spencer_and_murty.raytrace,
        iterable of arrays of length 3 (X,Y,Z)
    marker : str, optional
        marker style
    c : color
        anything matplotlib interprets as a color, strings, 3-tuples, 4-tuples, ...
    alpha : float
        opacity of the rays, 1=fully opaque, 0=fully transparent
    zorder : int
        stack order in the plot, higher z orders are on top of lower z orders
    s : float
        marker size or variable used for marker size
    axis : str, {'x', 'y'}
        which ray position to plot, x or y
    fig : matplotlib.figure.Figure
        A figure object
    ax : matplotlib.axes.Axis
        An axis object

    Returns
    -------
    matplotlib.figure.Figure
        A figure object
    matplotlib.axes.Axis
        An axis object

    """
    fig, ax = share_fig_ax(fig, ax)
    x = phist[-1, ..., 0]
    y = phist[-1, ..., 1]
    ax.scatter(x, y, c=c, s=s, marker=marker, alpha=alpha, zorder=zorder)
    return fig, ax
