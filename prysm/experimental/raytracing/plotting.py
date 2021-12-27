"""Plotting functions for raytraces."""

from prysm.plotting import share_fig_ax

from .surfaces import STYPE_REFLECT

import numpy as np  # always numpy, matplotlib only understands numpy


def plot_rays(phist, lw=1, c='r', alpha=1, zorder=3, x='z', y='y', fig=None, ax=None):
    """Plot rays in 2D.

    Parameters
    ----------
    phist : list or numpy.ndarray
        the first return from spencer_and_murty.raytrace,
        iterable of arrays of length 3 (X,Y,Z)
    lw : float, optional
        linewidth
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
    ax.plot(x, y, c=c, lw=lw, alpha=alpha, zorder=zorder)
    return fig, ax


def plot_optics(prescription, phist, mirror_backing=None, points=100,
                lw=1, c='k', alpha=1, zorder=4,
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
        z = surf.P[2]
        if surf.typ == STYPE_REFLECT:
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

            sag = surf.F(xpt, ypt)
            sag += z
            sag[mask] = np.nan
            # TODO: mirror backing
            ax.plot(sag, ploty, c=c, lw=lw, alpha=alpha, zorder=zorder)

    return fig, ax


def plot_transverse_ray_aberration(phist, lw=1, c='r', alpha=1, zorder=3, axis='y', fig=None, ax=None):
    """Plot the transverse ray aberration for a single ray fan.

    Parameters
    ----------
    phist : list or numpy.ndarray
        the first return from spencer_and_murty.raytrace,
        iterable of arrays of length 3 (X,Y,Z)
    lw : float, optional
        linewidth
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
    ax.plot(x, y, c=c, lw=lw, alpha=alpha, zorder=zorder)
    return fig, ax
