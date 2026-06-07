"""Plotting for coating designs."""

from prysm.plotting import share_fig_ax
from prysm.mathops import array_to_true_numpy

from .stack import RTA, field_at_depth, internal_fields
from .monitoring import monitoring_trace

import numpy as np  # matplotlib expects host numpy arrays


def _to_np(x):
    return np.asarray(array_to_true_numpy(x))


def _boundary_depths(stack):
    th = _to_np(stack.thicknesses)
    return np.concatenate([[0.0], np.cumsum(th)])


def _rta_pol(stack, wvls, theta, pol):
    """R, T, total-A versus wvls for a polarization ('s', 'p', or 'avg')."""
    if pol == 'avg':
        Rs, Ts, As = _rta_pol(stack, wvls, theta, 's')
        Rp, Tp, Ap = _rta_pol(stack, wvls, theta, 'p')
        return 0.5 * (Rs + Rp), 0.5 * (Ts + Tp), 0.5 * (As + Ap)
    R, T, A = RTA(stack, wvls, theta, pol)
    R = _to_np(R)
    T = _to_np(T)
    A_total = 1.0 - R - T
    return R, T, A_total


def plot_spectrum(stack, wvls, theta=0.0, pol='avg', quantities=('R', 'T'),
                  fig=None, ax=None):
    """Plot reflectance / transmittance / absorptance versus wavelength.

    Parameters
    ----------
    stack : Stack
    wvls : ndarray
        wavelengths to evaluate, microns.
    theta : float, optional
        angle of incidence, radians.
    pol : str, optional
        's', 'p', or 'avg' (unpolarized mean).
    quantities : sequence of str, optional
        any of 'R', 'T', 'A' to draw.
    fig, ax : optional
        matplotlib figure / axis.

    Returns
    -------
    (figure, axis)

    """
    wvls = _to_np(wvls)
    R, T, A = _rta_pol(stack, wvls, theta, pol)
    series = {'R': R, 'T': T, 'A': A}
    labels = {'R': 'reflectance', 'T': 'transmittance', 'A': 'absorptance'}
    fig, ax = share_fig_ax(fig, ax)
    for q in quantities:
        ax.plot(wvls, series[q], label=labels[q])
    ax.set_xlabel('wavelength [um]')
    ax.set_ylabel('fraction of incident power')
    ax.legend()
    return fig, ax


def plot_index_profile(stack, wvl=None, fig=None, ax=None):
    """Step plot of refractive index versus depth through the stack.

    Parameters
    ----------
    stack : Stack
    wvl : float, optional
        wavelength to resolve dispersive indices at (real part is drawn).
    fig, ax : optional

    Returns
    -------
    (figure, axis)

    """
    Z = _boundary_depths(stack)
    if wvl is None:
        ns = [n if not callable(n) else n(0.55) for n in stack.indices]
    else:
        ns = stack.resolved_indices(wvl)
    ns = _to_np([np.real(n) for n in ns])
    fig, ax = share_fig_ax(fig, ax)
    # draw each homogeneous layer as a flat segment
    for k in range(len(stack)):
        ax.plot([Z[k], Z[k + 1]], [ns[k], ns[k]], c='C0')
        if k > 0:
            ax.plot([Z[k], Z[k]], [ns[k - 1], ns[k]], c='C0', lw=0.75)
    ax.set_xlabel('depth [um]')
    ax.set_ylabel('refractive index')
    return fig, ax


def plot_field_intensity(stack, wvl, theta=0.0, pol='s', n_points=1000,
                         fig=None, ax=None):
    """Plot the standing-wave intensity |E(z)|^2 through the stack.

    Parameters
    ----------
    stack : Stack
    wvl : float
        wavelength, microns.
    theta : float, optional
        angle of incidence, radians.
    pol : str, optional
        's' or 'p'.
    n_points : int, optional
        depth samples.
    fig, ax : optional

    Returns
    -------
    (figure, axis)

    """
    Z = _boundary_depths(stack)
    z = np.linspace(0.0, float(Z[-1]), n_points)
    E, _ = field_at_depth(stack, z, wvl, theta, pol)
    intensity = _to_np(np.abs(E) ** 2)
    fig, ax = share_fig_ax(fig, ax)
    ax.plot(z, intensity, c='C3')
    for zb in Z[1:-1]:
        ax.axvline(zb, c='k', lw=0.5, alpha=0.3)
    ax.set_xlabel('depth [um]')
    ax.set_ylabel('|E|^2 (incident = 1)')
    return fig, ax


def plot_admittance(stack, wvl, theta=0.0, pol='s', n_points=2000,
                    fig=None, ax=None):
    """Plot the admittance diagram, the H/E locus through the stack.

    Parameters
    ----------
    stack : Stack
    wvl : float
        wavelength, microns.
    theta : float, optional
        angle of incidence, radians.
    pol : str, optional
        's' or 'p'.
    n_points : int, optional
        depth samples along the locus.
    fig, ax : optional

    Returns
    -------
    (figure, axis)

    """
    Z = _boundary_depths(stack)
    z = np.linspace(0.0, float(Z[-1]), n_points)
    E, H = field_at_depth(stack, z, wvl, theta, pol)
    Y = _to_np(H / E)
    fig, ax = share_fig_ax(fig, ax)
    ax.plot(np.real(Y), np.imag(Y), c='C2')
    # mark the boundary admittances
    Eb, Hb = internal_fields(stack, wvl, theta, pol)
    Yb = _to_np(Hb / Eb)
    ax.scatter(np.real(Yb), np.imag(Yb), c='k', s=12, zorder=4)
    ax.set_xlabel('Re(Y)  (admittance)')
    ax.set_ylabel('Im(Y)')
    ax.set_aspect('equal', adjustable='datalim')
    return fig, ax


def plot_monitoring_trace(stack, layer, monitor_wvl, theta=0.0, pol='s',
                          mode='R', n_points=400, max_factor=1.0,
                          fig=None, ax=None):
    """Plot the in-situ monitoring signal while one layer is deposited.

    Parameters
    ----------
    stack : Stack
    layer : int
        index (ambient side first) of the layer being deposited.
    monitor_wvl : float
        monitor wavelength, microns.
    theta, pol, mode : optional
        monitor geometry / quantity (see monitoring.monitoring_trace).
    n_points, max_factor : optional
        deposited-thickness sampling.
    fig, ax : optional

    Returns
    -------
    (figure, axis)

    """
    d, sig = monitoring_trace(stack, layer, monitor_wvl, theta=theta, pol=pol,
                              mode=mode, n_points=n_points, max_factor=max_factor)
    d = _to_np(d)
    sig = _to_np(sig)
    fig, ax = share_fig_ax(fig, ax)
    ax.plot(d, sig, c='C4')
    ax.set_xlabel('deposited thickness [um]')
    ax.set_ylabel(f'monitor signal ({mode})')
    return fig, ax


__all__ = [
    'plot_spectrum',
    'plot_index_profile',
    'plot_field_intensity',
    'plot_admittance',
    'plot_monitoring_trace',
]
