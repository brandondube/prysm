"""Optical-monitoring simulation for coating deposition."""

from functools import reduce

from prysm.conf import config
from prysm.mathops import np
from prysm.thinfilm import _cos_snell

from .stack import (
    Stack, _resolve, _admittance, _char_matrix, _eye2,
    stack_characteristic_matrices,
)


def _signal_curve(below_indices, below_thicknesses, grow_index, d_grid,
                  monitor_wvl, theta, pol, mode, n0, nsub):
    """Monitor signal of a growing layer over deposited thickness."""
    pol = pol.lower()
    if below_indices:
        sub = Stack(below_indices, below_thicknesses, nsub, n0)
        mats = stack_characteristic_matrices(sub, monitor_wvl, theta, pol)
        P_below = reduce(np.matmul, mats, _eye2())
    else:
        P_below = _eye2()

    n_k = _resolve(grow_index, monitor_wvl)
    cost_k = _cos_snell(n0, n_k, theta)
    eta_k = _admittance(n_k, cost_k, pol)
    dbeta = (2 * np.pi * n_k * cost_k) / monitor_wvl
    beta = dbeta * np.asarray(d_grid)
    Mk = _char_matrix(beta, np.broadcast_to(eta_k + 0j, beta.shape))
    A = Mk @ P_below[None]

    cost0 = np.cos(theta)
    cost_sub = _cos_snell(n0, nsub, theta)
    eta0 = _admittance(n0, cost0, pol)
    eta_sub = _admittance(nsub, cost_sub, pol)
    B = A[..., 0, 0] + A[..., 0, 1] * eta_sub
    C = A[..., 1, 0] + A[..., 1, 1] * eta_sub
    den = eta0 * B + C
    if mode == 'R':
        r = (eta0 * B - C) / den
        return np.abs(r) ** 2
    t = 2 * eta0 / den
    return np.real(eta_sub) / np.real(eta0) * np.abs(t) ** 2


def monitoring_trace(stack, layer, monitor_wvl, *, theta=0.0, pol='s', mode='R',
                     n_points=400, max_factor=1.0):
    """Monitor signal versus deposited thickness while growing one layer.

    Parameters
    ----------
    stack : Stack
        the target design.
    layer : int
        index (ambient side first) of the layer being deposited.
    monitor_wvl : float
        monitor wavelength, microns.
    theta : float, optional
        monitor angle of incidence, radians.
    pol : str, optional
        polarization, 's' or 'p'.
    mode : str, optional
        monitored quantity, 'R' or 'T'.
    n_points : int, optional
        samples along the deposited-thickness axis.
    max_factor : float, optional
        multiple of the target thickness spanned by the grid.

    Returns
    -------
    (ndarray, ndarray)
        deposited thickness grid and the monitor signal along it.

    """
    th = np.asarray(stack.thicknesses, dtype=config.precision)
    n0 = _resolve(stack.ambient_index, monitor_wvl)
    nsub = _resolve(stack.substrate_index, monitor_wvl)
    d_target = float(th[layer])
    d_grid = np.linspace(0.0, max_factor * d_target, n_points)
    sig = _signal_curve(list(stack.indices[layer + 1:]), th[layer + 1:],
                        stack.indices[layer], d_grid, monitor_wvl, theta, pol,
                        mode, n0, nsub)
    return d_grid, sig


def turning_points(d, signal):
    """Deposited thicknesses at the turning points (extrema) of a monitor trace.

    Parameters
    ----------
    d : ndarray
        deposited-thickness grid.
    signal : ndarray
        monitor signal along d.

    Returns
    -------
    ndarray
        the d values at which the signal has a local extremum.

    """
    d = np.asarray(d)
    s = np.asarray(signal)
    slope = np.sign(np.diff(s))
    idx = np.where(slope[:-1] != slope[1:])[0] + 1
    return d[idx]


def level_cut(d, signal, level, target=None):
    """Deposited thickness where the monitor signal crosses a level.

    Parameters
    ----------
    d : ndarray
        deposited-thickness grid.
    signal : ndarray
        monitor signal along d.
    level : float
        the termination level.
    target : float, optional
        when several crossings exist, return the one nearest this thickness;
        otherwise the first crossing.

    Returns
    -------
    float
        the deposited thickness at termination.

    """
    d = np.asarray(d)
    s = np.asarray(signal) - level
    sign = np.sign(s)
    idx = np.where(sign[:-1] != sign[1:])[0]
    if idx.size == 0:
        return float(d[np.argmin(np.abs(s))])
    crossings = []
    for i in idx:
        s0, s1 = s[i], s[i + 1]
        frac = 0.0 if s1 == s0 else -s0 / (s1 - s0)
        crossings.append(d[i] + frac * (d[i + 1] - d[i]))
    crossings = np.asarray(crossings)
    if target is None:
        return float(crossings[0])
    return float(crossings[np.argmin(np.abs(crossings - target))])


def cutoff_levels(stack, monitor_wvl, *, theta=0.0, pol='s', mode='R',
                  n_points=400):
    """Nominal monitor level at the end of each layer's deposition.

    For level-cut monitoring this is the signal value the monitor should read
    when each layer reaches its target thickness, given nominal layers below.
    """
    th = np.asarray(stack.thicknesses, dtype=config.precision)
    n0 = _resolve(stack.ambient_index, monitor_wvl)
    nsub = _resolve(stack.substrate_index, monitor_wvl)
    levels = []
    for k in range(len(stack)):
        sig = _signal_curve(list(stack.indices[k + 1:]), th[k + 1:],
                            stack.indices[k], np.array([th[k]]), monitor_wvl,
                            theta, pol, mode, n0, nsub)
        levels.append(float(sig[0]))
    return np.asarray(levels, dtype=config.precision)


def simulate_run(stack, monitor_wvl, *, strategy='level', turning_index=1,
                 signal_errors=None, thickness_errors=None, theta=0.0, pol='s',
                 mode='R', n_points=600, max_factor=1.8, levels=None):
    """Simulate a monitored deposition run.

    Parameters
    ----------
    stack : Stack
        target design.
    monitor_wvl : float
        monitor wavelength, microns.
    strategy : str, optional
        'level' or 'turning'.
    turning_index : int, optional
        which extremum to stop at for the turning strategy.
    signal_errors : ndarray, optional
        per-layer monitor-level error.
    thickness_errors : ndarray, optional
        per-layer thickness error.
    theta, pol, mode : optional
        monitor geometry / quantity.
    n_points, max_factor : optional
        monitor-trace sampling (see monitoring_trace).
    levels : ndarray, optional
        precomputed nominal cutoff levels.

    Returns
    -------
    Stack
        the as-built stack (same indices / substrate, realized thicknesses).

    """
    N = len(stack)
    th_nom = np.asarray(stack.thicknesses, dtype=config.precision)
    n0 = _resolve(stack.ambient_index, monitor_wvl)
    nsub = _resolve(stack.substrate_index, monitor_wvl)
    asbuilt = th_nom.copy()

    if strategy == 'level' and levels is None:
        levels = cutoff_levels(stack, monitor_wvl, theta=theta, pol=pol,
                               mode=mode, n_points=n_points)

    for k in range(N - 1, -1, -1):
        d_grid = np.linspace(1e-12, max_factor * th_nom[k], n_points)
        sig = _signal_curve(list(stack.indices[k + 1:]), asbuilt[k + 1:],
                            stack.indices[k], d_grid, monitor_wvl, theta, pol,
                            mode, n0, nsub)
        if strategy == 'turning':
            tps = turning_points(d_grid, sig)
            if tps.size >= turning_index:
                d_real = float(tps[turning_index - 1])
            else:
                d_real = float(th_nom[k])
            if thickness_errors is not None:
                d_real = d_real + float(thickness_errors[k])
        elif strategy == 'level':
            L = float(levels[k])
            if signal_errors is not None:
                L = L + float(signal_errors[k])
            d_real = level_cut(d_grid, sig, L, target=float(th_nom[k]))
        else:
            raise ValueError("strategy must be 'level' or 'turning'")
        asbuilt[k] = max(d_real, 0.0)

    return Stack(stack.indices, asbuilt, stack.substrate_index,
                 stack.ambient_index)


def monitoring_error_sensitivity(stack, monitor_wvl, design_wvls, *,
                                 strategy='level', theta=0.0, pol='s',
                                 design_pol='s', mode='R', eps=1e-4, **kwargs):
    """Jacobian of realized reflectance w.r.t. termination error.

    Parameters
    ----------
    stack : Stack
    monitor_wvl : float
    design_wvls : ndarray
        wavelengths at which the realized reflectance is evaluated.
    strategy : str, optional
    theta, pol, mode : optional
        monitor geometry.
    design_pol : str, optional
        polarization for the realized-reflectance readout.
    eps : float, optional
        finite-difference step in the termination-error units.

    Returns
    -------
    ndarray
        (len(design_wvls), N) sensitivity matrix.

    """
    from .stack import RTA
    N = len(stack)
    design_wvls = np.atleast_1d(np.asarray(design_wvls, dtype=config.precision))
    base = simulate_run(stack, monitor_wvl, strategy=strategy, theta=theta,
                        pol=pol, mode=mode, **kwargs)
    R0, _, _ = RTA(base, design_wvls, theta, design_pol)
    R0 = np.atleast_1d(R0)

    J = np.zeros((design_wvls.size, N), dtype=config.precision)
    for k in range(N):
        err = np.zeros(N, dtype=config.precision)
        err[k] = eps
        if strategy == 'turning':
            run = simulate_run(stack, monitor_wvl, strategy=strategy,
                               thickness_errors=err, theta=theta, pol=pol,
                               mode=mode, **kwargs)
        else:
            run = simulate_run(stack, monitor_wvl, strategy=strategy,
                               signal_errors=err, theta=theta, pol=pol,
                               mode=mode, **kwargs)
        Rk, _, _ = RTA(run, design_wvls, theta, design_pol)
        J[:, k] = (np.atleast_1d(Rk) - R0) / eps
    return J


def choose_monitor_wavelength(stack, candidates, design_wvls, *,
                             strategy='level', **kwargs):
    """Pick the monitor wavelength with lowest error sensitivity.

    Parameters
    ----------
    stack : Stack
    candidates : sequence of float
        candidate monitor wavelengths, microns.
    design_wvls : ndarray
        wavelengths at which realized reflectance is judged.
    strategy : str, optional
    kwargs
        forwarded to monitoring_error_sensitivity.

    Returns
    -------
    (float, ndarray)
        the best monitor wavelength and the per-candidate score array.

    """
    scores = []
    for wm in candidates:
        J = monitoring_error_sensitivity(stack, wm, design_wvls,
                                         strategy=strategy, **kwargs)
        scores.append(float(np.sqrt(np.sum(J * J))))
    scores = np.asarray(scores, dtype=config.precision)
    best = float(np.asarray(candidates)[int(np.argmin(scores))])
    return best, scores


__all__ = [
    'monitoring_trace',
    'turning_points',
    'level_cut',
    'cutoff_levels',
    'simulate_run',
    'monitoring_error_sensitivity',
    'choose_monitor_wavelength',
]
