"""Plotting helpers for optym convergence histories."""

from prysm.conf import config
from prysm.mathops import array_to_true_numpy, np
from prysm.plotting import share_fig_ax


_ALIASES = {
    'f': 'f',
    'cost': 'f',
    'objective': 'f',
    'g': 'g_norm',
    'gnorm': 'g_norm',
    'g_norm': 'g_norm',
    'gradient': 'g_norm',
    'gradient_norm': 'g_norm',
    'bounded': 'bounded',
    'bounds': 'bounded',
    'n_bounded': 'bounded',
    'bounded_variables': 'bounded',
}


def _records_from(result_or_records):
    if hasattr(result_or_records, 'records'):
        return list(result_or_records.records)
    if hasattr(result_or_records, 'history'):
        return list(result_or_records.history)
    return list(result_or_records)


def _as_tuple(quantities):
    if isinstance(quantities, str):
        quantities = (quantities,)
    return tuple(_normalize_quantity(quantity) for quantity in quantities)


def _normalize_quantity(quantity):
    try:
        return _ALIASES[quantity]
    except KeyError as exc:
        raise ValueError(f'unknown convergence quantity {quantity!r}') from exc


def _metadata(record):
    if isinstance(record, dict):
        return record
    return getattr(record, 'metadata', {})


def _record_value(record, *keys):
    if isinstance(record, dict):
        for key in keys:
            if key in record:
                return record[key]
        return None

    metadata = _metadata(record)
    for key in keys:
        if key in metadata:
            return metadata[key]
    for key in keys:
        if hasattr(record, key):
            return getattr(record, key)
    return None


def _iterations(records):
    iterations = []
    for idx, record in enumerate(records, start=1):
        if isinstance(record, dict):
            iterations.append(record.get('iteration', idx))
        else:
            iterations.append(getattr(record, 'iteration', idx))
    return np.asarray(iterations, dtype=float)


def _gradient_norm(record, norm):
    gnorm = _record_value(record, 'g_norm', 'gradient_norm')
    if gnorm is not None:
        return float(gnorm)

    g = _record_value(record, 'g', 'gradient')
    if g is None:
        raise ValueError('gradient norm requested, but no gradient data is available')
    g = np.asarray(g)
    if norm == np.inf or norm == 'inf':
        return float(np.max(np.abs(g)))
    return float(np.linalg.norm(g.ravel(), ord=norm))


def _active_bound_count(record, atol, rtol):
    direct = _record_value(
        record,
        'bounded_variables',
        'n_bounded',
        'active_bounds',
    )
    if direct is not None:
        direct = np.asarray(direct)
        if direct.ndim == 0:
            return float(direct)
        return float(np.sum(direct))

    active_inequalities = _record_value(record, 'active_inequalities')
    if active_inequalities is not None:
        return float(np.asarray(active_inequalities).size)

    if isinstance(record, dict):
        return None

    optimizer = getattr(record, 'optimizer', None)
    if optimizer is None or not hasattr(optimizer, 'l') or not hasattr(optimizer, 'u'):
        return None

    x = np.asarray(getattr(record, 'x_next', record.x))
    lower = np.asarray(optimizer.l)
    upper = np.asarray(optimizer.u)
    at_lower = np.isfinite(lower) & np.isclose(x, lower, rtol=rtol, atol=atol)
    at_upper = np.isfinite(upper) & np.isclose(x, upper, rtol=rtol, atol=atol)
    return float(np.sum(at_lower | at_upper))


def _series(records, quantity, gradient_norm, bounded_atol, bounded_rtol):
    out = []
    for record in records:
        if quantity == 'f':
            value = _record_value(record, 'f', 'cost', 'objective')
            if value is None:
                raise ValueError('f requested, but no objective data is available')
            out.append(float(value))
        elif quantity == 'g_norm':
            out.append(_gradient_norm(record, gradient_norm))
        elif quantity == 'bounded':
            value = _active_bound_count(record, bounded_atol, bounded_rtol)
            if value is None:
                raise ValueError('bounded requested, but no bound data is available')
            out.append(value)
    return np.asarray(out, dtype=float)


def _axis_list(ax):
    if isinstance(ax, (list, tuple)):
        return list(ax)
    if hasattr(ax, 'ravel'):
        return list(ax.ravel())
    return [ax]


def _label_for(quantity, gradient_norm):
    if quantity == 'f':
        return 'f'
    if quantity == 'g_norm':
        if gradient_norm == np.inf or gradient_norm == 'inf':
            return '||g|| inf'
        return f'||g|| {gradient_norm}'
    return 'bounded variables'


def plot_convergence(result_or_records, quantities=('f', 'g_norm'), *,
                     gradient_norm=np.inf, bounded_atol=1e-12,
                     bounded_rtol=1e-9, fig=None, ax=None, yscale='linear',
                     lw=None, marker=None, colors=None):
    """Plot optimizer convergence series against iteration.

    Parameters
    ----------
    result_or_records : OptimizationResult or sequence
        A run_until result, a sequence of StepRecord objects, or a sequence of
        history dictionaries.
    quantities : str or sequence of str, optional
        Series to plot.  Supported names are f, g_norm, and bounded.  Aliases
        include cost, objective, g, gnorm, gradient_norm, bounds, n_bounded,
        and bounded_variables.
    gradient_norm : float or str, optional
        Norm order used for g_norm.  The default is infinity norm.
    bounded_atol : float, optional
        Absolute tolerance for deciding whether a variable is on a box bound.
    bounded_rtol : float, optional
        Relative tolerance for deciding whether a variable is on a box bound.
    fig : matplotlib.figure.Figure, optional
        Figure containing the plot.
    ax : matplotlib.axes.Axis, optional
        Axis or array of axes containing the plot.
    yscale : str, optional
        Matplotlib y axis scale applied to all created axes.
    lw : float, optional
        Line width.
    marker : str, optional
        Matplotlib marker for each point.
    colors : sequence, optional
        Colors for each requested quantity.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the plot.
    matplotlib.axes.Axis or ndarray
        Axis object, or an array of axes when multiple quantities are plotted.

    """
    records = _records_from(result_or_records)
    if not records:
        raise ValueError('at least one convergence record is required')

    quantities = _as_tuple(quantities)
    if lw is None:
        lw = config.lw
    if colors is None:
        colors = (None,) * len(quantities)

    fig, ax = share_fig_ax(
        fig=fig,
        ax=ax,
        numax=len(quantities),
        sharex=True,
    )
    axes = _axis_list(ax)
    if len(axes) != len(quantities):
        raise ValueError('number of axes must match number of quantities')

    x = array_to_true_numpy(_iterations(records))
    for axis, quantity, color in zip(axes, quantities, colors):
        y = _series(
            records,
            quantity,
            gradient_norm,
            bounded_atol,
            bounded_rtol,
        )
        label = _label_for(quantity, gradient_norm)
        axis.plot(
            x,
            array_to_true_numpy(y),
            lw=lw,
            marker=marker,
            color=color,
            label=label,
        )
        axis.set_ylabel(label)
        axis.set_yscale(yscale)
        axis.grid(True, alpha=0.25)
        axis.legend()

    axes[-1].set_xlabel('iteration')
    return fig, ax
