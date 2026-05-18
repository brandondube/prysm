"""Tests for prysm.x.optym.plotting."""

import matplotlib
import numpy as np

matplotlib.use('Agg')

from matplotlib import pyplot as plt

from prysm.x.optym import (
    GradientDescent,
    MaxIterations,
    StepRecord,
    plot_convergence,
    run_until,
)


def quadratic_fg(x):
    f = float(0.5 * np.sum(x * x))
    return f, x.copy()


def test_plot_convergence_plots_f_and_gradient_norm():
    opt = GradientDescent(
        quadratic_fg, np.array([1.0, -2.0]), alpha=0.1,
    )
    result = run_until(opt, MaxIterations(3))

    fig, ax = plot_convergence(result, quantities=('f', 'g_norm'))
    axes = ax.ravel()

    np.testing.assert_allclose(
        axes[0].lines[0].get_ydata(),
        [2.5, 2.025, 1.64025],
    )
    np.testing.assert_allclose(
        axes[1].lines[0].get_ydata(),
        [2.0, 1.8, 1.62],
    )
    assert axes[0].get_ylabel() == 'f'
    assert axes[1].get_ylabel() == '||g|| inf'
    assert axes[1].get_xlabel() == 'iteration'
    plt.close(fig)


def test_plot_convergence_counts_active_box_bounds():
    class _BoundedOptimizer:
        l = np.array([0.0, -np.inf, 0.0])
        u = np.array([np.inf, 1.0, 2.0])

    opt = _BoundedOptimizer()
    record = StepRecord(
        optimizer=opt,
        iteration=1,
        x=np.array([1.0, 0.5, 1.0]),
        f=1.0,
        g=np.zeros(3),
        x_next=np.array([0.0, 1.0, 2.0]),
    )

    fig, ax = plot_convergence([record], quantities='bounded')
    np.testing.assert_array_equal(ax.lines[0].get_ydata(), [3.0])
    assert ax.get_ylabel() == 'bounded variables'
    plt.close(fig)


def test_plot_convergence_counts_active_inequalities_from_metadata():
    record = {
        'iteration': 2,
        'cost': 1.0,
        'active_inequalities': np.array([0, 3]),
    }

    fig, ax = plot_convergence([record], quantities='bounded')
    np.testing.assert_array_equal(ax.lines[0].get_xdata(), [2.0])
    np.testing.assert_array_equal(ax.lines[0].get_ydata(), [2.0])
    plt.close(fig)
