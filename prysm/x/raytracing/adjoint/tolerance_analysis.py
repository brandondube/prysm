"""TOR-style tolerance analysis on the adjoint Jacobian.

multi_objective_sensitivity assembles the full M x P Jacobian (M merit
objectives, P tolerance parameters) from one forward-with-intermediates pass
and M backward sweeps.  The remaining helpers are linear algebra on that
Jacobian: sensitivity / degradation tables, inverse sensitivity, RSS prediction,
compensator projection, and multi-objective budget allocation.
"""

from prysm.conf import config
from prysm.mathops import np

from prysm.x.raytracing._diff_raytrace import _assemble_seeds

from .backward_sweep import (
    _forward_with_intermediates,
    _backward_sweep,
    _precompute_shape_partials,
)


def _head_has_value(head):
    """True when a head reports a usable trace-based value.

    A design.Merit exposes the has_value flag (its value attribute is always
    present, so a plain hasattr cannot tell the stub from an override); other
    duck-typed heads fall back to hasattr.
    """
    flag = getattr(head, 'has_value', None)
    if flag is not None:
        return bool(flag)
    return hasattr(head, 'value')


class AdjointResult:
    """The M x P adjoint Jacobian plus labels and nominal merit values."""

    __slots__ = ('jacobian', 'head_names', 'param_names', 'nominals')

    def __init__(self, jacobian, head_names, param_names, nominals):
        self.jacobian = jacobian
        self.head_names = list(head_names)
        self.param_names = list(param_names)
        self.nominals = dict(nominals)

    def _row(self, head):
        if isinstance(head, int):
            return head
        return self.head_names.index(head)

    def sensitivity_for(self, head):
        """The (P,) gradient row for a named (or indexed) objective."""
        return self.jacobian[self._row(head)]

    def ranked_by(self, head):
        """Parameters sorted by abs(sensitivity) for one objective, descending."""
        row = self.sensitivity_for(head)
        order = np.argsort(-np.abs(row))
        return [(self.param_names[i], float(row[i])) for i in order]

    def to_dataframe(self):
        import pandas as pd
        return pd.DataFrame(np.asarray(self.jacobian),
                            index=self.head_names, columns=self.param_names)

    def __repr__(self):
        return (f'AdjointResult(M={len(self.head_names)}, '
                f'P={len(self.param_names)})')


def multi_objective_sensitivity(surfaces, P, S, wvl, seeds, heads, *,
                                tol_sag=None):
    """Assemble the M x P adjoint Jacobian: one forward pass, M backward sweeps.

    Parameters
    ----------
    surfaces : sequence of Surface
    P, S : ndarray
        launch positions / directions.
    wvl : float
        wavelength, microns.
    seeds : sequence of DiffSeed
        the P tolerance parameters (defines the Jacobian column order).
    heads : sequence of seedable merits
        the M objectives (defines the row order); each exposes
        seed(trace, prescription, wavelength) and optionally
        value(trace, prescription, wavelength) (see design.Merit).
    tol_sag : float

    Returns
    -------
    AdjointResult

    """
    seeds = list(seeds)
    heads = list(heads)
    n_params = len(seeds)
    trace, inter = _forward_with_intermediates(
        surfaces, P, S, wvl, tol_sag=tol_sag)
    Qdot_s, Rdot_s, nprimedot_s, shape_params, sag_partial_fns = \
        _assemble_seeds(len(surfaces), seeds, n_params)
    # the shape-DOF tangents are cotangent-independent; evaluate them once and
    # share them across all M sweeps rather than per objective.
    shape_partials = _precompute_shape_partials(
        surfaces, inter, shape_params, sag_partial_fns)

    J = np.zeros((len(heads), n_params), dtype=config.precision)
    nominals = {}
    for m, head in enumerate(heads):
        cot = head.seed(trace, surfaces, wvl)
        grad = _backward_sweep(surfaces, trace, inter, Qdot_s, Rdot_s,
                               nprimedot_s, shape_params, sag_partial_fns,
                               cot, shape_partials=shape_partials)
        direct = getattr(head, 'direct_gradient', None)
        if direct is not None:
            extra = direct(trace, surfaces, wvl, seeds)
            if extra is not None:
                grad = grad + extra
        J[m] = grad
        if _head_has_value(head):
            nominals[head.name] = head.value(trace, surfaces, wvl)

    head_names = [getattr(h, 'name', f'head{m}') for m, h in enumerate(heads)]
    param_names = [getattr(s, 'name', f'param{p}') or f'param{p}'
                   for p, s in enumerate(seeds)]
    return AdjointResult(J, head_names, param_names, nominals)


class ToleranceSensitivityTable:
    """Per-parameter sensitivities and per-step degradations for the objectives.

    steps is the (P,) vector of tolerance step sizes (one per parameter, in each
    parameter's own units).  degradation_at_step gives the linear merit change
    each parameter contributes at its step.
    """

    __slots__ = ('result', 'steps')

    def __init__(self, adjoint_result, steps):
        self.result = adjoint_result
        self.steps = np.asarray(steps, dtype=config.precision)

    def sensitivity(self):
        """abs(dF_m / dtau_p) matrix, (M, P)."""
        return np.abs(self.result.jacobian)

    def degradation_at_step(self):
        """dF_m/dtau_p * step_p matrix, (M, P)."""
        return self.result.jacobian * self.steps[None, :]

    def ranked_by(self, head):
        return self.result.ranked_by(head)


def inverse_sensitivity(J, budget, steps_min=None, steps_max=None):
    """Per-parameter tolerance giving exactly `budget` merit degradation.

    For each parameter p, tol_p = budget / abs(J[m, p]) with the tightest (most
    constraining) objective m taken.  Parameters with zero sensitivity are
    unconstrained (tolerance -> steps_max if given, else +inf).  Result is
    clipped to [steps_min, steps_max] when provided.

    Parameters
    ----------
    J : ndarray, (M, P)
    budget : float or ndarray (M,)
        allowed degradation per objective (scalar broadcasts to all).
    steps_min, steps_max : ndarray (P,), optional

    Returns
    -------
    tol : ndarray, (P,)

    """
    J = np.asarray(J, dtype=config.precision)
    absJ = np.abs(J)
    budget = np.broadcast_to(np.asarray(budget, dtype=config.precision),
                             (J.shape[0],))
    with np.errstate(divide='ignore', invalid='ignore'):
        per_obj = budget[:, None] / absJ           # (M, P)
    per_obj = np.where(absJ > 0, per_obj, np.inf)
    tol = per_obj.min(axis=0)                       # tightest objective
    if steps_max is not None:
        tol = np.minimum(tol, np.asarray(steps_max, dtype=config.precision))
    if steps_min is not None:
        tol = np.maximum(tol, np.asarray(steps_min, dtype=config.precision))
    return tol


def rss_prediction(J, sigmas):
    """Root-sum-square merit perturbation for independent tolerances.

    sigma_total_m = sqrt(sum_p (J[m, p] sigma_p)^2).

    Parameters
    ----------
    J : ndarray, (M, P)
    sigmas : ndarray, (P,)

    Returns
    -------
    rss : ndarray, (M,)

    """
    J = np.asarray(J, dtype=config.precision)
    sigmas = np.asarray(sigmas, dtype=config.precision)
    contrib = J * sigmas[None, :]
    return np.sqrt(np.sum(contrib * contrib, axis=1))


def compensated_jacobian(J, J_comp):
    """Project compensator DOFs out of the Jacobian.

    With K compensators having Jacobian J_comp (M, K), the optimal compensator
    motion that minimizes the merit shift is c = -pinv(J_comp) @ (J tau), so the
    effective (post-compensation) tolerance Jacobian is

        J_eff = (I - J_comp pinv(J_comp)) J ,

    the projection of J onto the orthogonal complement of the compensators'
    column space.  comp_motions = -pinv(J_comp) @ J gives dc/dtau.

    Parameters
    ----------
    J : ndarray, (M, P)
    J_comp : ndarray, (M, K)

    Returns
    -------
    J_eff : ndarray, (M, P)
    comp_motions : ndarray, (K, P)

    """
    J = np.asarray(J, dtype=config.precision)
    J_comp = np.asarray(J_comp, dtype=config.precision)
    pinv = np.linalg.pinv(J_comp)                   # (K, M)
    comp_motions = -pinv @ J                        # (K, P)
    J_eff = J + J_comp @ comp_motions               # (I - Jc pinv) J
    return J_eff, comp_motions


def multi_objective_budget(J, budgets):
    """Per-parameter tolerance satisfying all objective budgets at once.

    Minimax over objectives: tol_p = min_m budgets[m] / abs(J[m, p]).  Equivalent
    to inverse_sensitivity with a per-objective budget vector.

    Parameters
    ----------
    J : ndarray, (M, P)
    budgets : ndarray, (M,)

    Returns
    -------
    tol : ndarray, (P,)

    """
    return inverse_sensitivity(J, budgets)


__all__ = [
    'AdjointResult',
    'multi_objective_sensitivity',
    'ToleranceSensitivityTable',
    'inverse_sensitivity',
    'rss_prediction',
    'compensated_jacobian',
    'multi_objective_budget',
]
