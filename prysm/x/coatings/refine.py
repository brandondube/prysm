"""Analytic-gradient refinement of coating stacks."""

from prysm.conf import config
from prysm.mathops import np

from prysm.x.optym.optimizers import PrysmLBFGSB, run_until
from prysm.x.optym.least_squares import damped_least_squares
from prysm.x.optym.governors import (
    AnyGovernor,
    MaxIterations,
    FunctionTolerance,
    GradientTolerance,
)

from .merit import as_merit
from .problem import CoatingProblem


class CoatingResult:
    """Outcome of a coating refinement.

    Attributes
    ----------
    stack : Stack
        the optimized stack.
    x : ndarray
        the final design vector (variable thicknesses or indices).
    merit : float
        the merit value at the solution.
    success : bool
        whether the optimizer reported convergence.
    nit : int
        number of iterations taken.
    optimizer_result : object
        the raw result object returned by the underlying optym optimizer.

    """

    __slots__ = ('stack', 'x', 'merit', 'success', 'nit', 'optimizer_result')

    def __init__(self, stack, x, merit, success, nit, optimizer_result):
        self.stack = stack
        self.x = x
        self.merit = float(merit)
        self.success = bool(success)
        self.nit = int(nit)
        self.optimizer_result = optimizer_result

    def __repr__(self):
        return (f'CoatingResult(merit={self.merit:.3e}, '
                f'success={self.success}, nit={self.nit})')


def _box_bounds(n, bounds, min_thickness, max_thickness):
    if bounds is not None:
        lo, hi = bounds
        lb = np.full(n, lo, dtype=config.precision)
        ub = np.full(n, hi, dtype=config.precision)
    else:
        lb = np.full(n, min_thickness, dtype=config.precision)
        ub = (np.full(n, np.inf, dtype=config.precision)
              if max_thickness is None
              else np.full(n, max_thickness, dtype=config.precision))
    return lb, ub


def _as_constraint_list(constraints):
    if constraints is None:
        return []
    if callable(constraints):
        return [constraints]
    return list(constraints)


def _box_inequality_constraints(lb, ub):
    constraints = []
    if bool(np.any(np.isfinite(lb))):
        constraints.append(lambda x, lb=lb: np.asarray(x) - lb)
    if bool(np.any(np.isfinite(ub))):
        constraints.append(lambda x, ub=ub: ub - np.asarray(x))
    return constraints


def refine(stack, targets, *, method='lbfgsb', variable_layers=None,
           variables='thickness', bounds=None,
           min_thickness=0.0, max_thickness=None, maxiter=200,
           ftol=1e-12, gtol=1e-10, memory=10, **kwargs):
    """Refine a stack against a target.

    Parameters
    ----------
    stack : Stack
        the starting design.
    targets : MeritFunction, merit term, or sequence of terms
        the objective (normalized with merit.as_merit).
    method : {'lbfgsb', 'lm'}, optional
        bounded quasi-Newton ('lbfgsb') or damped least squares ('lm').
    variable_layers : sequence of int, optional
        which layers' design variable is free; default all.
    variables : {'thickness', 'index'}, optional
        optimize layer thickness or layer index.
    bounds : (float, float), optional
        explicit (lower, upper) box bounds on the design variable.
    min_thickness, max_thickness : float, optional
        box bounds on the thicknesses; default [0, inf).
    maxiter : int, optional
        iteration cap.
    ftol, gtol : float, optional
        function- and gradient-tolerance stop conditions (lbfgsb).
    memory : int, optional
        L-BFGS history length.
    kwargs
        forwarded to the underlying optimizer (damped_least_squares for 'lm').

    Returns
    -------
    CoatingResult

    """
    merit = as_merit(targets)
    problem = CoatingProblem(stack, merit, variable_layers=variable_layers,
                             variables=variables)
    x0 = problem.x0()
    n = x0.size
    lb, ub = _box_bounds(n, bounds, min_thickness, max_thickness)

    if method == 'lbfgsb':
        opt = PrysmLBFGSB(problem.fg, x0, memory=memory,
                          lower_bounds=lb, upper_bounds=ub, **kwargs)
        governor = AnyGovernor([
            MaxIterations(maxiter),
            FunctionTolerance(ftol),
            GradientTolerance(gtol),
        ])
        result = run_until(opt, governor, maxiter=maxiter)
        x = result.x
        success = result.success
        nit = result.nit
    elif method == 'lm':
        user_ineq = kwargs.pop('inequality_constraints', None)
        ineq = _as_constraint_list(user_ineq)
        ineq.extend(_box_inequality_constraints(lb, ub))
        result = damped_least_squares(
            problem, x0=x0, maxiter=maxiter,
            inequality_constraints=ineq or None, **kwargs)
        x = result.x
        success = result.success
        nit = result.nit
    else:
        raise ValueError("method must be 'lbfgsb' or 'lm'")

    final_stack = problem.stack_from_x(x)
    return CoatingResult(final_stack, x, merit.value(final_stack), success, nit,
                         result)


__all__ = ['refine', 'CoatingResult']
