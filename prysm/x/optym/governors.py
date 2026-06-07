"""Composable stop conditions for optym optimizers."""

from prysm.mathops import np


# StepRecord aliases x/g inputs.  Optimizers that reuse mutable buffers must
# snapshot before constructing records.


class StepRecord:
    """Observation of one completed optimizer step.

    Parameters
    ----------
    optimizer : Any
        Optimizer that produced the step.
    iteration : int
        One-based iteration count for the completed step.
    x : array_like
        Iterate at which the objective and gradient were evaluated.
    f : float
        Objective value at `x`.
    g : array_like
        Gradient at `x`.
    x_next : array_like
        Iterate after the optimizer step.
    metadata : Mapping, optional
        Additional optimizer-specific information for the completed step.

    """

    __slots__ = (
        'optimizer',
        'iteration',
        'x',
        'f',
        'g',
        'x_next',
        'metadata',
    )

    def __init__(self, optimizer, iteration, x, f, g, x_next,
                 metadata=None):
        self.optimizer = optimizer
        self.iteration = int(iteration)
        self.x = x
        self.f = float(f)
        self.g = g
        self.x_next = x_next
        self.metadata = {} if metadata is None else dict(metadata)


class GovernorDecision:
    """Decision returned by a governor.

    Parameters
    ----------
    stop : bool, optional
        If True, the optimizer runner should stop.
    success : bool, optional
        If True, the stop condition represents successful convergence.
    message : str, optional
        Human-readable reason for the decision.

    """

    __slots__ = ('stop', 'success', 'message')

    def __init__(self, stop=False, success=False, message=''):
        self.stop = bool(stop)
        self.success = bool(success)
        self.message = message

    def __bool__(self):
        """Return the stop flag."""
        return self.stop


class OptimizationResult:
    """Result from a governed optimizer run.

    Parameters
    ----------
    x : array_like
        Final optimizer iterate.
    decision : GovernorDecision
        Terminal decision that ended the run.
    records : Sequence of StepRecord
        Step records observed during the run.
    optimizer : Any, optional
        Optimizer used for the run.

    """

    __slots__ = (
        'x',
        'success',
        'message',
        'nit',
        'nfev',
        'njev',
        'decision',
        'records',
        'optimizer',
    )

    def __init__(self, x, decision, records, optimizer=None):
        self.x = x
        self.success = bool(decision.success)
        self.message = decision.message
        self.nit = len(records)
        self.nfev = getattr(optimizer, 'nfev', None)
        self.njev = getattr(optimizer, 'njev', None)
        self.decision = decision
        self.records = records
        self.optimizer = optimizer

    def __repr__(self):
        """Return a compact representation."""
        return (
            f'OptimizationResult(success={self.success}, '
            f'message={self.message!r}, nit={self.nit})'
        )


class Governor:
    """Base class for reusable optimizer stop conditions."""

    def observe(self, record):
        """Observe a step record."""
        return GovernorDecision(False, False, '')


class AnyGovernor(Governor):
    """Stop when any child governor stops.

    Parameters
    ----------
    governors : Iterable of Governor
        Child governors to observe each step.

    """

    def __init__(self, governors):
        self.governors = tuple(governors)

    def observe(self, record):
        """Return the first stopping child decision."""
        decisions = [governor.observe(record) for governor in self.governors]
        for decision in decisions:
            if decision.stop:
                return decision
        return GovernorDecision(False, False, '')


class AllGovernor(Governor):
    """Stop after every child governor has stopped at least once.

    Parameters
    ----------
    governors : Iterable of Governor
        Child governors to observe each step.

    """

    def __init__(self, governors):
        self.governors = tuple(governors)
        self._decisions = [None] * len(self.governors)

    def observe(self, record):
        """Stop once all child governors have stopped at least once."""
        for idx, governor in enumerate(self.governors):
            decision = governor.observe(record)
            if decision.stop:
                self._decisions[idx] = decision

        if self._decisions and all(dec is not None for dec in self._decisions):
            success = all(dec.success for dec in self._decisions)  # NOQA - guarded in the if; can't be none
            message = '; '.join(
                dec.message for dec in self._decisions if dec.message  # NOQA - guarded in the if; can't be none
            )
            return GovernorDecision(True, success, message)
        return GovernorDecision(False, False, '')


def _validate_nonnegative(value, name):
    """Validate that a scalar is nonnegative."""
    if value < 0:
        raise ValueError(f'{name} must be nonnegative')


def _vector_norm(x, norm):
    """Compute a vector norm as a Python float."""
    x = np.asarray(x)
    if x.size == 0:
        return 0.0
    if norm == np.inf or norm == 'inf':
        return float(np.max(np.abs(x)))
    return float(np.linalg.norm(x.ravel(), ord=norm))


class MaxIterations(Governor):
    """Stop after a fixed number of accepted optimizer steps.

    Parameters
    ----------
    n : int
        Maximum number of completed optimizer steps.

    """

    def __init__(self, n):
        n = int(n)
        _validate_nonnegative(n, 'n')
        self.n = n

    def observe(self, record):
        """Stop when record.iteration reaches n."""
        if record.iteration >= self.n:
            return GovernorDecision(
                True, False, 'maximum iterations reached',
            )
        return GovernorDecision(False, False, '')


class MaxEvaluations(Governor):
    """Stop when optimizer.nfev reaches a fixed limit.

    Parameters
    ----------
    n : int
        Maximum number of function evaluations.

    """

    def __init__(self, n):
        n = int(n)
        _validate_nonnegative(n, 'n')
        self.n = n

    def observe(self, record):
        """Stop when the optimizer reports nfev >= n."""
        nfev = getattr(record.optimizer, 'nfev', None)
        if nfev is not None and nfev >= self.n:
            return GovernorDecision(
                True, False, 'maximum function evaluations reached',
            )
        return GovernorDecision(False, False, '')


class FunctionTolerance(Governor):
    """Stop when consecutive objective values change by a small amount.

    Parameters
    ----------
    ftol : float
        Function-value tolerance.
    relative : bool, optional
        If True, scale the tolerance by the larger of 1 and the magnitudes of
        the consecutive function values.

    """

    def __init__(self, ftol, relative=True):
        _validate_nonnegative(float(ftol), 'ftol')
        self.ftol = float(ftol)
        self.relative = bool(relative)
        self._previous_f = None

    def observe(self, record):
        """Stop when consecutive function values differ by no more than ftol."""
        has_f_next = 'f_next' in record.metadata
        current_f = float(record.metadata.get('f_next', record.f))
        previous_f = self._previous_f
        if previous_f is None:
            if not has_f_next:
                self._previous_f = current_f
                return GovernorDecision(False, False, '')
            previous_f = record.f

        self._previous_f = current_f

        scale = 1.0
        if self.relative:
            scale = max(1.0, abs(previous_f), abs(current_f))
        if abs(previous_f - current_f) <= self.ftol * scale:
            return GovernorDecision(True, True, 'function tolerance reached')
        return GovernorDecision(False, False, '')


class GradientTolerance(Governor):
    """Stop when the gradient norm is below a threshold.

    Parameters
    ----------
    gtol : float
        Gradient norm tolerance.
    norm : int, float, or str, optional
        Norm order used to measure the gradient.

    """

    def __init__(self, gtol, norm=np.inf):
        _validate_nonnegative(float(gtol), 'gtol')
        self.gtol = float(gtol)
        self.norm = norm

    def observe(self, record):
        """Stop when the gradient norm is below gtol."""
        gnorm = _vector_norm(record.g, self.norm)
        if gnorm <= self.gtol:
            return GovernorDecision(True, True, 'gradient tolerance reached')
        return GovernorDecision(False, False, '')


class StepTolerance(Governor):
    """Stop when the optimizer step is below a threshold.

    Parameters
    ----------
    xtol : float
        Step norm tolerance.
    relative : bool, optional
        If True, scale the tolerance by the larger of 1 and the norm of the
        pre-step iterate.
    norm : int, float, or str, optional
        Norm order used to measure the step.

    """

    def __init__(self, xtol, relative=True, norm=np.inf):
        _validate_nonnegative(float(xtol), 'xtol')
        self.xtol = float(xtol)
        self.relative = bool(relative)
        self.norm = norm

    def observe(self, record):
        """Stop when the step norm is below xtol."""
        step_norm = _vector_norm(record.x_next - record.x, self.norm)
        scale = 1.0
        if self.relative:
            scale = max(1.0, _vector_norm(record.x, self.norm))
        if step_norm <= self.xtol * scale:
            return GovernorDecision(True, True, 'step tolerance reached')
        return GovernorDecision(False, False, '')


class ConstraintTolerance(Governor):
    """Stop when reported constraint violation is below a threshold.

    Parameters
    ----------
    tol : float
        Constraint violation tolerance.

    """

    def __init__(self, tol):
        _validate_nonnegative(float(tol), 'tol')
        self.tol = float(tol)

    def observe(self, record):
        """Stop when the reported constraint violation is below tol."""
        violation = record.metadata.get('constraint_violation', None)
        if violation is None:
            violation = getattr(record.optimizer, 'constraint_violation', None)
        if violation is not None and float(violation) <= self.tol:
            return GovernorDecision(
                True, True, 'constraint tolerance reached',
            )
        return GovernorDecision(False, False, '')
