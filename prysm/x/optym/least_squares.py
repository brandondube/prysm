"""Least-squares optimizers and helpers."""

from prysm.mathops import np

from .governors import (
    AllGovernor,
    AnyGovernor,
    ConstraintTolerance,
    FunctionTolerance,
    MaxIterations,
    StepRecord,
    StepTolerance,
)


class DampedLeastSquaresResult:
    """Result object returned by damped_least_squares."""

    __slots__ = (
        'x',
        'residuals',
        'cost',
        'success',
        'message',
        'nit',
        'nfev',
        'njev',
        'ncev',
        'lambda_eq',
        'lambda_ineq',
        'active_inequalities',
        'history',
    )

    def __init__(self, x, residuals, cost, success, message, nit, nfev,
                 njev, ncev, lambda_eq, lambda_ineq, active_inequalities,
                 history):
        self.x = x
        self.residuals = residuals
        self.cost = cost
        self.success = bool(success)
        self.message = message
        self.nit = int(nit)
        self.nfev = int(nfev)
        self.njev = int(njev)
        self.ncev = int(ncev)
        self.lambda_eq = lambda_eq
        self.lambda_ineq = lambda_ineq
        self.active_inequalities = active_inequalities
        self.history = history

    def __repr__(self):
        return (
            f'DampedLeastSquaresResult(success={self.success}, '
            f'cost={self.cost:.6g}, nit={self.nit}, '
            f'nfev={self.nfev})'
        )


class _DLSState:
    __slots__ = (
        'x',
        'residuals',
        'eq',
        'ineq',
        'cost',
        'violation',
        'nfev',
        'njev',
        'ncev',
        'lambda_eq',
        'lambda_ineq',
        'active',
        'history',
    )

    def __init__(self, x, residuals, eq, ineq):
        self.x = x
        self.residuals = residuals
        self.eq = eq
        self.ineq = ineq
        self.cost = _cost(residuals)
        self.violation = _constraint_violation(eq, ineq)
        self.nfev = 1
        self.njev = 0
        self.ncev = 1
        self.lambda_eq = np.zeros(0, dtype=float)
        self.lambda_ineq = np.zeros(0, dtype=float)
        self.active = np.zeros(0, dtype=int)
        self.history = []

    def result(self, success, message, iteration):
        return DampedLeastSquaresResult(
            self.x, self.residuals, self.cost, success, message, iteration,
            self.nfev, self.njev, self.ncev, self.lambda_eq,
            self.lambda_ineq, self.active, self.history,
        )


class _ResidualProblemView:
    __slots__ = ('problem', 'eq_constraints', 'ineq_constraints')

    def __init__(self, problem, equality_constraints, inequality_constraints):
        self.problem = problem
        self.eq_constraints = _as_constraint_list(equality_constraints)
        self.ineq_constraints = _as_constraint_list(inequality_constraints)

    def residuals(self, x):
        return np.asarray(self.problem.residuals(x), dtype=float).ravel()

    def eq(self, x):
        return _eval_constraint_vector(self.eq_constraints, x)

    def ineq(self, x):
        return _eval_constraint_vector(self.ineq_constraints, x)


def _as_constraint_list(constraints):
    if constraints is None:
        return ()
    if callable(constraints):
        return (constraints,)
    return tuple(constraints)


def _eval_constraint_vector(constraints, x):
    vals = []
    for constraint in constraints:
        value = np.asarray(constraint(x), dtype=float).ravel()
        vals.append(value)
    if not vals:
        return np.zeros(0, dtype=float)
    return np.concatenate(vals)


def _cost(residuals):
    return 0.5 * float(np.sum(residuals * residuals))


def _norm(x):
    return float(np.sqrt(np.sum(x * x)))


def _finite_difference_jacobian(fun, x, f0=None, step=1e-6):
    x = np.asarray(x, dtype=float)
    if f0 is None:
        f0 = np.asarray(fun(x), dtype=float).ravel()
    else:
        f0 = np.asarray(f0, dtype=float).ravel()

    jac = np.empty((f0.size, x.size), dtype=float)
    xflat = x.ravel()
    for j in range(x.size):
        h = float(step) * max(1.0, abs(float(xflat[j])))
        xp = x.copy()
        xm = x.copy()
        xp.ravel()[j] = xflat[j] + h
        xm.ravel()[j] = xflat[j] - h
        fp = np.asarray(fun(xp), dtype=float).ravel()
        fm = np.asarray(fun(xm), dtype=float).ravel()
        jac[:, j] = (fp - fm) / (2 * h)
    return jac


def _constraint_violation(eq, ineq):
    eq_norm = _norm(eq) if eq.size else 0.0
    if ineq.size:
        ineq_norm = _norm(np.minimum(ineq, 0.0))
    else:
        ineq_norm = 0.0
    return float(np.sqrt(eq_norm * eq_norm + ineq_norm * ineq_norm))


def _solve_kkt(H, grad, A, b):
    n = H.shape[0]
    m = A.shape[0]
    if m == 0:
        try:
            return np.linalg.solve(H, -grad), np.zeros(0, dtype=H.dtype)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, -grad, rcond=None)[0]
            return step, np.zeros(0, dtype=H.dtype)

    K = np.zeros((n + m, n + m), dtype=H.dtype)
    K[:n, :n] = H
    K[:n, n:] = A.T
    K[n:, :n] = A
    rhs = np.concatenate([-grad, b])
    try:
        sol = np.linalg.solve(K, rhs)
    except np.linalg.LinAlgError:
        sol = np.linalg.lstsq(K, rhs, rcond=None)[0]
    return sol[:n], sol[n:]


def _normal_matrix(residuals, jacobian, damping):
    H = jacobian.T @ jacobian
    if damping:
        H = H + float(damping) * np.eye(jacobian.shape[1], dtype=H.dtype)
    return H, jacobian.T @ residuals


def _constraint_matrix(active, Aeq, Aineq, eq, ineq):
    if len(active):
        A = np.vstack([Aeq, Aineq[active]]) if Aeq.size else Aineq[active]
        b_active = -ineq[active]
        b = np.concatenate([-eq, b_active]) if eq.size else b_active
        return A, b
    return Aeq, -eq


def _newly_violated_inequalities(ineq, Aineq, dx, active, constraint_tol):
    linear_ineq = ineq + Aineq @ dx
    return [
        idx for idx in np.nonzero(linear_ineq < -constraint_tol)[0]
        if idx not in active
    ]


def _inactive_multipliers(active, multipliers, n_eq, ineq, constraint_tol):
    active_multipliers = multipliers[n_eq:]
    return [
        active[i] for i, lm in enumerate(active_multipliers)
        if lm > constraint_tol and ineq[active[i]] >= -constraint_tol
    ]


def _multipliers(eq, ineq, active, raw_multipliers):
    lambda_eq = np.zeros(eq.size, dtype=float)
    lambda_ineq = np.zeros(ineq.size, dtype=float)
    if eq.size:
        lambda_eq = raw_multipliers[:eq.size]
    if len(active):
        lambda_ineq[np.asarray(active, dtype=int)] = raw_multipliers[eq.size:]
    return lambda_eq, lambda_ineq


def _active_set_step(state, J, Aeq, Aineq, damping, constraint_tol,
                     active_tol, max_active_iter):
    H, grad = _normal_matrix(state.residuals, J, damping)
    active = []
    if state.ineq.size:
        active = np.nonzero(state.ineq <= active_tol)[0].tolist()

    dx = np.zeros(J.shape[1], dtype=float)
    raw_multipliers = np.zeros(0, dtype=float)
    for _ in range(max_active_iter):
        A, b = _constraint_matrix(active, Aeq, Aineq, state.eq, state.ineq)
        dx, raw_multipliers = _solve_kkt(H, grad, A, b)

        if state.ineq.size:
            missing = _newly_violated_inequalities(
                state.ineq, Aineq, dx, active, constraint_tol,
            )
            if missing:
                active.extend(missing)
                active.sort()
                continue

        drop = _inactive_multipliers(
            active, raw_multipliers, state.eq.size, state.ineq, constraint_tol,
        )
        if drop:
            active = [idx for idx in active if idx not in drop]
            continue
        break

    lambda_eq, lambda_ineq = _multipliers(
        state.eq, state.ineq, active, raw_multipliers,
    )
    return dx, lambda_eq, lambda_ineq, np.asarray(active, dtype=int)


def _initial_x(problem, x0):
    if x0 is not None:
        return np.asarray(x0, dtype=float).copy()
    if not hasattr(problem, 'x0'):
        raise TypeError('x0 is required when problem has no x0 method')
    return np.asarray(problem.x0(), dtype=float)


def _initial_state(view, x):
    return _DLSState(x, view.residuals(x), view.eq(x), view.ineq(x))


def _constraint_jacobians(view, state, fd_step):
    Aeq = _finite_difference_jacobian(
        view.eq, state.x, f0=state.eq, step=fd_step,
    ) if state.eq.size else np.zeros((0, state.x.size), dtype=float)
    Aineq = _finite_difference_jacobian(
        view.ineq, state.x, f0=state.ineq, step=fd_step,
    ) if state.ineq.size else np.zeros((0, state.x.size), dtype=float)
    return Aeq, Aineq


def _trial_state(view, x):
    return _DLSState(x, view.residuals(x), view.eq(x), view.ineq(x))


def _accept_trial(state, trial, ftol, constraint_tol):
    feasible = trial.violation <= constraint_tol
    cost_ok = trial.cost <= state.cost + ftol * max(1.0, state.cost)
    if state.violation > constraint_tol:
        return trial.violation < state.violation
    return feasible and cost_ok


def _line_search(view, state, dx, ftol, constraint_tol, max_line_search):
    alpha = 1.0
    evaluations = 0
    for _ in range(max_line_search + 1):
        trial = _trial_state(view, state.x + alpha * dx)
        evaluations += 1
        if _accept_trial(state, trial, ftol, constraint_tol):
            return alpha, trial, evaluations
        alpha *= 0.5
    return None, None, evaluations


def _record_history(state, trial, step_norm, alpha):
    state.history.append({
        'x': trial.x.copy(),
        'cost': trial.cost,
        'constraint_violation': trial.violation,
        'step_norm': step_norm,
        'alpha': alpha,
        'active_inequalities': state.active.copy(),
    })


def _copy_trial_into_state(state, trial):
    state.x = trial.x
    state.residuals = trial.residuals
    state.eq = trial.eq
    state.ineq = trial.ineq
    state.cost = trial.cost
    state.violation = trial.violation


def _linearized_step(view, state, damping, fd_step, constraint_tol,
                     active_tol, max_active_iter):
    J = _finite_difference_jacobian(
        view.residuals, state.x, f0=state.residuals, step=fd_step,
    )
    state.nfev += 2 * state.x.size
    state.njev += 1
    grad = J.T @ state.residuals

    Aeq, Aineq = _constraint_jacobians(view, state, fd_step)
    if state.eq.size or state.ineq.size:
        state.ncev += 2 * state.x.size

    dx, state.lambda_eq, state.lambda_ineq, state.active = _active_set_step(
        state, J, Aeq, Aineq, damping, constraint_tol,
        active_tol, max_active_iter,
    )
    return dx, grad


class DampedLeastSquares:
    """Constrained damped least-squares optimizer with a step method.

    Parameters
    ----------
    problem : object
        Object with a residuals(x) method.
    x0 : ndarray, optional
        Starting vector.  If omitted, problem.x0() is used.
    equality_constraints : callable or sequence of callables, optional
        Functions returning values that must equal zero.
    inequality_constraints : callable or sequence of callables, optional
        Functions returning values that must be greater than or equal to zero.
    damping : float, optional
        Scalar added to the diagonal of J.T @ J.

    Notes
    -----
    Constraints are imposed on each local linearized DLS subproblem by
    KKT equations.  Linear constraints are exact in the subproblem;
    nonlinear constraints converge through sequential linearization.
    Inequality constraints use the convention g(x) >= 0.

    """
    def __init__(self, problem, x0=None, *, equality_constraints=None,
                 inequality_constraints=None, damping=1e-6,
                 maxiter=25, xtol=1e-10, ftol=1e-12,
                 constraint_tol=1e-10, active_tol=1e-10,
                 fd_step=1e-6, max_active_iter=20,
                 max_line_search=12):
        """Create a new constrained damped least-squares optimizer."""
        self.problem = problem
        self.view = _ResidualProblemView(
            problem, equality_constraints, inequality_constraints,
        )
        self.state = _initial_state(self.view, _initial_x(problem, x0))
        self.x0 = self.state.x.copy()
        self.x = self.state.x
        self.damping = damping
        self.maxiter = int(maxiter)
        self.xtol = xtol
        self.ftol = ftol
        self.constraint_tol = constraint_tol
        self.active_tol = active_tol
        self.fd_step = fd_step
        self.max_active_iter = max_active_iter
        self.max_line_search = max_line_search
        self.iter = 0
        self.done = False
        self.success = False
        self.message = ''
        self.last_step_norm = None
        self.last_alpha = None
        self.last_step_metadata = {}
        self._governor = AnyGovernor([
            StepTolerance(xtol, relative=True),
            AllGovernor([
                FunctionTolerance(ftol, relative=True),
                ConstraintTolerance(constraint_tol),
            ]),
            MaxIterations(self.maxiter),
        ])
        self._result_iteration = 0

    def _finish(self, success, message, iteration):
        self.done = True
        self.success = bool(success)
        self.message = message
        self._result_iteration = int(iteration)

    @property
    def nfev(self):
        """Number of residual function evaluations."""
        return self.state.nfev

    @property
    def njev(self):
        """Number of residual Jacobian evaluations."""
        return self.state.njev

    @property
    def ncev(self):
        """Number of constraint function evaluations."""
        return self.state.ncev

    @property
    def constraint_violation(self):
        """Current combined equality and inequality constraint violation."""
        return self.state.violation

    def _metadata(self, step_norm, alpha, accepted, f_next=None):
        if f_next is None:
            f_next = self.state.cost
        return {
            'step_norm': step_norm,
            'alpha': alpha,
            'constraint_violation': self.state.violation,
            'active_inequalities': self.state.active.copy(),
            'lambda_eq': self.state.lambda_eq.copy(),
            'lambda_ineq': self.state.lambda_ineq.copy(),
            'f_next': f_next,
            'accepted': accepted,
        }

    def _observe_governor(self, iteration, x, f, g):
        record = StepRecord(
            optimizer=self,
            iteration=iteration,
            x=x,
            f=f,
            g=g,
            x_next=self.x,
            metadata=self.last_step_metadata,
        )
        return self._governor.observe(record)

    def _finish_from_decision(self, decision, iteration):
        message = decision.message
        success = decision.success
        if 'function tolerance reached' in message:
            message = 'cost tolerance reached'
            success = self.state.violation <= self.constraint_tol
        elif message == 'maximum iterations reached':
            success = self.state.violation <= self.constraint_tol
        elif message == 'step tolerance reached':
            success = self.state.violation <= self.constraint_tol
        self._finish(success, message, iteration)

    def result(self):
        """Return the current result object."""
        return self.state.result(
            self.success, self.message, self._result_iteration,
        )

    def step(self):
        """Perform one iteration of optimization.

        Returns
        -------
        tuple
            (x, f, g), where f and g are the cost and least-squares
            gradient evaluated at x before the update.  After a successful
            step, self.x holds the updated iterate.

        """
        if self.done:
            raise StopIteration(self.result())

        iteration = self.iter + 1
        x = self.state.x
        f = self.state.cost
        dx, g = _linearized_step(
            self.view, self.state, self.damping, self.fd_step,
            self.constraint_tol, self.active_tol, self.max_active_iter,
        )

        step_norm = _norm(dx)
        self.last_step_norm = step_norm
        x_norm = _norm(self.state.x)
        if step_norm <= self.xtol * (self.xtol + x_norm):
            self.last_alpha = None
            self.last_step_metadata = self._metadata(
                step_norm, None, False,
            )
            decision = self._observe_governor(iteration, x, f, g)
            self._finish_from_decision(decision, iteration - 1)
            return x, f, g

        alpha, trial, evaluations = _line_search(
            self.view, self.state, dx, self.ftol, self.constraint_tol,
            self.max_line_search,
        )
        self.last_alpha = alpha
        self.state.nfev += evaluations
        self.state.ncev += evaluations
        if trial is None:
            self.last_step_metadata = self._metadata(
                step_norm, alpha, False,
            )
            self.last_step_metadata['line_search_failed'] = True
            self._finish(False, 'line search failed', iteration)
            return x, f, g

        f_next = trial.cost
        _record_history(self.state, trial, step_norm, alpha)
        _copy_trial_into_state(self.state, trial)
        self.x = self.state.x
        self.iter += 1
        self.last_step_metadata = self._metadata(
            step_norm, alpha, True, f_next=f_next,
        )

        decision = self._observe_governor(self.iter, x, f, g)
        if decision.stop:
            self._finish_from_decision(decision, self.iter)

        return x, f, g

    def run(self):
        """Run until DLS reaches its configured stopping condition."""
        if self.maxiter <= 0 and not self.done:
            success = self.state.violation <= self.constraint_tol
            self._finish(success, 'maximum iterations reached', 0)
        while not self.done:
            self.step()
        return self.result()


def damped_least_squares(problem, x0=None, *, equality_constraints=None,
                         inequality_constraints=None, damping=1e-6,
                         maxiter=25, xtol=1e-10, ftol=1e-12,
                         constraint_tol=1e-10, active_tol=1e-10,
                         fd_step=1e-6, max_active_iter=20,
                         max_line_search=12):
    """Run constrained damped least squares to completion."""
    optimizer = DampedLeastSquares(
        problem,
        x0=x0,
        equality_constraints=equality_constraints,
        inequality_constraints=inequality_constraints,
        damping=damping,
        maxiter=maxiter,
        xtol=xtol,
        ftol=ftol,
        constraint_tol=constraint_tol,
        active_tol=active_tol,
        fd_step=fd_step,
        max_active_iter=max_active_iter,
        max_line_search=max_line_search,
    )
    return optimizer.run()
