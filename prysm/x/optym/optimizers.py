"""Various optimization algorithms."""

from prysm.mathops import np

from .governors import (
    GovernorDecision,
    OptimizationResult,
    StepRecord,
)
from .problem import as_problem

from ._lbfgsb import LBFGSB  # NOQA - exporting optimizer
from ._prysm_lbfgsb import PrysmLBFGSB  # NOQA - exporting optimizer


def runN(optimizer, N):
    """Perform N iterations of optimization.

    Parameters
    ----------
    optimizer : Any
        any optimizer from this file, or any type which implements
            def step(self): -> (xk, fk, gk)
                pass
    N : int
        number of iterations to perform

    Returns
    -------
    generator
        yielding (xk, fk, gk) at each iteration

    """
    for _ in range(N):
        yield optimizer.step()


def _stop_iteration_decision(exc):
    value = exc.value
    success = bool(getattr(value, 'success', True))
    message = getattr(value, 'message', 'optimizer stopped')
    if not message:
        message = 'optimizer stopped'
    return GovernorDecision(True, success, message)


def _step_metadata(optimizer):
    metadata = getattr(optimizer, 'last_step_metadata', {})
    if metadata is None:
        return {}
    return metadata


def _as_bound_array(bound, x0, default):
    if bound is None:
        return np.full(x0.shape, default, dtype=x0.dtype)

    bound = np.asarray(bound, dtype=x0.dtype)
    if bound.shape == x0.shape:
        return bound
    if bound.size == x0.size:
        return bound.reshape(x0.shape)
    raise ValueError('bounds must have the same shape or size as x0')


def _init_bounds(optimizer, x0, lower_bounds, upper_bounds):
    lower_bounds = _as_bound_array(lower_bounds, x0, -np.inf)
    upper_bounds = _as_bound_array(upper_bounds, x0, np.inf)
    if bool(np.any(lower_bounds > upper_bounds)):
        raise ValueError('lower_bounds must be <= upper_bounds')

    optimizer.l = lower_bounds  # NOQA - mirrors L-BFGS-B naming
    optimizer.u = upper_bounds
    optimizer._has_bounds = bool(
        np.any(np.isfinite(lower_bounds)) or np.any(np.isfinite(upper_bounds))
    )
    optimizer.x = _project_bounds(optimizer, optimizer.x)
    optimizer.last_step_metadata = {}


def _project_bounds(optimizer, x):
    if not optimizer._has_bounds:
        return x
    return np.minimum(np.maximum(x, optimizer.l), optimizer.u)


def _project_gradient(optimizer, g):
    """Zero gradient components blocked by active box constraints."""
    if not optimizer._has_bounds:
        return g

    x = optimizer.x
    at_lower = np.isfinite(optimizer.l) & (x <= optimizer.l) & (g > 0)
    at_upper = np.isfinite(optimizer.u) & (x >= optimizer.u) & (g < 0)
    blocked = at_lower | at_upper
    if bool(np.any(blocked)):
        return np.where(blocked, np.zeros_like(g), g)
    return g


def _store_bounded_step_metadata(optimizer, g_step):
    if not optimizer._has_bounds:
        optimizer.last_step_metadata = {}
        return

    x = optimizer.x
    at_lower = np.isfinite(optimizer.l) & (x <= optimizer.l)
    at_upper = np.isfinite(optimizer.u) & (x >= optimizer.u)
    active_bounds = at_lower | at_upper
    optimizer.last_step_metadata = {
        'projected_gradient': g_step,
        'active_bounds': active_bounds,
        'bounded_variables': int(active_bounds.sum()),
    }


def run_until(optimizer, governor, *, maxiter=None):
    """Run an optimizer until a governor decides to stop.

    Parameters
    ----------
    optimizer : Any
        Optimizer exposing a step method returning x, f, g.  After each step,
        optimizer.x is expected to hold the post-step iterate.
    governor : Governor
        Stop condition that observes each completed step.
    maxiter : int, optional
        Safety cap for runner-owned iterations.  This is independent of any
        MaxIterations governor supplied by the caller.

    Returns
    -------
    OptimizationResult
        Result with the final optimizer.x, terminal decision, and step records.

    """
    records = []
    if maxiter is not None:
        maxiter = int(maxiter)
        if maxiter <= 0:
            decision = GovernorDecision(
                True, False, 'maximum iterations reached',
            )
            return OptimizationResult(
                getattr(optimizer, 'x', None), decision, records, optimizer,
            )

    iteration = 0
    while maxiter is None or iteration < maxiter:
        iteration += 1
        try:
            x, f, g = optimizer.step()
        except StopIteration as exc:
            decision = _stop_iteration_decision(exc)
            return OptimizationResult(
                getattr(optimizer, 'x', None), decision, records, optimizer,
            )

        record = StepRecord(
            optimizer=optimizer,
            iteration=iteration,
            x=x,
            f=f,
            g=g,
            x_next=optimizer.x,
            metadata=_step_metadata(optimizer),
        )
        records.append(record)
        decision = governor.observe(record)
        if decision.stop:
            return OptimizationResult(optimizer.x, decision, records, optimizer)

    decision = GovernorDecision(True, False, 'maximum iterations reached')
    return OptimizationResult(optimizer.x, decision, records, optimizer)


class _Accumulator:
    """Shared state for accumulator-based optimizers (AdaGrad / RMSProp).

    Holds a per-parameter accumulator initialized to zeros that subclass
    step methods update with some function of the gradient.  The
    problem, x, alpha, bounds, eps, iter state is the same as
    every other optimizer in this module.

    Subclasses with additional hyperparameters override __init__ to call
    super().__init__(fg, x0, alpha, lower_bounds, upper_bounds) and then set
    their extras.  Bounds are enforced by projecting iterates into the box;
    active-bound gradient components that point out of the box are zeroed
    before updating the accumulator.

    Subclass contract for step:
        return (x, f, g) where (f, g) were evaluated at x (the
        iterate before this step's update) and self.x holds the updated
        iterate.

    """
    def __init__(self, fg, x0, alpha, lower_bounds=None, upper_bounds=None):
        self.problem = as_problem(fg)
        self.x0 = x0
        self.alpha = alpha
        self.x = x0.copy()
        _init_bounds(self, x0, lower_bounds, upper_bounds)
        self.accumulator = np.zeros_like(self.x)
        self.eps = np.finfo(x0.dtype).eps
        self.iter = 0


class _MomentBased:
    """Shared state for moment-based optimizers (Adam / RAdam / AdaMomentum / Yogi).

    Holds first- and second-moment buffers m and v initialized to
    zeros that subclass step methods update from the gradient.  Two
    decay rates beta1 and beta2 live here too.  Bounds are enforced by
    projecting iterates into the box; active-bound gradient components that
    point out of the box are zeroed before updating the moments.

    Subclasses with additional hyperparameters (e.g. RAdam's rhoinf)
    override __init__ to call super().__init__(fg, x0, alpha, beta1,
    beta2, lower_bounds, upper_bounds) and then set their extras.

    Subclass contract for step: see _Accumulator.

    """
    def __init__(self, fg, x0, alpha, beta1=0.9, beta2=0.999,
                 lower_bounds=None, upper_bounds=None):
        self.problem = as_problem(fg)
        self.x0 = x0
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.x = x0.copy()
        _init_bounds(self, x0, lower_bounds, upper_bounds)
        self.m = np.zeros_like(x0)
        self.v = np.zeros_like(x0)
        self.eps = np.finfo(x0.dtype).eps
        self.iter = 0


class GradientDescent:
    r"""Gradient Descent optimization routine.

    Gradient Descent travels a constant step size alpha along the negative of
    the gradient on each iteration.  The update is:

    .. math::
        x_{k+1} = x_k - α g_k

    where g is the gradient vector

    The user may anneal alpha over the course
    of optimization if they wish.  The cost function is not used, nor higher
    order information.
    """
    def __init__(self, fg, x0, alpha, lower_bounds=None, upper_bounds=None):
        """Create a new GradientDescent optimizer.

        Parameters
        ----------
        fg : callable
            a function which returns (f, g) where f is the scalar cost, and
            g is the vector gradient.
        x0 : callable
            the parameter vector immediately prior to optimization
        alpha : float
            the step size
            the user may mutate self.alpha over the course of optimization
            with no negative effects (except optimization blowing up from a bad
            choice of new alpha)
        lower_bounds, upper_bounds : ndarray, optional
            per-variable hard bounds.  None means unconstrained on that side.

        """
        self.problem = as_problem(fg)
        self.x0 = x0
        self.alpha = alpha
        self.x = x0.copy()
        _init_bounds(self, x0, lower_bounds, upper_bounds)
        self.iter = 0

    def step(self):
        """Perform one iteration of optimization."""
        f, g = self.problem.fg(self.x)
        g_step = _project_gradient(self, g)
        x = self.x
        self.x = _project_bounds(self, x - self.alpha*g_step)
        self.iter += 1
        _store_bounded_step_metadata(self, g_step)
        return x, f, g


class AdaGrad(_Accumulator):
    r"""Adaptive Gradient Descent optimization routine.

    Gradient Descent has the same step size for each parameter.  Adagrad self-
    learns a unique step size for each parameter based on accumulation of the
    square of the gradient over the course of optimization.  The update is:

    .. math::
        s_k &= s_{k-1} + (g*g) \\
        x_{k+1} &= x_k - α g_k / \sqrt{s_k \,}

    The purpose of the square and square root operations is essentially to destroy
    the sign of g in the denomenator gain.  An alternative may be to simply do
    s_k = s_(k-1) + abs(g), which would have less numerical precision issues.

    Ref [1] describes a ""fully connected"" version of AdaGrad that is a cousin
    of sorts to BFGS, storing an NxN matrix.  This is intractible for large N.
    The implementation here is sister to most implementations in the wild, and
    is the "Diagonal" implementation, which stores no information about the
    relationship between "spatial" elements of the gradient vector.  Only the
    temporal relationship between the gradient and its past is stored.

    Constructed with AdaGrad(fg, x0, alpha) — see _Accumulator
    for the shared init signature.

    References
    ----------
    [1] Duchi, John, Hazan, Elad and Singer, Yoram. "Adaptive Subgradient
        Methods for Online Learning and Stochastic Optimization."
        https://doi.org/10.5555/1953048.2021068

    """

    def step(self):
        """Perform one iteration of optimization."""
        f, g = self.problem.fg(self.x)
        g_step = _project_gradient(self, g)
        self.accumulator += (g_step*g_step)
        x = self.x
        step = self.alpha * g_step / (np.sqrt(self.accumulator)+self.eps)
        self.x = _project_bounds(self, x - step)
        self.iter += 1
        _store_bounded_step_metadata(self, g_step)
        return x, f, g


class RMSProp(_Accumulator):
    r"""RMSProp optimization routine.

    RMSProp keeps a moving average of the squared gradient of each parameter.

    It is very similar to AdaGrad, except that the decay built into it allows
    it to forget old gradients.  This makes it often superior for non-convex
    problems, where navigation from one valley into another poisons AdaGrad, but
    RMSProp will eventually forget about the old valley.

    The update is:

    .. math::
        s_k &= γ * s_{k-1} + (1-γ)*(g*g) \\
        x_{k+1} &= x_k - α g_k / \sqrt{s_k \,}

    The decay terms gamma form a "moving average" that is squared, with the
    square root in the gain it is a "root mean square."

    RMSProp is an unpublished algorithm.  Its source is Ref [1]

    References
    ----------
    [1] Geoffrey Hinton, Nitish Srivastava, Kevin Swersky
        "Neural Networks for Machine Learning
        Lecture 6a Overview of mini-­‐batch gradient descent"
        U Toronto, CSC 321
        https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf

    """
    def __init__(self, fg, x0, alpha, gamma=0.9,
                 lower_bounds=None, upper_bounds=None):
        super().__init__(fg, x0, alpha, lower_bounds, upper_bounds)
        self.gamma = gamma

    def step(self):
        """Perform one iteration of optimization."""
        gamma = self.gamma
        f, g = self.problem.fg(self.x)
        g_step = _project_gradient(self, g)
        self.accumulator = gamma*self.accumulator + (1-gamma)*(g_step*g_step)
        x = self.x
        step = self.alpha * g_step / (np.sqrt(self.accumulator)+self.eps)
        self.x = _project_bounds(self, x - step)
        self.iter += 1
        _store_bounded_step_metadata(self, g_step)
        return x, f, g


class Adam(_MomentBased):
    r"""ADAM optimization routine.

    ADAM, or "Adaptive moment estimation" uses moving average estimates of the
    mean of the gradient and of its "uncentered variance".  This causes the
    algorithm to combine several properties of AdaGrad and RMSPRop, as well as
    perform a form of self-annealing, where the step size will naturally decay
    as the optimizer converges.  This can cause ADAM to recover itself after
    diverging, if the divergence is not too extreme.

    The update is:

    .. math::
        m &\equiv \text{mean} \\
        v &\equiv \text{variance} \\
        m_k &= β_1 m_{k-1} + (1-β_1) * g \\
        v_k &= β_2 v_{k-1} + (1-β_2) * (g*g) \\
        \hat{m}_k &= m_k / (1 - β_1^k) \\
        \hat{v}_k &= v_k / (1 - β_2^k) \\
        x_{k+1} &= x_k - α * \hat{m}_k / \sqrt{\hat{v}_k \,} \\

    Constructed with Adam(fg, x0, alpha, beta1=0.9, beta2=0.999) —
    see _MomentBased for the shared init signature.

    References
    ----------
    [1] Kingma, Diederik and Ba, Jimmy. "Adam: A Method for Stochastic Optimization"
        http://arxiv.org/abs/1412.6980

    """

    def step(self):
        """Perform one iteration of optimization."""
        self.iter += 1
        beta1 = self.beta1
        beta2 = self.beta2
        f, g = self.problem.fg(self.x)
        g_step = _project_gradient(self, g)
        # update momentum estimates
        self.m = beta1*self.m + (1-beta1) * g_step
        self.v = beta2*self.v + (1-beta2) * (g_step*g_step)

        mhat = self.m / (1 - beta1**self.iter)
        vhat = self.v / (1 - beta2**self.iter)

        x = self.x
        step = self.alpha * mhat/(np.sqrt(vhat)+self.eps)
        self.x = _project_bounds(self, x - step)
        _store_bounded_step_metadata(self, g_step)
        return x, f, g


class RAdam(_MomentBased):
    r"""RADAM optimization routine.

    RADAM or Rectified ADAM is a modification of ADAM, which seeks to remove
    any need for warmup time with ADAM, and to stabilize the variance estimate
    or second moment that ADAM uses.  These properties make RADAM more invariant
    to the choice of hyperparameters, especially alpha, and avoid distorting
    the trajectory of optimization in early iterations.

    The update is:

    .. math::
        m &\equiv \text{mean} \\
        v &\equiv \text{variance} \\
        \rho_\infty &= \frac{2}{1-β_2} - 1 \\
        m_k &= β_1 m_{k-1} + (1-β_1) * g \\
        v_k &= β_2 v_{k-1} + (1-β_2) * (g*g) \\
        \hat{m}_k &= m_k / (1 - β_1^k) \\
        \rho_k &= \rho_\infty - \frac{2 k β_2^k}{1-β_2^k} \\
        \text{if}& \rho_k > 5 \\
            \qquad l_k &= \sqrt{\frac{1 - β_2^k}{\sqrt{v_k}}} \\
            \qquad r_k &= \sqrt{\frac{(\rho_k - 4)(\rho_k-2)\rho_\infty}{(\rho_\infty-4)(\rho_\infty-2)\rho_t}} \\
            \qquad x_{k+1} &= x_k - α r_k \hat{m}_k l_k \\
        \text{else}& \\
            \qquad x_{k+1} &= x_k - α \hat{m}_k

    References
    ----------
    [1] Liu, Liyuan and Jiang, Haoming and He, Pengcheng and Chen, Weizhu and Liu Xiaodong, and Gao, Jianfeng and Han Jiawei. "On the Variance of the Adaptive Learning Rate and Beyond"
        http://arxiv.org/abs/1412.6980

    """
    def __init__(self, fg, x0, alpha, beta1=0.9, beta2=0.999,
                 lower_bounds=None, upper_bounds=None):
        super().__init__(fg, x0, alpha, beta1, beta2,
                         lower_bounds, upper_bounds)
        self.rhoinf = 2 / (1 - beta2) - 1

    def step(self):
        """Perform one iteration of optimization."""
        self.iter += 1
        k = self.iter
        beta1 = self.beta1
        beta2 = self.beta2
        beta2k = beta2**k

        f, g = self.problem.fg(self.x)
        g_step = _project_gradient(self, g)
        gsq = g_step*g_step
        # update momentum estimates
        self.m = beta1*self.m + (1-beta1) * g_step
        self.v = beta2*self.v + (1-beta2) * (gsq)
        # torch exp_avg_sq.mul_(beta2).addcmul_(grad,grad,value=1-beta2)
        # == v

        # going to use this many times, local lookup is cheaper
        rhoinf = self.rhoinf
        rho = rhoinf - (2*k*beta2k)/(1-beta2k)
        x = self.x
        if rho >= 5:  # 5 was 4 in the paper, but PyTorch uses 5, most others too
            # l = np.sqrt((1-beta2k)/self.v)  # NOQA
            # commented out l exactly as in paper
            # seems to blow up all the time, must be a typo; missing sqrt(v)
            # torch computes vhat same as ADAM, assume that's the typo
            mhat = self.m / (1 - beta1**k)
            l = np.sqrt(1 - beta2k) / (np.sqrt(self.v)+self.eps)  # NOQA
            num = (rho - 4) * (rho - 2) * rhoinf
            den = (rhoinf - 4) * (rhoinf - 2) * rho
            r = np.sqrt(num/den)
            self.x = _project_bounds(self, x - self.alpha * r * mhat * l)
        else:
            self.x = _project_bounds(self, x - self.alpha * g_step)
        _store_bounded_step_metadata(self, g_step)
        return x, f, g


class AdaMomentum(_MomentBased):
    r"""AdaMomentum optimization routine.

    AdaMomentum is an algorithm that is extremely similar to Adam, differing
    only in the calculation of v.  The idea is to reduce teh variance of the
    correction, which can plausibly improve the generality of found solutions,
    i.e. enter wider local/global minima.

    The update is:

    .. math::
        m &\equiv \text{mean} \\
        v &\equiv \text{variance} \\
        m_k &= β_1 m_{k-1} + (1-β_1) * g \\
        v_k &= β_2 v_{k-1} + (1-β_2) * m_k^2 \\
        \hat{m}_k &= m_k / (1 - β_1^k) \\
        \hat{v}_k &= v_k / (1 - β_2^k) \\
        x_{k+1} &= x_k - α * \hat{m}_k / \sqrt{\hat{v}_k \,} \\

    Constructed with AdaMomentum(fg, x0, alpha, beta1=0.9, beta2=0.999) —
    see _MomentBased for the shared init signature.

    References
    ----------
    [1] Wang, Yizhou and Kang, Yue and Qin, Can and Wang, Huan and Xu, Yi and Zhang Yulun and Fu, Yun. "Rethinking Adam: A Twofold Exponential Moving Average Approach"
        https://arxiv.org/abs/2106.11514

    """

    def step(self):
        """Perform one iteration of optimization."""
        self.iter += 1
        beta1 = self.beta1
        beta2 = self.beta2
        f, g = self.problem.fg(self.x)
        g_step = _project_gradient(self, g)
        # update momentum estimates
        self.m = beta1*self.m + (1-beta1) * g_step
        self.v = beta2*self.v + (1-beta2) * (self.m*self.m) + self.eps

        mhat = self.m / (1 - beta1**self.iter)
        vhat = self.v / (1 - beta2**self.iter)

        x = self.x
        self.x = _project_bounds(self, x - self.alpha * mhat/np.sqrt(vhat))
        _store_bounded_step_metadata(self, g_step)
        return x, f, g


class Yogi(_MomentBased):
    r"""YOGI optimization routine.

    YOGI is a modification of ADAM, which replaces the multiplicative update
    to the exponentially moving averaged estimate of the variance of the gradient
    with an additive one.  The premise for this is that multiplicative update
    causes ADAM to forget past gradients too quickly, tailoring it to more
    localized behavior in the second order momentum term.  The additive update
    in YOGI essentially makes the second order momentum update over a larger
    space.  This allows Yogi to perform better than ADAM for some non-convex
    or otherwise tumultuous cost functions, which have many changes in local
    curvature.

    The update is:

    .. math::
        m &\equiv \text{mean} \\
        v &\equiv \text{variance} \\
        m_k &= β_1 m_{k-1} + (1-β_1) * g \\
        v_k &= v_{k-1} - (1-β_2) * \text{sign}(v_{k-1} - (g^2))*(g^2) \\
        x_{k+1} &= x_k - α * m_k / \sqrt{v_k \,} \\

    Constructed with Yogi(fg, x0, alpha, beta1=0.9, beta2=0.999) —
    see _MomentBased for the shared init signature.

    References
    ----------
    [1] Zaheer, Manzil and Reddi, Sashank and Sachan, Devendra and Kale, Satyen and Kumar, Sanjiv. "Adaptive Methods for Nonconvex Optimization"
        https://papers.nips.cc/paper_files/paper/2018/hash/90365351ccc7437a1309dc64e4db32a3-Abstract.html

    """

    def step(self):
        """Perform one iteration of optimization."""
        self.iter += 1
        beta1 = self.beta1
        beta2 = self.beta2
        f, g = self.problem.fg(self.x)
        g_step = _project_gradient(self, g)
        gsq = g_step*g_step
        # update momentum estimates
        self.m = beta1*self.m + (1-beta1) * g_step
        self.v = self.v - (1-beta2) * np.sign(self.v - gsq)*gsq

        mhat = self.m  # for symmetry to ADAM
        vhat = np.sqrt(self.v+self.eps)

        x = self.x
        step = self.alpha * mhat/(np.sqrt(vhat)+self.eps)
        self.x = _project_bounds(self, x - step)
        _store_bounded_step_metadata(self, g_step)
        return x, f, g
