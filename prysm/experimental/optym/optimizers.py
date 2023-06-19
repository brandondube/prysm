"""Various optimization algorithms."""
import warnings

from scipy.optimize import _lbfgsb

from prysm.mathops import np


class GradientDescent:
    """Gradient Descent optimization routine.

    Gradient Descent travels a constant step size alpha along the negative of
    the gradient on each iteration.  The update is:

    x_(k+1) = x_k - α g_k

    where g is the gradient vector

    The user may anneal alpha over the course
    of optimization if they wish.  The cost function is not used, nor higher
    order information.
    """
    def __init__(self, fg, x0, alpha):
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

        """
        self.fg = fg
        self.x0 = x0
        self.alpha = alpha
        self.x = x0.copy()
        self.iter = 0

    def step(self):
        """Perform one iteration of optimization."""
        f, g = self.fg(self.x)
        self.x -= self.alpha*g
        self.iter += 1
        return self.x, f, g

    def runN(self, N):
        """Perform N iterations of optimization."""
        for _ in range(N):
            yield self.step()


class AdaGrad:
    """Adaptive Gradient Descent optimization routine.

    Gradient Descent has the same step size for each parameter.  Adagrad self-
    learns a unique step size for each parameter based on accumulation of the
    square of the gradient over the course of optimization.  The update is:

        s_k = s_(k-1) + (g*g)
        x_(k+1) = x_k - α g_k / sqrt(s_k)

    The purpose of the square and square root operations is essentially to destroy
    the sign of g in the denomenator gain.  An alternative may be to simply do
    s_k = s_(k-1) + abs(g), which would have less numerical precision issues.

    Ref [1] describes a ""fully connected"" version of AdaGrad that is a cousin
    of sorts to BFGS, storing an NxN matrix.  This is intractible for large N.
    The implementation here is sister to most implementations in the wild, and
    is the "Diagonal" implementation, which stores no information about the
    relationship between "spatial" elements of the gradient vector.  Only the
    temporal relationship between the gradient and its past is stored.

    References
    ----------
    [1] Duchi, John, Hazan, Elad and Singer, Yoram. "Adaptive Subgradient
        Methods for Online Learning and Stochastic Optimization."
        https://doi.org/10.5555/1953048.2021068

    """
    def __init__(self, fg, x0, alpha):
        """Create a new AdaGrad optimizer.

        Parameters
        ----------
        fg : callable
            a function which returns (f, g) where f is the scalar cost, and
            g is the vector gradient.
        x0 : callable
            the parameter vector immediately prior to optimization
        alpha : float
            the step size

        """
        self.fg = fg
        self.x0 = x0
        self.alpha = alpha
        self.x = x0.copy()
        self.accumulator = np.zeros_like(self.x)
        self.eps = np.finfo(x0.dtype).eps
        self.iter = 0

    def step(self):
        """Perform one iteration of optimization."""
        f, g = self.fg(self.x)
        self.accumulator += (g*g)
        self.x -= self.alpha * g / np.sqrt(self.accumulator+self.eps)
        self.iter += 1
        return self.x, f, g

    def runN(self, N):
        """Perform N iterations of optimization."""
        for _ in range(N):
            yield self.step()


class RMSProp:
    """RMSProp optimization routine.

    RMSProp keeps a moving average of the squared gradient of each parameter.

    It is very similar to AdaGrad, except that the decay built into it allows
    it to forget old gradients.  This makes it often superior for non-convex
    problems, where navigation from one valley into another poisons AdaGrad, but
    RMSProp will eventually forget about the old valley.

    The update is:

        s_k = γ * s_(k-1) + (1-γ)*(g*g)
        x_(k+1) = x_k - α g_k / sqrt(s_k)

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
    def __init__(self, fg, x0, alpha, gamma=0.9):
        """Create a new RMSProp optimizer.

        Parameters
        ----------
        fg : callable
            a function which returns (f, g) where f is the scalar cost, and
            g is the vector gradient.
        x0 : callable
            the parameter vector immediately prior to optimization
        alpha : float
            the step size
        gamma : float
            the decay rate of the accumulated squared gradient

        """
        self.fg = fg
        self.x0 = x0
        self.alpha = alpha
        self.gamma = gamma
        self.x = x0.copy()
        self.accumulator = np.zeros_like(self.x)
        self.eps = np.finfo(x0.dtype).eps
        self.iter = 0

    def step(self):
        """Perform one iteration of optimization."""
        gamma = self.gamma
        f, g = self.fg(self.x)
        self.accumulator = gamma*self.accumulator + (1-gamma)*(g*g)
        self.x -= self.alpha * g / np.sqrt(self.accumulator+self.eps)
        self.iter += 1
        return self.x, f, g

    def runN(self, N):
        """Perform N iterations of optimization."""
        for _ in range(N):
            yield self.step()


class ADAM:
    """ADAM optimization routine.

    ADAM, or "Adaptive moment estimation" uses moving average estimates of the
    mean of the gradient and of its "uncentered variance".  This causes the
    algorithm to combine several properties of AdaGrad and RMSPRop, as well as
    perform a form of self-annealing, where the step size will naturally decay
    as the optimizer converges.  This can cause ADAM to recover itself after
    diverging, if the divergence is not too extreme.

    The update is:
        m = mean
        v = variance

        m_k = β_1 m_(k-1) + (1-β_1) * g
        v_k = β_2 v_(k-1) + (1-β_2) * (g*g)

        mhat_k = m_k / (1 - β_1^k)
        mhat_v = v_k / (1 - β_2^k)

        x_(k+1) = x_k - α * mhat_k / sqrt(vhat_k)

    References
    ----------
    [1] Kingma, Diederik and Ba, Jimmy. "Adam: A Method for Stochastic Optimization"
        http://arxiv.org/abs/1412.6980

    """
    def __init__(self, fg, x0, alpha, beta1=0.9, beta2=0.999):
        """Create a new ADAM optimizer.

        Parameters
        ----------
        fg : callable
            a function which returns (f, g) where f is the scalar cost, and
            g is the vector gradient.
        x0 : callable
            the parameter vector immediately prior to optimization
        alpha : float
            the step size
        beta1 : float
            the decay rate of the first moment (mean of gradient)
        beta2 : float
            the decay rate of the second moment (uncentered variance)

        """
        self.fg = fg
        self.x0 = x0
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.x = x0.copy()
        self.m = np.zeros_like(x0)
        self.v = np.zeros_like(x0)
        self.eps = np.finfo(x0.dtype).eps
        self.iter = 0

    def step(self):
        """Perform one iteration of optimization."""
        self.iter += 1
        beta1 = self.beta1
        beta2 = self.beta2
        f, g = self.fg(self.x)
        # update momentum estimates
        self.m = beta1*self.m + (1-beta1) * g
        self.v = beta2*self.v + (1-beta2) * (g*g)

        mhat = self.m / (1 - beta1**self.iter)
        vhat = self.v / (1 - beta2**self.iter)

        self.x -= self.alpha * mhat/(np.sqrt(vhat+self.eps))
        return self.x, f, g

    def runN(self, N):
        """Perform N iterations of optimization."""
        for _ in range(N):
            yield self.step()


class F77LBFGSB:
    """Limited Memory Broyden Fletcher Goldfarb Shannon optimizer, variant B (L-BFGS-B).

    L-BFGS-B is a Quasi-Newton method which uses the previous m gradient vectors
    to perform the BFGS update, which itself is an approximation of Newton's
    Method.

    The "L" in L-BFGS is Limited Memory, due to this m*n storage requirement,
    where m is a small integer (say 10 to 30), and n is the number of variables.

    At its core, L-BFGS solves the BFGS update using an adaptive line search,
    satisfying the strong Wolfe conditions, which guarantee that it does not
    move uphill.

    Variant B (BFGS-B) incorporates subspace minimization, which further
    accelerates convergence.

    Subspace minimization is the practice of forming a lower-dimensional "manifold"
    (essentially, enclosing Euclidean geometry) for the problem at a given
    iteration, and then exactly solving for the minimum of that manifold.

    The combination of subspace minimization and a quasi-newton update give
    L-BFGS-B exponential convergence, where it may converge by an order of
    magnitude in cost or more on each iteration.

    This wrapper around Jorge Nocedal's Fortran code made available through
    SciPy attenpts to defeat the built-in convergence tests of lbfgsb.f, but
    is not always successful due to the nature of floating point arithmetic.
    Unlike all other classes in this file, L-BFGS-B may refuse to step(), and
    may stop early in a runN or run_to call.  A warning will be generated in
    such instances.

    References
    ----------
    [1] Jorge Nocedal, "Updating Quasi-Newton Matricies with Limited Storage"
        https://doi.org/10.2307/2006193

    [2] Richard H. Byrd, Peihuang Lu, and Jorge Nocedal "A Limited-Memory
        Algorithm For Bound-Constrained Optimization"
        https://doi.org/10.1137/0916069

    [3] Ciyou Zhu, Richard H. Byrd, Peihuang Lu, and Jorge Nocedal "Algorithm 778:
        L-BFGS-B: Fortran subroutines for large-scale bound-constrained optimization"
        https://doi.org/10.1145/279232.279236

    [4] José Luis Morales and Jorge Nocedal, "Remark on “algorithm 778: L-BFGS-B:
        Fortran subroutines for large-scale bound constrained optimization”
        https://doi.org/10.1145/2049662.2049669

    """
    def __init__(self, fg, x0, memory=10, lower_bounds=None, upper_bounds=None):
        """Create a new L-BFGS-B optimizer.

        Parameters
        ----------
        fg : callable
            a function which returns (f, g) where f is the scalar cost, and
            g is the vector gradient.
        x0 : callable
            the parameter vector immediately prior to optimization
        memory : int
            the number of recent gradient vectors to use in performing the
            approximate Newton's step
        lower_bounds : numpy.ndarray, optional
            vector of same size as x0 containing the hard lower bounds for the
            variables; if None, unconstrained lb
        upper_bounds : numpy.ndarray, optional
            vector of same size as x0 containing the hard upper bounds for the
            variables; if None, unconstrained ub

        """
        self.fg = fg
        self.x0 = x0
        self.n = len(x0)  # n = n vars
        self.m = memory

        # create the work arrays Fortran needs
        fint_dtype = _lbfgsb.types.intvar.dtype
#         ffloat_dtype = x0.dtype  maybe can uncomment this someday, but probably not.
        ffloat_dtype = np.float64

        # todo: f77 code explodes for f32 dtype?
        if lower_bounds is None:
            lower_bounds = np.full(self.n, -np.Inf, dtype=ffloat_dtype)

        if upper_bounds is None:
            upper_bounds = np.full(self.n, np.Inf, dtype=ffloat_dtype)

        # nbd is an array of integers for Fortran
        #         nbd(i)=0 if x(i) is unbounded,
        #                1 if x(i) has only a lower bound,
        #                2 if x(i) has both lower and upper bounds, and
        #                3 if x(i) has only an upper bound.
        nbd = np.zeros(self.n, dtype=fint_dtype)
        self.l = lower_bounds  # NOQA
        self.u = upper_bounds
        finite_lower_bound = np.isfinite(self.l)
        finite_upper_bound = np.isfinite(self.u)
        # unbounded case handled in init as zeros
        lower_but_not_upper_bound = finite_lower_bound & ~finite_upper_bound
        upper_but_not_lower_bound = finite_upper_bound & ~finite_lower_bound
        both_bounds = finite_lower_bound & finite_upper_bound
        nbd[lower_but_not_upper_bound] = 1
        nbd[both_bounds]               = 2  # NOQA
        nbd[upper_but_not_lower_bound] = 3
        self.nbd = nbd

        # much less complicated initializations
        m, n = self.m, self.n
        self.x = x0.copy()
        self.f = np.array([0], dtype=ffloat_dtype)
        self.g = np.zeros([self.n], dtype=ffloat_dtype)
        # see lbfgsb.f for this size
        # error in the docstring, see line 240 to 252
        self.wa = np.zeros(2 * m * n + 11 * m ** 2 + 5 * n + 8 * m, dtype=ffloat_dtype)
        self.iwa = np.zeros(3*n, dtype=fint_dtype)
        self.task = np.zeros(1, dtype='S60')  # S60 = <= 60 character wide byte array
        self.csave = np.zeros(1, dtype='S60')
        self.lsave = np.zeros(4, dtype=fint_dtype)
        self.isave = np.zeros(44, dtype=fint_dtype)
        self.dsave = np.zeros(29, dtype=ffloat_dtype)
        self.task[:] = 'START'

        self.iter = 0

        # try to prevent F77 driver from ever stopping on its own
        # cannot use NaN or Inf, Fortran comparisons do not work
        # properly, so pick unreasonably small numbers.
        # TODO: would a negative number be better here?
        self.factr = 1e-999
        self.pgtol = 1e-999

        # other stuff to be added to the interface later
        self.maxls = 30
        self.iprint = 1

    def _call_fortran(self):
        _lbfgsb.setulb(self.m, self.x, self.l, self.u, self.nbd, self.f, self.g,
                       self.factr, self.pgtol, self.wa, self.iwa, self.task, self.iprint,
                       self.csave, self.lsave, self.isave, self.dsave, self.maxls)

    def _view_s(self):
        m, n = self.m, self.n
        # flat => matrix storage => truncate to only valid rows
        return self.wa[0:m*n].reshape(m, n)[:self._valid_space_sy]

    def _view_y(self):
        m, n = self.m, self.n
        # flat => matrix storage => truncate to only valid rows
        return self.wa[m*n:2*m*n].reshape(m, n)[:self._valid_space_sy]

    @property
    def _nbfgs_updates(self):
        return self.isave[30]

    @property
    def _valid_space_sy(self):
        return min(self._nbfgs_updates, self.m)

    def step(self):
        """Perform one iteration of optimization."""
        self.iter += 1  # increment first so that while loop is self-breaking
        while self._nbfgs_updates < self.iter:
            # call F77 mutates all of the class's state
            self._call_fortran()
            # strip null bytes/termination and any ASCII white space
            task = self.task.tobytes().strip(b'\x00').strip()
            if task.startswith(b'FG'):
                f, g = self.fg(self.x)
                if g.ndim != 1:
                    g = g.ravel()

                self.f[:] = f
                self.g[:] = g
                self._call_fortran()

            if _fortran_died(task):
                msg = task.decode('UTF-8')
                raise ValueError("the Fortran L-BFGS-B driver thinks something is wrong with the problem and gave the message " + msg)

            if _fortran_converged(task):
                raise StopIteration

            if _fortran_major_iter_complete(task):
                break

        return self.x, self.f, self.g

    def runN(self, N):
        """Perform N iterations of optimization."""
        for i in range(N):
            try:
                yield self.step()
            except StopIteration:
                warnings.warn(f'L-BFGS-B can make no further progress; performed {i}/N iterations')
                break

    def run_to(self, N):
        """Run the optimizer until its iteration count equals N."""
        while self.iter < N:
            try:
                yield self.step()
            except StopIteration:
                warnings.warn(f'L-BFGS-B can make no further progress; stopped on iteration {self.iter}/N iterations')
                break


def _fortran_died(task):
    return task.startswith(b'STOP')


def _fortran_converged(task):
    return task.startswith(b'CONV')


def _fortran_major_iter_complete(task):
    return task.startswith(b'NEW_X')
