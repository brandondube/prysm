"""L-BFGS-B optimizer, wrapper around low-level scipy interface."""

import warnings

import numpy as np  # can't be our shim since this interacts with C
import scipy
from scipy.optimize import _lbfgsb

from .problem import as_problem


def _scipy_has_c_lbfgsb():
    """True if scipy ships the C port of L-BFGS-B (>= 1.15).

    SciPy 1.15 replaced the Fortran 77 L-BFGS-B driver with a C port.  The
    setulb signature changed: csave and iprint were dropped, an ln_task
    output array was added, and task is now an int array (not a 60-byte
    char array) whose first element encodes the request code.
    """
    parts = scipy.__version__.split(".")
    try:
        major, minor = int(parts[0]), int(parts[1])
    except (ValueError, IndexError):
        return False
    return (major, minor) >= (1, 15)


# Task codes returned by scipy >= 1.15's C L-BFGS-B driver in task[0].
# See scipy.optimize._lbfgsb_py.task_messages.
_C_TASK_NEW_X = 1
_C_TASK_FG = 3
_C_TASK_CONVERGENCE = 4
_C_TASK_STOP = 5


class _LBFGSBBase:
    """Shared scaffolding for L-BFGS-B drivers (F77 and C ports).

    Holds the part of the L-BFGS-B state that is identical between the two
    drivers — problem, bounds, nbd, x, g, wa, iwa,
    lsave/isave/dsave, the magic factr=pgtol=0 settings to defeat
    early convergence, and the step / run_to orchestration.

    The pieces that differ between the F77 and C drivers are factored out
    into a small set of hooks that subclasses implement:

    * _int_dtype() — integer dtype for the work arrays.
    * _init_driver_state() — allocate driver-specific buffers (f,
      task, csave/ln_task as applicable, plus the initial
      'START' poke for F77).
    * _call_driver() — one invocation of setulb with the
      driver-appropriate argument list.
    * _decode_task() — return one of 'FG', 'NEW_X',
      'CONVERGENCE', 'STOP', or None.
    * _store_fg(f, g) — write f and g into the driver's buffers.
    * _read_f() — current f as a Python float.
    * _task_diagnostic() — human-readable task string for error messages.

    """

    def __init__(self, fg, x0, memory=10, lower_bounds=None, upper_bounds=None):
        """Create a new L-BFGS-B optimizer.

        Parameters
        ----------
        fg : callable or Problem
            either fg(x) -> (f, g) or a Problem-shaped object; see
            :func:`as_problem`.
        x0 : ndarray
            the parameter vector immediately prior to optimization.
        memory : int
            number of recent gradient vectors used for the approximate
            Newton step (typical 10–30).
        lower_bounds, upper_bounds : ndarray, optional
            hard bounds per variable; None is unconstrained.

        """
        self.problem = as_problem(fg)
        self.x0 = x0
        self.n = len(x0)
        self.m = memory

        # f77 driver explodes for f32 dtype — keep both drivers on float64
        ffloat_dtype = np.float64
        int_dtype = self._int_dtype()  # NOQA - defined by subclasses.

        if lower_bounds is None:
            lower_bounds = np.full(self.n, -np.inf, dtype=ffloat_dtype)
        if upper_bounds is None:
            upper_bounds = np.full(self.n, np.inf, dtype=ffloat_dtype)

        # nbd[i] = 0 unbounded, 1 lb only, 2 both, 3 ub only
        nbd = np.zeros(self.n, dtype=int_dtype)
        self.l = lower_bounds  # NOQA
        self.u = upper_bounds
        finite_l = np.isfinite(self.l)
        finite_u = np.isfinite(self.u)
        nbd[finite_l & ~finite_u] = 1
        nbd[finite_l & finite_u] = 2  # NOQA
        nbd[~finite_l & finite_u] = 3
        self.nbd = nbd

        m, n = self.m, self.n
        self._x = x0.copy()
        self._g = np.zeros(n, dtype=ffloat_dtype)
        # wa sizing per lbfgsb.f (11*m**2 == 11*m*m)
        self.wa = np.zeros(2 * m * n + 11 * m * m + 5 * n + 8 * m, dtype=ffloat_dtype)
        self.iwa = np.zeros(3 * n, dtype=int_dtype)
        self.lsave = np.zeros(4, dtype=int_dtype)
        self.isave = np.zeros(44, dtype=int_dtype)
        self.dsave = np.zeros(29, dtype=ffloat_dtype)

        self.iter = 0
        self.nfev = 0
        self.last_step_metadata = {}
        self._last_eval_x = None
        self._last_eval_f = None
        self._last_eval_g = None
        # factr<0 makes F77 core-dump; the 0 here neutralizes the built-in
        # convergence tests so the user controls termination.
        self.factr = 0
        self.pgtol = 0
        self.maxls = 30

        self._init_driver_state()  # NOQA - defined by subclasses.

    @property
    def x(self):
        """Current iterate as a snapshot, not the driver's mutable buffer."""
        return self._x.copy()

    @property
    def g(self):
        """Current iterate as a snapshot, not the driver's mutable buffer."""
        return self._g.copy()

    def _view_s(self):
        m, n = self.m, self.n
        return self.wa[0 : m * n].reshape(m, n)[: self._valid_space_sy]

    def _view_y(self):
        m, n = self.m, self.n
        return self.wa[m * n : 2 * m * n].reshape(m, n)[: self._valid_space_sy]

    @property
    def _nbfgs_updates(self):
        # The driver's internal BFGS-update counter.  The slot it lives in
        # has shifted between scipy versions (F77 → C port re-laid isave),
        # so do not use this for control flow — count NEW_X events in
        # step() instead.  Kept for read-only introspection of the
        # raw driver state.
        return self.isave[30]

    @property
    def _valid_space_sy(self):
        return min(self._nbfgs_updates, self.m)

    def _record_eval(self, x, f, g):
        self._last_eval_x = x.copy()
        self._last_eval_f = float(f)
        self._last_eval_g = g.copy()

    def _cached_eval_matches(self, x):
        if self._last_eval_x is None:
            return False
        return np.array_equal(self._last_eval_x, x)

    def step(self):
        """Perform one iteration of optimization.

        Drives the underlying setulb call sequence until it signals a
        completed iteration (NEW_X), termination (CONVERGENCE), or an
        error (STOP).  Each NEW_X corresponds to one outer iteration.
        """
        x_start = self._x.copy()
        f_start = None
        g_start = None

        if self._cached_eval_matches(x_start):
            f_start = self._last_eval_f
            g_start = self._last_eval_g.copy()

        # setulb's task state-machine fires several FG requests during
        # its line search before declaring NEW_X.  Loop until that
        # happens; bail on CONVERGENCE/STOP.
        while True:
            self._call_driver()  # NOQA - defined by subclasses.
            task = self._decode_task()  # NOQA - defined by subclasses.
            if task == "FG":
                f, g = self.problem.fg(self._x.copy())
                self.nfev += 1
                if g.ndim != 1:
                    g = g.ravel()
                self._record_eval(self._x, f, g)
                if np.array_equal(self._x, x_start):
                    f_start = float(f)
                    g_start = g.copy()
                self._store_fg(f, g)  # NOQA - defined by subclasses.
                continue

            if task == "STOP":
                raise ValueError(
                    "L-BFGS-B driver thinks something is wrong with the "
                    f"problem; task: {self._task_diagnostic()}"
                )  # NOQA - defined by subclasses.
            if task == "CONVERGENCE":
                raise StopIteration
            if task == "NEW_X":
                break

            raise RuntimeError(
                f"L-BFGS-B driver returned an unknown task: {self._task_diagnostic()}"
            )

        self.iter += 1

        if f_start is None:
            f_start, g_start = self.problem.fg(x_start.copy())
            self.nfev += 1
            if g_start.ndim != 1:
                g_start = g_start.ravel()
            self._record_eval(x_start, f_start, g_start)

        self.last_step_metadata = {
            "task": self._task_diagnostic(),
            "nbfgs_updates": int(self._nbfgs_updates),
        }
        return x_start.copy(), float(f_start), g_start.copy()

    def run_to(self, N):
        """Run the optimizer until its iteration count equals N."""
        while self.iter < N:
            try:
                yield self.step()
            except StopIteration:
                warnings.warn(
                    f"L-BFGS-B can make no further progress; stopped on iteration {self.iter}/N iterations"
                )
                break


class F77LBFGSB(_LBFGSBBase):
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

    def _int_dtype(self):
        return _lbfgsb.types.intvar.dtype

    def _init_driver_state(self):
        self.f = np.array([0], dtype=np.float64)
        self.task = np.zeros(1, dtype="S60")  # 60-byte ASCII task string
        self.csave = np.zeros(1, dtype="S60")
        self.task[:] = "START"
        self.iprint = 0

    def _call_driver(self):
        _lbfgsb.setulb(
            self.m,
            self._x,
            self.l,
            self.u,
            self.nbd,
            self.f,
            self._g,
            self.factr,
            self.pgtol,
            self.wa,
            self.iwa,
            self.task,
            self.iprint,
            self.csave,
            self.lsave,
            self.isave,
            self.dsave,
            self.maxls,
        )

    def _decode_task(self):
        task = self.task.tobytes().strip(b"\x00").strip()
        if task.startswith(b"FG"):
            return "FG"
        if task.startswith(b"NEW_X"):
            return "NEW_X"
        if task.startswith(b"CONV"):
            return "CONVERGENCE"
        if task.startswith(b"STOP"):
            return "STOP"
        return None

    def _store_fg(self, f, g):
        self.f[:] = f
        self._g[:] = g

    def _read_f(self):
        return float(self.f[0])

    def _task_diagnostic(self):
        return self.task.tobytes().strip(b"\x00").strip().decode("UTF-8")


class CLBFGSB(_LBFGSBBase):
    """L-BFGS-B optimizer using the SciPy >= 1.15 C driver.

    Behaves identically to :class:`F77LBFGSB` from the user's perspective,
    but targets the C port of L-BFGS-B that replaced the Fortran 77 driver
    in SciPy 1.15.  The differences vs. the Fortran driver are internal:

    * csave and iprint are no longer arguments to setulb.
    * A new two-element integer ln_task workspace array is required.
    * task is a length-2 integer array; task[0] carries the request
      code (3=FG, 1=NEW_X, 4=CONVERGENCE, 5=STOP).

    See :class:`F77LBFGSB` for parameter documentation; this class accepts
    the same arguments and exposes the same step/run_to interface.
    """

    def _int_dtype(self):
        # mirror the C driver's dtype choice (depends on ILP64 LAPACK)
        try:
            from scipy.linalg.lapack import HAS_ILP64

            return np.int64 if HAS_ILP64 else np.int32
        except ImportError:
            return np.int32

    def _init_driver_state(self):
        int_dtype = self.nbd.dtype  # match the base's choice
        # f is a 0-d float64 array in the C interface, not a 1-element array
        self.f = np.array(0.0, dtype=np.float64)
        self.task = np.zeros(2, dtype=int_dtype)
        self.ln_task = np.zeros(2, dtype=int_dtype)

    def _call_driver(self):
        _lbfgsb.setulb(
            self.m,
            self._x,
            self.l,
            self.u,
            self.nbd,
            self.f,
            self._g,
            self.factr,
            self.pgtol,
            self.wa,
            self.iwa,
            self.task,
            self.lsave,
            self.isave,
            self.dsave,
            self.maxls,
            self.ln_task,
        )

    def _decode_task(self):
        code = int(self.task[0])
        if code == _C_TASK_FG:
            return "FG"
        if code == _C_TASK_NEW_X:
            return "NEW_X"
        if code == _C_TASK_CONVERGENCE:
            return "CONVERGENCE"
        if code == _C_TASK_STOP:
            return "STOP"
        return None

    def _store_fg(self, f, g):
        self.f[...] = f
        self._g[:] = g

    def _read_f(self):
        return float(self.f)

    def _task_diagnostic(self):
        return f"code {int(self.task[0])}"


# Public alias — picks the right driver for the installed SciPy.
# F77LBFGSB and CLBFGSB remain available for callers who want a specific one.
LBFGSB = CLBFGSB if _scipy_has_c_lbfgsb() else F77LBFGSB
