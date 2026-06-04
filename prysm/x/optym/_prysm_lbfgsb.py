"""L-BFGS-B (Byrd-Lu-Nocedal-Zhu 1995 + Morales-Nocedal 2011), pure-Python,
backend-shim friendly.

Compact L-BFGS representation, generalized Cauchy point, primal direct
subspace minimization with the Morales-Nocedal 2011 projection refinement.
All vector and matrix arithmetic flows through prysm.mathops.np so the same
code runs on numpy, cupy, or any other shim-compatible backend.  Scalars
from the Wolfe line search live on the host; an n-sized iterate, n-sized
gradient, m*n history, and 2m-sized intermediates live on the backend.

Behaves like every other class in optimizers.py: step returns
(x_pre, f_pre, g_pre); the only StopIteration it raises is when the line
search reports no acceptable step length.  Termination otherwise is the
governor's responsibility.
"""
import math

from prysm.mathops import linalg, np

from .problem import as_problem


class _OptimizerStop:
    """StopIteration payload consumed by run_until."""

    __slots__ = ('success', 'message')

    def __init__(self, success, message):
        self.success = bool(success)
        self.message = message


def _wolfe_eval(problem, xk, pk, alpha, dot):
    """Evaluate f, directional derivative, and gradient at xk + alpha*pk."""
    f, g = problem.fg(xk + alpha * pk)
    if g.ndim != 1:
        g = g.ravel()
    return float(f), float(dot(g, pk)), g


def _wolfe_quadmin(a, fa, fpa, b, fb):
    """Quadratic interpolant minimizer for scalar Wolfe zoom."""
    try:
        db = b - a
        B = (fb - fa - fpa * db) / (db * db)
        xmin = a - fpa / (2.0 * B)
    except ArithmeticError:
        return None
    if not math.isfinite(xmin):
        return None
    return xmin


def _wolfe_cubicmin(a, fa, fpa, b, fb, c, fc):
    """Cubic interpolant minimizer for scalar Wolfe zoom."""
    try:
        db = b - a
        dc = c - a
        denom = (db * dc) ** 2 * (db - dc)
        rhs0 = fb - fa - fpa * db
        rhs1 = fc - fa - fpa * dc
        A = (dc * dc * rhs0 - db * db * rhs1) / denom
        B = (-dc ** 3 * rhs0 + db ** 3 * rhs1) / denom
        radical = B * B - 3.0 * A * fpa
        xmin = a + (-B + math.sqrt(radical)) / (3.0 * A)
    except (ArithmeticError, ValueError):
        return None
    if not math.isfinite(xmin):
        return None
    return xmin


def _strong_wolfe_zoom(
    problem, xk, pk,
    a_lo, a_hi,
    phi_lo, phi_hi,
    derphi_lo,
    phi0, derphi0,
    c1, c2, dot,
):
    """Zoom stage for the optimizer-local strong-Wolfe line search."""
    maxiter = 10
    delta1 = 0.2
    delta2 = 0.1
    phi_rec = phi0
    a_rec = 0.0

    for i in range(maxiter + 1):
        dalpha = a_hi - a_lo
        if dalpha < 0:
            a, b = a_hi, a_lo
        else:
            a, b = a_lo, a_hi

        a_j = None
        if i > 0:
            cchk = delta1 * dalpha
            a_j = _wolfe_cubicmin(
                a_lo, phi_lo, derphi_lo,
                a_hi, phi_hi,
                a_rec, phi_rec,
            )
            if a_j is not None and (a_j > b - cchk or a_j < a + cchk):
                a_j = None

        if a_j is None:
            qchk = delta2 * dalpha
            a_j = _wolfe_quadmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi)
            if a_j is None or a_j > b - qchk or a_j < a + qchk:
                a_j = a_lo + 0.5 * dalpha

        phi_aj, derphi_aj, g_aj = _wolfe_eval(problem, xk, pk, a_j, dot)
        if (phi_aj > phi0 + c1 * a_j * derphi0) or (phi_aj >= phi_lo):
            phi_rec = phi_hi
            a_rec = a_hi
            a_hi = a_j
            phi_hi = phi_aj
            continue

        if abs(derphi_aj) <= -c2 * derphi0:
            return a_j, phi_aj, derphi_aj, g_aj

        if derphi_aj * (a_hi - a_lo) >= 0:
            phi_rec = phi_hi
            a_rec = a_hi
            a_hi = a_lo
            phi_hi = phi_lo
        else:
            phi_rec = phi_lo
            a_rec = a_lo
        a_lo = a_j
        phi_lo = phi_aj
        derphi_lo = derphi_aj

    return None, None, None, None


def _strong_wolfe_lean(
    problem, xk, pk,
    fg_at_xk,
    maxalpha=None,
    c1=1e-4,
    c2=0.9,
    maxiter=10,
):
    """Strong-Wolfe line search specialized for PrysmLBFGSB.step().

    This is intentionally internal: it assumes the caller already owns a
    Problem object and precomputed (f, g) at xk.  Keeping the bracket and
    zoom state explicit avoids per-step closures and cache containers in
    the public line-search helper while preserving the same returned
    accepted gradient.
    """
    fk, gk = fg_at_xk
    dot = np.dot
    phi0 = float(fk)
    derphi0 = float(dot(gk, pk))

    alpha0 = 0.0
    alpha1 = 1.0
    if maxalpha is not None:
        alpha1 = min(alpha1, maxalpha)

    phi_a0 = phi0
    derphi_a0 = derphi0
    phi_a1, derphi_a1, g_a1 = _wolfe_eval(problem, xk, pk, alpha1, dot)

    for i in range(maxiter):
        if alpha1 == 0 or (maxalpha is not None and alpha0 == maxalpha):
            break

        if (phi_a1 > phi0 + c1 * alpha1 * derphi0) or (i > 0 and phi_a1 >= phi_a0):
            return _strong_wolfe_zoom(
                problem, xk, pk,
                alpha0, alpha1,
                phi_a0, phi_a1,
                derphi_a0,
                phi0, derphi0,
                c1, c2, dot,
            )

        if abs(derphi_a1) <= -c2 * derphi0:
            return alpha1, phi_a1, derphi_a1, g_a1

        if derphi_a1 >= 0:
            return _strong_wolfe_zoom(
                problem, xk, pk,
                alpha1, alpha0,
                phi_a1, phi_a0,
                derphi_a1,
                phi0, derphi0,
                c1, c2, dot,
            )

        alpha2 = 2.0 * alpha1
        if maxalpha is not None:
            alpha2 = min(alpha2, maxalpha)

        alpha0 = alpha1
        alpha1 = alpha2
        phi_a0 = phi_a1
        derphi_a0 = derphi_a1
        phi_a1, derphi_a1, g_a1 = _wolfe_eval(problem, xk, pk, alpha1, dot)

    return None, None, None, None


class PrysmLBFGSB:
    """L-BFGS-B optimizer.

    Limited-memory BFGS with box constraints.  Implements the algorithm of
    Byrd, Lu, Nocedal, and Zhu, "A Limited-Memory Algorithm for Bound
    Constrained Optimization", SIAM J. Sci. Comput. 16(5), 1995, with the
    subspace-minimization refinement of Morales and Nocedal, "Remark on
    Algorithm 778", ACM Trans. Math. Softw. 38(1), 2011: rather than
    truncating the path from x_k to the unconstrained subspace minimizer
    x_hat at the first bound it hits, the iterate is taken to be the
    componentwise projection of x_hat into the box whenever that direction
    is descent, falling back to BLNZ truncation only when projection
    yields an ascent direction.

    Choose this over the scipy-backed LBFGSB when any of: (1) the
    objective lives on a non-numpy backend (cupy, pytorch) where the
    scipy Fortran wrapper would force a host round-trip per fg call;
    (2) you want to run at config.precision = 32, which the scipy
    driver refuses; (3) you want the governor to drive termination
    without inheriting the scipy task-string state machine.  The pure-
    Python implementation closes the gap with the scipy driver as n grows
    in the Rosenbrock CPU benchmark, but remains slower for tiny problems
    due to Python per-call overhead.

    Parameters
    ----------
    fg : callable or Problem
        either fg(x) -> (f, g) or a Problem-shaped object; see
        as_problem.
    x0 : ndarray
        the parameter vector immediately prior to optimization.  Its dtype
        sets the dtype of every internal array.
    memory : int
        number of recent (s, y) pairs retained for the L-BFGS approximation
        (typical 10-30).
    lower_bounds, upper_bounds : ndarray, optional
        per-variable hard bounds.  None means unconstrained on that side;
        +/-inf entries in an explicit array work the same way.
    c1, c2 : float
        Wolfe constants for the line search.
    maxls : int
        bracket-phase iteration cap for the line search.

    """

    def __init__(self, fg, x0, memory=10,
                 lower_bounds=None, upper_bounds=None,
                 c1=1e-4, c2=0.9, maxls=30):
        self.problem = as_problem(fg)
        self.x0 = x0
        self.x = x0.copy()
        self.n = x0.size
        self.m = memory

        dtype = x0.dtype
        self._eps = float(np.finfo(dtype).eps)  # curvature-condition floor
        if lower_bounds is None:
            lower_bounds = np.full(self.n, -np.inf, dtype=dtype)
        if upper_bounds is None:
            upper_bounds = np.full(self.n, np.inf, dtype=dtype)
        self.l = lower_bounds  # NOQA - mirrors the math
        self.u = upper_bounds
        # When no coord has a finite bound, the Cauchy point + subspace
        # solve reduces to the plain L-BFGS step -H_k g.  Skip both and
        # save one _W + one _M build per step.
        self._has_bounds = bool(
            np.any(np.isfinite(lower_bounds)) or np.any(np.isfinite(upper_bounds))
        )

        # history: each row is one (s, y) pair; ys[i] caches s_i . y_i.
        self.S = np.zeros((memory, self.n), dtype=dtype)
        self.Y = np.zeros((memory, self.n), dtype=dtype)
        self.ys = np.zeros(memory, dtype=dtype)
        self.rho = np.zeros(memory, dtype=dtype)
        self._alphas = np.empty(memory, dtype=dtype)
        self._k = 0  # valid history count, capped at memory
        self._hist_next = 0  # next slot to overwrite once history is full

        # theta = (y_{k-1} . y_{k-1}) / (s_{k-1} . y_{k-1}); 1 before any
        # update so the initial Hessian approximation is the identity.
        self.theta = np.asarray(1, dtype=dtype)

        self.c1 = c1
        self.c2 = c2
        self.maxls = maxls

        self.iter = 0
        self.nfev = 0
        self.last_step_metadata = {}
        # number of coords driven to a bound by the last _cauchy sweep;
        # set each call, read by step() for telemetry without a reduction.
        self._cauchy_breaks = 0
        # Reusable internal work arrays.  These are deliberately lazy:
        # unbounded problems never allocate the large bounded-path scratch,
        # and bounded problems grow each buffer only to the largest shape
        # reached so far.
        self._scratch = {}

    def _scratch_array(self, name, shape, dtype=None):
        """Return a reusable scratch array of exactly shape.

        Buffers grow but never shrink: when the cached array is larger than
        requested a same-rank, same-dtype slice view is returned.  The common
        case (exact match, or a fresh allocation) returns the buffer itself,
        avoiding a per-call view object in the inner loop.
        """
        if dtype is None:
            dtype = self.x.dtype
        shape = tuple(shape)

        arr = self._scratch.get(name)
        if (arr is None or arr.dtype != dtype or arr.ndim != len(shape)
                or any(h < n for h, n in zip(arr.shape, shape))):
            arr = np.empty(shape, dtype=dtype)
            self._scratch[name] = arr
            return arr

        if arr.shape == shape:
            return arr
        return arr[tuple(slice(0, n) for n in shape)]

    # ---------------- compact L-BFGS representation ----------------
    #
    # After k history pairs (s_0, y_0), ..., (s_{k-1}, y_{k-1}), the
    # L-BFGS Hessian approximation has the compact form
    #
    #     B_k = θ_k I − W_k M_k⁻¹ W_kᵀ                            (1)
    #
    # with
    #
    #     W_k = [ Y_k,  θ_k S_k ]      shape (n, 2k)
    #     M_k = [ −D_k       L_kᵀ           ]                   shape (2k, 2k)
    #           [  L_k    θ_k S_kᵀ S_k     ]
    #     D_k = diag(s_0·y_0, ..., s_{k-1}·y_{k-1})
    #     L_k = strict lower triangular of S_kᵀ Y_k
    #     θ_k = (y_{k-1}·y_{k-1}) / (s_{k-1}·y_{k-1})
    #
    # See Byrd, Nocedal, Schnabel, "Representations of quasi-Newton
    # matrices and their use in limited memory methods", Math. Prog.
    # 63 (1994).
    #
    # Storage convention: self.S, self.Y are (memory, n) row-major, so
    # the paper's S_k is self.S[:k].T and likewise for Y_k.
    #
    # The inverse H_k = B_k⁻¹ comes from Sherman–Morrison–Woodbury
    # applied to (1):
    #
    #     H_k g = (1/θ) g  −  (1/θ²) W_k N_k⁻¹ W_kᵀ g            (2)
    #     N_k   = −M_k + (1/θ_k) W_kᵀ W_k                      shape (2k, 2k)
    #
    # so a single 2k×2k solve gives the direction.

    def _hist_slot(self, i):
        """Physical row for logical history index i, ordered oldest-newest."""
        if self._k < self.m:
            return i
        return (self._hist_next + i) % self.m

    def _ordered_rows(self, rows):
        """History rows in oldest-newest order, using at most two chunks."""
        k = self._k
        if k < self.m or self._hist_next == 0:
            return rows[:k]

        i = self._hist_next
        return np.concatenate([rows[i:], rows[:i]], axis=0)

    def _ordered_rows_work(self, rows, name):
        """History rows in oldest-newest order, copying only after wrap."""
        k = self._k
        if k < self.m or self._hist_next == 0:
            return rows[:k]

        out = self._scratch_array(name, (k,) + rows.shape[1:], rows.dtype)
        i = self._hist_next
        n_tail = self.m - i
        out[:n_tail] = rows[i:]
        out[n_tail:] = rows[:i]
        return out

    def _W(self):
        """The (n, 2k) compact matrix W_k = [Y_k, θ_k S_k]."""
        k = self._k
        W = self._scratch_array('_compact_W', (self.n, 2 * k), self.x.dtype)
        Y = self._ordered_rows_work(self.Y, '_ordered_Y')
        S = self._ordered_rows_work(self.S, '_ordered_S')
        W[:, :k] = Y.T
        np.multiply(S.T, self.theta, out=W[:, k:])
        return W

    def _M(self):
        """The (2k, 2k) middle matrix from the compact representation."""
        k = self._k
        dtype = self.x.dtype
        S = self._ordered_rows_work(self.S, '_ordered_S')
        Y = self._ordered_rows_work(self.Y, '_ordered_Y')
        ys = self._ordered_rows_work(self.ys, '_ordered_ys')
        SY = self._scratch_array('_compact_SY', (k, k), dtype)
        SS = self._scratch_array('_compact_SS', (k, k), dtype)
        M = self._scratch_array('_compact_M', (2 * k, 2 * k), dtype)

        np.matmul(S, Y.T, out=SY)  # (i,j) = s_i . y_j
        np.matmul(S, S.T, out=SS)  # (i,j) = s_i . s_j
        M.fill(0)
        diag_idx = self._scratch_array('_compact_diag_idx', (k,), int)
        diag_idx[:] = np.arange(k)
        M[diag_idx, diag_idx] = -ys
        # strict lower triangular of S Y^T, filled by columns/rows.  k is the
        # small L-BFGS memory parameter, so this avoids full k-by-k temporaries.
        for i in range(1, k):
            row = SY[i, :i]
            M[k + i, :i] = row
            M[:i, k + i] = row
        np.multiply(SS, self.theta, out=M[k:, k:])
        return M

    def _Bv(self, v):
        """Apply B_k v using the compact form."""
        if self._k == 0:
            return self.theta * v
        W = self._W()
        M = self._M()
        return self.theta * v - W @ np.linalg.solve(M, W.T @ v)

    def _Hg(self, g):
        """Return the L-BFGS descent direction −H_k g.

        Uses the Sherman–Morrison–Woodbury formula on the compact form
        of B_k; one (2k, 2k) solve per call.
        """
        if self._k == 0:
            return -g
        W = self._W()
        M = self._M()
        theta = self.theta
        WtW = W.T @ W
        N = -M + WtW / theta
        Wtg = W.T @ g
        Ninv_Wtg = np.linalg.solve(N, Wtg)
        Hg = g / theta - (W @ Ninv_Wtg) / (theta * theta)
        return -Hg

    # ---------------- generalized Cauchy point ----------------
    #
    # The projected gradient path is
    #
    #     x(t) = P(x − t g, l, u)
    #
    # where P clamps componentwise into the box.  This path is piecewise
    # linear; the model
    #
    #     m(z) = f + gᵀ(z − x) + ½ (z − x)ᵀ B (z − x)
    #
    # restricted to the path is piecewise quadratic in t.  The Cauchy
    # point is the first local minimizer.
    #
    # Breakpoints: t_i is the time at which coord i first hits a bound.
    #     g_i > 0, l_i finite  ⇒  t_i = (x_i − l_i) / g_i
    #     g_i < 0, u_i finite  ⇒  t_i = (x_i − u_i) / g_i
    #     otherwise            ⇒  t_i = ∞
    #
    # Sorting the breakpoints gives consecutive linear segments.  On the
    # j-th segment the direction is d_j (= −g on free coords, 0 on
    # already-active coords); the model derivatives along the path are
    #
    #     m'(t)  = gᵀd_j + (x(t) − x)ᵀ B d_j        # linear in t
    #     m''(t) = d_jᵀ B d_j                       # constant on segment
    #
    # We track these incrementally with two backend auxiliaries:
    #
    #     c = Wᵀ (x(t) − x)            advances as  c += Δt · p
    #     p = Wᵀ d                     advances as  p += g_b · w_b
    #                                  at each breakpoint, where w_b = W[b, :]
    #
    # The jump in (m', m'') across the b-th breakpoint comes from
    # d_{j+1} = d_j + g_b e_b (zeroing the active component is a rank-one
    # change to d).  Expanding (d_{j+1})ᵀ B (d_{j+1}) − d_jᵀ B d_j with
    # B = θI − W M⁻¹ Wᵀ gives the update formulas implemented below.
    # See BLNZ §4 (Algorithm CP) for the derivation.

    def _compute_breakpoints(self, g):
        """Return (t, sorted_indices) for the projected gradient path."""
        x = self.x
        l = self.l
        u = self.u
        dtype = x.dtype
        t = self._scratch_array('_break_t', (self.n,), dtype)
        work = self._scratch_array('_break_work', (self.n,), dtype)
        pos_g = self._scratch_array('_break_pos_g', (self.n,), bool)
        neg_g = self._scratch_array('_break_neg_g', (self.n,), bool)
        hit = self._scratch_array('_break_hit', (self.n,), bool)

        t.fill(np.inf)

        np.greater(g, 0, out=pos_g)
        np.less(g, 0, out=neg_g)

        np.isfinite(l, out=hit)
        np.logical_and(hit, pos_g, out=hit)
        np.subtract(x, l, out=work)
        np.divide(work, g, out=work, where=hit)
        np.copyto(t, work, where=hit)

        np.isfinite(u, out=hit)
        np.logical_and(hit, neg_g, out=hit)
        np.subtract(x, u, out=work)
        np.divide(work, g, out=work, where=hit)
        np.copyto(t, work, where=hit)

        sorted_indices = np.argsort(t)
        return t, sorted_indices

    def _next_dt_min(self, df, ddf):
        """Trial step length to the model's minimum along the current d.

        Returns +inf when the model is non-convex along d (so the loop
        advances to the next breakpoint), 0 when the gradient already
        points uphill, and −df/ddf otherwise.
        """
        if df >= 0:
            return 0.0
        if ddf <= 0:
            return np.inf
        return -df / ddf

    def _next_dt_min_vec(self, df, ddf):
        """Vectorized form of _next_dt_min.  Works element-wise on
        arrays of df and ddf; same branching semantics applied
        coordinate by coordinate.
        """
        out = self._scratch_array('_cauchy_dt_min_at_iter', df.shape, df.dtype)
        safe_ddf = self._scratch_array('_cauchy_safe_ddf', df.shape, df.dtype)
        mask = self._scratch_array('_cauchy_dt_min_mask', df.shape, bool)

        safe_ddf[:] = ddf
        np.less_equal(ddf, 0, out=mask)
        np.copyto(safe_ddf, 1.0, where=mask)

        np.divide(df, safe_ddf, out=out)
        out *= -1
        np.copyto(out, np.inf, where=mask)

        np.greater_equal(df, 0, out=mask)
        np.copyto(out, 0.0, where=mask)
        return out

    def _cauchy(self, g, W=None, M=None, M_lu=None):
        """Generalized Cauchy point along the projected gradient path.

        Implements BLNZ Algorithm CP.  W, M, and an LU factorization of
        M (as returned by linalg.lu_factor) default to freshly built
        matrices; callers that already have them in hand (step()) pass
        them in to avoid the duplicate build.  Returns
            xc        - (n,) Cauchy point
            c         - (2k,) Wᵀ(xc − x) for the subspace solve
            free_mask - (n,) bool, True where xc is interior

        Batched-precompute design (replaces the original per-breakpoint
        solves):

        - p_j (= p before the j-th breakpoint's rank-1 update) is a
          cumulative sum  p_0 + Σ_{i<j} g_{b_i} w_{b_i}  -- one big
          cumsum over the breakpoint-ordered W rows.
        - c_j is a similar cumsum involving the sequence of dt * p_i.
        - M⁻¹ p_j and M⁻¹ c_j over every j are then two single batched
          lu_solve calls against M (since M doesn't change during the
          sweep), and the per-iteration w_b·M⁻¹{p,c} dot products fall
          out of one einsum each.
        - The per-iteration M⁻¹ w_b appears as a column of M⁻¹ Wᵀ that
          was already precomputed for the wMw diagonal.

        The result is two batched solves and a handful of einsums up
        front, then a fully vectorized (df, ddf, dt_min) state machine:
        ddf is a pure cumsum and df is a cumsum given ddf, so the whole
        recurrence and its first-minimizer break index reduce to two
        cumsums and one argmax — no Python loop and no per-breakpoint
        backend calls.
        """
        x = self.x
        n = self.n
        k = self._k
        theta = self.theta
        dtype = x.dtype

        t, sorted_indices = self._compute_breakpoints(g)

        # initial direction is the projected steepest descent ray;
        # already-active coords get d=0 inside the loop via their
        # t_i=0 breakpoint being processed first.
        d = self._scratch_array('_cauchy_d', (n,), dtype)
        free_mask = self._scratch_array('_cauchy_free_mask', (n,), bool)
        xc = self._scratch_array('_cauchy_xc', (n,), dtype)
        np.negative(g, out=d)
        free_mask.fill(True)
        xc[:] = x

        # 0-d backend scalar, not a host float: gtg only ever seeds the
        # df/ddf cumsum arrays (device) below, so keeping it on the
        # backend saves a device->host sync in the common path.  It is
        # materialized to host only in the no-finite-breakpoint branch.
        gtg = g @ g

        # Permute gradient and breakpoint time arrays into visit order once.
        g_sorted = self._scratch_array('_cauchy_g_sorted', (n,), dtype)
        t_sorted = self._scratch_array('_cauchy_t_sorted', (n,), dtype)
        np.take(g, sorted_indices, out=g_sorted)
        np.take(t, sorted_indices, out=t_sorted)
        # treat 'beyond the last finite breakpoint' as a hard stop; the
        # rest of the loop body is skipped via the t_b == inf early-exit.
        finite_mask = self._scratch_array('_cauchy_finite_mask', (n,), bool)
        np.isfinite(t_sorted, out=finite_mask)
        n_active = int(finite_mask.sum())

        # Per-breakpoint scalar inputs always needed when n_active > 0,
        # regardless of whether history is empty (k == 0).
        if n_active > 0:
            idx_a = sorted_indices[:n_active]
            t_a = t_sorted[:n_active]
            g_a = g_sorted[:n_active]
            # dt_j = t_a[j] - t_a[j-1] (t_a[-1] := 0); cumulative path
            # segment lengths shared by the vectorized state machine and
            # (when k > 0) the C_before cumsum.
            dt_seq = self._scratch_array('_cauchy_dt_seq', (n_active,), dtype)
            dt_seq[0] = t_a[0]
            dt_seq[1:] = t_a[1:] - t_a[:-1]

        if k == 0:
            df_init = -gtg
            ddf_init = theta * gtg
            p_final = self._scratch_array('_cauchy_p_final', (0,), dtype)
            c_final = self._scratch_array('_cauchy_c_final', (0,), dtype)
            # No history → no rank-1 corrections from W·M⁻¹W^T.  The
            # per-breakpoint Δdf, Δddf reduce to just the g² and θ·g²·t
            # terms below.
            wb_Mp_arr = wb_Mc_arr = wMw_b_arr = None
        else:
            if W is None:
                W = self._W()
            if M is None:
                M = self._M()
            if M_lu is None:
                M_lu = linalg.lu_factor(M)

            p_init = W.T @ d                              # (2k,)
            Mp_init = linalg.lu_solve(M_lu, p_init)       # (2k,)
            df_init = -gtg
            ddf_init = theta * gtg - p_init @ Mp_init

            if n_active == 0:
                # No breakpoints in finite time — the projected-gradient
                # path is unconstrained on this step.  Skip the batched
                # precompute (its dt_seq would degenerate to inf and the
                # lu_solve would choke on a NaN).
                p_final = p_init
                c_final = self._scratch_array('_cauchy_c_final', (2 * k,), dtype)
                c_final.fill(0)
                wb_Mp_arr = wb_Mc_arr = wMw_b_arr = None
            else:
                # MinvWT used here for the wMw diagonal; recomputed inside
                # _subspace_solve via its own M_lu re-use.
                MinvWT = linalg.lu_solve(M_lu, W.T)            # (2k, n)
                wMw_diag = self._scratch_array('_cauchy_wMw_diag',
                                               (n,), dtype)
                np.einsum('ij,ji->i', W, MinvWT, out=wMw_diag)  # (n,)

                W_a = self._scratch_array('_cauchy_W_a',
                                          (n_active, 2 * k), dtype)
                np.take(W, idx_a, axis=0, out=W_a)              # (n_a, 2k)

                # Build "p before iter j" and "c before iter j" via the
                # closed-form cumulative sums that fall out of the rank-1
                # recurrences in BLNZ Algorithm CP.
                dP_seq = self._scratch_array('_cauchy_dP_seq',
                                             (n_active, 2 * k), dtype)
                cum_dP = self._scratch_array('_cauchy_cum_dP',
                                             (n_active, 2 * k), dtype)
                P_before = self._scratch_array('_cauchy_P_before',
                                               (n_active, 2 * k), dtype)
                np.multiply(W_a, g_a[:, None], out=dP_seq)     # (n_a, 2k)
                np.cumsum(dP_seq, axis=0, out=cum_dP)          # (n_a, 2k)
                P_before[0] = p_init
                if n_active > 1:
                    P_before[1:] = cum_dP[:-1]
                    P_before[1:] += p_init

                dC_seq = self._scratch_array('_cauchy_dC_seq',
                                             (n_active, 2 * k), dtype)
                cum_dC = self._scratch_array('_cauchy_cum_dC',
                                             (n_active, 2 * k), dtype)
                C_before = self._scratch_array('_cauchy_C_before',
                                               (n_active, 2 * k), dtype)
                np.multiply(P_before, dt_seq[:, None], out=dC_seq)
                np.cumsum(dC_seq, axis=0, out=cum_dC)          # (n_a, 2k)
                C_before[0].fill(0)
                if n_active > 1:
                    C_before[1:] = cum_dC[:-1]

                # One batched lu_solve replaces n_active separate
                # M-solves.  lu_solve takes the RHS as (m, k); the
                # transposes turn our (n_a, 2k) row layout into
                # column-vectors and back.
                Mp_array = linalg.lu_solve(M_lu, P_before.T).T  # (n_a, 2k)
                Mc_array = linalg.lu_solve(M_lu, C_before.T).T  # (n_a, 2k)

                wb_Mp_arr = self._scratch_array('_cauchy_wb_Mp',
                                                (n_active,), dtype)
                wb_Mc_arr = self._scratch_array('_cauchy_wb_Mc',
                                                (n_active,), dtype)
                np.einsum('ij,ij->i', W_a, Mp_array, out=wb_Mp_arr)
                np.einsum('ij,ij->i', W_a, Mc_array, out=wb_Mc_arr)
                wMw_b_arr = self._scratch_array('_cauchy_wMw_b',
                                                (n_active,), dtype)
                np.take(wMw_diag, idx_a, out=wMw_b_arr)           # (n_a,)

        # ---- Vectorized (df, ddf, dt_min) state machine ----
        #
        # In the per-breakpoint recurrence the only sequentially-coupled
        # quantities are df and ddf.  ddf is a pure cumulative sum (its
        # update at iter j doesn't depend on df); df at iter j depends on
        # ddf at iter j via dt·ddf, but ddf is already cumulative-known.
        # So we compute the full (n_active+1,) arrays of df / ddf / dt_min
        # in two cumsums and pick out the break index with one argmax.
        # That replaces the Python scalar loop entirely and gives the GPU
        # a path that never round-trips a scalar to host inside the
        # sweep.
        if n_active == 0:
            K = 0
            # Scalar branch path: materialize the two 0-d seeds to host.
            dt_min_final = self._next_dt_min(float(df_init), float(ddf_init))
            t_old_final = 0.0
        else:
            ddf_step = self._scratch_array('_cauchy_ddf_step',
                                           (n_active,), dtype)
            df_alpha = self._scratch_array('_cauchy_df_alpha',
                                           (n_active,), dtype)
            active_work = self._scratch_array('_cauchy_active_work',
                                              (n_active,), dtype)
            if k > 0:
                np.multiply(g_a, g_a, out=ddf_step)
                df_alpha[:] = ddf_step
                ddf_step *= -theta
                np.multiply(g_a, wb_Mp_arr, out=active_work)
                active_work *= 2.0
                ddf_step -= active_work
                np.multiply(df_alpha, wMw_b_arr, out=active_work)
                ddf_step -= active_work

                np.multiply(df_alpha, t_a, out=active_work)
                active_work *= theta
                df_alpha -= active_work
                np.multiply(g_a, wb_Mc_arr, out=active_work)
                df_alpha -= active_work
            else:
                np.multiply(g_a, g_a, out=ddf_step)
                df_alpha[:] = ddf_step
                ddf_step *= -theta
                np.multiply(df_alpha, t_a, out=active_work)
                active_work *= theta
                df_alpha -= active_work
            # ddf_at_iter[j] = ddf_init + sum(ddf_step[:j])
            ddf_at_iter = self._scratch_array('_cauchy_ddf_at_iter',
                                              (n_active + 1,), dtype)
            ddf_at_iter[0] = ddf_init
            np.cumsum(ddf_step, out=ddf_at_iter[1:])
            ddf_at_iter[1:] += ddf_init
            # df_at_iter[j+1] = df_at_iter[j] + dt_seq[j]*ddf_at_iter[j] + df_alpha[j]
            df_step = self._scratch_array('_cauchy_df_step',
                                          (n_active,), dtype)
            np.multiply(dt_seq, ddf_at_iter[:-1], out=df_step)
            df_step += df_alpha
            df_at_iter = self._scratch_array('_cauchy_df_at_iter',
                                             (n_active + 1,), dtype)
            df_at_iter[0] = df_init
            np.cumsum(df_step, out=df_at_iter[1:])
            df_at_iter[1:] += df_init

            dt_min_at_iter = self._next_dt_min_vec(df_at_iter, ddf_at_iter)

            # Break check at iter j: dt_min_at_iter[j] < dt_seq[j].
            # First True index along the active prefix is K.  np.argmax
            # returns 0 on an all-False input, so check that separately.
            break_pred = self._scratch_array('_cauchy_break_pred',
                                             (n_active,), bool)
            np.less(dt_min_at_iter[:n_active], dt_seq, out=break_pred)
            if bool(break_pred.any()):
                K = int(np.argmax(break_pred))
            else:
                K = n_active
            dt_min_final = float(dt_min_at_iter[K])
            t_old_final = 0.0 if K == 0 else float(t_a[K - 1])

        # K is exactly the number of coords driven to a bound during the
        # sweep, so free_mask ends with n - K free entries.  Stash it for
        # step()'s telemetry: free_mask.sum() == n - K identically, so the
        # caller derives both free-var and break counts from K and skips a
        # device->host reduction sync.
        self._cauchy_breaks = K

        # Batch-write the per-breakpoint backend updates.  K is the
        # number of iterations that ran; visited == sorted_indices[:K].
        if K > 0:
            visited = sorted_indices[:K]
            visited_t = t_sorted[:K]
            visited_g = g_sorted[:K]
            # z_b = -t_b * g_b, xc[b] = x[b] + z_b
            visited_x = self._scratch_array('_cauchy_visited_x', (K,), dtype)
            visited_step = self._scratch_array('_cauchy_visited_step', (K,), dtype)
            np.take(x, visited, out=visited_x)
            np.multiply(visited_t, visited_g, out=visited_step)
            visited_x -= visited_step
            xc[visited] = visited_x
            d[visited] = 0
            free_mask[visited] = False

        if k > 0 and n_active > 0:
            # p_K and c_K are the values _at_ iter K, recovered from the
            # precomputed prefix-sum arrays.
            if K < n_active:
                p = P_before[K]
                c = C_before[K]
            else:
                # ran every active breakpoint; last update is dP_seq /
                # dC_seq at index n_active-1, which P_before / C_before
                # don't include because they store BEFORE-iter state.
                p = self._scratch_array('_cauchy_p_final', (2 * k,), dtype)
                c = self._scratch_array('_cauchy_c_final', (2 * k,), dtype)
                np.add(P_before[n_active - 1], dP_seq[n_active - 1], out=p)
                np.add(C_before[n_active - 1], dC_seq[n_active - 1], out=c)
        else:
            p = p_final
            c = c_final

        dt_min = dt_min_final
        t_old = t_old_final

        # final partial step along the active segment's d
        if not np.isfinite(dt_min):
            # exhausted all breakpoints with a non-convex model; cap
            # at the last breakpoint and stop.
            dt_min = 0.0
        else:
            dt_min = max(dt_min, 0.0)

        # free coords get xc[i] = x[i] + (t_old + dt_min) * d[i] (d still
        # holds -g for free coords); already-active coords keep the bound
        # value written during the sweep.
        np.multiply(d, t_old + dt_min, out=d)
        d += x
        np.copyto(xc, d, where=free_mask)
        if k > 0:
            c_update = self._scratch_array('_cauchy_c_update', (2 * k,), dtype)
            np.multiply(p, dt_min, out=c_update)
            c += c_update

        return xc, c, free_mask

    # ---------------- subspace minimization ----------------
    #
    # Given the Cauchy point xc and the free index set F, refine xc by
    # exactly minimizing the quadratic model on the free subspace with
    # active variables fixed at their bound values.
    #
    # The first-order condition Z^T ∇m(xc + Z d_F) = 0 reduces to
    #
    #     B_F d_F  =  −r_c
    #     r_c      =  Z^T (g + B z_c)
    #              =  g[F] + θ z_c[F] − (W M⁻¹ c)[F]
    #     B_F      =  Z^T B Z  =  θ I_|F| − W_F M⁻¹ W_Fᵀ
    #
    # where Z is the n × |F| selector for free variables (so Z^T A is just
    # row-slicing by free_mask) and z_c = xc − x.
    #
    # Sherman–Morrison–Woodbury on B_F gives
    #
    #     B_F⁻¹  =  (1/θ) I  −  (1/θ²) W_F N⁻¹ W_Fᵀ
    #     N      =  −M + (1/θ) W_Fᵀ W_F        shape (2k, 2k)
    #
    # so the subspace step requires a single 2k×2k solve plus row-sliced
    # backend mat–vecs.
    #
    # The unconstrained subspace minimizer x_hat = xc + Δx may sit outside
    # the box.  BLNZ 1995 handle this by truncating the path from x_k to
    # x_hat at the first bound it hits; this can give a uselessly small
    # step on problems where one coord wants a long move and another sits
    # near a bound.  Morales-Nocedal 2011 (ACM TOMS 38(1) Art. 7) replace
    # the truncation with the componentwise projection
    #
    #     x_bar = P(x_hat, l, u)
    #
    # and take direction d = x_bar - x_k.  d is feasible at α=1 by
    # construction, so the line search runs over [0, 1].  When d turns out
    # to be an ascent direction (gᵀd ≥ 0) the algorithm reverts to the
    # original BLNZ truncation rule.  See step() for the switch.

    def _subspace_solve(self, xc, c, free_mask, g, W=None, M=None, M_lu=None,
                        n_free=None):
        """Return the subspace displacement Δx (full length, zero on
        active coords) such that xc + Δx minimizes the quadratic model
        on the free subspace.  W, M, and an LU factorization of M are
        accepted from the caller to share work with _cauchy within the
        same outer step.
        """
        n = self.n
        k = self._k
        dtype = self.x.dtype
        dx = self._scratch_array('_subspace_dx', (n,), dtype)
        dx.fill(0)

        if n_free is None:
            if not bool(free_mask.any()):
                return dx
        elif n_free <= 0:
            return dx

        zc = self._scratch_array('_subspace_zc', (n,), dtype)
        np.subtract(xc, self.x, out=zc)
        theta = float(self.theta)

        if k == 0:
            # B = θI, so B_F^{-1} = (1/θ) I and Δx_F = -(1/θ) r_c.
            rc_full = self._scratch_array('_subspace_rc_full', (n,), dtype)
            np.multiply(zc, theta, out=rc_full)
            rc_full += g
            rc_full /= -theta
            np.copyto(dx, rc_full, where=free_mask)
            return dx

        if W is None:
            W = self._W()
        if M is None:
            M = self._M()
        if M_lu is None:
            M_lu = linalg.lu_factor(M)

        Mc = linalg.lu_solve(M_lu, c)
        # r_c, full-length, then row-slice to F.
        rc_full = self._scratch_array('_subspace_rc_full', (n,), dtype)
        np.multiply(zc, theta, out=rc_full)
        rc_full += g
        WMc = self._scratch_array('_subspace_WMc', (n,), dtype)
        np.matmul(W, Mc, out=WMc)
        rc_full -= WMc

        if n_free is None:
            idx_F = np.nonzero(free_mask)[0]
            n_free = idx_F.size
        elif n_free == n:
            idx_F = None
        else:
            idx_F = np.nonzero(free_mask)[0]

        if idx_F is None:
            rc_F = rc_full
            W_F = W
        else:
            rc_F = self._scratch_array('_subspace_rc_F', (n_free,), dtype)
            W_F = self._scratch_array('_subspace_W_F', (n_free, 2 * k), dtype)
            np.take(rc_full, idx_F, out=rc_F)
            np.take(W, idx_F, axis=0, out=W_F)   # (|F|, 2k)

        N = self._scratch_array('_subspace_N', (2 * k, 2 * k), dtype)
        np.matmul(W_F.T, W_F, out=N)         # (2k, 2k)
        N /= theta
        N -= M
        WF_rcF = self._scratch_array('_subspace_WF_rcF', (2 * k,), dtype)
        np.matmul(W_F.T, rc_F, out=WF_rcF)
        # One-shot solve here; not worth factoring N up front.  Using
        # np.linalg.solve (vs linalg.solve) sidesteps scipy's noisy
        # ill-conditioning warnings — N can be poorly scaled near
        # convergence without that being a correctness problem.
        N_inv = np.linalg.solve(N, WF_rcF)
        # Δx_F = -(1/θ) r_c + (1/θ²) W_F N⁻¹ W_Fᵀ r_c
        dx_F = self._scratch_array('_subspace_dx_F', (n_free,), dtype)
        np.multiply(rc_F, -1.0 / theta, out=dx_F)
        WF_N_inv = self._scratch_array('_subspace_WF_N_inv', (n_free,), dtype)
        np.matmul(W_F, N_inv, out=WF_N_inv)
        WF_N_inv /= theta * theta
        dx_F += WF_N_inv
        if idx_F is None:
            dx[:] = dx_F
        else:
            dx[idx_F] = dx_F
        return dx

    def _max_alpha_inside_box(self, x, p):
        """Largest α ∈ [0, 1] keeping x + α p inside the box."""
        # vars with p > 0 face the upper bound; p < 0 face the lower.
        bound_alpha = self._scratch_array('_bound_alpha', x.shape, x.dtype)
        work = self._scratch_array('_bound_work', x.shape, x.dtype)
        pos = self._scratch_array('_bound_pos', x.shape, bool)
        neg = self._scratch_array('_bound_neg', x.shape, bool)

        bound_alpha.fill(np.inf)
        np.greater(p, 0, out=pos)
        np.less(p, 0, out=neg)

        np.subtract(self.u, x, out=work)
        np.divide(work, p, out=work, where=pos)
        np.copyto(bound_alpha, work, where=pos)

        np.subtract(self.l, x, out=work)
        np.divide(work, p, out=work, where=neg)
        np.copyto(bound_alpha, work, where=neg)

        alpha_max = float(np.min(bound_alpha))
        if alpha_max > 1.0:
            return 1.0
        if alpha_max < 0.0:
            return 0.0
        return alpha_max

    def _two_loop(self, g):
        """Return the L-BFGS search direction -H_k g via two-loop recursion.

        Nocedal & Wright, "Numerical Optimization" (2006), Algorithm 7.4.
        H_0 is the diagonal scaling (s_{k-1} . y_{k-1}) / (y_{k-1} . y_{k-1}),
        which is the standard choice and equivalent to 1/theta_k in the
        compact representation.

        Used both for cross-validating the compact-form direction _Hg and as
        the production unconstrained-step path, where it is several times
        faster than _Hg because it never materializes the (n, 2k) W or
        (2k, 2k) M matrices.
        """
        k = self._k
        q = g.copy()
        alphas = self._alphas
        S = self.S
        Y = self.Y
        rho = self.rho
        dot = np.dot
        for i in range(k - 1, -1, -1):
            slot = self._hist_slot(i)
            alpha_i = dot(S[slot], q) * rho[slot]
            q -= alpha_i * Y[slot]
            alphas[i] = alpha_i

        if k > 0:
            slot = self._hist_slot(k - 1)
            y_last = Y[slot]
            gamma = self.ys[slot] / dot(y_last, y_last)
            r = gamma * q
        else:
            r = q

        for i in range(k):
            slot = self._hist_slot(i)
            beta_i = dot(Y[slot], r) * rho[slot]
            r += S[slot] * (alphas[i] - beta_i)

        return -r

    def _update_history(self, s, y):
        """Append a new (s, y) pair, dropping the oldest if at capacity.

        The curvature condition s . y > eps * y . y guards the BFGS
        positive-definiteness invariant; non-conforming pairs are silently
        skipped (Nocedal & Wright Sec. 7.2).
        """
        dot = np.dot
        sy = float(dot(s, y))
        yy = float(dot(y, y))
        if sy <= self._eps * max(yy, 1.0):
            return

        m = self.m
        if self._k < m:
            slot = self._k
            self._k += 1
        else:
            # Circular buffer: overwrite the oldest pair and advance the
            # logical start.  Consumers read history in oldest-newest order
            # via _hist_slot or _ordered_rows, so no physical roll is needed.
            slot = self._hist_next
            self._hist_next = (self._hist_next + 1) % m

        self.S[slot] = s
        self.Y[slot] = y
        self.ys[slot] = sy
        self.rho[slot] = 1 / sy
        self.theta = np.asarray(yy / sy, dtype=self.x.dtype)

    def step(self):
        """Perform one iteration of optimization.

        Compute the generalized Cauchy point, refine via subspace
        minimization, project the unconstrained subspace minimizer into
        the box (Morales-Nocedal 2011), and take a Wolfe line search
        along the resulting direction.  Falls back to BLNZ-style
        truncation when the projected direction is not descent.
        """
        x_pre = self.x
        f, g = self.problem.fg(self.x)
        self.nfev += 1
        if g.ndim != 1:
            g = g.ravel()

        if not self._has_bounds:
            # Cauchy + subspace solve reduces to the L-BFGS direction
            # -H_k g when no bound is finite.  The two-loop recursion
            # delivers -H_k g without materializing the (n, 2k) W or
            # (2k, 2k) M matrices, which is a clear win in the absence
            # of bounds where neither is reused.
            p = self._two_loop(g)
            alpha_max = None
            free_mask_count = self.n
            cauchy_breaks = 0
            subspace_mode = 'unbounded'
        else:
            # Build W, M, and an LU factor of M once per outer iteration;
            # _cauchy and _subspace_solve both consume them.  The LU
            # factor is shared by every per-breakpoint M-solve in the
            # Cauchy sweep and the one in _subspace_solve, replacing
            # O(breakpoints) full factorizations with O(breakpoints)
            # back-substitutions.
            if self._k > 0:
                W = self._W()
                M = self._M()
                M_lu = linalg.lu_factor(M)
            else:
                W = None
                M = None
                M_lu = None

            xc, c_vec, free_mask = self._cauchy(g, W, M, M_lu)
            n_free = self.n - self._cauchy_breaks
            dx = self._subspace_solve(xc, c_vec, free_mask, g, W, M, M_lu,
                                      n_free=n_free)
            x_hat = self._scratch_array('_step_x_hat', (self.n,), self.x.dtype)
            np.add(xc, dx, out=x_hat)

            # Morales-Nocedal 2011: project x_hat into the box and use that
            # as the trial point.  If the resulting direction is descent,
            # take the full unit step as the line-search ceiling.  Otherwise
            # revert to the BLNZ truncation rule that walks from x_k toward
            # x_hat until the first bound is reached.
            x_bar = self._scratch_array('_step_x_bar', (self.n,), self.x.dtype)
            np.maximum(x_hat, self.l, out=x_bar)
            np.minimum(x_bar, self.u, out=x_bar)
            p_proj = self._scratch_array('_step_p_proj', (self.n,), self.x.dtype)
            np.subtract(x_bar, self.x, out=p_proj)
            gp_proj = float(g @ p_proj)
            if gp_proj < 0.0:
                p = p_proj
                alpha_max = 1.0
                subspace_mode = 'projected'
            else:
                p = self._scratch_array('_step_p_hat', (self.n,), self.x.dtype)
                np.subtract(x_hat, self.x, out=p)
                alpha_max = self._max_alpha_inside_box(self.x, p)
                subspace_mode = 'truncated'
            # _cauchy set self._cauchy_breaks = K; free_mask has exactly
            # n - K free entries, so derive both without re-reducing it.
            cauchy_breaks = self._cauchy_breaks
            free_mask_count = n_free

        # No room to move, or the Cauchy+subspace direction degenerated
        # to zero (e.g. every coord is at a bound being pushed into).
        if (alpha_max is not None and alpha_max <= 0.0) or not bool(np.any(p)):
            self.last_step_metadata = {'reason': 'no_descent'}
            raise StopIteration(_OptimizerStop(False, 'no descent direction'))

        alpha, _, _, g_new = _strong_wolfe_lean(
            self.problem, self.x, p,
            fg_at_xk=(f, g),
            maxalpha=alpha_max,
            c1=self.c1, c2=self.c2, maxiter=self.maxls,
        )

        if alpha is None:
            self.last_step_metadata = {'reason': 'linesearch_fail'}
            raise StopIteration(_OptimizerStop(False, 'line search failed'))

        # The Wolfe search has already evaluated (f, g) at xk + alpha*pk
        # via its internal cache; reuse the gradient instead of paying
        # an extra fg for it.
        if g_new.ndim != 1:
            g_new = g_new.ravel()

        s = self._scratch_array('_step_s', (self.n,), self.x.dtype)
        np.multiply(p, alpha, out=s)
        x_new = self.x + s
        if self._has_bounds:
            # Defensive clip: the Wolfe search may walk fractionally past a
            # bound under floating-point roundoff even with maxalpha set.
            # The clipped point is within an ULP of x_new, so the cached
            # g_new is still the right representative for the curvature update.
            np.maximum(x_new, self.l, out=x_new)
            np.minimum(x_new, self.u, out=x_new)
            # honor the clip in the history update too
            np.subtract(x_new, self.x, out=s)

        y = self._scratch_array('_step_y', (self.n,), self.x.dtype)
        np.subtract(g_new, g, out=y)
        self._update_history(s, y)

        self.x = x_new
        self.iter += 1
        self.last_step_metadata = {
            'alpha': float(alpha),
            'free_vars': free_mask_count,
            'cauchy_breaks': cauchy_breaks,
            'subspace_solved': True,
            'subspace_mode': subspace_mode,
        }
        return x_pre, float(f), g
