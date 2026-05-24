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
        :func:`as_problem`.
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

    def _W(self):
        """The (n, 2k) compact matrix W_k = [Y_k, θ_k S_k]."""
        Y_T = self._ordered_rows(self.Y).T
        S_T = self._ordered_rows(self.S).T
        return np.concatenate([Y_T, self.theta * S_T], axis=1)

    def _M(self):
        """The (2k, 2k) middle matrix from the compact representation."""
        S = self._ordered_rows(self.S)
        Y = self._ordered_rows(self.Y)
        SY = S @ Y.T               # (i,j) = s_i · y_j
        SS = S @ S.T               # (i,j) = s_i · s_j
        D = np.diag(self._ordered_rows(self.ys))
        # strict lower triangular (zero out diagonal and above)
        L = SY - np.triu(SY)
        top = np.concatenate([-D, L.T], axis=1)
        bot = np.concatenate([L, self.theta * SS], axis=1)
        return np.concatenate([top, bot], axis=0)

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
        t = np.full(self.n, np.inf, dtype=dtype)
        pos_g = g > 0
        neg_g = g < 0
        # guard divisor with np.where; masked-off entries are discarded
        safe_g = np.where(g != 0, g, 1)
        lower_hit = pos_g & np.isfinite(l)
        upper_hit = neg_g & np.isfinite(u)
        t = np.where(lower_hit, (x - l) / safe_g, t)
        t = np.where(upper_hit, (x - u) / safe_g, t)
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

    @staticmethod
    def _next_dt_min_vec(df, ddf):
        """Vectorized form of _next_dt_min.  Works element-wise on
        arrays of df and ddf; same branching semantics applied
        coordinate by coordinate via np.where.
        """
        # divisor guard: ddf<=0 entries will be masked out below, but
        # the np.where eager-evaluates both branches, so feed a safe
        # value where ddf would be zero/negative.
        safe_ddf = np.where(ddf > 0, ddf, 1.0)
        return np.where(
            df >= 0, 0.0,
            np.where(ddf <= 0, np.inf, -df / safe_ddf),
        )

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
        d = -g.copy()
        free_mask = np.ones(n, dtype=bool)
        xc = x.copy()

        # 0-d backend scalar, not a host float: gtg only ever seeds the
        # df/ddf cumsum arrays (device) below, so keeping it on the
        # backend saves a device->host sync in the common path.  It is
        # materialized to host only in the no-finite-breakpoint branch.
        gtg = g @ g

        # Permute gradient and breakpoint time arrays into visit order once.
        g_sorted = g[sorted_indices]
        t_sorted = t[sorted_indices]
        # treat 'beyond the last finite breakpoint' as a hard stop; the
        # rest of the loop body is skipped via the t_b == inf early-exit.
        finite_mask = np.isfinite(t_sorted)
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
            dt_seq = np.empty(n_active, dtype=dtype)
            dt_seq[0] = t_a[0]
            dt_seq[1:] = t_a[1:] - t_a[:-1]

        if k == 0:
            df_init = -gtg
            ddf_init = theta * gtg
            p_final = np.zeros(0, dtype=dtype)
            c_final = np.zeros(0, dtype=dtype)
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
                c_final = np.zeros(2 * k, dtype=dtype)
                wb_Mp_arr = wb_Mc_arr = wMw_b_arr = None
            else:
                # MinvWT used here for the wMw diagonal; recomputed inside
                # _subspace_solve via its own M_lu re-use.
                MinvWT = linalg.lu_solve(M_lu, W.T)            # (2k, n)
                wMw_diag = np.einsum('ij,ji->i', W, MinvWT)    # (n,)

                W_a = W[idx_a, :]                              # (n_a, 2k)

                # Build "p before iter j" and "c before iter j" via the
                # closed-form cumulative sums that fall out of the rank-1
                # recurrences in BLNZ Algorithm CP.
                dP_seq = g_a[:, None] * W_a                    # (n_a, 2k)
                cum_dP = np.cumsum(dP_seq, axis=0)             # (n_a, 2k)
                P_before = np.concatenate(
                    [p_init[None, :], p_init[None, :] + cum_dP[:-1, :]],
                    axis=0,
                )                                              # (n_a, 2k)

                dC_seq = dt_seq[:, None] * P_before            # (n_a, 2k)
                cum_dC = np.cumsum(dC_seq, axis=0)             # (n_a, 2k)
                C_before = np.concatenate(
                    [np.zeros((1, 2 * k), dtype=dtype), cum_dC[:-1, :]],
                    axis=0,
                )                                              # (n_a, 2k)

                # One batched lu_solve replaces n_active separate
                # M-solves.  lu_solve takes the RHS as (m, k); the
                # transposes turn our (n_a, 2k) row layout into
                # column-vectors and back.
                Mp_array = linalg.lu_solve(M_lu, P_before.T).T  # (n_a, 2k)
                Mc_array = linalg.lu_solve(M_lu, C_before.T).T  # (n_a, 2k)

                wb_Mp_arr = np.einsum('ij,ij->i', W_a, Mp_array)  # (n_a,)
                wb_Mc_arr = np.einsum('ij,ij->i', W_a, Mc_array)  # (n_a,)
                wMw_b_arr = wMw_diag[idx_a]                       # (n_a,)

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
            if k > 0:
                ddf_step = (-theta * g_a * g_a
                            - 2.0 * g_a * wb_Mp_arr
                            - g_a * g_a * wMw_b_arr)
                df_alpha = (g_a * g_a - theta * g_a * g_a * t_a
                            - g_a * wb_Mc_arr)
            else:
                ddf_step = -theta * g_a * g_a
                df_alpha = g_a * g_a - theta * g_a * g_a * t_a
            # ddf_at_iter[j] = ddf_init + sum(ddf_step[:j])
            ddf_at_iter = np.empty(n_active + 1, dtype=dtype)
            ddf_at_iter[0] = ddf_init
            ddf_at_iter[1:] = ddf_init + np.cumsum(ddf_step)
            # df_at_iter[j+1] = df_at_iter[j] + dt_seq[j]*ddf_at_iter[j] + df_alpha[j]
            df_step = dt_seq * ddf_at_iter[:-1] + df_alpha
            df_at_iter = np.empty(n_active + 1, dtype=dtype)
            df_at_iter[0] = df_init
            df_at_iter[1:] = df_init + np.cumsum(df_step)

            dt_min_at_iter = self._next_dt_min_vec(df_at_iter, ddf_at_iter)

            # Break check at iter j: dt_min_at_iter[j] < dt_seq[j].
            # First True index along the active prefix is K.  np.argmax
            # returns 0 on an all-False input, so check that separately.
            break_pred = dt_min_at_iter[:n_active] < dt_seq
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
            xc[visited] = x[visited] - visited_t * visited_g
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
                p = P_before[n_active - 1] + dP_seq[n_active - 1]
                c = C_before[n_active - 1] + dC_seq[n_active - 1]
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
        xc = np.where(free_mask, x + (t_old + dt_min) * d, xc)
        if k > 0:
            c = c + dt_min * p

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

    def _subspace_solve(self, xc, c, free_mask, g, W=None, M=None, M_lu=None):
        """Return the subspace displacement Δx (full length, zero on
        active coords) such that xc + Δx minimizes the quadratic model
        on the free subspace.  W, M, and an LU factorization of M are
        accepted from the caller to share work with _cauchy within the
        same outer step.
        """
        n = self.n
        k = self._k
        dtype = self.x.dtype
        dx = np.zeros(n, dtype=dtype)

        if not bool(free_mask.any()):
            return dx

        zc = xc - self.x
        theta = float(self.theta)

        if k == 0:
            # B = θI, so B_F^{-1} = (1/θ) I and Δx_F = -(1/θ) r_c.
            rc_F = g[free_mask] + theta * zc[free_mask]
            dx_F = -rc_F / theta
            dx[free_mask] = dx_F
            return dx

        if W is None:
            W = self._W()
        if M is None:
            M = self._M()
        if M_lu is None:
            M_lu = linalg.lu_factor(M)

        Mc = linalg.lu_solve(M_lu, c)
        # r_c, full-length, then row-slice to F.
        rc_full = g + theta * zc - W @ Mc
        rc_F = rc_full[free_mask]
        W_F = W[free_mask, :]                # (|F|, 2k)

        N = -M + (W_F.T @ W_F) / theta       # (2k, 2k)
        WF_rcF = W_F.T @ rc_F
        # One-shot solve here; not worth factoring N up front.  Using
        # np.linalg.solve (vs linalg.solve) sidesteps scipy's noisy
        # ill-conditioning warnings — N can be poorly scaled near
        # convergence without that being a correctness problem.
        N_inv = np.linalg.solve(N, WF_rcF)
        # Δx_F = -(1/θ) r_c + (1/θ²) W_F N⁻¹ W_Fᵀ r_c
        dx_F = -rc_F / theta + (W_F @ N_inv) / (theta * theta)
        dx[free_mask] = dx_F
        return dx

    def _max_alpha_inside_box(self, x, p):
        """Largest α ∈ [0, 1] keeping x + α p inside the box."""
        # vars with p > 0 face the upper bound; p < 0 face the lower.
        safe_p = np.where(p != 0, p, 1)
        bound_alpha = np.full_like(x, np.inf)
        pos = p > 0
        neg = p < 0
        bound_alpha = np.where(pos, (self.u - x) / safe_p, bound_alpha)
        bound_alpha = np.where(neg, (self.l - x) / safe_p, bound_alpha)
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

        Used both for cross-validating the compact-form direction _Hg
        (Phase 3 tests) and as the production unconstrained-step path,
        where it is several times faster than _Hg because it never
        materializes the (n, 2k) W or (2k, 2k) M matrices.
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
        eps = float(np.finfo(self.x.dtype).eps)
        if sy <= eps * max(yy, 1.0):
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
        x_pre = self.x.copy()
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
            dx = self._subspace_solve(xc, c_vec, free_mask, g, W, M, M_lu)
            x_hat = xc + dx

            # Morales-Nocedal 2011: project x_hat into the box and use that
            # as the trial point.  If the resulting direction is descent,
            # take the full unit step as the line-search ceiling.  Otherwise
            # revert to the BLNZ truncation rule that walks from x_k toward
            # x_hat until the first bound is reached.
            x_bar = np.minimum(np.maximum(x_hat, self.l), self.u)
            p_proj = x_bar - self.x
            gp_proj = float(g @ p_proj)
            if gp_proj < 0.0:
                p = p_proj
                alpha_max = 1.0
                subspace_mode = 'projected'
            else:
                p = x_hat - self.x
                alpha_max = self._max_alpha_inside_box(self.x, p)
                subspace_mode = 'truncated'
            # _cauchy set self._cauchy_breaks = K; free_mask has exactly
            # n - K free entries, so derive both without re-reducing it.
            cauchy_breaks = self._cauchy_breaks
            free_mask_count = self.n - cauchy_breaks

        # No room to move, or the Cauchy+subspace direction degenerated
        # to zero (e.g. every coord is at a bound being pushed into).
        if (alpha_max is not None and alpha_max <= 0.0) or not bool(np.any(p)):
            self.last_step_metadata = {'reason': 'no_descent'}
            raise StopIteration

        alpha, _, _, g_new = _strong_wolfe_lean(
            self.problem, self.x, p,
            fg_at_xk=(f, g),
            maxalpha=alpha_max,
            c1=self.c1, c2=self.c2, maxiter=self.maxls,
        )

        if alpha is None:
            self.last_step_metadata = {'reason': 'linesearch_fail'}
            raise StopIteration

        # The Wolfe search has already evaluated (f, g) at xk + alpha*pk
        # via its internal cache; reuse the gradient instead of paying
        # an extra fg for it.
        if g_new.ndim != 1:
            g_new = g_new.ravel()

        s = alpha * p
        x_new = self.x + s
        if self._has_bounds:
            # Defensive clip: the Wolfe search may walk fractionally past a
            # bound under floating-point roundoff even with maxalpha set.
            # The clipped point is within an ULP of x_new, so the cached
            # g_new is still the right representative for the curvature update.
            x_new = np.minimum(np.maximum(x_new, self.l), self.u)
            s = x_new - self.x  # honor the clip in the history update too

        y = g_new - g
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
        return x_pre, float(f), g.copy()
