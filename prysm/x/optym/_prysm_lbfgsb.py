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
from prysm.mathops import linalg, np

from .linesearch import ls_strong_wolfe
from .problem import as_problem


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
    Python implementation matches or beats the scipy driver on CPU at
    n >= 1000 in the Rosenbrock CPU benchmark and is roughly 2x slower
    for tiny problems (n < 10) due to Python per-call overhead.

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
        self._k = 0  # valid history count, capped at memory

        # theta = (y_{k-1} . y_{k-1}) / (s_{k-1} . y_{k-1}); 1 before any
        # update so the initial Hessian approximation is the identity.
        self.theta = np.asarray(1, dtype=dtype)

        self.c1 = c1
        self.c2 = c2
        self.maxls = maxls

        self.iter = 0
        self.nfev = 0
        self.last_step_metadata = {}

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

    def _W(self):
        """The (n, 2k) compact matrix W_k = [Y_k, θ_k S_k]."""
        k = self._k
        Y_T = self.Y[:k].T
        S_T = self.S[:k].T
        return np.concatenate([Y_T, self.theta * S_T], axis=1)

    def _M(self):
        """The (2k, 2k) middle matrix from the compact representation."""
        k = self._k
        S = self.S[:k]
        Y = self.Y[:k]
        SY = S @ Y.T               # (i,j) = s_i · y_j
        SS = S @ S.T               # (i,j) = s_i · s_j
        D = np.diag(self.ys[:k])
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

    def _cauchy(self, g, W=None, M=None):
        """Generalized Cauchy point along the projected gradient path.

        Implements BLNZ Algorithm CP.  W and M default to freshly built
        matrices; callers that already have them in hand (step()) pass
        them in to avoid the duplicate build.  Returns
            xc        - (n,) Cauchy point
            c         - (2k,) Wᵀ(xc − x) for the subspace solve
            free_mask - (n,) bool, True where xc is interior
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

        gtg = float(g @ g)

        if k == 0:
            c = np.zeros(0, dtype=dtype)
            p = np.zeros(0, dtype=dtype)
            df = -gtg
            ddf = float(theta) * gtg
        else:
            if W is None:
                W = self._W()
            if M is None:
                M = self._M()
            c = np.zeros(2 * k, dtype=dtype)
            p = W.T @ d
            Mp = np.linalg.solve(M, p)
            df = -gtg
            ddf = float(theta) * gtg - float(p @ Mp)

        dt_min = self._next_dt_min(df, ddf)
        t_old = 0.0

        for j in range(n):
            b = int(sorted_indices[j])
            t_b = float(t[b])
            if not np.isfinite(t_b):
                break
            dt = t_b - t_old
            if dt_min < dt:
                break

            # advance c, df, ddf to t_b (still on segment j), then apply
            # the rank-one jump for the new active variable b.
            if k > 0:
                c = c + dt * p
            df_at_tb = df + dt * ddf

            g_b = float(g[b])
            z_b = -t_b * g_b
            xc[b] = x[b] + z_b

            if k > 0:
                w_b = W[b, :]
                Mc = np.linalg.solve(M, c)
                Mp_seg = np.linalg.solve(M, p)
                Mw = np.linalg.solve(M, w_b)
                wb_Mc = float(w_b @ Mc)
                wb_Mp = float(w_b @ Mp_seg)
                wb_Mw = float(w_b @ Mw)
                df = (df_at_tb + g_b * g_b + g_b * float(theta) * z_b
                      - g_b * wb_Mc)
                ddf = (ddf - float(theta) * g_b * g_b
                       - 2.0 * g_b * wb_Mp - g_b * g_b * wb_Mw)
                p = p + g_b * w_b
            else:
                df = df_at_tb + g_b * g_b + g_b * float(theta) * z_b
                ddf = ddf - float(theta) * g_b * g_b

            d[b] = 0
            free_mask[b] = False
            t_old = t_b
            dt_min = self._next_dt_min(df, ddf)

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

    def _subspace_solve(self, xc, c, free_mask, g, W=None, M=None):
        """Return the subspace displacement Δx (full length, zero on
        active coords) such that xc + Δx minimizes the quadratic model
        on the free subspace.  W and M are accepted from the caller to
        avoid rebuilding the compact representation matrices.
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

        Mc = np.linalg.solve(M, c)
        # r_c, full-length, then row-slice to F.
        rc_full = g + theta * zc - W @ Mc
        rc_F = rc_full[free_mask]
        W_F = W[free_mask, :]                # (|F|, 2k)

        N = -M + (W_F.T @ W_F) / theta       # (2k, 2k)
        WF_rcF = W_F.T @ rc_F
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
        alphas = [None] * k
        for i in range(k - 1, -1, -1):
            alpha_i = np.dot(self.S[i], q) / self.ys[i]
            q = q - alpha_i * self.Y[i]
            alphas[i] = alpha_i

        if k > 0:
            y_last = self.Y[k - 1]
            gamma = self.ys[k - 1] / np.dot(y_last, y_last)
            r = gamma * q
        else:
            r = q

        for i in range(k):
            beta_i = np.dot(self.Y[i], r) / self.ys[i]
            r = r + self.S[i] * (alphas[i] - beta_i)

        return -r

    def _update_history(self, s, y):
        """Append a new (s, y) pair, dropping the oldest if at capacity.

        The curvature condition s . y > eps * y . y guards the BFGS
        positive-definiteness invariant; non-conforming pairs are silently
        skipped (Nocedal & Wright Sec. 7.2).
        """
        sy = float(np.dot(s, y))
        yy = float(np.dot(y, y))
        eps = float(np.finfo(self.x.dtype).eps)
        if sy <= eps * max(yy, 1.0):
            return

        m = self.m
        if self._k < m:
            slot = self._k
            self._k += 1
        else:
            # roll the buffer left and write to the last slot.  Use an
            # explicit .copy() of the source slice; some backends do not
            # promise correct behavior for overlapping in-place slice
            # assignments.
            self.S[:-1] = self.S[1:].copy()
            self.Y[:-1] = self.Y[1:].copy()
            self.ys[:-1] = self.ys[1:].copy()
            slot = m - 1

        self.S[slot] = s
        self.Y[slot] = y
        self.ys[slot] = sy
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
            alpha_max = np.inf
            free_mask_count = self.n
            cauchy_breaks = 0
            subspace_mode = 'unbounded'
        else:
            # Build W and M once per outer iteration; _cauchy and
            # _subspace_solve both consume them.
            if self._k > 0:
                W = self._W()
                M = self._M()
            else:
                W = None
                M = None

            xc, c_vec, free_mask = self._cauchy(g, W, M)
            dx = self._subspace_solve(xc, c_vec, free_mask, g, W, M)
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
            free_mask_count = int(free_mask.sum())
            cauchy_breaks = self.n - free_mask_count

        # No room to move, or the Cauchy+subspace direction degenerated
        # to zero (e.g. every coord is at a bound being pushed into).
        if alpha_max <= 0.0 or float(np.max(np.abs(p))) == 0.0:
            self.last_step_metadata = {'reason': 'no_descent'}
            raise StopIteration

        alpha, _, _, g_new = ls_strong_wolfe(
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
