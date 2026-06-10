"""Pure-Python, backend-shim friendly L-BFGS-B.

Implements BLNZ 1995 with the Morales-Nocedal 2011 projection refinement.
Array work goes through prysm.mathops.np; Wolfe scalars live on the host.
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
    Algorithm 778", ACM Trans. Math. Softw. 38(1), 2011.  The subspace
    minimizer is projected into the box when that direction is descent;
    otherwise the original BLNZ truncation rule is used.

    Parameters
    ----------
    fg : callable or Problem
        fg(x) -> (f, g) or a Problem-shaped object.
    x0 : ndarray
        the parameter vector immediately prior to optimization.  Its dtype
        sets the dtype of every internal array.
    memory : int
        Number of recent (s, y) pairs retained.
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
        # match the bounds to x0's dtype so e.g. float64 bounds cannot
        # promote float32 arithmetic anywhere downstream.
        if lower_bounds is None:
            lower_bounds = np.full(self.n, -np.inf, dtype=dtype)
        else:
            lower_bounds = np.asarray(lower_bounds, dtype=dtype)
        if upper_bounds is None:
            upper_bounds = np.full(self.n, np.inf, dtype=dtype)
        else:
            upper_bounds = np.asarray(upper_bounds, dtype=dtype)
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
        # initial breakpoint-sweep chunk; grows geometrically within a
        # sweep.  The sweep usually stops within the first few breakpoints,
        # so this bounds the common-case work of _cauchy.
        self._cauchy_chunk = 64
        # Lazy reusable work arrays.
        self._scratch = {}

    def _scratch_array(self, name, shape, dtype=None):
        """Return a reusable scratch array of exactly shape.

        Buffers grow but never shrink; oversize buffers return a slice view.
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

    def _WM(self):
        """Build W_k and M_k together.

        W is returned transposed -- Wt, shape (2k, n) -- so that its
        construction and every downstream consumer (gathers, batched
        solves, einsums) operate along contiguous rows.  The history ring
        buffer is consumed in physical order; only the small (k, k)
        products are permuted into logical (oldest-newest) order.
        """
        k = self._k
        m = self.m
        dtype = self.x.dtype
        theta = self.theta
        S = self.S[:k]
        Y = self.Y[:k]
        ys = self.ys[:k]

        Wt = self._scratch_array('_compact_Wt', (2 * k, self.n), dtype)
        if k == m and self._hist_next:
            # wrapped ring buffer: physical row order != logical order.
            order = (np.arange(k) + self._hist_next) % m
            np.take(Y, order, axis=0, out=Wt[:k])
            np.take(S, order, axis=0, out=Wt[k:])
            Wt[k:] *= theta
        else:
            order = None
            Wt[:k] = Y
            np.multiply(S, theta, out=Wt[k:])

        SY = self._scratch_array('_compact_SY', (k, k), dtype)
        SS = self._scratch_array('_compact_SS', (k, k), dtype)
        M = self._scratch_array('_compact_M', (2 * k, 2 * k), dtype)
        np.matmul(S, Y.T, out=SY)  # (i,j) = s_i . y_j, physical order
        np.matmul(S, S.T, out=SS)  # (i,j) = s_i . s_j, physical order
        if order is not None:
            SY = SY[order][:, order]
            SS = SS[order][:, order]
            ys = ys[order]
        M.fill(0)
        idx = np.arange(k)
        M[idx, idx] = -ys
        L = np.tril(SY, -1)  # strict lower triangle of S Yᵀ
        M[k:, :k] = L
        M[:k, k:] = L.T
        np.multiply(SS, theta, out=M[k:, k:])
        return Wt, M

    def _W(self):
        """The (n, 2k) compact matrix W_k = [Y_k, θ_k S_k]."""
        return self._WM()[0].T

    def _M(self):
        """The (2k, 2k) middle matrix from the compact representation."""
        return self._WM()[1]

    def _Bv(self, v):
        """Apply B_k v using the compact form."""
        if self._k == 0:
            return self.theta * v
        Wt, M = self._WM()
        return self.theta * v - Wt.T @ np.linalg.solve(M, Wt @ v)

    def _Hg(self, g):
        """Return the L-BFGS descent direction −H_k g.

        Uses the Sherman–Morrison–Woodbury formula on the compact form
        of B_k; one (2k, 2k) solve per call.
        """
        if self._k == 0:
            return -g
        Wt, M = self._WM()
        theta = self.theta
        N = -M + (Wt @ Wt.T) / theta
        Ninv_Wtg = np.linalg.solve(N, Wt @ g)
        Hg = g / theta - (Wt.T @ Ninv_Wtg) / (theta * theta)
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

    def _cauchy(self, g, Wt=None, M=None, M_lu=None):
        """Generalized Cauchy point along the projected gradient path.

        Implements BLNZ Algorithm CP.  Optional Wt, M, and M_lu are reused
        from step() when available.  Returns the Cauchy point, W.T@(xc-x),
        and a mask of interior variables.

        The breakpoint recurrence is evaluated in chunks: within a chunk it
        reduces to prefix sums plus batched solves against M, and the sweep
        exits at the first chunk containing the path minimizer.  Chunks grow
        geometrically from self._cauchy_chunk, so the common case (minimizer
        within the first few breakpoints) does O(chunk) work while the worst
        case stays within ~15% of one full-width pass.
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

        # Seed the sweep state: df/ddf as host scalars (the chunk loop syncs
        # at its break predicate anyway), p = Wᵀd and c = 0 carried in
        # reusable buffers.  With no history there are no W M⁻¹ Wᵀ
        # correction terms and p, c are empty placeholders.
        gtg = g @ g
        if k == 0:
            df = -float(gtg)
            ddf = float(theta * gtg)
            p = self._scratch_array('_cauchy_p_carry', (0,), dtype)
            c = self._scratch_array('_cauchy_c_carry', (0,), dtype)
        else:
            if Wt is None or M is None:
                Wt, M = self._WM()
            if M_lu is None:
                M_lu = linalg.lu_factor(M, check_finite=False)

            p_init = Wt @ d                               # (2k,)
            Mp_init = linalg.lu_solve(M_lu, p_init, check_finite=False)
            df = -float(gtg)
            ddf = float(theta * gtg - p_init @ Mp_init)
            p = self._scratch_array('_cauchy_p_carry', (2 * k,), dtype)
            c = self._scratch_array('_cauchy_c_carry', (2 * k,), dtype)
            p[:] = p_init
            c.fill(0)

        # ---- Chunked (df, ddf, dt_min) state machine ----
        #
        # Within a chunk, df and ddf reduce to cumulative sums and the first
        # minimizer to an argmax over the breakpoint predicate -- one host
        # sync per chunk instead of one per breakpoint.
        K = None
        dt_min_final = None
        t_prev = 0.0
        lo = 0
        chunk = self._cauchy_chunk
        while lo < n_active:
            hi = min(lo + chunk, n_active)
            m_c = hi - lo
            idx_c = sorted_indices[lo:hi]
            t_c = t_sorted[lo:hi]
            g_c = g_sorted[lo:hi]
            # dt_c[j] is the path length from breakpoint j-1 (or the
            # carry-in point) to breakpoint j.
            dt_c = self._scratch_array('_cauchy_dt_c', (m_c,), dtype)
            dt_c[0] = t_c[0] - t_prev
            dt_c[1:] = t_c[1:] - t_c[:-1]

            if k > 0:
                W_cT = self._scratch_array('_cauchy_W_cT', (2 * k, m_c), dtype)
                np.take(Wt, idx_c, axis=1, out=W_cT)
                # w_bᵀ M⁻¹ w_b for this chunk's rows only.
                MinvWcT = linalg.lu_solve(M_lu, W_cT, check_finite=False)
                wMw_b = self._scratch_array('_cauchy_wMw_b', (m_c,), dtype)
                np.einsum('ib,ib->b', W_cT, MinvWcT, out=wMw_b)

                # Prefix states before each breakpoint in BLNZ Algorithm CP,
                # held transposed -- (2k, m_c) -- so the cumulative sums run
                # along contiguous memory.
                dP_T = self._scratch_array('_cauchy_dP_T', (2 * k, m_c), dtype)
                P_T = self._scratch_array('_cauchy_P_T', (2 * k, m_c), dtype)
                np.multiply(W_cT, g_c, out=dP_T)         # dP_T[:, b] = g_b w_b
                P_T[:, 0] = p
                if m_c > 1:
                    np.cumsum(dP_T[:, :-1], axis=1, out=P_T[:, 1:])
                    P_T[:, 1:] += p[:, None]

                dC_T = self._scratch_array('_cauchy_dC_T', (2 * k, m_c), dtype)
                C_T = self._scratch_array('_cauchy_C_T', (2 * k, m_c), dtype)
                np.multiply(P_T, dt_c, out=dC_T)         # dC_T[:, b] = Δt_b P_b
                C_T[:, 0] = c
                if m_c > 1:
                    np.cumsum(dC_T[:, :-1], axis=1, out=C_T[:, 1:])
                    C_T[:, 1:] += c[:, None]

                # One batched solve replaces m_c separate M-solves.
                Mp_T = linalg.lu_solve(M_lu, P_T, check_finite=False)
                Mc_T = linalg.lu_solve(M_lu, C_T, check_finite=False)

                wb_Mp = self._scratch_array('_cauchy_wb_Mp', (m_c,), dtype)
                wb_Mc = self._scratch_array('_cauchy_wb_Mc', (m_c,), dtype)
                np.einsum('ib,ib->b', W_cT, Mp_T, out=wb_Mp)
                np.einsum('ib,ib->b', W_cT, Mc_T, out=wb_Mc)

            ddf_step = self._scratch_array('_cauchy_ddf_step', (m_c,), dtype)
            df_alpha = self._scratch_array('_cauchy_df_alpha', (m_c,), dtype)
            work = self._scratch_array('_cauchy_active_work', (m_c,), dtype)
            # Δddf and Δdf at each breakpoint; the w_b terms are the rank-one
            # corrections from W M⁻¹ Wᵀ and vanish when history is empty.
            np.multiply(g_c, g_c, out=ddf_step)
            df_alpha[:] = ddf_step
            ddf_step *= -theta
            if k > 0:
                np.multiply(g_c, wb_Mp, out=work)
                work *= 2.0
                ddf_step -= work
                np.multiply(df_alpha, wMw_b, out=work)
                ddf_step -= work
            np.multiply(df_alpha, t_c, out=work)
            work *= theta
            df_alpha -= work
            if k > 0:
                np.multiply(g_c, wb_Mc, out=work)
                df_alpha -= work

            # ddf_at[j] = ddf + sum(ddf_step[:j])
            ddf_at = self._scratch_array('_cauchy_ddf_at_iter',
                                         (m_c + 1,), dtype)
            ddf_at[0] = ddf
            np.cumsum(ddf_step, out=ddf_at[1:])
            ddf_at[1:] += ddf
            # df_at[j+1] = df_at[j] + dt_c[j]*ddf_at[j] + df_alpha[j]
            df_step = self._scratch_array('_cauchy_df_step', (m_c,), dtype)
            np.multiply(dt_c, ddf_at[:-1], out=df_step)
            df_step += df_alpha
            df_at = self._scratch_array('_cauchy_df_at_iter',
                                        (m_c + 1,), dtype)
            df_at[0] = df
            np.cumsum(df_step, out=df_at[1:])
            df_at[1:] += df

            dt_min_at = self._next_dt_min_vec(df_at, ddf_at)

            # Break check at iter j: dt_min_at[j] < dt_c[j].  np.argmax
            # returns 0 on an all-False input, so check that separately.
            break_pred = self._scratch_array('_cauchy_break_pred',
                                             (m_c,), bool)
            np.less(dt_min_at[:m_c], dt_c, out=break_pred)
            if bool(break_pred.any()):
                j = int(np.argmax(break_pred))
                K = lo + j
                dt_min_final = float(dt_min_at[j])
                t_old = t_prev if j == 0 else float(t_c[j - 1])
                if k > 0:
                    # p and c at iter K, before its breakpoint update.
                    p = P_T[:, j]
                    c = C_T[:, j]
                break

            # No minimizer in this chunk; carry the state past it.
            df = float(df_at[m_c])
            ddf = float(ddf_at[m_c])
            t_prev = float(t_c[-1])
            if k > 0:
                # Prefix arrays hold before-states; add the final update.
                np.add(P_T[:, -1], dP_T[:, -1], out=p)
                np.add(C_T[:, -1], dC_T[:, -1], out=c)
            lo = hi
            chunk *= 8

        if K is None:
            # Path exhausted without an interior minimizer (or no finite
            # breakpoints at all): the model minimum lies past the last
            # breakpoint along the carried direction.
            K = n_active
            dt_min_final = self._next_dt_min(df, ddf)
            t_old = t_prev

        # K is the number of coords driven to a bound during the sweep.
        self._cauchy_breaks = K

        # Batch-write the visited breakpoint updates.
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

        dt_min = dt_min_final

        # final partial step along the active segment's d
        if not np.isfinite(dt_min):
            # Exhausted all breakpoints with a non-convex model.
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

    def _subspace_solve(self, xc, c, free_mask, g, Wt=None, M=None, M_lu=None,
                        n_free=None):
        """Return the full-length free-subspace displacement Δx."""
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
        # 0-d backend scalar; every use below is out= or in-place, so it
        # never promotes the dtype and never forces a device->host sync.
        theta = self.theta

        if k == 0:
            # B = θI, so B_F^{-1} = (1/θ) I and Δx_F = -(1/θ) r_c.
            rc_full = self._scratch_array('_subspace_rc_full', (n,), dtype)
            np.multiply(zc, theta, out=rc_full)
            rc_full += g
            rc_full /= -theta
            np.copyto(dx, rc_full, where=free_mask)
            return dx

        if Wt is None or M is None:
            Wt, M = self._WM()
        if M_lu is None:
            M_lu = linalg.lu_factor(M, check_finite=False)

        Mc = linalg.lu_solve(M_lu, c, check_finite=False)
        # r_c, full-length, then row-slice to F.
        rc_full = self._scratch_array('_subspace_rc_full', (n,), dtype)
        np.multiply(zc, theta, out=rc_full)
        rc_full += g
        WMc = self._scratch_array('_subspace_WMc', (n,), dtype)
        np.matmul(Wt.T, Mc, out=WMc)
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
            W_FT = Wt
        else:
            rc_F = self._scratch_array('_subspace_rc_F', (n_free,), dtype)
            W_FT = self._scratch_array('_subspace_W_FT', (2 * k, n_free), dtype)
            np.take(rc_full, idx_F, out=rc_F)
            np.take(Wt, idx_F, axis=1, out=W_FT)   # (2k, |F|)

        N = self._scratch_array('_subspace_N', (2 * k, 2 * k), dtype)
        np.matmul(W_FT, W_FT.T, out=N)       # (2k, 2k)
        N /= theta
        N -= M
        WF_rcF = self._scratch_array('_subspace_WF_rcF', (2 * k,), dtype)
        np.matmul(W_FT, rc_F, out=WF_rcF)
        # One-shot solve here; not worth factoring N up front.  Using
        # np.linalg.solve (vs linalg.solve) sidesteps scipy's noisy
        # ill-conditioning warnings — N can be poorly scaled near
        # convergence without that being a correctness problem.
        N_inv = np.linalg.solve(N, WF_rcF)
        # Δx_F = -(1/θ) r_c + (1/θ²) W_F N⁻¹ W_Fᵀ r_c
        dx_F = self._scratch_array('_subspace_dx_F', (n_free,), dtype)
        np.multiply(rc_F, -1.0 / theta, out=dx_F)
        WF_N_inv = self._scratch_array('_subspace_WF_N_inv', (n_free,), dtype)
        np.matmul(W_FT.T, N_inv, out=WF_N_inv)
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

        The returned array is a reusable scratch buffer, valid until the
        next call.
        """
        k = self._k
        q = self._scratch_array('_two_loop_q', (self.n,), g.dtype)
        tmp = self._scratch_array('_two_loop_tmp', (self.n,), g.dtype)
        q[:] = g
        alphas = self._alphas
        S = self.S
        Y = self.Y
        rho = self.rho
        dot = np.dot
        for i in range(k - 1, -1, -1):
            slot = self._hist_slot(i)
            alpha_i = dot(S[slot], q) * rho[slot]
            np.multiply(Y[slot], alpha_i, out=tmp)
            q -= tmp
            alphas[i] = alpha_i

        if k > 0:
            slot = self._hist_slot(k - 1)
            y_last = Y[slot]
            q *= self.ys[slot] / dot(y_last, y_last)  # gamma = 1/theta

        for i in range(k):
            slot = self._hist_slot(i)
            beta_i = dot(Y[slot], q) * rho[slot]
            np.multiply(S[slot], alphas[i] - beta_i, out=tmp)
            q += tmp

        np.negative(q, out=q)
        return q

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
            # Unbounded case: two-loop recursion gives -H_k g directly.
            p = self._two_loop(g)
            alpha_max = None
            n_free = self.n
            cauchy_breaks = 0
            subspace_mode = 'unbounded'
        else:
            # Wt, M, and M_lu are shared by the Cauchy and subspace stages.
            if self._k > 0:
                Wt, M = self._WM()
                M_lu = linalg.lu_factor(M, check_finite=False)
            else:
                Wt = None
                M = None
                M_lu = None

            xc, c_vec, free_mask = self._cauchy(g, Wt, M, M_lu)
            n_free = self.n - self._cauchy_breaks
            dx = self._subspace_solve(xc, c_vec, free_mask, g, Wt, M, M_lu,
                                      n_free=n_free)
            x_hat = self._scratch_array('_step_x_hat', (self.n,), self.x.dtype)
            np.add(xc, dx, out=x_hat)

            # Morales-Nocedal projection, with BLNZ truncation as fallback.
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
            cauchy_breaks = self._cauchy_breaks

        # No room to move, or the direction degenerated to zero.
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

        # Reuse the gradient already computed by the Wolfe search.
        if g_new.ndim != 1:
            g_new = g_new.ravel()

        s = self._scratch_array('_step_s', (self.n,), self.x.dtype)
        np.multiply(p, alpha, out=s)
        x_new = self.x + s
        if self._has_bounds:
            # Clip possible roundoff beyond a bound.
            np.maximum(x_new, self.l, out=x_new)
            np.minimum(x_new, self.u, out=x_new)
            np.subtract(x_new, self.x, out=s)

        y = self._scratch_array('_step_y', (self.n,), self.x.dtype)
        np.subtract(g_new, g, out=y)
        self._update_history(s, y)

        self.x = x_new
        self.iter += 1
        self.last_step_metadata = {
            'alpha': float(alpha),
            'free_vars': n_free,
            'cauchy_breaks': cauchy_breaks,
            'subspace_solved': True,
            'subspace_mode': subspace_mode,
        }
        return x_pre, float(f), g
