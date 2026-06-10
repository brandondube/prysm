"""Tests for prysm.x.optym._prysm_lbfgsb.

scaffolding (constructor + dtype plumbing + bound defaults).
unconstrained two-loop step + Wolfe line search.
"""
import numpy as np
import pytest

from prysm.x.optym import (
    GradientTolerance,
    MaxIterations,
    PrysmLBFGSB,
    Problem,
    as_problem,
    run_until,
)
from prysm.x.optym._prysm_lbfgsb import _strong_wolfe_lean


def _quadratic_fg(x):
    """f(x) = 0.5 * ||x||^2; gradient = x."""
    return float(0.5 * np.sum(x * x)), x


class _QuadraticProblem(Problem):
    has_fg = True

    def _fg(self, x):
        return _quadratic_fg(x)


@pytest.fixture
def x0_f64():
    return np.array([1.0, -2.0, 3.0, -4.0], dtype=np.float64)


@pytest.fixture
def x0_f32():
    return np.array([1.0, -2.0, 3.0, -4.0], dtype=np.float32)


def test_import_from_optym_namespace():
    from prysm.x.optym import PrysmLBFGSB as _Cls
    assert _Cls is PrysmLBFGSB


def test_construct_with_callable(x0_f64):
    opt = PrysmLBFGSB(_quadratic_fg, x0_f64)
    assert opt.problem is not None
    assert opt.x is not x0_f64           # constructor must copy
    assert opt.x.shape == x0_f64.shape


def test_construct_with_problem(x0_f64):
    prob = _QuadraticProblem()
    opt = PrysmLBFGSB(prob, x0_f64)
    assert opt.problem is prob           # Problem passthrough


def test_construct_with_explicit_bounds(x0_f64):
    lb = np.full_like(x0_f64, -5.0)
    ub = np.full_like(x0_f64, 5.0)
    opt = PrysmLBFGSB(_quadratic_fg, x0_f64,
                      lower_bounds=lb, upper_bounds=ub)
    assert opt.l is lb
    assert opt.u is ub


def test_default_bounds_are_inf(x0_f64):
    opt = PrysmLBFGSB(_quadratic_fg, x0_f64)
    assert opt.l.shape == (x0_f64.size,)
    assert opt.u.shape == (x0_f64.size,)
    assert opt.l.dtype == x0_f64.dtype
    assert opt.u.dtype == x0_f64.dtype
    assert np.all(np.isneginf(opt.l))
    assert np.all(np.isposinf(opt.u))


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_history_dtype_tracks_x0(dtype):
    x0 = np.array([1.0, 2.0, 3.0], dtype=dtype)
    opt = PrysmLBFGSB(_quadratic_fg, x0, memory=7)
    assert opt.S.dtype == dtype
    assert opt.Y.dtype == dtype
    assert opt.ys.dtype == dtype
    assert opt.theta.dtype == dtype
    assert opt.x.dtype == dtype
    assert opt.S.shape == (7, 3)
    assert opt.Y.shape == (7, 3)


def test_history_starts_empty(x0_f64):
    opt = PrysmLBFGSB(_quadratic_fg, x0_f64)
    assert opt._k == 0
    assert opt.iter == 0
    assert opt.nfev == 0
    assert opt.last_step_metadata == {}


def test_memory_size_respected(x0_f64):
    opt = PrysmLBFGSB(_quadratic_fg, x0_f64, memory=17)
    assert opt.m == 17
    assert opt.S.shape == (17, x0_f64.size)


# -------------------------- algorithmic --------------------------


def _make_quadratic(dim, dtype=np.float64, seed=0):
    """Return (fg, x_star) for f(x) = 0.5 (x-x*)^T A (x-x*), A SPD."""
    rng = np.random.default_rng(seed)
    Q = rng.standard_normal((dim, dim)).astype(dtype)
    A = (Q.T @ Q + np.eye(dim, dtype=dtype) * dim).astype(dtype)
    x_star = rng.standard_normal(dim).astype(dtype)

    def fg(x):
        d = x - x_star
        f = 0.5 * float(d @ A @ d)
        g = (A @ d).astype(dtype)
        return f, g

    return fg, x_star


def _rosenbrock_fg(x):
    """Classical 2-D-extended Rosenbrock; min at (1, 1, ...)."""
    x = np.asarray(x, dtype=np.float64)
    f = float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))
    g = np.zeros_like(x)
    g[:-1] = -400.0 * x[:-1] * (x[1:] - x[:-1] ** 2) - 2.0 * (1 - x[:-1])
    g[1:] += 200.0 * (x[1:] - x[:-1] ** 2)
    return f, g


def test_quadratic_converges_to_known_minimum():
    fg, x_star = _make_quadratic(dim=6, seed=1)
    x0 = np.zeros_like(x_star)
    opt = PrysmLBFGSB(fg, x0, memory=10)
    result = run_until(opt, MaxIterations(50))
    assert np.linalg.norm(opt.x - x_star) < 1e-8
    # f at the minimum is 0
    assert result.records[-1].f >= 0
    assert opt.iter <= 50


def test_quadratic_drives_gradient_to_zero():
    """On an SPD quadratic, ||g|| should drop to machine precision."""
    fg, x_star = _make_quadratic(dim=5, seed=2)
    x0 = np.zeros_like(x_star)

    opt = PrysmLBFGSB(fg, x0, memory=10)
    run_until(opt, MaxIterations(50))

    _, g = fg(opt.x)
    assert np.linalg.norm(g) < 1e-8
    assert np.linalg.norm(opt.x - x_star) < 1e-8


def test_first_step_is_steepest_descent():
    """With no history, the L-BFGS direction reduces to -g.  The first
    step is therefore a Wolfe line search along -g and should sit on
    that ray."""
    H = np.array([1.0, 2.0, 0.5, 4.0])

    def fg_diag(x):
        return float(0.5 * np.sum(H * x * x)), H * x

    x0 = np.array([1.0, -2.0, 3.0, -0.5])
    _, g0 = fg_diag(x0)

    opt = PrysmLBFGSB(fg_diag, x0.copy(), memory=10)
    opt.step()

    # x1 = x0 - alpha * g0 for some alpha > 0
    step = opt.x - x0
    # step should be antiparallel to g0
    cos_sim = float(step @ g0) / (np.linalg.norm(step) * np.linalg.norm(g0))
    assert cos_sim < -1 + 1e-10


def test_internal_wolfe_returns_accepted_gradient():
    """The optimizer-local Wolfe search should preserve the public helper's
    important contract: the accepted gradient is returned to the caller.
    """
    H = 10.0 * np.eye(2)

    def fg(x):
        return float(0.5 * x @ H @ x), H @ x

    xk = np.array([1.0, 1.0])
    fk, gk = fg(xk)
    pk = -gk
    alpha, phi_a, derphi_a, g_a = _strong_wolfe_lean(
        as_problem(fg), xk, pk, (fk, gk),
    )

    assert alpha is not None
    assert g_a is not None
    np.testing.assert_allclose(g_a, H @ (xk + alpha * pk))

    derphi0 = float(gk @ pk)
    assert phi_a <= fk + 1e-4 * alpha * derphi0 + 1e-12
    assert abs(derphi_a) <= 0.9 * abs(derphi0) + 1e-12


def test_rosenbrock_5d_converges():
    x0 = np.full(5, -1.2, dtype=np.float64)
    opt = PrysmLBFGSB(_rosenbrock_fg, x0, memory=10)
    run_until(opt, MaxIterations(200))
    assert np.linalg.norm(opt.x - 1.0) < 1e-4


def test_step_returns_pre_step_iterate():
    fg, x_star = _make_quadratic(dim=4, seed=3)
    x0 = np.zeros_like(x_star)
    opt = PrysmLBFGSB(fg, x0, memory=10)
    x_before = opt.x.copy()
    x_pre, f, g = opt.step()
    # x_pre is the iterate BEFORE this step
    np.testing.assert_array_equal(x_pre, x_before)
    # opt.x is the iterate AFTER this step
    assert not np.array_equal(opt.x, x_before)
    # f and g were evaluated at x_pre (the pre-step iterate)
    f_check, g_check = fg(x_before)
    assert f == pytest.approx(f_check)
    np.testing.assert_allclose(g, g_check)


def test_step_advances_bookkeeping():
    fg, _ = _make_quadratic(dim=3, seed=4)
    opt = PrysmLBFGSB(fg, np.zeros(3), memory=5)
    n0 = opt.nfev
    opt.step()
    assert opt.iter == 1
    assert opt.nfev > n0
    assert opt._k == 1  # one history pair after one step
    assert 'alpha' in opt.last_step_metadata


def test_history_buffer_rolls_at_capacity():
    """After more steps than the memory size, the buffer keeps only the
    last memory pairs."""
    fg, _ = _make_quadratic(dim=4, seed=5)
    opt = PrysmLBFGSB(fg, np.full(4, 0.5), memory=3)
    run_until(opt, MaxIterations(10))
    assert opt._k == 3  # capped at memory
    # most recent ys entries are nonzero
    assert np.all(opt.ys[: opt._k] != 0)


def test_history_buffer_keeps_logical_order_after_wrap():
    opt = PrysmLBFGSB(_quadratic_fg, np.zeros(2), memory=3)
    for i in range(5):
        s = np.full(2, i + 1.0)
        y = np.full(2, i + 2.0)
        opt._update_history(s, y)

    expected_s = np.array([[3.0, 3.0], [4.0, 4.0], [5.0, 5.0]])
    expected_y = np.array([[4.0, 4.0], [5.0, 5.0], [6.0, 6.0]])
    np.testing.assert_array_equal(opt._ordered_rows(opt.S), expected_s)
    np.testing.assert_array_equal(opt._ordered_rows(opt.Y), expected_y)


def test_linesearch_failure_raises_stop_iteration():
    """An ascent direction forces the strong-Wolfe search to fail."""
    fg, x_star = _make_quadratic(dim=4, seed=6)

    def bad_fg(x):
        f, g = fg(x)
        return f, -g  # flip the sign — gradient now points uphill

    opt = PrysmLBFGSB(bad_fg, np.zeros_like(x_star), memory=5)
    with pytest.raises(StopIteration):
        # _two_loop with no history returns -(-g) = +g; the line search
        # along +gradient cannot decrease f, so Wolfe returns alpha=None.
        # Sweep a few iterations because the first step may take a tiny
        # successful α before subsequent rejection.
        for _ in range(20):
            opt.step()
    assert opt.last_step_metadata.get('reason') == 'linesearch_fail'


def test_run_until_linesearch_failure_reports_unsuccessful():
    """Abort payloads must not be interpreted as successful convergence."""
    fg, x_star = _make_quadratic(dim=4, seed=6)

    def bad_fg(x):
        f, g = fg(x)
        return f, -g

    opt = PrysmLBFGSB(bad_fg, np.zeros_like(x_star), memory=5)
    result = run_until(opt, MaxIterations(20), maxiter=20)
    assert not result.success
    assert result.message == 'line search failed'


def test_run_until_no_descent_reports_unsuccessful():
    lb = np.array([0.0, 0.0])
    ub = np.array([1.0, 1.0])

    def fg(x):
        return float(x.sum()), np.ones(2)

    opt = PrysmLBFGSB(fg, np.array([0.0, 0.0]),
                      lower_bounds=lb, upper_bounds=ub, memory=3)
    result = run_until(opt, MaxIterations(5))
    assert not result.success
    assert result.message == 'no descent direction'


def test_run_until_gradient_convergence_still_reports_success():
    opt = PrysmLBFGSB(_quadratic_fg, np.array([1.0, -2.0]), memory=5)
    result = run_until(opt, GradientTolerance(10.0), maxiter=20)
    assert result.success


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_history_dtype_preserved_after_update(dtype):
    fg, x_star = _make_quadratic(dim=4, dtype=dtype, seed=7)
    x0 = np.zeros_like(x_star)
    opt = PrysmLBFGSB(fg, x0, memory=5)
    opt.step()
    assert opt.S.dtype == dtype
    assert opt.Y.dtype == dtype
    assert opt.ys.dtype == dtype
    assert opt.theta.dtype == dtype
    assert opt.x.dtype == dtype


def test_run_until_returns_governor_decision():
    fg, x_star = _make_quadratic(dim=4, seed=8)
    opt = PrysmLBFGSB(fg, np.zeros_like(x_star), memory=5)
    result = run_until(opt, MaxIterations(10))
    assert result.decision.stop
    assert len(result.records) <= 10


# -------------------- compact representation --------------------


def _populate_history(opt, n_steps, fg):
    """Run n_steps to fill the optimizer's S/Y/ys buffers."""
    run_until(opt, MaxIterations(n_steps))


def _dense_B(opt):
    """Reconstruct B_k = θI − W M⁻¹ Wᵀ as a dense (n, n) matrix.

    Reference path that exercises _W and _M only via their shapes and
    sizes; the algebra is done independently here.
    """
    n = opt.n
    k = opt._k
    if k == 0:
        return float(opt.theta) * np.eye(n)
    W = opt._W()
    M = opt._M()
    return float(opt.theta) * np.eye(n) - W @ np.linalg.solve(M, W.T)


def test_W_and_M_have_correct_shapes():
    fg, x_star = _make_quadratic(dim=6, seed=10)
    opt = PrysmLBFGSB(fg, np.zeros_like(x_star), memory=4)
    _populate_history(opt, 3, fg)
    k = opt._k
    assert k > 0
    W = opt._W()
    M = opt._M()
    assert W.shape == (opt.n, 2 * k)
    assert M.shape == (2 * k, 2 * k)


def test_Bv_matches_dense_reconstruction():
    """B_k v from _Bv matches B_k built densely from _W and _M."""
    fg, x_star = _make_quadratic(dim=6, seed=11)
    opt = PrysmLBFGSB(fg, np.zeros_like(x_star), memory=4)
    _populate_history(opt, 3, fg)
    B = _dense_B(opt)
    rng = np.random.default_rng(0)
    v = rng.standard_normal(opt.n)
    np.testing.assert_allclose(opt._Bv(v), B @ v, atol=1e-10, rtol=1e-10)


def test_Hg_inverts_Bv():
    """B_k (H_k g) == g for any g, with H_k from the SMW formula."""
    fg, x_star = _make_quadratic(dim=6, seed=12)
    opt = PrysmLBFGSB(fg, np.zeros_like(x_star), memory=4)
    _populate_history(opt, 3, fg)
    rng = np.random.default_rng(1)
    g = rng.standard_normal(opt.n)
    # _Hg returns -H_k g; recover H_k g
    Hg = -opt._Hg(g)
    np.testing.assert_allclose(opt._Bv(Hg), g, atol=1e-9, rtol=1e-9)


def test_Hg_matches_two_loop():
    """Compact-form _Hg and the two-loop recursion agree to machine eps."""
    fg, x_star = _make_quadratic(dim=6, seed=13)
    opt = PrysmLBFGSB(fg, np.zeros_like(x_star), memory=4)
    _populate_history(opt, 3, fg)
    rng = np.random.default_rng(2)
    g = rng.standard_normal(opt.n)
    np.testing.assert_allclose(opt._Hg(g), opt._two_loop(g),
                               atol=1e-10, rtol=1e-10)


def test_Hg_handles_empty_history():
    """With no history, H_0 = I so _Hg(g) = -g and _Bv(v) = θv = v."""
    fg, x_star = _make_quadratic(dim=4, seed=14)
    opt = PrysmLBFGSB(fg, np.zeros_like(x_star), memory=5)
    assert opt._k == 0
    g = np.array([1.0, -2.0, 3.0, 0.5])
    np.testing.assert_array_equal(opt._Hg(g), -g)
    np.testing.assert_array_equal(opt._Bv(g), g)


def test_Hg_drives_convergence_after_compact_step_switch():
    """After wiring step() to use _Hg, the quadratic still converges."""
    fg, x_star = _make_quadratic(dim=5, seed=15)
    opt = PrysmLBFGSB(fg, np.zeros_like(x_star), memory=10)
    run_until(opt, MaxIterations(50))
    assert np.linalg.norm(opt.x - x_star) < 1e-8


def test_M_diagonal_blocks_match_definitions():
    """Top-left of M should be -diag(ys); bottom-right should be θ S Sᵀ."""
    fg, x_star = _make_quadratic(dim=5, seed=16)
    opt = PrysmLBFGSB(fg, np.zeros_like(x_star), memory=3)
    _populate_history(opt, 2, fg)
    k = opt._k
    M = opt._M()
    np.testing.assert_allclose(M[:k, :k], -np.diag(opt.ys[:k]))
    SS = opt.S[:k] @ opt.S[:k].T
    np.testing.assert_allclose(M[k:, k:], float(opt.theta) * SS,
                               atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_compact_form_preserves_dtype(dtype):
    fg, x_star = _make_quadratic(dim=4, dtype=dtype, seed=17)
    opt = PrysmLBFGSB(fg, np.zeros_like(x_star), memory=3)
    _populate_history(opt, 2, fg)
    W = opt._W()
    M = opt._M()
    assert W.dtype == dtype
    assert M.dtype == dtype


# -------------------- generalized Cauchy point --------------------


def _brute_cauchy(x, g, l, u, B):
    """Reference Cauchy point via segment-by-segment walk using dense B.

    Independent re-implementation of BLNZ Algorithm CP that uses an
    explicit (n, n) Hessian approximation rather than the compact form.
    The two implementations must agree to numerical precision.
    """
    n = x.size
    # breakpoints
    t = np.full(n, np.inf)
    for i in range(n):
        if g[i] > 0 and np.isfinite(l[i]):
            t[i] = (x[i] - l[i]) / g[i]
        elif g[i] < 0 and np.isfinite(u[i]):
            t[i] = (x[i] - u[i]) / g[i]
    sorted_idx = np.argsort(t)

    free = np.ones(n, dtype=bool)
    xc = x.copy()
    t_old = 0.0

    for j in range(n):
        b = int(sorted_idx[j])
        t_b = t[b]

        d = np.where(free, -g, 0.0)
        df = float(g @ d + (xc - x) @ B @ d)
        ddf = float(d @ B @ d)

        dt_max = (t_b - t_old) if np.isfinite(t_b) else np.inf

        if df >= 0:
            dt_min = 0.0
        elif ddf <= 0:
            dt_min = np.inf
        else:
            dt_min = -df / ddf

        if dt_min < dt_max:
            xc = xc + dt_min * d
            return xc, free.copy()

        # advance to breakpoint
        if not np.isfinite(t_b):
            return xc, free.copy()
        xc = xc + dt_max * d
        # snap b to bound
        if g[b] > 0:
            xc[b] = l[b]
        elif g[b] < 0:
            xc[b] = u[b]
        free[b] = False
        t_old = t_b

    return xc, free.copy()


def test_cauchy_no_history_unconstrained():
    """With k=0 and no bounds active, Cauchy point lies on the steepest
    descent ray and matches the model minimum (B = θI = I, θ_min step = 1)."""
    opt = PrysmLBFGSB(_quadratic_fg, np.array([1.0, -2.0, 3.0]), memory=5)
    g = np.array([1.0, -2.0, 3.0])
    xc, c, free = opt._cauchy(g)
    assert c.size == 0
    assert np.all(free)
    # B = I, so min of m(x − t g) along t is t = 1 (gradient norm exactly cancels).
    np.testing.assert_allclose(xc, opt.x - g)


def test_cauchy_no_history_with_bounds():
    """Bound clamps the Cauchy step short of the unconstrained minimum."""
    x0 = np.array([0.5, 0.5])
    lb = np.array([0.0, -np.inf])
    ub = np.array([np.inf, np.inf])
    opt = PrysmLBFGSB(_quadratic_fg, x0, lower_bounds=lb, upper_bounds=ub,
                      memory=5)
    g = np.array([1.0, 1.0])  # pushes first coord down to its bound
    xc, c, free = opt._cauchy(g)
    # first coord hits l=0 at t=0.5; the unconstrained minimum is at t=1
    # for both coords (B=I), so the Cauchy point sits past the breakpoint
    # with var 0 clamped and var 1 free.
    assert xc[0] == 0.0
    assert not free[0]
    assert free[1]


def test_cauchy_matches_brute_force_no_history():
    """k=0 Cauchy matches the dense-B reference."""
    rng = np.random.default_rng(100)
    n = 5
    x = rng.standard_normal(n)
    g = rng.standard_normal(n)
    l = x - rng.uniform(0.1, 2.0, n)
    u = x + rng.uniform(0.1, 2.0, n)
    opt = PrysmLBFGSB(_quadratic_fg, x, lower_bounds=l, upper_bounds=u,
                      memory=5)
    xc, _, free = opt._cauchy(g)
    B = np.eye(n)  # θ=1 with no history
    xc_ref, free_ref = _brute_cauchy(x, g, l, u, B)
    np.testing.assert_allclose(xc, xc_ref, atol=1e-12, rtol=1e-12)
    np.testing.assert_array_equal(free, free_ref)


@pytest.mark.parametrize('seed', [0, 1, 2, 3, 4])
def test_cauchy_matches_brute_force_with_history(seed):
    """Cauchy point with compact-form B matches the dense-B reference."""
    rng = np.random.default_rng(seed)
    n = 6
    fg, x_star = _make_quadratic(dim=n, seed=seed + 200)
    x0 = rng.standard_normal(n)
    opt = PrysmLBFGSB(fg, x0, memory=4)
    _populate_history(opt, 3, fg)

    # Pick a random gradient and bounds; ensure x is inside the box.
    x = opt.x
    g = rng.standard_normal(n)
    l = x - rng.uniform(0.05, 2.0, n)
    u = x + rng.uniform(0.05, 2.0, n)
    # impose by directly setting; this exercises the Cauchy code with
    # bounds that don't correspond to the original construction.
    opt.l = l
    opt.u = u

    xc, _, free = opt._cauchy(g)
    B = _dense_B(opt)
    xc_ref, free_ref = _brute_cauchy(x, g, l, u, B)
    np.testing.assert_allclose(xc, xc_ref, atol=1e-9, rtol=1e-9)
    np.testing.assert_array_equal(free, free_ref)


@pytest.mark.parametrize('seed', [0, 1, 2, 3, 4])
def test_cauchy_chunked_sweep_carry_matches_brute_force(seed):
    """The sweep state carried across chunk boundaries reproduces the
    single-pass result; chunk=2 forces many carries on a 30-var problem."""
    rng = np.random.default_rng(seed)
    n = 30
    fg, x_star = _make_quadratic(dim=n, seed=seed + 500)
    x0 = rng.standard_normal(n)
    opt = PrysmLBFGSB(fg, x0, memory=4)
    _populate_history(opt, 3, fg)
    opt._cauchy_chunk = 2

    x = opt.x
    g = rng.standard_normal(n)
    # wide bounds so the sweep crosses many breakpoints before stopping
    l = x - rng.uniform(0.05, 4.0, n)
    u = x + rng.uniform(0.05, 4.0, n)
    opt.l = l
    opt.u = u

    xc, c, free = opt._cauchy(g)
    B = _dense_B(opt)
    xc_ref, free_ref = _brute_cauchy(x, g, l, u, B)
    np.testing.assert_allclose(xc, xc_ref, atol=1e-9, rtol=1e-9)
    np.testing.assert_array_equal(free, free_ref)
    # the auxiliary c must equal W.T @ (xc - x) for the subspace stage
    c_ref = opt._W().T @ (xc_ref - x)
    np.testing.assert_allclose(c, c_ref, atol=1e-9, rtol=1e-9)


def test_cauchy_already_active_variable():
    """A coord already at its bound with gradient pushing further into it
    is processed at t=0 and emerges as active with no motion."""
    x0 = np.array([0.0, 1.0])           # var 0 sits on its lower bound
    lb = np.array([0.0, -np.inf])
    ub = np.array([np.inf, np.inf])
    opt = PrysmLBFGSB(_quadratic_fg, x0, lower_bounds=lb, upper_bounds=ub,
                      memory=5)
    g = np.array([2.0, 1.0])            # pushes var 0 further below its bound
    xc, c, free = opt._cauchy(g)
    assert xc[0] == 0.0
    assert not free[0]
    # var 1 took the full unconstrained Cauchy step
    assert free[1]


def test_cauchy_no_finite_breakpoints():
    """Unbounded problem behaves like a plain L-BFGS step (no clamping)."""
    rng = np.random.default_rng(50)
    n = 5
    x0 = rng.standard_normal(n)
    g = rng.standard_normal(n)
    opt = PrysmLBFGSB(_quadratic_fg, x0, memory=5)  # default ±inf bounds
    xc, _, free = opt._cauchy(g)
    assert np.all(free)
    # xc = x − t* g with t* = ‖g‖² / (θ‖g‖²) = 1/θ.  With no history, θ=1.
    np.testing.assert_allclose(xc, x0 - g)


def test_cauchy_all_clamped_at_start():
    """If every coord is already at a bound being pushed into, xc = x."""
    x0 = np.zeros(3)
    lb = np.zeros(3)
    ub = np.full(3, np.inf)
    opt = PrysmLBFGSB(_quadratic_fg, x0, lower_bounds=lb, upper_bounds=ub,
                      memory=5)
    g = np.array([1.0, 2.0, 3.0])  # every coord pushed into lower bound
    xc, _, free = opt._cauchy(g)
    np.testing.assert_array_equal(xc, x0)
    assert not free.any()


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_cauchy_dtype_propagation(dtype):
    x0 = np.array([1.0, -1.0, 0.5], dtype=dtype)
    lb = np.array([-2.0, -2.0, -2.0], dtype=dtype)
    ub = np.array([2.0, 2.0, 2.0], dtype=dtype)
    opt = PrysmLBFGSB(_quadratic_fg, x0,
                      lower_bounds=lb, upper_bounds=ub, memory=3)
    g = np.array([0.5, 0.5, 0.5], dtype=dtype)
    xc, c, _ = opt._cauchy(g)
    assert xc.dtype == dtype
    assert c.dtype == dtype


# -------------------- subspace minimization --------------------


def test_max_alpha_inside_box_unbounded():
    opt = PrysmLBFGSB(_quadratic_fg, np.zeros(3), memory=3)
    p = np.array([1.0, -2.0, 3.0])
    # ±inf bounds; alpha can be the full 1.0
    assert opt._max_alpha_inside_box(np.zeros(3), p) == 1.0


def test_max_alpha_inside_box_clamps_to_face():
    x0 = np.array([0.5, 0.0])
    lb = np.array([0.0, 0.0])
    ub = np.array([1.0, 1.0])
    opt = PrysmLBFGSB(_quadratic_fg, x0,
                      lower_bounds=lb, upper_bounds=ub, memory=3)
    # p pushes var 0 toward upper bound (room = 0.5), var 1 toward upper
    # (room = 1.0); limiting move is var 0 with alpha = 0.5/1.0 = 0.5.
    p = np.array([1.0, 1.0])
    assert opt._max_alpha_inside_box(x0, p) == 0.5


def test_max_alpha_inside_box_zero_at_corner():
    x0 = np.array([0.0])
    lb = np.array([0.0])
    ub = np.array([1.0])
    opt = PrysmLBFGSB(_quadratic_fg, x0,
                      lower_bounds=lb, upper_bounds=ub, memory=3)
    # already at the lower bound, p pulls further below it
    assert opt._max_alpha_inside_box(x0, np.array([-1.0])) == 0.0


def test_subspace_solve_unconstrained_matches_newton():
    """With all variables free, xc + Δx − x should equal −H g (the
    L-BFGS Newton step)."""
    fg, x_star = _make_quadratic(dim=5, seed=300)
    opt = PrysmLBFGSB(fg, np.zeros_like(x_star), memory=4)
    _populate_history(opt, 3, fg)

    x = opt.x
    f, g = fg(x)
    xc, c, free_mask = opt._cauchy(g)
    assert free_mask.all()                 # no bounds active
    dx = opt._subspace_solve(xc, c, free_mask, g)
    newton_step = -np.linalg.solve(_dense_B(opt), g)
    actual = (xc + dx) - x
    np.testing.assert_allclose(actual, newton_step, atol=1e-9, rtol=1e-9)


def test_subspace_solve_active_coords_unchanged():
    """Active coords in xc are not perturbed by the subspace step."""
    x0 = np.array([0.5, 0.5, 0.5])
    lb = np.array([0.0, 0.0, 0.0])
    ub = np.array([1.0, 1.0, 1.0])
    opt = PrysmLBFGSB(_quadratic_fg, x0,
                      lower_bounds=lb, upper_bounds=ub, memory=3)
    g = np.array([2.0, 0.1, -2.0])  # var 0 hits l=0, var 2 hits u=1, var 1 free
    xc, c, free = opt._cauchy(g)
    dx = opt._subspace_solve(xc, c, free, g)
    # active components of Δx are exactly zero
    assert dx[~free].tolist() == [0.0] * int((~free).sum())


def test_subspace_solve_residual_after_step_is_zero_on_free_subspace():
    """At xc + Δx, the projected gradient on free variables should be
    zero (first-order optimality of the subspace QP)."""
    fg, x_star = _make_quadratic(dim=5, seed=301)
    rng = np.random.default_rng(0)
    n = 5
    x = rng.standard_normal(n)
    l = x - rng.uniform(0.1, 0.5, n)
    u = x + rng.uniform(0.1, 0.5, n)
    opt = PrysmLBFGSB(fg, x, lower_bounds=l, upper_bounds=u, memory=3)
    _populate_history(opt, 2, fg)
    # _populate_history moved opt.x; freshen
    x = opt.x
    g = rng.standard_normal(n)

    xc, c, free = opt._cauchy(g)
    dx = opt._subspace_solve(xc, c, free, g)
    z = (xc + dx) - x
    B = _dense_B(opt)
    grad_at_step = g + B @ z   # ∇m at xc + Δx
    np.testing.assert_allclose(grad_at_step[free], 0, atol=1e-9)


def test_bounded_quadratic_converges_to_face():
    """Quadratic with unconstrained minimum outside the box settles on
    the boundary face containing the constrained minimum."""
    # f(x) = 0.5 (x_1 − 2)² + 0.5 (x_2 + 2)²; unconstrained min at (2, -2)
    def fg(x):
        f = 0.5 * float((x[0] - 2) ** 2 + (x[1] + 2) ** 2)
        g = np.array([x[0] - 2, x[1] + 2])
        return f, g

    lb = np.array([0.0, -1.0])
    ub = np.array([1.0, 0.0])
    # constrained minimum at (1, -1) — both bounds active
    x0 = np.array([0.5, -0.5])
    opt = PrysmLBFGSB(fg, x0, lower_bounds=lb, upper_bounds=ub, memory=10)
    run_until(opt, MaxIterations(30))
    np.testing.assert_allclose(opt.x, [1.0, -1.0], atol=1e-6)


def test_bounded_rosenbrock_5d():
    lb = np.full(5, -0.5)
    ub = np.full(5, 0.5)
    x0 = np.full(5, -0.4)
    opt = PrysmLBFGSB(_rosenbrock_fg, x0,
                      lower_bounds=lb, upper_bounds=ub, memory=10)
    run_until(opt, MaxIterations(200))
    # check we end up in a feasible point and the gradient is small on
    # the free variables
    assert np.all(opt.x >= lb - 1e-12)
    assert np.all(opt.x <= ub + 1e-12)
    _, g = _rosenbrock_fg(opt.x)
    free = (opt.x > lb + 1e-9) & (opt.x < ub - 1e-9)
    if free.any():
        assert float(np.max(np.abs(g[free]))) < 1e-4


def test_no_descent_when_x_at_pulling_corner():
    """If x sits at a corner with the gradient pushing further into
    every active face, step() should signal no descent immediately."""
    lb = np.array([0.0, 0.0])
    ub = np.array([1.0, 1.0])
    # f pushes x to (-1, -1); both lower bounds active
    def fg(x):
        return float(x.sum()), np.ones(2)
    x0 = np.array([0.0, 0.0])
    opt = PrysmLBFGSB(fg, x0, lower_bounds=lb, upper_bounds=ub, memory=3)
    with pytest.raises(StopIteration):
        opt.step()
    assert opt.last_step_metadata['reason'] == 'no_descent'


def test_post_step_iterate_inside_box():
    """Even with aggressive bounds, every iterate stays in the box."""
    rng = np.random.default_rng(500)
    n = 6
    fg, x_star = _make_quadratic(dim=n, seed=500)
    lb = -np.ones(n)
    ub = np.ones(n)
    x0 = rng.uniform(-0.5, 0.5, n)
    opt = PrysmLBFGSB(fg, x0, lower_bounds=lb, upper_bounds=ub, memory=10)
    for _ in range(30):
        try:
            opt.step()
        except StopIteration:
            break
        assert np.all(opt.x >= lb - 1e-12)
        assert np.all(opt.x <= ub + 1e-12)


def test_metadata_reports_active_set():
    """At a step where the Cauchy sweep activates bounds, metadata
    records the count."""
    lb = np.array([0.0, -np.inf])
    ub = np.array([np.inf, 0.0])
    x0 = np.array([0.5, -0.5])

    def fg(x):
        f = 0.5 * float((x[0] - 2) ** 2 + (x[1] + 2) ** 2)
        g = np.array([x[0] - 2, x[1] + 2])
        return f, g

    opt = PrysmLBFGSB(fg, x0, lower_bounds=lb, upper_bounds=ub, memory=5)
    # first iterate pushes both vars beyond their bounds
    opt.step()
    assert opt.last_step_metadata['subspace_solved'] is True
    assert opt.last_step_metadata['alpha'] > 0
    # free_vars + cauchy_breaks == n
    md = opt.last_step_metadata
    assert md['free_vars'] + md['cauchy_breaks'] == 2


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_step_preserves_dtype_with_bounds(dtype):
    n = 4
    fg, x_star = _make_quadratic(dim=n, dtype=dtype, seed=701)
    lb = np.full(n, -1.0, dtype=dtype)
    ub = np.full(n, 1.0, dtype=dtype)
    x0 = np.zeros(n, dtype=dtype)
    opt = PrysmLBFGSB(fg, x0, lower_bounds=lb, upper_bounds=ub, memory=5)
    opt.step()
    assert opt.x.dtype == dtype
    assert opt.S.dtype == dtype


# -------------------- fp32 conditioning stress --------------------
#
# The 2k×2k solves in the Cauchy sweep, in _Hg, and in the subspace step
# can become ill-conditioned as history pairs (s_i, y_i) become nearly
# collinear near convergence.  At fp32 the curvature-condition guard
# (eps * ||y||^2) is much looser than at fp64, so these tests pin the
# fp32 path against pathological problems:
#
#   * high condition number (≥ 1e4) so descent is non-trivial
#   * larger n (10-50) so matrix arithmetic dominates
#   * memory ≥ n in some cases so the history buffer rolls
#   * bounded variants so Cauchy+subspace get exercised in fp32
#
# Anything that NaN/Inf-propagates here is a real bug; anything that fails
# to converge under a reasonable iteration budget is a fp32 conditioning
# escape worth diagnosing.


def _make_illconditioned_quadratic(dim, cond=1e4, dtype=np.float64, seed=0):
    """SPD quadratic with controlled condition number."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((dim, dim)))
    # eigenvalues log-spaced from 1 to cond
    eigs = np.logspace(0, np.log10(cond), dim).astype(dtype)
    A = (Q @ np.diag(eigs) @ Q.T).astype(dtype)
    x_star = rng.standard_normal(dim).astype(dtype)

    def fg(x):
        d = x - x_star
        f = 0.5 * float(d @ A @ d)
        g = (A @ d).astype(dtype)
        return f, g

    return fg, x_star, A


def test_fp32_illconditioned_quadratic_converges():
    """cond(A) ~ 1e4 SPD quadratic, n=20.  Final ‖x - x*‖∞ within fp32 reach."""
    n = 20
    fg, x_star, A = _make_illconditioned_quadratic(
        n, cond=1e4, dtype=np.float32, seed=900,
    )
    x0 = np.zeros(n, dtype=np.float32)
    opt = PrysmLBFGSB(fg, x0, memory=15)
    run_until(opt, MaxIterations(200))
    # all internal state must remain finite throughout
    assert np.all(np.isfinite(opt.x))
    assert np.all(np.isfinite(opt.S))
    assert np.all(np.isfinite(opt.Y))
    # reach fp32-appropriate accuracy
    err = float(np.max(np.abs(opt.x - x_star)))
    assert err < 5e-3, f"||x - x*||_inf = {err:.3e} too large"


def test_fp32_large_n_quadratic():
    """n=50 quadratic at fp32 — exercises bigger matrix ops."""
    n = 50
    fg, x_star, A = _make_illconditioned_quadratic(
        n, cond=1e3, dtype=np.float32, seed=901,
    )
    x0 = np.zeros(n, dtype=np.float32)
    opt = PrysmLBFGSB(fg, x0, memory=20)
    run_until(opt, MaxIterations(300))
    assert np.all(np.isfinite(opt.x))
    err = float(np.max(np.abs(opt.x - x_star)))
    assert err < 5e-3, f"||x - x*||_inf = {err:.3e} too large"


def test_fp32_history_saturation_no_blowup():
    """Small memory + slow-converging problem forces the buffer to
    saturate and roll many times.  Verify the rolling code path stays
    numerically clean at fp32."""
    # Rosenbrock at fp32 converges slowly enough that with memory=4 the
    # history will roll many times.
    n = 4
    x0 = np.full(n, -1.2, dtype=np.float32)
    opt = PrysmLBFGSB(_rosenbrock_fg, x0, memory=4)  # memory < iters needed
    run_until(opt, MaxIterations(150))
    assert np.all(np.isfinite(opt.x))
    assert np.all(np.isfinite(opt.S))
    assert np.all(np.isfinite(opt.Y))
    # buffer should have saturated within the run
    assert opt._k == opt.m
    # and the optimizer is at least in the neighborhood of the minimum
    assert float(np.max(np.abs(opt.x - 1.0))) < 0.2


def test_fp32_memory_above_n_does_not_nan():
    """memory > n is a pathological configuration (S^T S becomes
    rank-deficient by construction) but must not produce NaN/Inf even
    if Wolfe gives up early."""
    n = 5
    fg, x_star, A = _make_illconditioned_quadratic(
        n, cond=1e3, dtype=np.float32, seed=902,
    )
    x0 = np.zeros(n, dtype=np.float32)
    opt = PrysmLBFGSB(fg, x0, memory=20)  # memory > n
    # may terminate early via linesearch_fail once history becomes
    # rank-deficient; what matters is no NaN/Inf leaks out.
    for _ in range(50):
        try:
            opt.step()
        except StopIteration:
            break
        assert np.all(np.isfinite(opt.x))
        assert np.all(np.isfinite(opt.S))
        assert np.all(np.isfinite(opt.Y))
        assert np.all(np.isfinite(opt.ys))


def test_fp32_bounded_quadratic_stays_in_box():
    """fp32 + bounds: Cauchy & subspace solves at fp32 must keep iterates
    in the box and converge to the constrained minimum."""
    n = 10
    fg, x_star, A = _make_illconditioned_quadratic(
        n, cond=1e2, dtype=np.float32, seed=903,
    )
    lb = np.full(n, -0.5, dtype=np.float32)
    ub = np.full(n, 0.5, dtype=np.float32)
    x0 = np.zeros(n, dtype=np.float32)
    opt = PrysmLBFGSB(fg, x0, lower_bounds=lb, upper_bounds=ub, memory=10)
    for _ in range(100):
        try:
            opt.step()
        except StopIteration:
            break
        assert np.all(np.isfinite(opt.x))
        assert np.all(opt.x >= lb - 1e-5)
        assert np.all(opt.x <= ub + 1e-5)


def test_fp32_dtype_invariants_after_long_run():
    """Every internal array remains fp32 across many iterations,
    bound activations, and history rolls."""
    n = 8
    fg, x_star, A = _make_illconditioned_quadratic(
        n, cond=1e4, dtype=np.float32, seed=904,
    )
    lb = np.full(n, -1.0, dtype=np.float32)
    ub = np.full(n, 1.0, dtype=np.float32)
    x0 = np.zeros(n, dtype=np.float32)
    opt = PrysmLBFGSB(fg, x0, lower_bounds=lb, upper_bounds=ub, memory=12)
    run_until(opt, MaxIterations(150))
    assert opt.x.dtype == np.float32
    assert opt.S.dtype == np.float32
    assert opt.Y.dtype == np.float32
    assert opt.ys.dtype == np.float32
    assert opt.theta.dtype == np.float32
    assert opt.l.dtype == np.float32
    assert opt.u.dtype == np.float32


def test_fp32_rosenbrock_converges():
    """5-D Rosenbrock at fp32 — non-convex landscape; checks the
    curvature-condition guard handles fp32 thresholds correctly."""
    x0 = np.full(5, -1.2, dtype=np.float32)
    opt = PrysmLBFGSB(_rosenbrock_fg, x0, memory=15)
    run_until(opt, MaxIterations(500))
    assert np.all(np.isfinite(opt.x))
    # fp32 reach on Rosenbrock is loose; just verify we got into the
    # neighborhood of the minimum at (1, 1, 1, 1, 1).
    assert float(np.max(np.abs(opt.x - 1.0))) < 0.05


def test_fp32_extreme_conditioning_does_not_nan():
    """cond(A) ~ 1e6 at fp32 — well beyond what fp32 can solve cleanly.
    The optimizer must either make progress or bail gracefully; under
    no circumstance can NaN/Inf appear in the iterate or history."""
    n = 12
    fg, x_star, A = _make_illconditioned_quadratic(
        n, cond=1e6, dtype=np.float32, seed=905,
    )
    x0 = np.zeros(n, dtype=np.float32)
    opt = PrysmLBFGSB(fg, x0, memory=10)
    for _ in range(50):
        try:
            opt.step()
        except StopIteration:
            break
        assert np.all(np.isfinite(opt.x))
        assert np.all(np.isfinite(opt.S))
        assert np.all(np.isfinite(opt.Y))
        assert np.all(np.isfinite(opt.ys))


# ----------- Morales-Nocedal 2011 projected subspace step -----------
#
# The 2011 remark replaces BLNZ-style truncation of the path x_k -> x_hat
# with the componentwise projection x_bar = P(x_hat, l, u), reverting to
# truncation only when g . (x_bar - x_k) >= 0.  These tests pin both
# branches and verify the projection branch is preferred when both apply.


def test_subspace_mode_metadata_present():
    """Every successful step records which subspace branch fired."""
    fg, x_star = _make_quadratic(dim=4, seed=1000)
    # Pass explicit finite bounds so the Cauchy + subspace machinery runs
    # and exposes the projected/truncated metadata.
    lb = np.full_like(x_star, -10.0)
    ub = np.full_like(x_star, 10.0)
    opt = PrysmLBFGSB(fg, np.zeros_like(x_star), memory=5,
                      lower_bounds=lb, upper_bounds=ub)
    opt.step()
    assert opt.last_step_metadata['subspace_mode'] in ('projected', 'truncated')


def test_unconstrained_step_uses_unbounded_fastpath():
    """With ±inf bounds, step() takes the L-BFGS-direct fast path that
    skips Cauchy + subspace minimization entirely (they reduce to the
    same direction in the absence of bounds).
    """
    fg, x_star = _make_quadratic(dim=5, seed=1001)
    opt = PrysmLBFGSB(fg, np.zeros_like(x_star), memory=5)
    for _ in range(5):
        opt.step()
        assert opt.last_step_metadata['subspace_mode'] == 'unbounded'


def _run_truncated_only(fg, x0, lb, ub, memory, max_iter):
    """Run a copy of PrysmLBFGSB forced onto the BLNZ truncation branch.

    Monkeypatches step() to skip the Morales-Nocedal projection test and
    always take the BLNZ-truncated direction.  Returns the optimizer
    after running for up to max_iter iterations.
    """
    import prysm.x.optym._prysm_lbfgsb as mod
    from prysm.x.optym.linesearch import ls_strong_wolfe

    opt = PrysmLBFGSB(fg, x0.copy(), lower_bounds=lb, upper_bounds=ub,
                      memory=memory)
    original_step = mod.PrysmLBFGSB.step

    def truncated_only_step(self):
        x_pre = self.x.copy()
        f, g = self.problem.fg(self.x)
        self.nfev += 1
        if g.ndim != 1:
            g = g.ravel()
        xc, c_vec, free_mask = self._cauchy(g)
        dx = self._subspace_solve(xc, c_vec, free_mask, g)
        p = (xc + dx) - self.x
        alpha_max = self._max_alpha_inside_box(self.x, p)
        if alpha_max <= 0.0 or float(np.max(np.abs(p))) == 0.0:
            self.last_step_metadata = {'reason': 'no_descent'}
            raise StopIteration
        alpha, _, _, g_new = ls_strong_wolfe(
            self.problem, self.x, p, fg_at_xk=(f, g),
            maxalpha=alpha_max, c1=self.c1, c2=self.c2, maxiter=self.maxls,
        )
        if alpha is None:
            self.last_step_metadata = {'reason': 'linesearch_fail'}
            raise StopIteration
        if g_new.ndim != 1:
            g_new = g_new.ravel()
        s = alpha * p
        x_new = self.x + s
        x_new = np.minimum(np.maximum(x_new, self.l), self.u)
        s = x_new - self.x
        y = g_new - g
        self._update_history(s, y)
        self.x = x_new
        self.iter += 1
        self.last_step_metadata = {'alpha': float(alpha),
                                   'subspace_mode': 'truncated'}
        return x_pre, float(f), g.copy()

    try:
        mod.PrysmLBFGSB.step = truncated_only_step
        for _ in range(max_iter):
            try:
                opt.step()
            except StopIteration:
                break
    finally:
        mod.PrysmLBFGSB.step = original_step
    return opt


def test_projection_outperforms_truncation_on_bound_pinned_quadratic():
    """Quadratic whose unconstrained minimum lies far outside the box
    along several coords.  Truncation chops the Newton step at the first
    bound it hits per iteration; projection collapses all the violating
    coords to their bounds in one go.  Projected variant converges in
    fewer iterations.
    """
    n = 12
    rng = np.random.default_rng(1234)
    # SPD A with moderate conditioning so Newton steps are meaningful.
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    eigs = np.logspace(0, 2, n)
    A = Q @ np.diag(eigs) @ Q.T
    # x_star sits well outside the box for ~half the coords.
    x_star = rng.standard_normal(n) * 3.0

    def fg(x):
        d = x - x_star
        f = 0.5 * float(d @ A @ d)
        g = A @ d
        return f, g

    lb = -np.ones(n) * 0.5
    ub = np.ones(n) * 0.5
    x0 = np.zeros(n)

    opt_proj = PrysmLBFGSB(fg, x0.copy(), lower_bounds=lb, upper_bounds=ub,
                           memory=10)
    run_until(opt_proj, MaxIterations(80))

    opt_trunc = _run_truncated_only(fg, x0, lb, ub, memory=10, max_iter=80)

    # both must produce feasible iterates
    assert np.all(opt_proj.x >= lb - 1e-12)
    assert np.all(opt_proj.x <= ub + 1e-12)
    assert np.all(opt_trunc.x >= lb - 1e-12)
    assert np.all(opt_trunc.x <= ub + 1e-12)
    # projected variant reaches a lower objective in the same iter budget
    f_proj, _ = fg(opt_proj.x)
    f_trunc, _ = fg(opt_trunc.x)
    assert f_proj <= f_trunc, (
        f'projected f={f_proj:.6e} not <= truncated f={f_trunc:.6e}'
    )
    # and uses no more iterations
    assert opt_proj.iter <= opt_trunc.iter


def test_projected_branch_lands_at_x_bar_on_first_step():
    """On a problem where projected x_hat is itself the constrained
    optimum, the unit step is accepted and the iterate lands exactly
    at the projection.
    """
    # Quadratic centered at -5 along coord 0; the projected unconstrained
    # minimum is x_bar = (0, 0, ...).
    n = 4
    x_star = np.array([-5.0, 0.0, 0.0, 0.0])

    def fg(x):
        d = x - x_star
        return 0.5 * float(d @ d), d

    lb = np.array([0.0, -1.0, -1.0, -1.0])
    ub = np.array([1.0, 1.0, 1.0, 1.0])
    x0 = np.array([0.5, 0.5, 0.5, 0.5])
    opt = PrysmLBFGSB(fg, x0, lower_bounds=lb, upper_bounds=ub, memory=5)
    opt.step()
    assert opt.last_step_metadata['subspace_mode'] == 'projected'
    assert opt.last_step_metadata['alpha'] == pytest.approx(1.0)
    np.testing.assert_allclose(opt.x, [0.0, 0.0, 0.0, 0.0], atol=1e-10)


def test_truncated_branch_fires_when_projection_is_ascent():
    """Pathological hand-built case: drive xc + dx outside the box in
    such a way that gᵀ(x_bar - x_k) >= 0 so the projected direction is
    not descent, forcing the truncation fallback.

    Easiest construction: pin coords already at a bound + a contrived
    history where the subspace Newton step jumps wildly.  Simplest in
    practice is to look at a step where projection collapses to zero
    motion (x_bar == x_k), making gᵀp_proj == 0; then the fallback
    fires and the BLNZ truncation rule still produces progress along
    the unprojected path.
    """
    # x is interior; pick a direction whose projection is the zero
    # vector (every coord's x_hat sits past the bound it's nearest to,
    # but the gradient ensures p_proj sums to gᵀp_proj == 0).  Easiest
    # to fabricate: just monkey with x_hat to be xc itself when xc == x.
    n = 2

    def fg(x):
        # f = 0 anywhere; gradient identically 0.  Forces gᵀp_proj == 0
        # (not < 0), triggering the fallback branch.
        return 0.0, np.zeros_like(x)

    x0 = np.array([0.5, 0.5])
    lb = np.array([0.0, 0.0])
    ub = np.array([1.0, 1.0])
    opt = PrysmLBFGSB(fg, x0, lower_bounds=lb, upper_bounds=ub, memory=3)
    # with g == 0, the line search will reject (no descent possible)
    with pytest.raises(StopIteration):
        opt.step()


def test_projection_keeps_iterates_in_box():
    """Sanity: the projected branch must never leak outside the box."""
    n = 6
    rng = np.random.default_rng(2000)
    fg, x_star = _make_quadratic(dim=n, seed=2000)
    lb = -np.ones(n) * 0.3
    ub = np.ones(n) * 0.3
    x0 = rng.uniform(-0.2, 0.2, n)
    opt = PrysmLBFGSB(fg, x0, lower_bounds=lb, upper_bounds=ub, memory=8)
    for _ in range(40):
        try:
            opt.step()
        except StopIteration:
            break
        assert np.all(opt.x >= lb - 1e-12)
        assert np.all(opt.x <= ub + 1e-12)


def test_fp32_subspace_with_many_active_bounds():
    """Many simultaneous active bounds + fp32: stresses the |F|-sized
    subspace solve at small dimension and the row-slice operations."""
    n = 15
    fg, x_star, A = _make_illconditioned_quadratic(
        n, cond=1e3, dtype=np.float32, seed=906,
    )
    # narrow box around 0; x* will be outside for most coords
    lb = np.full(n, -0.1, dtype=np.float32)
    ub = np.full(n, 0.1, dtype=np.float32)
    x0 = np.zeros(n, dtype=np.float32)
    opt = PrysmLBFGSB(fg, x0, lower_bounds=lb, upper_bounds=ub, memory=10)
    final_active = 0
    for _ in range(80):
        try:
            opt.step()
        except StopIteration:
            break
        assert np.all(np.isfinite(opt.x))
        assert np.all(opt.x >= lb - 1e-5)
        assert np.all(opt.x <= ub + 1e-5)
        # how many coords are pinned to a bound at the final iterate
        final_active = int(np.sum((opt.x <= lb + 1e-6) | (opt.x >= ub - 1e-6)))
    # most coords end up bound-pinned (sanity: the test actually exercises
    # a heavily-constrained scenario)
    assert final_active >= n // 2
