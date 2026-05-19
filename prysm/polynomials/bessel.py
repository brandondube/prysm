"""Integer-order Bessel functions."""
from functools import wraps
from math import lgamma

from prysm.mathops import np

from prysm.conf import config

from . import _bessel_coeffs as _c


_SQRT_TWO_OVER_PI = 0.79788456080286535588
_PI_OVER_4 = 0.78539816339744830962
_3PI_OVER_4 = 2.35619449019234492885


def _is_scalar(x):
    return not hasattr(x, 'shape') and not hasattr(x, '__len__')


def _as_float_array(x):
    """Return x as a backend floating array."""
    return np.asarray(x, dtype=config.precision)


def _scalar_safe(fn):
    """Marshal scalar x to array on input, unwrap on output.

    The wrapped function must take x as its last positional argument and
    return a single array.  Functions returning tuples (e.g. besselj_adjacent)
    do not use this decorator.
    """
    @wraps(fn)
    def wrapper(*args):
        x = args[-1]
        scalar = _is_scalar(x)
        out = fn(*args[:-1], _as_float_array(x))
        return out.item() if scalar else out
    return wrapper


def _series_tol(x):
    try:
        return 50 * np.finfo(x.dtype).eps
    except (TypeError, ValueError):
        return 1e-14


def _horner_hl(coeffs, x):
    """Evaluate a polynomial via Horner; coeffs ordered high to low."""
    out = coeffs[0]
    for c in coeffs[1:]:
        out = out * x + c
    return out


def _horner_hl_one(coeffs, x):
    """Evaluate a high-to-low polynomial with implicit leading coefficient 1."""
    out = x + coeffs[0]
    for c in coeffs[1:]:
        out = out * x + c
    return out


def _chbevl(coeffs, x):
    """Evaluate a Chebyshev expansion using the Cephes coefficient ordering."""
    b0 = coeffs[0] + 0 * x
    b1 = 0 * x
    b2 = b1
    for c in coeffs[1:]:
        b2 = b1
        b1 = b0
        b0 = x * b1 - b2 + c
    return 0.5 * (b0 - b2)


def _j_asymptotic(ax, phase, p_num, p_den, q_num, q_den):
    """Evaluate the shared large-argument Cephes form for J_0 and J_1."""
    safe_ax = np.where(ax == 0, 1, ax)
    w = 5 / safe_ax
    q = w * w
    p = _horner_hl(p_num, q) / _horner_hl(p_den, q)
    q = _horner_hl(q_num, q) / _horner_hl_one(q_den, q)
    return _SQRT_TWO_OVER_PI / np.sqrt(safe_ax) * (
        np.cos(phase) * p - w * np.sin(phase) * q
    )


def _modified_besseli_cephes(x, small_coeffs, large_coeffs, odd=False):
    """Evaluate I_0 or I_1 from the Cephes Chebyshev expansions."""
    ax = np.abs(x)

    small = np.exp(ax) * _chbevl(small_coeffs, 0.5 * ax - 2)
    if odd:
        small = ax * small

    safe_ax = np.where(ax == 0, 1, ax)
    large = np.exp(ax) / np.sqrt(safe_ax) * _chbevl(large_coeffs, 32 / safe_ax - 2)
    out = np.where(ax <= 8, small, large)
    if odd:
        out = np.where(x < 0, -out, out)
    return out


def _besselk_cephes(x, small_coeffs, large_coeffs, modified_i, log_sign,
                    divide_small_poly=False):
    """Evaluate K_0 or K_1 from the Cephes Chebyshev expansions."""
    safe_x = np.where(x == 0, 1, x)
    x_for_small = np.where(x <= 2, x, 1)
    y_small = x_for_small * x_for_small

    small_poly = _chbevl(small_coeffs, y_small - 2)
    if divide_small_poly:
        small_poly = small_poly / safe_x
    small = log_sign * np.log(0.5 * safe_x) * modified_i(x_for_small) + small_poly

    large = np.exp(-safe_x) / np.sqrt(safe_x) * _chbevl(large_coeffs, 8 / safe_x - 2)
    out = np.where(x <= 2, small, large)
    return np.where(x == 0, np.inf, out)


def _besselj_series(n, x):
    """Power series for J_n(x), used where forward recurrence is unstable."""
    if n < 0:
        sign = -1 if n & 1 else 1
        return sign * _besselj_series(-n, x)

    if n == 0:
        term = np.ones_like(x)
    else:
        ax = np.abs(x)
        safe_ax = np.where(ax == 0, 1, ax)
        log_abs_term = n * np.log(0.5 * safe_ax) - lgamma(n + 1)
        term = np.exp(log_abs_term)
        term = np.where(ax == 0, 0, term)
        if n & 1:
            term = np.where(x < 0, -term, term)

    out = term
    y = -0.25 * x * x
    tol = _series_tol(x)
    for k in range(1, 200):
        term = term * y / (k * (n + k))
        out = out + term
        if bool(np.all(np.abs(term) <= tol * (1 + np.abs(out)))):
            break

    return out


def _miller_start(n):
    """Backward-recurrence start index for J_n: Miller heuristic."""
    return n + int((40 * n) ** 0.5) + 20


def _besselj_miller_seq(max_n, x):
    """Miller backward recurrence producing J_0(x)..J_max_n(x) in one pass.

    Callers must mask the result to elements where backward recurrence is
    appropriate; elements with very small or zero |x| return junk or zero and
    should be overwritten via series or direct evaluation.
    """
    ax = np.abs(x)
    safe_x = np.where(ax == 0, 1, x)
    m = _miller_start(max_n)
    jp1 = np.zeros_like(x)
    jcur = np.ones_like(x)
    out = np.zeros((max_n + 1, *x.shape), dtype=x.dtype)
    norm = np.zeros_like(x)
    if m % 2 == 0:
        norm = norm + 2 * jcur

    for k in range(m, 0, -1):
        mag = np.maximum(np.abs(jp1), np.maximum(np.abs(jcur), np.abs(norm)))
        scale = np.where(mag > 1e150, 1e-150, 1)
        if bool(np.any(scale != 1)):
            jp1 = jp1 * scale
            jcur = jcur * scale
            norm = norm * scale
            out = out * scale
        jm1 = (2 * k / safe_x) * jcur - jp1
        order = k - 1
        if order <= max_n:
            out[order] = jm1
        if order == 0:
            norm = norm + jm1
        elif order % 2 == 0:
            norm = norm + 2 * jm1
        jp1, jcur = jcur, jm1

    out = out / norm
    return np.where(ax == 0, np.zeros_like(x), out)


def _besselj_miller(n, x):
    """Miller backward recurrence for a single J_n(x), O(1) memory.

    Tracks only the target order's value rather than allocating the full
    (n+1, *x.shape) array required by the seq version.
    """
    if n < 0:
        sign = -1 if n & 1 else 1
        return sign * _besselj_miller(-n, x)
    if n == 0:
        return besselj0(x)
    if n == 1:
        return besselj1(x)

    ax = np.abs(x)
    safe_x = np.where(ax == 0, 1, x)
    m = _miller_start(n)
    jp1 = np.zeros_like(x)
    jcur = np.ones_like(x)
    target = np.zeros_like(x)
    norm = np.zeros_like(x)
    if m % 2 == 0:
        norm = norm + 2 * jcur

    for k in range(m, 0, -1):
        mag = np.maximum(np.abs(jp1), np.maximum(np.abs(jcur), np.abs(norm)))
        if bool(np.any(mag > 1e150)):
            scale = np.where(mag > 1e150, 1e-150, 1)
            jp1 = jp1 * scale
            jcur = jcur * scale
            norm = norm * scale
            target = target * scale
        jm1 = (2 * k / safe_x) * jcur - jp1
        order = k - 1
        if order == n:
            target = jm1
        if order == 0:
            norm = norm + jm1
        elif order % 2 == 0:
            norm = norm + 2 * jm1
        jp1, jcur = jcur, jm1

    out = target / norm
    return np.where(ax == 0, np.zeros_like(x), out)


@_scalar_safe
def besselj0(x):
    """Bessel function of the first kind of order 0."""
    ax = np.abs(x)
    z = ax * ax

    very_small = 1 - z / 4
    small = (
        (z - _c.J0_DR1) * (z - _c.J0_DR2)
        * _horner_hl(_c.J0_RP, z)
        / _horner_hl_one(_c.J0_RQ, z)
    )
    small = np.where(ax < 1e-5, very_small, small)

    large = _j_asymptotic(ax, ax - _PI_OVER_4, _c.J0_PP, _c.J0_PQ, _c.J0_QP, _c.J0_QQ)
    return np.where(ax <= 5, small, large)


@_scalar_safe
def besselj1(x):
    """Bessel function of the first kind of order 1."""
    ax = np.abs(x)
    z = x * x

    small = (
        x * (z - _c.J1_Z1) * (z - _c.J1_Z2)
        * _horner_hl(_c.J1_RP, z)
        / _horner_hl_one(_c.J1_RQ, z)
    )

    large = _j_asymptotic(ax, ax - _3PI_OVER_4, _c.J1_PP, _c.J1_PQ, _c.J1_QP, _c.J1_QQ)
    large = np.where(x < 0, -large, large)
    return np.where(ax <= 5, small, large)


def _besselj_forward_seq(max_n, x):
    """Return all J_n(x), n from 0 to max_n, by forward recurrence."""
    out = np.empty((max_n + 1, *x.shape), dtype=x.dtype)
    out[0] = besselj0(x)
    if max_n == 0:
        return out

    out[1] = besselj1(x)
    safe_x = np.where(x == 0, 1, x)
    for n in range(1, max_n):
        out[n + 1] = (2 * n / safe_x) * out[n] - out[n - 1]

    return out


def _besselj_positive_order(n, x):
    """Evaluate J_n(x) for n >= 0 using the stable region for each element."""
    if n == 0:
        return besselj0(x)
    if n == 1:
        return besselj1(x)

    ax = np.abs(x)
    out = np.zeros_like(x)

    series_mask = ax <= 0.5 * n
    if bool(np.any(series_mask)):
        series_x = np.where(series_mask, x, 0)
        out = np.where(series_mask, _besselj_series(n, series_x), out)
    miller_mask = (ax > 0.5 * n) & (ax <= n)
    if bool(np.any(miller_mask)):
        miller_x = np.where(miller_mask, x, n)
        out = np.where(miller_mask, _besselj_miller(n, miller_x), out)
    forward_mask = ax > n
    if bool(np.any(forward_mask)):
        forward_x = np.where(forward_mask, x, n + 1)
        out = np.where(forward_mask, _besselj_forward_seq(n, forward_x)[n], out)
    return out


def _besselj_all_orders(max_n, x):
    """Return J_0..J_max_n(x) using one batched call per stability region.

    Single Miller and forward passes cover their respective regions for all
    orders simultaneously; only the series region still loops over n, since
    each order has its own series.
    """
    out = np.empty((max_n + 1, *x.shape), dtype=x.dtype)
    out[0] = besselj0(x)
    if max_n == 0:
        return out
    out[1] = besselj1(x)
    if max_n == 1:
        return out

    ax = np.abs(x)

    forward_mask = ax > max_n
    if bool(np.any(forward_mask)):
        forward_x = np.where(forward_mask, x, max_n + 1)
        fseq = _besselj_forward_seq(max_n, forward_x)
        out = np.where(forward_mask[None], fseq, out)

    miller_mask = (ax > 0.5 * max_n) & (ax <= max_n)
    if bool(np.any(miller_mask)):
        miller_x = np.where(miller_mask, x, max_n)
        mseq = _besselj_miller_seq(max_n, miller_x)
        # leave out[0], out[1] alone: canonical besselj0/besselj1 are tighter
        # than the Miller-normalized values for those orders.
        out[2:] = np.where(miller_mask[None], mseq[2:], out[2:])

    series_mask = ax <= 0.5 * max_n
    if bool(np.any(series_mask)):
        series_x = np.where(series_mask, x, 0)
        for n in range(2, max_n + 1):
            out[n] = np.where(series_mask, _besselj_series(n, series_x), out[n])

    return out


def _modified_besseli0(x):
    """Modified Bessel function I_0(x), private helper for K_0."""
    return _modified_besseli_cephes(x, _c.I0_SMALL, _c.I0_LARGE)


def _modified_besseli1(x):
    """Modified Bessel function I_1(x), private helper for K_1."""
    return _modified_besseli_cephes(x, _c.I1_SMALL, _c.I1_LARGE, odd=True)


@_scalar_safe
def besselj(n, x):
    """Bessel function of the first kind of integer order n."""
    if n < 0:
        sign = -1 if n & 1 else 1
        return sign * _besselj_positive_order(-n, x)
    return _besselj_positive_order(n, x)


def besselj_seq(ns, x):
    """Bessel functions of the first kind for sorted integer orders ns."""
    x = _as_float_array(x)
    ns = list(ns)
    if len(ns) == 0:
        return np.empty((0, *x.shape), dtype=x.dtype)

    max_n = max(abs(n) for n in ns)
    all_orders = _besselj_all_orders(max_n, x)

    out = np.empty((len(ns), *x.shape), dtype=x.dtype)
    for idx, n in enumerate(ns):
        if n < 0:
            sign = -1 if n & 1 else 1
            out[idx] = sign * all_orders[-n]
        else:
            out[idx] = all_orders[n]
    return out


def besselj_adjacent(n, x):
    """Return J_{n-1}(x) and J_n(x) from one recurrence."""
    if n < 1:
        raise ValueError('n must be >= 1')

    vals = besselj_seq((n - 1, n), x)
    return vals[0], vals[1]


def besselj_ratio_jnm1(n, x):
    """Return J_{n-1}(x) / J_n(x), using the adjacent-order path."""
    jnm1, jn = besselj_adjacent(n, x)
    return jnm1 / jn


@_scalar_safe
def besselk0(x):
    """Modified Bessel function of the second kind of order 0."""
    return _besselk_cephes(x, _c.K0_SMALL, _c.K0_LARGE, _modified_besseli0, -1)


@_scalar_safe
def besselk1(x):
    """Modified Bessel function of the second kind of order 1."""
    return _besselk_cephes(
        x, _c.K1_SMALL, _c.K1_LARGE, _modified_besseli1, 1,
        divide_small_poly=True,
    )


@_scalar_safe
def besselk(n, x):
    """Modified Bessel function of the second kind of integer order n."""
    n = abs(n)
    if n == 0:
        return besselk0(x)
    if n == 1:
        return besselk1(x)

    safe_x = np.where(x == 0, 1, x)
    km2 = besselk0(x)
    km1 = besselk1(x)
    for order in range(1, n):
        k = km2 + (2 * order / safe_x) * km1
        km2, km1 = km1, k

    return np.where(x == 0, np.inf, km1)


def besselk_seq(ns, x):
    """Modified Bessel functions of the second kind for integer orders ns."""
    x = _as_float_array(x)
    ns = list(ns)
    if len(ns) == 0:
        return np.empty((0, *x.shape), dtype=x.dtype)

    max_n = max(abs(n) for n in ns)
    all_orders = np.empty((max_n + 1, *x.shape), dtype=x.dtype)
    all_orders[0] = besselk0(x)
    if max_n > 0:
        all_orders[1] = besselk1(x)
        safe_x = np.where(x == 0, 1, x)
        for order in range(1, max_n):
            all_orders[order + 1] = all_orders[order - 1] + (2 * order / safe_x) * all_orders[order]
        all_orders = np.where(x == 0, np.inf, all_orders)

    out = np.empty((len(ns), *x.shape), dtype=x.dtype)
    for idx, n in enumerate(ns):
        out[idx] = all_orders[abs(n)]
    return out


def besselk_adjacent(n, x):
    """Return K_{n-1}(x) and K_n(x) from one recurrence."""
    if n < 1:
        raise ValueError('n must be >= 1')

    vals = besselk_seq((n - 1, n), x)
    return vals[0], vals[1]


@_scalar_safe
def besselk_ratio_knm1(n, x):
    """Return K_{n-1}(x) / K_n(x) without forming high-order K_n values."""
    if n < 1:
        raise ValueError('n must be >= 1')

    safe_x = np.where(x == 0, 1, x)
    ratio = besselk0(x) / besselk1(x)
    for order in range(1, n):
        ratio = 1 / (ratio + 2 * order / safe_x)

    return np.where(x == 0, 0, ratio)
