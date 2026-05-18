"""Integer-order Bessel functions."""
from math import factorial

from prysm.mathops import np

from prysm.conf import config


_TWO_OVER_PI = 0.6366197723675813


def _as_float_array(x):
    """Return x as a backend floating array."""
    x = np.asarray(x, dtype=config.precision)
    if x.dtype.kind not in 'fc':
        x = x + 0.0
    return x


def _maybe_scalar(out, scalar):
    if scalar:
        return out.item()
    return out


def _series_tol(x):
    try:
        return 50 * np.finfo(x.dtype).eps
    except (TypeError, ValueError):
        return 1e-14


def _besselj_series(n, x):
    """Power series for J_n(x), used where forward recurrence is unstable."""
    x = _as_float_array(x)
    if n < 0:
        sign = -1 if n & 1 else 1
        return sign * _besselj_series(-n, x)

    if n == 0:
        term = np.ones_like(x)
    else:
        term = (0.5 * x) ** n / factorial(n)

    out = term.copy()
    y = -0.25 * x * x
    tol = _series_tol(x)
    for k in range(1, 200):
        term = term * y / (k * (n + k))
        out = out + term
        if bool(np.all(np.abs(term) <= tol * (1 + np.abs(out)))):
            break

    return out


def _besselj_miller(n, x):
    """Miller backward recurrence for J_n(x)."""
    if n < 0:
        sign = -1 if n & 1 else 1
        return sign * _besselj_miller(-n, x)
    if n == 0:
        return besselj0(x)
    if n == 1:
        return besselj1(x)

    x = _as_float_array(x)
    ax = np.abs(x)
    safe_x = np.where(ax <= 0.5 * n, 1e30, x)
    m = n + int((40 * n) ** 0.5) + 20
    jp1 = np.zeros_like(x)
    jcur = np.ones_like(x)
    wanted = jcur if m == n else None
    norm = np.zeros_like(x)
    if m % 2 == 0:
        norm = norm + 2 * jcur

    for k in range(m, 0, -1):
        jm1 = (2 * k / safe_x) * jcur - jp1
        order = k - 1
        if order == n:
            wanted = jm1
        if order == 0:
            norm = norm + jm1
        elif order % 2 == 0:
            norm = norm + 2 * jm1
        jp1, jcur = jcur, jm1

    out = wanted / norm
    return np.where(ax == 0, np.zeros_like(x), out)


def _replace_where(mask, original, replacement):
    return np.where(mask, replacement, original)


def _modified_besseli0(x):
    """Modified Bessel function I_0(x), private helper for K_0."""
    x = _as_float_array(x)
    ax = np.abs(x)

    # The small-|x| fit is a polynomial in (x / 3.75)^2; 3.75 is baked into
    # the approximation coefficients and matches the branch point below.
    y_small = (x / 3.75) ** 2
    small = (
        1.0
        + y_small * (
            3.5156229
            + y_small * (
                3.0899424
                + y_small * (
                    1.2067492
                    + y_small * (
                        0.2659732
                        + y_small * (0.0360768 + y_small * 0.0045813)
                    )
                )
            )
        )
    )

    safe_ax = np.where(ax == 0, 1, ax)
    # The large-|x| branch factors out exp(|x|) / sqrt(|x|), then corrects it
    # with a polynomial in 3.75 / |x|.
    y_large = 3.75 / safe_ax
    large = (
        np.exp(safe_ax)
        / np.sqrt(safe_ax)
        * (
            0.39894228
            + y_large * (
                0.01328592
                + y_large * (
                    0.00225319
                    + y_large * (
                        -0.00157565
                        + y_large * (
                            0.00916281
                            + y_large * (
                                -0.02057706
                                + y_large * (
                                    0.02635537
                                    + y_large * (-0.01647633 + y_large * 0.00392377)
                                )
                            )
                        )
                    )
                )
            )
        )
    )
    return np.where(ax < 3.75, small, large)


def _modified_besseli1(x):
    """Modified Bessel function I_1(x), private helper for K_1."""
    x = _as_float_array(x)
    ax = np.abs(x)

    # The small-|x| fit is odd, so the fitted even polynomial is multiplied by
    # x after using the same (x / 3.75)^2 scale as I_0.
    y_small = (x / 3.75) ** 2
    small = x * (
        0.5
        + y_small * (
            0.87890594
            + y_small * (
                0.51498869
                + y_small * (
                    0.15084934
                    + y_small * (
                        0.02658733
                        + y_small * (0.00301532 + y_small * 0.00032411)
                    )
                )
            )
        )
    )

    safe_ax = np.where(ax == 0, 1, ax)
    # The large-|x| branch uses the same exp(|x|) / sqrt(|x|) envelope as I_0,
    # with order-specific coefficients in 3.75 / |x|.
    y_large = 3.75 / safe_ax
    large = (
        np.exp(safe_ax)
        / np.sqrt(safe_ax)
        * (
            0.39894228
            + y_large * (
                -0.03988024
                + y_large * (
                    -0.00362018
                    + y_large * (
                        0.00163801
                        + y_large * (
                            -0.01031555
                            + y_large * (
                                0.02282967
                                + y_large * (
                                    -0.02895312
                                    + y_large * (0.01787654 - y_large * 0.00420059)
                                )
                            )
                        )
                    )
                )
            )
        )
    )
    large = np.where(x < 0, -large, large)
    return np.where(ax < 3.75, small, large)


def besselj0(x):
    """Bessel function of the first kind of order 0."""
    scalar = not hasattr(x, 'shape') and not hasattr(x, '__len__')
    x = _as_float_array(x)
    ax = np.abs(x)
    y = x * x

    # For |x| < 8, J_0 is approximated by a rational function in x^2.
    small_num = (
        57568490574.0
        + y * (
            -13362590354.0
            + y * (
                651619640.7
                + y * (
                    -11214424.18
                    + y * (77392.33017 + y * -184.9052456)
                )
            )
        )
    )
    small_den = (
        57568490411.0
        + y * (
            1029532985.0
            + y * (
                9494680.718
                + y * (59272.64853 + y * (267.8532712 + y))
            )
        )
    )
    small = small_num / small_den

    safe_ax = np.where(ax == 0, 1, ax)
    # For |x| >= 8, use the oscillatory asymptotic form with polynomial
    # corrections in (8 / |x|)^2.
    z = 8.0 / safe_ax
    y2 = z * z
    xx = safe_ax - 0.785398164
    large_p = (
        1.0
        + y2 * (
            -0.1098628627e-2
            + y2 * (
                0.2734510407e-4
                + y2 * (-0.2073370639e-5 + y2 * 0.2093887211e-6)
            )
        )
    )
    large_q = (
        -0.1562499995e-1
        + y2 * (
            0.1430488765e-3
            + y2 * (
                -0.6911147651e-5
                + y2 * (0.7621095161e-6 - y2 * 0.934935152e-7)
            )
        )
    )
    large = np.sqrt(_TWO_OVER_PI / safe_ax) * (np.cos(xx) * large_p - z * np.sin(xx) * large_q)

    out = np.where(ax < 8, small, large)
    return _maybe_scalar(out, scalar)


def besselj1(x):
    """Bessel function of the first kind of order 1."""
    scalar = not hasattr(x, 'shape') and not hasattr(x, '__len__')
    x = _as_float_array(x)
    ax = np.abs(x)
    y = x * x

    # For |x| < 8, J_1 is x times a rational function in x^2, preserving the
    # odd symmetry of the function.
    small_num = x * (
        72362614232.0
        + y * (
            -7895059235.0
            + y * (
                242396853.1
                + y * (
                    -2972611.439
                    + y * (15704.48260 + y * -30.16036606)
                )
            )
        )
    )
    small_den = (
        144725228442.0
        + y * (
            2300535178.0
            + y * (
                18583304.74
                + y * (99447.43394 + y * (376.9991397 + y))
            )
        )
    )
    small = small_num / small_den

    safe_ax = np.where(ax == 0, 1, ax)
    # For |x| >= 8, use the oscillatory asymptotic form with polynomial
    # corrections in (8 / |x|)^2.
    z = 8.0 / safe_ax
    y2 = z * z
    xx = safe_ax - 2.356194491
    large_p = (
        1.0
        + y2 * (
            0.183105e-2
            + y2 * (
                -0.3516396496e-4
                + y2 * (0.2457520174e-5 + y2 * -0.240337019e-6)
            )
        )
    )
    large_q = (
        0.04687499995
        + y2 * (
            -0.2002690873e-3
            + y2 * (
                0.8449199096e-5
                + y2 * (-0.88228987e-6 + y2 * 0.105787412e-6)
            )
        )
    )
    large = np.sqrt(_TWO_OVER_PI / safe_ax) * (np.cos(xx) * large_p - z * np.sin(xx) * large_q)
    large = np.where(x < 0, -large, large)

    out = np.where(ax < 8, small, large)
    return _maybe_scalar(out, scalar)


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


def besselj(n, x):
    """Bessel function of the first kind of integer order n."""
    scalar = not hasattr(x, 'shape') and not hasattr(x, '__len__')
    if n < 0:
        sign = -1 if n & 1 else 1
        return sign * besselj(-n, x)

    if n == 0:
        return besselj0(x)
    if n == 1:
        return besselj1(x)

    x = _as_float_array(x)
    ax = np.abs(x)
    seq = _besselj_forward_seq(n, x)
    out = seq[n]
    series_mask = ax <= 0.5 * n
    if bool(np.any(series_mask)):
        series_x = np.where(series_mask, x, 0)
        out = _replace_where(series_mask, out, _besselj_series(n, series_x))
    miller_mask = (ax > 0.5 * n) & (ax <= n)
    if bool(np.any(miller_mask)):
        miller_x = np.where(miller_mask, x, n)
        out = _replace_where(miller_mask, out, _besselj_miller(n, miller_x))
    return _maybe_scalar(out, scalar)


def besselj_seq(ns, x):
    """Bessel functions of the first kind for sorted integer orders ns."""
    if not hasattr(ns, '__len__'):
        ns = list(ns)
    x = _as_float_array(x)
    ns = list(ns)
    if len(ns) == 0:
        return np.empty((0, *x.shape), dtype=x.dtype)

    max_n = max(abs(n) for n in ns)
    forward = _besselj_forward_seq(max_n, x)
    ax = np.abs(x)
    out = np.empty((len(ns), *x.shape), dtype=x.dtype)
    for idx, n in enumerate(ns):
        sign = 1
        order = n
        if n < 0:
            order = -n
            sign = -1 if order & 1 else 1

        elem = forward[order]
        if order > 1:
            series_mask = ax <= 0.5 * order
            if bool(np.any(series_mask)):
                series_x = np.where(series_mask, x, 0)
                elem = _replace_where(series_mask, elem, _besselj_series(order, series_x))
            miller_mask = (ax > 0.5 * order) & (ax <= order)
            if bool(np.any(miller_mask)):
                miller_x = np.where(miller_mask, x, order)
                elem = _replace_where(miller_mask, elem, _besselj_miller(order, miller_x))
        out[idx] = sign * elem

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


def besselk0(x):
    """Modified Bessel function of the second kind of order 0."""
    scalar = not hasattr(x, 'shape') and not hasattr(x, '__len__')
    x = _as_float_array(x)
    x_for_small = np.where(x <= 2, x, 1)
    # For x <= 2, K_0 has a logarithmic singular term plus a polynomial in
    # (x / 2)^2; x_for_small keeps the unused branch numerically harmless.
    y_small = 0.25 * x_for_small * x_for_small
    safe_x = np.where(x == 0, 1, x)

    small = (
        -np.log(0.5 * safe_x) * _modified_besseli0(x_for_small)
        + (
            -0.57721566
            + y_small * (
                0.42278420
                + y_small * (
                    0.23069756
                    + y_small * (
                        0.03488590
                        + y_small * (
                            0.00262698
                            + y_small * (0.00010750 + y_small * 0.00000740)
                        )
                    )
                )
            )
        )
    )

    safe_large_x = np.where(x == 0, 1, x)
    # For x > 2, K_0 is the decaying exp(-x) / sqrt(x) envelope times a
    # correction polynomial in 2 / x.
    y_large = 2 / safe_large_x
    large = (
        np.exp(-safe_large_x)
        / np.sqrt(safe_large_x)
        * (
            1.25331414
            + y_large * (
                -0.07832358
                + y_large * (
                    0.02189568
                    + y_large * (
                        -0.01062446
                        + y_large * (
                            0.00587872
                            + y_large * (-0.00251540 + y_large * 0.00053208)
                        )
                    )
                )
            )
        )
    )

    out = np.where(x <= 2, small, large)
    out = np.where(x == 0, np.inf, out)
    return _maybe_scalar(out, scalar)


def besselk1(x):
    """Modified Bessel function of the second kind of order 1."""
    scalar = not hasattr(x, 'shape') and not hasattr(x, '__len__')
    x = _as_float_array(x)
    x_for_small = np.where(x <= 2, x, 1)
    # For x <= 2, K_1 combines the log-weighted I_1 term with a polynomial in
    # (x / 2)^2, then divides by x to capture the 1 / x singularity.
    y_small = 0.25 * x_for_small * x_for_small
    safe_x = np.where(x == 0, 1, x)

    small = (
        np.log(0.5 * safe_x) * _modified_besseli1(x_for_small)
        + (
            1.0
            + y_small * (
                0.15443144
                + y_small * (
                    -0.67278579
                    + y_small * (
                        -0.18156897
                        + y_small * (
                            -0.01919402
                            + y_small * (-0.00110404 + y_small * -0.00004686)
                        )
                    )
                )
            )
        )
        / safe_x
    )

    safe_large_x = np.where(x == 0, 1, x)
    # For x > 2, K_1 shares the decaying exp(-x) / sqrt(x) envelope with K_0,
    # using order-specific correction coefficients in 2 / x.
    y_large = 2 / safe_large_x
    large = (
        np.exp(-safe_large_x)
        / np.sqrt(safe_large_x)
        * (
            1.25331414
            + y_large * (
                0.23498619
                + y_large * (
                    -0.03655620
                    + y_large * (
                        0.01504268
                        + y_large * (
                            -0.00780353
                            + y_large * (0.00325614 + y_large * -0.00068245)
                        )
                    )
                )
            )
        )
    )

    out = np.where(x <= 2, small, large)
    out = np.where(x == 0, np.inf, out)
    return _maybe_scalar(out, scalar)


def besselk(n, x):
    """Modified Bessel function of the second kind of integer order n."""
    scalar = not hasattr(x, 'shape') and not hasattr(x, '__len__')
    n = abs(n)
    if n == 0:
        return besselk0(x)
    if n == 1:
        return besselk1(x)

    x = _as_float_array(x)
    safe_x = np.where(x == 0, 1, x)
    km2 = besselk0(x)
    km1 = besselk1(x)
    for order in range(1, n):
        k = km2 + (2 * order / safe_x) * km1
        km2, km1 = km1, k

    out = np.where(x == 0, np.inf, km1)
    return _maybe_scalar(out, scalar)


def besselk_seq(ns, x):
    """Modified Bessel functions of the second kind for integer orders ns."""
    if not hasattr(ns, '__len__'):
        ns = list(ns)
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


def besselk_ratio_knm1(n, x):
    """Return K_{n-1}(x) / K_n(x), using the adjacent-order path."""
    knm1, kn = besselk_adjacent(n, x)
    return knm1 / kn
