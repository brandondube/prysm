"""Line searching routines."""
from prysm.mathops import np

import numpy as truenp

from .problem import as_problem

def _cubicmin(a, fa, fpa, b, fb, c, fc):
    """
    Finds the minimizer for a cubic polynomial that goes through the
    points (a,fa), (b,fb), and (c,fc) with derivative at a of fpa.

    If no minimizer can be found, return None.

    """
    # f(x) = A *(x-a)^3 + B*(x-a)^2 + C*(x-a) + D
    # this function operates on quasi-scalars; do not run on GPU
    np = truenp
    with np.errstate(divide='raise', over='raise', invalid='raise'):
        try:
            C = fpa
            db = b - a
            dc = c - a
            denom = (db * dc) ** 2 * (db - dc)
            d1 = np.empty((2, 2))
            d1[0, 0] = dc ** 2
            d1[0, 1] = -db ** 2
            d1[1, 0] = -dc ** 3
            d1[1, 1] = db ** 3
            [A, B] = np.dot(d1, np.asarray([fb - fa - C * db,
                                            fc - fa - C * dc]).flatten())
            A /= denom
            B /= denom
            radical = B * B - 3 * A * C
            xmin = a + (-B + np.sqrt(radical)) / (3 * A)
        except ArithmeticError:
            return None
    if not np.isfinite(xmin):
        return None
    return xmin


def _quadmin(a, fa, fpa, b, fb):
    """
    Finds the minimizer for a quadratic polynomial that goes through
    the points (a,fa), (b,fb) with derivative at a of fpa.

    """
    # f(x) = B*(x-a)^2 + C*(x-a) + D
    # this function operates on quasi-scalars; do not run on GPU
    np = truenp
    with np.errstate(divide='raise', over='raise', invalid='raise'):
        try:
            D = fa
            C = fpa
            db = b - a * 1.0
            B = (fb - D - C * db) / (db * db)
            xmin = a - C / (2.0 * B)
        except ArithmeticError:
            return None
    if not np.isfinite(xmin):
        return None
    return xmin


def _zoom(a_lo, a_hi, phi_lo, phi_hi, derphi_lo,
          phi, derphi, phi0, derphi0, c1, c2, extra_condition):
    """Zoom stage of approximate linesearch satisfying strong Wolfe conditions.

    Part of the optimization algorithm in `scalar_search_wolfe2`.

    Notes
    -----
    Implements Algorithm 3.6 (zoom) in Wright and Nocedal,
    'Numerical Optimization', 1999, pp. 61.

    """

    maxiter = 10
    i = 0
    delta1 = 0.2  # cubic interpolant check
    delta2 = 0.1  # quadratic interpolant check
    phi_rec = phi0
    a_rec = 0
    while True:
        # interpolate to find a trial step length between a_lo and
        # a_hi Need to choose interpolation here. Use cubic
        # interpolation and then if the result is within delta *
        # dalpha or outside of the interval bounded by a_lo or a_hi
        # then use quadratic interpolation, if the result is still too
        # close, then use bisection

        dalpha = a_hi - a_lo
        if dalpha < 0:
            a, b = a_hi, a_lo
        else:
            a, b = a_lo, a_hi

        # minimizer of cubic interpolant
        # (uses phi_lo, derphi_lo, phi_hi, and the most recent value of phi)
        #
        # if the result is too close to the end points (or out of the
        # interval), then use quadratic interpolation with phi_lo,
        # derphi_lo and phi_hi if the result is still too close to the
        # end points (or out of the interval) then use bisection

        if (i > 0):
            cchk = delta1 * dalpha
            a_j = _cubicmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi,
                            a_rec, phi_rec)
        if (i == 0) or (a_j is None) or (a_j > b - cchk) or (a_j < a + cchk):
            qchk = delta2 * dalpha
            a_j = _quadmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi)
            if (a_j is None) or (a_j > b-qchk) or (a_j < a+qchk):
                a_j = a_lo + 0.5*dalpha

        # Check new value of a_j

        phi_aj = phi(a_j)
        if (phi_aj > phi0 + c1*a_j*derphi0) or (phi_aj >= phi_lo):
            phi_rec = phi_hi
            a_rec = a_hi
            a_hi = a_j
            phi_hi = phi_aj
        else:
            derphi_aj = derphi(a_j)
            if abs(derphi_aj) <= -c2*derphi0 and extra_condition(a_j, phi_aj):
                a_star = a_j
                val_star = phi_aj
                valprime_star = derphi_aj
                break
            if derphi_aj*(a_hi - a_lo) >= 0:
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
        i += 1
        if (i > maxiter):
            # Failed to find a conforming step size
            a_star = None
            val_star = None
            valprime_star = None
            break
    return a_star, val_star, valprime_star


def ls_strong_wolfe(problem, xk, pk, fg_at_xk=None, maxalpha=None, c1=1e-4, c2=0.9, maxiter=10):
    """Line search satisfying the strong Wolfe conditions.

    Finds a step length alpha along direction pk from xk satisfying
        f(xk + alpha*pk) <= f(xk) + c1 * alpha * (g(xk) . pk)        (sufficient decrease)
        |g(xk + alpha*pk) . pk| <= c2 * |g(xk) . pk|                 (curvature)

    Implements Algorithm 3.5 of Wright & Nocedal, "Numerical Optimization",
    1999, pp. 60. The zoom stage (Algorithm 3.6) lives in `_zoom` above.

    Parameters
    ----------
    problem : Problem or callable
        A Problem instance, any object exposing fg(x), or a plain callable
        fg(x) -> (f, g).  Callables are wrapped by as_problem.
    xk, pk : ndarray
        current iterate and search direction.
    fg_at_xk : tuple, optional
        precomputed (f, g) at xk; saves one problem evaluation.
    maxalpha : float, optional
        upper bound on the step length.
    c1, c2 : float
        Wolfe constants. Standard defaults: c1=1e-4, c2=0.9.
    maxiter : int
        bracket-phase iteration cap (10 is plenty in practice).

    Returns
    -------
    alpha : float or None
        accepted step length, or None if no acceptable alpha was found.
    phi_a : float or None
        f(xk + alpha*pk) at the accepted alpha.
    derphi_a : float or None
        g(xk + alpha*pk) . pk at the accepted alpha.

    """
    problem = as_problem(problem)
    if fg_at_xk is None:
        fg_at_xk = problem.fg(xk)
    fk, gk = fg_at_xk

    # _zoom needs phi(alpha) and derphi(alpha) as separate callables. Each
    # alpha produces a fresh xk+alpha*pk array, so the Problem's internal
    # identity cache cannot bridge phi-then-derphi at the same alpha. Cache
    # the joint fg result here, keyed on alpha.
    _cache_alpha = [None]
    _cache_val = [None, None]  # phi, derphi

    def _eval(alpha):
        if _cache_alpha[0] != alpha:
            fa, ga = problem.fg(xk + alpha * pk)
            _cache_alpha[0] = alpha
            _cache_val[0] = fa
            _cache_val[1] = np.dot(ga, pk)
        return _cache_val[0], _cache_val[1]

    def phi(alpha):
        return _eval(alpha)[0]

    def derphi(alpha):
        return _eval(alpha)[1]

    phi0 = fk
    derphi0 = np.dot(gk, pk)

    alpha0 = 0.0
    alpha1 = 1.0
    if maxalpha is not None:
        alpha1 = min(alpha1, maxalpha)

    phi_a0 = phi0
    derphi_a0 = derphi0
    phi_a1, _ = _eval(alpha1)

    def _extra(a, p):
        return True

    alpha_star = None
    phi_star = None
    derphi_star = None

    for i in range(maxiter):
        if alpha1 == 0 or (maxalpha is not None and alpha0 == maxalpha):
            break

        # sufficient-decrease violation, or non-decreasing on a non-first step
        # → minimum is bracketed between alpha0 and alpha1; zoom in.
        if (phi_a1 > phi0 + c1 * alpha1 * derphi0) or (i > 0 and phi_a1 >= phi_a0):
            alpha_star, phi_star, derphi_star = _zoom(
                alpha0, alpha1, phi_a0, phi_a1, derphi_a0,
                phi, derphi, phi0, derphi0, c1, c2, _extra,
            )
            break

        derphi_a1 = derphi(alpha1)
        # strong Wolfe curvature condition satisfied → accept alpha1
        if abs(derphi_a1) <= -c2 * derphi0:
            alpha_star = alpha1
            phi_star = phi_a1
            derphi_star = derphi_a1
            break

        # positive slope at alpha1 → minimum is between alpha1 and alpha0; zoom
        # with the interval reversed.
        if derphi_a1 >= 0:
            alpha_star, phi_star, derphi_star = _zoom(
                alpha1, alpha0, phi_a1, phi_a0, derphi_a1,
                phi, derphi, phi0, derphi0, c1, c2, _extra,
            )
            break

        # extrapolate
        alpha2 = 2.0 * alpha1
        if maxalpha is not None:
            alpha2 = min(alpha2, maxalpha)

        alpha0 = alpha1
        alpha1 = alpha2
        phi_a0 = phi_a1
        derphi_a0 = derphi_a1
        phi_a1, _ = _eval(alpha1)

    return alpha_star, phi_star, derphi_star
