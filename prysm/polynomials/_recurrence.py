"""Shared three-term recurrence helpers for the polynomial families.

These primitives are not part of the public API.
"""

from prysm.mathops import np


def _seq_by_recurrence(ns, x, seed0, seed1, step):
    """Evaluate a three-term recurrence at sorted orders ns.

    P_0 = seed0, P_1 = seed1, P_k = step(k, P_{k-1}, P_{k-2}) for k >= 2.

    Parameters
    ----------
    ns : sized iterable
        orders to return, ascending
    x : ndarray
        evaluation grid; supplies shape and dtype
    seed0, seed1 : ndarray or scalar
        P_0, P_1
    step : callable
        step(k, Pkm1, Pkm2) -> P_k

    Returns
    -------
    ndarray
        has shape (len(ns),) followed by x.shape

    """
    if not hasattr(ns, '__len__'):
        ns = list(ns)
    min_i = 0
    out = np.empty((len(ns), *x.shape), dtype=x.dtype)
    if ns[min_i] == 0:
        out[min_i] = seed0
        min_i += 1

    if min_i == len(ns):
        return out

    if ns[min_i] == 1:
        out[min_i] = seed1
        min_i += 1

    if min_i == len(ns):
        return out

    Pnm2, Pnm1 = seed0, seed1
    max_n = ns[-1]
    for k in range(2, max_n + 1):
        Pn = step(k, Pnm1, Pnm2)
        Pnm2, Pnm1 = Pnm1, Pn
        if ns[min_i] == k:
            out[min_i] = Pn
            min_i += 1

    return out


def _seq_by_recurrence_with_der(ns, x, seed0, seed1, dseed0, dseed1, step):
    """Evaluate a three-term recurrence and its x-derivative at sorted orders ns.

    Value track seeded/stepped as in _seq_by_recurrence; the derivative track
    is seeded by dseed0, dseed1 and stepped jointly with the value track by
    step(k, Pkm1, Pkm2, Dkm1, Dkm2) -> (Pk, Dk).

    Returns
    -------
    ndarray, ndarray
        value and derivative, each shape (len(ns),) followed by x.shape

    """
    if not hasattr(ns, '__len__'):
        ns = list(ns)
    min_i = 0
    out = np.empty((len(ns), *x.shape), dtype=x.dtype)
    dout = np.empty_like(out)
    if ns[min_i] == 0:
        out[min_i] = seed0
        dout[min_i] = dseed0
        min_i += 1

    if min_i == len(ns):
        return out, dout

    if ns[min_i] == 1:
        out[min_i] = seed1
        dout[min_i] = dseed1
        min_i += 1

    if min_i == len(ns):
        return out, dout

    Pnm2, Pnm1 = seed0, seed1
    Dnm2, Dnm1 = dseed0, dseed1
    max_n = ns[-1]
    for k in range(2, max_n + 1):
        Pn, Dn = step(k, Pnm1, Pnm2, Dnm1, Dnm2)
        Pnm2, Pnm1 = Pnm1, Pn
        Dnm2, Dnm1 = Dnm1, Dn
        if ns[min_i] == k:
            out[min_i] = Pn
            dout[min_i] = Dn
            min_i += 1

    return out, dout
