"""Hermite Polynomials.

He (Probabilist's) and H (Physicist's) Hermite polynomials share a single
three-term recurrence parameterized by an integer kind:

    kind = 1  =>  Probabilist's He   (Ax = x,  C = n-1)
    kind = 2  =>  Physicist's  H     (Ax = 2x, C = 2(n-1))

Equivalently, Ax = kind * x and C = kind * (n-1) for both families.
The derivative also collapses to a single formula:

    P'_n = kind * n * P_{n-1}

"""

from prysm.mathops import np

from ._recurrence import _seq_by_recurrence


def _hermite_value(n, x, kind):
    """Hermite polynomial of order n.  kind is 1 (He) or 2 (H)."""
    if n == 0:
        return np.ones_like(x)
    kx = kind * x
    if n == 1:
        return kx
    # P_2 = kx * P_1 - kind * P_0 = (kx)^2 - kind
    P2 = kx * kx - kind
    if n == 2:
        return P2

    Pnm2 = kx
    Pnm1 = P2
    for nn in range(3, n + 1):
        Pn = kx * Pnm1 - kind * (nn - 1) * Pnm2
        Pnm2, Pnm1 = Pnm1, Pn

    return Pn


def _hermite_value_seq(ns, x, kind):
    """Hermite polynomials at sorted orders ns.  See _hermite_value."""
    kx = kind * x

    def step(k, Pnm1, Pnm2):
        return kx * Pnm1 - kind * (k - 1) * Pnm2

    return _seq_by_recurrence(ns, x, 1, kx, step)


def _hermite_der_seq(ns, x, kind):
    """Derivative of Hermite polynomials at sorted orders ns.

    The derivative identity is uniform across kinds:

        d/dx He_n(x) = n * He_{n-1}(x)
        d/dx H_n(x)  = 2n * H_{n-1}(x)  =  kind * n * P_{n-1}(x)
    """
    if not hasattr(ns, '__len__'):
        ns = list(ns)
    out = np.empty((len(ns), *x.shape), dtype=x.dtype)
    lead = 0
    while lead < len(ns) and ns[lead] == 0:
        out[lead] = 0
        lead += 1

    if lead == len(ns):
        return out

    shifted = [n - 1 for n in ns[lead:]]
    Pns = _hermite_value_seq(shifted, x, kind)
    for i, n in enumerate(ns[lead:], start=lead):
        out[i] = kind * n * Pns[i - lead]

    return out


def hermite_He(n, x):
    """Probabilist's Hermite polynomial He_n at points x."""
    return _hermite_value(n, x, kind=1)


def hermite_He_seq(ns, x):
    """Probabilist's Hermite polynomials He_n at sorted orders ns and points x."""
    return _hermite_value_seq(ns, x, kind=1)


def hermite_He_der(n, x):
    """First derivative of He_n at points x."""
    if n == 0:
        return np.zeros_like(x)
    return n * hermite_He(n - 1, x)


def hermite_He_der_seq(ns, x):
    """First derivative of He_n at sorted orders ns and points x."""
    return _hermite_der_seq(ns, x, kind=1)


def hermite_H(n, x):
    """Physicist's Hermite polynomial H_n at points x."""
    return _hermite_value(n, x, kind=2)


def hermite_H_seq(ns, x):
    """Physicist's Hermite polynomials H_n at sorted orders ns and points x."""
    return _hermite_value_seq(ns, x, kind=2)


def hermite_H_der(n, x):
    """First derivative of H_n at points x."""
    if n == 0:
        return np.zeros_like(x)
    return 2 * n * hermite_H(n - 1, x)


def hermite_H_der_seq(ns, x):
    """First derivative of H_n at sorted orders ns and points x."""
    return _hermite_der_seq(ns, x, kind=2)
