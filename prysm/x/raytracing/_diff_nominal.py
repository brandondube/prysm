"""Nominal scalar kernels shared by forward and reverse AD."""

from prysm.mathops import np, row_dot


def refract_nominal(n, nprime, S_loc, n_hat):
    """Nominal refract scalars used by tangent and adjoint paths."""
    cosI = row_dot(n_hat, S_loc)
    mu = n / nprime
    one_minus = 1.0 - cosI * cosI
    sinT2 = mu * mu * one_minus
    with np.errstate(invalid='ignore'):
        cosT = np.sqrt(1.0 - sinT2)
    sign = np.sign(cosI)
    factor = sign * cosT - mu * cosI
    return cosI, mu, one_minus, cosT, sign, factor


def eic_nominal(P, S, C, kappa):
    """Nominal EIC closing scalars for s_tilde = -b - g / h."""
    r = P - C[None, :]
    b = row_dot(S, r)
    rr = row_dot(r, r)
    m = b * b - rr
    k = float(kappa)
    disc = 1.0 + k * k * m
    disc[disc < 0] = 0.0
    w = np.sqrt(disc)
    wsafe = np.where(w == 0, 1.0, w)
    g = k * m
    h = 1.0 + w
    return r, b, m, k, wsafe, g, h
