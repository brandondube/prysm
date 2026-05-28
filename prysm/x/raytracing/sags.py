"""Sag and derivative kernels for sequential raytracing surfaces."""

from prysm.mathops import np
from prysm.coordinates import cart_to_polar
from prysm.polynomials.qpoly import compute_z_Q2d, compute_z_zprime_Q2d

def product_rule(u, v, du, dv):
    """The product rule of calculus, d/dx uv = u dv + v du."""
    return u * dv + v * du


def gradient_to_unit_normal(Fx, Fy):
    """Unit normal for a sag surface from x/y sag gradients."""
    inv_mag = 1.0 / np.sqrt(1.0 + Fx * Fx + Fy * Fy)
    return np.stack([-Fx * inv_mag, -Fy * inv_mag, inv_mag], axis=-1)


def plane_sag_and_normal(x, y):
    """Sag and unit normal for the local plane z = 0."""
    z = np.zeros_like(x)
    n_hat = np.zeros(x.shape + (3,), dtype=x.dtype)
    n_hat[..., 2] = 1.0
    return z, n_hat


def phi_conic(c, k, rhosq):
    """'phi' for a conicc.

    phi = sqrt(1 - c^2 rho^2)

    Parameters
    ----------
    c : float
        curvature, reciprocal radius of curvature
    k : float
        kappa, conic constant
    rhosq : ndarray
        squared radial coordinate (non-normalized)

    Returns
    -------
    ndarray
        phi term

    """
    csq = c * c
    return np.sqrt(1 - (1 + k) * csq * rhosq)


def der_direction_cosine_conic(c, k, rho, rhosq=None, phi=None):
    """Derivative term needed for the product rule and Q type aspheres.

    sag z(rho) = 1/phi * (weighted sum of Q polynomials)

    The return of this function is the derivative of 1/phi required
    for completing the product rule on the surface's derivative.

    Parameters
    ----------
    c : float
        curvature, reciprocal radius of curvature
    k : float
        kappa, conic constant
    rho : ndarray
        radial coordinate (non-normalized)
    rhosq : ndarray
        squared radial coordinate (non-normalized)
        rho ** 2 if None
    phi : ndarray, optional
        (1 - c^2 r^2)^.5
        computed if not provided
        many surface types utilize phi; its computation can be
        de-duplicated by passing the optional argument

    Returns
    -------
    ndarray
        d/drho of (1/phi)

    """
    csq = c * c
    if rhosq is None:
        rhosq = rho * rho
    if phi is None:
        phi = phi_conic(c, k, rhosq)

    num = -csq * (k-1) * rho
    den = phi * phi * phi
    return num / den

def conic_sag_and_normal(c, k, X, Y):
    """Sag and unit normal for a conicoid evaluated at coordinates X, Y."""
    if c == 0.0:
        return plane_sag_and_normal(X, Y)
    A = X * X + Y * Y
    with np.errstate(invalid='ignore'):
        phi = phi_conic(c, k, A)
        phi = np.where(np.isnan(phi), 0.0, phi)
        mag_sq = 1.0 - k * c * c * A
        mag = np.sqrt(np.where(mag_sq < 0.0, 1.0, mag_sq))
    z = conic_sag(c, k, A, phi=phi)
    n_hat = np.stack([-c * X / mag, -c * Y / mag, phi / mag], axis=-1)
    return z, n_hat


def sphere_sag(c, rhosq, phi=None):
    """Sag of a spherical surface.

    Parameters
    ----------
    c : float
        surface curvature
    rhosq : ndarray
        radial coordinate squared
        e.g. for a 15 mm half-diameter optic,
        rho = 0 .. 15
        rhosq = 0 .. 225
        there is no requirement on rectilinear sampling or array
        dimensionality
    phi : ndarray, optional
        (1 - c^2 r^2)^.5
        computed if not provided
        many surface types utilize phi; its computation can be
        de-duplicated by passing the optional argument

    Returns
    -------
    ndarray
        surface sag

    """
    if phi is None:
        csq = c * c
        phi = np.sqrt(1 - csq * rhosq)

    return (c * rhosq) / (1 + phi)


def sphere_sag_der(c, rho, phi=None):
    """Derivative of the sag of a spherical surface.

    Parameters
    ----------
    c : float
        surface curvature
    rho : ndarray
        radial coordinate
        e.g. for a 15 mm half-diameter optic,
        rho = 0 .. 15
        there is no requirement on rectilinear sampling or array
        dimensionality
    phi : ndarray, optional
        (1 - c^2 r^2)^.5
        computed if not provided
        many surface types utilize phi; its computation can be
        de-duplicated by passing the optional argument

    Returns
    -------
    ndarray
        derivative of surface sag

    """
    if phi is None:
        csq = c ** 2
        rhosq = rho * rho
        phi = np.sqrt(1 - csq * rhosq)
    return (c * rho) / phi


def conic_sag(c, kappa, rhosq, phi=None):
    """Sag of a conic surface.

    Parameters
    ----------
    c : float
        surface curvature
    kappa : float
        conic constant
    rhosq : ndarray
        radial coordinate squared
        e.g. for a 15 mm half-diameter optic,
        rho = 0 .. 15
        rhosq = 0 .. 225
        there is no requirement on rectilinear sampling or array
        dimensionality
    phi : ndarray, optional
        (1 - (1+kappa) c^2 r^2)^.5
        computed if not provided
        many surface types utilize phi; its computation can be
        de-duplicated by passing the optional argument

    Returns
    -------
    ndarray
        surface sag

    """
    if phi is None:
        phi = phi_conic(c, kappa, rhosq)

    return (c * rhosq) / (1 + phi)


def conic_sag_der(c, kappa, rho, phi=None):
    """Derivative of the sag of a conic surface.

    Parameters
    ----------
    c : float
        surface curvature
    kappa : float
        conic constant, 0=sphere, 1=parabola, etc
    rho : ndarray
        radial coordinate
        e.g. for a 15 mm half-diameter optic,
        rho = 0 .. 15
        there is no requirement on rectilinear sampling or array
        dimensionality
    phi : ndarray, optional
        (1 - (1+kappa) c^2 r^2)^.5
        computed if not provided
        many surface types utilize phi; its computation can be
        de-duplicated by passing the optional argument

    Returns
    -------
    ndarray
        surface sag

    """
    if phi is None:
        phi = phi_conic(c, kappa, rho * rho)

    return (c * rho) / phi


def conic_sag_der_xy(c, kappa, x, y, phi=None):
    """Cartesian partial derivatives of the sag of an on-axis conic.

    Equivalent to conic_sag_der composed with the polar-to-Cartesian chain rule,
    but written directly in (x,y) so there is no 1/r singularity at the origin.

    Parameters
    ----------
    c : float
        surface curvature
    kappa : float
        conic constant
    x : ndarray
        x coordinate (non-normalized)
    y : ndarray
        y coordinate (non-normalized)
    phi : ndarray, optional
        (1 - (1+kappa) c^2 (x^2+y^2))^.5
        computed if not provided

    Returns
    -------
    ndarray, ndarray
        dz/dx, dz/dy

    """
    if phi is None:
        phi = phi_conic(c, kappa, x * x + y * y)
    return (c * x) / phi, (c * y) / phi


def _conic_base_xy(c, kappa, x, y):
    """Sag and Cartesian partial derivatives of an on-axis conic at (x, y).

    Convenience wrapper around conic_sag and conic_sag_der_xy that shares a
    single phi computation.  Used by the polynomial-deformed surface sag_and_normal
    closures to write themselves as conic-base + perturbation.

    Returns
    -------
    z, dz/dx, dz/dy : ndarray

    """
    rsq = x * x + y * y
    phi = phi_conic(c, kappa, rsq)
    z = conic_sag(c, kappa, rsq, phi=phi)
    ddx, ddy = conic_sag_der_xy(c, kappa, x, y, phi=phi)
    return z, ddx, ddy


def _conic_base_xy_sag(c, kappa, x, y):
    """Sag-only sibling of _conic_base_xy."""
    return conic_sag(c, kappa, x * x + y * y)


def _add_conic_base_derivatives(c, kappa, x, y, z_p, ddx_p, ddy_p):
    """Add an on-axis conic base to a polynomial perturbation.

    The caller is responsible for any chain-rule rescaling of (ddx_p, ddy_p)
    needed to bring them into unnormalized-coordinate units before reaching
    here.

    """
    z_c, ddx_c, ddy_c = _conic_base_xy(c, kappa, x, y)
    return z_c + z_p, ddx_c + ddx_p, ddy_c + ddy_p


def _add_conic_base_sag(c, kappa, x, y, z_p):
    """Sag-only sibling of _add_conic_base_derivatives."""
    return _conic_base_xy_sag(c, kappa, x, y) + z_p


def even_asphere_sag(c, kappa, coefs, rsq):
    """Sag of an even asphere: conic base plus polynomial in r^2.

    z(rho) = conic_sag(c, kappa, rho^2)
           + coefs[0] * rho^4 + coefs[1] * rho^6 + coefs[2] * rho^8 + ...

    Parameters
    ----------
    c : float
        vertex curvature
    kappa : float
        conic constant
    coefs : iterable of float
        polynomial coefficients [a4, a6, a8, ...] for powers r^4, r^6, r^8, ...
        Empty / None makes the surface a pure conic.
    rsq : ndarray
        radial coordinate squared, x^2 + y^2

    Returns
    -------
    ndarray
        surface sag

    """
    z = conic_sag(c, kappa, rsq)
    if coefs is None or len(coefs) == 0:
        return z
    # Horner over rsq:  poly = a4 + a6*rsq + a8*rsq^2 + ...,  then * rsq^2
    p = 0.0
    for a in reversed(coefs):
        p = p * rsq + a
    return z + p * rsq * rsq


def even_asphere_sag_der_xy(c, kappa, coefs, x, y, phi=None):
    """Cartesian partial derivatives of an even asphere.

    z = z_conic(rho^2) + sum_i coefs[i] * rho^(2(i+2))
    dz/dx = (c x / phi) + 2 x * sum_i (i+2) * coefs[i] * rho^(2(i+1))
          = conic_der_x + 2 x * rho^2 * Horner(d_coefs)
    where d_coefs[i] = (i + 2) * coefs[i].

    Parameters
    ----------
    c, kappa : float
        conic curvature and conic constant
    coefs : iterable of float
        even-asphere coefficients (see even_asphere_sag)
    x, y : ndarray
        Cartesian coordinates
    phi : ndarray, optional
        sqrt(1 - (1+kappa) c^2 (x^2 + y^2)); computed if not provided

    Returns
    -------
    ndarray, ndarray
        dz/dx, dz/dy

    """
    rsq = x * x + y * y
    if phi is None:
        phi = phi_conic(c, kappa, rsq)
    dx_c, dy_c = conic_sag_der_xy(c, kappa, x, y, phi=phi)
    if coefs is None or len(coefs) == 0:
        return dx_c, dy_c
    # d_coefs[i] = (i + 2) * coefs[i]; Horner over rsq
    d_coefs = [(i + 2) * a for i, a in enumerate(coefs)]
    p = 0.0
    for a in reversed(d_coefs):
        p = p * rsq + a
    common = 2.0 * rsq * p
    return dx_c + x * common, dy_c + y * common


def _q2d_conic_aggregate_with_derivs(x, y, r, t, dx=0, dy=0):
    """Base-conic aggregate and polar derivatives for Q2D."""
    X = x + dx
    Y = y + dy
    A = X * X + Y * Y
    cost = np.cos(t)
    sint = np.sin(t)
    dA_dr = 2.0 * (X * cost + Y * sint)
    dA_dt = 2.0 * r * (-X * sint + Y * cost)
    return A, dA_dr, dA_dt


def _q2d_sigma_inv_der(c, k, x, y, r, t, dx=0, dy=0):
    """Polar derivatives of Q2D's base-conic 1/sigma factor."""
    A, dA_dr, dA_dt = _q2d_conic_aggregate_with_derivs(x, y, r, t, dx, dy)
    csq = c * c
    phi = phi_conic(c, k, A)
    phi_cubed = phi * phi * phi
    u = np.sqrt(1.0 - k * csq * A)
    common = csq * (((1.0 + k) * u) / (2.0 * phi_cubed)
                    - k / (2.0 * phi * u))
    return common * dA_dr, common * dA_dt


def _q2d_conic_base_terms(c, k, x, y, r, t, dx=0, dy=0):
    """Base conic sag, polar derivatives, and sigma for Q2D."""
    base_sag, n_hat = conic_sag_and_normal(c, k, x + dx, y + dy)
    sigma = n_hat[..., 2]
    ddx = -n_hat[..., 0] / sigma
    ddy = -n_hat[..., 1] / sigma
    cost = np.cos(t)
    sint = np.sin(t)
    base_primer = ddx * cost + ddy * sint
    base_primet = r * (-ddx * sint + ddy * cost)
    return base_sag, base_primer, base_primet, sigma


def Q2d_and_der(cm0, ams, bms, x, y, normalization_radius, c, k, dx=0, dy=0):
    """Q-type freeform surface, with base (perhaps shifted) conicoic.

    Returns the sag and the polar derivatives (dz/dr, dz/dt).  Q2D.sag_and_normal
    converts to Cartesian (dz/dx, dz/dy) via the chain rule with
    a small on-axis patch (r=0 has a removable 0/0).

    Parameters
    ----------
    cm0 : iterable
        surface coefficients when m=0 (inside curly brace, top line, Eq. B.1)
        span n=0 .. len(cms)-1 and mus tbe fully dense
    ams : iterable of iterables
        ams[0] are the coefficients for the m=1 cosine terms,
        ams[1] for the m=2 cosines, and so on.  Same order n rules as cm0
    bms : iterable of iterables
        same as ams, but for the sine terms
        ams and bms must be the same length - that is, if an azimuthal order m
        is presnet in ams, it must be present in bms.  The azimuthal orders
        need not have equal radial expansions.

        For example, if ams extends to m=3, then bms must reach m=3
        but, if the ams for m=3 span n=0..5, it is OK for the bms to span n=0..3,
        or any other value, even just [0].
    x : ndarray
        X coordinates
    y : ndarray
        Y coordinates
    normalization_radius : float
        radius by which to normalize rho to produce u
    c : float
        curvature, reciprocal radius of curvature
    k : float
        kappa, conic constant
    dx : float
        shift of the base conic in x
    dy : float
        shift of the base conic in y

    Returns
    -------
    ndarray, ndarray, ndarray
        sag, dsag/drho, dsag/dtheta

    """
    # Q portion.  vec_to_grid=False: cart_to_polar's default auto-meshgrids
    # 1D inputs into a 2D grid, which is the wrong semantic for ray-trace
    # use where (x, y) are paired per-point; opt out explicitly here.
    r, t = cart_to_polar(x, y, vec_to_grid=False)
    r2 = r / normalization_radius
    # content of curly braces in B.1 from oe-20-3-2483
    z, zprimer, zprimet = compute_z_zprime_Q2d(cm0, ams, bms, r2, t)

    base_sag, base_primer, base_primet, sigma = _q2d_conic_base_terms(
        c, k, x, y, r, t, dx, dy,
    )

    # Eq. 5.1/5.2
    sigma = 1 / sigma
    sigmaprimer, sigmaprimet = _q2d_sigma_inv_der(c, k, x, y, r, t, dx, dy)

    zprimer /= normalization_radius
    zprimer2 = product_rule(sigma, z, sigmaprimer, zprimer)
    zprimet2 = product_rule(sigma, z, sigmaprimet, zprimet)
    z *= sigma
    z += base_sag
    zprimer2 += base_primer
    zprimet2 += base_primet
    return z, zprimer2, zprimet2


def Q2d_sag(cm0, ams, bms, x, y, normalization_radius, c, k, dx=0, dy=0):
    """Sag-only sibling of Q2d_and_der.

    Uses the polynomial-side compute_z_Q2d (which avoids the j=1 Clenshaw
    recurrence) and skips the off_axis_conic derivative and sigma-derivative
    work that Q2d_and_der needs for dz/dr, dz/dt.

    """
    r, t = cart_to_polar(x, y, vec_to_grid=False)
    r2 = r / normalization_radius
    z = compute_z_Q2d(cm0, ams, bms, r2, t)
    base_sag, _, _, sigma = _q2d_conic_base_terms(c, k, x, y, r, t, dx, dy)
    sigma_inv = 1 / sigma
    return base_sag + sigma_inv * z

__all__ = [
    'gradient_to_unit_normal',
    'plane_sag_and_normal',
    'conic_sag_and_normal',
    'product_rule',
    'phi_conic',
    'der_direction_cosine_conic',
    'sphere_sag',
    'sphere_sag_der',
    'conic_sag',
    'conic_sag_der',
    'conic_sag_der_xy',
    'even_asphere_sag',
    'even_asphere_sag_der_xy',
    'Q2d_and_der',
    'Q2d_sag',
    '_conic_base_xy',
    '_conic_base_xy_sag',
    '_add_conic_base_derivatives',
    '_add_conic_base_sag',
]
