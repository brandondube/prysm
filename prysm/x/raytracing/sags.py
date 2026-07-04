"""Sag and derivative kernels for sequential raytracing surfaces."""

from prysm.conf import config
from prysm.mathops import np
from prysm.coordinates import cart_to_polar
from prysm.polynomials import zernike_nm, zernike_nm_der_xy
from prysm.polynomials.qpoly import compute_z_Q2d, compute_z_zprime_Q2d


def fd_step(finite_difference_step, *arrs):
    """Central-difference step, scaled to the coordinate magnitude.

    Uses finite_difference_step when provided.
    """
    if finite_difference_step is not None:
        return np.asarray(finite_difference_step, dtype=config.precision)
    eps = np.sqrt(np.finfo(config.precision).eps)
    mag = 1.0
    for a in arrs:
        mag = np.maximum(mag, np.abs(a))
    return eps * mag


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
    """'phi' for a conic.

    phi = sqrt(1 - (1 + k) c^2 rho^2)
    The conic convention is k = -1 for a parabola and k = 0 for a sphere.

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


def _der_inv_phi_conic_rhosq(c, k, rhosq, phi=None):
    """Derivative of 1 / phi with respect to rho squared."""
    csq = c * c
    if phi is None:
        phi = phi_conic(c, k, rhosq)
    return 0.5 * (1.0 + k) * csq / (phi * phi * phi)


def der_direction_cosine_conic(c, k, rho, rhosq=None, phi=None):
    """Derivative term needed for the product rule and Q type aspheres.

    For z(rho) = (weighted sum of Q polynomials) / phi, returns
    d(1 / phi) / drho.

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
        (1 - (1 + k) c^2 r^2)^.5, computed if not provided.

    Returns
    -------
    ndarray
        d/drho of (1/phi)

    """
    if rhosq is None:
        rhosq = rho * rho
    if phi is None:
        phi = phi_conic(c, k, rhosq)

    return 2.0 * rho * _der_inv_phi_conic_rhosq(c, k, rhosq, phi=phi)

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
        radial coordinate squared.
    phi : ndarray, optional
        (1 - c^2 r^2)^.5, computed if not provided.

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
        radial coordinate.
    phi : ndarray, optional
        (1 - c^2 r^2)^.5, computed if not provided.

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
        radial coordinate squared.
    phi : ndarray, optional
        (1 - (1+kappa) c^2 r^2)^.5, computed if not provided.

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
        conic constant, where -1 is a parabola and 0 is a sphere
    rho : ndarray
        radial coordinate.
    phi : ndarray, optional
        (1 - (1+kappa) c^2 r^2)^.5, computed if not provided.

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


def conic_sag_hessian(c, kappa, x, y, phi=None):
    """Cartesian second derivatives of an on-axis conic sag.

    Returns (sag_xx, sag_xy, sag_yy), the entries of the sag Hessian, needed
    by the differential ray trace's normal tangent.  Written directly in
    (x, y) with no 1/r singularity; reduces to the sphere Hessian at kappa=0.

    Parameters
    ----------
    c : float
        surface curvature
    kappa : float
        conic constant
    x, y : ndarray
        Cartesian coordinates (non-normalized)
    phi : ndarray, optional
        sqrt(1 - (1+kappa) c^2 (x^2 + y^2)); computed if not provided

    Returns
    -------
    ndarray, ndarray, ndarray
        sag_xx, sag_xy, sag_yy

    """
    if phi is None:
        phi = phi_conic(c, kappa, x * x + y * y)
    beta = (1.0 + kappa) * c * c
    phi3 = phi * phi * phi
    sag_xx = c * (1.0 - beta * y * y) / phi3
    sag_xy = c * beta * x * y / phi3
    sag_yy = c * (1.0 - beta * x * x) / phi3
    return sag_xx, sag_xy, sag_yy


def conic_sag_param_partials(c, kappa, x, y, name, phi=None):
    """Partials of conic sag and sag-gradient wrt a shape parameter.

    For name in {'c', 'k'}, returns (sag_t, gx_t, gy_t):
    d(sag), d(dz/dx), and d(dz/dy) wrt the named parameter at fixed (x, y).

    Parameters
    ----------
    c : float
        surface curvature
    kappa : float
        conic constant
    x, y : ndarray
        Cartesian coordinates (non-normalized)
    name : str
        'c' for curvature, 'k' for conic constant
    phi : ndarray, optional
        sqrt(1 - (1+kappa) c^2 (x^2 + y^2)); computed if not provided

    Returns
    -------
    ndarray, ndarray, ndarray
        d sag / d param, d(dz/dx) / d param, d(dz/dy) / d param

    """
    A = x * x + y * y
    if phi is None:
        phi = phi_conic(c, kappa, A)
    phi3 = phi * phi * phi
    one_plus_phi = 1.0 + phi
    if name == 'c':
        sag_t = A / one_plus_phi + (1.0 + kappa) * c * c * A * A / (
            phi * one_plus_phi * one_plus_phi)
        gx_t = x / phi3
        gy_t = y / phi3
    elif name == 'k':
        c3 = c * c * c
        sag_t = c3 * A * A / (2.0 * phi * one_plus_phi * one_plus_phi)
        gx_t = c3 * x * A / (2.0 * phi3)
        gy_t = c3 * y * A / (2.0 * phi3)
    else:
        raise ValueError(f"conic shape parameter must be 'c' or 'k', got {name!r}")
    return sag_t, gx_t, gy_t


def zernike_irregularity_partials(n, m, x, y, normalization_radius, norm=True):
    """Sag and sag-gradient partials of one Zernike surface-irregularity term.

    For delta z = a * Z_n^m(x / R, y / R), the amplitude tangents are:

        d(sag)/da   = Z_n^m(x / R, y / R)
        d(dz/dx)/da = (1 / R) dZ_n^m/dx
        d(dz/dy)/da = (1 / R) dZ_n^m/dy

    The sag value uses polar Zernikes; gradients use zernike_nm_der_xy.  With
    norm=True, unit amplitude is unit RMS over the disk of radius R.  Z_2^2 and
    Z_2^-2 are the Rimmer cylinder terms (Applied Optics 9(3), 533-537, 1970).

    Parameters
    ----------
    n : int
        Zernike radial order.
    m : int
        Zernike azimuthal order.
    x, y : ndarray
        Cartesian coordinates (non-normalized, surface length units).
    normalization_radius : float
        radius R used to normalize x, y before Zernike evaluation.
    norm : bool, optional
        if True (default), orthonormal (unit-RMS) Zernikes; else zero-to-peak.

    Returns
    -------
    ndarray, ndarray, ndarray
        d sag / da, d(dz/dx) / da, d(dz/dy) / da

    """
    R = float(normalization_radius)
    xn = x / R
    yn = y / R
    rho = np.sqrt(xn * xn + yn * yn)
    theta = np.arctan2(yn, xn)
    sag_t = zernike_nm(n, m, rho, theta, norm=norm)
    dzdx, dzdy = zernike_nm_der_xy(n, m, xn, yn, norm=norm)
    return sag_t, dzdx / R, dzdy / R


def _conic_base_xy(c, kappa, x, y):
    """Sag and Cartesian partial derivatives of an on-axis conic at (x, y).

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

    ddx_p and ddy_p must already be in unnormalized-coordinate units.

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

    z = z_conic(rho^2) + sum_i coefs[i] * rho^(2(i+2));
    dz/dx = conic_der_x + 2 x rho^2 Horner(d_coefs),
    d_coefs[i] = (i + 2) coefs[i].

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
    u = np.sqrt(1.0 - k * csq * A)
    d_inv_phi_dA = _der_inv_phi_conic_rhosq(c, k, A, phi=phi)
    d_u_dA = -0.5 * k * csq / u
    common = u * d_inv_phi_dA + d_u_dA / phi
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
    """Q-type freeform surface, with base (perhaps shifted) conicoid.

    Returns sag and polar derivatives (dz/dr, dz/dt).

    Parameters
    ----------
    cm0 : iterable
        m=0 coefficients (Eq. B.1), dense in n.
    ams : iterable of iterables
        cosine coefficients; ams[m-1] holds azimuthal order m.
    bms : iterable of iterables
        sine coefficients, with the same azimuthal orders as ams.
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
    'conic_sag_hessian',
    'conic_sag_param_partials',
    'zernike_irregularity_partials',
    'even_asphere_sag',
    'even_asphere_sag_der_xy',
    'Q2d_and_der',
    'Q2d_sag',
    '_conic_base_xy',
    '_conic_base_xy_sag',
    '_add_conic_base_derivatives',
    '_add_conic_base_sag',
]
