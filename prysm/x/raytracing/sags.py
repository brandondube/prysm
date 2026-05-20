"""Sag and derivative kernels for sequential raytracing surfaces."""

from prysm.mathops import np
from prysm.coordinates import cart_to_polar
from prysm.polynomials.qpoly import compute_z_Q2d, compute_z_zprime_Q2d


def product_rule(u, v, du, dv):
    """The product rule of calculus, d/dx uv = u dv + v du."""
    return u * dv + v * du


def phi_spheroid(c, k, rhosq):
    """'phi' for a spheroid.

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


def der_direction_cosine_spheroid(c, k, rho, rhosq=None, phi=None):
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
        phi = phi_spheroid(c, k, rhosq)

    num = -csq * (k-1) * rho
    den = phi * phi * phi
    return num / den


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
        phi = phi_spheroid(c, kappa, rhosq)

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
        phi = phi_spheroid(c, kappa, rho * rho)

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
        phi = phi_spheroid(c, kappa, x * x + y * y)
    return (c * x) / phi, (c * y) / phi


def _conic_base_xy(c, kappa, x, y):
    """Sag and Cartesian partial derivatives of an on-axis conic at (x, y).

    Convenience wrapper around conic_sag and conic_sag_der_xy that shares a
    single phi computation.  Used by the polynomial-deformed surface FFp
    closures to write themselves as conic-base + perturbation.

    Returns
    -------
    z, dz/dx, dz/dy : ndarray

    """
    rsq = x * x + y * y
    phi = phi_spheroid(c, kappa, rsq)
    z = conic_sag(c, kappa, rsq, phi=phi)
    ddx, ddy = conic_sag_der_xy(c, kappa, x, y, phi=phi)
    return z, ddx, ddy


def _conic_base_xy_F(c, kappa, x, y):
    """Sag-only sibling of _conic_base_xy."""
    return conic_sag(c, kappa, x * x + y * y)


def _add_conic_base_FFp(c, kappa, x, y, z_p, ddx_p, ddy_p):
    """Add an on-axis conic base to a polynomial perturbation.

    The caller is responsible for any chain-rule rescaling of (ddx_p, ddy_p)
    needed to bring them into unnormalized-coordinate units before reaching
    here.

    """
    z_c, ddx_c, ddy_c = _conic_base_xy(c, kappa, x, y)
    return z_c + z_p, ddx_c + ddx_p, ddy_c + ddy_p


def _add_conic_base_F(c, kappa, x, y, z_p):
    """Sag-only sibling of _add_conic_base_FFp."""
    return _conic_base_xy_F(c, kappa, x, y) + z_p


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
        phi = phi_spheroid(c, kappa, rsq)
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


def _off_axis_aggregate(r, t, dx, dy):
    """Squared distance from off-axis section (r, t) to base conic vertex.

    Returns A = (r·cos t + dx)² + (r·sin t + dy)² = r² + 2 r (dx cos t + dy sin t) + dx² + dy²,
    the rhosq used in the off-axis conic phi factor.

    """
    oblique = 2 * r * (dx * np.cos(t) + dy * np.sin(t))
    return r * r + oblique + dx * dx + dy * dy


def _off_axis_aggregate_with_derivs(r, t, dx, dy):
    """Aggregate + radial and half-azimuthal derivatives.

    Returns (A, dA/dr, half_dA/dt).  dA/dt = 2 * half_dA/dt; the
    half-derivative is the building block that appears in the closed-form
    sag derivatives so the consumer can multiply by 2 if it wants the full
    azimuthal derivative.

    """
    cost = np.cos(t)
    sint = np.sin(t)
    oblique = 2 * r * (dx * cost + dy * sint)
    ddr_oblique = 2 * r + 2 * (dx * cost + dy * sint)
    ddt_oblique_half = r * (-dx * sint + dy * cost)
    agg = r * r + oblique + dx * dx + dy * dy
    return agg, ddr_oblique, ddt_oblique_half


def off_axis_conic_sag_der_xy(c, kappa, x, y, dx=0, dy=0, phi=None):
    """Cartesian partial derivatives of the sag of an off-axis conic.

    Singularity-free everywhere inside the aperture (where phi > 0), including
    at the local origin x=y=0.

    Parameters
    ----------
    c : float
        axial curvature of the conic
    kappa : float
        conic constant
    x : ndarray
        local x coordinate (off-axis section centered at x=y=0)
    y : ndarray
        local y coordinate
    dx : float
        shift of the surface in x with respect to the base conic vertex
    dy : float
        shift of the surface in y with respect to the base conic vertex
    phi : ndarray, optional
        (1 - (1+kappa) c^2 ((x+dx)^2 + (y+dy)^2))^.5
        computed if not provided

    Returns
    -------
    ndarray, ndarray
        dz/dx, dz/dy

    """
    X = x + dx
    Y = y + dy
    if phi is None:
        phi = phi_spheroid(c, kappa, X * X + Y * Y)
    return (c * X) / phi, (c * Y) / phi


def off_axis_conic_sag(c, kappa, r, t, dx, dy=0):
    """Sag of an off-axis conicoid.

    Parameters
    ----------
    c : float
        axial curvature of the conic
    kappa : float
        conic constant
    r : ndarray
        radial coordinate, where r=0 is centered on the off-axis section
    t : ndarray
        azimuthal coordinate
    dx : float
        shift of the surface in x with respect to the base conic vertex
    dy : float
        shift of the surface in y with respect to the base conic vertex

    Returns
    -------
    ndarray
        surface sag, z(x,y)

    """
    agg = _off_axis_aggregate(r, t, dx, dy)
    return (c * agg) / (1 + phi_spheroid(c, kappa, agg))


def off_axis_conic_der(c, kappa, r, t, dx, dy=0):
    """Radial and azimuthal derivatives of an off-axis conic.

    Parameters
    ----------
    c : float
        axial curvature of the conic
    kappa : float
        conic constant
    r : ndarray
        radial coordinate, where r=0 is centered on the off-axis section
    t : ndarray
        azimuthal coordinate
    dx : float
        shift of the surface in x with respect to the base conic vertex
    dy : float
        shift of the surface in y with respect to the base conic vertex

    Returns
    -------
    ndarray, ndarray
        d/dr(z), d/dt(z)

    """
    agg, ddr_oblique, ddt_oblique_half = _off_axis_aggregate_with_derivs(r, t, dx, dy)
    ddt_oblique = 2 * ddt_oblique_half
    csq = c * c
    c3 = csq * c
    phi = phi_spheroid(c, kappa, agg)
    phip1 = 1 + phi
    phip1sq = phip1 * phip1

    # d/dr
    term1 = (c * ddr_oblique) / phip1
    term2 = (c3 * (1 + kappa) * ddr_oblique * agg) / ((2 * phi) * phip1sq)
    dr = term1 + term2

    # d/dt
    term1 = (c * ddt_oblique) / phip1
    term2 = (c3 * (1 + kappa) * ddt_oblique_half * agg) / (phi * phip1sq)
    dt = term1 + term2

    return dr, dt


def off_axis_conic_sigma(c, kappa, r, t, dx, dy=0):
    """Lowercase sigma (direction cosine projection term) for an off-axis conic.

    See Eq. (5.2) of oe-20-3-2483.

    Parameters
    ----------
    c : float
        axial curvature of the conic
    kappa : float
        conic constant
    r : ndarray
        radial coordinate, where r=0 is centered on the off-axis section
    t : ndarray
        azimuthal coordinate
    dx : float
        shift of the surface in x with respect to the base conic vertex
    dy : float
        shift of the surface in y with respect to the base conic vertex

    Returns
    -------
    sigma(r,t)

    """
    agg = _off_axis_aggregate(r, t, dx, dy)
    csq = c * c
    num = phi_spheroid(c, kappa, agg)
    den = np.sqrt(1 - kappa * csq * agg)  # flipped sign, 1-kappa (NOT a spheroid phi)
    return num / den


def off_axis_conic_sigma_der(c, kappa, r, t, dx, dy=0):
    """Derivatives of 1/off_axis_conic_sigma.

    See Eq. (5.2) of oe-20-3-2483.

    Parameters
    ----------
    c : float
        axial curvature of the conic
    kappa : float
        conic constant
    r : ndarray
        radial coordinate, where r=0 is centered on the off-axis section
    t : ndarray
        azimuthal coordinate
    dx : float
        shift of the surface in x with respect to the base conic vertex
    dy : float
        shift of the surface in y with respect to the base conic vertex

    Returns
    -------
    ndarray, ndarray
        d/dr(z), d/dt(z)

    """
    agg, ddr_oblique, ddt_oblique_half = _off_axis_aggregate_with_derivs(r, t, dx, dy)
    csq = c * c
    # 1/sigma = sqrt(1 - kappa c^2 A) / sqrt(1 - (1+kappa) c^2 A) = u/phi
    # let phi = sqrt(1 - (1+kappa) c^2 A), notquitephi (= u) = sqrt(1 - kappa c^2 A)
    # d/dx (u/phi) = u'/phi + u * d/dx (1/phi)
    #              = u'/phi - u*phi'/phi^2
    # phi' = -(1+kappa) c^2 A' / (2 phi)  =>  -u*phi'/phi^2 = u (1+kappa) c^2 A' / (2 phi^3)
    # u'   = -kappa c^2 A' / (2 u)        =>  u'/phi = -kappa c^2 A' / (2 u phi)
    phi = phi_spheroid(c, kappa, agg)
    phi_cubed = phi * phi * phi  # = (1 - (1+kappa) c^2 A) ^ (3/2)
    notquitephi = np.sqrt(1 - kappa * csq * agg)

    # d/dr
    term1 = (csq * (1 + kappa) * ddr_oblique * notquitephi) / (2 * phi_cubed)
    term2 = (-csq * kappa * ddr_oblique) / (2 * phi * notquitephi)
    dr = term1 + term2

    # d/dt — same structure but with d/dt of aggregate_term (= 2 * ddt_oblique_half)
    term1 = (csq * (1 + kappa) * ddt_oblique_half * notquitephi) / phi_cubed
    term2 = (-csq * kappa * ddt_oblique_half) / (phi * notquitephi)
    dt = term1 + term2
    return dr, dt


def Q2d_and_der(cm0, ams, bms, x, y, normalization_radius, c, k, dx=0, dy=0):
    """Q-type freeform surface, with base (perhaps shifted) conicoic.

    Returns the sag and the polar derivatives (dz/dr, dz/dt).  Q2DSag.FFp
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

    base_sag = off_axis_conic_sag(c, k, r, t, dx, dy)
    base_primer, base_primet = off_axis_conic_der(c, k, r, t, dx, dy)

    # Eq. 5.1/5.2
    sigma = off_axis_conic_sigma(c, k, r, t, dx, dy)
    sigma = 1 / sigma
    sigmaprimer, sigmaprimet = off_axis_conic_sigma_der(c, k, r, t, dx, dy)

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
    base_sag = off_axis_conic_sag(c, k, r, t, dx, dy)
    sigma_inv = 1 / off_axis_conic_sigma(c, k, r, t, dx, dy)
    return base_sag + sigma_inv * z

__all__ = [
    'product_rule',
    'phi_spheroid',
    'der_direction_cosine_spheroid',
    'sphere_sag',
    'sphere_sag_der',
    'conic_sag',
    'conic_sag_der',
    'conic_sag_der_xy',
    'even_asphere_sag',
    'even_asphere_sag_der_xy',
    'off_axis_conic_sag',
    'off_axis_conic_der',
    'off_axis_conic_sigma',
    'off_axis_conic_sigma_der',
    'off_axis_conic_sag_der_xy',
    'Q2d_and_der',
    'Q2d_sag',
    '_conic_base_xy',
    '_conic_base_xy_F',
    '_add_conic_base_FFp',
    '_add_conic_base_F',
]
