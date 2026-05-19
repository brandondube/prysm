"""Surface types and calculus."""

from prysm.mathops import np
from prysm.conf import config
from prysm.coordinates import (
    cart_to_polar,
    promote_3d_point,
    coerce_3d_rotation,
    apply_tilt_decenter,
)
from prysm.polynomials.qpoly import compute_z_Q2d, compute_z_zprime_Q2d

from prysm.polynomials import (
    cheby1_2d_sum,
    cheby1_2d_sum_der_xy,
    jacobi_radial_sum,
    jacobi_radial_sum_der_xy,
    xy_sum,
    xy_sum_der_xy,
    zernike_sum,
    zernike_sum_der_xy,
)

from .spencer_and_murty import (
    newton_raphson_solve_s,
    intersect as _newton_intersect,
    SURFACE_INTERSECTION_DEFAULT_MAXITER,
    STYPE_REFLECT,
    STYPE_REFRACT,
    STYPE_EVAL,
)


def circular_aperture(radius, x0=0.0, y0=0.0):
    """Create a circular surface aperture predicate.

    Parameters
    ----------
    radius : float
        aperture radius in the same units as the surface coordinates.
    x0, y0 : float, optional
        aperture center in the surface's local coordinate system.

    Returns
    -------
    callable
        function of x and y returning True for points inside the circle.

    """
    radius = float(radius)
    x0 = float(x0)
    y0 = float(y0)
    rsq = radius * radius

    def aperture(x, y):
        dx = x - x0
        dy = y - y0
        return dx * dx + dy * dy <= rsq

    return aperture


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

    Returns the sag and the polar derivatives (dz/dr, dz/dt).  Surface.q2d's
    FFp closure converts to Cartesian (dz/dx, dz/dy) via the chain rule with
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


def ray_plane_intersect(P, S, return_valid=False):
    """Intersect rays P + t*S with the local-frame plane Z = 0.

    Parameters
    ----------
    P : ndarray
        shape (N, 3) ray origins in the surface's local frame
    S : ndarray
        shape (N, 3) unit direction cosines
    return_valid : bool, optional
        if True, also return a length-N boolean mask flagging rays whose
        Z-direction is nonzero (rays parallel to the plane never intersect).

    Returns
    -------
    Q : ndarray
        shape (N, 3) intersection points
    n : ndarray
        shape (N, 3) surface normals in S&M form (-Fx, -Fy, 1) = (0, 0, 1)
    valid : ndarray, optional
        shape (N,) boolean; only returned when return_valid is True.

    """
    P, S = np.atleast_2d(P, S)
    Sz = S[..., 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        t = -P[..., 2] / Sz
        Q = P + t[..., np.newaxis] * S
    n = np.zeros(Q.shape, dtype=Q.dtype)
    n[..., 2] = 1.0
    if return_valid:
        return Q, n, (Sz != 0)
    return Q, n


def _conic_quadratic_t(c, kappa, P1, S, dx, dy):
    """Solve the conic-intersection quadratic for the smaller-|t| root.

    Assumes P1 sits on the surface's vertex tangent plane (Z=0).  Returns
    (t, disc_nonneg) where disc_nonneg flags rays whose quadratic had a real
    root.  Caller must wrap in np.errstate(divide='ignore', invalid='ignore')
    since this routine silently swallows 1/0 and sqrt(<0) for missed rays.

    """
    Sx = S[..., 0]
    Sy = S[..., 1]
    Sz = S[..., 2]
    Xp = P1[..., 0] + dx
    Yp = P1[..., 1] + dy
    A_ = 1.0 + kappa * Sz * Sz
    B_ = Xp * Sx + Yp * Sy - Sz / c
    C_ = Xp * Xp + Yp * Yp
    disc = B_ * B_ - A_ * C_
    disc_nonneg = (disc >= 0)
    disc = np.where(disc_nonneg, disc, np.zeros_like(disc))
    sqrt_disc = np.sqrt(disc)
    # quadratic roots, with paraboloid-axial-ray fallback (A_==0 collapses
    # to the linear equation 2 B_ t + C_ = 0).
    a_is_zero = (A_ == 0)
    safe_A = np.where(a_is_zero, 1.0, A_)
    t1 = (-B_ - sqrt_disc) / safe_A
    t2 = (-B_ + sqrt_disc) / safe_A
    safe_B = np.where(B_ == 0, 1.0, B_)
    t_lin = -C_ / (2.0 * safe_B)
    t1 = np.where(a_is_zero, t_lin, t1)
    t2 = np.where(a_is_zero, t_lin, t2)
    # vertex-side intersection: smaller |t|.
    t = np.where(np.abs(t1) <= np.abs(t2), t1, t2)
    return t, disc_nonneg


def ray_conic_intersect(P, S, c, kappa, dx=0.0, dy=0.0, return_valid=False):
    """Intersect rays P + t*S with a (possibly off-axis) conicoid.

    Surface implicit form (from sag z = c*A / (1 + sqrt(1 - (1+k)c^2 A))
    with A = (X+dx)^2 + (Y+dy)^2):

        (X + dx)^2 + (Y + dy)^2 + (1 + k) Z^2 - 2 Z / c = 0

    Substituting X = Px + t Sx, ..., yields a quadratic in t.  This
    routine first projects P onto the vertex tangent plane (Z = 0) via
    P1 = P - (Pz / Sz) S, matching the convention used by the Newton-Raphson
    path in spencer_and_murty.intersect.  In the projected frame Pz = 0, so
        A_ = 1 + k Sz^2
        B_ = (P1x + dx) Sx + (P1y + dy) Sy - Sz / c
        C_ = (P1x + dx)^2 + (P1y + dy)^2

    The root with smaller |t| is selected — the intersection nearest the
    vertex tangent plane.  When A_ == 0 (paraboloid + axial ray) the
    quadratic degenerates to a linear equation t = -C_ / (2 B_).

    Parameters
    ----------
    P : ndarray
        shape (N, 3) ray origins in the surface's local frame
    S : ndarray
        shape (N, 3) unit direction cosines
    c : float
        vertex curvature (1 / R)
    kappa : float
        conic constant (-1 = parabola, 0 = sphere, < -1 = hyperbola, > -1 = ellipse)
    dx, dy : float
        off-axis shift of the surface relative to the parent conic vertex;
        dx = dy = 0 for an on-axis surface
    return_valid : bool, optional
        if True, also return a length-N boolean mask flagging rays that
        actually intersect the surface (positive discriminant) and land
        inside the supporting region (phi > 0).

    Returns
    -------
    Q : ndarray
        shape (N, 3) intersection points
    n : ndarray
        shape (N, 3) surface normals in S&M form (-Fx, -Fy, 1)
    valid : ndarray, optional
        shape (N,) boolean; only returned when return_valid is True.

    """
    if c == 0.0:
        # The conic z = c A / (1 + sqrt(1 - (1+k) c^2 A)) collapses to z = 0
        # when c=0; the closed-form quadratic divides by c, so dispatch to the
        # plane intersector to avoid 1/0.  The off-axis shift dx/dy is
        # meaningless for c=0 (a plane has no vertex) and is silently ignored.
        return ray_plane_intersect(P, S, return_valid=return_valid)
    P, S = np.atleast_2d(P, S)
    Sz = S[..., 2]
    # invalid arithmetic (1/0, sqrt(-x)) is expected for rays that miss the
    # surface; we expose those via the valid mask, not via FP warnings.
    with np.errstate(divide='ignore', invalid='ignore'):
        # project P onto the vertex tangent plane (matches Newton path convention)
        s0 = -P[..., 2] / Sz
        P1 = P + s0[..., np.newaxis] * S
        t, disc_nonneg = _conic_quadratic_t(c, kappa, P1, S, dx, dy)
        Q = P1 + t[..., np.newaxis] * S
        Xq = Q[..., 0] + dx
        Yq = Q[..., 1] + dy
        phi_arg = 1.0 - (1.0 + kappa) * c * c * (Xq * Xq + Yq * Yq)
        # rays that landed outside the supporting region get phi_arg < 0; clip
        # before sqrt so the normal computation doesn't propagate NaNs through
        # the valid rays in the same batch.
        phi = np.sqrt(np.where(phi_arg < 0, np.zeros_like(phi_arg), phi_arg))
        nx = -c * Xq / phi
        ny = -c * Yq / phi
    nz = np.ones_like(nx)
    n = np.stack([nx, ny, nz], axis=-1)
    if return_valid:
        return Q, n, disc_nonneg & (phi_arg >= 0)
    return Q, n


def ray_sphere_intersect(P, S, c, return_valid=False):
    """Intersect rays P + t*S with a sphere of curvature c, vertex at origin.

    Thin wrapper over ray_conic_intersect with kappa=0.

    Parameters
    ----------
    P : ndarray
        shape (N, 3) ray origins
    S : ndarray
        shape (N, 3) unit direction cosines
    c : float
        vertex curvature (1 / R)
    return_valid : bool, optional
        forwarded to ray_conic_intersect

    Returns
    -------
    Q : ndarray
        intersection points
    n : ndarray
        surface normals in S&M form
    valid : ndarray, optional
        shape (N,) boolean; only returned when return_valid is True.

    """
    return ray_conic_intersect(P, S, c, 0.0, return_valid=return_valid)


def _ensure_P_vec(P):
    return promote_3d_point(P, dtype=config.precision)


def _none_or_rotmat(R):
    return coerce_3d_rotation(R)


def _apply_tilt_decenter(P, R, tilt, decenter, tilt_radians):
    """Combine a base (P, R) with a (tilt, decenter) perturbation.

    decenter is added to the surface position vector P.  tilt is
    converted to a rotation matrix via make_rotation_matrix and composed
    with R as R_total = R @ R_tilt — i.e. the perturbation acts in
    the surface's *nominal local frame* (apply tilt first, then any base
    rotation).  For a surface with no nominal R this collapses to
    R_total = R_tilt.

    Either or both perturbations may be None.

    """
    return apply_tilt_decenter(P, R, tilt=tilt, decenter=decenter,
                               tilt_radians=tilt_radians,
                               dtype=config.precision)


def _map_stype(typ):
    if isinstance(typ, int):
        return typ

    typ_lc = typ.lower()
    if typ_lc in ('refl', 'reflect'):
        return STYPE_REFLECT
    if typ_lc in ('refr', 'refract'):
        return STYPE_REFRACT
    if typ_lc == 'eval':
        return STYPE_EVAL
    raise ValueError(
        f'unknown surface type {typ!r}; expected one of '
        "'refl'/'reflect', 'refr'/'refract', 'eval', or an STYPE_* int."
    )


def _validate_n_and_typ(n, typ):
    if typ == STYPE_REFRACT and n is None:
        raise ValueError('refractive surfaces must have a refractive index function, not None')


class Surface:
    """A surface for raytracing.

    Contract for subclasses:
    - Every surface exposes two callables:
      - F(x, y) -> z, the sag-only kernel.  Cheaper than FFp; used by
        plotting and any consumer that doesn't need partial derivatives.
      - FFp(x, y) -> (z, dz/dx, dz/dy), sag plus Cartesian partials.  Used
        by sag_normal and by the default Newton-Raphson intersect path.
    - Surface.intersect defaults to Newton-Raphson on FFp.  Subclasses that
      admit a closed-form ray intersection (e.g. plane, sphere, conic)
      should override intersect and call the appropriate ray_*_intersect
      free function; subclasses with no analytic intersection should leave
      the default path in place.

    """
    def __init__(self, typ, P, n, FFp, F, R=None, params=None, bounding=None,
                 aperture=None, tilt=None, decenter=None, tilt_radians=False,
                 grating=None):
        """Create a new surface for raytracing.

        Parameters
        ----------
        typ : int or str
            if an int, must be one of the STYPE constants
            if a str, must be something in the set {'refl', 'reflect', 'refr', 'refract', 'eval'}
            the type of surface (reflection, refraction, no ray bend)
        P : ndarray
            global surface position, [X,Y,Z]
        n : callable n(wvl) -> refractive index
            a function which returns the index of refraction at the given wavelength
        FFp : callable of signature FFp(x, y) -> (z, dz/dx, dz/dy)
            sag plus Cartesian first partial derivatives at (x, y).
        F : callable of signature F(x, y) -> z
            sag only at (x, y).  Used by plotting and other sag-only
            consumers to skip the derivative work in FFp.
        R : ndarray
            rotation matrix, may be None
        params : dict, optional
            surface type specific parameters
        bounding : dict, optional
            bounding geometry description
            at the moment, outer_radius and inner_radius are the only values
            which are used for anything.  More will be added in the future
        aperture : callable, optional
            a function (x, y) -> bool returning True for points inside
            the surface's aperture and False outside.  raytrace calls
            this on each ray's intersection point; rays falling outside are
            flagged in the trace's status array as clipped and skipped on
            all subsequent surfaces.  If None (the default) every ray is
            considered to be inside the surface.
        tilt : iterable of length 3, optional
            (rz, ry, rx) rotation angles describing a tilt perturbation
            of the surface in its own local frame.  Composed with R as
            R_total = R @ R_tilt so the tilt acts before any nominal
            rotation; with R=None the tilt becomes the surface's full
            rotation.  Angles in degrees by default; pass
            tilt_radians=True to switch.
        decenter : iterable of length 3, optional
            (dx, dy, dz) translation perturbation, added directly to
            P.  Convenient for sensitivity analyses (see
            prysm.x.raytracing.sensitivity) without having to build a
            new surface object.
        tilt_radians : bool, optional
            if True, tilt is interpreted in radians; default degrees.
        grating : tuple (period, vector, order), optional
            opt-in grating modifier.  period is the groove period in
            length units (same as the rest of the system); vector is
            a length-3 unit vector specifying the grating direction in
            the surface's local frame; order is the integer
            diffraction order.  When set, raytrace applies the grating
            equation after Snell/reflect.  Evanescent diffraction is
            flagged in the trace's status as STATUS_TIR (see
            raytrace docstring).

        """
        typ = _map_stype(typ)
        P = _ensure_P_vec(P)
        R = _none_or_rotmat(R)
        P, R = _apply_tilt_decenter(P, R, tilt, decenter, tilt_radians)
        _validate_n_and_typ(n, typ)

        self.typ = typ
        self.P = P
        self.n = n
        self.FFp = FFp
        self.F = F
        self.R = R
        self.params = params
        self.bounding = bounding
        self.aperture = aperture
        self.grating = grating

    def sag_normal(self, x, y):
        """Sag z and normal [Fx, Fy, Fz] of the surface at the point (x,y).

        Parameters
        ----------
        x : ndarray
            x coordinate, non-normalized
        y : ndarray
            y coordinate, non-normalized

        Returns
        -------
        ndarray
            surface sag in Z

        """
        z, Fx, Fy = self.FFp(x, y)
        # faster than ones
        Fz = np.array([1.], dtype=config.precision)
        Fz = np.broadcast_to(Fz, Fx.shape)
        # F(X,Y,Z) = 0 = Z - F(x,Y)
        # d/dx, d/dy have leading - term, dz = 1 always
        der = np.stack([-Fx, -Fy, Fz], axis=1)
        return z, der

    def diffract(self, S_specular, r, n_post, wvl):
        """Apply per-ray grating diffraction to S_specular.

        Implements the vector grating equation for the post-Snell/reflect
        direction cosines:

            n_post * S'_tan = n_post * S_specular_tan + (m * lambda / d) * g_tan

        where the tangent components are taken w.r.t. the per-ray surface
        normal, g_tan is the supplied grating vector projected into the
        tangent plane, d the groove period, m the diffraction order,
        and lambda the vacuum wavelength.  The normal component of
        S' is recovered from |S'| = 1 with a sign chosen to match
        the propagation direction of S_specular.

        Parameters
        ----------
        S_specular : ndarray, shape (N, 3)
            Direction cosines after refraction or reflection.
        r : ndarray, shape (N, 3)
            Surface normal in S&M form (-Fx, -Fy, 1); unit-normalized
            internally.
        n_post : float
            Refractive index of the post-bend medium.  Equal to the
            refracted index for a refraction grating; equal to the pre-bend
            index for a reflection grating.
        wvl : float
            Vacuum wavelength, in the same length unit as the grating
            period.

        Returns
        -------
        S_diff : ndarray, shape (N, 3)
            Diffracted direction cosines.  Rays whose diffracted order is
            evanescent (tangential magnitude > 1) are returned unchanged
            and flagged via valid.
        valid : ndarray, shape (N,) of bool
            False where the diffraction order is evanescent.

        """
        if self.grating is None:
            return S_specular, np.ones(S_specular.shape[:-1], dtype=bool)
        period, g_vec, order = self.grating
        g_vec = np.asarray(g_vec, dtype=S_specular.dtype)
        # unit normal per ray
        n_norm = np.sqrt((r * r).sum(-1, keepdims=True))
        n_hat = r / n_norm
        # grating vector in cycles/length (q = g_vec / d), projected onto the
        # per-ray tangent plane: q_tan = q - (q . n_hat) n_hat
        q = g_vec / period
        q_dot_n = (q * n_hat).sum(-1, keepdims=True)
        q_tan = q - q_dot_n * n_hat
        # specular tangent
        s_dot_n = (S_specular * n_hat).sum(-1, keepdims=True)
        s_specular_tan = S_specular - s_dot_n * n_hat
        # diffracted tangent
        s_diff_tan = s_specular_tan + (order * wvl / n_post) * q_tan
        tan_sq = (s_diff_tan * s_diff_tan).sum(-1)
        valid = tan_sq <= 1.0
        normal_mag = np.sqrt(np.where(valid, 1.0 - tan_sq, np.zeros_like(tan_sq)))
        # preserve propagation direction of the specular ray
        sign = np.sign(s_dot_n[..., 0])
        S_diff = s_diff_tan + (sign * normal_mag)[..., np.newaxis] * n_hat
        # leave evanescent rays at their specular direction; raytrace flags
        # them via status.
        S_diff = np.where(valid[..., np.newaxis], S_diff, S_specular)
        return S_diff, valid

    def intersect(self, P, S, eps=None, maxiter=None, return_valid=False):
        """Intersect rays P + t*S with this surface, returning (Q, normal).

        Default implementation uses Newton-Raphson iteration on FFp.
        Subclasses that admit a closed-form ray intersection should override
        this method and call the appropriate ray_*_intersect free function;
        subclasses with no analytic form should leave this default in
        place.

        Parameters
        ----------
        P : ndarray
            shape (N, 3) ray origins in this surface's local frame
        S : ndarray
            shape (N, 3) unit direction cosines
        eps : float, optional
            Newton convergence tolerance; ignored by analytic subclasses
        maxiter : int, optional
            Newton iteration limit; ignored by analytic subclasses
        return_valid : bool, optional
            if True, also return a length-N boolean mask of rays that
            successfully intersected this surface.  Failures are NaN'd out
            in Q and normal regardless.

        Returns
        -------
        Q : ndarray
            shape (N, 3) intersection points
        n : ndarray
            shape (N, 3) surface normals in S&M form (-Fx, -Fy, 1)
        valid : ndarray, optional
            shape (N,) boolean; only returned when return_valid is True.

        """
        if maxiter is None:
            maxiter = SURFACE_INTERSECTION_DEFAULT_MAXITER
        return _newton_intersect(P, S, self.sag_normal, eps=eps, maxiter=maxiter,
                                 return_valid=return_valid)

    @classmethod
    def conic(cls, c, k, typ, P, n=None, R=None, bounding=None, aperture=None,
              tilt=None, decenter=None, tilt_radians=False, grating=None):
        """Conic surface type.

        for documentation on typ, P, N, R, and bounding see the docstring for
        Surface.__init__

        Parameters
        ----------
        c : float
            vertex curvature
        k : float
            conic constant
            -1 = parabola
            0 = sphere
            < - 1 = hyperbola
            > - 1 = ellipse

        Returns
        -------
        Surface
            a conic surface

        """
        params = dict()
        params['c'] = c
        params['k'] = k

        def FFp(x, y):
            return _conic_base_xy(params['c'], params['k'], x, y)

        def F(x, y):
            return _conic_base_xy_F(params['c'], params['k'], x, y)

        return Conic(typ=typ, P=P, n=n, FFp=FFp, F=F, R=R, params=params,
                     bounding=bounding, aperture=aperture,
                     tilt=tilt, decenter=decenter, tilt_radians=tilt_radians, grating=grating)

    @classmethod
    def off_axis_conic(cls, c, k, typ, P, dx=0, dy=0, n=None, R=None,
                       bounding=None, aperture=None,
                       tilt=None, decenter=None, tilt_radians=False, grating=None):
        """Off-axis conic surface type.

        for documentation on typ, P, N, R, and bounding see the docstring for
        Surface.__init__

        Parameters
        ----------
        c : float
            vertex curvature
        k : float
            conic constant
            -1 = parabola
            0 = sphere
            < - 1 = hyperbola
            > - 1 = ellipse
        dx : float
            off-axis distance in x
        dy : float
            off-axis distance in y

        Returns
        -------
        Surface
            a conic surface

        """
        params = dict()
        params['c'] = c
        params['k'] = k
        params['dx'] = dx
        params['dy'] = dy

        def FFp(x, y):
            c_, k_ = params['c'], params['k']
            X = x + params['dx']
            Y = y + params['dy']
            aggregate = X * X + Y * Y
            phi = phi_spheroid(c_, k_, aggregate)
            z = (c_ * aggregate) / (1 + phi)
            ddx = (c_ * X) / phi
            ddy = (c_ * Y) / phi
            return z, ddx, ddy

        def F(x, y):
            c_, k_ = params['c'], params['k']
            X = x + params['dx']
            Y = y + params['dy']
            return conic_sag(c_, k_, X * X + Y * Y)

        return OffAxisConic(typ=typ, P=P, n=n, FFp=FFp, F=F, R=R, params=params,
                            bounding=bounding, aperture=aperture,
                            tilt=tilt, decenter=decenter,
                            tilt_radians=tilt_radians, grating=grating)

    @classmethod
    def plane(cls, typ, P, n=None, R=None, bounding=None, aperture=None,
              tilt=None, decenter=None, tilt_radians=False, grating=None):
        """A plane normal to its local Z axis.

        for documentation on typ, P, N, R, and bounding see the docstring for
        Surface.__init__

        Returns
        -------
        Surface
            a planar surface

        """

        def FFp(x, y):
            zero = np.array([0.], dtype=x.dtype)
            zero_up = np.broadcast_to(zero, x.shape)
            return zero_up, zero_up, zero_up

        def F(x, y):
            zero = np.array([0.], dtype=x.dtype)
            return np.broadcast_to(zero, x.shape)

        return Plane(typ=typ, P=P, n=n, FFp=FFp, F=F, R=R, bounding=bounding,
                     aperture=aperture,
                     tilt=tilt, decenter=decenter, tilt_radians=tilt_radians, grating=grating)

    @classmethod
    def sphere(cls, c, typ, P, n, R=None, bounding=None, aperture=None,
               tilt=None, decenter=None, tilt_radians=False, grating=None):
        """A spherical surface.

        for documentation on typ, P, N, R, and bounding see the docstring for
        Surface.__init__

        Parameters
        ----------
        c : float
            vertex curvature

        Returns
        -------
        Surface
            a spherical surface

        """
        params = dict()
        params['c'] = c

        def FFp(x, y):
            c_ = params['c']
            rsq = x * x + y * y
            phi = phi_spheroid(c_, 0.0, rsq)
            z = sphere_sag(c_, rsq, phi=phi)
            dx = (c_ * x) / phi
            dy = (c_ * y) / phi
            return z, dx, dy

        def F(x, y):
            return sphere_sag(params['c'], x * x + y * y)

        return Sphere(typ=typ, P=P, n=n, FFp=FFp, F=F, R=R, params=params,
                      bounding=bounding, aperture=aperture,
                      tilt=tilt, decenter=decenter, tilt_radians=tilt_radians, grating=grating)

    @classmethod
    def even_asphere(cls, c, k, coefs, typ, P, n=None, R=None,
                     bounding=None, aperture=None,
                     tilt=None, decenter=None, tilt_radians=False, grating=None):
        """An even asphere: conic base plus polynomial in r^2.

        for documentation on typ, P, N, R, and bounding see the docstring for
        Surface.__init__

        Parameters
        ----------
        c : float
            vertex curvature
        k : float
            conic constant
        coefs : iterable of float
            polynomial coefficients [a4, a6, a8, ...] multiplying r^4, r^6, r^8, ...
            in the surface sag.  Empty list / None makes the surface a pure conic.

        Returns
        -------
        EvenAsphere
            an even-asphere surface

        """
        coefs = tuple(coefs) if coefs is not None else ()
        params = dict(c=c, k=k, coefs=coefs)

        def FFp(x, y):
            c_, k_, coefs_ = params['c'], params['k'], params['coefs']
            rsq = x * x + y * y
            phi = phi_spheroid(c_, k_, rsq)
            z = even_asphere_sag(c_, k_, coefs_, rsq)
            dx, dy = even_asphere_sag_der_xy(c_, k_, coefs_, x, y, phi=phi)
            return z, dx, dy

        def F(x, y):
            c_, k_, coefs_ = params['c'], params['k'], params['coefs']
            return even_asphere_sag(c_, k_, coefs_, x * x + y * y)

        return EvenAsphere(typ=typ, P=P, n=n, FFp=FFp, F=F, R=R, params=params,
                           bounding=bounding, aperture=aperture,
                           tilt=tilt, decenter=decenter,
                           tilt_radians=tilt_radians, grating=grating)

    @classmethod
    def q2d(cls, c, k, normalization_radius, cm0, ams, bms, typ, P,
            dx=0, dy=0, n=None, R=None, bounding=None, aperture=None,
            tilt=None, decenter=None, tilt_radians=False, grating=None):
        """A Forbes Q-2D freeform surface.

        Conic base plus the Q-2D polynomial expansion (Forbes, oe-20-3-2483).

        Parameters
        ----------
        c, k : float
            base conic curvature and conic constant
        normalization_radius : float
            radius by which to normalize rho (the Q polynomials are defined
            on a unit disk; rho/normalization_radius should lie in [0, 1]
            inside the aperture).
        cm0 : iterable of float
            coefficients for the m=0 Q radial expansion (axisymmetric piece)
        ams, bms : iterable of iterables of float
            ams[m-1] is the coefficient list for cos(m theta) Q radial
            expansion; same for bms and sin(m theta).  ams and
            bms must have the same length but may have different per-m
            radial extents.
        typ, P, n, R, bounding, aperture : see Surface.__init__
        dx, dy : float
            off-axis shift of the base conic vertex (same convention as
            Surface.off_axis_conic).

        Returns
        -------
        SurfaceQ2D
            a Q-2D freeform surface

        """
        cm0 = tuple(cm0) if cm0 is not None else (0.0,)
        ams = tuple(tuple(am) for am in ams)
        bms = tuple(tuple(bm) for bm in bms)
        params = dict(c=c, k=k, normalization_radius=float(normalization_radius),
                      cm0=cm0, ams=ams, bms=bms, dx=dx, dy=dy)

        def FFp(x, y):
            c_, k_ = params['c'], params['k']
            cm0_, ams_, bms_ = params['cm0'], params['ams'], params['bms']
            norm_r = params['normalization_radius']
            dx_, dy_ = params['dx'], params['dy']
            z, dr, dt = Q2d_and_der(cm0_, ams_, bms_, x, y, norm_r, c_, k_,
                                    dx=dx_, dy=dy_)
            # polar -> Cartesian via chain rule; r=0 has a 0/0 that we patch
            # to (0, 0) — for typical raytraces no ray lands exactly at the
            # vertex, and on-axis rays of an axisymmetric surface have zero
            # gradient by construction.  Asymmetric (m=1) modes do have a
            # finite gradient at r=0 but raytrace's seeded-Newton starts from
            # the conic root which is also at r=0 only for an exactly axial
            # ray; in that case the m=1 contribution to dz/dx, dz/dy is small
            # enough to be absorbed by the next Newton iteration.
            rsq = x * x + y * y
            r = np.sqrt(rsq)
            on_axis = (r == 0)
            safe_r = np.where(on_axis, 1.0, r)
            cost = x / safe_r
            sint = y / safe_r
            ddx = dr * cost - dt * sint / safe_r
            ddy = dr * sint + dt * cost / safe_r
            if np.any(on_axis):
                ddx = np.where(on_axis, np.zeros_like(ddx), ddx)
                ddy = np.where(on_axis, np.zeros_like(ddy), ddy)
            return z, ddx, ddy

        def F(x, y):
            return Q2d_sag(params['cm0'], params['ams'], params['bms'],
                           x, y, params['normalization_radius'],
                           params['c'], params['k'],
                           dx=params['dx'], dy=params['dy'])

        return SurfaceQ2D(typ=typ, P=P, n=n, FFp=FFp, F=F, R=R, params=params,
                          bounding=bounding, aperture=aperture,
                          tilt=tilt, decenter=decenter,
                          tilt_radians=tilt_radians, grating=grating)

    @classmethod
    def zernike(cls, c, k, normalization_radius, nms, coefs, typ, P,
                n=None, R=None, bounding=None, aperture=None,
                tilt=None, decenter=None, tilt_radians=False, grating=None,
                norm=True):
        """Zernike-deformed surface: conic base plus a Zernike-coefficient sum.

        sag(x, y) = conic_sag(c, k, r^2)
                  + sum_i coefs[i] * Z_{n_i, m_i}(x / R_n, y / R_n)

        where R_n = normalization_radius and the Zernike polynomials are
        evaluated on the unit disk.  Cartesian derivatives come from
        zernike_sum_der_xy which is singularity-free at the origin.

        for documentation on typ, P, n, R, bounding see the docstring for
        Surface.__init__.

        Parameters
        ----------
        c, k : float
            base conic curvature and conic constant
        normalization_radius : float
            radius by which to normalize (x, y) before Zernike evaluation;
            the Zernike polynomials are defined on the unit disk so
            sqrt(x^2+y^2)/normalization_radius should be <= 1 inside the
            aperture.
        nms : iterable of (int, int)
            (n, m) Zernike indices; parallel to coefs.
        coefs : iterable of float
            coefficients, parallel to nms.  Treated as orthonormal
            (unit-RMS) Zernike weights when norm=True (the default), or
            zero-to-peak weights when norm=False.
        norm : bool, optional
            see coefs.

        Returns
        -------
        SurfaceZernike

        """
        nms = tuple((int(nn), int(mm)) for nn, mm in nms)
        coefs = tuple(float(co) for co in coefs)
        if len(nms) != len(coefs):
            raise ValueError(
                f'nms and coefs must be parallel; got {len(nms)} and {len(coefs)}'
            )
        params = dict(c=c, k=k,
                      normalization_radius=float(normalization_radius),
                      nms=nms, coefs=coefs, norm=bool(norm))

        def FFp(x, y):
            norm_r = params['normalization_radius']
            z_p, ddx_p, ddy_p = zernike_sum_der_xy(
                params['coefs'], params['nms'],
                x / norm_r, y / norm_r, norm=params['norm'])
            return _add_conic_base_FFp(params['c'], params['k'], x, y,
                                       z_p, ddx_p / norm_r, ddy_p / norm_r)

        def F(x, y):
            norm_r = params['normalization_radius']
            z_p = zernike_sum(params['coefs'], params['nms'],
                              x / norm_r, y / norm_r, norm=params['norm'])
            return _add_conic_base_F(params['c'], params['k'], x, y, z_p)

        return SurfaceZernike(typ=typ, P=P, n=n, FFp=FFp, F=F, R=R, params=params,
                              bounding=bounding, aperture=aperture,
                              tilt=tilt, decenter=decenter,
                              tilt_radians=tilt_radians, grating=grating)

    @classmethod
    def xy(cls, c, k, normalization_radius, mns, coefs, typ, P,
           n=None, R=None, bounding=None, aperture=None,
           tilt=None, decenter=None, tilt_radians=False, grating=None):
        """XY polynomial surface: conic base plus a sum of x^m y^n monomials.

        sag(x, y) = conic_sag(c, k, r^2)
                  + sum_i coefs[i] * (x / R_n)^{m_i} * (y / R_n)^{n_i}

        Constant term (m, n) == (0, 0) is allowed and behaves as a piston
        offset of the surface vertex.

        Parameters
        ----------
        c, k : float
            base conic curvature and conic constant
        normalization_radius : float
            radius by which to normalize (x, y) before monomial
            evaluation.  Pass 1.0 for the un-normalized Zemax XY surface
            convention.
        mns : iterable of (int, int)
            (m, n) integer powers; parallel to coefs.
        coefs : iterable of float
            coefficients, parallel to mns.

        Returns
        -------
        SurfaceXY

        """
        mns = tuple((int(mm), int(nn)) for mm, nn in mns)
        coefs = tuple(float(co) for co in coefs)
        if len(mns) != len(coefs):
            raise ValueError(
                f'mns and coefs must be parallel; got {len(mns)} and {len(coefs)}'
            )
        params = dict(c=c, k=k,
                      normalization_radius=float(normalization_radius),
                      mns=mns, coefs=coefs)

        def FFp(x, y):
            norm_r = params['normalization_radius']
            z_p, ddx_p, ddy_p = xy_sum_der_xy(params['coefs'], params['mns'],
                                              x / norm_r, y / norm_r,
                                              cartesian_grid=False)
            return _add_conic_base_FFp(params['c'], params['k'], x, y,
                                       z_p, ddx_p / norm_r, ddy_p / norm_r)

        def F(x, y):
            norm_r = params['normalization_radius']
            z_p = xy_sum(params['coefs'], params['mns'],
                         x / norm_r, y / norm_r, cartesian_grid=False)
            return _add_conic_base_F(params['c'], params['k'], x, y, z_p)

        return SurfaceXY(typ=typ, P=P, n=n, FFp=FFp, F=F, R=R, params=params,
                         bounding=bounding, aperture=aperture,
                         tilt=tilt, decenter=decenter,
                         tilt_radians=tilt_radians, grating=grating)

    @classmethod
    def chebyshev(cls, c, k, x_norm, y_norm, mns, coefs, typ, P,
                  n=None, R=None, bounding=None, aperture=None,
                  tilt=None, decenter=None, tilt_radians=False, grating=None):
        """Tensor-product Chebyshev (T-polynomial) freeform surface.

        sag(x, y) = conic_sag(c, k, r^2)
                  + sum_i coefs[i] * T_{m_i}(x / x_norm) * T_{n_i}(y / y_norm)

        The polynomials are orthogonal on [-1, 1] x [-1, 1], so for
        meaningful expansion x / x_norm and y / y_norm should fall in
        that range inside the aperture.  Common for rectangular freeform
        sections.

        Parameters
        ----------
        c, k : float
            base conic curvature and conic constant
        x_norm, y_norm : float
            normalization half-widths along the local x- and y-axes.
        mns : iterable of (int, int)
            (m, n) Chebyshev orders along x and y, parallel to coefs.
        coefs : iterable of float
            coefficients, parallel to mns.

        Returns
        -------
        SurfaceChebyshev

        """
        mns = tuple((int(mm), int(nn)) for mm, nn in mns)
        coefs = tuple(float(co) for co in coefs)
        if len(mns) != len(coefs):
            raise ValueError(
                f'mns and coefs must be parallel; got {len(mns)} and {len(coefs)}'
            )
        params = dict(c=c, k=k,
                      x_norm=float(x_norm), y_norm=float(y_norm),
                      mns=mns, coefs=coefs)

        def FFp(x, y):
            xn = params['x_norm']
            yn = params['y_norm']
            z_p, ddx_p, ddy_p = cheby1_2d_sum_der_xy(
                params['coefs'], params['mns'], x / xn, y / yn, xn, yn)
            return _add_conic_base_FFp(params['c'], params['k'], x, y,
                                       z_p, ddx_p, ddy_p)

        def F(x, y):
            z_p = cheby1_2d_sum(params['coefs'], params['mns'],
                                x / params['x_norm'], y / params['y_norm'])
            return _add_conic_base_F(params['c'], params['k'], x, y, z_p)

        return SurfaceChebyshev(typ=typ, P=P, n=n, FFp=FFp, F=F, R=R, params=params,
                                bounding=bounding, aperture=aperture,
                                tilt=tilt, decenter=decenter,
                                tilt_radians=tilt_radians, grating=grating)

    @classmethod
    def jacobi(cls, c, k, normalization_radius, alpha, beta, ns, coefs, typ, P,
               n=None, R=None, bounding=None, aperture=None,
               tilt=None, decenter=None, tilt_radians=False, grating=None):
        """Axisymmetric Jacobi-radial freeform: conic base plus a Jacobi
        polynomial expansion in u = 2 (r / R_n)^2 - 1.

        sag(x, y) = conic_sag(c, k, r^2)
                  + sum_i coefs[i] * P_{n_i}^{(alpha, beta)}(2 (r / R_n)^2 - 1)

        Generalizes the Forbes Q-bfs / Q-con axisymmetric basis: with
        alpha=beta=-1/2 this reduces to the Chebyshev-T radial basis,
        alpha=beta=0 to Legendre.  Singularity-free at the origin
        because du/dx = 4 x / R_n^2 is smooth in Cartesian coordinates.

        Parameters
        ----------
        c, k : float
            base conic curvature and conic constant
        normalization_radius : float
            radius by which to normalize r; r / normalization_radius
            should be <= 1 inside the aperture for the basis to lie on the
            Jacobi argument's natural [-1, 1] range.
        alpha, beta : float
            Jacobi weight parameters.  Re-using the same (alpha, beta)
            pair across surfaces preserves the orthogonality structure.
        ns : iterable of int
            Jacobi radial orders, parallel to coefs.
        coefs : iterable of float
            coefficients, parallel to ns.

        Returns
        -------
        SurfaceJacobi

        """
        ns = tuple(int(nn) for nn in ns)
        coefs = tuple(float(co) for co in coefs)
        if len(ns) != len(coefs):
            raise ValueError(
                f'ns and coefs must be parallel; got {len(ns)} and {len(coefs)}'
            )
        params = dict(c=c, k=k,
                      normalization_radius=float(normalization_radius),
                      alpha=float(alpha), beta=float(beta),
                      ns=ns, coefs=coefs)

        def FFp(x, y):
            z_p, ddx_p, ddy_p = jacobi_radial_sum_der_xy(
                params['coefs'], params['ns'],
                params['alpha'], params['beta'],
                x, y, params['normalization_radius'])
            return _add_conic_base_FFp(params['c'], params['k'], x, y,
                                       z_p, ddx_p, ddy_p)

        def F(x, y):
            z_p = jacobi_radial_sum(
                params['coefs'], params['ns'],
                params['alpha'], params['beta'],
                x, y, params['normalization_radius'])
            return _add_conic_base_F(params['c'], params['k'], x, y, z_p)

        return SurfaceJacobi(typ=typ, P=P, n=n, FFp=FFp, F=F, R=R, params=params,
                             bounding=bounding, aperture=aperture,
                             tilt=tilt, decenter=decenter,
                             tilt_radians=tilt_radians, grating=grating)

    @classmethod
    def toroid(cls, c_x, c_y, k_y, coefs_y, typ, P,
               n=None, R=None, bounding=None, aperture=None,
               tilt=None, decenter=None, tilt_radians=False, grating=None):
        """Zemax-style toroidal surface.

        sag(x, y) = even_asphere_sag(c_y, k_y, coefs_y, y^2)
                  + sphere_sag(c_x, x^2)

        The Y profile is an even asphere with vertex curvature c_y,
        conic constant k_y, and polynomial coefficients coefs_y
        ([a4, a6, ...] multiplying y^4, y^6, ...).  The X profile
        is a cylindrical-sphere arc of curvature c_x.  Common in
        cylindrical-lens and anamorphic optics.

        Parameters
        ----------
        c_x : float
            X-direction (cylindrical) curvature
        c_y : float
            Y-direction vertex curvature
        k_y : float
            Y-direction conic constant
        coefs_y : iterable of float
            Y even-asphere polynomial coefficients [a4, a6, a8, ...]
            multiplying y^4, y^6, y^8, ... in the Y sag.  Empty / None
            makes the Y profile a pure conic.

        Returns
        -------
        Toroid

        """
        coefs_y = tuple(coefs_y) if coefs_y is not None else ()
        params = dict(c_x=float(c_x), c_y=float(c_y), k_y=float(k_y),
                      coefs_y=coefs_y)

        def FFp(x, y):
            c_x_ = params['c_x']
            c_y_ = params['c_y']
            k_y_ = params['k_y']
            coefs_y_ = params['coefs_y']
            xsq = x * x
            ysq = y * y
            # X-direction: cylindrical sphere of curvature c_x
            phi_x = phi_spheroid(c_x_, 0.0, xsq)
            z_x = sphere_sag(c_x_, xsq, phi=phi_x)
            ddx = (c_x_ * x) / phi_x
            # Y-direction: even asphere along y (treat x = 0 in the 2D helper)
            zero = np.zeros_like(y)
            z_y = even_asphere_sag(c_y_, k_y_, coefs_y_, ysq)
            _, ddy = even_asphere_sag_der_xy(c_y_, k_y_, coefs_y_, zero, y)
            return z_x + z_y, ddx, ddy

        def F(x, y):
            z_x = sphere_sag(params['c_x'], x * x)
            z_y = even_asphere_sag(params['c_y'], params['k_y'],
                                   params['coefs_y'], y * y)
            return z_x + z_y

        return Toroid(typ=typ, P=P, n=n, FFp=FFp, F=F, R=R, params=params,
                      bounding=bounding, aperture=aperture,
                      tilt=tilt, decenter=decenter,
                      tilt_radians=tilt_radians, grating=grating)

    @classmethod
    def biconic(cls, c_x, c_y, k_x, k_y, typ, P,
                n=None, R=None, bounding=None, aperture=None,
                tilt=None, decenter=None, tilt_radians=False, grating=None):
        """Biconic surface: independent conics on each axis.

        sag(x, y) = (c_x x^2 + c_y y^2) /
                    (1 + sqrt(1 - (1+k_x) c_x^2 x^2 - (1+k_y) c_y^2 y^2))

        Generalizes the on-axis conic to anisotropic curvature and conic
        constant.  Reduces to Conic when c_x == c_y and
        k_x == k_y.

        Parameters
        ----------
        c_x, c_y : float
            X- and Y-direction vertex curvatures
        k_x, k_y : float
            X- and Y-direction conic constants

        Returns
        -------
        Biconic

        """
        params = dict(c_x=float(c_x), c_y=float(c_y),
                      k_x=float(k_x), k_y=float(k_y))

        def FFp(x, y):
            c_x_ = params['c_x']
            c_y_ = params['c_y']
            kx_ = params['k_x']
            ky_ = params['k_y']
            xsq = x * x
            ysq = y * y
            one_plus_kx = 1.0 + kx_
            one_plus_ky = 1.0 + ky_
            phi_arg = 1 - one_plus_kx * c_x_ * c_x_ * xsq \
                       - one_plus_ky * c_y_ * c_y_ * ysq
            phi = np.sqrt(phi_arg)
            one_plus_phi = 1 + phi
            num = c_x_ * xsq + c_y_ * ysq
            z = num / one_plus_phi
            # dz/dx = c_x x [2 phi (1+phi) + N (1+kx) c_x] / (phi (1+phi)^2)
            # dz/dy similarly with (1+ky) c_y
            two_phi_one_plus_phi = 2 * phi * one_plus_phi
            den = phi * one_plus_phi * one_plus_phi
            ddx = c_x_ * x * (two_phi_one_plus_phi + num * one_plus_kx * c_x_) / den
            ddy = c_y_ * y * (two_phi_one_plus_phi + num * one_plus_ky * c_y_) / den
            return z, ddx, ddy

        def F(x, y):
            c_x_ = params['c_x']
            c_y_ = params['c_y']
            xsq = x * x
            ysq = y * y
            phi = np.sqrt(1 - (1.0 + params['k_x']) * c_x_ * c_x_ * xsq
                            - (1.0 + params['k_y']) * c_y_ * c_y_ * ysq)
            return (c_x_ * xsq + c_y_ * ysq) / (1 + phi)

        return Biconic(typ=typ, P=P, n=n, FFp=FFp, F=F, R=R, params=params,
                       bounding=bounding, aperture=aperture,
                       tilt=tilt, decenter=decenter,
                       tilt_radians=tilt_radians, grating=grating)


class Plane(Surface):
    """A plane normal to its local Z axis, with closed-form ray intersection."""

    _analytic_intersect = True

    def intersect(self, P, S, eps=None, maxiter=None, return_valid=False):
        return ray_plane_intersect(P, S, return_valid=return_valid)


class Sphere(Surface):
    """A spherical surface with closed-form ray-sphere intersection."""

    _analytic_intersect = True

    def intersect(self, P, S, eps=None, maxiter=None, return_valid=False):
        return ray_sphere_intersect(P, S, self.params['c'], return_valid=return_valid)


class Conic(Surface):
    """An on-axis conic surface with closed-form ray-conicoid intersection."""

    _analytic_intersect = True

    def intersect(self, P, S, eps=None, maxiter=None, return_valid=False):
        p = self.params
        return ray_conic_intersect(P, S, p['c'], p['k'], return_valid=return_valid)


class OffAxisConic(Surface):
    """An off-axis section of a conicoid with closed-form ray intersection."""

    _analytic_intersect = True

    def intersect(self, P, S, eps=None, maxiter=None, return_valid=False):
        p = self.params
        return ray_conic_intersect(P, S, p['c'], p['k'], dx=p['dx'], dy=p['dy'],
                                   return_valid=return_valid)


class _ConicSeededNewtonSurface(Surface):
    """Internal base for surfaces with a (possibly notional) conic base plus
    a perturbation.

    The implicit surface equation is non-rational in the ray parameter so
    there is no closed-form intersection.  Newton-Raphson is seeded with
    the closed-form conic intersection; the perturbation is typically
    small enough that convergence is reached in 1-2 iterations versus many
    starting from t=0.

    Default seed reads params['c'], params['k'], optional
    params['dx'], params['dy'].  Subclasses with a non-trivial
    seed (e.g. Toroid, Biconic, which have no single base conic)
    override _seed_conic().

    """

    def _seed_conic(self):
        """Return (c, k, dx, dy) for the Newton-seed conic.

        Default reads from self.params; override when the seed must be
        derived from other parameters.

        """
        p = self.params
        return p['c'], p['k'], p.get('dx', 0.0), p.get('dy', 0.0)

    def intersect(self, P, S, eps=None, maxiter=None, return_valid=False):
        # Not a closed-form intersection — Newton seeded from the base conic's
        # analytic root.  _analytic_intersect intentionally left False so the
        # raytrace engine maps a non-convergence to STATUS_NEWTON, not
        # STATUS_MISS.
        if maxiter is None:
            maxiter = SURFACE_INTERSECTION_DEFAULT_MAXITER
        P, S = np.atleast_2d(P, S)
        Sz = S[..., 2]
        # project onto the vertex tangent plane (Newton's reference frame);
        # P1.z == 0 by construction so the segment length from P1 to Q_conic
        # along S equals Q_conic.z / Sz.
        s0 = -P[..., 2] / Sz
        P1 = P + s0[..., np.newaxis] * S
        c_seed, k_seed, dx_seed, dy_seed = self._seed_conic()
        Q_conic, _ = ray_conic_intersect(P1, S, c_seed, k_seed,
                                         dx=dx_seed, dy=dy_seed)
        s1 = Q_conic[..., 2] / Sz
        return newton_raphson_solve_s(P1, S, self.sag_normal,
                                      s1=s1, eps=eps, maxiter=maxiter,
                                      return_valid=return_valid)


class EvenAsphere(_ConicSeededNewtonSurface):
    """Even asphere: conic base plus polynomial in r^2."""


class SurfaceQ2D(_ConicSeededNewtonSurface):
    """Forbes Q-2D freeform surface: (possibly off-axis) conic base plus a
    Q-polynomial expansion in normalized polar coordinates.
    """


class SurfaceZernike(_ConicSeededNewtonSurface):
    """Conic base plus a Zernike-coefficient sum on the unit disk.

    Cartesian derivatives are computed directly via zernike_sum_der_xy,
    so the surface normal is smooth across the entire disk including the
    origin.

    """


class SurfaceXY(_ConicSeededNewtonSurface):
    """Conic base plus an XY polynomial sag (sum of c_{ij} x^i y^j)."""


class SurfaceChebyshev(_ConicSeededNewtonSurface):
    """Conic base plus a tensor-product Chebyshev-T sag on a rectangular
    normalization region.
    """


class SurfaceJacobi(_ConicSeededNewtonSurface):
    """Conic base plus an axisymmetric Jacobi-radial sag in the Jacobi
    argument u = 2 (r / R_n)^2 - 1.

    With alpha = beta = -1/2 this reduces to the Chebyshev-T radial
    basis; with alpha = beta = 0 to Legendre.
    """


class Toroid(_ConicSeededNewtonSurface):
    """Zemax-style toroidal surface: cylindrical sphere in X plus an even
    asphere in Y.  Newton-Raphson intersection seeded by a sphere with
    curvature (c_x + c_y) / 2.

    """

    def _seed_conic(self):
        p = self.params
        c_seed = 0.5 * (p['c_x'] + p['c_y'])
        return c_seed, 0.0, 0.0, 0.0


class Biconic(_ConicSeededNewtonSurface):
    """Biconic surface: independent vertex curvatures and conic constants
    on the X and Y axes.

    Newton-Raphson intersection seeded by a sphere with the average
    curvature and conic constant.

    """

    def _seed_conic(self):
        p = self.params
        c_seed = 0.5 * (p['c_x'] + p['c_y'])
        k_seed = 0.5 * (p['k_x'] + p['k_y'])
        return c_seed, k_seed, 0.0, 0.0
