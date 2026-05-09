"""Surface types and calculus."""

from prysm.mathops import np
from prysm.conf import config
from prysm.coordinates import cart_to_polar, make_rotation_matrix
from prysm.polynomials.qpoly import compute_z_zprime_Q2d
from prysm.polynomials import hermite_He_seq, lstsq


def find_zero_indices_2d(x, y, tol=1e-8):
    """Find the (y,x) indices into x and y where x==y==0.

    Reserved for the SurfaceQ2D path (no live caller inside this module yet).

    """
    # assuming we're FFT-centered, we will never do the ifs
    # this probably blows up if zero is not in the array
    lookup = tuple(s//2 for s in x.shape)
    x0 = x[lookup]
    if x0 > tol:
        lookup2 = (lookup[0], lookup[1]+1)
        x1 = x[lookup2]
        dx = x1-x0
        shift_samples = (x0 / dx)
        lookup = (lookup[0], lookup[1]+shift_samples)
    y0 = y[lookup]
    if y0 > tol:
        lookup2 = (lookup[0]+1, lookup[1])
        y1 = y[lookup2]
        dy = y1-y0
        shift_samples = (y0 / dy)
        lookup = (lookup[0]+shift_samples, lookup[1])

    return lookup


def fix_zero_singularity(arr, x, y, fill='xypoly', order=2):
    """Fix a singularity at the origin of arr by polynomial interpolation.

    Reserved for the SurfaceQ2D path (no live caller inside this module yet);
    the conic-family Surface subclasses use the singularity-free
    *_sag_der_xy helpers and do not need this.

    Parameters
    ----------
    arr : ndarray
        array of dimension 2 to modify at the origin (x==y==0)
    x : ndarray
        array of dimension 2 of X coordinates
    y : ndarray
        array of dimension 2 of Y coordinates
    fill : str, optional, {'xypoly'}
        how to fill.  Not used/hard-coded to X/Y polynomials, but made an arg
        today in case it may be added future for backwards compatibility
    order : int
        polynomial order to fit

    Returns
    -------
    ndarray
        arr (modified in-place)

    """
    zloc = find_zero_indices_2d(x, y)
    min_y = zloc[0]-order
    max_y = zloc[0]+order+1
    min_x = zloc[1]-order
    max_x = zloc[1]+order+1
    # newaxis schenanigans to get broadcasting right without
    # meshgrid
    ypts = np.arange(min_y, max_y)[:, np.newaxis]
    xpts = np.arange(min_x, max_x)[np.newaxis, :]
    window = arr[ypts, xpts].copy()
    c = [s//2 for s in window.shape]
    window[c] = np.nan
    # no longer need xpts, ypts
    # really don't care about fp64 vs fp32 (very small arrays)
    xpts = xpts.astype(float)
    ypts = ypts.astype(float)
    # use Hermite polynomials as
    # XY polynomial-like basis orthogonal
    # over the infinite plane
    # H0 = 1
    # H1 = x
    # H2 = x^2 - 1, and so on
    # H0(x) and H0(y) are both the constant 1, so only include
    # the m=0 term once (under x) to keep the basis full rank.
    xbasis = hermite_He_seq(np.arange(order+1), xpts)
    ybasis = hermite_He_seq(np.arange(1, order+1), ypts)
    # convert 1D modes to 2D for lstsq
    xbasis = [np.broadcast_to(mode, (ypts.size, xpts.size)) for mode in xbasis]
    ybasis = [np.broadcast_to(mode, (ypts.size, xpts.size)) for mode in ybasis]
    basis_set = np.asarray([*xbasis, *ybasis])
    coefs = lstsq(basis_set, window)
    projected = np.dot(basis_set[:, c[0], c[1]], coefs)
    arr[zloc] = projected
    return arr


def surface_normal_from_cylindrical_derivatives(fp, ft, r, t):
    """Use polar derivatives to compute Cartesian surface normals.

    Reserved for the SurfaceQ2D path (no live caller inside this module yet);
    surfaces with closed-form Cartesian derivatives should use the
    *_sag_der_xy helpers and skip this conversion entirely.

    Parameters
    ----------
    fp : ndarray
        derivative of f w.r.t. r
    ft : ndarray
        derivative of f w.r.t. t
    r : ndarray
        radial coordinates
    t : ndarray
        azimuthal coordinates

    Returns
    -------
    ndarray, ndarray
        x, y derivatives; will contain a singularity where r=0,
        see fix_zero_singularity

    """
    cost = np.cos(t)
    sint = np.sin(t)
    x = fp * cost - 1/r * ft * sint
    y = fp * sint + 1/r * ft * cost
    return x, y


def surface_normal_from_cartesian_derivatives(fx, fy, r, t):
    """Use Cartesian derivatives to compute polar surface normals.

    Reserved utility (no live caller inside this module).

    Parameters
    ----------
    fx : ndarray
        derivative of f w.r.t. x
    fy : ndarray
        derivative of f w.r.t. y
    r : ndarray
        radial coordinates
    t : ndarray
        azimuthal coordinates

    Returns
    -------
    ndarray, ndarray
        r, t derivatives; will contain a singularity where r=0,
        see fix_zero_singularity

    """
    cost = np.cos(t)
    sint = np.sin(t)
    fr = fx * cost + fy * sint
    ft = -fx * r * sint + fy * r * cost
    return fr, ft


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
        csq = c * c
        phi = np.sqrt(1 - (1+kappa) * csq * rhosq)

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
        csq = c ** 2
        rhosq = rho * rho
        phi = np.sqrt(1 - (1+kappa) * csq * rhosq)

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
        csq = c * c
        phi = np.sqrt(1 - (1 + kappa) * csq * (x * x + y * y))
    return (c * x) / phi, (c * y) / phi


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
        phi = np.sqrt(1 - (1 + kappa) * c * c * rsq)
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
        csq = c * c
        phi = np.sqrt(1 - (1 + kappa) * csq * (X * X + Y * Y))
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
    cost = np.cos(t)
    sint = np.sin(t)
    oblique_term = 2 * r * (dx * cost + dy * sint)
    ssq = dx * dx + dy * dy
    aggregate_term = r * r + oblique_term + ssq
    num = c * aggregate_term
    csq = c * c
    den = 1 + np.sqrt(1 - (1 + kappa) * csq * aggregate_term)
    return num / den


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
    cost = np.cos(t)
    sint = np.sin(t)
    oblique_term = 2 * r * (dx * cost + dy * sint)
    ddr_oblique = 2 * r + 2 * (dx * cost + dy * sint)
    ddt_oblique_ = r * (-dx * sint + dy * cost)
    ddt_oblique = 2 * ddt_oblique_
    ssq = dx * dx + dy * dy

    aggregate_term = r * r + oblique_term + ssq
    csq = c * c
    c3 = csq * c
    # d/dr first
    num = c * ddr_oblique
    phi_kernel = (1 + kappa) * csq * aggregate_term
    phi = np.sqrt(1 - phi_kernel)
    phip1 = 1 + phi
    phip1sq = phip1 * phip1
    den = phip1
    term1 = num / den

    num = c3 * (1+kappa)*ddr_oblique * aggregate_term
    den = (2 * phi) * phip1sq
    term2 = num / den
    dr = term1 + term2

    # d/dt
    num = c * ddt_oblique
    den = phip1
    term1 = num / den

    num = c3 * (1+kappa) * ddt_oblique_ * aggregate_term
    den = phi * phip1sq
    term2 = num / den
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
    cost = np.cos(t)
    sint = np.sin(t)
    oblique_term = 2 * r * (dx * cost + dy * sint)
    ssq = dx * dx + dy * dy
    aggregate_term = r * r + oblique_term + ssq
    csq = c * c
    num = np.sqrt(1 - (1+kappa) * csq * aggregate_term)
    den = np.sqrt(1 - kappa * csq * aggregate_term)  # flipped sign, 1-kappa
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
    cost = np.cos(t)
    sint = np.sin(t)
    oblique_term = 2 * r * (dx * cost + dy * sint)
    ddr_oblique = 2 * r + 2 * (dx * cost + dy * sint)
    ddt_oblique_ = r * (-dx * sint + dy * cost)
    ssq = dx * dx + dy * dy

    aggregate_term = r * r + oblique_term + ssq
    csq = c * c
    # 1/sigma = sqrt(1 - kappa c^2 A) / sqrt(1 - (1+kappa) c^2 A) = u/phi
    # let phi = sqrt(1 - (1+kappa) c^2 A), notquitephi (= u) = sqrt(1 - kappa c^2 A)
    # d/dx (u/phi) = u'/phi + u * d/dx (1/phi)
    #              = u'/phi - u*phi'/phi^2
    # phi' = -(1+kappa) c^2 A' / (2 phi)  =>  -u*phi'/phi^2 = u (1+kappa) c^2 A' / (2 phi^3)
    # u'   = -kappa c^2 A' / (2 u)        =>  u'/phi = -kappa c^2 A' / (2 u phi)
    phi_kernel = (1 + kappa) * csq * aggregate_term
    phi = np.sqrt(1 - phi_kernel)
    notquitephi = np.sqrt(1 - kappa * csq * aggregate_term)

    # d/dr
    num = csq * (1 + kappa) * ddr_oblique * notquitephi
    den = 2 * (1 - phi_kernel) ** (3/2)
    term1 = num / den

    num = -csq * kappa * ddr_oblique
    den = 2 * phi * notquitephi
    term2 = num / den
    dr = term1 + term2

    # d/dt — same structure but with d/dt of aggregate_term (= 2 * ddt_oblique_)
    num = csq * (1 + kappa) * ddt_oblique_ * notquitephi
    den = (1 - phi_kernel) ** (3/2)
    term1 = num / den

    num = -csq * kappa * ddt_oblique_
    den = phi * notquitephi
    term2 = num / den
    dt = term1 + term2
    return dr, dt


def Q2d_and_der(cm0, ams, bms, x, y, normalization_radius, c, k, dx=0, dy=0):
    """Q-type freeform surface, with base (perhaps shifted) conicoic.

    Reserved for the SurfaceQ2D path (no live caller inside this module yet).
    When wired up, the consumer will pair this with
    surface_normal_from_cylindrical_derivatives and fix_zero_singularity to
    obtain Cartesian normals from the polar derivatives this returns.

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
    P, S = np.atleast_2d(P, S)
    Sx = S[..., 0]
    Sy = S[..., 1]
    Sz = S[..., 2]
    # invalid arithmetic (1/0, sqrt(-x)) is expected for rays that miss the
    # surface; we expose those via the valid mask, not via FP warnings.
    with np.errstate(divide='ignore', invalid='ignore'):
        # project P onto the vertex tangent plane (matches Newton path convention)
        s0 = -P[..., 2] / Sz
        P1 = P + s0[..., np.newaxis] * S
        Xp = P1[..., 0] + dx
        Yp = P1[..., 1] + dy
        one_plus_k = 1.0 + kappa
        A_ = 1.0 + kappa * Sz * Sz
        B_ = Xp * Sx + Yp * Sy - Sz / c
        C_ = Xp * Xp + Yp * Yp
        disc = B_ * B_ - A_ * C_
        disc_nonneg = (disc >= 0)
        disc = np.where(disc_nonneg, disc, np.zeros_like(disc))
        sqrt_disc = np.sqrt(disc)

        # quadratic root, picking the one with smaller |t| (vertex-side intersection)
        a_is_zero = (A_ == 0)
        safe_A = np.where(a_is_zero, 1.0, A_)
        t1 = (-B_ - sqrt_disc) / safe_A
        t2 = (-B_ + sqrt_disc) / safe_A
        # paraboloid + axial-ray fallback (linear equation 2 B_ t + C_ = 0)
        safe_B = np.where(B_ == 0, 1.0, B_)
        t_lin = -C_ / (2.0 * safe_B)
        t1 = np.where(a_is_zero, t_lin, t1)
        t2 = np.where(a_is_zero, t_lin, t2)
        t = np.where(np.abs(t1) <= np.abs(t2), t1, t2)

        Q = P1 + t[..., np.newaxis] * S
        Xq = Q[..., 0] + dx
        Yq = Q[..., 1] + dy
        phi_arg = 1.0 - one_plus_k * c * c * (Xq * Xq + Yq * Yq)
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
    if not hasattr(P, '__iter__'):
        P = np.array([0, 0, P], dtype=config.precision)
    else:
        # iterable
        P2 = np.zeros(3, dtype=config.precision)
        P2[-len(P):] = P
        P = P2

    return np.asarray(P).astype(config.precision)


def _none_or_rotmat(R):
    if R is None:
        return None
    if type(R) in (list, tuple):
        R = make_rotation_matrix(R)

    return R


def _apply_tilt_decenter(P, R, tilt, decenter, tilt_radians):
    """Combine a base (P, R) with a (tilt, decenter) perturbation.

    ``decenter`` is added to the surface position vector ``P``.  ``tilt`` is
    converted to a rotation matrix via ``make_rotation_matrix`` and composed
    with ``R`` as ``R_total = R @ R_tilt`` — i.e. the perturbation acts in
    the surface's *nominal local frame* (apply tilt first, then any base
    rotation).  For a surface with no nominal R this collapses to
    ``R_total = R_tilt``.

    Either or both perturbations may be None.

    """
    if decenter is not None:
        decenter = np.asarray(decenter, dtype=config.precision)
        if decenter.shape != (3,):
            raise ValueError(
                f'decenter must be a length-3 vector, got shape {decenter.shape}'
            )
        P = P + decenter
    if tilt is not None:
        R_tilt = make_rotation_matrix(tilt, radians=tilt_radians)
        if R is None:
            R = R_tilt
        else:
            R = R @ R_tilt
    return P, R


def _map_stype(typ):
    if isinstance(typ, int):
        return typ

    typ = typ.lower()
    if typ in ('refl', 'reflect'):
        return STYPE_REFLECT

    if typ in ('refr', 'refract'):
        return STYPE_REFRACT

    if typ == 'eval':
        return STYPE_EVAL


def _validate_n_and_typ(n, typ):
    if typ == STYPE_REFRACT and n is None:
        raise ValueError('refractive surfaces must have a refractive index function, not None')
    return


STYPE_REFLECT = -1
STYPE_REFRACT = -2
STYPE_EVAL =    -3  # NOQA


class Surface:
    """A surface for raytracing."""
    def __init__(self, typ, P, n, FFp, R=None, params=None, bounding=None,
                 aperture=None, tilt=None, decenter=None, tilt_radians=False):
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
        FFp : callable of signature F(x,y) -> z, [Nx, Ny]
            a function which returns the surface sag at point x, y as well as
            the X and Y partial derivatives at that point
        R : ndarray
            rotation matrix, may be None
        params : dict, optional
            surface type specific parameters
        bounding : dict, optional
            bounding geometry description
            at the moment, outer_radius and inner_radius are the only values
            which are used for anything.  More will be added in the future
        aperture : callable, optional
            a function ``(x, y) -> bool`` returning True for points inside
            the surface's aperture and False outside.  ``raytrace`` calls
            this on each ray's intersection point; rays falling outside are
            flagged in the trace's status array as clipped and skipped on
            all subsequent surfaces.  If None (the default) every ray is
            considered to be inside the surface.
        tilt : iterable of length 3, optional
            ``(rz, ry, rx)`` rotation angles describing a tilt perturbation
            of the surface in its own local frame.  Composed with ``R`` as
            ``R_total = R @ R_tilt`` so the tilt acts before any nominal
            rotation; with ``R=None`` the tilt becomes the surface's full
            rotation.  Angles in degrees by default; pass
            ``tilt_radians=True`` to switch.
        decenter : iterable of length 3, optional
            ``(dx, dy, dz)`` translation perturbation, added directly to
            ``P``.  Convenient for sensitivity analyses (see
            ``prysm.x.raytracing.sensitivity``) without having to build a
            new surface object.
        tilt_radians : bool, optional
            if True, ``tilt`` is interpreted in radians; default degrees.

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
        self.R = R
        self.params = params
        self.bounding = bounding
        self.aperture = aperture

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

    def intersect(self, P, S, eps=None, maxiter=None, return_valid=False):
        """Intersect rays P + t*S with this surface, returning (Q, normal).

        Default implementation uses Newton-Raphson iteration on the FFp
        callback.  Subclasses may override with closed-form intersections.

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
        # lazy import to avoid circular dependency with spencer_and_murty
        from .spencer_and_murty import (
            intersect as _newton_intersect,
            SURFACE_INTERSECTION_DEFAULT_MAXITER,
        )
        if maxiter is None:
            maxiter = SURFACE_INTERSECTION_DEFAULT_MAXITER
        return _newton_intersect(P, S, self.sag_normal, eps=eps, maxiter=maxiter,
                                 return_valid=return_valid)

    @classmethod
    def conic(cls, c, k, typ, P, n=None, R=None, bounding=None, aperture=None,
              tilt=None, decenter=None, tilt_radians=False):
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
            c_, k_ = params['c'], params['k']
            rsq = x * x + y * y
            csq = c_ * c_
            phi = np.sqrt(1 - (1 + k_) * csq * rsq)
            z = conic_sag(c_, k_, rsq, phi=phi)
            dx, dy = conic_sag_der_xy(c_, k_, x, y, phi=phi)
            return z, dx, dy

        return Conic(typ=typ, P=P, n=n, FFp=FFp, R=R, params=params,
                     bounding=bounding, aperture=aperture,
                     tilt=tilt, decenter=decenter, tilt_radians=tilt_radians)

    @classmethod
    def off_axis_conic(cls, c, k, typ, P, dx=0, dy=0, n=None, R=None,
                       bounding=None, aperture=None,
                       tilt=None, decenter=None, tilt_radians=False):
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
            dx_, dy_ = params['dx'], params['dy']
            X = x + dx_
            Y = y + dy_
            aggregate = X * X + Y * Y
            csq = c_ * c_
            phi = np.sqrt(1 - (1 + k_) * csq * aggregate)
            z = (c_ * aggregate) / (1 + phi)
            ddx = (c_ * X) / phi
            ddy = (c_ * Y) / phi
            return z, ddx, ddy

        return OffAxisConic(typ=typ, P=P, n=n, FFp=FFp, R=R, params=params,
                            bounding=bounding, aperture=aperture,
                            tilt=tilt, decenter=decenter,
                            tilt_radians=tilt_radians)

    @classmethod
    def plane(cls, typ, P, n=None, R=None, bounding=None, aperture=None,
              tilt=None, decenter=None, tilt_radians=False):
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

        return Plane(typ=typ, P=P, n=n, FFp=FFp, R=R, bounding=bounding,
                     aperture=aperture,
                     tilt=tilt, decenter=decenter, tilt_radians=tilt_radians)

    @classmethod
    def sphere(cls, c, typ, P, n, R=None, bounding=None, aperture=None,
               tilt=None, decenter=None, tilt_radians=False):
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
            phi = np.sqrt(1 - (c_ * c_) * rsq)
            z = sphere_sag(c_, rsq, phi=phi)
            dx = (c_ * x) / phi
            dy = (c_ * y) / phi
            return z, dx, dy

        return Sphere(typ=typ, P=P, n=n, FFp=FFp, R=R, params=params,
                      bounding=bounding, aperture=aperture,
                      tilt=tilt, decenter=decenter, tilt_radians=tilt_radians)

    @classmethod
    def even_asphere(cls, c, k, coefs, typ, P, n=None, R=None,
                     bounding=None, aperture=None,
                     tilt=None, decenter=None, tilt_radians=False):
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
            phi = np.sqrt(1 - (1 + k_) * c_ * c_ * rsq)
            z = even_asphere_sag(c_, k_, coefs_, rsq)
            dx, dy = even_asphere_sag_der_xy(c_, k_, coefs_, x, y, phi=phi)
            return z, dx, dy

        return EvenAsphere(typ=typ, P=P, n=n, FFp=FFp, R=R, params=params,
                           bounding=bounding, aperture=aperture,
                           tilt=tilt, decenter=decenter,
                           tilt_radians=tilt_radians)

    @classmethod
    def q2d(cls, c, k, normalization_radius, cm0, ams, bms, typ, P,
            dx=0, dy=0, n=None, R=None, bounding=None, aperture=None,
            tilt=None, decenter=None, tilt_radians=False):
        """A Forbes Q-2D freeform surface.

        Conic base plus the Q-2D polynomial expansion (Forbes, oe-20-3-2483).

        Parameters
        ----------
        c, k : float
            base conic curvature and conic constant
        normalization_radius : float
            radius by which to normalize rho (the Q polynomials are defined
            on a unit disk; ``rho/normalization_radius`` should lie in [0, 1]
            inside the aperture).
        cm0 : iterable of float
            coefficients for the m=0 Q radial expansion (axisymmetric piece)
        ams, bms : iterable of iterables of float
            ``ams[m-1]`` is the coefficient list for ``cos(m theta)`` Q radial
            expansion; same for ``bms`` and ``sin(m theta)``.  ``ams`` and
            ``bms`` must have the same length but may have different per-m
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

        return SurfaceQ2D(typ=typ, P=P, n=n, FFp=FFp, R=R, params=params,
                          bounding=bounding, aperture=aperture,
                          tilt=tilt, decenter=decenter,
                          tilt_radians=tilt_radians)


class Plane(Surface):
    """A plane normal to its local Z axis, with closed-form ray intersection."""

    def intersect(self, P, S, eps=None, maxiter=None, return_valid=False):
        return ray_plane_intersect(P, S, return_valid=return_valid)


class Sphere(Surface):
    """A spherical surface with closed-form ray-sphere intersection."""

    def intersect(self, P, S, eps=None, maxiter=None, return_valid=False):
        return ray_sphere_intersect(P, S, self.params['c'], return_valid=return_valid)


class Conic(Surface):
    """An on-axis conic surface with closed-form ray-conicoid intersection."""

    def intersect(self, P, S, eps=None, maxiter=None, return_valid=False):
        p = self.params
        return ray_conic_intersect(P, S, p['c'], p['k'], return_valid=return_valid)


class OffAxisConic(Surface):
    """An off-axis section of a conicoid with closed-form ray intersection."""

    def intersect(self, P, S, eps=None, maxiter=None, return_valid=False):
        p = self.params
        return ray_conic_intersect(P, S, p['c'], p['k'], dx=p['dx'], dy=p['dy'],
                                   return_valid=return_valid)


class EvenAsphere(Surface):
    """Even asphere: conic base plus polynomial in r^2.

    No closed-form ray intersection exists in general, but the asphere is
    typically a small perturbation of the base conic, so we seed Newton-
    Raphson with the closed-form conic intersection.  Convergence is
    typically reached in 1-2 iterations versus many starting from t=0.

    """

    def intersect(self, P, S, eps=None, maxiter=None, return_valid=False):
        # lazy import to avoid the spencer_and_murty <-> surfaces import cycle
        from .spencer_and_murty import (
            newton_raphson_solve_s,
            SURFACE_INTERSECTION_DEFAULT_MAXITER,
        )
        if maxiter is None:
            maxiter = SURFACE_INTERSECTION_DEFAULT_MAXITER
        P, S = np.atleast_2d(P, S)
        Sz = S[..., 2]
        # project onto vertex tangent plane (Newton's reference frame)
        s0 = -P[..., 2] / Sz
        P1 = P + s0[..., np.newaxis] * S
        # closed-form conic intersection in the tangent-plane frame; since
        # P1.z == 0, the segment length from P1 to Q_conic along S equals
        # Q_conic.z / Sz.
        Q_conic, _ = ray_conic_intersect(P1, S, self.params['c'], self.params['k'])
        s1 = Q_conic[..., 2] / Sz
        return newton_raphson_solve_s(P1, S, self.sag_normal,
                                      s1=s1, eps=eps, maxiter=maxiter,
                                      return_valid=return_valid)


class SurfaceQ2D(Surface):
    """Forbes Q-2D freeform surface: off-axis conic base plus a Q-polynomial
    expansion in normalized polar coordinates.

    Like ``EvenAsphere``, the implicit surface equation is non-rational in
    the ray parameter so there is no closed-form intersection.  We seed
    Newton-Raphson with the closed-form conic intersection (which is the
    base conic of the surface, dx/dy-shifted as needed); the Q expansion
    is typically a microns-scale perturbation, so Newton converges in a
    handful of iterations.

    """

    def intersect(self, P, S, eps=None, maxiter=None, return_valid=False):
        from .spencer_and_murty import (
            newton_raphson_solve_s,
            SURFACE_INTERSECTION_DEFAULT_MAXITER,
        )
        if maxiter is None:
            maxiter = SURFACE_INTERSECTION_DEFAULT_MAXITER
        P, S = np.atleast_2d(P, S)
        Sz = S[..., 2]
        # project onto the vertex tangent plane (Newton's reference frame)
        s0 = -P[..., 2] / Sz
        P1 = P + s0[..., np.newaxis] * S
        p = self.params
        Q_conic, _ = ray_conic_intersect(P1, S, p['c'], p['k'],
                                         dx=p.get('dx', 0.0),
                                         dy=p.get('dy', 0.0))
        s1 = Q_conic[..., 2] / Sz
        return newton_raphson_solve_s(P1, S, self.sag_normal,
                                      s1=s1, eps=eps, maxiter=maxiter,
                                      return_valid=return_valid)
