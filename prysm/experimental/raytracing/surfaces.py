"""Spherical surfaces."""

from prysm.mathops import np
from prysm.coordinates import cart_to_polar
from prysm.polynomials.qpoly import compute_z_zprime_Q2d


def product_rule(u, v, du, dv):
    """The product rule of calculus, d/dx uv = u dv v du."""
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
    rhosq : numpy.ndarray
        squared radial coordinate (non-normalized)

    Returns
    -------
    numpy.ndarray
        phi term

    """
    csq = c * c
    return np.sqrt(1 - (1 - k) * csq * rhosq)


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
    rho : numpy.ndarray
        radial coordinate (non-normalized)
    rhosq : numpy.ndarray
        squared radial coordinate (non-normalized)
        rho ** 2 if None

    Returns
    -------
    numpy.ndarray
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
    rhosq : numpy.ndarray
        radial coordinate squared
        e.g. for a 15 mm half-diameter optic,
        rho = 0 .. 15
        rhosq = 0 .. 225
        there is no requirement on rectilinear sampling or array
        dimensionality
    phi : numpy.ndarray, optional
        (1 - c^2 r^2)^.5
        computed if not provided
        many surface types utilize phi; its computation can be
        de-duplicated by passing the optional argument

    Returns
    -------
    numpy.ndarray
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
    rho : numpy.ndarray
        radial coordinate
        e.g. for a 15 mm half-diameter optic,
        rho = 0 .. 15
        there is no requirement on rectilinear sampling or array
        dimensionality
    phi : numpy.ndarray, optional
        (1 - c^2 r^2)^.5
        computed if not provided
        many surface types utilize phi; its computation can be
        de-duplicated by passing the optional argument

    Returns
    -------
    numpy.ndarray
        derivative of surface sag

    """
    if phi is None:
        csq = c ** 2
        rhosq = rho * rho
        phi = np.sqrt(1 - csq * rhosq)
    return (c * rho) / phi


def conic_sag(c, kappa, rhosq, phi=None):
    """Sag of a spherical surface.

    Parameters
    ----------
    c : float
        surface curvature
    kappa : float
        conic constant
    rhosq : numpy.ndarray
        radial coordinate squared
        e.g. for a 15 mm half-diameter optic,
        rho = 0 .. 15
        rhosq = 0 .. 225
        there is no requirement on rectilinear sampling or array
        dimensionality
    phi : numpy.ndarray, optional
        (1 - (1+kappa) c^2 r^2)^.5
        computed if not provided
        many surface types utilize phi; its computation can be
        de-duplicated by passing the optional argument

    Returns
    -------
    numpy.ndarray
        surface sag

    """
    if phi is None:
        csq = c * c
        phi = np.sqrt(1 - (1-kappa) * csq * rhosq)

    return (c * rhosq) / (1 + phi)


def conic_sag_der(c, kappa, rho, phi=None):
    """Sag of a spherical surface.

    Parameters
    ----------
    c : float
        surface curvature
    kappa : float
        conic constant
    rho : numpy.ndarray
        radial coordinate
        e.g. for a 15 mm half-diameter optic,
        rho = 0 .. 15
        there is no requirement on rectilinear sampling or array
        dimensionality
    phi : numpy.ndarray, optional
        (1 - (1+kappa) c^2 r^2)^.5
        computed if not provided
        many surface types utilize phi; its computation can be
        de-duplicated by passing the optional argument

    Returns
    -------
    numpy.ndarray
        surface sag

    """
    if phi is None:
        csq = c ** 2
        rhosq = rho * rho
        phi = np.sqrt(1 - (1-kappa) * csq * rhosq)

    return (c * rho) / phi


def off_axis_conic_sag(c, kappa, r, t, dx, dy=0):
    """Sag of an off-axis conicoid

    Parameters
    ----------
    c : float
        axial curvature of the conic
    kappa : float
        conic constant
    r : numpy.ndarray
        radial coordinate, where r=0 is centered on the off-axis section
    t : numpy.ndarray
        azimuthal coordinate
    dx : float
        shift of the surface in x with respect to the base conic vertex,
        mutually exclusive to dy (only one may be nonzero)
        use dx=0 when dy != 0
    dy : float
        shift of the surface in y with respect to the base conic vertex

    Returns
    -------
    numpy.ndarray
        surface sag, z(x,y)

    """
    if dy != 0 and dx != 0:
        raise ValueError('only one of dx/dy may be nonzero')

    if dx != 0:
        s = dx
        oblique_term = 2 * s * r * np.cos(t)
    else:
        s = dy
        oblique_term = 2 * s * r * np.sin(t)

    aggregate_term = r * r + oblique_term + s * s
    num = c * aggregate_term
    csq = c * c
    # typo in paper; 1+k => 1-k
    den = 1 + np.sqrt(1 - (1 - kappa) * csq * aggregate_term)
    return num / den


def off_axis_conic_der(c, kappa, r, t, dx, dy=0):
    """Radial and azimuthal derivatives of an off-axis conic.

    Parameters
    ----------
    c : float
        axial curvature of the conic
    kappa : float
        conic constant
    r : numpy.ndarray
        radial coordinate, where r=0 is centered on the off-axis section
    t : numpy.ndarray
        azimuthal coordinate
    dx : float
        shift of the surface in x with respect to the base conic vertex,
        mutually exclusive to dy (only one may be nonzero)
        use dx=0 when dy != 0
    dy : float
        shift of the surface in y with respect to the base conic vertex

    Returns
    -------
    numpy.ndarray, numpy.ndarray
        d/dr(z), d/dt(z)

    """
    if dy != 0 and dx != 0:
        raise ValueError('only one of dx/dy may be nonzero')

    cost = np.cos(t)
    sint = np.sin(t)
    if dx != 0:
        s = dx
        oblique_term = 2 * s * r * cost
        ddr_oblique = 2 * r + 2 * s * cost
        # I accept the evil in writing this the way I have
        # to deduplicate the computation
        ddt_oblique_ = r*(-s)*sint
        ddt_oblique = 2 * ddt_oblique_
    else:
        s = dy
        oblique_term = 2 * s * r * sint
        ddr_oblique = 2 * r + 2 * s * sint
        ddt_oblique_ = r*s*cost
        ddt_oblique = 2 * ddt_oblique_

    aggregate_term = r * r + oblique_term + s * s
    csq = c * c
    c3 = csq * c
    # d/dr first
    num = c * ddr_oblique
    phi_kernel = (1 - kappa) * csq * aggregate_term
    phi = np.sqrt(1 - phi_kernel)
    phip1 = 1 + phi
    phip1sq = phip1 * phip1
    den = phip1
    term1 = num / den

    num = c3 * (1-kappa)*ddr_oblique * aggregate_term
    den = (2 * phi) * phip1sq
    term2 = num / den
    dr = term1 + term2

    # d/dt
    num = c * ddt_oblique
    den = phip1
    term1 = num / den

    num = c3 * (1-kappa) * ddt_oblique_ * aggregate_term
    den = phi * phip1sq
    term2 = num / den
    dt = term1 + term2

    return dr, dt


def off_axis_conic_sigma(c, kappa, r, t, dx, dy=0):
    """sigma (direction cosine projection term) for an off-axis conic.

    See Eq. (5.2) of oe-20-3-2483.

    Parameters
    ----------
    c : float
        axial curvature of the conic
    kappa : float
        conic constant
    r : numpy.ndarray
        radial coordinate, where r=0 is centered on the off-axis section
    t : numpy.ndarray
        azimuthal coordinate
    dx : float
        shift of the surface in x with respect to the base conic vertex,
        mutually exclusive to dy (only one may be nonzero)
        use dx=0 when dy != 0
    dy : float
        shift of the surface in y with respect to the base conic vertex

    Returns
    -------
    sigma(r,t)

    """
    if dy != 0 and dx != 0:
        raise ValueError('only one of dx/dy may be nonzero')

    if dx != 0:
        s = dx
        oblique_term = 2 * s * r * np.cos(t)
    else:
        s = dy
        oblique_term = 2 * s * r * np.sin(t)

    aggregate_term = r * r + oblique_term + s * s
    csq = c * c
    num = np.sqrt(1 - (1-kappa) * csq * aggregate_term)
    den = np.sqrt(1 + kappa * csq * aggregate_term)  # flipped sign, 1-kappa
    return num / den


def off_axis_conic_sigma_der(c, kappa, r, t, dx, dy=0):
    """Lowercase sigma (direction cosine projection term) for an off-axis conic.

    See Eq. (5.2) of oe-20-3-2483.

    Parameters
    ----------
    c : float
        axial curvature of the conic
    kappa : float
        conic constant
    r : numpy.ndarray
        radial coordinate, where r=0 is centered on the off-axis section
    t : numpy.ndarray
        azimuthal coordinate
    dx : float
        shift of the surface in x with respect to the base conic vertex,
        mutually exclusive to dy (only one may be nonzero)
        use dx=0 when dy != 0
    dy : float
        shift of the surface in y with respect to the base conic vertex

    Returns
    -------
    numpy.ndarray, numpy.ndarray
        d/dr(z), d/dt(z)

    """
    if dy != 0 and dx != 0:
        raise ValueError('only one of dx/dy may be nonzero')

    cost = np.cos(t)
    sint = np.sin(t)
    if dx != 0:
        s = dx
        oblique_term = 2 * s * r * cost
        ddr_oblique = 2 * r + 2 * s * cost
        # I accept the evil in writing this the way I have
        # to deduplicate the computation
        ddt_oblique_ = r*(-s)*sint
        ddt_oblique = 2 * ddt_oblique_
    else:
        s = dy
        oblique_term = 2 * s * r * sint
        ddr_oblique = 2 * r + 2 * s * sint
        ddt_oblique_ = r*s*cost
        ddt_oblique = 2 * ddt_oblique_

    aggregate_term = r * r + oblique_term + s * s
    csq = c * c
    # d/dr first
    phi_kernel = (1 - kappa) * csq * aggregate_term
    phi = np.sqrt(1 - phi_kernel)
    notquitephi = np.sqrt(1 + kappa * csq * aggregate_term)
    num = csq * ddr_oblique * phi
    den = 2 * (1 - csq * kappa * aggregate_term) ** (3/2)
    term1 = num / den

    num = csq * (1 - kappa) * ddr_oblique
    den = 2 * phi * notquitephi  # slight difference in writing (2*phi*phi)
    term2 = num / den
    dr = term1 - term2

    # d/dt
    num = csq * (1 - kappa) * ddt_oblique_
    den = phi * notquitephi
    term1 = num/den

    num = csq * kappa * ddt_oblique_ * phi
    den = (csq * kappa * aggregate_term + 1) ** (3/2)
    term2 = num / den
    dt = term1 + term2  # minus in writing, but sine/cosine
    # dt *= kappa
    return dr, dt


def Q2d_and_der(cm0, ams, bms, x, y, normalization_radius, c, k, dx=0, dy=0):
    """Q-type freeform surface, with base (perhaps shifted) conicoic.

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
    x : numpy.ndarray
        X coordinates
    y : numpy.ndarray
        Y coordinates
    normalization_radius : float
        radius by which to normalize rho to produce u
    c : float
        curvature, reciprocal radius of curvature
    k : float
        kappa, conic constant
    rhosq : numpy.ndarray
        squared radial coordinate (non-normalized)
    dx : float
        shift of the base conic in x
    dy : float
        shift of the base conic in y

    Returns
    -------
    numpy.ndarray, numpy.ndarray, numpy.ndarray
        sag, dsag/drho, dsag/dtheta

    """
    # Q portion
    r, t = cart_to_polar(x, y)
    r2 = r / normalization_radius
    # content of curly braces in B.1 from oe-20-3-2483
    z, zprimer, zprimet = compute_z_zprime_Q2d(cm0, ams, bms, r2, t)

    base_sag = off_axis_conic_sag(c, k, r, t, dx, dy)
    base_primer, base_primet = off_axis_conic_der(c, k, r, t, dx, dy)

    # Eq. 5.1/5.2
    sigma = off_axis_conic_sigma(c, k, r, t, dx, dy)
    sigmaprimer, sigmaprimet = off_axis_conic_sigma_der(c, k, r, t, dx, dy)

    # u = 1/phi
    # du = d/dr(1/phi)
    # v = q
    # dv = (q der)

    # print('zt')
    # print(zprimet)
    zprimer /= normalization_radius
    zprimer2 = product_rule(sigma, z, sigmaprimer, zprimer)
    # zprimet2 = zprimet
    zprimet2 = product_rule(sigma, z, sigmaprimet, zprimet)
    # print('zt2')
    # print(zprimet2)
    # zprimet2 = zprimet
    # don't need to adjust azimuthal derivative
    z *= sigma
    # zprimet *= sigma
    z += base_sag
    zprimer2 += base_primer
    zprimet2 += base_primet
    # print('zt2+bt')
    # print(zprimet2)
    return z, zprimer2, zprimet2
