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
    r /= normalization_radius
    z, zprimer, zprimet = compute_z_zprime_Q2d(cm0, ams, bms, r, t)

    if dx != 0:
        x = x + dx
    if dy != 0:
        y = y + dy

    # no matter what need to do this again because of normalization radius
    r, t = cart_to_polar(x, y)

    rsq = r * r
    phi = phi_spheroid(c, k, rsq)
    base_sag = conic_sag(c, k, rsq, phi=phi)
    base_sag_der = conic_sag_der(c, k, r, phi=phi)

    q_prefix = 1 / phi
    q_prefix_der = der_direction_cosine_spheroid(c, k, r, rhosq=rsq, phi=phi)

    # u = 1/phi
    # du = d/dr(1/phi)
    # v = q
    # dv = (q der)

    zprimer /= normalization_radius
    zprimer2 = product_rule(q_prefix, z, q_prefix_der, zprimer)
    # don't need to adjust azimuthal derivative
    z *= q_prefix
    zprimet *= q_prefix
    z += base_sag
    zprimer2 += base_sag_der
    return z, zprimer2, zprimet
