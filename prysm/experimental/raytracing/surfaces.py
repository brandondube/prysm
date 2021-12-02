"""Spherical surfaces."""

from prysm.mathops import np


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
        csq = c ** 2
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
        csq = c ** 2
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
