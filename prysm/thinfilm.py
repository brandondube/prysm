"""Tools for performing thin film calculations."""

from prysm.mathops import engine as e


def brewsters_angle(n0, n1, deg=True):
    """Compute the Brewster's angle at a given interface.

    Parameters
    ----------
    n0 : `float`
        refractive index on the "left" of the boundary
    n1 : `float`
        refractive index on the "right" of the boundary
    deg : `bool`, optional
        if True, convert output to degrees

    """
    ang = e.arctan2(n1, n0)
    if deg:
        return e.degres(ang)
    else:
        return ang


def critical_angle(n0, n1):
    """Minimum angle for total internal reflection at an interface.

    Parameters
    ----------
    n0 : `float`
        index of refraction of the "left" material
    n1 : `float`
        index of refraction of the "right" material

    Returns
    -------
    `float`
        the angle in degrees at which TIR begins to occur

    """
    return e.degrees(e.arcsin(n1/n0))


def snell_aor(n0, n1, theta):
    """Compute the angle of refraction using Snell's law.

    Parameters
    ----------
    n0 : `float`
        index of refraction of the "left" material
    n1 : `float`
        idnex of refraction of the "right" material
    theta : `float`
        angle of incidence in degrees

    Returns
    -------
    `float`
        angle of refraction

    """
    return e.arcsin(n0/n1 * e.sin(e.radians(theta)))


def fresnel_rs(n0, n1, theta0, theta1):
    """Compute the "r sub s" fresnel coefficient.

    This is associated with reflection of the s-polarized electric field.

    Parameters
    ----------
    n0 : `float`
        refractive index of the "left" material
    n1 : `float`
        refractive index of the "right" material
    theta0 : `float`
        angle of incidence, radians
    theta1 : `float`
        angle of reflection, radians

    Returns
    -------
    `float`
        the fresnel coefficient "r sub s"

    """
    num = n0 * e.cos(theta0) - n1 * e.cos(theta1)
    den = n1 * e.cos(theta0) + n1 * e.cos(theta1)
    return num / den


def fresnel_ts(n0, n1, theta0, theta1):
    """Compute the "t sub s" fresnel coefficient.

    This is associated with transmission of the s-polarized electric field.

    Parameters
    ----------
    n0 : `float`
        refractive index of the "left" material
    n1 : `float`
        refractive index of the "right" material
    theta0 : `float`
        angle of incidence, radians
    theta1 : `float`
        angle of refraction, radians

    Returns
    -------
    `float`
        the fresnel coefficient "t sub s"

    """
    num = 2 * n0 * e.cos(theta0)
    den = n0 * e.cos(theta0) + n1 * e.cos(theta1)
    return num / den


def fresnel_rp(n0, n1, theta0, theta1):
    """Compute the "r sub p" fresnel coefficient.

    This is associated with reflection of the p-polarized electric field.

    Parameters
    ----------
    n0 : `float`
        refractive index of the "left" material
    n1 : `float`
        refractive index of the "right" material
    theta0 : `float`
        angle of incidence, radians
    theta1 : `float`
        angle of reflection, radians

    Returns
    -------
    `float`
        the fresnel coefficient "r sub p"

    """
    num = n0 * e.cos(theta1) - n1 * e.cos(theta0)
    den = n0 * e.cos(theta1) + n1 * e.cos(theta0)
    return num / den


def fresnel_tp(n0, n1, theta0, theta1):
    """Compute the "t sub p" fresnel coefficient.

    This is associated with transmission of the p-polarized electric field.

    Parameters
    ----------
    n0 : `float`
        refractive index of the "left" material
    n1 : `float`
        refractive index of the "right" material
    theta0 : `float`
        angle of incidence, radians
    theta1 : `float`
        angle of refraction, radians

    Returns
    -------
    `float`
        the fresnel coefficient "t sub p"

    """
    num = 2 * n0 * e.cos(theta0)
    den = n0 * e.cos(theta1) + n1 * e.cos(theta0)
    return num / den
