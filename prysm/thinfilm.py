"""Tools for performing thin film calculations."""
from functools import reduce

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


def characteristic_matrix_p(lambda_, d, n, theta):
    """Compute the characteristic matrix M_j^p for a layer of a material stack and p-polarized light.

    Uses (4.49) to compute (4.55) from BYU optics book, edition 2015

    Parameters
    ----------
    lambda_ : `float`
        wavelength of light, microns
    d : `float`
        thickness of the layer, microns
    n : `float` or `complex`
        refractive index of the layer
    theta : `float`
        angle of incidence, degrees

    Returns
    -------
    `numpy.array`
        a 2x2 matrix

    """
    theta = e.radians(theta)
    k = (2 * e.pi) / lambda_
    cost = e.cos(theta)
    beta = k * d * cost
    sinb, cosb = e.sin(beta), e.cos(beta)

    upper_right = -1j * sinb * cost / n
    lower_left = -1j * n * sinb / cost
    return e.array([
        [cosb, upper_right],
        [lower_left, cosb]
    ])


def characteristic_matrix_s(lambda_, d, n, theta):
    """Compute the characteristic matrix M_j^p for a layer of a material stack and s-polarized light.

    Uses (4.49) to compute (4.55) from BYU optics book, edition 2015

    Parameters
    ----------
    lambda_ : `float`
        wavelength of light, microns
    d : `float`
        thickness of the layer, microns
    n : `float` or `complex`
        refractive index of the layer
    theta : `float`
        angle of incidence, degrees

    Returns
    -------
    `numpy.array`
        a 2x2 matrix

    """
    theta = e.radians(theta)
    k = (2 * e.pi) / lambda_
    cost = e.cos(theta)
    beta = k * d * cost
    sinb, cosb = e.sin(beta), e.cos(beta)

    upper_right = -1j * sinb / (cost * n)
    lower_left = -1j * n * sinb * cost
    return e.array([
        [cosb, upper_right],
        [lower_left, cosb]
    ])


def multilayer_matrix_p(n0, theta0, characteristic_matricies, nnp1, theta_np1):
    """Reduce a multilayer problem to give the 2x2 matric A^p.

    Computes (4.58) from BYU optics book.

    Parameters
    ----------
    n0 : `float` or `complex`
        refractive index of the first medium
    theta0 : `float`
        angle of incidence on the first medium, degrees
    characteristic_matricies : `iterable` of `numpy.ndarray` each of which of shape 2x2
        the characteristic matricies of each layer
    nnp1 : `float` or `complex`
        refractive index of the final medium
    theta_np1 : `float`
        angle of incidence on final medium, degrees

    Returns
    -------
    `numpy.ndarray`
        2x2 matrix A^s

    """
    theta0 = e.radians(theta0)
    cost0 = e.cos(theta0)
    term1 = 1 / (2 * n0 * cost0)

    theta_np1 = e.radians(theta_np1)

    term2 = e.array([
        [n0, cost0],
        [n0, -cost0]
    ])
    term3 = reduce(e.multiply, characteristic_matricies)  # reduce does M1 * M2 * M3 [...]
    term4 = e.array([
        [e.cos(theta_np1), 0],
        [nnp1, 0]
    ])
    return term1 * term2 * term3 * term4


def multilayer_matrix_s(n0, theta0, characteristic_matricies, nnp1, theta_np1):
    """Reduce a multilayer problem to give the 2x2 matric A^s.

    Computes (4.62) from BYU optics book.

    Parameters
    ----------
    n0 : `float` or `complex`
        refractive index of the first medium
    theta0 : `float`
        angle of incidence on the first medium, degrees
    characteristic_matricies : `iterable` of `numpy.ndarray` each of which of shape 2x2
        the characteristic matricies of each layer
    nnp1 : `float` or `complex`
        refractive index of the final medium
    theta_np1 : `float`
        angle of incidence on final medium, degrees

    Returns
    -------
    `numpy.ndarray`
        2x2 matrix A^s

    """
    theta0 = e.radians(theta0)
    cost0 = e.cos(theta0)
    term1 = 1 / (2 * n0 * cost0)
    n0cost0 = n0 * cost0

    theta_np1 = e.radians(theta_np1)

    term2 = e.array([
        [n0cost0, 1],
        [n0cost0, -1]
    ])
    term3 = reduce(e.multiply, characteristic_matricies)  # reduce does M1 * M2 * M3 [...]
    term4 = e.array([
        [1, 0],
        [nnp1 * e.cos(theta_np1), 0]
    ])
    return term1 * term2 * term3 * term4


def rtot(Amat):
    """Compute rtot, the equivalent total fresnel coefficient for a multilayer stack.

    Parameters
    ----------
    Amat : `numpy.ndarray`
        2x2 array

    Returns
    -------
    `float` or `complex`
        the value of rtot, either s or p.

    """
    return Amat[1, 0] / Amat[0, 0]


def ttot(Amat):
    """Compute ttot, the equivalent total fresnel coefficient for a multilayer stack.

    Parameters
    ----------
    Amat : `numpy.ndarray`
        2x2 array

    Returns
    -------
    `float` or `complex`
        the value of rtot, either s or p.

    """
    return 1 / Amat[0, 0]


# where am
# matrix s typed out, unsure if impl is correct
# or if this is the best interface
# matrix p is pasted from s but not converted to p
# need to reduce to effective r, t coefs but
# need to carefully sketch out the problem to get directions, etc, correct
