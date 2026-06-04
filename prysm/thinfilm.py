"""Tools for performing thin film calculations."""
from .mathops import np


def brewsters_angle(n0, n1, deg=True):
    """Compute the Brewster's angle at a given interface.

    Parameters
    ----------
    n0 : float
        refractive index on the "left" of the boundary
    n1 : float
        refractive index on the "right" of the boundary
    deg : bool, optional
        if True, convert output to degrees

    """
    ang = np.arctan2(n1, n0)
    if deg:
        return np.degrees(ang)
    else:
        return ang


def critical_angle(n0, n1, deg=True):
    """Minimum angle for total internal reflection at an interface.

    Parameters
    ----------
    n0 : float
        index of refraction of the "left" material
    n1 : float
        index of refraction of the "right" material
    deg : bool, optional
        if true, returns degrees, else radians

    Returns
    -------
    float
        the angle in degrees at which TIR begins to occur

    """
    ang = np.arcsin(n0/n1)
    if deg:
        return np.degrees(ang)

    return ang


def snell_aor(n0, n1, theta, deg=True):
    """Compute the angle of refraction using Snell's law.

    Parameters
    ----------
    n0 : float
        index of refraction of the "left" material
    n1 : float
        index of refraction of the "right" material
    theta : float
        angle of incidence, in degrees if deg=True
    deg : bool, optional
        if True, theta is interpreted as an angle in degrees

    Returns
    -------
    float
        angle of refraction

    """
    if deg:
        theta = np.radians(theta)
    return np.lib.scimath.arcsin(n0/n1 * np.sin(theta))


def _cos_snell(n0, n1, theta):
    """Compute cos(theta_1) from Snell's law."""
    sint = n0/n1 * np.sin(theta)
    cost = np.lib.scimath.sqrt(1 - sint * sint)
    tir = (np.imag(sint) == 0) & (np.real(sint) > 1)
    return np.where(tir, -cost, cost)


def fresnel_rs(n0, n1, theta0, theta1):
    """Compute the "r sub s" fresnel coefficient.

    This is associated with reflection of the s-polarized electric field.

    Parameters
    ----------
    n0 : float
        refractive index of the "left" material
    n1 : float
        refractive index of the "right" material
    theta0 : float
        angle of incidence, radians
    theta1 : float
        angle of reflection, radians

    Returns
    -------
    float
        the fresnel coefficient "r sub s"

    """
    num = n0 * np.cos(theta0) - n1 * np.cos(theta1)
    den = n0 * np.cos(theta0) + n1 * np.cos(theta1)
    return num / den


def fresnel_ts(n0, n1, theta0, theta1):
    """Compute the "t sub s" fresnel coefficient.

    This is associated with transmission of the s-polarized electric field.

    Parameters
    ----------
    n0 : float
        refractive index of the "left" material
    n1 : float
        refractive index of the "right" material
    theta0 : float
        angle of incidence, radians
    theta1 : float
        angle of refraction, radians

    Returns
    -------
    float
        the fresnel coefficient "t sub s"

    """
    num = 2 * n0 * np.cos(theta0)
    den = n0 * np.cos(theta0) + n1 * np.cos(theta1)
    return num / den


def fresnel_rp(n0, n1, theta0, theta1):
    """Compute the "r sub p" fresnel coefficient.

    This is associated with reflection of the p-polarized electric field.

    Parameters
    ----------
    n0 : float
        refractive index of the "left" material
    n1 : float
        refractive index of the "right" material
    theta0 : float
        angle of incidence, radians
    theta1 : float
        angle of reflection, radians

    Returns
    -------
    float
        the fresnel coefficient "r sub p"

    """
    num = n0 * np.cos(theta1) - n1 * np.cos(theta0)
    den = n0 * np.cos(theta1) + n1 * np.cos(theta0)
    return num / den


def fresnel_tp(n0, n1, theta0, theta1):
    """Compute the "t sub p" fresnel coefficient.

    This is associated with transmission of the p-polarized electric field.

    Parameters
    ----------
    n0 : float
        refractive index of the "left" material
    n1 : float
        refractive index of the "right" material
    theta0 : float
        angle of incidence, radians
    theta1 : float
        angle of refraction, radians

    Returns
    -------
    float
        the fresnel coefficient "t sub p"

    """
    num = 2 * n0 * np.cos(theta0)
    den = n0 * np.cos(theta1) + n1 * np.cos(theta0)
    return num / den


def _as_layer_arrays(indices, thicknesses):
    """Validate and broadcast layer index and thickness arrays."""
    indices = np.asarray(indices)
    thicknesses = np.asarray(thicknesses)

    if indices.ndim == 0:
        indices = indices[None]

    if thicknesses.ndim == 0:
        thicknesses = thicknesses[None]

    try:
        indices, thicknesses = np.broadcast_arrays(indices, thicknesses)
    except ValueError as exc:
        raise ValueError('indices and thicknesses must be broadcastable to the same shape') from exc

    if indices.ndim < 1 or indices.shape[0] == 0:
        raise ValueError('indices and thicknesses must contain at least one film layer')

    return indices, thicknesses


def multilayer_stack_rt(indices, thicknesses, wavelength, polarization, substrate_index, aoi=0, ambient_index=1):
    """Compute r and t for a given stack of materials.

    Parameters
    ----------
    indices : ndarray
        refractive index for each film layer.  The first axis is the layer
        axis; any trailing axes are vectorized calculation dimensions.
    thicknesses : ndarray
        thickness of each film layer, microns.  Must be broadcastable to the
        same shape as indices.
    wavelength : float
        wavelength of light, microns
    polarization : str, {'p', 's'}
        the polarization state
    substrate_index : float or ndarray
        refractive index of the medium after the final film layer.  May be a
        scalar, or broadcastable to the trailing dimensions of indices and
        thicknesses.
    aoi : float, optional
        angle of incidence, degrees
    ambient_index : float, optional
        The refractive index the film is immersed in, defaults to 1 (vacuum)

    Returns
    -------
    (float, float)
        r, t coefficients

    """
    # digest inputs a little bit
    polarization = polarization.lower()
    if polarization not in ('p', 's'):
        raise ValueError("unknown polarization, use p or s")

    aoi = np.radians(aoi)
    indices, thicknesses = _as_layer_arrays(indices, thicknesses)
    layer_shape = indices.shape
    nlayers = layer_shape[0]
    calculation_shape = layer_shape[1:]

    if indices.ndim > 1:
        indices = indices.reshape((nlayers, -1))
        thicknesses = thicknesses.reshape((nlayers, -1))

        substrate_index = np.asarray(substrate_index)
        try:
            substrate_index = np.broadcast_to(substrate_index, calculation_shape).reshape(-1)
        except ValueError as exc:
            raise ValueError('substrate_index must be broadcastable to the trailing layer dimensions') from exc

    cost0 = np.cos(aoi)
    term1 = 1 / (2 * ambient_index * cost0)

    m00 = m01 = m10 = m11 = None
    for layer in range(nlayers):
        n = indices[layer]
        d = thicknesses[layer]
        cost = _cos_snell(ambient_index, n, aoi)
        beta = (2 * np.pi * n * d * cost) / wavelength
        sinb, cosb = np.sin(beta), np.cos(beta)

        if polarization == 'p':
            upper_right = -1j * sinb * cost / n
            lower_left = -1j * n * sinb / cost
        else:
            upper_right = -1j * sinb / (cost * n)
            lower_left = -1j * n * sinb * cost

        if layer == 0:
            m00 = cosb
            m01 = upper_right
            m10 = lower_left
            m11 = cosb
            continue

        new00 = m00 * cosb + m01 * lower_left
        new01 = m00 * upper_right + m01 * cosb
        new10 = m10 * cosb + m11 * lower_left
        m11 = m10 * upper_right + m11 * cosb
        m00, m01, m10 = new00, new01, new10

    substrate_cost = _cos_snell(ambient_index, substrate_index, aoi)

    if polarization == 'p':
        q0 = m00 * substrate_cost + m01 * substrate_index
        q1 = m10 * substrate_cost + m11 * substrate_index
        A00 = term1 * (ambient_index * q0 + cost0 * q1)
        A10 = term1 * (ambient_index * q0 - cost0 * q1)
    else:
        substrate_admittance = substrate_index * substrate_cost
        q0 = m00 + m01 * substrate_admittance
        q1 = m10 + m11 * substrate_admittance
        ambient_admittance = ambient_index * cost0
        A00 = term1 * (ambient_admittance * q0 + q1)
        A10 = term1 * (ambient_admittance * q0 - q1)

    r = A10 / A00
    t = 1 / A00

    if indices.ndim > 1:
        r = r.reshape(calculation_shape)
        t = t.reshape(calculation_shape)
    return r, t
