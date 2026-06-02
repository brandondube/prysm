"""Tools for performing thin film calculations."""
from functools import reduce

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


def characteristic_matrix_p(lambda_, d, n, theta):
    """Compute the characteristic matrix M_j^p for a layer of a material stack and p-polarized light.

    Uses (4.49) to compute (4.55) from BYU optics book, edition 2015

    Parameters
    ----------
    lambda_ : float
        wavelength of light, microns
    d : float
        thickness of the layer, microns
    n : float or complex
        refractive index of the layer
    theta : float
        angle of incidence, radians

    Returns
    -------
    numpy.array
        a 2x2 matrix

    """
    # BDD 2021-11-19: supports ND d, n automatically, no changes
    # d, n as shape (10,10) -> return is (2,2,10,10)
    k = (2 * np.pi * n) / lambda_
    cost = np.cos(theta)
    beta = k * d * cost
    sinb, cosb = np.sin(beta), np.cos(beta)

    upper_right = -1j * sinb * cost / n
    lower_left = -1j * n * sinb / cost
    return np.asarray([
        [cosb, upper_right],
        [lower_left, cosb]
    ])


def characteristic_matrix_s(lambda_, d, n, theta):
    """Compute the characteristic matrix M_j^p for a layer of a material stack and s-polarized light.

    Uses (4.49) to compute (4.55) from BYU optics book, edition 2015

    Parameters
    ----------
    lambda_ : float
        wavelength of light, microns
    d : float
        thickness of the layer, microns
    n : float or complex
        refractive index of the layer
    theta : float
        angle of incidence, radians

    Returns
    -------
    numpy.array
        a 2x2 matrix

    """
    k = (2 * np.pi * n) / lambda_
    cost = np.cos(theta)
    beta = k * d * cost
    sinb, cosb = np.sin(beta), np.cos(beta)

    upper_right = -1j * sinb / (cost * n)
    lower_left = -1j * n * sinb * cost
    return np.asarray([
        [cosb, upper_right],
        [lower_left, cosb]
    ])


def multilayer_matrix_p(n0, theta0, characteristic_matrices, nnp1, theta_np1):
    """Reduce a multilayer problem to give the 2x2 matrix A^p.

    Computes (4.58) from BYU optics book.

    Parameters
    ----------
    n0 : float or complex
        refractive index of the first medium
    theta0 : float
        angle of incidence on the first medium, radians
    characteristic_matrices : iterable of ndarray each of which of shape 2x2
        the characteristic matrices of each layer
    nnp1 : float or complex
        refractive index of the final medium
    theta_np1 : float
        angle of incidence on final medium, radians

    Returns
    -------
    ndarray
        2x2 matrix A^s

    """
    # there are a lot of guards in this function that look weird
    # basically, we may have characteristic metricies for multiple
    # thicknesses/indices along the first dim (N, 2, 2) instead of (2, 2)
    # numpy matmul is designed to do "batch" compuations in this case, but
    # we need to make sure all of our scalars, etc, give arrays of the same
    # shape.  I.e., matmul(scalar, (3,2,2)) is illegal where dot(scalar, (3,2,2))
    # was not.  The "noise" in this function is there to take care of those parts

    # there may be some performance left on the table because of the moveaxes
    # all over the place in this function, I'm not precisely sure.
    cost0 = np.cos(theta0)
    term1 = 1 / (2 * n0 * cost0)

    term2 = np.asarray([
        [n0, cost0],
        [n0, -cost0]
    ])

    if len(characteristic_matrices) > 1:
        term3 = reduce(np.matmul, characteristic_matrices)  # reduce does M1 * M2 * M3 [...]
    else:
        term3 = characteristic_matrices[0]

    # non-scalar case
    if hasattr(theta_np1, '__len__'):
        term4 = np.asarray([
            [np.cos(theta_np1), np.broadcast_to(0, theta_np1.shape)],
            [nnp1,              np.broadcast_to(0, theta_np1.shape)]
        ])
    else:
        term4 = np.asarray([
            [np.cos(theta_np1), 0],
            [nnp1,              0]
        ])

    if term2.ndim > 2:
        term2 = np.moveaxis(term2, 2, 0)

    if term4.ndim > 2:
        term4 = np.moveaxis(term4, 2, 0)

    if hasattr(term1, '__len__') and len(term1) > 1:
        term12 = np.tensordot(term2, term1, axes=(0, 0))
    else:
        term12 = np.dot(term1, term2)

    return reduce(np.matmul, (term12, term3, term4))


def multilayer_matrix_s(n0, theta0, characteristic_matrices, nnp1, theta_np1):
    """Reduce a multilayer problem to give the 2x2 matrix A^s.

    Computes (4.62) from BYU optics book.

    Parameters
    ----------
    n0 : float or complex
        refractive index of the first medium
    theta0 : float
        angle of incidence on the first medium, radians
    characteristic_matrices : iterable of ndarray each of which of shape 2x2
        the characteristic matrices of each layer
    nnp1 : float or complex
        refractive index of the final medium
    theta_np1 : float
        angle of incidence on final medium, radians

    Returns
    -------
    ndarray
        2x2 matrix A^s

    """
    cost0 = np.cos(theta0)
    term1 = 1 / (2 * n0 * cost0)
    n0cost0 = n0 * cost0

    term2 = np.asarray([
        [n0cost0, np.broadcast_to(1, n0cost0.shape)],
        [n0cost0, np.broadcast_to(-1, n0cost0.shape)]
    ])
    if len(characteristic_matrices) > 1:
        term3 = reduce(np.matmul, characteristic_matrices)
    else:
        term3 = characteristic_matrices[0]

    # non-scalar case
    if hasattr(theta_np1, '__len__'):
        term4 = np.asarray([
            [np.broadcast_to(1, theta_np1.shape), np.broadcast_to(0, theta_np1.shape)],
            [nnp1 * np.cos(theta_np1),            np.broadcast_to(0, theta_np1.shape)]
        ])
    else:
        term4 = np.asarray([
            [1,                       0],
            [nnp1 * np.cos(theta_np1), 0]
        ])

    if term2.ndim > 2:
        term2 = np.moveaxis(term2, 2, 0)

    if term4.ndim > 2:
        term4 = np.moveaxis(term4, 2, 0)

    if hasattr(term1, '__len__') and len(term1) > 1:
        term12 = np.tensordot(term2, term1, axes=(0, 0))
    else:
        term12 = np.dot(term1, term2)

    return reduce(np.matmul, (term12, term3, term4))


def rtot(Amat):
    """Compute rtot, the equivalent total fresnel coefficient for a multilayer stack.

    Parameters
    ----------
    Amat : ndarray
        2x2 array

    Returns
    -------
    float or complex
        the value of rtot, either s or p.

    """
    # ... to support batch computation
    return Amat[..., 1, 0] / Amat[..., 0, 0]


def ttot(Amat):
    """Compute ttot, the equivalent total fresnel coefficient for a multilayer stack.

    Parameters
    ----------
    Amat : ndarray
        2x2 array

    Returns
    -------
    float or complex
        the value of rtot, either s or p.

    """
    return 1 / Amat[..., 0, 0]


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
