"""Tools for performing thin film calculations."""
from functools import reduce

from .mathops import np
from .conf import config


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


def snell_aor(n0, n1, theta, degrees=True):
    """Compute the angle of refraction using Snell's law.

    Parameters
    ----------
    n0 : float
        index of refraction of the "left" material
    n1 : float
        index of refraction of the "right" material
    theta : float
        angle of incidence, in degrees if degrees=True
    degrees : bool, optional
        if True, theta is interpreted as an angle in degrees

    Returns
    -------
    float
        angle of refraction

    """
    if degrees:
        theta = np.radians(theta)
    return np.lib.scimath.arcsin(n0/n1 * np.sin(theta))


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
    den = n1 * np.cos(theta0) + n1 * np.cos(theta1)
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
    characteristic_matrices : iterable of numpy.ndarray each of which of shape 2x2
        the characteristic matrices of each layer
    nnp1 : float or complex
        refractive index of the final medium
    theta_np1 : float
        angle of incidence on final medium, radians

    Returns
    -------
    numpy.ndarray
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

    if hasattr(theta_np1, '__len__') and len(theta_np1 > 1):
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
    characteristic_matrices : iterable of numpy.ndarray each of which of shape 2x2
        the characteristic matrices of each layer
    nnp1 : float or complex
        refractive index of the final medium
    theta_np1 : float
        angle of incidence on final medium, radians

    Returns
    -------
    numpy.ndarray
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

    if hasattr(theta_np1, '__len__') and len(theta_np1 > 1):
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
        term4 = np.moveaxis(term4, 2, 0)

    if hasattr(term1, '__len__') and len(term1) > 1:
        term12 = np.tensordot(term2, term1, axes=(0, 0))
    else:
        term12 = np.dot(term1, term2)

    return reduce(np.matmul, (term12, term3, term4))

    return reduce(np.dot, (term1, term2, term3, term4))


def rtot(Amat):
    """Compute rtot, the equivalent total fresnel coefficient for a multilayer stack.

    Parameters
    ----------
    Amat : numpy.ndarray
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
    Amat : numpy.ndarray
        2x2 array

    Returns
    -------
    float or complex
        the value of rtot, either s or p.

    """
    return 1 / Amat[..., 0, 0]


def multilayer_stack_rt(stack, wavelength, polarization, aoi=0, assume_vac_ambient=True):
    """Compute r and t for a given stack of materials.

    An infinitely thick layer of vacuum is assumed to proceed stack
    if assume_vac_ambient is True

    Parameters
    ----------
    polarization : str, {'p', 's'}
        the polarization state
    stack : numpy.ndarray
        array which has final dimensions of [n, t]
        where n is the index and t is the thickness in microns.
        i.e., stack[0] is N dimensional index of refraction, and
              stack[1] is N dimensional thickness

        For example, if stack is of shape (2, 100, 100)
        then the computation is for a 100x100 spatial arrangement of index
        and thickness

        iterable of tuples, which looks like [(n1, t1), (n2, t2) ...] also work
    wavelength : float
        wavelength of light, microns
    aoi : float, optional
        angle of incidence, degrees
    assume_vac_ambient : bool, optional
        if True, prepends an infinitely thick layer of vacuum to the stack
        if False, prepend the ambient index but *NOT* a thickness

    Returns
    -------
    (float, float)
        r, t coefficients

    """
    # digest inputs a little bit
    polarization = polarization.lower()
    aoi = np.radians(aoi)
    stack = np.asarray(stack)
    indices = stack[:, 0, ...]  # : = all layers, ... = keep other dims as present
    thicknesses = stack[:, 1, ...]

    # input munging:
    # downstream routines require shape (N, 2, 2) for batched matmul
    # input shape is (Nlayers, 2, ...more)
    # we are ultimately going to loop over Nlayers, but we need to flatten
    # the last dimension(s) and move them to the front
    # then within this function, because things do not naturally align to the
    # first axis, there are lots of awkward checks to see if we're multi-dimensional,
    # and if we are do the same thing but called slightly differently
    #
    # there's no way (that I know of) around that

    nlayers = len(stack)
    if indices.ndim > 1:
        indices = np.moveaxis(indices.reshape((nlayers, -1)), 1, 0)
        thicknesses = np.moveaxis(thicknesses.reshape((nlayers, -1)), 1, 0)

    angles = np.empty(thicknesses.shape, dtype=config.precision_complex)
    # do the first loop by hand to handle ambient vacuum gracefully
    if angles.ndim > 1:
        angles[:, 0] = aoi
        if assume_vac_ambient:
            result = snell_aor(1, indices[:, 0], aoi, degrees=False)
            angles[:, 1] = result
        else:
            angles[:, 1] = snell_aor(indices[:, 0], indices[:, 1], aoi, degrees=False)
        for i in range(2, thicknesses.shape[1]):
            angles[:, i] = snell_aor(indices[:, i-1], indices[:, i], angles[:, i-1], degrees=False)
    else:
        angles[0] = aoi
        if assume_vac_ambient:
            result = snell_aor(1, indices[0], aoi, degrees=False)
            angles[1] = result
        else:
            angles[1] = snell_aor(indices[0], indices[1], aoi, degrees=False)
        for i in range(2, len(thicknesses)):
            angles[i] = snell_aor(indices[i-1], indices[i], angles[i-1], degrees=False)

    if polarization == 'p':
        fn1 = characteristic_matrix_p
        fn2 = multilayer_matrix_p
    elif polarization == 's':
        fn1 = characteristic_matrix_s
        fn2 = multilayer_matrix_s
    else:
        raise ValueError("unknown polarization, use p or s")

    Mjs = []
    if angles.ndim > 1:
        for i in range(angles.shape[1]):
            Mjs.append(fn1(wavelength, thicknesses[:, i], indices[:, i], angles[:, i]))
    else:
        for i in range(len(angles)):
            Mjs.append(fn1(wavelength, thicknesses[i], indices[i], angles[i]))

    if Mjs[0].ndim > 2:
        Mjs = [np.moveaxis(M, 2, 0) for M in Mjs]

    if angles.ndim > 1:
        if assume_vac_ambient:
            i1 = np.broadcast_to(1, indices.shape[0])
            A = fn2(i1, angles[:, 0], Mjs, indices[:, -1], angles[:, -1])
        else:
            A = fn2(indices[:, 0], angles[:, 0], Mjs, indices[:, -1], angles[:, -1])
    else:
        if assume_vac_ambient:
            A = fn2(1, angles[0], Mjs, indices[-1], angles[-1])
        else:
            A = fn2(indices[0], angles[0], Mjs, indices[-1], angles[-1])

    r = rtot(A)
    t = ttot(A)
    if indices.ndim > 1:
        r = r.reshape(stack.shape[2:])
        t = t.reshape(stack.shape[2:])
    return r, t
