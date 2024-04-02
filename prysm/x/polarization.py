"Jones and Mueller Calculus"
from prysm.mathops import np
from prysm.conf import config
from prysm import propagation
from prysm.coordinates import make_xy_grid,cart_to_polar
import functools


# supported functions for jones_decorator
supported_propagation_funcs = ['focus','unfocus','focus_fixed_sampling','angular_spectrum']


U = np.array([[1, 0,   0,  1],
                [1, 0,   0, -1],
                [0, 1,   1,  0],
                [0, 1j, -1j, 0]]) / np.sqrt(2)

def _empty_pol_vector(shape=None):
    """Returns an empty array to populate with jones vector elements.

    Parameters
    ----------
    shape : list, optional
        shape to prepend to the jones vector array. shape = [32,32] returns an array of shape [32,32,2,1]
        where the matrix is assumed to be in the last indices. Defaults to None, which returns a [2] array.

    Returns
    -------
    numpy.ndarray
        The empty array of specified shape
    """

    if shape is None:
        
        shape = (2)

    else:

        shape = (*shape,2,1)

    return np.zeros(shape, dtype=config.precision_complex)

def linear_pol_vector(angle, degrees=True):
    """Returns a linearly polarized jones vector at a specified angle

    Parameters
    ----------
    angle : float
        angle that the linear polarization is oriented with respect to the horizontal axis
    shape : list, optional
        shape to prepend to the jones vector array. shape = [32,32] returns an array of shape [32,32,2,1]
        where the matrix is assumed to be in the last indices. Defaults to None, which returns a [2] array.

    Returns
    -------
    numpy.ndarray
        linear jones vector
    """

    if degrees:
        angle = angle * np.pi / 180

    cost = np.cos(angle)
    sint = np.sin(angle)

    if hasattr(angle,'ndim'):
        pol_vector = _empty_pol_vector(shape=angle.shape)
        pol_vector[...,0,0] = cost
        pol_vector[...,1,0] = sint
    else:
        pol_vector = _empty_pol_vector(shape=None)
        pol_vector[0] = cost
        pol_vector[1] = sint

    return pol_vector

def circular_pol_vector(handedness='left',shape=None):
    """Returns a circularly polarized jones vector

    Parameters
    ----------
    shape : list, optional
        shape to prepend to the jones vector array. shape = [32,32] returns an array of shape [32,32,2,1]
        where the matrix is assumed to be in the last indices. Defaults to None, which returns a [2] array.

    Returns
    -------
    numpy.ndarray
        circular jones vector
    """

    pol_vector = _empty_pol_vector(shape=shape)
    pol_vector[...,0] = 1/np.sqrt(2)
    if handedness == 'left':
        pol_vector[...,1] = 1j/np.sqrt(2)
    elif handedness == 'right':
        pol_vector[...,1] = -1j/np.sqrt(2)
    else:
        raise ValueError(f"unknown handedness {handedness}, use 'left' or 'right''")

    return pol_vector


def _empty_jones(shape=None):
    """Returns an empty array to populate with jones matrix elements.

    Parameters
    ----------
    shape : list
        shape to prepend to the jones matrix array. shape = [32,32] returns an array of shape [32,32,2,2]
        where the matrix is assumed to be in the last indices. Defaults to None, which returns a 2x2 array.

    Returns
    -------
    numpy.ndarray
        The empty array of specified shape
    """

    if shape is None:

        shape = (2, 2)

    else:

        shape = (*shape, 2, 2)

    return np.zeros(shape, dtype=config.precision_complex)


def jones_rotation_matrix(theta, shape=None):
    """A rotation matrix for rotating the coordinate system transverse to propagation.
    source: https://en.wikipedia.org/wiki/Rotation_matrix.

    Parameters
    ----------
    theta : float
        angle in radians to rotate the jones matrix with respect to the x-axis.

    shape : list
        shape to prepend to the jones matrix array. shape = [32,32] returns an array of shape [32,32,2,2]
        where the matrix is assumed to be in the last indices. Defaults to None, which returns a 2x2 array.

    Returns
    -------
    numpy.ndarray
        2D rotation matrix
    """

    jones = _empty_jones(shape=shape)
    cost = np.cos(theta)
    sint = np.sin(theta)
    jones[..., 0, 0] = cost
    jones[..., 0, 1] = sint
    jones[..., 1, 0] = -sint
    jones[..., 1, 1] = cost

    return jones


def linear_retarder(retardance, theta=0, shape=None):
    """Generates a homogenous linear retarder jones matrix.

    Parameters
    ----------
    retardance : float
        phase delay experienced by the slow state in radians.

    theta : float
        angle in radians the linear retarder is rotated with respect to the x-axis.
        Defaults to 0.

    shape : list
        shape to prepend to the jones matrix array. shape = [32,32] returns an array of shape [32,32,2,2]
        where the matrix is assumed to be in the last indices. Defaults to None, which returns a 2x2 array.


    Returns
    -------
    retarder : numpy.ndarray
        numpy array containing the retarder matrices
    """

    retphasor = np.exp(1j*retardance)

    jones = _empty_jones(shape=shape)

    jones[..., 0, 0] = 1
    jones[..., 1, 1] = retphasor

    retarder = jones_rotation_matrix(-theta) @ jones @ jones_rotation_matrix(theta)

    return retarder


def linear_diattenuator(alpha, theta=0, shape=None):
    """Generates a homogenous linear diattenuator jones matrix.

    Parameters
    ----------
    alpha : float
        Fraction of the light that passes through the partially transmitted channel.
        If 1, this is an unpolarizing plate. If 0, this is a perfect polarizer

    theta : float
        angle in radians the linear retarder is rotated with respect to the x-axis.
        Defaults to 0.

    shape : list
        shape to prepend to the jones matrix array. shape = [32,32] returns an array of shape [32,32,2,2]
        where the matrix is assumed to be in the last indices. Defaults to None, which returns a 2x2 array.


    Returns
    -------
    diattenuator : numpy.ndarray
        numpy array containing the diattenuator matrices
    """
    assert (alpha >= 0) and (alpha <= 1), f"alpha cannot be less than 0 or greater than 1, got: {alpha}"

    jones = _empty_jones(shape=shape)
    jones[..., 0, 0] = 1
    jones[..., 1, 1] = alpha

    diattenuator = jones_rotation_matrix(-theta) @ jones @ jones_rotation_matrix(theta)

    return diattenuator


def half_wave_plate(theta=0, shape=None):
    """Make a half wave plate jones matrix. Just a wrapper for linear_retarder.

    Parameters
    ----------
    theta : float
        angle in radians the linear retarder is rotated with respect to the x-axis.
        Defaults to 0.
    shape : list
        shape to prepend to the jones matrix array. shape = [32,32] returns an array of shape [32,32,2,2]
        where the matrix is assumed to be in the last indices. Defaults to None, which returns a 2x2 array.

    Returns
    -------
    linear_retarder
        a linear retarder with half-wave retardance
    """
    return linear_retarder(np.pi, theta=theta, shape=shape)


def quarter_wave_plate(theta=0, shape=None):
    """Make a quarter wave plate jones matrix. Just a wrapper for linear_retarder.

    Parameters
    ----------
    theta : float
        angle in radians the linear retarder is rotated with respect to the x-axis.
        Defaults to 0.
    shape : list, optional
        shape to prepend to the jones matrix array. shape = [32,32] returns an array of shape [32,32,2,2]
        where the matrix is assumed to be in the last indices. Defaults to None, which returns a 2x2 array.

    Returns
    -------
    linear_retarder
        a linear retarder with quarter-wave retardance
    """
    return linear_retarder(np.pi / 2, theta=theta, shape=shape)


def linear_polarizer(theta=0, shape=None):
    """Make a linear polarizer jones matrix. Just a wrapper for linear_diattenuator.

    Returns
    -------
    theta : float
        angle in radians the linear retarder is rotated with respect to the x-axis.
        Defaults to 0.
    shape : list
        shape to prepend to the jones matrix array. shape = [32,32] returns an array of shape [32,32,2,2]
        where the matrix is assumed to be in the last indices. Defaults to None, which returns a 2x2 array.

    Returns
    -------
    linear_diattenuator
        a linear diattenuator with unit diattenuation
    """

    return linear_diattenuator(0, theta=theta, shape=shape)

def vector_vortex_retarder(charge, shape, retardance=np.pi, theta=0):
    """generate a phase-only spatially-varying vector vortex retarder (VVR)

    This model follows Eq (7) in D. Mawet. et al. (2009)
    https://opg.optica.org/oe/fulltext.cfm?uri=oe-17-3-1902&id=176231 (open access)

    Parameters
    ----------
    charge : float
        topological charge of the vortex, typically an interger
    shape : tuple of int
        shape of the VR array
    retardance : float
        phase difference between the ordinary and extraordinary modes, by default np.pi or half a wave
    theta : float, optional
        angle in radians to rotate the vortex by, by default 0

    Returns
    -------
    _type_
        _description_
    """
    
    vvr_lhs = _empty_jones(shape=[shape,shape])
    vvr_rhs = _empty_jones(shape=[shape,shape])

    # create the dimensions
    x,y = make_xy_grid(shape,diameter=1)
    r,t = cart_to_polar(x,y)
    t *= charge

    # precompute retardance
    cost = np.cos(t)
    sint = np.sin(t)
    jcosr = -1j*np.cos(retardance/2)
    jsinr = np.sin(retardance/2)

    # build jones matrices
    vvr_lhs[...,0,0] = cost
    vvr_lhs[...,0,1] = sint
    vvr_lhs[...,1,0] = sint
    vvr_lhs[...,1,1] = -cost
    vvr_lhs *= jsinr

    vvr_rhs[...,0,0] = jcosr
    vvr_rhs[...,0,0] = jcosr

    vvr = vvr_lhs + vvr_rhs

    vvr = jones_rotation_matrix(-theta) @ vvr @ jones_rotation_matrix(theta)

    return vvr

def broadcast_kron(a,b):
    """broadcasted kronecker product of two N,M,...,2,2 arrays. Used for jones -> mueller conversion
    In the unbroadcasted case, this output looks like

    out = [a[0,0]*b,a[0,1]*b]
          [a[1,0]*b,a[1,1]*b]

    where out is a N,M,...,4,4 array. I wrote this to work for generally shaped kronecker products where the matrix
    is contained in the last two axes, but it's only tested for the Nx2x2 case

    Parameters
    ----------
    a : numpy.ndarray
        N,M,...,2,2 array used to scale b in kronecker product
    b : numpy.ndarray
        N,M,...,2,2 array used to form block matrices in kronecker product

    Returns
    -------
    out
        N,M,...,4,4 array
    """

    return np.einsum('...ik,...jl',a,b).reshape([*a.shape[:-2],int(a.shape[-2]*b.shape[-2]),int(a.shape[-1]*b.shape[-1])])

def jones_to_mueller(jones, broadcast=True):
    """Construct a Mueller Matrix given a Jones Matrix. From Chipman, Lam, and Young Eq (6.99).

    Parameters
    ----------
    jones : ndarray with final dimensions 2x2
        The complex-valued jones matrices to convert into mueller matrices
    broadcast : bool
        Whether to use the experimental `broadcast_kron` to compute the conversion in a broadcast fashion, by default True

    Returns
    -------
    M : np.ndarray
        Mueller matrix
    """

    if broadcast:
        jprod = broadcast_kron(np.conj(jones), jones)
    else:
        jprod = np.kron(np.conj(jones), jones)

    M = np.real(U @ jprod @ np.linalg.inv(U))
    return M

def pauli_spin_matrix(index, shape=None):
    """Generates a pauli spin matrix used for Jones matrix data reduction. From CLY Eq 6.108.

    Parameters
    ----------
    index : int
        0 - the identity matrix
        1 - a linear half-wave retarder oriented horizontally
        2 - a linear half-wave retarder oriented 45 degrees
        3 - a circular half-wave retarder
    shape : list, optional
        shape to prepend to the jones matrix array.
        shape = [32,32] returns an array of shape [32,32,2,2]
        where the matrix is assumed to be in the last indices.
        Default returns a 2x2 array

    Returns
    -------
    jones
        pauli spin matrix of index specified
    """

    jones = _empty_jones(shape=shape)

    assert index in (0, 1, 2, 3), f"index should be 0,1,2, or 3. Got {index}"

    if index == 0:
        jones[..., 0, 0] = 1
        jones[..., 1, 1] = 1

    elif index == 1:
        jones[..., 0, 0] = 1
        jones[..., 1, 1] = -1

    elif index == 2:
        jones[..., 0, 1] = 1
        jones[..., 1, 0] = 1

    elif index == 3:
        jones[..., 0, 1] = -1j
        jones[..., 1, 0] = 1j

    return jones


def pauli_coefficients(jones):
    """Compute the pauli coefficients of a jones matrix.

    Parameters
    ----------
    jones : numpy.ndarray
        complex jones matrix to decompose


    Returns
    -------
    c0,c1,c2,c3
        complex coefficients of pauli matrices
    """

    c0 = (jones[..., 0, 0] + jones[..., 1, 1]) / 2
    c1 = (jones[..., 0, 0] - jones[..., 1, 1]) / 2
    c2 = (jones[..., 0, 1] + jones[..., 1, 0]) / 2
    c3 = 1j*(jones[..., 0, 1] - jones[..., 1, 0]) / 2

    return c0, c1, c2, c3


def jones_adapter(prop_func):
    """wrapper around prysm.propagation functions to support polarized field propagation

    Parameters
    ----------
    prop_func : callable
        propagation function to decorate

    Notes
    -----
    There isn't anything particularly special about polarized field propagation. We simply 
    leverage the independence of the 4 "polarized" components of an optical system expressed
    as a Jones matrix

    J = [
        [J00,J01],
        [J10,J11]
    ]

    The elements of this matrix can be propagated as incoherent wavefronts to express the polarized
    response of an optical system. All `jones_adapter` does is call a given propagation function
    4 times, one for each element of the Jones matrix.

    Returns
    -------
    callable
        decorated propagation function
    """

    @functools.wraps(prop_func)
    def wrapper(*args,**kwargs):

        
        # this is a function
        wavefunction = args[0]
        if len(args) > 1:
            other_args = args[1:]
        else:
            other_args = ()

        if wavefunction.ndim == 2:
            # pass through non-jones case
            return prop_func(*args,**kwargs)

        J00 = wavefunction[...,0,0]
        J01 = wavefunction[...,0,1]
        J10 = wavefunction[...,1,0]
        J11 = wavefunction[...,1,1]
        tmp = []
        for E in [J00, J01, J10, J11]:
            ret = prop_func(E, *other_args, **kwargs)
            tmp.append(ret)
        
        out = np.empty([*ret.shape,2,2],dtype=ret.dtype)
        out[...,0,0] = tmp[0]
        out[...,0,1] = tmp[1]
        out[...,1,0] = tmp[2]
        out[...,1,1] = tmp[3]
        
        return out
    
    return wrapper

def add_jones_propagation(funcs_to_change=supported_propagation_funcs):
    """apply decorator to supported propagation functions

    Parameters
    ----------
    funcs_to_change : list, optional
        list of propagation functions to add polarized field propagation to, by default supported_propagation_funcs
    """

    for name,func in vars(propagation).items():
        if name in funcs_to_change:
            setattr(propagation, name, jones_adapter(func))

def apply_polarization_to_field(field):
    """Extends the dimensions of a scalar field to be compatible with jones calculus

    Parameters
    ----------
    field : numpy.ndarray
        scalar field of shape M x N

    Returns
    -------
    numpy.ndarray
        jones matrix field of shape M x N x 1 x 1
    """

    field = field[..., np.newaxis, np.newaxis]
    
    return field


