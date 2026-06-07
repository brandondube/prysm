"""Thin-film stack utilities with internal field access.

Layers are ambient-side first.  Angles are radians; thicknesses and wavelengths
are microns.  Characteristic matrices use trailing 2x2 axes and broadcast over
the leading sample axes.
"""

from prysm.conf import config
from prysm.mathops import np
from prysm.thinfilm import _cos_snell


def _resolve(index, wvl):
    """Resolve a constant, callable, or material index at wavelength wvl."""
    nk = getattr(index, 'nk', None)
    if callable(nk):
        return nk(wvl)
    if callable(index):
        return index(wvl)
    return index


def _admittance(n, cost, pol):
    """Tilted optical admittance eta for a medium of index n at cos(theta)."""
    if pol == 'p':
        return n / cost
    return n * cost


def _char_matrix(beta, eta):
    """Per-layer characteristic matrix.

    Parameters
    ----------
    beta : ndarray or float
        phase thickness, possibly complex.
    eta : ndarray or float
        tilted optical admittance.

    """
    cosb = np.cos(beta) + 0j
    sinb = np.sin(beta)
    m01 = -1j * sinb / eta
    m10 = -1j * eta * sinb
    row0 = np.stack([cosb, m01], axis=-1)
    row1 = np.stack([m10, cosb], axis=-1)
    return np.stack([row0, row1], axis=-2)


def _matvec(M, v):
    """Batched matrix-vector product: (*calc, 2, 2) applied to (*calc, 2)."""
    return (M @ v[..., None])[..., 0]


def _eye2():
    """2x2 complex identity (broadcasts against any calc shape under matmul)."""
    return np.eye(2) + 0j


class Stack:
    """A multilayer thin-film stack.

    Parameters
    ----------
    indices : sequence
        per-layer refractive index, ambient side first.
    thicknesses : sequence or ndarray
        per-layer physical thicknesses, microns.
    substrate_index : float, complex, or callable
        index of the medium after the last layer.
    ambient_index : float, complex, or callable, optional
        index of the incidence medium.

    """

    __slots__ = ('indices', 'thicknesses', 'substrate_index', 'ambient_index')

    def __init__(self, indices, thicknesses, substrate_index, ambient_index=1.0):
        indices = list(indices)
        thicknesses = np.asarray(thicknesses, dtype=config.precision)
        if thicknesses.ndim == 0:
            thicknesses = np.full(len(indices), thicknesses,
                                  dtype=config.precision)
        if len(indices) != thicknesses.shape[0]:
            raise ValueError('indices and thicknesses must describe the same number of layers')
        self.indices = indices
        self.thicknesses = thicknesses
        self.substrate_index = substrate_index
        self.ambient_index = ambient_index

    def __len__(self):
        return self.thicknesses.shape[0]

    def resolved_indices(self, wvl):
        """List of per-layer indices evaluated at wavelength wvl."""
        return [_resolve(n, wvl) for n in self.indices]

    def __repr__(self):
        return f'Stack({len(self)} layers, substrate={self.substrate_index!r})'


def stack_characteristic_matrices(stack, wvl, theta0, pol):
    """Per-layer characteristic matrices for a stack.

    Parameters
    ----------
    stack : Stack
    wvl : float or ndarray
        wavelength, microns.
    theta0 : float or ndarray
        angle of incidence in the ambient medium, radians.
    pol : str
        polarization, 'p' or 's'.

    Returns
    -------
    list of ndarray
        one matrix per layer, ambient side first.

    """
    pol = pol.lower()
    n0 = _resolve(stack.ambient_index, wvl)
    matrices = []
    for n, d in zip(stack.resolved_indices(wvl), stack.thicknesses):
        cost = _cos_snell(n0, n, theta0)
        beta = (2 * np.pi * n * d * cost) / wvl
        eta = _admittance(n, cost, pol)
        matrices.append(_char_matrix(beta, eta))
    return matrices


def forward_products(matrices):
    """Cumulative left products of a matrix stack.

    Parameters
    ----------
    matrices : sequence of ndarray
        per-layer characteristic matrices, ambient side first.

    Returns
    -------
    list of ndarray
        length N+1; entry 0 is the identity.

    """
    L = [_eye2()]
    for M in matrices:
        L.append(L[-1] @ M)
    return L


def backward_products(matrices):
    """Cumulative right products of a matrix stack.

    Parameters
    ----------
    matrices : sequence of ndarray
        per-layer characteristic matrices, ambient side first.

    Returns
    -------
    list of ndarray
        length N+1; entry N is the identity.

    """
    N = len(matrices)
    R = [None] * (N + 1)
    R[N] = _eye2()
    for k in range(N - 1, -1, -1):
        R[k] = matrices[k] @ R[k + 1]
    return R


def _evaluate(stack, wvl, theta0, pol):
    """Shared transfer-matrix forward pass."""
    pol = pol.lower()
    if pol not in ('p', 's'):
        raise ValueError("unknown polarization, use 'p' or 's'")

    n0 = _resolve(stack.ambient_index, wvl)
    nsub = _resolve(stack.substrate_index, wvl)
    cost0 = np.cos(theta0)
    cost_sub = _cos_snell(n0, nsub, theta0)
    eta0 = _admittance(n0, cost0, pol)
    eta_sub = _admittance(nsub, cost_sub, pol)

    matrices = stack_characteristic_matrices(stack, wvl, theta0, pol)
    R = backward_products(matrices)
    M = R[0]  # full assembly product M_1 ... M_N

    # B, C are the front-side tangential fields normalized to a unit
    # substrate-side electric field: [B, C] = M [1, eta_sub].
    B = M[..., 0, 0] + M[..., 0, 1] * eta_sub
    C = M[..., 1, 0] + M[..., 1, 1] * eta_sub
    denom = eta0 * B + C
    r = (eta0 * B - C) / denom
    t = 2 * eta0 / denom

    # substrate-side state for a unit-amplitude incident wave, then the field at
    # every boundary by a single backward sweep of partial products.
    v_sub = np.stack([t, t * eta_sub], axis=-1)
    fields = np.stack([_matvec(Rk, v_sub) for Rk in R], axis=0)
    return matrices, R, eta0, eta_sub, r, t, fields


def stack_rt(stack, wvl, theta0, pol):
    """Amplitude reflection and transmission coefficients.

    Parameters
    ----------
    stack : Stack
    wvl : float or ndarray
        wavelength, microns.
    theta0 : float or ndarray
        angle of incidence in the ambient medium, radians.
    pol : str
        polarization, 'p' or 's'.

    Returns
    -------
    (ndarray, ndarray)
        r and t for a unit-amplitude incident wave.

    """
    _, _, _, _, r, t, _ = _evaluate(stack, wvl, theta0, pol)
    return r, t


def internal_fields(stack, wvl, theta0, pol):
    """Tangential electric and magnetic field at every boundary.

    Parameters
    ----------
    stack : Stack
    wvl : float or ndarray
        wavelength, microns.
    theta0 : float or ndarray
        angle of incidence in the ambient medium, radians.
    pol : str
        polarization, 'p' or 's'.

    Returns
    -------
    (ndarray, ndarray)
        E and H; leading axis indexes boundaries from the ambient side.

    """
    *_, fields = _evaluate(stack, wvl, theta0, pol)
    return fields[..., 0], fields[..., 1]


def field_at_depth(stack, z, wvl, theta0, pol):
    """Tangential field at arbitrary depth(s) inside the stack.

    Parameters
    ----------
    stack : Stack
    z : float or ndarray
        depth(s) into the stack from the ambient side, microns.
    wvl : float
        wavelength, microns.
    theta0 : float
        angle of incidence in the ambient medium, radians.
    pol : str
        polarization, 'p' or 's'.

    Returns
    -------
    (ndarray, ndarray)
        E, H at each depth, shaped like z.

    """
    pol = pol.lower()
    z = np.asarray(z, dtype=config.precision)
    N = len(stack)
    if N == 0:
        raise ValueError('field_at_depth requires at least one layer')

    n0 = _resolve(stack.ambient_index, wvl)
    ns = np.asarray(stack.resolved_indices(wvl))
    ds = stack.thicknesses

    E, H = internal_fields(stack, wvl, theta0, pol)

    # boundary depths Z[0..N]; layer j spans [Z[j], Z[j+1]].
    Z = np.concatenate([np.zeros(1, dtype=config.precision), np.cumsum(ds)])
    if bool(np.any((z < 0) | (z > Z[-1]))):
        raise ValueError('z must lie within the coating stack')
    li = np.clip(np.searchsorted(Z, z, side='right') - 1, 0, N - 1)

    n_z = ns[li]
    cost_z = _cos_snell(n0, n_z, theta0)
    eta_z = _admittance(n_z, cost_z, pol)
    t_below = Z[li + 1] - z
    beta_z = (2 * np.pi * n_z * t_below * cost_z) / wvl
    Mz = _char_matrix(beta_z, eta_z)

    # propagate up from the substrate-side boundary of the containing layer.
    v_bottom = np.stack([E[li + 1], H[li + 1]], axis=-1)
    f = _matvec(Mz, v_bottom)
    return f[..., 0], f[..., 1]


def RTA(stack, wvl, theta0, pol):
    """Reflectance, transmittance, and per-layer absorptance.

    Parameters
    ----------
    stack : Stack
    wvl : float or ndarray
        wavelength, microns.
    theta0 : float or ndarray
        angle of incidence in the ambient medium, radians.
    pol : str
        polarization, 'p' or 's'.

    Returns
    -------
    (ndarray, ndarray, ndarray)
        R, T, and A; A has one leading entry per layer.

    """
    _, _, eta0, eta_sub, r, t, fields = _evaluate(stack, wvl, theta0, pol)
    R = np.abs(r) ** 2
    T = np.real(eta_sub) / np.real(eta0) * np.abs(t) ** 2

    E = fields[..., 0]
    H = fields[..., 1]
    # net power flux toward the substrate at each boundary, normalized to the
    # incident power: flux[0] = 1 - R, flux[N] = T.  Each layer absorbs the
    # difference, so the per-layer split telescopes to the global A = 1 - R - T.
    flux = np.real(E * np.conj(H)) / np.real(eta0)
    A = flux[:-1] - flux[1:]
    return R, T, A
