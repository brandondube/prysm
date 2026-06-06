"""Stack representation + field / partial-product engine.

This is the foundation the rest of the coatings package is built on.  The core
2x2 transfer-matrix primitives in prysm.thinfilm collapse a stack into the two
A-matrix coefficients needed for r and t and throw away the intermediate state.
Needle optimization, analytic gradients, field-constrained design, and
monitoring simulation all need that intermediate state -- the partial products
of the per-layer characteristic matrices and the internal electric/magnetic
field at every boundary.  This module computes it once and exposes it.

Conventions
-----------
The per-layer characteristic matrix follows the optical-admittance form used by
prysm.thinfilm.multilayer_stack_rt (the -i time convention),

    M_j = [[cos b_j,            -i sin b_j / eta_j],
           [-i eta_j sin b_j,    cos b_j         ]]

with phase thickness b_j = (2 pi / lambda) n_j d_j cos(theta_j) and tilted
admittance eta_j = n_j / cos(theta_j) for p polarization, n_j cos(theta_j) for
s.  The matrix relates the tangential fields at the two boundaries of a layer,

    [E, H]_top = M_j [E, H]_bottom,

so the assembly relation is [E, H]_front = (M_1 ... M_N) [E, H]_substrate.  A
unit-amplitude incident wave fixes the substrate-side state to t [1, eta_sub],
from which the field at any boundary follows by a partial product and r, t, R,
T, and per-layer absorptance follow from the boundary fields.

Matrices carry the 2x2 in their trailing two axes, shape (*calc, 2, 2), where
calc is whatever shape the wavelength / angle broadcast to.  This lets numpy
matmul batch the products directly, without the axis juggling thinfilm needs for
its (2, 2, *calc) layout.

Angles are in radians here (this is the low-level engine); thinfilm.snell_aor
and np.radians bridge from degrees.  Thicknesses and wavelengths are microns.
The engine is backend-pure (no float() coercion) so the same forward pass is
reused by the Phase 2 analytic-gradient adjoint.
"""

from prysm.conf import config
from prysm.mathops import np
from prysm.thinfilm import _cos_snell


def _resolve(index, wvl):
    """Resolve a possibly-dispersive index to its value at wvl."""
    if callable(index):
        return index(wvl)
    return index


def _admittance(n, cost, pol):
    """Tilted optical admittance eta for a medium of index n at cos(theta)."""
    if pol == 'p':
        return n / cost
    return n * cost


def _char_matrix(beta, eta):
    """Per-layer characteristic matrix of shape (*calc, 2, 2).

    Parameters
    ----------
    beta : ndarray or float
        phase thickness (2 pi / lambda) n d cos(theta), possibly complex.
    eta : ndarray or float
        tilted optical admittance of the layer for the polarization.

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
    """A multilayer thin-film stack: ordered layers between ambient and substrate.

    Layers are ordered from the ambient (incidence) side toward the substrate.
    Each layer index, the substrate index, and the ambient index may be a fixed
    value (real or complex) or a callable n(wavelength) dispersion model -- the
    glass / air callables from prysm.x.materials work directly.

    Parameters
    ----------
    indices : sequence
        per-layer refractive index, ambient side first.  Each entry is a fixed
        value or a callable n(wvl).
    thicknesses : sequence or ndarray
        per-layer physical thickness, microns.  Broadcast to one value per
        layer; these are the design variables for refinement.
    substrate_index : float, complex, or callable
        index of the medium after the last layer.
    ambient_index : float, complex, or callable, optional
        index of the incidence medium, default 1.0 (vacuum / air).

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

    Thin wrapper that resolves the per-layer Snell angle (referenced to the
    ambient via the invariant n sin(theta) = const) and admittance, then builds
    each layer matrix.

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
        one (*calc, 2, 2) matrix per layer, ambient side first.

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
    """Cumulative left products L_k = M_1 ... M_k.

    Parameters
    ----------
    matrices : sequence of ndarray
        per-layer characteristic matrices, ambient side first.

    Returns
    -------
    list of ndarray
        length N+1; entry k is the product of the first k matrices, with entry
        0 the identity and entry N the full assembly product.

    """
    L = [_eye2()]
    for M in matrices:
        L.append(L[-1] @ M)
    return L


def backward_products(matrices):
    """Cumulative right products R_k = M_{k+1} ... M_N.

    Parameters
    ----------
    matrices : sequence of ndarray
        per-layer characteristic matrices, ambient side first.

    Returns
    -------
    list of ndarray
        length N+1; entry k is the product of the matrices below boundary k,
        with entry N the identity and entry 0 the full assembly product.  R_k
        applied to the substrate-side field vector gives the field at boundary k.

    """
    N = len(matrices)
    R = [None] * (N + 1)
    R[N] = _eye2()
    for k in range(N - 1, -1, -1):
        R[k] = matrices[k] @ R[k + 1]
    return R


def _evaluate(stack, wvl, theta0, pol):
    """Shared forward pass.

    Returns matrices, backward products R, eta0, eta_sub, r, t, and the boundary
    field vectors (N+1, *calc, 2) where component 0 is tangential E and 1 is
    tangential H.  r and t are the Macleod-normalized amplitude coefficients for
    a unit-amplitude incident wave.
    """
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
    """Amplitude reflection and transmission coefficients of a stack.

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
        r, t amplitude coefficients.  r matches thinfilm.multilayer_stack_rt; t
        is normalized so the substrate-side tangential E equals t for a
        unit-amplitude incident wave (the p-polarization t differs from
        thinfilm's by a cos(theta0)/cos(theta_sub) substrate-column factor).

    """
    _, _, _, _, r, t, _ = _evaluate(stack, wvl, theta0, pol)
    return r, t


def internal_fields(stack, wvl, theta0, pol):
    """Tangential electric and magnetic field at every boundary.

    Boundary k lies between layer k and layer k+1 (boundary 0 is ambient/first
    layer, boundary N is last layer/substrate); the fields are continuous across
    each boundary so a single value per boundary is unambiguous.  Amplitudes are
    for a unit-amplitude incident wave; |E|^2 is the standing-wave intensity.

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
        E, H each of shape (N+1, *calc); the leading axis indexes boundaries
        from the ambient side.

    """
    *_, fields = _evaluate(stack, wvl, theta0, pol)
    return fields[..., 0], fields[..., 1]


def field_at_depth(stack, z, wvl, theta0, pol):
    """Tangential field at arbitrary depth(s) inside the stack.

    Depth is measured from the front boundary (z = 0 at the ambient/first-layer
    interface) increasing toward the substrate.  Within the layer containing z,
    the field is propagated from that layer's substrate-side boundary by a
    partial characteristic matrix, so the result is continuous with
    internal_fields at the boundaries.

    This sampler takes scalar wvl, theta0, and pol (one illumination condition)
    and an array of depths -- the usual mode for a standing-wave |E(z)|^2 plot
    or a needle z-sweep.

    Parameters
    ----------
    stack : Stack
    z : float or ndarray
        depth(s) into the stack, microns.
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
    """Reflectance, transmittance, and per-layer absorptance of a stack.

    The per-layer absorptance is the drop in the net power flux toward the
    substrate across the layer, recovered from the boundary fields -- the split
    r and t alone cannot give.  By construction R + sum(A) + T = 1, and for a
    lossless stack every A is zero.

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
        R, T, and A; A has shape (N, *calc) with one absorptance per layer,
        ambient side first.

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
