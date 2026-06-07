"""Needle optimization for coating synthesis."""

from prysm.conf import config
from prysm.mathops import np
from prysm.thinfilm import _cos_snell

from .stack import Stack, _resolve, _admittance, _char_matrix
from .diff import _dchar_dbeta
from .merit import as_merit
from .refine import refine


def _boundary_depths(stack):
    th = np.asarray(stack.thicknesses, dtype=config.precision)
    return np.concatenate([np.zeros(1, dtype=config.precision), np.cumsum(th)])


def _needle_P_for_sample(fwd, c_M, needle_material, z, Z):
    """P(z) contribution from one ForwardEval / assembled-matrix cotangent pair."""
    stack = fwd.stack
    wvl, theta0, pol = fwd.wvl, fwd.theta0, fwd.pol
    N = len(stack)
    calc_shape = np.shape(fwd.r)
    ndc = len(calc_shape)

    # thin-layer generator at this illumination (broadcast over calc)
    n0 = _resolve(stack.ambient_index, wvl)
    nn = _resolve(needle_material, wvl)
    cost_n = _cos_snell(n0, nn, theta0)
    eta_n = _admittance(nn, cost_n, pol)
    beta_dd_n = (2 * np.pi * nn * cost_n) / wvl
    eta_n_b = np.broadcast_to(eta_n + 0j, calc_shape)
    G = np.broadcast_to(beta_dd_n + 0j, calc_shape)[..., None, None] \
        * _dchar_dbeta(np.zeros(calc_shape), eta_n_b)

    # per-z host layer and partial matrices above / below z
    j = np.clip(np.searchsorted(Z, z, side='right') - 1, 0, N - 1)
    top_t = z - Z[j]
    bot_t = Z[j + 1] - z
    extra = (1,) * ndc
    top_b = top_t.reshape((-1,) + extra)
    bot_b = bot_t.reshape((-1,) + extra)

    dbdd = np.stack([np.broadcast_to(d + 0j, calc_shape) for d in fwd.dbeta_dd], 0)
    etas = np.stack([np.broadcast_to(e + 0j, calc_shape) for e in fwd.etas], 0)
    dbdd_j = dbdd[j]
    etas_j = etas[j]
    M_top = _char_matrix(dbdd_j * top_b, etas_j)
    M_bot = _char_matrix(dbdd_j * bot_b, etas_j)

    # broadcast end-cap identities to the calc shape before stacking.
    tshape = calc_shape + (2, 2)
    Lstack = np.stack([np.broadcast_to(Lk + 0j, tshape) for Lk in fwd.L], 0)
    Rstack = np.stack([np.broadcast_to(Rk + 0j, tshape) for Rk in fwd.R], 0)
    Lz = Lstack[j] @ M_top
    Rz = M_bot @ Rstack[j + 1]
    dM = Lz @ G[None] @ Rz

    contrib = np.real(np.sum(np.conj(c_M)[None] * dM, axis=(-2, -1)))
    if ndc:
        contrib = np.sum(contrib, axis=tuple(range(1, contrib.ndim)))
    return contrib


def needle_function(stack, targets, needle_material, z):
    """Merit derivative for inserting a needle material at depth z.

    Parameters
    ----------
    stack : Stack
    targets : MeritFunction, term, or sequence of terms
        reflectance / transmittance objective.
    needle_material : float, complex, or callable
        the candidate needle index (or dispersion callable).
    z : float or ndarray
        depth(s) into the stack, microns, measured from the ambient side.

    Returns
    -------
    ndarray
        P(z), same shape as z; negative values lower the merit.

    """
    merit = as_merit(targets)
    z = np.atleast_1d(np.asarray(z, dtype=config.precision))
    Z = _boundary_depths(stack)
    P = np.zeros(z.shape, dtype=config.precision)
    for term in merit.terms:
        for fwd, c_M in term.assembly_seeds(stack):
            P = P + _needle_P_for_sample(fwd, c_M, needle_material, z, Z)
    return P


def insert_needle(stack, z, material, thickness=1e-3, return_index=False):
    """Insert a layer at depth z, splitting the host layer.

    Parameters
    ----------
    stack : Stack
    z : float
        insertion depth, microns.
    material : float, complex, or callable
        needle index.
    thickness : float, optional
        seed thickness of the inserted needle, microns.
    return_index : bool, optional
        if True, also return the layer index of the inserted needle.

    Returns
    -------
    Stack
        the new stack, or (Stack, int) when return_index is True.

    """
    Z = _boundary_depths(stack)
    N = len(stack)
    if N == 0:
        raise ValueError('insert_needle requires at least one layer')
    total = float(Z[-1])
    z = float(z)
    if z < 0.0 or z > total:
        raise ValueError('z must lie within the coating stack')
    j = int(np.clip(np.searchsorted(Z, z, side='right') - 1, 0, N - 1))
    top_t = float(z - Z[j])
    bot_t = float(Z[j + 1] - z)

    idx = list(stack.indices)
    th = list(np.asarray(stack.thicknesses, dtype=config.precision))
    new_idx = idx[:j] + [idx[j], material, idx[j]] + idx[j + 1:]
    new_th = th[:j] + [top_t, float(thickness), bot_t] + th[j + 1:]
    inserted = Stack(new_idx, new_th, stack.substrate_index, stack.ambient_index)
    if return_index:
        return inserted, j + 1
    return inserted


def _same_material(a, b):
    if callable(a) or callable(b):
        return a is b
    return bool(np.isclose(a, b))


def cleanup(stack, prune_tol=2e-3, keep_indices=None):
    """Drop sub-tolerance layers and merge adjacent same-material layers.

    Parameters
    ----------
    stack : Stack
    prune_tol : float
        layers thinner than this (microns) are removed.
    keep_indices : sequence of int, optional
        layers exempt from pruning; used to protect a freshly inserted needle
        until the refine step has a chance to grow it.

    Returns
    -------
    Stack

    """
    idx = list(stack.indices)
    th = list(np.asarray(stack.thicknesses, dtype=config.precision))
    keep_indices = set(() if keep_indices is None else keep_indices)

    kept_idx = []
    kept_th = []
    for k, (i, t) in enumerate(zip(idx, th)):
        if t >= prune_tol or k in keep_indices:
            kept_idx.append(i)
            kept_th.append(t)

    midx = []
    mth = []
    for i, t in zip(kept_idx, kept_th):
        if midx and _same_material(midx[-1], i):
            mth[-1] = mth[-1] + t
        else:
            midx.append(i)
            mth.append(t)
    return Stack(midx, mth, stack.substrate_index, stack.ambient_index)


class NeedleResult:
    """Outcome of needle synthesis.

    Attributes
    ----------
    stack : Stack
        the synthesized stack.
    merit : float
        the merit value of the final stack.
    n_layers : int
        number of layers in the final stack.
    iterations : int
        number of needle insertions performed.
    success : bool
        True if the loop stopped because no insertion improved the merit
        (a stationary design), False if a budget or iteration cap was hit.

    """

    __slots__ = ('stack', 'merit', 'n_layers', 'iterations', 'success')

    def __init__(self, stack, merit, iterations, success):
        self.stack = stack
        self.merit = float(merit)
        self.n_layers = len(stack)
        self.iterations = int(iterations)
        self.success = bool(success)

    def __repr__(self):
        return (f'NeedleResult(merit={self.merit:.3e}, '
                f'n_layers={self.n_layers}, iterations={self.iterations}, '
                f'success={self.success})')


def synthesize(stack0, targets, materials, *, z_samples=240, max_layers=40,
               max_iters=30, tol=1e-9, prune_tol=2e-3, seed_thickness=1e-3,
               refine_kwargs=None):
    """Grow a multilayer design by repeated needle insertion and refinement.

    Parameters
    ----------
    stack0 : Stack
        starting design (one or two layers is typical).
    targets : MeritFunction, term, or sequence of terms
        reflectance / transmittance objective.
    materials : sequence
        material pool, as indices or dispersion callables.
    z_samples : int, optional
        depth-grid resolution for the P(z) sweep.
    max_layers : int, optional
        layer-count budget.
    max_iters : int, optional
        maximum needle insertions.
    tol : float, optional
        stationarity tolerance on min P(z).
    prune_tol : float, optional
        thin-layer prune threshold, microns.
    seed_thickness : float, optional
        seed thickness of each inserted needle, microns.
    refine_kwargs : dict, optional
        forwarded to refine.

    Returns
    -------
    NeedleResult

    """
    merit = as_merit(targets)
    materials = list(materials)
    if not materials:
        raise ValueError('materials pool is empty')
    refine_kwargs = dict(refine_kwargs or {})

    stack = refine(stack0, merit, **refine_kwargs).stack
    stationary = False
    iterations = 0
    for iterations in range(1, max_iters + 1):
        if len(stack) >= max_layers:
            break
        total = float(np.sum(np.asarray(stack.thicknesses)))
        if total <= 0 or len(stack) == 0:
            break
        z = np.linspace(0.0, total, z_samples)

        best_P = np.inf
        best_mat = None
        best_z = None
        for mat in materials:
            P = needle_function(stack, merit, mat, z)
            i = int(np.argmin(P))
            if P[i] < best_P:
                best_P = float(P[i])
                best_mat = mat
                best_z = float(z[i])

        if best_P >= -tol:
            stationary = True
            break

        stack, inserted_idx = insert_needle(
            stack, best_z, best_mat, thickness=seed_thickness, return_index=True)
        stack = refine(stack, merit, **refine_kwargs).stack
        cleaned = cleanup(stack, prune_tol=prune_tol,
                          keep_indices=[inserted_idx])
        if len(cleaned) == 0:
            stack = cleaned
            break
        if len(cleaned) != len(stack):
            stack = refine(cleaned, merit, **refine_kwargs).stack
        else:
            stack = cleaned

    return NeedleResult(stack, merit.value(stack), iterations, stationary)


__all__ = [
    'needle_function',
    'insert_needle',
    'cleanup',
    'synthesize',
    'NeedleResult',
]
