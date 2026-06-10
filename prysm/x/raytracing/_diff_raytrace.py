"""Forward-mode differential ray tracing."""

from prysm.conf import config
from prysm.mathops import np, row_dot

from .spencer_and_murty import (
    STYPE_REFLECT,
    STYPE_REFRACT,
    raytrace,
    valid_mask,
)
from ._line_math import normalize_vector
from .opt import hopkins_eic_closing, _chief_axis_perp_norm
from .analysis import (
    _pupil_center_chief_index,
    _filtered_chief_index,
    _apply_field_and_output,
)
from ._meta import object_space_index, image_space_index, system_stop_index


# ---------- broadcasting helpers --------------------------------------------

def _dot_nt(a, b):
    """Dot a nominal vector (N, 3) with a tangent vector (N, 3, P) -> (N, P)."""
    return np.sum(a[..., None] * b, axis=1)


def _matvec_t(R, b):
    """Apply a (3, 3) matrix to a tangent vector (N, 3, P) -> (N, 3, P)."""
    return np.einsum('ij,njp->nip', R, b)


def _dmatvec(Rdot, v):
    """Apply a tangent matrix (3, 3, P) to a nominal vector (N, 3) -> (N, 3, P)."""
    return np.einsum('ijp,nj->nip', Rdot, v)


def _dmatTvec(Rdot, v):
    """Apply the transpose of a tangent matrix (3,3,P) to (N,3) -> (N,3,P)."""
    return np.einsum('jip,nj->nip', Rdot, v)


# ---------- differential primitives (1:1 with spencer_and_murty) ------------

def d_transform_local(Reff, Q, P, S, Pdot, Sdot, Qdot, Rdot):
    """Differential of transform_to_local_coords.

    P0 = R (P - Q),  S_loc = R S.
    Pdot0 = R (Pdot - Qdot) + Rdot (P - Q);  Slocdot = R Sdot + Rdot S.
    """
    Pmq = P - Q
    P0 = (Reff @ Pmq.T).T
    S_loc = (Reff @ S.T).T
    P0dot = _matvec_t(Reff, Pdot - Qdot[None, :, :]) + _dmatvec(Rdot, Pmq)
    S_locdot = _matvec_t(Reff, Sdot) + _dmatvec(Rdot, S)
    return P0, S_loc, P0dot, S_locdot


def d_intersect(P0, S_loc, Q_loc, n_hat, P0dot, S_locdot,
                hessian, dsag_param, dgx_param, dgy_param):
    """Differential of the implicit ray/surface intersection.

    Parameters
    ----------
    P0, S_loc, Q_loc, n_hat : ndarray, (N, 3)
        local ray origin, direction, intersection point, unit normal.
    P0dot, S_locdot : ndarray, (N, 3, P)
        tangents of P0 and S_loc.
    hessian : tuple of ndarray
        (sag_xx, sag_xy, sag_yy), each (N,).
    dsag_param, dgx_param, dgy_param : ndarray, (N, P)
        explicit (parameter) partials of sag and its gradient at (X, Y).

    Returns
    -------
    dPj : ndarray, (N, 3, P)
        tangent of the local intersection point.
    dn_hat : ndarray, (N, 3, P)
        tangent of the local unit normal.

    """
    nz = n_hat[..., 2]
    g = n_hat / nz[:, None]                       # g_z == 1
    s_total = row_dot(Q_loc - P0, S_loc)
    g_dot_S = row_dot(g, S_loc)
    g_dot_P0dot = _dot_nt(g, P0dot)
    g_dot_Slocdot = _dot_nt(g, S_locdot)
    sdot = -(g_dot_P0dot + s_total[:, None] * g_dot_Slocdot
             - dsag_param) / g_dot_S[:, None]
    dPj = (P0dot + sdot[:, None, :] * S_loc[:, :, None]
           + s_total[:, None, None] * S_locdot)
    dXj = dPj[:, 0, :]
    dYj = dPj[:, 1, :]
    sxx, sxy, syy = hessian
    dgx = sxx[:, None] * dXj + sxy[:, None] * dYj + dgx_param
    dgy = sxy[:, None] * dXj + syy[:, None] * dYj + dgy_param
    dg = np.stack([-dgx, -dgy, np.zeros_like(dgx)], axis=1)
    n_dot_dg = _dot_nt(n_hat, dg)
    # divide by ||g|| == 1 / nz  ->  multiply by nz
    dn_hat = (dg - n_hat[:, :, None] * n_dot_dg[:, None, :]) * nz[:, None, None]
    return dPj, dn_hat


def d_refract(n, nprime, S_loc, n_hat, S_locdot, dn_hat, ndot_pre, ndot_post):
    """Differential of refract.

    S' = mu S + (sign(cosI) cosT - mu cosI) n_hat,  mu = n / n'.
    Index tangents enter via mu_dot = (ndot n' - n ndot') / n'^2.
    """
    cosI = row_dot(n_hat, S_loc)
    dcosI = _dot_nt(S_loc, dn_hat) + _dot_nt(n_hat, S_locdot)
    mu = n / nprime
    mu_dot = (ndot_pre * nprime - n * ndot_post) / (nprime * nprime)  # (P,)
    one_minus = 1.0 - cosI * cosI
    sinT2 = mu * mu * one_minus
    with np.errstate(invalid='ignore'):
        cosT = np.sqrt(1.0 - sinT2)
    dsinT2 = (2.0 * mu * mu_dot[None, :] * one_minus[:, None]
              - 2.0 * mu * mu * cosI[:, None] * dcosI)
    # near the critical angle cosT -> 0 and dcosT = -dsinT2 / (2 cosT) blows up;
    # an exact-TIR ray already carries cosT = NaN (and a NaN forward Sprime that
    # the analysis layer filters out).  Zero the non-finite derivative rather
    # than let inf/NaN silently poison an otherwise-valid sensitivity column.
    with np.errstate(divide='ignore', invalid='ignore'):
        dcosT = -dsinT2 / (2.0 * cosT[:, None])
    dcosT[~np.isfinite(dcosT)] = 0.0
    sign = np.sign(cosI)
    factor = sign * cosT - mu * cosI
    dfactor = (sign[:, None] * dcosT - mu * dcosI
               - mu_dot[None, :] * cosI[:, None])
    Sprime = mu * S_loc + factor[:, None] * n_hat
    dSprime = (S_loc[:, :, None] * mu_dot[None, None, :] + mu * S_locdot
               + n_hat[:, :, None] * dfactor[:, None, :]
               + factor[:, None, None] * dn_hat)
    return Sprime, dSprime


def d_reflect(S_loc, n_hat, S_locdot, dn_hat):
    """Differential of reflect.  S' = S - 2 (S . n_hat) n_hat."""
    cosI = row_dot(S_loc, n_hat)
    dcosI = _dot_nt(n_hat, S_locdot) + _dot_nt(S_loc, dn_hat)
    Sprime = S_loc - 2.0 * cosI[:, None] * n_hat
    dSprime = S_locdot - 2.0 * (n_hat[:, :, None] * dcosI[:, None, :]
                                + cosI[:, None, None] * dn_hat)
    return Sprime, dSprime


def d_transform_global(Reff, Q, Q_loc, Sprime, dPj, dSprime, Qdot, Rdot):
    """Differential of transform_to_global_coords (inverse of d_transform_local).

    Pjp1 = R^T Q_loc + Q.
    dPjp1 = R^T dPj + (dR)^T Q_loc + Qdot;  dSjp1 = R^T dSprime + (dR)^T Sprime.
    """
    Rt = Reff.T
    dPjp1 = (_matvec_t(Rt, dPj) + _dmatTvec(Rdot, Q_loc) + Qdot[None, :, :])
    dSjp1 = (_matvec_t(Rt, dSprime) + _dmatTvec(Rdot, Sprime))
    return dPjp1, dSjp1


def d_opl_segment(n_pre, n_pre_dot, seg, dseg):
    """Differential of an OPL segment L = n_pre ||seg||.

    dL = n_pre_dot ||seg|| + n_pre (seg . dseg) / ||seg||.
    """
    seg_len = np.sqrt(row_dot(seg, seg))
    return (n_pre_dot[None, :] * seg_len[:, None]
            + n_pre * _dot_nt(seg, dseg) / seg_len[:, None])


# ---------- perturbation seeds ----------------------------------------------

# rotation generators dR/dangle at angle 0 (radians), matching
# coordinates.make_rotation_matrix's Rx/Ry/Rz blocks.
_GEN = {
    'x': np.asarray([[0., 0., 0.], [0., 0., -1.], [0., 1., 0.]]),
    'y': np.asarray([[0., 0., 1.], [0., 0., 0.], [-1., 0., 0.]]),
    'z': np.asarray([[0., -1., 0.], [1., 0., 0.], [0., 0., 0.]]),
}


class DiffSeed:
    """Tangent seed for one perturbation parameter.

    Attributes
    ----------
    pose : dict
        {surface_index: (Qdot, Rdot)} vertex and rotation tangents.
    shape : tuple or None
        (surface_index, name) of a shape DOF.
    sag_partials : tuple or None
        (surface_index, fn) for explicit sag/sag-gradient partials.
    index : tuple or None
        (surface_index, value) tangent of the medium after that surface.
    name : str
        label for reporting.

    """

    __slots__ = ('pose', 'shape', 'sag_partials', 'index', 'name')

    def __init__(self, pose=None, shape=None, sag_partials=None, index=None,
                 name=''):
        self.pose = dict(pose) if pose else {}
        self.shape = shape
        self.sag_partials = sag_partials
        self.index = index
        self.name = str(name)

    def __repr__(self):
        return f'DiffSeed(name={self.name!r})'


def seed_curvature(surface, name='c'):
    """Seed for a curvature (DLR) tolerance on a surface's shape DOF 'c'."""
    return DiffSeed(shape=(surface, 'c'), name=name)


def seed_conic(surface, name='k'):
    """Seed for a conic-constant tolerance on a surface's shape DOF 'k'."""
    return DiffSeed(shape=(surface, 'k'), name=name)


def seed_shape_param(surface, param_name, name=None):
    """Seed for an arbitrary scalar shape DOF (freeform coefficient, etc.)."""
    return DiffSeed(shape=(surface, param_name), name=name or param_name)


def seed_irregularity(surface, n, m, normalization_radius, *, norm=True,
                      name=None):
    """Seed for a Zernike surface-irregularity tolerance.

    """
    from .sags import zernike_irregularity_partials  # local: keep the kernel
    # module's import surface free of the optional irregularity helper

    def partials(x, y):
        return zernike_irregularity_partials(n, m, x, y, normalization_radius,
                                             norm=norm)

    return DiffSeed(sag_partials=(surface, partials),
                    name=name or f'irr_Z{n}_{m}')


def seed_decenter(surface, axis, name=None):
    """Seed for a decenter tolerance: vertex moves along axis."""
    idx = {'x': 0, 'y': 1, 'z': 2}[axis]
    q = np.zeros(3, dtype=config.precision)
    q[idx] = 1.0
    return DiffSeed(pose={surface: (q, None)}, name=name or f'decenter_{axis}')


def seed_despace(surfaces, name='despace'):
    """Seed for a despace/thickness tolerance."""
    q_plus = np.array([0., 0., 1.], dtype=config.precision)
    pose = {}
    for sidx, sgn in surfaces:
        pose[sidx] = (sgn * q_plus, None)
    return DiffSeed(pose=pose, name=name)


def seed_tilt(surface, axis, R_nominal=None, name=None):
    """Seed for a tilt (BTX/BTY) tolerance about a local axis, in radians.

    R_total = R_nominal @ R_tilt(a), so Rdot = R_nominal @ generator.  Pass
    R_nominal=None for an untilted surface (identity).
    """
    G = _GEN[axis].astype(config.precision)
    Rdot = G if R_nominal is None else (np.asarray(R_nominal, dtype=config.precision) @ G)
    q = np.zeros(3, dtype=config.precision)
    return DiffSeed(pose={surface: (q, Rdot)}, name=name or f'tilt_{axis}')


def seed_index(surface, name='index'):
    """Seed for an index (DLN) tolerance on the medium following a surface."""
    return DiffSeed(index=(surface, 1.0), name=name)


# ---------- map tolerance.Perturbation -> DiffSeed --------------------------

def _is_surface_row(row):
    """True for a SurfaceRow (emits a Surface); False for a CoordBreak."""
    return hasattr(row, 'build_shape')


def _row_to_surface_index(rows, row_idx):
    """Compiled-surface index of the SurfaceRow at rows[row_idx].

    CoordBreak rows emit no Surface, so a surface row's index in the compiled
    list is the count of surface rows preceding it.
    """
    count = 0
    for i, row in enumerate(rows):
        if i == row_idx:
            if not _is_surface_row(row):
                raise ValueError(
                    f'row {row_idx} is a coordinate break, not a surface, and '
                    'has no shape DOF to differentiate')
            return count
        if _is_surface_row(row):
            count += 1
    raise IndexError(f'row index {row_idx} out of range ({len(rows)} rows)')


def _shape_dof_name(row, off):
    """Resolve a shape-DOF offset to its parameter name on a SurfaceRow."""
    for key, (start, length) in row.key_offsets.items():
        if start <= off < start + length:
            if length == 1:
                return key
            raise NotImplementedError(
                f'tolerance on element {off - start} of the vector DOF {key!r} '
                'is not mapped to a differential seed; freeform-coefficient '
                '(CYN/CYD) sensitivities use the FD sensitivity_table fallback')
    raise KeyError(f'no shape DOF at offset {off}')


def _as_rot(R):
    """A concrete 3x3 rotation (identity when the surface stores R=None)."""
    return _eye3() if R is None else np.asarray(R, dtype=config.precision)


def _pose_tangents_via_layout(perturbation, h):
    """Per-surface (Qdot, Rdot) of the compiled layout wrt one DOF."""
    ld = perturbation.lensdata
    nominal = perturbation.nominal
    try:
        perturbation.set(nominal + h)
        sp = ld.to_surfaces()
        Pp = [np.array(s.P, dtype=config.precision) for s in sp]
        Rp = [_as_rot(s.R) for s in sp]
        perturbation.set(nominal - h)
        sm = ld.to_surfaces()
        Pm = [np.array(s.P, dtype=config.precision) for s in sm]
        Rm = [_as_rot(s.R) for s in sm]
    finally:
        perturbation.reset()

    inv2h = 1.0 / (2.0 * h)
    pose = {}
    for j in range(len(Pp)):
        Qdot = (Pp[j] - Pm[j]) * inv2h
        Rdot = (Rp[j] - Rm[j]) * inv2h
        r_nz = bool(np.any(Rdot))
        if bool(np.any(Qdot)) or r_nz:
            pose[j] = (Qdot, Rdot if r_nz else None)
    return pose


def seed_from_perturbation(perturbation, *, pose_step=1e-6):
    """Build the DiffSeed matching a tolerance.Perturbation on a LensData.

    Shape tangents use sag_param_partials; pose tangents are obtained by
    finite-differencing the compiled layout.

    """
    group, row_idx, off = perturbation.slot
    ld = perturbation.lensdata
    name = perturbation.name or f'{group}{row_idx}'

    shape = None
    if group == 'shape':
        surf_idx = _row_to_surface_index(ld.rows, row_idx)
        shape = (surf_idx, _shape_dof_name(ld.rows[row_idx], off))

    pose = _pose_tangents_via_layout(perturbation, pose_step)
    return DiffSeed(pose=pose, shape=shape, name=name)


def seeds_from_perturbations(perturbations, *, pose_step=1e-6):
    """Map a sequence of tolerance.Perturbations to DiffSeeds (one per).

    The returned seeds define the trailing parameter axis of
    raytrace_with_tangents / wavefront_with_tangents in the given order.  See
    seed_from_perturbation for the per-category mapping and its limitations.
    """
    return [seed_from_perturbation(p, pose_step=pose_step)
            for p in perturbations]


def _assemble_seeds(n_surfaces, seeds, n_params):
    """Bucket the per-parameter seeds into per-surface explicit-tangent arrays."""
    dt = config.precision
    Qdot = [np.zeros((3, n_params), dtype=dt) for _ in range(n_surfaces)]
    Rdot = [np.zeros((3, 3, n_params), dtype=dt) for _ in range(n_surfaces)]
    nprimedot = [np.zeros(n_params, dtype=dt) for _ in range(n_surfaces)]
    shape_params = [[] for _ in range(n_surfaces)]  # list of (p, name)
    sag_partial_fns = [[] for _ in range(n_surfaces)]  # list of (p, fn)
    for p, seed in enumerate(seeds):
        for sidx, (Qd, Rd) in seed.pose.items():
            Qdot[sidx][:, p] = Qdot[sidx][:, p] + np.asarray(Qd, dtype=dt)
            if Rd is not None:
                Rdot[sidx][:, :, p] = Rdot[sidx][:, :, p] + np.asarray(Rd, dtype=dt)
        if seed.shape is not None:
            sidx, pname = seed.shape
            shape_params[sidx].append((p, pname))
        if seed.sag_partials is not None:
            sidx, fn = seed.sag_partials
            sag_partial_fns[sidx].append((p, fn))
        if seed.index is not None:
            sidx, val = seed.index
            nprimedot[sidx][p] = nprimedot[sidx][p] + float(val)
    return Qdot, Rdot, nprimedot, shape_params, sag_partial_fns


def _paraxial_matmul_tangent(A, Adot, M, Mdot):
    """Compose ABCD nominal/tangent matrices with a trailing parameter axis."""
    Mdot_next = (np.einsum('ijp,jk->ikp', Adot, M)
                 + np.einsum('ij,jkp->ikp', A, Mdot))
    return A @ M, Mdot_next


def _paraxial_walk_matrix_tangent(surfaces, wvl, n_start, n_start_dot,
                                  zdot_s, cdot_s, nprimedot_s, *,
                                  start_index=0, end_index=None,
                                  include_end_surface=True):
    """ABCD walk plus derivatives with respect to DiffSeed parameters."""
    from .paraxial import _paraxial_curvature

    n_params = n_start_dot.shape[0]
    M = np.eye(2, dtype=config.precision)
    Mdot = np.zeros((2, 2, n_params), dtype=config.precision)
    n = float(n_start)
    ndot = np.asarray(n_start_dot, dtype=config.precision).copy()
    z_prev = float(surfaces[start_index].P[2])
    if end_index is None:
        end_index = len(surfaces) - 1

    for k in range(start_index, len(surfaces)):
        if k > end_index:
            break
        surf = surfaces[k]
        if k > start_index:
            t = float(surf.P[2]) - z_prev
            tdot = zdot_s[k] - zdot_s[k - 1]
            T = np.array([[1.0, t / n], [0.0, 1.0]], dtype=config.precision)
            Tdot = np.zeros((2, 2, n_params), dtype=config.precision)
            Tdot[0, 1] = tdot / n - t * ndot / (n * n)
            M, Mdot = _paraxial_matmul_tangent(T, Tdot, M, Mdot)

        if include_end_surface or k != end_index:
            c = _paraxial_curvature(surf)
            cdot = cdot_s[k]
            if surf.typ == STYPE_REFRACT:
                nprime = float(surf.material.n(wvl))
                nprime_dot = nprimedot_s[k]
            elif surf.typ == STYPE_REFLECT:
                nprime = -n
                nprime_dot = -ndot
            else:
                z_prev = float(surf.P[2])
                continue

            power = (nprime - n) * c
            power_dot = (nprime_dot - ndot) * c + (nprime - n) * cdot
            R = np.array([[1.0, 0.0], [-power, 1.0]], dtype=config.precision)
            Rdot = np.zeros((2, 2, n_params), dtype=config.precision)
            Rdot[1, 0] = -power_dot
            M, Mdot = _paraxial_matmul_tangent(R, Rdot, M, Mdot)
            n = float(nprime)
            ndot = np.asarray(nprime_dot, dtype=config.precision).copy()

        z_prev = float(surf.P[2])

    return M, n, Mdot, ndot


def paraxial_exit_pupil_z_tangents(prescription, wvl, seeds, *,
                                   stop_index=None):
    """Derivative of first_order(...).xp_z with respect to DiffSeed entries."""
    from .paraxial import _first_order_surfaces

    seeds = list(seeds)
    n_params = len(seeds)
    stop_index = system_stop_index(prescription, stop_index)
    if stop_index is None:
        return np.zeros(n_params, dtype=config.precision)

    surfaces = _first_order_surfaces(prescription)
    n_surfaces = len(surfaces)
    k = int(stop_index)
    if k < 0 or k >= n_surfaces:
        raise IndexError(
            f'stop_index {k} out of range for prescription of length '
            f'{n_surfaces}'
        )

    Qdot_s, _, nprimedot_s, shape_params, _ = _assemble_seeds(
        n_surfaces, seeds, n_params)
    zdot_s = np.array([Qdot[2] for Qdot in Qdot_s], dtype=config.precision)
    cdot_s = np.zeros((n_surfaces, n_params), dtype=config.precision)
    for sidx, entries in enumerate(shape_params):
        for p, pname in entries:
            if pname in ('c', 'c_y'):
                cdot_s[sidx, p] = cdot_s[sidx, p] + 1.0

    n_object = object_space_index(surfaces, wvl)
    ndot_object = np.zeros(n_params, dtype=config.precision)
    _, n_at_stop, _, ndot_at_stop = _paraxial_walk_matrix_tangent(
        surfaces, wvl, n_object, ndot_object, zdot_s, cdot_s, nprimedot_s,
        end_index=k, include_end_surface=False)
    M_from_stop, n_image, Mdot_from_stop, ndot_image = \
        _paraxial_walk_matrix_tangent(
            surfaces, wvl, n_at_stop, ndot_at_stop, zdot_s, cdot_s,
            nprimedot_s, start_index=k)

    B = float(M_from_stop[0, 1])
    D = float(M_from_stop[1, 1])
    if abs(D) < 1e-30:
        return np.zeros(n_params, dtype=config.precision)
    Bdot = Mdot_from_stop[0, 1]
    Ddot = Mdot_from_stop[1, 1]
    xp_distance_dot = (
        -(Bdot * n_image + B * ndot_image) / D
        + B * n_image * Ddot / (D * D)
    )
    return zdot_s[-1] + xp_distance_dot


# ---------- the differential trace ------------------------------------------

class DiffTraceResult:
    """Nominal trace plus the propagated tangent bundle.

    Attributes
    ----------
    trace : RayTraceResult
        the authoritative nominal trace (P, S, OPL, status histories).
    Pdot, Sdot : ndarray, (jj+1, N, 3, P)
        position / direction tangent histories, parallel to trace.P / trace.S.
    Ldot : ndarray, (jj+1, N, P)
        per-segment OPL tangent history, parallel to trace.OPL (Ldot[0] == 0).
    n_params : int

    """

    __slots__ = ('trace', 'Pdot', 'Sdot', 'Ldot', 'n_params')

    def __init__(self, trace, Pdot, Sdot, Ldot):
        self.trace = trace
        self.Pdot = Pdot
        self.Sdot = Sdot
        self.Ldot = Ldot
        self.n_params = Pdot.shape[-1]

    @property
    def P(self):
        return self.trace.P

    @property
    def S(self):
        return self.trace.S

    @property
    def OPL(self):
        return self.trace.OPL

    @property
    def status(self):
        return self.trace.status

    def __repr__(self):
        return (f'DiffTraceResult(N_rays={self.Pdot.shape[1]}, '
                f'N_surfaces={self.Pdot.shape[0] - 1}, '
                f'n_params={self.n_params})')


def _eye3():
    return np.eye(3, dtype=config.precision)


def _reject_gratings(surfaces):
    """Refuse to differentiate a prescription containing a grating surface.

    The forward (d_*) and reverse (adj_*) Spencer & Murty primitives have no
    grating term -- a grating surface traces as a plain mirror/lens in the AD
    stacks, silently yielding wrong derivatives.  Fail loudly until a grating
    diffraction primitive is added on both sides.
    """
    if any(getattr(s, 'grating', None) is not None for s in surfaces):
        raise NotImplementedError(
            'differential/adjoint raytrace does not model grating diffraction; '
            'remove the grating or add a grating AD primitive before tracing '
            'tangents/cotangents through it.')


def raytrace_with_tangents(surfaces, P, S, wvl, seeds, tol_sag=None):
    """Trace (P, S) and propagate the tangent bundle for every seed.

    Parameters
    ----------
    surfaces : sequence of Surface
        the compiled prescription.
    P, S : ndarray, (N, 3) or (3,)
        launch positions and direction cosines.
    wvl : float
        wavelength, microns.
    seeds : sequence of DiffSeed
        one per perturbation parameter; defines the trailing parameter axis.
    tol_sag : float, optional
        Newton convergence tolerance, forwarded to the nominal trace.

    Returns
    -------
    DiffTraceResult

    """
    _reject_gratings(surfaces)
    seeds = list(seeds)
    n_params = len(seeds)
    P = np.asarray(P)
    S = np.asarray(S)
    if P.ndim == 1:
        P = P[None, :]
        S = S[None, :]
    P = P.astype(config.precision)
    S = S.astype(config.precision)
    n_rays = P.shape[0]
    jj = len(surfaces)

    # keep the per-surface Interaction objects so the nominal local intersection
    # and normal come straight off the forward trace -- no second Newton solve.
    trace = raytrace(surfaces, P, S, wvl, tol_sag=tol_sag, keep_intermediates=True)

    Qdot_s, Rdot_s, nprimedot_s, shape_params, sag_partial_fns = _assemble_seeds(
        jj, seeds, n_params)

    Pdot_hist = np.zeros((jj + 1, n_rays, 3, n_params), dtype=config.precision)
    Sdot_hist = np.zeros((jj + 1, n_rays, 3, n_params), dtype=config.precision)
    Ldot_hist = np.zeros((jj + 1, n_rays, n_params), dtype=config.precision)

    Pdot = Pdot_hist[0]
    Sdot = Sdot_hist[0]
    # launch medium is intrinsic to the surfaces (leading eval object material,
    # else air); no seed perturbs the object medium, so its tangent is zero.
    nj = float(object_space_index(surfaces, wvl))
    nj_dot = np.zeros(n_params, dtype=config.precision)

    Pdot_prev = Pdot
    for j, surf in enumerate(surfaces):
        Reff = _eye3() if surf.R is None else np.asarray(surf.R, dtype=config.precision)
        Q = np.asarray(surf.P, dtype=config.precision)
        Pj_prev = trace.P[j]                # nominal global position (authoritative)
        Pj_cur = trace.P[j + 1]
        Qdot_j = Qdot_s[j]
        Rdot_j = Rdot_s[j]

        # Step I: to local (P0/S_loc here are bit-identical to inter.P0/S_loc)
        P0, S_loc, P0dot, S_locdot = d_transform_local(
            Reff, Q, Pj_prev, trace.S[j], Pdot, Sdot, Qdot_j, Rdot_j)

        # nominal local intersection + normal, captured from the forward trace
        inter = trace.intermediates[j]
        Q_loc, n_hat = inter.Q_loc, inter.n_hat

        # shape derivatives at the intersection
        Xj = Q_loc[..., 0]
        Yj = Q_loc[..., 1]
        hessian = surf.shape.sag_hessian(Xj, Yj)
        dsag_param = np.zeros((n_rays, n_params), dtype=config.precision)
        dgx_param = np.zeros((n_rays, n_params), dtype=config.precision)
        dgy_param = np.zeros((n_rays, n_params), dtype=config.precision)
        for p, pname in shape_params[j]:
            sag_t, gx_t, gy_t = surf.shape.sag_param_partials(Xj, Yj, pname)
            dsag_param[:, p] = sag_t
            dgx_param[:, p] = gx_t
            dgy_param[:, p] = gy_t
        # explicit (non-DOF) sag partials, e.g. a Zernike irregularity term
        for p, fn in sag_partial_fns[j]:
            sag_t, gx_t, gy_t = fn(Xj, Yj)
            dsag_param[:, p] = dsag_param[:, p] + sag_t
            dgx_param[:, p] = dgx_param[:, p] + gx_t
            dgy_param[:, p] = dgy_param[:, p] + gy_t

        # Step II: intersect
        dPj, dn_hat = d_intersect(P0, S_loc, Q_loc, n_hat, P0dot, S_locdot,
                                  hessian, dsag_param, dgx_param, dgy_param)

        # Step III: bend
        if surf.typ == STYPE_REFRACT:
            nprime = float(surf.material.n(wvl))
            nprimedot = nprimedot_s[j]
            Sprime, dSprime = d_refract(nj, nprime, S_loc, n_hat,
                                        S_locdot, dn_hat, nj_dot, nprimedot)
            n_post, n_post_dot = nprime, nprimedot
        elif surf.typ == STYPE_REFLECT:
            Sprime, dSprime = d_reflect(S_loc, n_hat, S_locdot, dn_hat)
            n_post, n_post_dot = nj, nj_dot
        else:  # eval / no interaction
            Sprime, dSprime = S_loc, S_locdot
            n_post, n_post_dot = nj, nj_dot

        # Step IV: to global
        dPjp1, dSjp1 = d_transform_global(Reff, Q, Q_loc, Sprime,
                                          dPj, dSprime, Qdot_j, Rdot_j)

        # Step V: OPL segment (index preceding surface j is nj / nj_dot)
        seg = Pj_cur - Pj_prev
        dseg = dPjp1 - Pdot_prev
        Ldot_hist[j + 1] = d_opl_segment(nj, nj_dot, seg, dseg)

        Pdot_hist[j + 1] = dPjp1
        Sdot_hist[j + 1] = dSjp1
        Pdot, Sdot = dPjp1, dSjp1
        Pdot_prev = dPjp1
        if surf.typ == STYPE_REFRACT:
            nj, nj_dot = n_post, n_post_dot

    return DiffTraceResult(trace, Pdot_hist, Sdot_hist, Ldot_hist)


# ---------- dOPD / dtau : differential of the reference-sphere OPD ----------

def _dot_vt(u, Vt):
    """Dot a nominal 3-vector u (3,) with a tangent 3-vector Vt (3, P) -> (P,)."""
    return np.sum(u[:, None] * Vt, axis=0)


def d_closest_point_on_axis(P, S, Pdot, Sdot, axis_point, axis_dir):
    """Exit-pupil point on the optical axis and its tangent.

    Returns (P_xp (3,), P_xp_dot (3, P)).
    """
    A = np.asarray(P, dtype=config.precision)
    Sc = np.asarray(S, dtype=config.precision)
    B = np.asarray(axis_point, dtype=config.precision)
    Sa = normalize_vector(np.asarray(axis_dir, dtype=config.precision), axis=-1)

    w = A - B
    wdot = Pdot
    a = np.dot(Sc, Sc)
    adot = 2.0 * _dot_vt(Sc, Sdot)
    b = np.dot(Sc, Sa)
    bdot = _dot_vt(Sa, Sdot)
    c = np.dot(Sa, Sa)
    d = np.dot(Sc, w)
    ddot = _dot_vt(w, Sdot) + _dot_vt(Sc, wdot)
    e = np.dot(Sa, w)
    edot = _dot_vt(Sa, wdot)
    denom = a * c - b * b
    denomdot = adot * c - 2.0 * b * bdot
    num = a * e - b * d
    numdot = adot * e + a * edot - bdot * d - b * ddot
    t = num / denom
    tdot = (numdot * denom - num * denomdot) / (denom * denom)
    P_xp = B + t * Sa
    P_xp_dot = Sa[:, None] * tdot[None, :]
    return P_xp, P_xp_dot


def d_eic_closing(P, S, Pdot, Sdot, C, Cdot, kappa, kappa_dot):
    """Tangent of the determinate EIC closing segment s_tilde.

    Forward-mode derivative of opt.hopkins_eic_closing's per-ray segment
    s_tilde = -b - kappa m / (1 + sqrt(1 + kappa^2 m)), b = S.(P - C),
    m = b^2 - |P - C|^2, with respect to the ray (P, S), the reference center
    C, and the reference-sphere curvature kappa = 1/R.  The determinate
    (cancellation-free) replacement for the reference-sphere segment tangent.

    Returns
    -------
    sdot : ndarray, (N, Pp)
        per-ray, per-parameter tangent of the closing segment.

    """
    r = P - C[None, :]
    rdot = Pdot - Cdot[None, :, :]
    b = row_dot(S, r)
    bdot = _dot_nt(r, Sdot) + _dot_nt(S, rdot)
    rr = row_dot(r, r)
    rrdot = 2.0 * _dot_nt(r, rdot)
    m = b * b - rr
    mdot = 2.0 * b[:, None] * bdot - rrdot
    k = float(kappa)
    disc = 1.0 + k * k * m
    disc[disc < 0] = 0.0
    discdot = 2.0 * k * kappa_dot[None, :] * m[:, None] + k * k * mdot
    w = np.sqrt(disc)
    wsafe = np.where(w == 0, 1.0, w)
    wdot = discdot / (2.0 * wsafe[:, None])
    g = k * m
    gdot = kappa_dot[None, :] * m[:, None] + k * mdot
    h = 1.0 + w
    # s = -b - g / h;  h_dot = w_dot
    sdot = -bdot - (gdot * h[:, None] - g[:, None] * wdot) / (h[:, None] ** 2)
    return sdot


def wavefront_with_tangents(surfaces, P, S, wavelength, seeds, *,
                            chief_index=None,
                            axis_point=None, axis_dir=None, P_xp=None,
                            field=None, output='length'):
    """OPD and per-seed OPD tangents on the chief reference sphere.

    Returns
    -------
    opd : ndarray, (Nvalid,)
        nominal OPD, chief == 0.
    x_pupil, y_pupil : ndarray, (Nvalid,)
        launch (x, y) pupil coordinates (chief-relative).
    dW : ndarray, (Nvalid, P)
        per-tolerance wavefront-derivative maps, column p = dOPD/dtau_p.

    """
    P = np.asarray(P).astype(config.precision)
    S = np.asarray(S).astype(config.precision)
    n_object = object_space_index(surfaces, wavelength)
    res = raytrace_with_tangents(surfaces, P, S, wavelength, seeds)
    trace = res.trace

    if chief_index is None:
        chief_index = _pupil_center_chief_index(P)
    valid = valid_mask(trace.status, trace.P[-1])
    if not valid[chief_index]:
        raise ValueError('chief ray is invalid; cannot define reference sphere')

    P_chief_final = trace.P[-1, chief_index]
    S_chief_final = trace.S[-1, chief_index]
    Pdot_chief = res.Pdot[-1, chief_index]   # (3, P)
    Sdot_chief = res.Sdot[-1, chief_index]   # (3, P)

    C = P_chief_final
    Cdot = Pdot_chief
    if P_xp is None:
        if axis_point is None:
            axis_point = np.zeros(3, dtype=config.precision)
        if axis_dir is None:
            axis_dir = np.array([0., 0., 1.], dtype=config.precision)
        if _chief_axis_perp_norm(S_chief_final, axis_dir) < 1e-6:
            raise ValueError(
                'cannot locate the exit pupil from a near-axial chief ray; '
                'pass P_xp to anchor the reference sphere'
            )
        P_xp, P_xp_dot = d_closest_point_on_axis(
            P_chief_final, S_chief_final, Pdot_chief, Sdot_chief,
            axis_point, axis_dir)
    else:
        P_xp = np.asarray(P_xp, dtype=config.precision)
        P_xp_dot = np.zeros((3, res.n_params), dtype=config.precision)
    delta = P_xp - C
    delta_dot = P_xp_dot - Cdot
    R = float(np.sqrt(np.sum(delta * delta)))
    if R <= 1e-12:
        raise ValueError(
            'reference-sphere radius is degenerate; pass a nondegenerate P_xp'
        )
    Rdot = _dot_vt(delta, delta_dot) / R
    # reference-sphere curvature kappa = 1/R is the closing's determinate handle
    kappa = 1.0 / R
    kappa_dot = -Rdot / (R * R)

    filtered_chief = _filtered_chief_index(valid, chief_index)

    n_image = image_space_index(surfaces, wavelength, fallback=n_object)
    # nominal value via the determinate EIC closing -- identical to
    # analysis.wavefront, so the nominal cannot drift between the two paths.
    opd = hopkins_eic_closing(trace.P[:, valid], trace.S[:, valid],
                              trace.OPL[:, valid],
                              center=P_chief_final, curvature=kappa,
                              n_image=n_image, chief_index=filtered_chief)

    P_last_f = trace.P[-1, valid]
    S_last_f = trace.S[-1, valid]
    Pdot_last_f = res.Pdot[-1][valid]
    Sdot_last_f = res.Sdot[-1][valid]
    Ldot_total_f = res.Ldot.sum(axis=0)[valid]

    sdot = d_eic_closing(P_last_f, S_last_f,
                         Pdot_last_f, Sdot_last_f,
                         C, Cdot, kappa, kappa_dot)
    opl_total_dot = Ldot_total_f + n_image * sdot
    opd_dot = opl_total_dot - opl_total_dot[filtered_chief][None, :]

    x_pupil = P[valid, 0] - P[chief_index, 0]
    y_pupil = P[valid, 1] - P[chief_index, 1]

    # field tilt and length/waves scaling are tau-independent factors, so the
    # same scale that converts opd converts the tangent maps (shared with
    # analysis.wavefront so the nominal opd cannot drift between the two).
    opd, scale = _apply_field_and_output(opd, x_pupil, y_pupil, field, output,
                                         wavelength)
    dW = opd_dot * scale

    return opd, x_pupil, y_pupil, dW


__all__ = [
    'd_transform_local',
    'd_intersect',
    'd_refract',
    'd_reflect',
    'd_transform_global',
    'd_opl_segment',
    'd_closest_point_on_axis',
    'd_eic_closing',
    'DiffSeed',
    'DiffTraceResult',
    'raytrace_with_tangents',
    'wavefront_with_tangents',
    'seed_curvature',
    'seed_conic',
    'seed_shape_param',
    'seed_irregularity',
    'seed_decenter',
    'seed_despace',
    'seed_tilt',
    'seed_index',
    'seed_from_perturbation',
    'seeds_from_perturbations',
]
