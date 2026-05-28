"""Manual forward-mode differential ray trace (tangent bundle through the kernel).

This is the engine behind first-order wavefront-differential tolerancing:
trace the nominal system once and propagate, alongside the ray state, a tangent
(Pdot, Sdot, Ldot) = d(P, S, OPL)/dtau for every perturbation parameter at once.
The tangents carry a trailing parameter axis, so one nominal trace yields every
tolerance's sensitivity.

The primitives mirror spencer_and_murty 1:1 (d_transform_local, d_intersect,
d_refract, d_reflect, d_transform_global, d_opl_segment) and are exact
closed-form derivatives, not autograd.  Each was validated against central
finite differences of the actual kernel.

Layout convention (N rays, P parameters):
    nominal vector   (N, 3)        tangent vector   (N, 3, P)
    nominal scalar   (N,)          tangent scalar   (N, P)
    pose tangent     Qdot (3, P), Rdot (3, 3, P)   (ray-independent)
    index tangent    (P,)          (ray-independent)

Shape derivatives needed: the sag Hessian (sag_hessian) and the explicit
parameter partials (sag_param_partials) of the perturbed surface.  The
unnormalized normal g and its magnitude come for free from the nominal unit
normal (g = n_hat / n_hat_z, ||g|| = 1 / n_hat_z), so the intersection and
normal tangents work for any shape that supplies a Hessian.
"""

from prysm.conf import config
from prysm.mathops import np, row_dot

from .spencer_and_murty import (
    DEFAULT_TOL_SAG,
    STYPE_REFLECT,
    STYPE_REFRACT,
    raytrace,
)
from ._line_math import normalize_vector
from .opt import opd_from_raytrace, _valid_mask
from .analysis import (
    _pupil_center_chief_index,
    _filtered_chief_index,
    _apply_field_and_output,
)


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

    Implicit equation F = Z - sag(X, Y; theta) = 0 along Pj = P0 + s S_loc.
    The unnormalized normal g = (-sag_x, -sag_y, 1) and ||g|| are recovered
    from the nominal unit normal (g = n_hat / n_hat_z).  The implicit function
    theorem gives sdot; the Hessian + explicit shape partials give dn_hat.

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
    cosT = np.sqrt(1.0 - sinT2)
    dsinT2 = (2.0 * mu * mu_dot[None, :] * one_minus[:, None]
              - 2.0 * mu * mu * cosI[:, None] * dcosI)
    dcosT = -dsinT2 / (2.0 * cosT[:, None])
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
    """Explicit tangent of one perturbation parameter on a compiled surface list.

    A parameter perturbs the system through explicit tangents that activate at
    the surface(s) it touches; everything left unset contributes a zero tangent.

    Attributes
    ----------
    pose : dict
        {surface_index: (Qdot (3,), Rdot (3, 3) or None)} vertex-position and
        rotation-matrix tangents.  A dict so a thickness/despace can fan out to
        every downstream surface.
    shape : tuple or None
        (surface_index, name) naming a shape DOF ('c', 'k', or a freeform
        coefficient resolved by the shape's sag_param_partials).
    sag_partials : tuple or None
        (surface_index, fn) where fn(Xj, Yj) -> (sag_t, gx_t, gy_t) supplies
        the explicit sag/sag-gradient partials of a perturbation that is NOT a
        stored shape DOF -- e.g. a Zernike surface irregularity (CYN/CYD).  The
        partials are evaluated at the nominal local intersection; the nominal
        sag/Hessian are untouched.  A simple callable so any analytic departure
        term plugs in without modifying the surface classes.
    index : tuple or None
        (surface_index, value) tangent of the index FOLLOWING that surface;
        fans forward to every downstream segment until the next refraction.
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
    """Seed for a Zernike surface-irregularity (CYN/CYD) tolerance.

    Models the irregularity as an added departure delta z = a Z_n^m(x/R, y/R)
    on the named surface and seeds the tangent wrt its amplitude a (sag length
    units).  Unlike a shape DOF this need not be present on the nominal surface
    -- the partials come from sags.zernike_irregularity_partials evaluated at
    the nominal intersection, so a plain Sphere/Conic can be toleranced for
    irregularity it does not nominally carry.  CYN is (n, m) = (2, 2) (cylinder
    along the axes), CYD is (2, -2) (45-degree cylinder); any (n, m) is allowed.

    """
    from .sags import zernike_irregularity_partials  # local: keep the kernel
    # module's import surface free of the optional irregularity helper

    def partials(x, y):
        return zernike_irregularity_partials(n, m, x, y, normalization_radius,
                                             norm=norm)

    return DiffSeed(sag_partials=(surface, partials),
                    name=name or f'irr_Z{n}_{m}')


def seed_decenter(surface, axis, name=None):
    """Seed for a decenter (DSX/DSY) tolerance: vertex moves along axis.

    Analytic pose primitive: validated directly against finite differences. The
    Perturbation->seed mapping (seed_from_perturbation) does NOT use this; it
    derives pose tangents from the compiled layout (_pose_tangents_via_layout)
    so coord-break framing and solves are handled uniformly.
    """
    idx = {'x': 0, 'y': 1, 'z': 2}[axis]
    q = np.zeros(3, dtype=config.precision)
    q[idx] = 1.0
    return DiffSeed(pose={surface: (q, None)}, name=name or f'decenter_{axis}')


def seed_despace(surfaces, name='despace'):
    """Seed for a despace/thickness (DLZ/DLT) tolerance.

    surfaces is an iterable of (surface_index, sign): each named vertex moves
    sign units along +z.  A despace is a single surface with sign +1; a
    thickness fans out to every downstream surface (sign flips after a fold).

    Analytic pose primitive: validated directly against finite differences. The
    Perturbation->seed mapping derives the fan-out from the compiled layout
    instead (see seed_decenter).
    """
    q_plus = np.array([0., 0., 1.], dtype=config.precision)
    pose = {}
    for sidx, sgn in surfaces:
        pose[sidx] = (sgn * q_plus, None)
    return DiffSeed(pose=pose, name=name)


def seed_tilt(surface, axis, R_nominal=None, name=None):
    """Seed for a tilt (BTX/BTY) tolerance about a local axis, in radians.

    R_total = R_nominal @ R_tilt(a), so Rdot = R_nominal @ generator.  Pass
    R_nominal=None for an untilted surface (identity).

    Analytic pose primitive: validated directly against finite differences. The
    Perturbation->seed mapping derives Rdot from the compiled layout instead
    (see seed_decenter).
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
    """Per-surface (Qdot, Rdot) of the compiled layout wrt one DOF, by central FD.

    Recompiles the LensData with the targeted DOF at nominal +/- h and central-
    differences every compiled Surface's vertex P and rotation R.  This is the
    explicit pose tangent the differential trace consumes, and it captures
    thickness fan-out (mirror-fold aware), coordinate-break frame propagation,
    and any solve-/pickup-induced pose motion uniformly -- without re-deriving
    the layout by hand.  The layout is an O(surfaces) recompile, not a re-trace,
    so the wavefront-differential speed advantage holds.

    Returns the seed's pose dict {surface_index: (Qdot (3,), Rdot (3,3) or None)},
    sparse over the surfaces the DOF actually moves.
    """
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

    Maps a lens-design tolerance set onto the differential trace:

    - curvature (DLR) / radius -> the surface's shape DOF 'c'
    - conic -> the surface's shape DOF 'k'
    - thickness (DLT) / despace (DLZ) -> downstream pose fan-out, mirror-fold
      and solve aware
    - tilt (BTX/BTY) / decenter (DSX/DSY) on a coordinate break -> the pose
      tangents that break induces on every surface it reframes

    Shape tangents go through the surface's analytic sag_param_partials; pose
    tangents come from a central difference of the compiled layout (an
    O(surfaces) recompile, not a re-trace).  pose_step is that layout-FD step in
    the DOF's own units; the layout is exactly linear in translations and smooth
    in rotations, so it is accurate well below the trace-FD validation floor.  A
    shape perturbation still gets its layout-FD pose tangents, so a curvature or
    conic change that moves the image plane through an image-distance solve is
    captured too.

    Limitations: freeform-coefficient (CYN/CYD) and index (DLN) tolerances are
    not LensData DOF slots and stay on the FD sensitivity_table; a pickup that
    couples one shape DOF to another is not differentiated through (the seed
    activates only the perturbed surface).

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


def raytrace_with_tangents(surfaces, P, S, wvl, seeds, n_ambient=1.0,
                           n_ambient_dot=None, tol_sag=DEFAULT_TOL_SAG):
    """Trace (P, S) and propagate the tangent bundle for every seed.

    Runs the authoritative nominal trace via spencer_and_murty.raytrace, then
    propagates (Pdot, Sdot, Ldot) through the differential primitives using the
    nominal positions/normals recomputed in lockstep with the kernel.

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
    n_ambient : float, optional
        object-space index.
    n_ambient_dot : ndarray (P,), optional
        tangent of the ambient index (rarely nonzero).
    tol_sag : float, optional
        Newton convergence tolerance, forwarded to the nominal trace.

    Returns
    -------
    DiffTraceResult

    """
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

    trace = raytrace(surfaces, P, S, wvl, n_ambient=n_ambient, tol_sag=tol_sag)

    Qdot_s, Rdot_s, nprimedot_s, shape_params, sag_partial_fns = _assemble_seeds(
        jj, seeds, n_params)

    Pdot_hist = np.zeros((jj + 1, n_rays, 3, n_params), dtype=config.precision)
    Sdot_hist = np.zeros((jj + 1, n_rays, 3, n_params), dtype=config.precision)
    Ldot_hist = np.zeros((jj + 1, n_rays, n_params), dtype=config.precision)

    Pdot = Pdot_hist[0]
    Sdot = Sdot_hist[0]
    nj = float(n_ambient)
    if n_ambient_dot is None:
        nj_dot = np.zeros(n_params, dtype=config.precision)
    else:
        nj_dot = np.asarray(n_ambient_dot, dtype=config.precision)

    Pdot_prev = Pdot
    for j, surf in enumerate(surfaces):
        Reff = _eye3() if surf.R is None else np.asarray(surf.R, dtype=config.precision)
        Q = np.asarray(surf.P, dtype=config.precision)
        Pj_prev = trace.P[j]                # nominal global position (authoritative)
        Pj_cur = trace.P[j + 1]
        Qdot_j = Qdot_s[j]
        Rdot_j = Rdot_s[j]

        # Step I: to local
        P0, S_loc, P0dot, S_locdot = d_transform_local(
            Reff, Q, Pj_prev, trace.S[j], Pdot, Sdot, Qdot_j, Rdot_j)

        # nominal local intersection + normal
        Q_loc, n_hat = surf.intersect(P0, S_loc, tol_sag=tol_sag)

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
            nprime = float(surf.n(wvl))
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

    Differentiates opt._closest_approach_on_axis (the foot of the common
    perpendicular from the chief ray to the axis line).  The axis
    (axis_point, axis_dir) is fixed; only the chief ray (P, S) moves.

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


def d_intersect_reference_sphere(P, S, Pdot, Sdot, C, Cdot, R, Rdot):
    """Tangent of the reference-sphere segment length t for a ray bundle.

    Differentiates spencer_and_murty.intersect_reference_sphere with BOTH the
    sphere center C and radius R moving (the chief image point and exit-pupil
    distance both shift under a perturbation).

    Returns tdot (N, P).
    """
    dvec = P - C[None, :]
    dvecdot = Pdot - Cdot[None, :, :]
    b = row_dot(S, dvec)
    bdot = _dot_nt(dvec, Sdot) + _dot_nt(S, dvecdot)
    cc = row_dot(dvec, dvec) - R * R
    ccdot = 2.0 * _dot_nt(dvec, dvecdot) - 2.0 * R * Rdot[None, :]
    disc = b * b - cc
    disc = np.where(disc < 0, np.zeros_like(disc), disc)
    discdot = 2.0 * b[:, None] * bdot - ccdot
    sqrt_disc = np.sqrt(disc)
    safe = sqrt_disc == 0
    sqrt_disc_safe = np.where(safe, 1.0, sqrt_disc)
    sqrt_disc_dot = discdot / (2.0 * sqrt_disc_safe[:, None])
    tdot = -bdot - sqrt_disc_dot
    return tdot


def wavefront_with_tangents(surfaces, P, S, wavelength, seeds, *,
                            n_ambient=1.0, chief_index=None,
                            axis_point=None, axis_dir=None, P_xp=None,
                            field=None, output='length'):
    """Trace, compute OPD on the chief reference sphere, and its tangent maps.

    The differential analog of analysis.wavefront: returns the nominal OPD plus
    the per-tolerance wavefront-derivative maps dW_p = dOPD/dtau_p (one column
    per seed).  The reference sphere centers on the chief image point C and has
    radius R = ||P_xp - C||; both C and (for the auto-located exit pupil) P_xp
    move with the perturbation, and that motion is differentiated in closed
    form.

    Parameters mirror analysis.wavefront; seeds defines the trailing parameter
    axis (see raytrace_with_tangents).  output='length' returns dW in length
    units (prysm OPD sign: longer OPL positive); output='waves' returns
    -dOPD/lambda (lens-design wavefront-error convention), assuming mm / um.

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
    res = raytrace_with_tangents(surfaces, P, S, wavelength, seeds,
                                 n_ambient=n_ambient)
    trace = res.trace

    if chief_index is None:
        chief_index = _pupil_center_chief_index(P)
    valid = _valid_mask(trace.status, trace.P[-1])
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
        P_xp, P_xp_dot = d_closest_point_on_axis(
            P_chief_final, S_chief_final, Pdot_chief, Sdot_chief,
            axis_point, axis_dir)
    else:
        P_xp = np.asarray(P_xp, dtype=config.precision)
        P_xp_dot = np.zeros((3, res.n_params), dtype=config.precision)
    delta = P_xp - C
    delta_dot = P_xp_dot - Cdot
    R = float(np.sqrt(np.sum(delta * delta)))
    Rdot = _dot_vt(delta, delta_dot) / R

    filtered_chief = _filtered_chief_index(valid, chief_index)

    opd = opd_from_raytrace(trace.P[:, valid], trace.S[:, valid],
                            trace.OPL[:, valid],
                            P_img=P_chief_final, P_xp=P_xp,
                            n_image=n_ambient, chief_index=filtered_chief)

    P_last_f = trace.P[-1, valid]
    S_last_f = trace.S[-1, valid]
    Pdot_last_f = res.Pdot[-1][valid]
    Sdot_last_f = res.Sdot[-1][valid]
    Ldot_total_f = res.Ldot.sum(axis=0)[valid]

    tdot = d_intersect_reference_sphere(P_last_f, S_last_f,
                                        Pdot_last_f, Sdot_last_f,
                                        C, Cdot, R, Rdot)
    opl_total_dot = Ldot_total_f + n_ambient * tdot
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
    'd_intersect_reference_sphere',
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
