"""Spencer & Murty's General Ray-Trace algorithm."""

# Obsolete numpy-only fast path, kept as a note for archaeology.
# from numpy.core.umath_tests import inner1d

from prysm.mathops import np, row_dot

SURFACE_INTERSECTION_DEFAULT_MAXITER = 100
DEFAULT_TOL_SAG = 1e-12

# Surface-type constants live here to avoid an import cycle with surfaces.py.
# Measurement types do not bend rays; test them with _is_measurement_surf.
STYPE_REFLECT = -1
STYPE_REFRACT = -2
STYPE_EVAL = -3  # NOQA  intermediate measurement surface (neither object nor image)
STYPE_OBJ = -4   # object endpoint (row 0): object distance + object-space medium
STYPE_IMG = -5   # image endpoint (last row): measurement surface (often flat)


def _is_measurement_surf(typ):
    """True for a non-bending measurement surface (EVAL, OBJECT, or IMAGE).

    Keys off the interaction type, not the shape: an EVAL/OBJECT/IMAGE surface
    can carry any geometry (a curved measurement surface is valid); it simply
    records the ray's state without refracting or reflecting it.
    """
    return typ in (STYPE_EVAL, STYPE_OBJ, STYPE_IMG)

# Per-ray status encoding for raytrace().  status is a complex array of shape
# (N_rays,); the imaginary part picks the failure mode (0 = ok), the real
# part records which surface the outcome was decided at.
#
#   imag      meaning
#   ----      -------
#   0         valid; ray passed every surface in the prescription.  Real
#             part = number of surfaces traversed.
#   +1        Newton-Raphson did not converge at surface int(real).
#   +2        ray was clipped by surface int(real)'s aperture.
#   -1        analytic intersection had negative discriminant at surface
#             int(real) (geometry says the ray never touches the surface).
#   -2        total internal reflection at surface int(real).
#   -3        diffracted grating order is evanescent at surface int(real).
#
# Sign of imag distinguishes the two failure families: positive = ray made it
# to the surface but couldn't be processed; negative = ray never made it.
STATUS_OK = 0
STATUS_NEWTON = 1   # numerical: Newton-Raphson didn't converge
STATUS_CLIP = 2     # numerical: aperture clipped
STATUS_MISS = -1    # geometric: no analytic intersection
STATUS_TIR = -2     # geometric: total internal reflection
STATUS_EVANESCENT = -3  # geometric: diffracted order does not propagate

_STATUS_LABELS = {
    STATUS_OK: 'OK',
    STATUS_NEWTON: 'NEWTON',
    STATUS_CLIP: 'CLIPPED',
    STATUS_MISS: 'MISS',
    STATUS_TIR: 'TIR',
    STATUS_EVANESCENT: 'EVANESCENT',
}


class RayTraceResult:
    """Structured return type for raytrace.

    Provides attribute access (result.P, .S, .OPL, .status).

    """

    __slots__ = ('P', 'S', 'OPL', 'status', 'status_record', 'intermediates')

    def __init__(self, P, S, OPL, status, intermediates=None):
        self.P = P
        self.S = S
        self.OPL = OPL
        self.status = status
        self.status_record = RayStatus.from_encoded(status)
        # per-surface Interaction objects when raytrace(keep_intermediates=True);
        # the AD stacks read these instead of re-running the Newton intersect.
        self.intermediates = intermediates

    def __repr__(self):
        return (
            f'RayTraceResult(N_rays={self.status.shape[0]}, '
            f'N_surfaces={self.P.shape[0] - 1}, '
            f'valid={int(valid_mask(self.status).sum())})'
        )


class RayStatus:
    """Structured view of per-ray trace status."""

    __slots__ = ('surface', 'code')

    def __init__(self, surface, code):
        self.surface = surface
        self.code = code

    @classmethod
    def from_encoded(cls, status):
        return cls(status.real.astype(int), status.imag.astype(int))

    @property
    def encoded(self):
        return self.surface + 1j * self.code

    @property
    def text(self):
        """Human-readable status strings."""
        return decode_status(self.encoded)


def _decode_status_scalar(status):
    """Decode one complex ray status value."""
    surface = int(status.real)
    code = int(status.imag)
    label = _STATUS_LABELS.get(code, f'UNKNOWN({code})')
    if code == STATUS_OK:
        return label
    return f'{label} at surface {surface}'


def decode_status(status):
    """Decode raytrace's compact complex status encoding.

    Parameters
    ----------
    status : complex or ndarray
        Encoded status value or array.  The real part stores the 1-based
        surface index and the imaginary part stores the STATUS_* code.

    Returns
    -------
    str or ndarray
        A scalar input returns a string.  An array input returns an object
        array of strings with matching shape.
    """
    arr = np.asarray(status)
    if arr.ndim == 0:
        return _decode_status_scalar(arr.item())
    decoded = [_decode_status_scalar(v) for v in arr.ravel()]
    return np.asarray(decoded, dtype=object).reshape(arr.shape)


def _finite_ray_mask(P):
    """Return a bool mask for rays with finite position coordinates."""
    P = np.asarray(P)
    return np.isfinite(P).all(axis=-1)


def valid_mask(status, P=None):
    """Reduce status and optional positions to a bool valid-ray mask.

    Parameters
    ----------
    status : ndarray or None
        Encoded ray status.  If None, validity is derived from P.
    P : ndarray, optional
        Ray positions.  When supplied, non-finite ray positions are also
        rejected.

    Returns
    -------
    ndarray or None
        Boolean valid-ray mask.  If both status and P are None, returns None.
    """
    if status is None:
        if P is None:
            return None
        return _finite_ray_mask(P)

    valid = np.asarray(status).imag == STATUS_OK
    if P is not None:
        valid = valid & _finite_ray_mask(P)
    return valid


def resolve_tol_sag(tol_sag, dtype):
    """Resolve the surface-residual convergence tolerance.

    Parameters
    ----------
    tol_sag : float or None
        absolute convergence tolerance on the surface residual Z - sag.  None
        selects a dtype-aware default: 1e-12 in float64, relaxed toward 100x
        the machine epsilon of the working dtype so a float32 backend (or
        config.precision == 32) can actually reach it.  The residual Z - sag
        can never drop below a few epsilon in the working precision, so a fixed
        1e-12 is unreachable in float32 and would strand every Newton surface
        at maxiter.
    dtype : numpy dtype
        working precision of the ray buffer; sets the tolerance floor.

    Returns
    -------
    float
        the resolved tolerance.

    """
    if tol_sag is None:
        return max(DEFAULT_TOL_SAG, float(np.finfo(dtype).eps) * 100.0)
    return tol_sag


def newton_raphson_solve_s(P1, S, sag_and_normal, s1=0.0,
                           tol_sag=None,
                           maxiter=SURFACE_INTERSECTION_DEFAULT_MAXITER):
    """Use Newton-Raphson iteration to solve for intersection between a ray and surface.

    Parameters
    ----------
    P1 : ndarray
        shape (3,) or (N,3), any float dtype
        position (X1,Y1,Z1) at in the plane normal to the surface vertex
        Eq. 7 from Spencer & Murty, except we keep Z1 so we can utilize vector algebra
    S : ndarray
        shape (3,) or (N,3), any float dtype
        (k,l,m) incident direction cosines
    sag_and_normal : callable
        Function returning surface sag and unit normal at x, y.
    s1 : float
        initial guess for the length along the ray from (X1, Y1, 0) to reach the surface
    tol_sag : float
        Absolute convergence tolerance on the surface residual Z - sag.
    maxiter : int
        maximum number of iterations to allow

    Returns
    -------
    Pj, n_hat, valid : ndarray, ndarray, ndarray
        final position of the ray intersection, and the unit surface normal
        at that point, plus a length-N boolean convergence mask.

    """
    dtype = P1.dtype
    tol_sag = resolve_tol_sag(tol_sag, dtype)
    nrays = P1.shape[0]
    # Single-alloc init: s1 may be a scalar (broadcasts to all rays) OR a
    # (nrays,) array of per-ray seeds (from the conic-seeded Newton path).
    # numpy assignment handles both in one writeable buffer at the right dtype.
    sj_full = np.empty(nrays, dtype=dtype)
    sj_full[...] = s1
    # Pj/n outputs default to NaN/invalid: rays dropped below (non-finite
    # input) and rays that never converge both keep these defaults.
    Pj_out = np.full((nrays, 3), np.nan, dtype=dtype)
    n_out = np.full((nrays, 3), np.nan, dtype=dtype)
    valid = np.zeros(nrays, dtype=bool)
    # Drop rays that arrive non-finite -- clipped/failed upstream and forwarded
    # as NaN by the raytrace kernel, or a missed conic seed (NaN s1).  Newton
    # cannot make progress from a non-finite state (the first residual is NaN,
    # which never satisfies the convergence test), so left in the active set
    # they would iterate all the way to maxiter on every surface.  Dropping
    # them here yields the identical NaN / invalid result at no cost.
    finite = (np.isfinite(P1).all(axis=-1)
              & np.isfinite(S).all(axis=-1)
              & np.isfinite(sj_full))
    # mask maps working-buffer-index -> original-ray-index, so converged rays
    # scatter back to the right Pj_out/n_out slot.  sj_work / S_work / P1_work
    # shrink in lock-step with mask each iteration, so subsequent iterations
    # operate on a buffer the size of the still-active set rather than
    # fancy-indexing into the full inputs every time.  The all-finite case (the
    # common one) skips the slice entirely so the hot path makes no copies.
    if finite.all():
        mask = np.arange(nrays)
        sj_work = sj_full
        S_work = S
        P1_work = P1
    else:
        mask = np.flatnonzero(finite)
        if mask.size == 0:
            return Pj_out, n_out, valid
        sj_work = sj_full[mask]
        S_work = S[mask]
        P1_work = P1[mask]
    for _ in range(maxiter):
        Pj = P1_work + sj_work[:, np.newaxis] * S_work
        Xj = Pj[..., 0]
        Yj = Pj[..., 1]
        Zj = Pj[..., 2]
        sagj, n_hat = sag_and_normal(Xj, Yj)
        Fj = Zj - sagj

        converged = np.abs(Fj) < tol_sag
        if converged.any():
            insert_idx = mask[converged]
            Pj_out[insert_idx] = Pj[converged]
            n_out[insert_idx] = n_hat[converged]
            valid[insert_idx] = True
            survive = ~converged
            mask = mask[survive]
            if mask.size == 0:
                break
            sj_work = sj_work[survive]
            S_work = S_work[survive]
            P1_work = P1_work[survive]
            n_hat = n_hat[survive]
            Fj = Fj[survive]

        with np.errstate(divide='ignore', invalid='ignore'):
            Fpj = row_dot(S_work, n_hat) / n_hat[..., 2]
        sj_work = sj_work - Fj / Fpj

    # Rays still in mask never converged: they keep the NaN / invalid defaults.
    return Pj_out, n_out, valid


def intersect(P0, S, sag_and_normal, s1=0,
              tol_sag=None,
              maxiter=SURFACE_INTERSECTION_DEFAULT_MAXITER):
    """Find the intersection of a ray and a surface.

    Parameters
    ----------
    P0 : ndarray
        shape (3,) or (N,3), any float dtype
        position of the ray, in local coordinates (but Z not necessarily zero)
        Eq. 3 Spencer & Murty
    S : ndarray
        shape (3,) or (N,3), any float dtype
        (k,l,m) incident direction cosines
    sag_and_normal : callable
        Function returning surface sag and unit normal at x, y.
    s1 : float
        initial guess for the length along the ray from (X1, Y1, 0) to reach the surface
    tol_sag : float
        absolute convergence tolerance on the surface residual Z - sag.
    maxiter : int
        maximum number of iterations to allow

    Returns
    -------
    Pj, n_hat, valid : ndarray, ndarray, ndarray
        final position of the ray intersection, and the unit surface normal
        at that point, plus a length-N boolean convergence mask.

    """
    # tol_sag is resolved at the leaf (newton_raphson_solve_s) where the
    # working dtype is known, so it scales with float32/float64 backends.
    # batch support -- ellipsis skip any early dimensions, then replace
    # dot with a multiply
    P0, S = np.atleast_2d(P0, S)
    # go to z=0
    Z0 = P0[..., 2]
    m = S[..., 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        s0 = -Z0/m
    # Eq. 7, in vector form (extra computation on Z is cheaper than breaking apart P and S)
    # the newaxis on s0 turns (N,) -> (N,1) so that multiply's broadcast rules work
    P1 = P0 + s0[:, np.newaxis] * S
    # P1 is (N,3)
    # then use newton's method to find and go to the intersection
    return newton_raphson_solve_s(P1, S, sag_and_normal, s1,
                                  tol_sag=tol_sag, maxiter=maxiter)


def transform_to_global_coords(XYZ, P, S, R=None):
    """Transform the coordinates XYZ from local coordinates about P back to global coordinates.

    Parameters
    ----------
    XYZ : ndarray
        shape (3,) or (N,3), any float dtype
        local coordinates [X,Y,Z] along the final dimension
    P : ndarray
        shape (3,), any float dtype
        point defining the origin of the local coordinate frame, [X0,Y0,Z0]
    S : ndarray
        shape (3,) or (N,3), any float dtype
        (k,l,m) incident direction cosines
    R : ndarray
        shape (3,3), any float dtype
        the surface's lab-to-local rotation, the same matrix
        transform_to_local_coords takes; its transpose is applied here

    Returns
    -------
    ndarray, ndarray
        rotated XYZ coordinates, rotated direction cosines

    """
    if R is not None:
        XYZ = np.matmul(XYZ, R)
        S = np.matmul(S, R)

    XYZ = XYZ + P
    return XYZ, S


def transform_to_local_coords(XYZ, P, S, R=None):
    """Transform the coordinates XYZ to local coordinates about P, plausibly rotated by R.

    Parameters
    ----------
    XYZ : ndarray
        shape (3,) or (N,3), any float dtype
        "world" coordinates [X,Y,Z] along the final dimension
    P : ndarray
        shape (3,), any float dtype
        point defining the origin of the local coordinate frame, [X0,Y0,Z0]
    S : ndarray
        shape (3,) or (N,3), any float dtype
        (k,l,m) incident direction cosines
    R : ndarray
        shape (3,3), any float dtype
        rotation matrix to apply, if the surface is tilted

    Returns
    -------
    ndarray, ndarray
        rotated XYZ coordinates, rotated direction cosines

    """
    XYZ2 = XYZ - P
    if R is not None:
        Rt = np.swapaxes(R, -1, -2)
        XYZ2 = np.matmul(XYZ2, Rt)
        S = np.matmul(S, Rt)

    return XYZ2, S


def refract(n, nprime, S, n_hat):
    """Solve Snell's law for the exitant direction cosines.

    Parameters
    ----------
    n : float
        preceeding index of refraction
    nprime : float
        following index of refraction
    S : ndarray
        shape (3,) or (N,3), any float dtype
        (k,l,m) incident direction cosines
    n_hat : ndarray
        shape (3,) or (N,3), any float dtype
        unit surface normals.

    Returns
    -------
    ndarray
        Sprime, a length 3 vector containing the exitant direction cosines

    """
    S, n_hat = np.atleast_2d(S, n_hat)
    mu = n/nprime
    cosI = row_dot(n_hat, S)
    sinT_sq = mu * mu * (1.0 - cosI * cosI)
    with np.errstate(invalid='ignore'):
        cosT = np.sqrt(1.0 - sinT_sq)
    factor = np.sign(cosI) * cosT - mu * cosI
    return mu * S + factor[:, np.newaxis] * n_hat


def reflect(S, n_hat):
    """Reflect a ray off of a surface.

    Parameters
    ----------
    S : ndarray
        shape (3,) or (N,3), any float dtype
        (k,l,m) incident direction cosines
    n_hat : ndarray
        shape (3,) or (N,3), any float dtype
        unit surface normals.

    Returns
    -------
    ndarray
        Sprime, the exitant direction cosines

    """
    # at least 2D turns (3,) -> (1,3) where 1 = batch reflect count
    # this allows us to use the same code for vector operations on many
    # S, r or on one S, r
    S, n_hat = np.atleast_2d(S, n_hat)
    cosI = row_dot(S, n_hat)
    return S - 2.0 * cosI[:, np.newaxis] * n_hat


def _launch_medium_index(surfaces, wvl):
    """Index of the medium the bundle launches in.

    The object medium is carried by the surface sequence itself: when the
    leading surface is an eval (object) surface, its material's .n(wvl) is the
    launch medium; otherwise the launch medium is air (n = 1).  Kept tensor
    clean (no float coercion) so a tolerance that perturbs the object medium
    threads through autograd.
    """
    if len(surfaces) > 0:
        first = surfaces[0]
        if _is_measurement_surf(getattr(first, 'typ', None)):
            material = getattr(first, 'material', None)
            if material is not None:
                return material.n(wvl)
    return 1.0


def raytrace(surfaces, P, S, wvl, tol_sag=None, keep_intermediates=False):
    """Perform a raytrace through a sequence of surfaces.

    Parameters
    ----------
    surfaces : iterable
        surfaces to trace through.
    P : ndarray
        shape (3,) or (N, 3), starting positions.
    S : ndarray
        shape (3,) or (N, 3), starting direction cosines.
    wvl : float
        wavelength of light, microns.
    tol_sag : float, optional
        convergence tolerance for Newton surface intersections.
    keep_intermediates : bool, optional
        when True, attach the per-surface Interaction objects (local-frame
        intersection, normal, post-bend direction, ...) to the result so the
        differential / adjoint stacks can read them instead of re-running the
        Newton intersect.  Off by default -- the common analysis path does not
        need them and they pin extra arrays alive.

    Returns
    -------
    RayTraceResult
        position, direction, OPL, and status histories.

    """
    # Analytic misses and Newton non-convergence get distinct status codes.
    P = np.asarray(P)
    S = np.asarray(S)
    # promote 1D single-ray inputs to 2D batch shape for the duration of the
    # trace; squeeze the trailing batch dim back off before returning
    squeeze_batch = (P.ndim == 1)
    if squeeze_batch:
        P = P[np.newaxis, :]
        S = S[np.newaxis, :]
    jj = len(surfaces)
    n_rays = P.shape[0]
    P_hist = np.empty((jj+1, *P.shape), dtype=P.dtype)
    S_hist = np.empty((jj+1, *S.shape), dtype=P.dtype)
    OPL_hist = np.zeros((jj+1, *P.shape[:-1]), dtype=P.dtype)
    status = np.zeros(n_rays, dtype=np.complex128)
    Pj = P
    Sj = S
    P_hist[0] = P
    S_hist[0] = S
    # the launch medium is intrinsic to the surfaces: a leading eval (object)
    # surface carries the object-space material, otherwise the bundle launches
    # in air.  To launch inside glass, prepend an eval surface with that glass.
    nj = _launch_medium_index(surfaces, wvl)
    intermediates = [] if keep_intermediates else None
    for j, surf in enumerate(surfaces):
        surf_idx = j + 1  # 1-based index recorded in status.real
        # Surface physics live in Surface.interact; the kernel tracks history.
        # The first segment may be signed when the launch plane sits past S1.
        step = surf.interact(Pj, Sj, nj, wvl, tol_sag=tol_sag,
                             first_segment=(j == 0))

        active = valid_mask(status)
        failed = active & (step.code != STATUS_OK)
        if failed.any():
            # first failure wins: record the surface and code, drop from active.
            status[failed] = surf_idx + 1j * step.code[failed]
            active = active & ~failed

        Pjp1 = step.P
        Sjp1 = step.S
        OPL_hist[j+1] = step.opl
        if surf.typ == STYPE_REFRACT:
            nj = step.n_post

        inactive = ~active
        if inactive.any():
            Pjp1 = Pjp1.copy()
            Sjp1 = Sjp1.copy()
            Pjp1[inactive] = np.nan
            Sjp1[inactive] = np.nan
            OPL_hist[j+1][inactive] = np.nan
        P_hist[j+1] = Pjp1
        S_hist[j+1] = Sjp1
        Pj, Sj = Pjp1, Sjp1
        if intermediates is not None:
            intermediates.append(step)

    # rays that survived all surfaces: status.real records the highest
    # surface index reached (= jj for fully successful rays).
    fully_valid = valid_mask(status)
    status.real[fully_valid] = jj

    if squeeze_batch:
        P_hist = P_hist.squeeze(axis=1)
        S_hist = S_hist.squeeze(axis=1)
        OPL_hist = OPL_hist.squeeze(axis=1)
        # status stays as a length-1 array so attribute access is consistent
        # across single-ray and batched calls.

    return RayTraceResult(P_hist, S_hist, OPL_hist, status, intermediates)
