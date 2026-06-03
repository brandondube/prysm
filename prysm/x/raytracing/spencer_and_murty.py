"""Spencer & Murty's General Ray-Trace algorithm."""

# don't uncomment this line, this is a very obscure import
# not using it at the moment for GPU compatibility
# from numpy.core.umath_tests import inner1d

from prysm.conf import config
from prysm.mathops import np, row_dot

SURFACE_INTERSECTION_DEFAULT_MAXITER = 100
DEFAULT_TOL_SAG = 1e-12

# Surface-type constants live here (rather than in surfaces.py) so that
# spencer_and_murty.raytrace() and paraxial.system_matrix() can import them
# without cycling through surfaces.py.  surfaces.py imports them from here.
STYPE_REFLECT = -1
STYPE_REFRACT = -2
STYPE_EVAL = -3  # NOQA

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
#
# Sign of imag distinguishes the two failure families: positive = ray made it
# to the surface but couldn't be processed; negative = ray never made it.
STATUS_OK = 0
STATUS_NEWTON = 1   # numerical: Newton-Raphson didn't converge
STATUS_CLIP = 2     # numerical: aperture clipped
STATUS_MISS = -1    # geometric: no analytic intersection
STATUS_TIR = -2     # geometric: total internal reflection

_STATUS_LABELS = {
    STATUS_OK: 'OK',
    STATUS_NEWTON: 'NEWTON',
    STATUS_CLIP: 'CLIPPED',
    STATUS_MISS: 'MISS',
    STATUS_TIR: 'TIR',
}


class RayTraceResult:
    """Structured return type for raytrace.

    Provides attribute access (result.P, .S, .OPL, .status).

    """

    __slots__ = ('P', 'S', 'OPL', 'status', 'status_record')

    def __init__(self, P, S, OPL, status):
        self.P = P
        self.S = S
        self.OPL = OPL
        self.status = status
        self.status_record = RayStatus.from_encoded(status)

    def __repr__(self):
        return (
            f'RayTraceResult(N_rays={self.status.shape[0]}, '
            f'N_surfaces={self.P.shape[0] - 1}, '
            f'valid={int((self.status.imag == 0).sum())})'
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


def _failure_code_for_surface(surf):
    if getattr(surf, '_analytic_intersect', False):
        return STATUS_MISS
    return STATUS_NEWTON


def _record_failure(status, active, failed, surf_idx, code):
    failed = active & failed
    if failed.any():
        status[failed] = surf_idx + code * 1j
        active = active & ~failed
    return active


def _apply_aperture_status(surf, Pj, active, status, surf_idx):
    if surf.aperture is None or not active.any():
        return active
    inside = np.asarray(surf.aperture(Pj[..., 0], Pj[..., 1]), dtype=bool)
    return _record_failure(status, active, ~inside, surf_idx, STATUS_CLIP)


def _bend_rays(surf, Sj, n_hat, wvl, nj, active, status, surf_idx):
    if surf.typ == STYPE_REFLECT:
        return reflect(Sj, n_hat), nj, active
    if surf.typ == STYPE_REFRACT:
        nprime = surf.n(wvl)
        pre_refract = active.copy()
        Sjp1 = refract(nj, nprime, Sj, n_hat)
        tir = pre_refract & np.isnan(Sjp1).any(axis=-1)
        active = _record_failure(status, active, tir, surf_idx, STATUS_TIR)
        return Sjp1, nprime, active
    return Sj, nj, active


def _apply_grating_status(surf, Sjp1, r, n_post, wvl, active, status, surf_idx):
    if (surf.grating is None
            or surf.typ not in (STYPE_REFLECT, STYPE_REFRACT)
            or not active.any()):
        return Sjp1, active
    Sjp1_diff, valid_diff = surf.diffract(Sjp1, r, n_post, wvl)
    active = _record_failure(status, active, ~valid_diff, surf_idx, STATUS_TIR)
    return Sjp1_diff, active


def _mark_inactive_history(Pjp1, Sjp1, OPL_segment, active):
    inactive = ~active
    if inactive.any():
        Pjp1 = Pjp1.copy()
        Sjp1 = Sjp1.copy()
        Pjp1[inactive] = np.nan
        Sjp1[inactive] = np.nan
        OPL_segment[inactive] = np.nan
    return Pjp1, Sjp1


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
    sj_work = np.empty(nrays, dtype=dtype)
    sj_work[...] = s1
    # the Pj and n_hat to be returned; we keep these data structures around
    # so they can be adjusted within the loop
    Pj_out = np.empty_like(P1)
    n_out = np.empty((nrays, 3), dtype=dtype)
    # mask maps working-buffer-index -> original-ray-index, so converged rays
    # scatter back to the right Pj_out/r_out slot.  sj_work / S_work / P1_work
    # shrink in lock-step with mask each iteration, so subsequent iterations
    # operate on a buffer the size of the still-active set rather than
    # fancy-indexing into the full inputs every time.
    mask = np.arange(nrays)
    S_work = S
    P1_work = P1
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

    # NaN out rays which failed to converge (within maxiter)
    if mask.size > 0:
        Pj_out[mask] = np.nan
        n_out[mask] = np.nan
    valid = np.ones(nrays, dtype=bool)
    if mask.size > 0:
        valid[mask] = False
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
    """Transform the coordiantes XYZ from local coordinates about P back to global coordinates.

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
    if R is not None:
        XYZ, S = np.atleast_2d(XYZ, S)
        XYZ = np.matmul(R, XYZ[..., np.newaxis]).squeeze(-1)
        S = np.matmul(R, S[..., np.newaxis]).squeeze(-1)

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
        XYZ2, S = np.atleast_2d(XYZ2, S)
        # in regular matmul, 3x3 @ (3,) has a 1 appended to the dimension
        # of the second array to make it into a column vector
        # for batch compatibility, we do that manually
        XYZ2 = np.matmul(R, XYZ2[..., np.newaxis]).squeeze(-1)
        S = np.matmul(R, S[..., np.newaxis]).squeeze(-1)

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
    leading surface is an eval (object) surface, its material n(wvl) is the
    launch medium; otherwise the launch medium is air (n = 1).  Kept tensor
    clean (no float coercion) so a tolerance that perturbs the object medium
    threads through autograd.
    """
    if len(surfaces) > 0:
        first = surfaces[0]
        if getattr(first, 'typ', None) == STYPE_EVAL:
            n = getattr(first, 'n', None)
            if callable(n):
                return n(wvl)
    return 1.0


def raytrace(surfaces, P, S, wvl, tol_sag=None):
    """Perform a raytrace through a sequence of surfaces.

    Notes
    -----
    When P and S are single dimensional, a single ray is traced.

    When they have two dimensions, the first dimension is the "batch" and the
    second contains [X,Y,Z] and [k,l,m] for each ray in the batch.

    There is no internal ray aiming or other adjustment to P and S.

    In a batch raytrace, there is no reason all rows of P and S must belong to
    the same ray bundle.

    wvl does not matter and is not used in raytraces with only reflective
    surfaces

    A ray originating "at infinity" would have
    P = [Px, Py, -1e99]
    S = [0, 0, 1] # propagating in the +z direction
    though the value of P is not so important,
    since S defines the ray as moving in the +z direction only

    Parameters
    ----------
    surfaces : iterable
        the surfaces to trace through;
        a surface is defined by the interface:
        surf.sag_and_normal(x,y) -> z sag, unit normal
        surf.typ in {STYPE}
        surf.P, surface global coordinates, [X,Y,Z]
        surf.R, surface rotation matrix (may be None)
        surf.n(wvl) -> refractive index (wvl in um)
    P : ndarray
        shape (3,) or (N,3), any float dtype
        position (X0,Y0,Z0) at the outset of the raytrace
    S : ndarray
        shape (3,) or (N,3), any float dtype
        (k,l,m) starting direction cosines
    wvl : float
        wavelength of light, um
    tol_sag : float, optional
        convergence tolerance in sag for newton-raphson iteration intersecting
        with surfaces that cannot be analytically intersected

    Returns
    -------
    RayTraceResult
        Vanilla class carrying P (position history, (jj+1, ..., 3)),
        S (direction-cosine history, (jj+1, ..., 3)), OPL
        (per-segment optical path length history, (jj+1, ...)), and
        status (per-ray complex status, see module-level STATUS_*
        constants).

        OPL_hist[0] is zero by convention.  OPL_hist[j+1] is the OPL of the
        segment from P_hist[j] to P_hist[j+1] (i.e., the path through the
        medium preceding surface j).  The cumulative OPL up to surface j is
        OPL_hist[:j+2].sum(axis=0).

    Implementation Notes
    --------------------
    See Spencer & Murty, General Ray-Tracing Procedure JOSA 1961

    Steps (I, II, III, IV) utilize the functions:
    I   -> transform_to_local_coords
    II  -> Surface.intersect (analytic when the shape supplies it,
           otherwise Newton-Raphson via newton_raphson_solve_s)
    III -> reflect or refract
    IV  -> transform_to_global_coords

    Surface-level apertures (surf.aperture) are checked at step II's
    intersection point; rays falling outside are flagged STATUS_CLIP.  TIR
    is detected at step III by post-refract NaN inspection.  Once a ray's
    status becomes non-zero it is excluded from further status updates so
    only the first failure surface is recorded.

    """
    # Surfaces whose intersect() has a closed-form geometric solution get
    # STATUS_MISS on failure (negative discriminant / outside supporting
    # region); Newton-driven surfaces get STATUS_NEWTON for non-convergence.
    # Discrimination is via Surface._analytic_intersect, copied from the
    # shape object during construction.
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
    for j, surf in enumerate(surfaces):
        surf_idx = j + 1  # 1-based index recorded in status.real
        P0, Sj = transform_to_local_coords(Pj, surf.P, Sj, surf.R)
        Pj, r, valid = surf.intersect(P0, Sj, tol_sag=tol_sag)

        active = (status.imag == 0)
        if not valid.all():
            active = _record_failure(status, active, ~valid, surf_idx,
                                     _failure_code_for_surface(surf))

        active = _apply_aperture_status(surf, Pj, active, status, surf_idx)
        Sjp1, n_post, active = _bend_rays(surf, Sj, r, wvl, nj, active,
                                          status, surf_idx)
        Sjp1, active = _apply_grating_status(surf, Sjp1, r, n_post, wvl,
                                             active, status, surf_idx)

        if surf.R is None:
            Rt = None
        else:
            Rt = surf.R.T
        Pjp1, Sjp1 = transform_to_global_coords(Pj, surf.P, Sjp1, Rt)

        seg = Pjp1 - P_hist[j]
        seg_len = np.sqrt(np.sum(seg * seg, axis=-1))
        OPL_hist[j+1] = nj * seg_len
        if surf.typ == STYPE_REFRACT:
            nj = n_post

        Pjp1, Sjp1 = _mark_inactive_history(Pjp1, Sjp1, OPL_hist[j+1],
                                            active)
        P_hist[j+1] = Pjp1
        S_hist[j+1] = Sjp1
        Pj, Sj = Pjp1, Sjp1

    # rays that survived all surfaces: status.real records the highest
    # surface index reached (= jj for fully successful rays).
    fully_valid = (status.imag == 0)
    status.real[fully_valid] = jj

    if squeeze_batch:
        P_hist = P_hist.squeeze(axis=1)
        S_hist = S_hist.squeeze(axis=1)
        OPL_hist = OPL_hist.squeeze(axis=1)
        # status stays as a length-1 array so attribute access is consistent
        # across single-ray and batched calls.

    return RayTraceResult(P_hist, S_hist, OPL_hist, status)


def intersect_reference_sphere(P, S, C, R):
    """Intersect a batch of rays P + t*S with the sphere |X - C| = R.

    Picks the root nearer to P along +S — appropriate when rays are
    propagating toward the sphere center (converging beam) and the sphere
    sits between the last optical surface and the image point.  For a
    diverging beam (virtual exit pupil behind the optic) the same root
    convention still yields the upstream intersection.

    Parameters
    ----------
    P : ndarray
        shape (N,3), any float dtype.  Ray origins (typically the last surface).
    S : ndarray
        shape (N,3), any float dtype.  Ray direction cosines leaving P
    C : ndarray
        shape (3,), any float dtype.  Sphere center (image point).
    R : float
        sphere radius (|XP - image|).

    Returns
    -------
    Q, t : ndarray, ndarray
        intersection points (N,3) and signed segment lengths (N,) from P along S.

    """
    P, S = np.atleast_2d(P, S)
    C = np.asarray(C)
    d = P - C                                  # (N,3)
    b = row_dot(S, d)                       # S . (P-C)
    cc = row_dot(d, d) - R * R
    disc = b * b - cc
    _validate_reference_sphere_intersection(P, C, R, disc)
    # near-zero negative disc from FP noise -> clamp
    disc = np.where(disc < 0, np.zeros_like(disc), disc)
    sqrt_disc = np.sqrt(disc)
    # of the two roots t = -b +/- sqrt_disc, pick the one closer to P (smaller |t|).
    # for a converging ray with C ahead of P (b<0), -b - sqrt_disc and -b + sqrt_disc
    # straddle the closest approach; the upstream intersection is -b - sqrt_disc.
    t = -b - sqrt_disc
    Q = P + t[:, np.newaxis] * S
    return Q, t


def _reference_sphere_tolerances(P, C, R):
    """Absolute radius and discriminant tolerances for reference-sphere tests."""
    P = np.asarray(P)
    C = np.asarray(C)
    dtype = getattr(P, 'dtype', config.precision)
    rtol = 1e-6 if dtype == np.float32 else 1e-12
    d = P - C
    try:
        extent = float(np.sqrt(np.max(row_dot(d, d))))
    except TypeError:
        extent = 1.0
    scale = max(1.0, extent, abs(float(R)))
    return rtol * scale, rtol * scale * scale


def _validate_reference_sphere_intersection(P, C, R, disc):
    """Raise for degenerate reference spheres or wholesale missed intersections."""
    if not np.isfinite(R):
        raise ValueError(
            'reference-sphere radius is not finite; pass a finite P_xp or use '
            'the EIC planar-reference path for an infinite exit pupil'
        )
    radius_tol, disc_tol = _reference_sphere_tolerances(P, C, R)
    if abs(float(R)) <= radius_tol:
        raise ValueError(
            'reference-sphere radius is degenerate; pass P_xp or a resolvable '
            'stop/pupil route instead of auto-locating from an axial chief ray'
        )
    bad = disc < -disc_tol
    if bool(np.any(bad)):
        try:
            fraction = float(np.count_nonzero(bad)) / float(np.size(bad))
        except TypeError:
            fraction = 1.0
        if fraction >= 0.25:
            raise ValueError(
                'reference sphere does not intersect a substantial fraction '
                'of rays; check P_xp/reference-sphere geometry'
            )
