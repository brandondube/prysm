"""Spencer & Murty's General Ray-Trace algorithm."""

# don't uncomment this line, this is a very obscure import
# not using it at the moment for GPU compatability
# from numpy.core.umath_tests import inner1d

from prysm.mathops import np, row_dot

SURFACE_INTERSECTION_DEFAULT_MAXITER = 100

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


class RayTraceResult:
    """Structured return type for ``raytrace``.

    Provides attribute access (``result.P``, ``.S``, ``.OPL``, ``.status``)
    plus iteration support so ``P, S, OPL = raytrace(...)`` keeps working.

    """

    __slots__ = ('P', 'S', 'OPL', 'status')

    def __init__(self, P, S, OPL, status):
        self.P = P
        self.S = S
        self.OPL = OPL
        self.status = status

    def __iter__(self):
        # legacy 3-tuple unpacking: yield P, S, OPL (status is opt-in via
        # attribute access).
        yield self.P
        yield self.S
        yield self.OPL

    def __len__(self):
        return 3

    def __repr__(self):
        return (
            f'RayTraceResult(N_rays={self.status.shape[0]}, '
            f'N_surfaces={self.P.shape[0] - 1}, '
            f'valid={int((self.status.imag == 0).sum())})'
        )


def _sanitize_eps(eps, dtype):
    if eps is None:
        try:
            # 100x eps being hard-coded is a little not great, but user can
            # defeat with their own eps.  An editable module variable requires
            # globals() to be retrieved, which is icky.  Don't want to thread
            # a control for "fctr" all the way to the top for this.
            # note: some rays dead stall in fp64 at 7.105427357601002e-15
            # 100*eps ~= 2e-14
            return np.finfo(dtype).eps * 100
        except:  # NOQA - cannot predict error type in numpy-like libs
            return 1e-14

    return eps


def newton_raphson_solve_s(P1, S, FFp, s1=0.0,
                           eps=None,
                           maxiter=SURFACE_INTERSECTION_DEFAULT_MAXITER,
                           return_valid=False):
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
    FFp : callable of signature F(x,y) -> z, [Nx, Ny, Nz]
        a function which returns the surface sag at point x, y as well as
        the X, Y, Z partial derivatives at that point
    s1 : float
        initial guess for the length along the ray from (X1, Y1, 0) to reach the surface
    eps : float
        tolerance for convergence of Newton's method
    maxiter : int
        maximum number of iterations to allow
    return_valid : bool, optional
        if True, also return a length-N boolean mask indicating which rays
        converged within `maxiter`.  Non-converged rays still appear in
        Pj/r as NaN, but the mask makes the failure explicit.

    Returns
    -------
    Pj, r : ndarray, ndarray
        final position of the ray intersection, and the surface normal at that point.
        if return_valid is True, also returns a length-N boolean array.

    """
    dtype = P1.dtype
    eps = _sanitize_eps(eps, dtype)
    nrays = P1.shape[0]
    # Single-alloc init: s1 may be a scalar (broadcasts to all rays) OR a
    # (nrays,) array of per-ray seeds (from the conic-seeded Newton path).
    # numpy assignment handles both in one writeable buffer at the right dtype.
    sj_work = np.empty(nrays, dtype=dtype)
    sj_work[...] = s1
    # the Pj and r to be returned; we keep these three data structures around
    # so they can be adjusted within the loop
    Pj_out = np.empty_like(P1)
    r_out = np.empty((nrays, 3), dtype=dtype)
    # mask maps working-buffer-index -> original-ray-index, so converged rays
    # scatter back to the right Pj_out/r_out slot.  sj_work / S_work / P1_work
    # shrink in lock-step with mask each iteration, so subsequent iterations
    # operate on a buffer the size of the still-active set rather than
    # fancy-indexing into the full inputs every time.
    mask = np.arange(nrays)
    S_work = S
    P1_work = P1
    for j in range(maxiter):
        sj_bcast = sj_work[:, np.newaxis]
        Pj = P1_work + sj_bcast * S_work
        Xj = Pj[..., 0]
        Yj = Pj[..., 1]
        Zj = Pj[..., 2]
        sagj, r = FFp(Xj, Yj)
        Fj = Zj - sagj
        Fpj = row_dot(S_work, r)
        sjp1 = sj_work - Fj / Fpj

        delta = abs(sjp1 - sj_work)

        # this block of code stops computation on rays which have converged,
        # while allowing those which have not yet converged to progress,
        # over "time," the iterations of Newton-Raphson will speed up, in terms
        # of wall clock time.
        rays_which_converged = (delta < eps)
        insert_mask = mask[rays_which_converged]
        if insert_mask.size != 0:
            Pj_out[insert_mask] = Pj[rays_which_converged]
            r_out[insert_mask] = r[rays_which_converged]
            diverged = ~rays_which_converged
            mask = mask[diverged]
            if mask.size == 0:
                break  # all rays converged
            sj_work = sjp1[diverged]
            S_work = S_work[diverged]
            P1_work = P1_work[diverged]
        else:
            sj_work = sjp1

    # NaN out rays which failed to converge (within maxiter)
    if mask.size > 0:
        Pj_out[mask] = np.nan
        r_out[mask] = np.nan
    if return_valid:
        valid = np.ones(nrays, dtype=bool)
        if mask.size > 0:
            valid[mask] = False
        return Pj_out, r_out, valid
    return Pj_out, r_out


def intersect(P0, S, FFp, s1=0,
              eps=None,
              maxiter=SURFACE_INTERSECTION_DEFAULT_MAXITER,
              return_valid=False):
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
    FFp : callable of signature F(x,y) -> z, [Nx, Ny, Nz]
        a function which returns the surface sag at point x, y as well as
        the X, Y, Z partial derivatives at that point
    s1 : float
        initial guess for the length along the ray from (X1, Y1, 0) to reach the surface
    eps : float
        tolerance for convergence of Newton's method
    maxiter : int
        maximum number of iterations to allow
    return_valid : bool, optional
        if True, also return a length-N boolean mask flagging rays that
        converged within `maxiter`.

    Returns
    -------
    Pj, r : ndarray, ndarray
        final position of the ray intersection, and the surface normal at that point.
        if return_valid is True, also returns a length-N boolean array.

    """
    eps = _sanitize_eps(eps, P0.dtype)
    # batch support -- ellipsis skip any early dimensions, then replace
    # dot with a multiply
    P0, S = np.atleast_2d(P0, S)
    # go to z=0
    Z0 = P0[..., 2]
    m = S[..., 2]
    s0 = -Z0/m
    # Eq. 7, in vector form (extra computation on Z is cheaper than breaking apart P and S)
    # the newaxis on s0 turns (N,) -> (N,1) so that multiply's broadcast rules work
    P1 = P0 + s0[:, np.newaxis] * S
    # P1 is (N,3)
    # then use newton's method to find and go to the intersection
    return newton_raphson_solve_s(P1, S, FFp, s1, eps, maxiter,
                                  return_valid=return_valid)


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
        # for batch compatability, we do that manually
        XYZ2 = np.matmul(R, XYZ2[..., np.newaxis]).squeeze(-1)
        S = np.matmul(R, S[..., np.newaxis]).squeeze(-1)

    return XYZ2, S


def refract(n, nprime, S, r):
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
    r : ndarray
        shape (3,) or (N,3), any float dtype
        surface normals (Fx, Fy, 1)

    Returns
    -------
    ndarray
        Sprime, a length 3 vector containing the exitant direction cosines

    """
    mu = n/nprime
    musq = mu * mu
    cosI = row_dot(r, S)
    cosIsq = cosI * cosI
    # the inline newaxis-es are terrible for readability, but serve a performance purpose
    # broadcast the square root to 2D, so that fewer very expensive sqrt ops are done
    # then, in the second term, broadcast cosI for compatability with S and r
    # since it is needed there
    # TIR rays produce a negative sqrt argument; the resulting NaN is the
    # expected signal (raytrace inspects it to flag STATUS_TIR), so silence
    # the FP warning at the source.
    with np.errstate(invalid='ignore'):
        first_term = np.sqrt(1 - musq * (1 - cosIsq))[:, np.newaxis] * r
    second_term = mu * (S - cosI[:, np.newaxis] * r)
    return first_term + second_term


def reflect(S, r):
    """Reflect a ray off of a surface.

    Parameters
    ----------
    S : ndarray
        shape (3,) or (N,3), any float dtype
        (k,l,m) incident direction cosines
    r : ndarray
        shape (3,) or (N,3), any float dtype
        surface normals (Fx, Fy, 1)

    Returns
    -------
    ndarray
        Sprime, the exitant direction cosines

    """
    # at least 2D turns (3,) -> (1,3) where 1 = batch reflect count
    # this allows us to use the same code for vector operations on many
    # S, r or on one S, r
    S, r = np.atleast_2d(S, r)
    rnorm = row_dot(r, r)

    # paragraph above Eq. 45, mu=1
    # and see that definition of a including
    # mu=1 does not require multiply by mu (1)
    cosI = row_dot(S, r) / rnorm

    # newaxis for batch support, (N) -> (N,1) shape
    cosI = cosI[:, np.newaxis]
    return S - 2 * cosI * r


def raytrace(surfaces, P, S, wvl, n_ambient=1):
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
        surf.F(x,y) -> z sag
        surf.Fp(x,y) -> (Fx, Fy, 1) derivatives (with S&M convention for 1 in z)
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
    n_ambient : float
        ambient index of refraction (1=vacuum)

    Returns
    -------
    RayTraceResult
        Vanilla class carrying ``P`` (position history, ``(jj+1, ..., 3)``),
        ``S`` (direction-cosine history, ``(jj+1, ..., 3)``), ``OPL``
        (per-segment optical path length history, ``(jj+1, ...)``), and
        ``status`` (per-ray complex status, see module-level ``STATUS_*``
        constants).  Iterating the result yields the legacy 3-tuple
        ``(P, S, OPL)`` so ``P, S, OPL = raytrace(...)`` keeps working.

        OPL_hist[0] is zero by convention.  OPL_hist[j+1] is the OPL of the
        segment from P_hist[j] to P_hist[j+1] (i.e., the path through the
        medium preceding surface j).  The cumulative OPL up to surface j is
        OPL_hist[:j+2].sum(axis=0).

    Implementation Notes
    --------------------
    See Spencer & Murty, General Ray-Tracing Procedure JOSA 1961

    Steps (I, II, III, IV) utilize the functions:
    I   -> transform_to_local_coords
    II  -> Surface.intersect (analytic on Plane/Sphere/Conic/OffAxisConic,
           Newton-Raphson via newton_raphson_solve_s on the generic Surface)
    III -> reflect or refract
    IV  -> transform_to_global_coords

    Surface-level apertures (``surf.aperture``) are checked at step II's
    intersection point; rays falling outside are flagged STATUS_CLIP.  TIR
    is detected at step III by post-refract NaN inspection.  Once a ray's
    status becomes non-zero it is excluded from further status updates so
    only the first failure surface is recorded.

    """
    # Surfaces whose intersect() has a closed-form geometric solution get
    # STATUS_MISS on failure (negative discriminant / outside supporting
    # region); Newton-driven surfaces get STATUS_NEWTON for non-convergence.
    # Discrimination is via the ``_analytic_intersect`` class attribute set on
    # Plane/Sphere/Conic/OffAxisConic.
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
    nj = n_ambient
    for j, surf in enumerate(surfaces):
        surf_idx = j + 1  # 1-based index recorded in status.real
        # I - transform from global to local coordinates
        P0, Sj = transform_to_local_coords(Pj, surf.P, Sj, surf.R)
        # II - find ray intersection (analytic if available, else Newton-Raphson)
        Pj, r, valid = surf.intersect(P0, Sj, return_valid=True)
        # encode geometry / convergence failures: the failure code depends on
        # which intersect path the surface uses.
        active = (status.imag == 0)  # rays still propagating
        if not valid.all():
            failed = active & ~valid
            if failed.any():
                if getattr(surf, '_analytic_intersect', False):
                    code = STATUS_MISS
                else:
                    code = STATUS_NEWTON
                status[failed] = surf_idx + code * 1j
                active = active & valid
        # II.b - aperture clipping (after intersection, before bending)
        if surf.aperture is not None and active.any():
            inside = np.asarray(surf.aperture(Pj[..., 0], Pj[..., 1]),
                                dtype=bool)
            clipped = active & ~inside
            if clipped.any():
                status[clipped] = surf_idx + STATUS_CLIP * 1j
                active = active & inside
        # III - reflection or refraction
        if surf.typ == STYPE_REFLECT:
            Sjp1 = reflect(Sj, r)
            n_post = nj
        elif surf.typ == STYPE_REFRACT:
            nprime = surf.n(wvl)
            pre_refract = active.copy()
            Sjp1 = refract(nj, nprime, Sj, r)
            # TIR detection: any active ray that newly carries NaN
            tir = pre_refract & np.isnan(Sjp1).any(axis=-1)
            if tir.any():
                status[tir] = surf_idx + STATUS_TIR * 1j
                active = active & ~tir
            n_post = nprime
        else:
            # other surface types do not bend rays
            Sjp1 = Sj
            Pjp1 = Pj
            n_post = nj
        # III.b - grating diffraction (refr/refl only); evanescent ⇒ TIR-like
        if (surf.grating is not None
                and surf.typ in (STYPE_REFLECT, STYPE_REFRACT)
                and active.any()):
            Sjp1_diff, valid_diff = surf.diffract(Sjp1, r, n_post, wvl)
            evanescent = active & ~valid_diff
            if evanescent.any():
                status[evanescent] = surf_idx + STATUS_TIR * 1j
                active = active & valid_diff
            Sjp1 = Sjp1_diff

        # IV - back to world coordinates
        if surf.R is None:
            Rt = None
        else:
            # transformation matrix has inverse which is its transpose
            Rt = surf.R.T
        Pjp1, Sjp1 = transform_to_global_coords(Pj, surf.P, Sjp1, Rt)
        # geometric path of the segment we just traversed (medium index = nj),
        # computed in the global frame for batch+rotated-surface compatibility
        seg = Pjp1 - P_hist[j]
        seg_len = np.sqrt(np.sum(seg * seg, axis=-1))
        OPL_hist[j+1] = nj * seg_len
        if surf.typ == STYPE_REFRACT:
            nj = nprime

        # Once a ray fails, do not keep propagating finite coordinates through
        # downstream surfaces.  The first failure surface/mode is encoded in
        # status; histories from that surface onward are NaN so consumers
        # cannot mistake them for valid trace data.
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
        shape (N,3), any float dtype.  Ray direction cosines.
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
    # near-zero negative disc from FP noise -> clamp
    disc = np.where(disc < 0, np.zeros_like(disc), disc)
    sqrt_disc = np.sqrt(disc)
    # of the two roots t = -b +/- sqrt_disc, pick the one closer to P (smaller |t|).
    # for a converging ray with C ahead of P (b<0), -b - sqrt_disc and -b + sqrt_disc
    # straddle the closest approach; the upstream intersection is -b - sqrt_disc.
    t = -b - sqrt_disc
    Q = P + t[:, np.newaxis] * S
    return Q, t
