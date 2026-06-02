"""Minor optimization routines."""

from prysm.conf import config
from prysm.mathops import np, array_to_true_numpy
from . import spencer_and_murty
from ._line_math import (
    closest_point_on_line_to_line,
    line_intersection_params,
    unit_vector_between,
)


def _intersect_lines(P1, S1, P2, S2):
    """Find the slerp along the line (P1, S1) that results in intersection with
    the line (P2, S2).

    P = position, array shape (3,)
    S = direction cosines, array shape (3,)

    pair of two lines only.
    """
    return line_intersection_params(P1, S1, P2, S2)


def _establish_axis(P1, P2):
    """Given two points, establish an axis between them.

    Parameters
    ----------
    P1 : ndarray
        shape (3,), any float dtype
        first point
    P2 : ndarray
        shape (3,), any float dtype
        second point

    Returns
    -------
    ndarray, ndarray
        P1 (same exact PyObject) and direction cosine from P1 -> P2

    """
    return unit_vector_between(P1, P2)


def aim_rays(P, S, prescription, surface_index, target_xy, wvl,
             tol=1e-12, maxiter=20, strict=True, vary='position'):
    """Aim a bundle of rays so each lands at target_xy on a surface.

    Stop-aiming is a 2-input -> 2-output root find per ray: find the launch
    parameter (the transverse launch position, or the transverse launch
    direction) such that the ray lands at target_xy on
    prescription[surface_index].  The landing map (paraxial pupil-to-surface
    transfer plus mild aberration) is smooth and nearly linear, so a batched
    damped Newton on the per-ray 2x2 landing Jacobian solves the whole bundle
    in a handful of iterations.  Cost is ~3 batched traces per iteration,
    independent of the number of rays.

    Backend-pure (no scipy): runs natively on numpy, cupy, or torch.

    Parameters
    ----------
    P : ndarray
        shape (N, 3), launch positions.  Modified in place when vary is
        position; carried unchanged when vary is direction.
    S : ndarray
        shape (N, 3), launch direction cosines.  Modified in place (the
        transverse pair, z recomputed from |S| == 1) when vary is direction.
    prescription : sequence of Surface
        the system traced through during aiming.
    surface_index : int
        index of the aim surface; rays are driven to target_xy on it.
    target_xy : iterable
        target landing (x, y) on the aim surface; either a length-2 point
        (shared by every ray) or an (N, 2) array of per-ray targets (so a
        sampled pupil can be aimed to fill the stop).
    wvl : float
        wavelength for the aim trace, microns.
    tol : float, optional
        convergence tolerance on the landing-residual Euclidean norm.
    maxiter : int, optional
        Newton iteration cap.
    strict : bool, optional
        if True (default), raise RuntimeError listing any rays that did not
        converge; if False, return them at their best-effort parameter.
    vary : str, optional
        'position' (default) varies the transverse launch position (Px, Py)
        with the launch z and direction held fixed -- correct for collimated
        bundles.  'direction' varies the transverse direction cosines (Sx, Sy)
        (Sz recomputed to keep S a unit vector) with the launch position held
        fixed -- correct for a finite-conjugate bundle that must all emanate
        from one object point.

    Returns
    -------
    P, S : ndarray, ndarray
        shape (N, 3) launch positions and direction cosines, with the varied
        pair adjusted.
    converged : ndarray
        shape (N,), bool mask of rays that reached tol.

    """
    if vary not in ('position', 'direction'):
        raise ValueError(
            f"vary must be 'position' or 'direction', got {vary!r}")
    P = np.asarray(P).astype(config.precision).copy()
    S = np.asarray(S).astype(config.precision).copy()
    target = np.asarray(target_xy, dtype=config.precision)
    if target.ndim == 1:
        target = target.reshape(1, 2)
    trace_path = prescription[:surface_index + 1]

    if vary == 'direction':
        sz_sign = np.sign(S[:, 2])
        sz_sign = np.where(sz_sign == 0, 1.0, sz_sign)

        def apply(var):
            S[:, 0] = var[:, 0]
            S[:, 1] = var[:, 1]
            rem = 1.0 - var[:, 0] * var[:, 0] - var[:, 1] * var[:, 1]
            S[:, 2] = sz_sign * np.sqrt(np.clip(rem, 0.0, None))

        var0 = S[:, :2].copy()
    else:
        def apply(var):
            P[:, 0] = var[:, 0]
            P[:, 1] = var[:, 1]

        var0 = P[:, :2].copy()

    def landing(var):
        apply(var)
        tr = spencer_and_murty.raytrace(trace_path, P, S, wvl)
        return tr.P[-1, :, :2]

    eps = float(np.finfo(config.precision).eps)
    sqrt_eps = eps ** 0.5

    var = var0
    r = landing(var) - target
    rn = np.sqrt((r * r).sum(axis=1))
    dead = ~np.isfinite(rn)  # NaN landing (TIR / miss): cannot be aimed

    prev_max = np.inf
    for _ in range(int(maxiter)):
        active = ~dead
        if not bool(np.any(active)):
            break
        cur_max = float(np.max(np.where(active, rn, 0.0)))
        if cur_max < tol or cur_max >= prev_max:
            break
        prev_max = cur_max

        # forward-difference 2x2 Jacobian: columns d(x,y)/dvar0, d(x,y)/dvar1
        h = sqrt_eps * np.maximum(
            1.0, np.maximum(np.abs(var[:, 0]), np.abs(var[:, 1])))
        L0 = r + target
        var_dx = var.copy()
        var_dx[:, 0] = var_dx[:, 0] + h
        L_dx = landing(var_dx)
        var_dy = var.copy()
        var_dy[:, 1] = var_dy[:, 1] + h
        L_dy = landing(var_dy)

        a = (L_dx[:, 0] - L0[:, 0]) / h  # dx/dvar0
        c = (L_dx[:, 1] - L0[:, 1]) / h  # dy/dvar0
        b = (L_dy[:, 0] - L0[:, 0]) / h  # dx/dvar1
        d = (L_dy[:, 1] - L0[:, 1]) / h  # dy/dvar1

        det = a * d - b * c
        jac_scale = a * a + b * b + c * c + d * d
        singular = (~np.isfinite(det)) | (np.abs(det) < eps * jac_scale)

        rx = r[:, 0]
        ry = r[:, 1]
        safe_det = np.where(singular, 1.0, det)
        d0 = (-rx * d + b * ry) / safe_det  # closed-form 2x2 solve J@d = -r
        d1 = (rx * c - a * ry) / safe_det

        freeze = dead | singular
        delta = np.stack([np.where(freeze, 0.0, d0),
                          np.where(freeze, 0.0, d1)], axis=1)
        dead = dead | singular

        # damped step: backtrack if the active-ray max residual rose
        alpha = 1.0
        for _bt in range(40):
            var_try = var + alpha * delta
            r_try = landing(var_try) - target
            rn_try = np.sqrt((r_try * r_try).sum(axis=1))
            bad = ~np.isfinite(rn_try)
            cur = float(np.max(np.where((~dead) & (~bad), rn_try, 0.0)))
            if cur <= cur_max or alpha <= sqrt_eps:
                break
            alpha = alpha * 0.5

        # rays that went non-finite this step revert to their last good
        # parameter and are flagged dead; everyone else takes the step
        bad = ~np.isfinite(rn_try)
        keep_old = bad[:, np.newaxis]
        var = np.where(keep_old, var, var_try)
        r = np.where(keep_old, r, r_try)
        rn = np.where(bad, rn, rn_try)
        dead = dead | bad

    apply(var)
    converged = np.isfinite(rn) & (rn <= tol)

    if strict and not bool(np.all(converged)):
        bad_idx = array_to_true_numpy(np.where(~converged)[0]).tolist()
        n_bad = len(bad_idx)
        max_res = float(array_to_true_numpy(np.nanmax(np.where(dead, 0.0, rn))))
        raise RuntimeError(
            f'aim_rays failed to converge {n_bad} of {converged.shape[0]} '
            f'rays (indices {bad_idx}); worst finite residual {max_res:.3e}. '
            'Pass strict=False to return best-effort launch parameters.'
        )
    return P, S, converged


def _closest_approach_on_axis(P_chief, S_chief, axis_point, axis_dir):
    """Point on (axis_point, axis_dir) closest to the chief ray.

    For a centered system, the optical axis is z-hat through the origin and
    the exit (or entrance) pupil center lies where the chief ray crosses it.
    Real chief rays in 3D are usually skew to that axis, so we return the
    point on the axis at the foot of the common perpendicular.

    """
    return closest_point_on_line_to_line(P_chief, S_chief,
                                         axis_point, axis_dir)


def _pupil_on_axis(P_chief, S_chief, axis_p1, axis_p2):
    """Closest-approach point on the line (axis_p1, axis_p2) to the chief ray.

    Shared by locate_ep and locate_xp; the only difference between them is
    which pair of points defines the optical axis.

    """
    axis_p1 = np.asarray(axis_p1)
    S_axis = _establish_axis(axis_p1, np.asarray(axis_p2))
    return _closest_approach_on_axis(P_chief, S_chief, axis_p1, S_axis)


def locate_ep(P_chief, S_chief, P_obj, P_s1):
    """Locate the entrance pupil of a system.

    Defines the optical axis as the line through P_obj and P_s1, then finds
    the point on that axis closest to the chief ray.  For a coaxial system
    this reduces to the standard z-axis intersection; for tilted/decentered
    systems the user should supply a meaningful axis pair (P_obj, P_s1).

    Parameters
    ----------
    P_chief : ndarray
        any point on the chief ray (e.g. its starting position at the object)
    S_chief : ndarray
        chief ray direction cosines
    P_obj : iterable
        object-side point on the optical axis (commonly the object location)
    P_s1 : iterable
        a second axis point (commonly the first surface vertex).  Must differ
        from P_obj.

    Returns
    -------
    ndarray
        position of the entrance pupil (X,Y,Z)

    """
    return _pupil_on_axis(P_chief, S_chief, P_obj, P_s1)


def xp_reference_sphere(P_chief, S_chief, axis_point=None, axis_dir=None):
    """Compute the exit-pupil reference sphere for a single chief ray.

    The reference sphere is centered on the chief ray's image point (P_chief)
    and has radius |P_xp - P_chief|, where P_xp is the chief ray's closest
    approach to the optical axis (the line through axis_point parallel to
    axis_dir).  For a centered coaxial system, the optical axis is the
    z-axis through the origin (the defaults).

    For tilted/decentered systems, supply axis_point and axis_dir explicitly,
    or compute P_xp via independent means (e.g., from a bundle of chief rays
    from different fields) and pass it directly to opd_from_raytrace.

    Parameters
    ----------
    P_chief : ndarray
        shape (3,).  Position of the chief ray, typically at the image plane.
        This becomes the center of the reference sphere.
    S_chief : ndarray
        shape (3,).  Direction cosines of the chief ray after the last surface.
    axis_point : iterable, optional
        a point on the optical axis (default: origin)
    axis_dir : iterable, optional
        direction of the optical axis (default: +z)

    Returns
    -------
    C, R, P_xp : ndarray, float, ndarray
        sphere center (=P_chief), radius, exit pupil center

    """
    if axis_point is None:
        axis_point = np.zeros(3, dtype=np.asarray(P_chief).dtype)
    if axis_dir is None:
        axis_dir = np.array([0., 0., 1.], dtype=np.asarray(P_chief).dtype)
    C = np.asarray(P_chief)
    P_xp = _closest_approach_on_axis(P_chief, S_chief,
                                     np.asarray(axis_point),
                                     np.asarray(axis_dir))
    R = np.sqrt(np.sum((P_xp - C) ** 2))
    return C, float(R), P_xp


def _pupil_center_chief_index(P, valid=None):
    """Index of the launch ray nearest the pupil center -- the chief ray.

    For a symmetric sampling the launch bundle's centroid is the pupil center,
    even after the bundle was translated by entrance-pupil routing; pick the
    ray nearest it.  A fixed N // 2 is only the pupil center for fan / rect
    grids and lands on an arbitrary ring sample for a hex bundle.  Shared by
    wavefront(), the OPD primitives, the differential trace, and the adjoint
    merit heads so every reference-sphere anchor agrees on the chief ray.

    When valid is given, only rays passing the mask are eligible, so an
    obscured (annular) bundle whose geometric center sample is clipped still
    resolves to the surviving ray nearest the pupil center.
    """
    P = np.asarray(P)
    center = np.mean(P[:, :2], axis=0)
    d2 = np.sum((P[:, :2] - center) ** 2, axis=1)
    if valid is not None:
        d2 = np.where(valid, d2, np.inf)
    return int(np.argmin(d2))


def opd_from_raytrace(P_hist, S_hist, OPL_hist, P_img, P_xp,
                      n_image=1.0, chief_index=None):
    """Compute OPD on the exit-pupil reference sphere from a ray-trace.

    For each ray, the OPL is summed through the prescription, then extended
    along the final direction cosine until it intersects the reference sphere
    centered on P_img with radius |P_xp - P_img|.  OPD is reported relative
    to the chief ray (chief OPD = 0).

    The starting plane for OPL accumulation is whatever transverse plane the
    rays were launched on (e.g. z = const for a collimated fan).  Provided all
    rays in the bundle share that starting plane, the constant OPL offset
    cancels in the chief-relative subtraction.

    Parameters
    ----------
    P_hist : ndarray
        position history from raytrace, shape (jj+1, N, 3)
    S_hist : ndarray
        direction cosine history from raytrace, shape (jj+1, N, 3)
    OPL_hist : ndarray
        per-segment OPL history from raytrace, shape (jj+1, N).  OPL_hist[0]
        is zero by convention; OPL_hist[j+1] is the OPL of the segment ending
        at surface j.
    P_img : iterable
        image point — center of the reference sphere
    P_xp : iterable
        exit pupil center — sets the radius
    n_image : float
        index of refraction in image space (1=vacuum)
    chief_index : int, optional
        row index of the chief ray.  If None, defaults to the ray nearest the
        pupil center (correct for any sampling, including hex grids where N//2
        is an off-center ring sample).

    Returns
    -------
    opd : ndarray
        shape (N,).  Optical path difference at the reference sphere, in the
        same length units as the prescription.  Sign convention: chief == 0;
        rays whose OPL exceeds the chief's are positive.

    """
    P_img = np.asarray(P_img)
    P_xp = np.asarray(P_xp)
    R = np.sqrt(np.sum((P_xp - P_img) ** 2))

    P_last = P_hist[-1]
    S_last = S_hist[-1]
    Q, t = spencer_and_murty.intersect_reference_sphere(P_last, S_last, P_img, R)

    OPL_through = OPL_hist.sum(axis=0)
    OPL_total = OPL_through + n_image * t

    if chief_index is None:
        chief_index = _pupil_center_chief_index(P_hist[0])

    return OPL_total - OPL_total[chief_index]


def eic_distance(P_a, d_a, P_b, d_b):
    """Equally-inclined-chord distance from a reference point on ray b to ray a.

    Hopkins' classical relation between two pencils (Welford, Aberrations of
    Optical Systems, eq 8.x): the signed distance along ray a from its
    equally-inclined-chord point to P_a, given the second ray (P_b, d_b),
    evaluates as

        e = (d_a + d_b) . (P_a - P_b) / (1 + d_a . d_b).

    The chord that links the two rays makes equal angles with each, so e is
    the OPL contribution from the difference in start points -- the
    geometric piece of a Hopkins-style wavefront aberration calculation.
    Provided here as a primitive for users porting classical formulas; the
    OPD path in opd_from_raytrace_eic does not need it because prysm has the
    full OPL_hist along each ray in global coordinates.

    Parameters
    ----------
    P_a, d_a, P_b, d_b : ndarray
        any common shape; the last axis is xyz.

    Returns
    -------
    ndarray
        e, with P_a's leading shape.

    """
    dP = P_a - P_b
    num = ((d_a + d_b) * dP).sum(axis=-1)
    denom = 1.0 + (d_a * d_b).sum(axis=-1)
    return num / denom


def opd_from_raytrace_eic(P_hist, S_hist, OPL_hist, P_img, P_xp,
                          n_image=1.0, chief_index=None,
                          infinite_threshold=1.0e8):
    """OPD on the exit-pupil reference surface, robust to extreme conjugates.

    Same contract as opd_from_raytrace but uses a cancellation-free
    (Welford-rationalized) intersection of each ray with the reference
    sphere and falls back automatically to a planar reference through P_xp
    normal to the chief direction when the reference-sphere radius exceeds
    infinite_threshold.  The legacy form t = -b - sqrt(b**2 - cc) loses
    precision when |b| ~ sqrt(...) -- the regime reached for long
    conjugates -- because the two large terms cancel.  Here the cancelling
    branch is rationalized to t = cc / (-b + sqrt(...)), whose denominator
    is a sum of like-signed quantities; the non-cancelling branch is left
    in its original form, so the result is bit-identical to the legacy
    path for benign cases and merely more accurate for extreme ones.  The
    plane fallback handles the strict afocal limit where the reference
    sphere is not well defined at all.

    Parameters
    ----------
    P_hist, S_hist, OPL_hist : ndarray
        ray history from raytrace, shapes (jj+1, N, 3), (jj+1, N, 3),
        (jj+1, N).
    P_img : iterable
        chief image point -- sphere center.
    P_xp : iterable
        exit-pupil center -- sets the sphere radius R = |P_xp - P_img|.
    n_image : float
        index in image space.
    chief_index : int, optional
        row of the chief ray.  Default: the ray nearest the pupil center.
    infinite_threshold : float
        reference-sphere radius (in prescription length units) above which
        the planar reference is used; finite-conjugate systems are far
        below the default and behave identically to the legacy path.

    Returns
    -------
    opd : ndarray, shape (N,)
        OPD relative to the chief, sign matching opd_from_raytrace
        (longer ray OPL -> positive).

    """
    P_img = np.asarray(P_img)
    P_xp = np.asarray(P_xp)
    R = float(np.sqrt(np.sum((P_xp - P_img) ** 2)))
    P_last = P_hist[-1]
    S_last = S_hist[-1]
    OPL_through = OPL_hist.sum(axis=0)
    if chief_index is None:
        chief_index = _pupil_center_chief_index(P_hist[0])

    if (not np.isfinite(R)) or R > infinite_threshold:
        # planar reference through P_xp normal to the chief direction --
        # the Mikš limit of the reference sphere as R -> infinity.
        d_c = S_last[chief_index]
        denom = (S_last * d_c).sum(axis=-1)
        with np.errstate(divide='ignore', invalid='ignore'):
            s = ((P_xp - P_last) * d_c).sum(axis=-1) / denom
    else:
        # cancellation-free sphere intersection equivalent to
        # spencer_and_murty.intersect_reference_sphere's -b - sqrt root.
        g = P_last - P_img
        beta = (S_last * g).sum(axis=-1)
        gamma = (g * g).sum(axis=-1) - R * R
        disc = beta * beta - gamma
        disc = np.where(disc < 0, np.zeros_like(disc), disc)
        sq = np.sqrt(disc)
        # cancellation occurs when -beta and -sqrt are opposite-signed
        # (i.e. beta < 0, the converging-beam case); rationalize that branch.
        denom = -beta + sq
        safe = np.where(denom == 0, np.ones_like(denom), denom)
        s = np.where(beta < 0,
                     np.where(denom == 0, np.zeros_like(gamma), gamma / safe),
                     -beta - sq)

    OPL_total = OPL_through + n_image * s
    return OPL_total - OPL_total[chief_index]


def locate_xp(P_chief, S_chief, P_img, P_sk):
    """Locate the exit pupil of a system.

    Defines the optical axis as the line through P_img and P_sk, then finds
    the point on that axis closest to the chief ray.  For a coaxial system
    this reduces to the standard z-axis intersection; for tilted/decentered
    systems the user should supply a meaningful axis pair (P_img, P_sk).

    Parameters
    ----------
    P_chief : ndarray
        any point on the chief ray (e.g. final position at the image plane)
    S_chief : ndarray
        chief ray direction cosines (after the last surface)
    P_img : iterable
        image-side point on the optical axis (commonly the image point)
    P_sk : iterable
        a second axis point (commonly the last optical surface vertex —
        NOT the image plane).  Must differ from P_img.

    Returns
    -------
    ndarray
        position of the exit pupil (X,Y,Z)

    """
    return _pupil_on_axis(P_chief, S_chief, P_img, P_sk)


# ---------- spot statistics ----------

def _finite_ray_mask(P):
    """Return a bool mask for rays with finite position coordinates."""
    P = np.asarray(P)
    return np.isfinite(P).all(axis=-1)


def _valid_mask(status, P=None):
    """Reduce status and optional positions to a bool valid-ray mask."""
    if status is None:
        if P is None:
            return None
        return _finite_ray_mask(P)

    valid = np.asarray(status).imag == 0
    if P is not None:
        valid = valid & _finite_ray_mask(P)
    return valid


def spot_centroid(P_final, status=None):
    """Mean (x, y) position of valid rays at a surface plane.

    Parameters
    ----------
    P_final : ndarray
        shape (N, 3) — typically `raytrace(...).P[-1]`.
    status : ndarray, optional
        per-ray complex status from `raytrace` (or any equivalent).  If
        provided, rays with `status.imag != 0` are excluded.

    Returns
    -------
    ndarray
        shape (2,) — mean (x, y) of the valid rays.  Returns `[nan, nan]`
        if no rays are valid.

    """
    P_final = np.asarray(P_final)
    valid = _valid_mask(status, P_final)
    if valid is not None:
        P_final = P_final[valid]
    if P_final.shape[0] == 0:
        return np.array([np.nan, np.nan])
    return P_final[..., :2].mean(axis=0)


def rms_spot_radius(P_final, status=None, centroid=None):
    """RMS distance of valid rays from their centroid.

    Parameters
    ----------
    P_final : ndarray
        shape (N, 3) ray positions.
    status : ndarray, optional
        per-ray complex status; rays with `status.imag != 0` are excluded.
    centroid : ndarray, optional
        shape (2,) custom center for the RMS.  If None, the spot centroid
        of the valid rays is used.

    Returns
    -------
    float
        RMS spot radius in the (x, y) plane.  Returns `nan` if no rays
        are valid.

    """
    P_final = np.asarray(P_final)
    valid = _valid_mask(status, P_final)
    if valid is not None:
        P_final = P_final[valid]
    if P_final.shape[0] == 0:
        return float('nan')
    if centroid is None:
        centroid = P_final[..., :2].mean(axis=0)
    diffs = P_final[..., :2] - np.asarray(centroid)
    return float(np.sqrt(np.mean(np.sum(diffs * diffs, axis=-1))))


def geometric_psf_histogram(P_final, status=None, bins=64, extent=None):
    """2D histogram of valid rays' (x, y) positions — the geometric PSF.

    Parameters
    ----------
    P_final : ndarray
        shape (N, 3) ray positions.
    status : ndarray, optional
        per-ray complex status; rays with `status.imag != 0` are excluded.
    bins : int or [int, int]
        passed through to `np.histogram2d`; number of bins per axis.
    extent : list of (lo, hi) length 2, optional
        per-axis (x, y) histogram range.  If None, a square window is
        auto-fit around the valid-ray bounding box with a 5% margin.

    Returns
    -------
    H : ndarray
        2D histogram of shape `(nx, ny)`.
    xedges, yedges : ndarray
        bin edges along the two axes.

    """
    P_final = np.asarray(P_final)
    valid = _valid_mask(status, P_final)
    if valid is not None:
        P_final = P_final[valid]
    x = P_final[..., 0]
    y = P_final[..., 1]
    if extent is None:
        if x.size == 0:
            extent = [(-1.0, 1.0), (-1.0, 1.0)]
        else:
            cx = float(x.mean())
            cy = float(y.mean())
            r = max(float(np.abs(x - cx).max()),
                    float(np.abs(y - cy).max())) * 1.05
            r = max(r, 1e-12)  # avoid zero-width range
            extent = [(cx - r, cx + r), (cy - r, cy + r)]
    H, xedges, yedges = np.histogram2d(x, y, bins=bins, range=extent)
    return H, xedges, yedges
