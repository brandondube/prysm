"""Minor optimization routines."""

from prysm.conf import config
from prysm.mathops import np, array_to_true_numpy
from . import spencer_and_murty
from .spencer_and_murty import valid_mask
from ._line_math import (
    closest_point_on_line_to_line,
    normalize_vector,
    unit_vector_between,
)


def aim_rays(P, S, surfaces, surface_index, target_xy, wvl,
             tol=1e-12, maxiter=20, strict=True, vary='position'):
    """Aim a bundle of rays so each lands at target_xy on a surface.

    Parameters
    ----------
    P : ndarray
        shape (N, 3), launch positions.
    S : ndarray
        shape (N, 3), launch direction cosines.
    surfaces : sequence of Surface
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
        if True, raise when any ray did not converge.
    vary : str, optional
        'position' (default) or 'direction'.

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
    trace_path = surfaces[:surface_index + 1]

    if vary == 'direction':
        sz_sign = np.sign(S[:, 2])
        sz_sign[sz_sign == 0] = 1.0
        sz_anchor = sz_sign * np.abs(S[:, 2])

        def apply(var):
            sx = var[:, 0]
            sy = var[:, 1]
            norm = np.sqrt(sx * sx + sy * sy + sz_anchor * sz_anchor)
            zero_norm = norm == 0
            norm[zero_norm] = 1.0
            S[:, 0] = sx / norm
            S[:, 1] = sy / norm
            S[:, 2] = np.where(zero_norm, sz_sign, sz_anchor / norm)

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
        det[singular] = 1.0  # safe denominator; singular rays frozen below
        d0 = (-rx * d + b * ry) / det  # closed-form 2x2 solve J@d = -r
        d1 = (rx * c - a * ry) / det

        freeze = dead | singular
        delta = np.stack([d0, d1], axis=1)
        delta[freeze] = 0.0
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
        var_try[bad] = var[bad]
        r_try[bad] = r[bad]
        rn_try[bad] = rn[bad]
        var, r, rn = var_try, r_try, rn_try
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


def _chief_axis_perp_norm(S_chief, axis_dir):
    """Magnitude of the chief-ray direction perpendicular to an axis."""
    S_chief = np.asarray(S_chief)
    axis_dir = normalize_vector(np.asarray(axis_dir), axis=-1)
    s_parallel = np.sum(S_chief * axis_dir) * axis_dir
    s_perp = S_chief - s_parallel
    return float(np.sqrt(np.sum(s_perp * s_perp)))


def _pupil_on_axis(P_chief, S_chief, axis_p1, axis_p2):
    """Closest-approach point on the line (axis_p1, axis_p2) to the chief ray.

    Shared by locate_ep and locate_xp; the only difference between them is
    which pair of points defines the optical axis.

    """
    axis_p1 = np.asarray(axis_p1)
    S_axis = unit_vector_between(axis_p1, np.asarray(axis_p2))
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


def xp_reference_sphere(P_chief, S_chief, axis_point=None, axis_dir=None,
                        min_perp=1e-6):
    """Compute the exit-pupil reference sphere for a single chief ray.

    The reference sphere is centered on the chief ray's image point (P_chief)
    and has radius |P_xp - P_chief|, where P_xp is the chief ray's closest
    approach to the optical axis.

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
    min_perp : float, optional
        minimum chief slope perpendicular to the axis.

    Returns
    -------
    C, R, P_xp : ndarray, float, ndarray
        sphere center (=P_chief), radius, exit pupil center

    """
    if axis_point is None:
        axis_point = np.zeros(3, dtype=np.asarray(P_chief).dtype)
    if axis_dir is None:
        axis_dir = np.array([0., 0., 1.], dtype=np.asarray(P_chief).dtype)
    if _chief_axis_perp_norm(S_chief, axis_dir) < min_perp:
        raise ValueError(
            'cannot locate the exit pupil from a near-axial chief ray; pass '
            'P_xp or a resolvable stop/pupil route to anchor the reference '
            'sphere'
        )
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
        d2[~valid] = np.inf
    return int(np.argmin(d2))


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
    OPD path in hopkins_eic_closing does not need it because prysm has the
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


def reference_sphere_curvature(P_xp, center):
    """Curvature kappa = 1/R of the chief-image reference sphere.

    The reference sphere is centered on the image point center and passes
    through the exit-pupil point P_xp, so R = |P_xp - center|.  Returns the
    curvature kappa = 1/R, which is the determinate handle the EIC closing
    uses: a finite exit pupil gives kappa > 0, and an exit pupil at infinity
    (image-space telecentric / afocal) is the kappa = 0 limit -- signalled by
    P_xp is None.  Parametrizing by curvature keeps the closing finite through
    that limit, where the radius R itself diverges.

    Parameters
    ----------
    P_xp : iterable or None
        exit-pupil point; None means the exit pupil is at infinity (kappa 0).
    center : iterable
        reference-sphere center -- the chief-ray image point.

    Returns
    -------
    float
        kappa, the reference-sphere curvature.

    """
    if P_xp is None:
        return 0.0
    R = float(np.sqrt(np.sum((np.asarray(P_xp) - np.asarray(center)) ** 2)))
    if R <= 1e-12:
        raise ValueError(
            'reference-sphere radius is degenerate (the exit pupil coincides '
            'with the image point); pass a separated P_xp'
        )
    return 1.0 / R


def hopkins_eic_closing(P_hist, S_hist, OPL_hist, *, center, curvature,
                        n_image=1.0, chief_index=None):
    """Determinate equally-inclined-chord OPD on the chief-image reference sphere.

    Closes the wavefront the way Hopkins (1981) does -- referencing every ray
    to the chief-ray image point through the equally-inclined chord -- but in a
    branch-free, cancellation-free form parametrized by the reference sphere's
    finite center and its curvature kappa = 1/R.  The segment from the last
    surface to the sphere, t = -b - sqrt(b^2 - |r|^2 + R^2) with r = P_last -
    center and b = S_last . r, is rewritten by shifting off the common radius:

        s = t + R = -b - kappa |r|^2-corrected-term, exactly
        s = -b - kappa * m / (1 + sqrt(1 + kappa^2 * m)),   m = b^2 - |r|^2.

    The common -R is ray-independent and drops out under the chief reference,
    so n_image * s is the reference-sphere OPD to machine precision for a
    finite pupil, while staying finite as the exit pupil recedes to infinity
    (kappa -> 0, s -> -b).  No explicit radius, no exit-pupil coordinate, no
    sqrt-of-negative clamp branch, no infinite-pupil special case -- the single
    formula spans infinite-conjugate, afocal, telecentric, and
    non-telecentric image space.  The chief-image reference focus is built in,
    so Hopkins' separate focal-shift term e' is implicit here.

    Parameters
    ----------
    P_hist, S_hist, OPL_hist : ndarray
        ray history from raytrace, shapes (jj+1, N, 3), (jj+1, N, 3), (jj+1, N).
    center : iterable
        chief-ray image point -- the reference-sphere center.
    curvature : float
        reference-sphere curvature kappa = 1/R (see reference_sphere_curvature);
        0 for an exit pupil at infinity.
    n_image : float
        index in image space.
    chief_index : int, optional
        row of the chief ray.  Default: the ray nearest the pupil center.

    Returns
    -------
    opd : ndarray, shape (N,)
        OPD relative to the chief (longer ray OPL -> positive), in the same
        length units as the surfaces.

    """
    C = np.asarray(center)
    P_last = P_hist[-1]
    S_last = S_hist[-1]
    OPL_through = OPL_hist.sum(axis=0)
    if chief_index is None:
        chief_index = _pupil_center_chief_index(P_hist[0])
    k = float(curvature)
    r = P_last - C
    b = (S_last * r).sum(axis=-1)
    m = b * b - (r * r).sum(axis=-1)
    disc = 1.0 + k * k * m
    # the same domain as the explicit root: disc = (b^2 - |r|^2 + R^2)/R^2 >= 0
    # exactly when the ray meets the sphere.  Clamp FP noise on grazing rays.
    min_disc = float(np.min(array_to_true_numpy(disc)))
    tol = 64.0 * np.finfo(np.asarray(disc).dtype).eps
    if min_disc < -tol:
        raise ValueError(
            'ray does not intersect the reference sphere; check P_xp/center '
            'or use the telecentric curvature=0 limit'
        )
    disc = np.where(disc < 0.0, 0.0, disc)
    s = -b - k * m / (1.0 + np.sqrt(disc))
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

def centroid_referenced_rms(x, y, *, axis=-1, center=None):
    """RMS distance from a per-slice center, reduced over the sample axis.

    The shared spot-size numeric core.  x and y are matching transverse
    coordinate arrays; with center None the reference is their centroid (the
    nanmean over axis), so the result is the centroid-referenced RMS radius.
    Pass center=(cx, cy) to measure the RMS about an explicit point instead.
    The centroid and the reduction are NaN-aware, so a bundle may carry
    NaN-padded invalid samples (the grid spot metrics over a SpotGrid) or be
    pre-filtered down to its valid rays (the single-bundle spot metrics).

    Parameters
    ----------
    x, y : ndarray
        transverse coordinate arrays of the same shape.
    axis : int, optional
        the sample axis reduced; default the trailing axis.
    center : tuple of (ndarray or float), optional
        explicit (cx, cy) reference; default the per-slice centroid.

    Returns
    -------
    ndarray
        RMS radius with the sample axis removed.

    """
    x = np.asarray(x)
    y = np.asarray(y)
    if center is None:
        cx = np.nanmean(x, axis=axis, keepdims=True)
        cy = np.nanmean(y, axis=axis, keepdims=True)
    else:
        cx, cy = center
    dx = x - cx
    dy = y - cy
    return np.sqrt(np.nanmean(dx * dx + dy * dy, axis=axis))


def spot_centroid(P_final, status=None):
    """Mean (x, y) position of valid rays at a surface plane.

    Parameters
    ----------
    P_final : ndarray
        shape (N, 3) — typically `raytrace(...).P[-1]`.
    status : ndarray, optional
        per-ray complex status from `raytrace` (or any equivalent).  If
        provided, rays rejected by valid_mask are excluded.

    Returns
    -------
    ndarray
        shape (2,) — mean (x, y) of the valid rays.  Returns `[nan, nan]`
        if no rays are valid.

    """
    P_final = np.asarray(P_final)
    valid = valid_mask(status, P_final)
    if valid is not None:
        P_final = P_final[valid]
    if P_final.shape[0] == 0:
        return np.array([np.nan, np.nan], dtype=P_final.dtype)
    return P_final[..., :2].mean(axis=0)


def rms_spot_radius(P_final, status=None, centroid=None):
    """RMS distance of valid rays from their centroid.

    Parameters
    ----------
    P_final : ndarray
        shape (N, 3) ray positions.
    status : ndarray, optional
        per-ray complex status; rays rejected by valid_mask are excluded.
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
    valid = valid_mask(status, P_final)
    if valid is not None:
        P_final = P_final[valid]
    if P_final.shape[0] == 0:
        return float('nan')
    center = None if centroid is None else tuple(np.asarray(centroid))
    return float(centroid_referenced_rms(
        P_final[..., 0], P_final[..., 1], axis=0, center=center))


def geometric_psf_histogram(P_final, status=None, bins=64, extent=None):
    """2D histogram of valid rays' (x, y) positions — the geometric PSF.

    Parameters
    ----------
    P_final : ndarray
        shape (N, 3) ray positions.
    status : ndarray, optional
        per-ray complex status; rays rejected by valid_mask are excluded.
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
    valid = valid_mask(status, P_final)
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
