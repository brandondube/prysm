"""Ray/surface intersection helpers for sequential raytracing."""

from prysm.mathops import np, row_dot

from .spencer_and_murty import (
    DEFAULT_TOL_SAG,
    SURFACE_INTERSECTION_DEFAULT_MAXITER,
    intersect as newton_intersect,
    newton_raphson_solve_s,
    resolve_tol_sag,
)
from .sags import conic_sag_and_normal

# Grazing-incidence floor for departure-band width D/|cos(theta_inc)|.
COS_INCIDENCE_FLOOR = 1e-3

# Slack in the monotonicity certificate cos_inc - G*|S_t| > margin.
CERTIFICATE_MARGIN = 1e-3


def ray_plane_intersect(P, S):
    """Intersect rays P + t*S with the local-frame plane Z = 0.

    Parameters
    ----------
    P : ndarray
        shape (N, 3) ray origins in the surface's local frame
    S : ndarray
        shape (N, 3) unit direction cosines
    Returns
    -------
    Q : ndarray
        shape (N, 3) intersection points
    n : ndarray
        shape (N, 3) unit surface normals.
    valid : ndarray
        shape (N,) boolean.

    """
    P, S = np.atleast_2d(P, S)
    Sz = S[..., 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        t = -P[..., 2] / Sz
        Q = P + t[..., np.newaxis] * S
    n = np.zeros(Q.shape, dtype=Q.dtype)
    n[..., 2] = 1.0
    return Q, n, (Sz != 0)


def _conic_quadratic_coeffs(c, kappa, P1, S, dx, dy):
    """Coefficients of A t^2 + 2 B t + C for conic intersection."""
    Sx = S[..., 0]
    Sy = S[..., 1]
    Sz = S[..., 2]
    Xp = P1[..., 0] + dx
    Yp = P1[..., 1] + dy
    A_ = 1.0 + kappa * Sz * Sz
    B_ = Xp * Sx + Yp * Sy - Sz / c
    C_ = Xp * Xp + Yp * Yp
    return A_, B_, C_


def _conic_quadratic_t(c, kappa, P1, S, dx, dy):
    """Solve the conic-intersection quadratic for the vertex-side root.

    Uses Welford's rationalized form.  Returns (t, disc_nonneg).

    """
    Sz = S[..., 2]
    A_, B_, C_ = _conic_quadratic_coeffs(c, kappa, P1, S, dx, dy)
    disc = B_ * B_ - A_ * C_
    disc_nonneg = (disc >= 0)
    disc = np.where(disc_nonneg, disc, 0.0)
    sqrt_disc = np.sqrt(disc)
    # z_dir selects the vertex-side root along the ray propagation sense.
    # sign(c) comes from factoring c through Welford's rationalized form.
    sign_c = 1.0 if c > 0 else -1.0
    z_dir = np.ones_like(Sz)
    z_dir[Sz < 0] = -1.0
    denom = z_dir * sign_c * sqrt_disc - B_
    # denom == 0 only for a ray tangent to the surface at the vertex
    # (B_ == 0 and disc == 0); the intersection is the vertex itself, t = 0.
    vertex_tangent = (denom == 0)
    safe_denom = np.where(vertex_tangent, 1.0, denom)
    t = C_ / safe_denom
    t = np.where(vertex_tangent, 0.0, t)
    return t, disc_nonneg


def ray_conic_intersect(P, S, c, kappa, dx=0.0, dy=0.0):
    """Intersect rays P + t*S with a (possibly off-axis) conicoid.

    Parameters
    ----------
    P : ndarray
        shape (N, 3) ray origins in the surface's local frame
    S : ndarray
        shape (N, 3) unit direction cosines
    c : float
        vertex curvature (1 / R)
    kappa : float
        conic constant (-1 = parabola, 0 = sphere, < -1 = hyperbola, > -1 = ellipse)
    dx, dy : float
        off-axis shift of the surface relative to the parent conic vertex;
        dx = dy = 0 for an on-axis surface

    Returns
    -------
    Q : ndarray
        shape (N, 3) intersection points
    n : ndarray
        shape (N, 3) unit surface normals.
    valid : ndarray
        shape (N,) boolean.

    """
    if c == 0.0:
        # c=0 is a plane; dx/dy are meaningless.
        return ray_plane_intersect(P, S)
    P, S = np.atleast_2d(P, S)
    Sz = S[..., 2]
    # invalid arithmetic (1/0, sqrt(-x)) is expected for rays that miss the
    # surface; we expose those via the valid mask, not via FP warnings.
    with np.errstate(divide='ignore', invalid='ignore'):
        # project P onto the vertex tangent plane (matches Newton path convention)
        s0 = -P[..., 2] / Sz
        P1 = P + s0[..., np.newaxis] * S
        t, disc_nonneg = _conic_quadratic_t(c, kappa, P1, S, dx, dy)
        Q = P1 + t[..., np.newaxis] * S
        Xq = Q[..., 0] + dx
        Yq = Q[..., 1] + dy
        phi_arg = 1.0 - (1.0 + kappa) * c * c * (Xq * Xq + Yq * Yq)
        _, n = conic_sag_and_normal(c, kappa, Xq, Yq)
    return Q, n, disc_nonneg & (phi_arg >= 0)


def ray_sphere_intersect(P, S, c):
    """Intersect rays P + t*S with a sphere of curvature c.

    Parameters
    ----------
    P : ndarray
        shape (N, 3) ray origins
    S : ndarray
        shape (N, 3) unit direction cosines
    c : float
        vertex curvature (1 / R)
    Returns
    -------
    Q : ndarray
        intersection points
    n : ndarray
        unit surface normals.
    valid : ndarray
        shape (N,) boolean.

    """
    return ray_conic_intersect(P, S, c, 0.0)


# Cap on Lipschitz-march steps before a ray is rejected.
LIPSCHITZ_MARCH_MAXSTEPS = 256
# Switch from Lipschitz descent to local Newton near the first root.
NEWTON_SWITCH_FRACTION = 1e-2
# Radial margin applied to the characterized domain during the march.
MARCH_RADIUS_MARGIN = 1.1


def _domain_corridor(P1, S, s_lo, s_hi, domain_radius, dtype):
    """Clip each ray's band to where its transverse radius stays <= R.

    Rays that never enter the disk return with lo > hi.
    """
    Sx = S[..., 0]
    Sy = S[..., 1]
    Px = P1[..., 0]
    Py = P1[..., 1]
    a = Sx * Sx + Sy * Sy
    b = Px * Sx + Py * Sy
    c = Px * Px + Py * Py - domain_radius * domain_radius
    lo = np.array(s_lo, dtype=dtype).copy()
    hi = np.array(s_hi, dtype=dtype).copy()
    with np.errstate(divide='ignore', invalid='ignore'):
        # axial-ish ray: constant radius, so either all in or all out.
        disc = b * b - a * c
        sqrt_disc = np.sqrt(np.maximum(disc, 0.0))
        s_a = (-b - sqrt_disc) / a
        s_b = (-b + sqrt_disc) / a
    swept = a > 0
    real = swept & (disc >= 0)
    # Tighten swept rays to disk entry/exit.
    lo = np.where(real, np.maximum(lo, s_a), lo)
    hi = np.where(real, np.minimum(hi, s_b), hi)
    # Swept miss, or axial ray outside the disk: empty.
    empty = (swept & ~real) | (~swept & (c > 0))
    hi = np.where(empty, lo - 1.0, hi)
    return lo, hi


def _lipschitz_march_solve_s(sag_and_normal, P1, S, s_lo, s_hi,
                             sag_lipschitz, tol_sag, maxiter, domain_radius=None):
    """First-root solve by Lipschitz (sphere-tracing) descent from the floor.

    Steps abs(F) / Lip from s_lo and switches to local Newton near the root.

    """
    dtype = P1.dtype
    nrays = P1.shape[0]
    Pj_out = np.full((nrays, 3), np.nan, dtype=dtype)
    n_out = np.full((nrays, 3), np.nan, dtype=dtype)
    valid = np.zeros(nrays, dtype=bool)

    if domain_radius is not None:
        s_lo, s_hi = _domain_corridor(P1, S, s_lo, s_hi,
                                      MARCH_RADIUS_MARGIN * domain_radius, dtype)

    Sz = S[..., 2]
    S_t = np.sqrt(np.maximum(0.0, 1.0 - Sz * Sz))
    Lip = np.abs(Sz) + sag_lipschitz * S_t
    # Lip == 0 only for an in-plane ray over locally flat sag.
    Lip = np.where(Lip > 0.0, Lip, 1.0)

    # Drop rays with empty corridors, then keep a shrinking active set.
    keep = np.flatnonzero(np.asarray(s_lo <= s_hi))
    if keep.size == 0:
        return Pj_out, n_out, valid
    mask = keep
    P1_w = P1[keep]
    S_w = S[keep]
    Lip_w = Lip[keep]
    lo_w = np.array(s_lo, dtype=dtype)[keep].copy()
    hi_w = np.array(s_hi, dtype=dtype)[keep].copy()
    s = lo_w.copy()
    for _ in range(maxiter):
        Pj = P1_w + s[:, np.newaxis] * S_w
        with np.errstate(divide='ignore', invalid='ignore'):
            sagj, n_hat = sag_and_normal(Pj[..., 0], Pj[..., 1])
        Fj = Pj[..., 2] - sagj
        converged = np.abs(Fj) < tol_sag
        if converged.any():
            ins = mask[converged]
            Pj_out[ins] = Pj[converged]
            n_out[ins] = n_hat[converged]
            valid[ins] = True
        with np.errstate(divide='ignore', invalid='ignore'):
            step_lip = np.abs(Fj) / Lip_w
            # local Newton slope F'(s) = S . n_hat / n_hat_z
            Fp = row_dot(S_w, n_hat) / n_hat[..., 2]
            step_newton = -Fj / Fp
        # Switch to Newton only near the root and away from tangency.
        near = (np.isfinite(step_newton)
                & (np.abs(Fp) > COS_INCIDENCE_FLOOR)
                & (step_lip < NEWTON_SWITCH_FRACTION * (1.0 + np.abs(s))))
        s_new = np.where(near, s + step_newton, s + step_lip)
        # Clamp Newton to the corridor; descent alone detects passing s_hi.
        s_new = np.minimum(np.maximum(s_new, lo_w), hi_w)
        exhausted = (~near) & ~converged & (s + step_lip > hi_w)
        s = s_new
        survive = ~converged & ~exhausted & np.isfinite(Fj)
        if survive.all():
            continue
        mask = mask[survive]
        if mask.size == 0:
            break
        P1_w = P1_w[survive]
        S_w = S_w[survive]
        Lip_w = Lip_w[survive]
        lo_w = lo_w[survive]
        hi_w = hi_w[survive]
        s = s[survive]
    return Pj_out, n_out, valid


def bracketed_newton_solve_s(P1, S, sag_and_normal, s_lo, s_hi,
                             tol_sag=None,
                             maxiter=SURFACE_INTERSECTION_DEFAULT_MAXITER,
                             lipschitz=None, domain_radius=None):
    """First-root solve in a band by Lipschitz (sphere-tracing) descent.

    The Lipschitz bound makes the march provably unable to step over the first
    root, so it needs no segment scan or bracket-refinement heuristics.

    Parameters
    ----------
    P1 : ndarray
        shape (N, 3) ray origins on the surface vertex plane.
    S : ndarray
        shape (N, 3) unit direction cosines.
    sag_and_normal : callable
        function returning surface sag and unit normal at x, y.
    s_lo, s_hi : ndarray
        shape (N,) search band endpoints, path length along each ray from P1.
    tol_sag : float, optional
        absolute convergence tolerance on the surface residual Z - sag.
    maxiter : int, optional
        maximum number of iterations per solve.
    lipschitz : float
        max abs(grad sag) over the domain; required -- it is what guarantees the
        march finds the first root.
    domain_radius : float, optional
        radius of the characterized disk; clips the march to where the bound
        holds.

    Returns
    -------
    Pj, n_hat, valid : ndarray, ndarray, ndarray
        intersection points, unit surface normals, and a length-N boolean
        convergence mask.  Failed rays are NaN.

    """
    if lipschitz is None:
        raise ValueError(
            'bracketed_newton_solve_s requires a lipschitz bound (max |grad '
            'sag| over the domain); it guarantees the first-root march.')
    dtype = P1.dtype
    tol_sag = resolve_tol_sag(tol_sag, dtype)
    s_lo = np.asarray(s_lo, dtype=dtype)
    s_hi = np.asarray(s_hi, dtype=dtype)
    steps = max(maxiter, LIPSCHITZ_MARCH_MAXSTEPS)
    return _lipschitz_march_solve_s(sag_and_normal, P1, S, s_lo, s_hi,
                                    lipschitz, tol_sag, steps,
                                    domain_radius=domain_radius)


class ConicSeedMixin:
    """Mixin for shapes whose Newton solve should start from a conic root.

    Optional departure bounds police the conic-seeded Newton result and route
    uncertified rays through the bracketed/Lipschitz first-root rescue.

    """

    def seed_conic(self):
        p = self.params
        return p['c'], p['k'], p.get('dx', 0.0), p.get('dy', 0.0)

    def intersect(self, P, S, sag_and_normal,
                  tol_sag=None, maxiter=None,
                  departure=None, domain_radius=None,
                  departure_gradient=None, sag_lipschitz=None,
                  forward_only=False):
        """Intersect rays with the shape via conic-seeded Newton iteration.

        Parameters
        ----------
        P : ndarray
            shape (N, 3) ray origins in the surface local frame.
        S : ndarray
            shape (N, 3) unit direction cosines.
        sag_and_normal : callable
            function returning surface sag and unit normal at x, y.
        tol_sag : float, optional
            absolute convergence tolerance on the surface residual.
        maxiter : int, optional
            maximum Newton iterations.
        departure : float, optional
            max abs(sag - seed conic sag) over the characterized domain; enables
            the departure-band acceptance.
        domain_radius : float, optional
            radius of the characterized domain; the band is only policed for
            rays whose conic-seed hit falls inside it (outside, departure does
            not bound the surface and convergence alone decides).
        departure_gradient : float, optional
            max abs(grad(sag - seed conic)) over the domain.
        sag_lipschitz : float, optional
            max abs(grad sag) over the domain; enables the Lipschitz rescue.
        forward_only : bool, optional
            when True, reject roots behind the ray origin.  Only sound for the
            forward-propagating (non-folded) configurations the conic seed
            covers; mirror folds legitimately produce behind-origin local roots,
            so Surface.intersect does not pass this for analytic mirror shapes.

        Returns
        -------
        Q, n, valid : ndarray, ndarray, ndarray
            intersection points, unit surface normals, and a length-N boolean
            acceptance mask.

        """
        if maxiter is None:
            maxiter = SURFACE_INTERSECTION_DEFAULT_MAXITER
        P, S = np.atleast_2d(P, S)
        Sz = S[..., 2]
        with np.errstate(divide='ignore', invalid='ignore'):
            s0 = -P[..., 2] / Sz
            P1 = P + s0[..., np.newaxis] * S
            c_seed, k_seed, dx_seed, dy_seed = self.seed_conic()
            Q_conic, n_conic, hit_conic = ray_conic_intersect(
                P1, S, c_seed, k_seed, dx=dx_seed, dy=dy_seed)
            s1 = Q_conic[..., 2] / Sz
        Q, n, valid = newton_raphson_solve_s(P1, S, sag_and_normal,
                                             s1=s1, tol_sag=tol_sag,
                                             maxiter=maxiter)
        if departure is None and not forward_only:
            return Q, n, valid

        tol = resolve_tol_sag(tol_sag, P1.dtype)
        # Recover the Newton root parameter; NaNs fail the band compare.
        s_root = row_dot(Q - P1, S)

        if departure is not None and domain_radius is not None:
            cosi = np.abs(row_dot(S, n_conic))
            # Monotonicity certificate on the unfloored seed incidence.
            if departure_gradient is not None:
                S_t = np.sqrt(np.maximum(0.0, 1.0 - Sz * Sz))
                with np.errstate(invalid='ignore'):
                    certified = (cosi - departure_gradient * S_t) > CERTIFICATE_MARGIN
            else:
                certified = np.ones(cosi.shape, dtype=bool)
            # Grazing/NaN incidence gets the widest finite band.
            floored = ~(cosi >= COS_INCIDENCE_FLOOR)
            cosi[floored] = COS_INCIDENCE_FLOOR
            # Slack for Newton convergence noise in near-zero departure bands.
            band = (departure + 100.0 * tol * (1.0 + np.abs(s1))) / cosi
            rseed_sq = (Q_conic[..., 0] * Q_conic[..., 0]
                        + Q_conic[..., 1] * Q_conic[..., 1])
            seed_ok = hit_conic & np.isfinite(s1)
            police = seed_ok & (rseed_sq <= domain_radius * domain_radius)
            with np.errstate(invalid='ignore'):
                in_band = np.abs(s_root - s1) <= band
                # Departure bounds do not certify roots outside the domain.
                rroot_sq = (Q[..., 0] * Q[..., 0] + Q[..., 1] * Q[..., 1])
                in_domain = rroot_sq <= domain_radius * domain_radius
            # Preserve roots the previous band-only guard would have accepted.
            old_anchorless = ~seed_ok & ~in_domain
            prior_accept = (valid & (~police | (in_band & in_domain))
                            & ~old_anchorless)
            # Certified roots skip the rescue.
            certified_accept = valid & police & in_band & in_domain & certified
            accept = certified_accept
            # Rescue roots that are not certified.
            rescue = police & ~certified_accept
            lo = s1 - band
            hi = s1 + band
            if (~seed_ok).any() and c_seed != 0.0:
                # Closest-approach band for rays whose seed conic misses.
                A_, B_, C_ = _conic_quadratic_coeffs(c_seed, k_seed, P1, S,
                                                     dx_seed, dy_seed)
                z_max = abs(c_seed) * domain_radius * domain_radius / 2.0 \
                    + departure
                scale = 2.0 / abs(c_seed) + 2.0 * abs(1.0 + k_seed) * z_max
                d_imp = (departure + 100.0 * tol) * scale
                with np.errstate(divide='ignore', invalid='ignore'):
                    t_star = -B_ / A_
                    c_min = C_ - B_ * B_ / A_
                    wsq = (d_imp - c_min) / A_
                    rescuable = (~seed_ok & (A_ > 0) & (wsq >= 0)
                                 & np.isfinite(t_star))
                    w = np.sqrt(np.abs(wsq))
                lo[rescuable] = t_star[rescuable] - w[rescuable]
                hi[rescuable] = t_star[rescuable] + w[rescuable]
                rescue = rescue | rescuable
            if rescue.any():
                Qr, nr, vr = bracketed_newton_solve_s(
                    P1[rescue], S[rescue], sag_and_normal,
                    lo[rescue], hi[rescue],
                    tol_sag=tol_sag, maxiter=maxiter,
                    lipschitz=sag_lipschitz, domain_radius=domain_radius)
                ridx = np.flatnonzero(rescue)
                # The rescue wins where it converged.
                conv = ridx[vr]
                Q[conv] = Qr[vr]
                n[conv] = nr[vr]
                accept[conv] = True
                # If rescue stalls, preserve previous band-only accepts.
                stalled = ridx[~vr]
                accept[stalled[prior_accept[stalled]]] = True
                s_root = row_dot(Q - P1, S)
            # Non-rescued previous accepts keep their Newton root.
            accept = accept | (prior_accept & ~rescue)
            valid = accept

        if forward_only:
            with np.errstate(invalid='ignore'):
                backward = (s0 + s_root) < (-100.0 * tol * (1.0 + np.abs(s0)))
            valid = valid & ~backward
        return Q, n, valid


__all__ = [
    'COS_INCIDENCE_FLOOR',
    'CERTIFICATE_MARGIN',
    'LIPSCHITZ_MARCH_MAXSTEPS',
    'SURFACE_INTERSECTION_DEFAULT_MAXITER',
    'DEFAULT_TOL_SAG',
    'newton_intersect',
    'newton_raphson_solve_s',
    'bracketed_newton_solve_s',
    'ray_plane_intersect',
    'ray_sphere_intersect',
    'ray_conic_intersect',
    'ConicSeedMixin',
]
