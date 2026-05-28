"""Ray/surface intersection helpers for sequential raytracing."""

from prysm.mathops import np

from .spencer_and_murty import (
    DEFAULT_TOL_SAG,
    SURFACE_INTERSECTION_DEFAULT_MAXITER,
    intersect as newton_intersect,
    newton_raphson_solve_s,
)
from .sags import conic_sag_and_normal


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


def _conic_quadratic_t(c, kappa, P1, S, dx, dy):
    """Solve the conic-intersection quadratic for the vertex-side root.

    Uses Welford's rationalized form t = C / (z_dir*sqrt(D) - B), which
    selects the intersection nearest the vertex in the propagation
    direction (z_dir = sign(Sz)).  The quadratic A t^2 + 2 B t + C = 0 has
    roots (-B +/- sqrt(D))/A; the rationalized form is the algebraically
    identical C / (-B -/+ sqrt(D)) and is preferred because its denominator
    is a *sum* of like-signed terms, avoiding the subtractive cancellation
    of the (-B +/- sqrt(D))/A form when B and sqrt(D) are close in
    magnitude (steep conics, decentered or near-grazing rays).  Choosing the
    sign of sqrt(D) by z_dir picks the first surface crossing along +S
    rather than relying on a smaller-|t| heuristic, which can select the
    wrong sheet for tilted/decentered geometry.

    Assumes P1 sits on the surface's vertex tangent plane (Z=0).  Returns
    (t, disc_nonneg) where disc_nonneg flags rays whose quadratic had a real
    root.  Caller must wrap in np.errstate(divide='ignore', invalid='ignore')
    since this routine silently swallows 1/0 and sqrt(<0) for missed rays.

    The A_ == 0 paraboloid-axial-ray degeneracy that the older
    (-B +/- sqrt(D))/A form had to special-case does not arise here: this
    form never divides by A_.

    """
    Sx = S[..., 0]
    Sy = S[..., 1]
    Sz = S[..., 2]
    Xp = P1[..., 0] + dx
    Yp = P1[..., 1] + dy
    A_ = 1.0 + kappa * Sz * Sz
    B_ = Xp * Sx + Yp * Sy - Sz / c
    C_ = Xp * Xp + Yp * Yp
    disc = B_ * B_ - A_ * C_
    disc_nonneg = (disc >= 0)
    disc = np.where(disc_nonneg, disc, np.zeros_like(disc))
    sqrt_disc = np.sqrt(disc)
    # z_dir picks the vertex-side root along the ray's propagation sense;
    # sign(0) -> +1 so a ray travelling in the tangent plane still resolves.
    # sign(c) enters because factoring c out of Welford's rationalized form
    # leaves a |c| under the radical: sqrt(disc_Welford) = |c| sqrt(disc_here),
    # so dividing through by c carries (|c|/c) = sign(c) onto the root.
    sign_c = 1.0 if c > 0 else -1.0
    z_dir = np.where(Sz < 0, -np.ones_like(Sz), np.ones_like(Sz))
    denom = z_dir * sign_c * sqrt_disc - B_
    # denom == 0 only for a ray tangent to the surface at the vertex
    # (B_ == 0 and disc == 0); the intersection is the vertex itself, t = 0.
    safe_denom = np.where(denom == 0, np.ones_like(denom), denom)
    t = C_ / safe_denom
    t = np.where(denom == 0, np.zeros_like(t), t)
    return t, disc_nonneg


def ray_conic_intersect(P, S, c, kappa, dx=0.0, dy=0.0):
    """Intersect rays P + t*S with a (possibly off-axis) conicoid.

    Surface implicit form (from sag z = c*A / (1 + sqrt(1 - (1+k)c^2 A))
    with A = (X+dx)^2 + (Y+dy)^2):

        (X + dx)^2 + (Y + dy)^2 + (1 + k) Z^2 - 2 Z / c = 0

    Substituting X = Px + t Sx, ..., yields a quadratic in t.  This
    routine first projects P onto the vertex tangent plane (Z = 0) via
    P1 = P - (Pz / Sz) S, matching the convention used by the Newton-Raphson
    path in spencer_and_murty.intersect.  In the projected frame Pz = 0, so
        A_ = 1 + k Sz^2
        B_ = (P1x + dx) Sx + (P1y + dy) Sy - Sz / c
        C_ = (P1x + dx)^2 + (P1y + dy)^2

    The vertex-side root in the propagation direction is selected via
    Welford's rationalized form (see _conic_quadratic_t), which is robust
    to the subtractive cancellation and the paraboloid-axial-ray
    degeneracy of the naive (-B +/- sqrt(D))/A quadratic.

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
        # The conic z = c A / (1 + sqrt(1 - (1+k) c^2 A)) collapses to z = 0
        # when c=0; the closed-form quadratic divides by c, so dispatch to the
        # plane intersector to avoid 1/0.  The off-axis shift dx/dy is
        # meaningless for c=0 (a plane has no vertex) and is silently ignored.
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
    """Intersect rays P + t*S with a sphere of curvature c, vertex at origin.

    Thin wrapper over ray_conic_intersect with kappa=0.

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


class ConicSeedMixin:
    """Mixin for shapes whose Newton solve should start from a conic root."""

    def seed_conic(self):
        p = self.params
        return p['c'], p['k'], p.get('dx', 0.0), p.get('dy', 0.0)

    def intersect(self, P, S, sag_and_normal,
                  tol_sag=DEFAULT_TOL_SAG, maxiter=None):
        if maxiter is None:
            maxiter = SURFACE_INTERSECTION_DEFAULT_MAXITER
        P, S = np.atleast_2d(P, S)
        Sz = S[..., 2]
        with np.errstate(divide='ignore', invalid='ignore'):
            s0 = -P[..., 2] / Sz
            P1 = P + s0[..., np.newaxis] * S
            c_seed, k_seed, dx_seed, dy_seed = self.seed_conic()
            Q_conic, _, _ = ray_conic_intersect(P1, S, c_seed, k_seed,
                                                dx=dx_seed, dy=dy_seed)
            s1 = Q_conic[..., 2] / Sz
        return newton_raphson_solve_s(P1, S, sag_and_normal,
                                      s1=s1, tol_sag=tol_sag,
                                      maxiter=maxiter)

__all__ = [
    'SURFACE_INTERSECTION_DEFAULT_MAXITER',
    'DEFAULT_TOL_SAG',
    'newton_intersect',
    'newton_raphson_solve_s',
    'ray_plane_intersect',
    'ray_sphere_intersect',
    'ray_conic_intersect',
    'ConicSeedMixin',
]
