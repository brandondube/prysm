"""Forward-with-intermediates pass + reverse-mode (adjoint) sweep.

The forward pass re-runs the nominal Spencer & Murty trace surface by surface,
saving the per-surface nominal quantities the adjoint primitives need (local
origin / direction / intersection / normal, post-bend direction, the sag
Hessian, indices).  The backward sweep then walks the surfaces in reverse,
propagating a single merit cotangent (P_bar, S_bar, L_bar) and contracting the
pose / shape / index cotangents with the assembled seed arrays to build the
gradient w.r.t. every tolerance parameter at once.
"""

from prysm.conf import config
from prysm.mathops import np

from prysm.x.raytracing.spencer_and_murty import (
    raytrace,
    STYPE_REFRACT,
    STYPE_REFLECT,
)
from prysm.x.raytracing._diff_raytrace import _assemble_seeds, _eye3
from prysm.x.raytracing.opt import _valid_mask

from .primitives import (
    adj_opl_segment,
    adj_transform_global,
    adj_refract,
    adj_reflect,
    adj_intersect,
    adj_transform_local,
)

# safe dummy nominals substituted for clipped/failed rays so that their (zero)
# cotangents propagate as exact zeros instead of 0 * NaN.
_DUMMY_DIR = np.array([0.0, 0.0, 1.0], dtype=config.precision)


class SurfaceIntermediate:
    """Nominal quantities saved at one surface for the backward sweep."""

    __slots__ = ('Reff', 'Q', 'P_in', 'S_in', 'P0', 'S_loc', 'Q_loc',
                 'n_hat', 'Sprime', 'hessian', 'seg', 'typ', 'n_pre',
                 'nprime', 'index_source')

    def __init__(self, Reff, Q, P_in, S_in, P0, S_loc, Q_loc, n_hat, Sprime,
                 hessian, seg, typ, n_pre, nprime, index_source):
        self.Reff = Reff
        self.Q = Q
        self.P_in = P_in        # global incoming position (trace.P[j])
        self.S_in = S_in        # global incoming direction (trace.S[j])
        self.P0 = P0
        self.S_loc = S_loc
        self.Q_loc = Q_loc
        self.n_hat = n_hat
        self.Sprime = Sprime
        self.hessian = hessian
        self.seg = seg          # global segment trace.P[j+1] - trace.P[j]
        self.typ = typ
        self.n_pre = n_pre      # index of the medium preceding the surface
        self.nprime = nprime    # following index (refract only; else None)
        self.index_source = index_source  # surface that set n_pre, or -1


class TraceIntermediates:
    """Per-surface intermediates plus the global valid mask."""

    __slots__ = ('surfaces', 'valid')

    def __init__(self, surfaces, valid):
        self.surfaces = surfaces
        self.valid = valid


def _sanitize_vec(arr, valid, dummy):
    """Replace invalid-ray rows of an (N, 3) array with a finite dummy."""
    out = np.where(valid[:, None], arr, dummy[None, :])
    return out.astype(config.precision)


def _sanitize_scalar(arr, valid):
    return np.where(valid, arr, np.zeros_like(arr)).astype(config.precision)


def _forward_with_intermediates(surfaces, P, S, wvl, n_ambient=1.0,
                                tol_sag=None):
    """Nominal trace + saved per-surface intermediates for the adjoint.

    Returns (RayTraceResult, TraceIntermediates).
    """
    P = np.asarray(P)
    S = np.asarray(S)
    if P.ndim == 1:
        P = P[None, :]
        S = S[None, :]
    P = P.astype(config.precision)
    S = S.astype(config.precision)

    trace = raytrace(surfaces, P, S, wvl, n_ambient=n_ambient, tol_sag=tol_sag)
    valid = _valid_mask(trace.status, trace.P[-1])

    inters = []
    nj = float(n_ambient)
    index_source = -1
    for j, surf in enumerate(surfaces):
        Reff = _eye3() if surf.R is None else np.asarray(surf.R, dtype=config.precision)
        Q = np.asarray(surf.P, dtype=config.precision)
        P_in = trace.P[j]
        S_in = trace.S[j]

        # Step I (local) recomputed for the nominal intermediates.
        Pmq = P_in - Q
        P0 = (Reff @ Pmq.T).T
        S_loc = (Reff @ S_in.T).T

        Q_loc, n_hat, _ = surf.intersect(P0, S_loc, tol_sag=tol_sag)
        Xj = Q_loc[..., 0]
        Yj = Q_loc[..., 1]
        hessian = surf.shape.sag_hessian(Xj, Yj)

        if surf.typ == STYPE_REFRACT:
            nprime = float(surf.n(wvl))
            cosI = np.sum(n_hat * S_loc, axis=-1)
            mu = nj / nprime
            factor = (np.sign(cosI) * np.sqrt(1.0 - mu * mu * (1.0 - cosI * cosI))
                      - mu * cosI)
            Sprime = mu * S_loc + factor[:, None] * n_hat
        elif surf.typ == STYPE_REFLECT:
            nprime = None
            cosI = np.sum(S_loc * n_hat, axis=-1)
            Sprime = S_loc - 2.0 * cosI[:, None] * n_hat
        else:
            nprime = None
            Sprime = S_loc

        seg = trace.P[j + 1] - trace.P[j]

        # sanitize NaNs from clipped/failed rays
        P0 = _sanitize_vec(P0, valid, np.zeros(3, dtype=config.precision))
        S_loc = _sanitize_vec(S_loc, valid, _DUMMY_DIR)
        Q_loc = _sanitize_vec(Q_loc, valid, np.zeros(3, dtype=config.precision))
        n_hat = _sanitize_vec(n_hat, valid, _DUMMY_DIR)
        Sprime = _sanitize_vec(Sprime, valid, _DUMMY_DIR)
        seg = _sanitize_vec(seg, valid, _DUMMY_DIR)
        hessian = tuple(_sanitize_scalar(h, valid) for h in hessian)

        inters.append(SurfaceIntermediate(
            Reff, Q, P_in, S_in, P0, S_loc, Q_loc, n_hat, Sprime, hessian,
            seg, surf.typ, nj, nprime, index_source))

        if surf.typ == STYPE_REFRACT:
            nj = nprime
            index_source = j

    return trace, TraceIntermediates(inters, valid)


def _precompute_shape_partials(surfaces, intermediates, shape_params,
                               sag_partial_fns):
    """Per-surface (param_index, sag_t, gx_t, gy_t) tangents for the shape DOFs.

    sag_param_partials depends only on the surface shape and its nominal
    intersection points, not on the merit cotangent, so it is identical across
    every backward sweep.  Evaluating it once here lets multi_objective_sensi-
    tivity reuse it for all M objectives instead of re-running it per head --
    for freeform shapes whose base-class partials are themselves finite-
    difference sag evaluations, that is the dominant cost of the M-sweep
    Jacobian.  Returns a list parallel to surfaces; each entry is a (possibly
    empty) list of (param_index, sag_t, gx_t, gy_t) tuples.
    """
    surf_inters = intermediates.surfaces
    per_surface = []
    for j in range(len(surfaces)):
        entries = []
        if shape_params[j] or sag_partial_fns[j]:
            Xj = surf_inters[j].Q_loc[..., 0]
            Yj = surf_inters[j].Q_loc[..., 1]
            for p, pname in shape_params[j]:
                sag_t, gx_t, gy_t = surfaces[j].shape.sag_param_partials(
                    Xj, Yj, pname)
                entries.append((p, sag_t, gx_t, gy_t))
            for p, fn in sag_partial_fns[j]:
                sag_t, gx_t, gy_t = fn(Xj, Yj)
                entries.append((p, sag_t, gx_t, gy_t))
        per_surface.append(entries)
    return per_surface


def _backward_sweep(surfaces, trace, intermediates, Qdot_s, Rdot_s,
                    nprimedot_s, shape_params, sag_partial_fns,
                    cotangent_seed, n_ambient_dot=None, shape_partials=None):
    """One reverse sweep: cotangent seed -> gradient (P,) over all parameters.

    cotangent_seed is (P_bar, S_bar, L_bar): the merit's cotangent on the
    image-plane ray position, direction, and per-segment OPL (L_bar shared
    across all segments since merits here depend on the total OPL).

    shape_partials, if given, is the output of _precompute_shape_partials and
    supplies the cotangent-independent shape-DOF tangents; the sweep then only
    contracts them with its own dsag_bar/dgx_bar/dgy_bar instead of re-running
    sag_param_partials.  When None they are evaluated on the fly (single-sweep
    callers such as adjoint_gradient).
    """
    n_params = Qdot_s[0].shape[1] if Qdot_s else 0
    grad = np.zeros(n_params, dtype=config.precision)

    valid = intermediates.valid
    P_bar, S_bar, L_bar = cotangent_seed
    # zero the cotangent for clipped/failed rays
    P_bar = np.where(valid[:, None], P_bar, 0.0).astype(config.precision)
    S_bar = np.where(valid[:, None], S_bar, 0.0).astype(config.precision)
    L_bar = np.where(valid, L_bar, 0.0).astype(config.precision)

    surf_inters = intermediates.surfaces
    for j in range(len(surfaces) - 1, -1, -1):
        si = surf_inters[j]

        # --- OPL segment: depends on outgoing P[j+1] (+dseg) and incoming
        #     P[j] (-dseg); L_bar is shared across segments.
        n_bar, dseg_bar = adj_opl_segment(si.n_pre, si.seg, L_bar)
        P_bar_out = P_bar + dseg_bar          # cotangent of dPjp1
        P_in_from_opl = -dseg_bar             # carry onto incoming position

        # --- Step IV adjoint: to global
        dPj_bar, dSprime_bar, Qdot_bar_g, Rdot_bar_g = adj_transform_global(
            si.Reff, si.Q_loc, si.Sprime, P_bar_out, S_bar)

        # --- Step III adjoint: bend
        ndot_pre_bar = 0.0
        ndot_post_bar = 0.0
        if si.typ == STYPE_REFRACT:
            (S_locdot_bar_b, dn_hat_bar, ndot_pre_bar,
             ndot_post_bar) = adj_refract(si.n_pre, si.nprime, si.S_loc,
                                          si.n_hat, dSprime_bar)
        elif si.typ == STYPE_REFLECT:
            S_locdot_bar_b, dn_hat_bar = adj_reflect(si.S_loc, si.n_hat,
                                                     dSprime_bar)
        else:  # eval: Sprime = S_loc, no normal interaction
            S_locdot_bar_b = dSprime_bar
            dn_hat_bar = np.zeros_like(dSprime_bar)

        # --- Step II adjoint: intersect
        (P0dot_bar, S_locdot_bar_i, dsag_bar, dgx_bar,
         dgy_bar) = adj_intersect(si.P0, si.S_loc, si.Q_loc, si.n_hat,
                                  si.hessian, dPj_bar, dn_hat_bar)
        S_locdot_bar = S_locdot_bar_b + S_locdot_bar_i

        # --- Step I adjoint: to local
        Pdot_bar, Sdot_bar, Qdot_bar_l, Rdot_bar_l = adj_transform_local(
            si.Reff, si.P_in, si.Q, si.S_in, P0dot_bar, S_locdot_bar)
        Pdot_bar = Pdot_bar + P_in_from_opl

        # --- accumulate parameter gradient -----------------------------
        Qdot_bar = Qdot_bar_g + Qdot_bar_l
        Rdot_bar = Rdot_bar_g + Rdot_bar_l
        grad = grad + Qdot_bar @ Qdot_s[j]
        grad = grad + np.tensordot(Rdot_bar, Rdot_s[j], axes=([0, 1], [0, 1]))

        # shape DOFs (analytic sag_param_partials).  The tangents are cotangent-
        # independent, so reuse the precomputed set when one was supplied.
        if shape_partials is not None:
            for p, sag_t, gx_t, gy_t in shape_partials[j]:
                grad[p] = grad[p] + np.sum(dsag_bar * sag_t + dgx_bar * gx_t
                                           + dgy_bar * gy_t)
        elif shape_params[j] or sag_partial_fns[j]:
            Xj = si.Q_loc[..., 0]
            Yj = si.Q_loc[..., 1]
            for p, pname in shape_params[j]:
                sag_t, gx_t, gy_t = surfaces[j].shape.sag_param_partials(
                    Xj, Yj, pname)
                grad[p] = grad[p] + np.sum(dsag_bar * sag_t + dgx_bar * gx_t
                                           + dgy_bar * gy_t)
            for p, fn in sag_partial_fns[j]:
                sag_t, gx_t, gy_t = fn(Xj, Yj)
                grad[p] = grad[p] + np.sum(dsag_bar * sag_t + dgx_bar * gx_t
                                           + dgy_bar * gy_t)

        # index DOFs
        npre_bar = n_bar + ndot_pre_bar
        if si.index_source >= 0:
            grad = grad + npre_bar * nprimedot_s[si.index_source]
        elif n_ambient_dot is not None:
            grad = grad + npre_bar * n_ambient_dot
        if si.typ == STYPE_REFRACT:
            grad = grad + ndot_post_bar * nprimedot_s[j]

        # carry cotangent to the upstream surface's outgoing ray
        P_bar, S_bar = Pdot_bar, Sdot_bar

    return grad


def adjoint_gradient(surfaces, P, S, wvl, seeds, head, *, n_ambient=1.0,
                     n_ambient_dot=None, tol_sag=None):
    """Gradient of a scalar merit (given by head) w.r.t. every seed parameter.

    One forward-with-intermediates pass and one backward sweep.

    Parameters
    ----------
    surfaces : sequence of Surface
    P, S : ndarray
        launch positions / directions.
    wvl : float
        wavelength, microns.
    seeds : sequence of DiffSeed
        defines the trailing parameter axis (order preserved in the gradient).
    head : merit head
        object with a seed(trace, intermediates) -> (P_bar, S_bar, L_bar) method.
    n_ambient : float
    n_ambient_dot : ndarray (P,), optional
    tol_sag : float

    Returns
    -------
    grad : ndarray, (P,)

    """
    seeds = list(seeds)
    n_params = len(seeds)
    trace, intermediates = _forward_with_intermediates(
        surfaces, P, S, wvl, n_ambient=n_ambient, tol_sag=tol_sag)
    Qdot_s, Rdot_s, nprimedot_s, shape_params, sag_partial_fns = \
        _assemble_seeds(len(surfaces), seeds, n_params)
    cotangent_seed = head.seed(trace, intermediates)
    return _backward_sweep(surfaces, trace, intermediates, Qdot_s, Rdot_s,
                           nprimedot_s, shape_params, sag_partial_fns,
                           cotangent_seed, n_ambient_dot=n_ambient_dot)
