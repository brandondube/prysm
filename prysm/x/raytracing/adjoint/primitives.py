"""Adjoint primitives for differential ray tracing."""

from prysm.mathops import np, row_dot


# ---------- 1.1 OPL segment -------------------------------------------------

def adj_opl_segment(n_pre, seg, L_bar):
    """Adjoint of d_opl_segment: dL = ndot ||seg|| + n (seg . dseg)/||seg||.

    Parameters
    ----------
    n_pre : float
        index of the medium preceding the surface (nominal).
    seg : ndarray, (N, 3)
        nominal segment vector P_{j+1} - P_j.
    L_bar : ndarray, (N,)
        cotangent of the segment OPL.

    Returns
    -------
    n_bar : float
        cotangent accumulated onto the preceding-index tangent.
    dseg_bar : ndarray, (N, 3)
        cotangent of the segment vector (fans into P_bar at both endpoints).

    """
    seg_len = np.sqrt(row_dot(seg, seg))
    n_bar = np.sum(L_bar * seg_len)
    dseg_bar = L_bar[:, None] * n_pre * seg / seg_len[:, None]
    return n_bar, dseg_bar


# ---------- 1.2 transform to global -----------------------------------------

def adj_transform_global(Reff, Q_loc, Sprime, P_bar, S_bar):
    """Adjoint of d_transform_global.

    Forward: dPjp1 = R^T dPj + (dR)^T Q_loc + Qdot
             dSjp1 = R^T dSprime + (dR)^T Sprime.

    Parameters
    ----------
    Reff : ndarray, (3, 3)
        surface rotation (identity if untilted).
    Q_loc : ndarray, (N, 3)
        nominal local intersection point.
    Sprime : ndarray, (N, 3)
        nominal post-interaction local direction.
    P_bar, S_bar : ndarray, (N, 3)
        cotangents of dPjp1 and dSjp1.

    Returns
    -------
    dPj_bar, dSprime_bar : ndarray, (N, 3)
        cotangents of the pre-transform intersection / direction tangents.
    Qdot_bar : ndarray, (3,)
        cotangent of the vertex-position tangent.
    Rdot_bar : ndarray, (3, 3)
        cotangent of the rotation tangent.

    """
    dPj_bar = (Reff @ P_bar.T).T
    dSprime_bar = (Reff @ S_bar.T).T
    Qdot_bar = P_bar.sum(axis=0)
    Rdot_bar = (np.einsum('nj,ni->ji', Q_loc, P_bar)
                + np.einsum('nj,ni->ji', Sprime, S_bar))
    return dPj_bar, dSprime_bar, Qdot_bar, Rdot_bar


# ---------- 1.3 refract -----------------------------------------------------

def adj_refract(n, nprime, S_loc, n_hat, dSprime_bar):
    """Adjoint of d_refract.

    Reverses the forward chain dSprime <- dfactor <- dcosT <- dsinT2 <- dcosI
    and the index tangent mu_dot.

    Parameters
    ----------
    n, nprime : float
        indices before / after the surface (nominal).
    S_loc, n_hat : ndarray, (N, 3)
        nominal local incident direction and unit normal.
    dSprime_bar : ndarray, (N, 3)
        cotangent of the refracted direction tangent.

    Returns
    -------
    S_locdot_bar, dn_hat_bar : ndarray, (N, 3)
        cotangents of the incident direction and normal tangents.
    ndot_pre_bar, ndot_post_bar : float
        cotangents of the preceding / following index tangents.

    """
    cosI = row_dot(n_hat, S_loc)
    mu = n / nprime
    one_minus = 1.0 - cosI * cosI
    sinT2 = mu * mu * one_minus
    cosT = np.sqrt(1.0 - sinT2)
    sign = np.sign(cosI)

    # nominal factor not needed; only its derivative structure transposes.
    # --- back through dSprime = S_loc*mu_dot + mu*S_locdot
    #                            + n_hat*dfactor + factor*dn_hat
    factor = sign * cosT - mu * cosI
    S_locdot_bar = mu * dSprime_bar
    dn_hat_bar = factor[:, None] * dSprime_bar
    mu_dot_bar = np.sum(row_dot(S_loc, dSprime_bar))
    dfactor_bar = row_dot(n_hat, dSprime_bar)

    # --- back through dfactor = sign*dcosT - mu*dcosI - mu_dot*cosI
    dcosT_bar = sign * dfactor_bar
    dcosI_bar = -mu * dfactor_bar
    mu_dot_bar = mu_dot_bar - np.sum(cosI * dfactor_bar)

    # --- back through dcosT = -dsinT2 / (2 cosT)
    dsinT2_bar = -dcosT_bar / (2.0 * cosT)

    # --- back through dsinT2 = 2 mu mu_dot one_minus - 2 mu^2 cosI dcosI
    mu_dot_bar = mu_dot_bar + 2.0 * mu * np.sum(one_minus * dsinT2_bar)
    dcosI_bar = dcosI_bar - 2.0 * mu * mu * cosI * dsinT2_bar

    # --- back through dcosI = (S_loc . dn_hat) + (n_hat . S_locdot)
    dn_hat_bar = dn_hat_bar + S_loc * dcosI_bar[:, None]
    S_locdot_bar = S_locdot_bar + n_hat * dcosI_bar[:, None]

    # --- back through mu_dot = (ndot_pre*nprime - n*ndot_post)/nprime^2
    ndot_pre_bar = mu_dot_bar / nprime
    ndot_post_bar = -mu_dot_bar * n / (nprime * nprime)
    return S_locdot_bar, dn_hat_bar, ndot_pre_bar, ndot_post_bar


# ---------- 1.4 reflect -----------------------------------------------------

def adj_reflect(S_loc, n_hat, dSprime_bar):
    """Adjoint of d_reflect: dSprime = dSloc - 2 (n_hat dcosI + cosI dn_hat).

    Parameters
    ----------
    S_loc, n_hat : ndarray, (N, 3)
        nominal local incident direction and unit normal.
    dSprime_bar : ndarray, (N, 3)
        cotangent of the reflected direction tangent.

    Returns
    -------
    S_locdot_bar, dn_hat_bar : ndarray, (N, 3)

    """
    cosI = row_dot(S_loc, n_hat)
    dcosI_bar = -2.0 * row_dot(n_hat, dSprime_bar)
    S_locdot_bar = dSprime_bar + n_hat * dcosI_bar[:, None]
    dn_hat_bar = -2.0 * cosI[:, None] * dSprime_bar + S_loc * dcosI_bar[:, None]
    return S_locdot_bar, dn_hat_bar


# ---------- 1.5 intersect ---------------------------------------------------

def adj_intersect(P0, S_loc, Q_loc, n_hat, hessian, dPj_bar, dn_hat_bar):
    """Adjoint of d_intersect (the implicit ray/surface intersection).

    Reverses the implicit-function-theorem chain.  The nominal unnormalized
    normal g = n_hat / n_hat_z and the surface Hessian enter linearly; no new
    surface derivatives are needed.

    Parameters
    ----------
    P0, S_loc, Q_loc, n_hat : ndarray, (N, 3)
        nominal local ray origin, direction, intersection, unit normal.
    hessian : tuple of ndarray
        (sag_xx, sag_xy, sag_yy), each (N,), evaluated at the intersection.
    dPj_bar, dn_hat_bar : ndarray, (N, 3)
        cotangents of the local intersection tangent and normal tangent.

    Returns
    -------
    P0dot_bar, S_locdot_bar : ndarray, (N, 3)
        cotangents of the local origin / direction tangents.
    dsag_param_bar, dgx_param_bar, dgy_param_bar : ndarray, (N,)
        cotangents of the explicit shape-parameter partials (contracted later
        with the per-parameter sag_param_partials to form the gradient).

    """
    nz = n_hat[..., 2]
    g = n_hat / nz[:, None]                       # g_z == 1
    s_total = row_dot(Q_loc - P0, S_loc)
    g_dot_S = row_dot(g, S_loc)
    sxx, sxy, syy = hessian

    # --- back through dn_hat = (dg - n_hat (n_hat . dg)) * nz
    A_bar = dn_hat_bar * nz[:, None]
    n_dot_dg_bar = -row_dot(n_hat, A_bar)
    dg_bar = A_bar + n_hat * n_dot_dg_bar[:, None]

    # --- back through dg = [-dgx, -dgy, 0]
    dgx_bar = -dg_bar[:, 0]
    dgy_bar = -dg_bar[:, 1]

    # --- back through dgx = sxx dXj + sxy dYj + dgx_param  (and dgy)
    dgx_param_bar = dgx_bar
    dgy_param_bar = dgy_bar
    dXj_bar = sxx * dgx_bar + sxy * dgy_bar
    dYj_bar = sxy * dgx_bar + syy * dgy_bar

    # dXj, dYj are components 0, 1 of dPj; fold into the dPj cotangent
    dPj_bar_tot = dPj_bar.copy()
    dPj_bar_tot[:, 0] = dPj_bar_tot[:, 0] + dXj_bar
    dPj_bar_tot[:, 1] = dPj_bar_tot[:, 1] + dYj_bar

    # --- back through dPj = P0dot + sdot S_loc + s_total S_locdot
    P0dot_bar = dPj_bar_tot
    S_locdot_bar = s_total[:, None] * dPj_bar_tot
    sdot_bar = row_dot(S_loc, dPj_bar_tot)

    # --- back through sdot = -(g.P0dot + s_total (g.Slocdot) - dsag_param)/g.S
    num_bar = -sdot_bar / g_dot_S
    g_dot_P0dot_bar = num_bar
    g_dot_Slocdot_bar = s_total * num_bar
    dsag_param_bar = -num_bar

    # --- back through the two g-dots
    P0dot_bar = P0dot_bar + g * g_dot_P0dot_bar[:, None]
    S_locdot_bar = S_locdot_bar + g * g_dot_Slocdot_bar[:, None]
    return P0dot_bar, S_locdot_bar, dsag_param_bar, dgx_param_bar, dgy_param_bar


# ---------- 1.6 transform to local ------------------------------------------

def adj_transform_local(Reff, P, Q, S, P0dot_bar, S_locdot_bar):
    """Adjoint of d_transform_local.

    Forward: P0dot = R(Pdot - Qdot) + Rdot(P - Q);  S_locdot = R Sdot + Rdot S.

    Parameters
    ----------
    Reff : ndarray, (3, 3)
        surface rotation.
    P, Q, S : ndarray
        nominal global incoming position (N, 3), vertex (3,), direction (N, 3).
    P0dot_bar, S_locdot_bar : ndarray, (N, 3)
        cotangents of the local origin / direction tangents.

    Returns
    -------
    Pdot_bar, Sdot_bar : ndarray, (N, 3)
        cotangents of the incoming position / direction tangents.
    Qdot_bar : ndarray, (3,)
    Rdot_bar : ndarray, (3, 3)

    """
    Rt = Reff.T
    Pdot_bar = (Rt @ P0dot_bar.T).T
    Sdot_bar = (Rt @ S_locdot_bar.T).T
    Qdot_bar = -Pdot_bar.sum(axis=0)
    Pmq = P - Q
    Rdot_bar = (np.einsum('ni,nj->ij', P0dot_bar, Pmq)
                + np.einsum('ni,nj->ij', S_locdot_bar, S))
    return Pdot_bar, Sdot_bar, Qdot_bar, Rdot_bar


# ---------- 1.7 reference sphere --------------------------------------------

def adj_intersect_reference_sphere_full(P, S, C, R, t_bar):
    """Adjoint of intersect_reference_sphere's t wrt the ray AND the sphere.

    The full transpose of spencer_and_murty.intersect_reference_sphere's segment
    length t, returning cotangents for the ray (P, S) and for the sphere
    geometry (center C, radius R).  The C / R cotangents close the reference
    sphere's dependence on the chief ray (handled by the WFE merit head).

    Parameters
    ----------
    P, S : ndarray, (N, 3)
        nominal ray origins (last surface) and directions.
    C : ndarray, (3,)
        sphere center (chief image point).
    R : float
        sphere radius.
    t_bar : ndarray, (N,)
        cotangent of the reference-sphere segment length.

    Returns
    -------
    P_bar, S_bar : ndarray, (N, 3)
    C_bar : ndarray, (3,)
    R_bar : float

    """
    dvec = P - C[None, :]
    b = row_dot(S, dvec)
    cc = row_dot(dvec, dvec) - R * R
    disc = b * b - cc
    disc = np.where(disc < 0, np.zeros_like(disc), disc)
    sqrt_disc = np.sqrt(disc)
    safe = sqrt_disc == 0
    sqrt_disc_safe = np.where(safe, 1.0, sqrt_disc)

    # forward: t = -b - sqrt_disc;  sqrt_disc_dot = discdot/(2 sqrt_disc_safe)
    #   discdot = 2 b bdot - ccdot
    #   bdot = dvec . Sdot + S . (Pdot - Cdot)
    #   ccdot = 2 dvec . (Pdot - Cdot) - 2 R Rdot
    bdot_bar = -t_bar
    sqrt_disc_dot_bar = -t_bar
    discdot_bar = sqrt_disc_dot_bar / (2.0 * sqrt_disc_safe)
    bdot_bar = bdot_bar + 2.0 * b * discdot_bar
    ccdot_bar = -discdot_bar

    P_bar = 2.0 * dvec * ccdot_bar[:, None] + S * bdot_bar[:, None]
    S_bar = dvec * bdot_bar[:, None]
    C_bar = -(S * bdot_bar[:, None]).sum(axis=0) \
        - 2.0 * (dvec * ccdot_bar[:, None]).sum(axis=0)
    R_bar = -2.0 * R * np.sum(ccdot_bar)
    return P_bar, S_bar, C_bar, R_bar


def adj_intersect_reference_sphere(P, S, C, R, t_bar):
    """Adjoint of intersect_reference_sphere's segment length t (rays only).

    Holds the sphere center C and radius R fixed; their motion (which couples
    through the chief ray) is recovered separately via
    adj_intersect_reference_sphere_full.

    Parameters
    ----------
    P, S : ndarray, (N, 3)
    C : ndarray, (3,)
    R : float
    t_bar : ndarray, (N,)

    Returns
    -------
    P_bar, S_bar : ndarray, (N, 3)

    """
    P_bar, S_bar, _, _ = adj_intersect_reference_sphere_full(P, S, C, R, t_bar)
    return P_bar, S_bar


def adj_closest_point_on_axis(P, S, axis_point, axis_dir, P_xp_bar):
    """Adjoint of d_closest_point_on_axis (auto-located exit-pupil point).

    Transposes the foot of the common perpendicular from the chief ray (P, S)
    to the optical axis line.  The axis is fixed; cotangent flows back to the
    chief ray's position and direction.

    Parameters
    ----------
    P, S : ndarray, (3,)
        nominal chief-ray position and direction (single ray).
    axis_point, axis_dir : ndarray, (3,)
        a point on and the direction of the optical axis (axis_dir unit-norm).
    P_xp_bar : ndarray, (3,)
        cotangent of the located exit-pupil point.

    Returns
    -------
    P_bar, S_bar : ndarray, (3,)

    """
    A = np.asarray(P)
    Sc = np.asarray(S)
    B = np.asarray(axis_point)
    Sa = np.asarray(axis_dir)
    Sa = Sa / np.sqrt(np.sum(Sa * Sa))

    w = A - B
    a = np.dot(Sc, Sc)
    b = np.dot(Sc, Sa)
    c = np.dot(Sa, Sa)
    d = np.dot(Sc, w)
    e = np.dot(Sa, w)
    denom = a * c - b * b
    num = a * e - b * d

    # forward: P_xp_dot = Sa * tdot;  tdot = (numdot denom - num denomdot)/denom^2
    tdot_bar = np.dot(Sa, P_xp_bar)
    numdot_bar = tdot_bar / denom
    denomdot_bar = -tdot_bar * num / (denom * denom)

    # numdot = adot e + a edot - bdot d - b ddot
    adot_bar = numdot_bar * e
    edot_bar = numdot_bar * a
    bdot_bar = numdot_bar * (-d)
    ddot_bar = numdot_bar * (-b)
    # denomdot = adot c - 2 b bdot
    adot_bar = adot_bar + denomdot_bar * c
    bdot_bar = bdot_bar + denomdot_bar * (-2.0 * b)

    # adot = 2 Sc.Sdot; bdot = Sa.Sdot; edot = Sa.Pdot; ddot = w.Sdot + Sc.Pdot
    S_bar = 2.0 * Sc * adot_bar + Sa * bdot_bar + w * ddot_bar
    P_bar = Sa * edot_bar + Sc * ddot_bar
    return P_bar, S_bar


__all__ = [
    'adj_opl_segment',
    'adj_transform_global',
    'adj_refract',
    'adj_reflect',
    'adj_intersect',
    'adj_transform_local',
    'adj_intersect_reference_sphere',
    'adj_intersect_reference_sphere_full',
    'adj_closest_point_on_axis',
]
