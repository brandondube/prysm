"""Parabasal first-order analysis (firABCD-style) about a real chief ray."""

from prysm.conf import config
from prysm.mathops import np

from .launch import Field, Sampling, launch, _perp_basis
from .spencer_and_murty import STYPE_REFLECT, STYPE_REFRACT, reflect, valid_mask
from ._diff_raytrace import DiffSeed, raytrace_with_tangents
from ._resolve import trace_context
from ._first_order import format_first_order, initialize_slots


_SEED_NAMES = ('dx', 'dy', 'du', 'dv')

# slots of ParabasalFirstOrder that hold (x, y) section pairs
_PAIR_SLOTS = (
    'efl', 'bfl', 'ffl',
    'paraxial_image_distance', 'paraxial_image_z',
    'fno', 'na_image',
    'ep_z', 'xp_z', 'ep_distance', 'xp_distance',
    'stop_diameter', 'ep_diameter', 'xp_diameter',
)


def _resolve_field(system, field):
    """Resolve the chief-ray field: system resolver first, then literals."""
    resolver = getattr(system, 'field', None)
    if callable(resolver):
        try:
            return resolver(field)
        except IndexError:
            fields = getattr(system, 'fields', None)
            if (np.isscalar(field) and float(field) == 0.0
                    and fields is not None and len(fields) == 0):
                return Field(0.0, 0.0)
            raise
    if field is None:
        return Field(0.0, 0.0)
    if isinstance(field, Field):
        return field
    if np.isscalar(field):
        raise TypeError(
            'a literal field must be a (hx, hy) pair or a Field, not a bare '
            f'scalar; got {field!r}')
    return Field(float(field[0]), float(field[1]))


def _chief_tangent_trace(system, surfaces, fld, wvl):
    """Trace the chief with dx/dy/du/dv launch tangents in its T/S frame."""
    P0, S0 = launch(system, fld, wvl, Sampling.chief())
    e1, e2 = _perp_basis(S0[0])
    zero = np.zeros(3, dtype=config.precision)
    Pdot0 = np.stack([e1, e2, zero, zero], axis=-1)[None, ...]
    Sdot0 = np.stack([zero, zero, e1, e2], axis=-1)[None, ...]
    seeds = [DiffSeed(name=name) for name in _SEED_NAMES]
    return raytrace_with_tangents(surfaces, P0, S0, wvl, seeds,
                                  Pdot0=Pdot0, Sdot0=Sdot0)


def _raw_matrix(res, j_pos, j_dir, basis):
    """4x4 launch-to-surface map in the chief T/S frame at that surface.

    Rows are (x, y, theta_x, theta_y); columns are dx, dy, du, dv seeds.
    """
    e1, e2 = basis
    Pd = res.Pdot[j_pos][0]
    Sd = res.Sdot[j_dir][0]
    return np.stack([e1 @ Pd, e2 @ Pd, e1 @ Sd, e2 @ Sd], axis=0)


def _section(M, i):
    """The 2x2 (position, angle) block of section i (0 = x, 1 = y)."""
    p, q = (0, 2) if i == 0 else (1, 3)
    return float(M[p, p]), float(M[p, q]), float(M[q, p]), float(M[q, q])


def _axis_crossing(y, th):
    """Distance along the chief to a ray's axis crossing, or None."""
    if abs(th) < 1e-30:
        return None
    return -y / th


def _image_space_physical_index(surfaces, wvl, n_object):
    """Physical (positive) image-space index: last refracting material."""
    for surf in reversed(surfaces):
        if surf.typ == STYPE_REFRACT:
            return float(surf.material.n(wvl))
    return float(n_object)


def _section_parity(trace, surfaces, e1, e2, exit_basis):
    """Orientation of the transported launch frame at the image."""
    b1 = np.array(e1, dtype=config.precision, copy=True)
    b2 = np.array(e2, dtype=config.precision, copy=True)
    for j, surf in enumerate(surfaces):
        if surf.typ == STYPE_REFLECT:
            n_hat = trace.intermediates[j].n_hat[0]
            if surf.R is not None:
                n_hat = np.asarray(surf.R, dtype=config.precision).T @ n_hat
            # Householder transport of the frame across the mirror = the same
            # reflection the ray undergoes (S - 2 (S.n) n).
            b1 = reflect(b1, n_hat)[0]
            b2 = reflect(b2, n_hat)[0]
        S = trace.S[j + 1, 0]
        for k, b in enumerate((b1, b2)):
            b = b - float(b @ S) * S
            norm = float(np.sqrt(b @ b))
            if norm > 1e-12:
                b = b / norm
                if k == 0:
                    b1 = b
                else:
                    b2 = b
    e1x, e2x = exit_basis
    s1 = float(np.sign(b1 @ e1x)) or 1.0
    s2 = float(np.sign(b2 @ e2x)) or 1.0
    return s1, s2


def _collapse(pair):
    """Mean of an (x, y) pair; lone defined section if one is degenerate."""
    if pair is None:
        return None
    a, b = pair
    if a is None and b is None:
        return None
    if a is None:
        return b
    if b is None:
        return a
    return 0.5 * (a + b)


def _section_image_foci(res, at_inf):
    """Per-section paraxial image z (x_z, y_z) from a chief-tangent trace.

    Uses the position-seed column at infinite conjugates, angle-seed column at
    finite conjugates.  None means that section has no axis crossing.
    """
    trace = res.trace
    P_img = trace.P[-1, 0]
    S_img = trace.S[-1, 0]
    z_img = float(P_img[2])
    simz = float(S_img[2])
    M_li = _raw_matrix(res, -1, -1, _perp_basis(S_img))
    foci = []
    for i in (0, 1):
        A, B, C, D = _section(M_li, i)
        t = _axis_crossing(A, C) if at_inf else _axis_crossing(B, D)
        foci.append(None if t is None else z_img + t * simz)
    return M_li, tuple(foci)


class ParabasalFirstOrder:
    """Parabasal first-order properties about a chief ray."""

    __slots__ = (
        'wavelength', 'field', 'backend', 'force_sym',
        'n_object', 'n_image',
        'n_surfaces', 'n_refractive', 'n_reflective', 'n_eval',
        'total_track', 'stop_index', 'epd', 'abcd',
    ) + _PAIR_SLOTS

    def __init__(self):
        initialize_slots(self)

    def __repr__(self):
        title = f'ParabasalFirstOrder (backend: {self.backend})'
        return format_first_order(self, title, paired=True)


def _fill_metadata(out, ctx, fld, force_sym):
    """Scalar metadata shared by the parabasal and ynu-fallback paths."""
    surfaces = ctx.surfaces
    n_surfaces = len(surfaces)
    out.wavelength = ctx.wavelength
    out.field = fld
    out.force_sym = bool(force_sym)
    out.n_surfaces = n_surfaces
    out.n_refractive = sum(1 for s in surfaces if s.typ == STYPE_REFRACT)
    out.n_reflective = sum(1 for s in surfaces if s.typ == STYPE_REFLECT)
    out.n_eval = n_surfaces - out.n_refractive - out.n_reflective
    out.total_track = float(surfaces[-1].P[2]) - float(surfaces[0].P[2])
    if ctx.epd is not None:
        out.epd = ctx.epd
    if ctx.stop_index is not None:
        k = ctx.stop_index
        if k < 0 or k >= n_surfaces:
            raise IndexError(
                f'stop_index {k} out of range for surfaces of length '
                f'{n_surfaces}'
            )
        out.stop_index = k


def _fill_from_ynu(out, system, ctx):
    """Populate the section pairs from the scalar YNU walk (chief failed)."""
    wvl = ctx.wavelength
    epd = ctx.epd
    stop_index = ctx.stop_index
    resolver = getattr(system, '_ynu_first_order', None)
    if callable(resolver):
        fo = resolver(wvl=wvl, epd=epd, stop_index=stop_index)
    else:
        from .paraxial import ynu_first_order  # local: avoid a circular import
        fo = ynu_first_order(ctx.surfaces, wvl=wvl, epd=epd,
                             stop_index=stop_index)
    out.backend = 'ynu'
    out.n_object = fo.n_object
    out.n_image = fo.n_image
    for name in _PAIR_SLOTS:
        v = getattr(fo, name)
        setattr(out, name, None if v is None else (float(v), float(v)))


def first_order(system, field=None, wavelength=None, *, epd=None,
                stop_index=None, force_sym=False):
    """Parabasal first-order properties of a system about a chief ray.

    Parameters
    ----------
    system : OpticalSystem or sequence of Surface
        a system resolves wavelength/EPD/stop/field.
    field : int, tuple, or Field, optional
        chief-ray field; None is the first field or on-axis.
    wavelength : float, optional
        wavelength in microns; None resolves to the system reference.
    epd : float, optional
        entrance pupil diameter; defaults from the system aperture.
    stop_index : int, optional
        aperture-stop index; defaults from the system.
    force_sym : bool, optional
        collapse each (x, y) section pair to the mean of its sections,
        returning scalar attributes shaped like the classical report.

    Returns
    -------
    ParabasalFirstOrder
        computed properties; unavailable quantities are None.

    """
    ctx = trace_context(system, wavelength, chief=True, epd=epd,
                        stop_index=stop_index)
    surfaces = ctx.surfaces
    wvl = ctx.wavelength
    if len(surfaces) == 0:
        raise ValueError('surfaces is empty')
    fld = _resolve_field(system, field)

    out = ParabasalFirstOrder()
    _fill_metadata(out, ctx, fld, force_sym)

    res = _chief_tangent_trace(system, surfaces, fld, wvl)
    trace = res.trace
    valid = valid_mask(trace.status, trace.P[-1])
    chief_ok = (bool(valid[0])
                and bool(np.all(np.isfinite(res.Pdot[-1])))
                and bool(np.all(np.isfinite(res.Sdot[-1]))))

    if not chief_ok:
        _fill_from_ynu(out, system, ctx)
        if force_sym:
            for name in _PAIR_SLOTS:
                setattr(out, name, _collapse(getattr(out, name)))
        return out

    out.backend = 'parabasal'
    n_obj = ctx.n_object
    n_img_phys = _image_space_physical_index(surfaces, wvl, n_obj)
    n_img_signed = n_img_phys if out.n_reflective % 2 == 0 else -n_img_phys
    out.n_object = n_obj
    out.n_image = n_img_signed

    P0 = trace.P[0, 0]
    S0 = trace.S[0, 0]
    P_img = trace.P[-1, 0]
    S_img = trace.S[-1, 0]
    z0 = float(P0[2])
    s0z = float(S0[2])
    z_img = float(P_img[2])
    simz = float(S_img[2])
    at_inf = fld.kind == 'angle'

    basis_img = _perp_basis(S_img)
    M_li, image_foci = _section_image_foci(res, at_inf)
    out.abcd = M_li
    sigma = _section_parity(trace, surfaces, *_perp_basis(S0),
                            exit_basis=basis_img)
    M_ls = None
    if out.stop_index is not None:
        k = out.stop_index
        M_ls = _raw_matrix(res, k + 1, k, _perp_basis(trace.S[k, 0]))

    first_powered = None
    last_powered = None
    last_interacting = None
    from .paraxial import _paraxial_curvature  # local: demoted internal walk
    for surf in surfaces:
        if surf.typ not in (STYPE_REFLECT, STYPE_REFRACT):
            continue
        last_interacting = surf
        if _paraxial_curvature(surf) != 0.0:
            if first_powered is None:
                first_powered = surf
            last_powered = surf

    pairs = {name: [None, None] for name in _PAIR_SLOTS}
    for i in (0, 1):
        A, B, C, D = _section(M_li, i)
        C_red = sigma[i] * n_img_phys * C
        if abs(C_red) >= 1e-30:
            pairs['efl'][i] = -n_obj / C_red
            if out.epd is not None:
                pairs['fno'][i] = abs(pairs['efl'][i]) / out.epd
                pairs['na_image'][i] = abs(C_red) * out.epd / 2.0
            t_f = _axis_crossing(A, C)
            if t_f is not None and last_powered is not None:
                focal_z = z_img + t_f * simz
                pairs['bfl'][i] = focal_z - float(last_powered.P[2])
            if first_powered is not None:
                t_ffp = _axis_crossing(D, -C)
                if t_ffp is not None:
                    front_focal_z = z0 + t_ffp * s0z
                    pairs['ffl'][i] = (float(first_powered.P[2])
                                       - front_focal_z)
        if image_foci[i] is not None:
            pairs['paraxial_image_z'][i] = image_foci[i]
            if last_interacting is not None:
                pairs['paraxial_image_distance'][i] = (
                    image_foci[i] - float(last_interacting.P[2]))

        if M_ls is None:
            continue
        As, Bs, Cs, Ds = _section(M_ls, i)
        t_ep = _axis_crossing(Bs, -As)
        if t_ep is not None:
            pairs['ep_z'][i] = z0 + t_ep * s0z
            pairs['ep_distance'][i] = (pairs['ep_z'][i]
                                       - float(surfaces[0].P[2]))
        # exit pupil: the same stop-center ray carried to image space
        y_i = A * Bs - B * As
        th_i = C * Bs - D * As
        t_xp = _axis_crossing(y_i, th_i)
        if t_xp is not None:
            pairs['xp_z'][i] = z_img + t_xp * simz
            pairs['xp_distance'][i] = (pairs['xp_z'][i]
                                       - float(surfaces[-1].P[2]))

        if out.epd is None:
            continue
        pairs['ep_diameter'][i] = out.epd
        semi = out.epd / 2.0
        if at_inf:
            x_m, th_m = semi, 0.0
        elif t_ep is not None and abs(t_ep) >= 1e-30:
            x_m, th_m = 0.0, semi / t_ep
        else:
            continue
        stop_semi = abs(As * x_m + Bs * th_m)
        pairs['stop_diameter'][i] = 2.0 * stop_semi
        det_s = As * Ds - Bs * Cs
        if t_xp is not None and abs(det_s) >= 1e-30:
            A_a = (A * Ds - B * Cs) / det_s
            C_a = (C * Ds - D * Cs) / det_s
            xp_mag = A_a + t_xp * C_a
            pairs['xp_diameter'][i] = (pairs['stop_diameter'][i]
                                       * abs(xp_mag))

    for name in _PAIR_SLOTS:
        x, y = pairs[name]
        if x is None and y is None:
            continue
        value = (x, y)
        setattr(out, name, _collapse(value) if force_sym else value)

    return out


def parabasal_foci(system, field, wavelength=None):
    """T/S focus z for one field point via the parabasal tangents.

    Parameters
    ----------
    system : OpticalSystem or sequence of Surface
        an OpticalSystem resolves the wavelength and field; a bare surface
        sequence needs an explicit wavelength and a literal field.
    field : Field or resolvable field spec
    wavelength : float, optional
        in microns; None resolves to the system reference (OpticalSystem only).

    Returns
    -------
    x_z, y_z : float
        lab-frame z where the x and y section pencils focus.

    """
    ctx = trace_context(system, wavelength)
    surfaces = ctx.surfaces
    wvl = ctx.wavelength
    fld = _resolve_field(system, field)
    res = _chief_tangent_trace(system, surfaces, fld, wvl)
    trace = res.trace
    valid = valid_mask(trace.status, trace.P[-1])
    if not bool(valid[0]):
        return float('nan'), float('nan')
    _, foci = _section_image_foci(res, fld.kind == 'angle')
    return tuple(float('nan') if z is None else float(z) for z in foci)


__all__ = [
    'ParabasalFirstOrder',
    'first_order',
    'parabasal_foci',
]
