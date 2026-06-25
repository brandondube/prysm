"""Paraxial ABCD-matrix tracing for sequential systems."""

from prysm.conf import config
from prysm.mathops import np

from .spencer_and_murty import (
    STYPE_REFLECT, STYPE_REFRACT, _is_measurement_surf)
from ._meta import object_space_index


_AXIAL_GEOMETRY_TOL = 1e-12


class NonAxialSystemError(ValueError):
    """A surface sequence is outside the centered-axial first-order ABCD contract.

    Subclasses ValueError so existing broad handlers keep working; callers that
    can degrade gracefully (e.g. entrance-pupil resolution on a tilted system)
    catch this type instead of matching on the message text.
    """


def _require_wavelength(wvl):
    """Coerce a resolved wavelength to a float, raising when it is absent."""
    if wvl is None:
        raise ValueError(
            'wavelength must be resolved before calling a paraxial primitive; '
            'pass an explicit wvl or call via the OpticalSystem, which resolves '
            'None to the reference wavelength.'
        )
    return float(wvl)


def _as_surface_list(surfaces):
    """Validate that surfaces is a compiled list, not a system or LensData."""
    if hasattr(surfaces, 'to_surfaces'):
        raise TypeError(
            'paraxial primitives take a compiled surface list, not a system or '
            'LensData; pass system.to_surfaces().'
        )
    return list(surfaces)


def _as_float_scalar(value):
    """Coerce a backend scalar to a Python float."""
    return float(np.asarray(value))


def local_vertex_curvatures(surf):
    """Local x and local y vertex curvatures of a surface.

    The returned pair is (c_x, c_y), evaluated in the surface's local
    coordinate frame at x = y = 0.  Rotational surfaces have c_x == c_y.
    Toroid and Biconic expose independent local curvatures directly; other
    shapes use their sag Hessian when available and fall back to a scalar c.
    """
    shape = getattr(surf, 'shape', None)
    if shape is not None:
        params = getattr(shape, 'params', None) or {}
        if 'c_x' in params and 'c_y' in params:
            return float(params['c_x']), float(params['c_y'])
        hessian = getattr(shape, 'sag_hessian', None)
        if callable(hessian):
            try:
                zero = np.asarray(0.0, dtype=config.precision)
                c_x, _, c_y = hessian(zero, zero)
                return _as_float_scalar(c_x), _as_float_scalar(c_y)
            except NotImplementedError:
                pass
        if 'c' in params:
            c = float(params['c'])
            return c, c

    params = getattr(surf, 'params', None) or {}
    if 'c_x' in params and 'c_y' in params:
        return float(params['c_x']), float(params['c_y'])
    if 'c' in params:
        c = float(params['c'])
        return c, c
    return 0.0, 0.0

# TODO: should these be removed?  Not used?

def local_x_vertex_curvature(surf):
    """Local x-section vertex curvature of surf."""
    return local_vertex_curvatures(surf)[0]


def local_y_vertex_curvature(surf):
    """Local y-section vertex curvature of surf."""
    return local_vertex_curvatures(surf)[1]


def _paraxial_curvature(surf):
    """Local y-section vertex curvature of surf for paraxial calculations.

    The ABCD state vector in this module is (y, u), so the scalar matrix path
    consumes the local y curvature.  Use local_x_vertex_curvature and
    local_y_vertex_curvature when a caller needs both principal local sections.

    """
    return local_y_vertex_curvature(surf)


def _assert_first_order_geometry(surfaces):
    """Raise when a surface sequence is outside the axial ABCD contract."""
    for idx, surf in enumerate(surfaces):
        P = np.asarray(getattr(surf, 'P', (0.0, 0.0, 0.0)))
        if P.shape[0] >= 2 and not np.allclose(
                P[:2], 0.0, atol=_AXIAL_GEOMETRY_TOL, rtol=0.0):
            raise NonAxialSystemError(
                'paraxial first-order calculations require centered axial '
                f'geometry; surface {idx} has a decentered vertex.'
            )

        R = getattr(surf, 'R', None)
        if R is not None and not np.allclose(
                np.asarray(R), np.eye(3), atol=_AXIAL_GEOMETRY_TOL, rtol=0.0):
            raise NonAxialSystemError(
                'paraxial first-order calculations require centered axial '
                f'geometry; surface {idx} is tilted or rotated.'
            )

        shape = getattr(surf, 'shape', None)
        gradient = getattr(shape, '_sag_gradient', None)
        if callable(gradient):
            try:
                zero = np.asarray(0.0, dtype=config.precision)
                gx, gy = gradient(zero, zero)
                if (abs(_as_float_scalar(gx)) > _AXIAL_GEOMETRY_TOL
                        or abs(_as_float_scalar(gy)) > _AXIAL_GEOMETRY_TOL):
                    raise NonAxialSystemError(
                        'paraxial first-order calculations require the local '
                        f'vertex normal to be axial; surface {idx} has a '
                        'nonzero vertex slope.'
                    )
            except NonAxialSystemError:
                raise
            except NotImplementedError:
                pass


def _first_order_surfaces(surfaces):
    """Surface list validated for centered axial first-order analysis."""
    surfaces = _as_surface_list(surfaces)
    _assert_first_order_geometry(surfaces)
    return surfaces


def _translation_matrix(t, n):
    """ABCD translation in (y, u = n*theta)."""
    return np.array([[1.0, t / n], [0.0, 1.0]], dtype=config.precision)


def _refraction_matrix(c, n, n_prime):
    """ABCD refraction at curvature c from index n to n_prime."""
    P_pwr = (n_prime - n) * c
    return np.array([[1.0, 0.0], [-P_pwr, 1.0]], dtype=config.precision)


def _apply_surface_matrix(M, n, surf, wvl):
    """Apply one surface action to an ABCD matrix and running index."""
    c = _paraxial_curvature(surf)
    if surf.typ == STYPE_REFLECT:
        n_prime = -n
        return _refraction_matrix(c, n, n_prime) @ M, n_prime
    if surf.typ == STYPE_REFRACT:
        n_prime = float(surf.material.n(wvl))
        return _refraction_matrix(c, n, n_prime) @ M, n_prime
    return M, n


def _walk_matrix(surfaces, wvl, n_start, *,
                 end_index=None, include_end_surface=True):
    """Walk a surface sequence and compose its ABCD matrix from a start index."""
    surfaces = _first_order_surfaces(surfaces)
    M = np.eye(2, dtype=config.precision)
    n = float(n_start)
    z_prev = float(surfaces[0].P[2])
    if end_index is None:
        end_index = len(surfaces) - 1
    for k, surf in enumerate(surfaces):
        if k > end_index:
            break
        if k > 0:
            t = float(surf.P[2]) - z_prev
            M = _translation_matrix(t, n) @ M
        if include_end_surface or k != end_index:
            M, n = _apply_surface_matrix(M, n, surf, wvl)
        z_prev = float(surf.P[2])
    return M, n


def system_matrix(surfaces, wvl=None):
    """Compose the local-y 2x2 ABCD system matrix for a sequential system.

    Translation by the running index between surfaces, refraction (or
    reflection, with n' = -n) at each.  Requires centered axial geometry.

    Parameters
    ----------
    surfaces : sequence of Surface
        compiled surfaces; the object-space index comes from the object row.
    wvl : float
        wavelength in microns

    Returns
    -------
    M : ndarray
        2x2 system matrix in (y, u) coordinates.
    n_final : float
        signed index of refraction in image space.  Negative when the
        surfaces contain an odd number of reflections.

    """
    surfaces = _first_order_surfaces(surfaces)
    wvl = _require_wavelength(wvl)
    n_object = object_space_index(surfaces, wvl)
    return _walk_matrix(surfaces, wvl, n_object)


def paraxial_image_distance(surfaces, wvl=None):
    """Signed distance from the last powered vertex to the paraxial image plane.

    t = -A * n_final / C, where the image plane is where a collimated on-axis
    ray (y0, u0=0) returns to y=0.

    Parameters
    ----------
    surfaces : sequence of Surface
    wvl : float
        wavelength in microns

    Returns
    -------
    bfd : float
        signed lab-frame z-displacement from the last powered surface vertex
        to the paraxial image.

    Raises
    ------
    ValueError
        if the surfaces have no net paraxial power; collimated input
        stays collimated and there is no finite image distance.

    """
    surfaces = _as_surface_list(surfaces)
    # strip trailing measurement planes so the BFD references the last
    # interacting vertex, not the image plane's back-focal gap
    while len(surfaces) > 1 and _is_measurement_surf(
            getattr(surfaces[-1], 'typ', None)):
        surfaces = surfaces[:-1]
    M, n_final = system_matrix(surfaces, wvl=wvl)
    A = M[0, 0]
    C = M[1, 0]
    if abs(C) < 1e-30:
        raise ValueError(
            'paraxial system has no net power (system matrix entry C is '
            'zero); cannot solve for an image distance from a collimated '
            'input.'
        )
    return -A * n_final / C


def effective_focal_length(surfaces, wvl=None):
    """System effective focal length (EFL) from the ABCD matrix.

    EFL = -n_object / C; positive for a converging system seen from object space.
    """
    surfaces = _first_order_surfaces(surfaces)
    wvl = _require_wavelength(wvl)
    n_object = object_space_index(surfaces, wvl)
    M, _ = _walk_matrix(surfaces, wvl, n_object)
    C = M[1, 0]
    if abs(C) < 1e-30:
        raise ValueError(
            'paraxial system has no net power; EFL is infinite.'
        )
    return -float(n_object) / C


def back_focal_length(surfaces, wvl=None):
    """Distance from the last *powered* vertex to the rear focal point.

    Differs from paraxial_image_distance only when the last interacting
    surface is flat: BFL measures from the last curved vertex.
    """
    surfaces = _first_order_surfaces(surfaces)
    last_powered = None
    last_interacting = None
    for surf in surfaces:
        if surf.typ not in (STYPE_REFLECT, STYPE_REFRACT):
            continue
        last_interacting = surf
        if _paraxial_curvature(surf) != 0.0:
            last_powered = surf
    if last_powered is None:
        raise ValueError(
            'surfaces contain no powered surfaces; BFL is undefined.'
        )
    # paraxial_image_distance references the last interacting (refracting/
    # reflecting) vertex; translate that to the last *powered* vertex.
    bfd = paraxial_image_distance(surfaces, wvl=wvl)
    extra = float(last_interacting.P[2]) - float(last_powered.P[2])
    return bfd + extra


def front_focal_length(surfaces, wvl=None):
    """Distance from the front focal point to the first *powered* vertex.

    -D * n_object / C, translated to the first powered vertex.  Positive when
    the front focus lies upstream of it.
    """
    surfaces = _first_order_surfaces(surfaces)
    first_powered = None
    for surf in surfaces:
        if (_paraxial_curvature(surf) != 0.0
                and surf.typ in (STYPE_REFLECT, STYPE_REFRACT)):
            first_powered = surf
            break
    if first_powered is None:
        raise ValueError(
            'surfaces contain no powered surfaces; FFL is undefined.'
        )
    wvl = _require_wavelength(wvl)
    n_object = object_space_index(surfaces, wvl)
    M, _ = _walk_matrix(surfaces, wvl, n_object)
    C = M[1, 0]
    D = M[1, 1]
    if abs(C) < 1e-30:
        raise ValueError(
            'paraxial system has no net power; FFL is infinite.'
        )
    ffl_from_first_entry = -float(D) * float(n_object) / float(C)
    extra = float(first_powered.P[2]) - float(surfaces[0].P[2])
    return ffl_from_first_entry + extra


def _matrix_to_plane(surfaces, k, wvl, n_start):
    """ABCD from the first vertex to the plane of surfaces[k], no refraction at k.

    Returns (M, n_at_plane).  Used for pupil location, where the stop is an
    aperture in a plane rather than a refracting surface.
    """
    return _walk_matrix(surfaces, wvl, n_start,
                        end_index=k, include_end_surface=False)


def entrance_pupil_z(surfaces, wvl=None, stop_index=None):
    """Lab-frame z of the paraxial entrance pupil.

    The stop imaged into object space; launch() positions off-axis bundles
    through it.

    Parameters
    ----------
    surfaces : sequence of Surface
        compiled surfaces; the object-space index comes from the object
        surface material.
    wvl : float
        wavelength in microns.
    stop_index : int, optional
        index of the aperture stop within surfaces.

    Returns
    -------
    float or None
        lab-frame z of the entrance pupil, measured in the same frame as the
        surface vertices.  None when the pupil is undefined: no stop_index is
        available, the index is out of range, or the system is telecentric in
        object space (entrance pupil at infinity).

    """
    surfaces = _first_order_surfaces(surfaces)
    wvl = _require_wavelength(wvl)
    n_object = object_space_index(surfaces, wvl)
    if stop_index is None:
        return None
    k = int(stop_index)
    if k < 0 or k >= len(surfaces):
        return None
    M_to_stop, _ = _matrix_to_plane(surfaces, k, wvl, n_object)
    A_b = float(M_to_stop[0, 0])
    B_b = float(M_to_stop[0, 1])
    if abs(A_b) < 1e-30:
        return None  # telecentric: entrance pupil at infinity
    ep_distance = B_b * float(n_object) / A_b
    return float(surfaces[0].P[2]) + ep_distance


class FirstOrderProperties:
    """Paraxial first-order properties of a surface sequence."""

    __slots__ = (
        'wavelength', 'n_object', 'n_image',
        'n_surfaces', 'n_refractive', 'n_reflective', 'n_eval',
        'total_track',
        'efl', 'bfl', 'ffl',
        'paraxial_image_distance', 'paraxial_image_z',
        'epd', 'fno', 'na_image',
        'stop_index',
        'ep_z', 'xp_z', 'ep_distance', 'xp_distance',
        'stop_diameter', 'ep_diameter', 'xp_diameter',
    )

    def __init__(self):
        for s in self.__slots__:
            setattr(self, s, None)

    def __repr__(self):
        def row(label, value, fmt='.6g', suffix=''):
            if value is None:
                return None
            return f'  {label:<22s}: {format(value, fmt)}{suffix}'

        lines = ['FirstOrderProperties']
        lines.append(row('wavelength', self.wavelength, '.6g', ' um'))
        lines.append(
            f'  {"surfaces":<22s}: '
            f'{self.n_surfaces} ({self.n_refractive} refr, '
            f'{self.n_reflective} refl, {self.n_eval} eval)'
        )
        lines.append(row('total track', self.total_track))
        lines.append(row('n object', self.n_object))
        lines.append(row('n image (signed)', self.n_image))
        lines.append(row('EFL', self.efl))
        lines.append(row('BFL', self.bfl))
        lines.append(row('FFL', self.ffl))
        lines.append(row('paraxial image dist', self.paraxial_image_distance))
        lines.append(row('paraxial image z', self.paraxial_image_z))
        lines.append(row('EPD', self.epd))
        lines.append(row('F/#', self.fno, '.4g'))
        lines.append(row('NA image', self.na_image, '.4g'))
        if self.stop_index is not None:
            lines.append(f'  {"stop index":<22s}: {self.stop_index}')
        lines.append(row('EP z', self.ep_z))
        lines.append(row('EP distance from S1', self.ep_distance, '+.6g'))
        lines.append(row('XP z', self.xp_z))
        lines.append(row('XP distance from SN', self.xp_distance, '+.6g'))
        lines.append(row('stop diameter', self.stop_diameter))
        lines.append(row('EP diameter', self.ep_diameter))
        lines.append(row('XP diameter', self.xp_diameter))
        return '\n'.join(line for line in lines if line is not None)


def ynu_first_order(surfaces, wvl=None, *, epd=None, stop_index=None):
    """Paraxial first-order properties from the scalar YNU/ABCD matrix walk.

    Requires centered axial geometry.

    Parameters
    ----------
    surfaces : sequence of Surface
    wvl : float
        wavelength in microns.
    epd : float, optional
        entrance pupil diameter.
    stop_index : int, optional
        aperture-stop index.

    Returns
    -------
    FirstOrderProperties
        computed properties; unavailable quantities are None.

    """
    surfaces = _first_order_surfaces(surfaces)
    wvl = _require_wavelength(wvl)
    n_object = object_space_index(surfaces, wvl)
    epd = None if epd is None else float(epd)
    out = FirstOrderProperties()
    n_surfaces = len(surfaces)
    if n_surfaces == 0:
        raise ValueError('surfaces is empty')

    out.wavelength = float(wvl)
    out.n_object = float(n_object)
    out.n_surfaces = n_surfaces
    out.n_refractive = sum(1 for s in surfaces if s.typ == STYPE_REFRACT)
    out.n_reflective = sum(1 for s in surfaces if s.typ == STYPE_REFLECT)
    out.n_eval = n_surfaces - out.n_refractive - out.n_reflective
    out.total_track = float(surfaces[-1].P[2]) - float(surfaces[0].P[2])

    M, n_image_signed = _walk_matrix(surfaces, wvl, n_object)
    out.n_image = float(n_image_signed)
    A = float(M[0, 0])
    C = float(M[1, 0])
    D = float(M[1, 1])

    has_power = abs(C) >= 1e-30
    if has_power:
        out.efl = -float(n_object) / C
        dist_from_end = -A * out.n_image / C
        out.paraxial_image_z = float(surfaces[-1].P[2]) + dist_from_end
        first_powered = None
        last_powered = None
        last_interacting = None
        for surf in surfaces:
            if surf.typ not in (STYPE_REFLECT, STYPE_REFRACT):
                continue
            last_interacting = surf
            if _paraxial_curvature(surf) != 0.0:
                if first_powered is None:
                    first_powered = surf
                last_powered = surf
        # measured from the last reflecting/refracting surface, not from
        # surfaces[-1]: a system whose last entry is an image plane at
        # paraxial focus would otherwise always report ~0
        if last_interacting is not None:
            out.paraxial_image_distance = (
                out.paraxial_image_z - float(last_interacting.P[2]))
        else:
            out.paraxial_image_distance = dist_from_end
        if last_powered is not None:
            out.bfl = out.paraxial_image_z - float(last_powered.P[2])
        if first_powered is not None:
            ffl_from_first = -D * float(n_object) / C
            out.ffl = (ffl_from_first
                       + float(first_powered.P[2])
                       - float(surfaces[0].P[2]))

    if epd is not None:
        out.epd = float(epd)
        if has_power:
            out.fno = abs(out.efl) / out.epd
            out.na_image = abs(C) * out.epd / 2.0

    if stop_index is not None:
        k = int(stop_index)
        if k < 0 or k >= n_surfaces:
            raise IndexError(
                f'stop_index {k} out of range for surfaces of length '
                f'{n_surfaces}'
            )
        out.stop_index = k

        M_to_stop, n_at_stop = _matrix_to_plane(surfaces, k, wvl,
                                                n_object)
        M_from_stop, _ = _walk_matrix(surfaces[k:], wvl, n_at_stop)
        A_b = float(M_to_stop[0, 0])
        B_b = float(M_to_stop[0, 1])
        B_a = float(M_from_stop[0, 1])
        D_a = float(M_from_stop[1, 1])

        if abs(A_b) >= 1e-30:
            out.ep_distance = B_b * float(n_object) / A_b
            out.ep_z = float(surfaces[0].P[2]) + out.ep_distance
        if abs(D_a) >= 1e-30:
            out.xp_distance = -B_a * out.n_image / D_a
            out.xp_z = float(surfaces[-1].P[2]) + out.xp_distance

        if epd is not None:
            out.ep_diameter = out.epd
            if abs(A_b) >= 1e-30:
                out.stop_diameter = out.epd * abs(A_b)
                if abs(D_a) >= 1e-30:
                    A_a = float(M_from_stop[0, 0])
                    C_a = float(M_from_stop[1, 0])
                    det_from_stop = A_a * D_a - B_a * C_a
                    xp_mag = det_from_stop / D_a
                    out.xp_diameter = out.stop_diameter * abs(xp_mag)

    return out
