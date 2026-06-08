"""Paraxial ABCD-matrix tracing for sequential systems."""

from prysm.conf import config
from prysm.mathops import np

from .spencer_and_murty import STYPE_REFLECT, STYPE_REFRACT
from ._meta import (
    system_wavelength,
    system_epd,
    system_stop_index,
    object_space_index,
)


_AXIAL_GEOMETRY_TOL = 1e-12


def _as_surface_list(prescription):
    """Compiled Surface list for a LensData, or the supplied sequence."""
    if hasattr(prescription, 'to_surfaces'):
        return prescription.to_surfaces()
    return list(prescription)


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
            raise ValueError(
                'paraxial first-order calculations require centered axial '
                f'geometry; surface {idx} has a decentered vertex.'
            )

        R = getattr(surf, 'R', None)
        if R is not None and not np.allclose(
                np.asarray(R), np.eye(3), atol=_AXIAL_GEOMETRY_TOL, rtol=0.0):
            raise ValueError(
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
                    raise ValueError(
                        'paraxial first-order calculations require the local '
                        f'vertex normal to be axial; surface {idx} has a '
                        'nonzero vertex slope.'
                    )
            except ValueError:
                raise
            except NotImplementedError:
                pass


def _first_order_surfaces(prescription):
    """Surface list validated for centered axial first-order analysis."""
    surfaces = _as_surface_list(prescription)
    _assert_first_order_geometry(surfaces)
    return surfaces


def _translation_matrix(t, n):
    """ABCD translation by signed lab-frame distance t in medium of signed index n, acting on (y, u = n*theta)."""
    return np.array([[1.0, t / n], [0.0, 1.0]], dtype=config.precision)


def _refraction_matrix(c, n, n_prime):
    """ABCD refraction at a surface of curvature c going from index n to n_prime, acting on (y, u = n*theta).  A mirror is the n_prime = -n special case."""
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


def _walk_matrix(prescription, wvl, n_start, *,
                 end_index=None, include_end_surface=True):
    """Walk a prescription and compose its ABCD matrix from a starting index."""
    prescription = _first_order_surfaces(prescription)
    M = np.eye(2, dtype=config.precision)
    n = float(n_start)
    z_prev = float(prescription[0].P[2])
    if end_index is None:
        end_index = len(prescription) - 1
    for k, surf in enumerate(prescription):
        if k > end_index:
            break
        if k > 0:
            t = float(surf.P[2]) - z_prev
            M = _translation_matrix(t, n) @ M
        if include_end_surface or k != end_index:
            M, n = _apply_surface_matrix(M, n, surf, wvl)
        z_prev = float(surf.P[2])
    return M, n


def system_matrix(prescription, wvl=None):
    """Compose the local-y 2x2 ABCD system matrix for a sequential prescription.

    Walks the prescription in order: between consecutive surfaces a
    translation matrix is applied with the running index n; at each
    surface the refraction (or reflection, with n' = -n) matrix updates
    the state.  Eval planes contribute only the translation that brings the
    walk to their vertex.

    Parameters
    ----------
    prescription : sequence of Surface
        the prescription to analyse.  The scalar ABCD path assumes centered
        axial geometry and uses each surface's local y vertex curvature.
        Planes and eval surfaces do not contribute power.  When an
        OpticalSystem is passed, wvl defaults to its reference wavelength.
        The object-space index is taken from the object surface material (the
        leading eval row), else air.
    wvl : float or str, optional
        wavelength in microns (passed to each refractive surface's n
        callback).  A string names a wavelength of the system.  None
        (default) resolves to the reference wavelength, or 0.6328
        for a bare Surface sequence.  Has no effect on a purely reflective
        prescription.

    Returns
    -------
    M : ndarray
        2x2 system matrix in (y, u) coordinates.
    n_final : float
        signed index of refraction in image space.  Negative when the
        prescription contains an odd number of reflections.

    """
    surfaces = _first_order_surfaces(prescription)
    wvl = system_wavelength(prescription, wvl)
    n_object = object_space_index(surfaces, wvl)
    return _walk_matrix(surfaces, wvl, n_object)


def paraxial_image_distance(prescription, wvl=None):
    """Signed distance from the last surface vertex to the paraxial image plane for a collimated on-axis input.

    For a marginal ray launched at (y0, u0 = 0), the image plane is the
    z position where y returns to zero.  In matrix form, this is
    t = -A * n_final / C where A and C are entries of the
    system matrix and n_final is the (signed) image-space index.

    Parameters
    ----------
    prescription : sequence of Surface
    wvl : float
        wavelength in microns

    Returns
    -------
    bfd : float
        signed lab-frame z-displacement from prescription[-1].P[2] to
        the paraxial image.

    Raises
    ------
    ValueError
        if the prescription has no net paraxial power; collimated input
        stays collimated and there is no finite image distance.

    """
    M, n_final = system_matrix(prescription, wvl=wvl)
    A = M[0, 0]
    C = M[1, 0]
    if abs(C) < 1e-30:
        raise ValueError(
            'paraxial system has no net power (system matrix entry C is '
            'zero); cannot solve for an image distance from a collimated '
            'input.'
        )
    return -A * n_final / C


def effective_focal_length(prescription, wvl=None):
    """System effective focal length (EFL) from the ABCD matrix.

    EFL = -n_object / C, where C is the system matrix entry coupling input
    height to output reduced angle and n_object is the object-space index.
    Sign follows the usual convention: positive EFL for a converging system
    seen from object space.

    """
    surfaces = _first_order_surfaces(prescription)
    wvl = system_wavelength(prescription, wvl)
    n_object = object_space_index(surfaces, wvl)
    M, _ = _walk_matrix(surfaces, wvl, n_object)
    C = M[1, 0]
    if abs(C) < 1e-30:
        raise ValueError(
            'paraxial system has no net power; EFL is infinite.'
        )
    return -float(n_object) / C


def back_focal_length(prescription, wvl=None):
    """System back focal length (BFL) — distance from the last *powered* surface vertex to the rear focal point.

    Equivalent to paraxial_image_distance when the prescription ends at
    the last powered surface; if downstream eval planes are present, BFL
    measures from the last surface with non-zero curvature, while
    paraxial_image_distance measures from the very last entry.

    """
    surfaces = _first_order_surfaces(prescription)
    last_powered = None
    for surf in surfaces:
        if (_paraxial_curvature(surf) != 0.0
                and surf.typ in (STYPE_REFLECT, STYPE_REFRACT)):
            last_powered = surf
    if last_powered is None:
        raise ValueError(
            'prescription contains no powered surfaces; BFL is undefined.'
        )
    bfd_from_end = paraxial_image_distance(surfaces, wvl=wvl)
    extra = float(surfaces[-1].P[2]) - float(last_powered.P[2])
    return bfd_from_end + extra


def front_focal_length(prescription, wvl=None):
    """System front focal length (FFL) — distance from the front focal point to the first *powered* surface vertex.

    Sign convention: positive when the front focal point lies upstream of
    the first powered surface (the usual case for a converging system in
    air).  Computed from the system matrix as -D * n_object / C and
    then translated to the first powered vertex if leading eval planes are
    present.

    """
    surfaces = _first_order_surfaces(prescription)
    first_powered = None
    for surf in surfaces:
        if (_paraxial_curvature(surf) != 0.0
                and surf.typ in (STYPE_REFLECT, STYPE_REFRACT)):
            first_powered = surf
            break
    if first_powered is None:
        raise ValueError(
            'prescription contains no powered surfaces; FFL is undefined.'
        )
    wvl = system_wavelength(prescription, wvl)
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


def _matrix_to_plane(prescription, k, wvl, n_start):
    """ABCD from the first entry's vertex (with refraction) to the *plane* of prescription[k] — translation only, no refraction at k.

    Returns (M, n_at_plane).  Used for paraxial pupil location, where
    the stop is treated as an aperture in a plane rather than as a
    refracting surface.

    """
    return _walk_matrix(prescription, wvl, n_start,
                        end_index=k, include_end_surface=False)


def entrance_pupil_z(prescription, wvl=None, stop_index=None):
    """Lab-frame z of the paraxial entrance pupil.

    The entrance pupil is the image of the aperture stop in object space; its
    center is where an object-space chief ray crosses the axis.  This is the
    plane a collimated or finite-conjugate bundle must pass through for a
    field point to be sampled correctly, so launch() uses it to position
    off-axis bundles.

    Parameters
    ----------
    prescription : sequence of Surface
        when an OpticalSystem is passed, wvl and stop_index each default to the
        corresponding system metadata it carries; the object-space index comes
        from the object surface material.
    wvl : float or str, optional
        wavelength in microns (or a system wavelength name).
    stop_index : int, optional
        index of the aperture stop within prescription.

    Returns
    -------
    float or None
        lab-frame z of the entrance pupil, measured in the same frame as the
        surface vertices.  None when the pupil is undefined: no stop_index is
        available, the index is out of range, or the system is telecentric in
        object space (entrance pupil at infinity).

    """
    surfaces = _first_order_surfaces(prescription)
    wvl = system_wavelength(prescription, wvl)
    n_object = object_space_index(surfaces, wvl)
    stop_index = system_stop_index(prescription, stop_index)
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
    """Paraxial first-order properties of a prescription."""

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


def first_order(prescription, wvl=None, *, epd=None, stop_index=None):
    """Compute paraxial first-order properties of a prescription.

    Parameters
    ----------
    prescription : sequence of Surface
    wvl : float or str, optional
        wavelength in microns (or a system wavelength name).  None defaults
        to the reference wavelength, else 0.6328.
    epd : float, optional
        entrance pupil diameter.
    stop_index : int, optional
        aperture-stop index.

    Returns
    -------
    FirstOrderProperties
        computed properties; unavailable quantities are None.

    """
    surfaces = _first_order_surfaces(prescription)
    wvl = system_wavelength(prescription, wvl)
    n_object = object_space_index(surfaces, wvl)
    epd = system_epd(prescription, epd, wvl)
    stop_index = system_stop_index(prescription, stop_index)
    out = FirstOrderProperties()
    n_surfaces = len(surfaces)
    if n_surfaces == 0:
        raise ValueError('prescription is empty')

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
    B = float(M[0, 1])
    C = float(M[1, 0])
    D = float(M[1, 1])

    has_power = abs(C) >= 1e-30
    if has_power:
        out.efl = -float(n_object) / C
        out.paraxial_image_distance = -A * out.n_image / C
        out.paraxial_image_z = (float(surfaces[-1].P[2])
                                + out.paraxial_image_distance)
        first_powered = None
        last_powered = None
        for surf in surfaces:
            if (_paraxial_curvature(surf) != 0.0
                    and surf.typ in (STYPE_REFLECT, STYPE_REFRACT)):
                if first_powered is None:
                    first_powered = surf
                last_powered = surf
        if last_powered is not None:
            out.bfl = (out.paraxial_image_distance
                       + float(surfaces[-1].P[2])
                       - float(last_powered.P[2]))
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
                f'stop_index {k} out of range for prescription of length '
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
