"""Paraxial (ABCD-matrix) ray tracing for sequential systems.

The system matrix walks a prescription surface-by-surface, alternating
translations through gaps with surface refraction/reflection.  Mirrors flip
the sign of the running index of refraction; combined with signed lab-frame
gaps and a translation entry of t/n, this automatically handles
propagation-direction reversals at mirrors with no need to "unfold" the
system.

Matrices act on the column vector (y, u) where u = n * theta is the
reduced angle.  Coordinates are referenced to the vertex of each successive
surface; no leading translation is applied (the first surface vertex is the
input plane).

"""

from prysm.conf import config
from prysm.mathops import np

from .spencer_and_murty import STYPE_REFLECT, STYPE_REFRACT
from ._meta import (
    lensdata_wavelength,
    lensdata_epd,
    lensdata_stop_index,
    lensdata_n_ambient,
)


def _paraxial_curvature(surf):
    """Vertex curvature of surf for paraxial calculations.

    Surfaces carrying a c parameter (sphere, conic, off-axis conic, even
    asphere, Q-2D freeform) report that.  Planes (params is None) and
    unrecognised surface types are treated as having no power.

    """
    if hasattr(surf, 'shape'):
        return float(getattr(surf.shape, 'c', 0.0))
    if surf.params is not None and 'c' in surf.params:
        return float(surf.params['c'])
    return 0.0


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
        n_prime = float(surf.n(wvl))
        return _refraction_matrix(c, n, n_prime) @ M, n_prime
    return M, n


def _walk_matrix(prescription, wvl, n_ambient, *,
                 end_index=None, include_end_surface=True):
    """Walk a prescription and compose its ABCD matrix."""
    M = np.eye(2, dtype=config.precision)
    n = float(n_ambient)
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


def system_matrix(prescription, wvl=None, n_ambient=None):
    """Compose the 2x2 ABCD system matrix for a sequential prescription.

    Walks the prescription in order: between consecutive surfaces a
    translation matrix is applied with the running index n; at each
    surface the refraction (or reflection, with n' = -n) matrix updates
    the state.  Eval planes contribute only the translation that brings the
    walk to their vertex.

    Parameters
    ----------
    prescription : sequence of Surface
        the prescription to analyse.  Surfaces with a c entry in
        surf.params contribute paraxial power; planes and eval surfaces
        do not.  When a LensData is passed, wvl defaults to its reference
        wavelength.
    wvl : float or str, optional
        wavelength in microns (passed to each refractive surface's n
        callback).  A string names a wavelength of the LensData.  None
        (default) resolves to the LensData reference wavelength, or 0.6328
        for a bare Surface sequence.  Has no effect on a purely reflective
        prescription.
    n_ambient : float, optional
        Index in object space. Defaults from a LensData n_ambient when omitted,
        else 1.0.

    Returns
    -------
    M : ndarray
        2x2 system matrix in (y, u) coordinates.
    n_final : float
        signed index of refraction in image space.  Negative when the
        prescription contains an odd number of reflections.

    """
    wvl = lensdata_wavelength(prescription, wvl)
    n_ambient = lensdata_n_ambient(prescription, n_ambient)
    return _walk_matrix(prescription, wvl, n_ambient)


def paraxial_image_distance(prescription, wvl=None, n_ambient=None):
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
    n_ambient : float
        object-space index

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
    M, n_final = system_matrix(prescription, wvl=wvl, n_ambient=n_ambient)
    A = M[0, 0]
    C = M[1, 0]
    if abs(C) < 1e-30:
        raise ValueError(
            'paraxial system has no net power (system matrix entry C is '
            'zero); cannot solve for an image distance from a collimated '
            'input.'
        )
    return -A * n_final / C


def effective_focal_length(prescription, wvl=None, n_ambient=None):
    """System effective focal length (EFL) from the ABCD matrix.

    EFL = -n_ambient / C, where C is the system matrix entry coupling input
    height to output reduced angle.  Sign follows the usual convention:
    positive EFL for a converging system seen from object space.

    """
    n_ambient = lensdata_n_ambient(prescription, n_ambient)
    M, _ = system_matrix(prescription, wvl=wvl, n_ambient=n_ambient)
    C = M[1, 0]
    if abs(C) < 1e-30:
        raise ValueError(
            'paraxial system has no net power; EFL is infinite.'
        )
    return -float(n_ambient) / C


def back_focal_length(prescription, wvl=None, n_ambient=None):
    """System back focal length (BFL) — distance from the last *powered* surface vertex to the rear focal point.

    Equivalent to paraxial_image_distance when the prescription ends at
    the last powered surface; if downstream eval planes are present, BFL
    measures from the last surface with non-zero curvature, while
    paraxial_image_distance measures from the very last entry.

    """
    last_powered = None
    for surf in prescription:
        if _paraxial_curvature(surf) != 0.0 and surf.typ in (STYPE_REFLECT, STYPE_REFRACT):
            last_powered = surf
    if last_powered is None:
        raise ValueError(
            'prescription contains no powered surfaces; BFL is undefined.'
        )
    bfd_from_end = paraxial_image_distance(prescription, wvl=wvl,
                                           n_ambient=n_ambient)
    extra = float(prescription[-1].P[2]) - float(last_powered.P[2])
    return bfd_from_end + extra


def front_focal_length(prescription, wvl=None, n_ambient=None):
    """System front focal length (FFL) — distance from the front focal point to the first *powered* surface vertex.

    Sign convention: positive when the front focal point lies upstream of
    the first powered surface (the usual case for a converging system in
    air).  Computed from the system matrix as -D * n_object / C and
    then translated to the first powered vertex if leading eval planes are
    present.

    """
    first_powered = None
    for surf in prescription:
        if _paraxial_curvature(surf) != 0.0 and surf.typ in (STYPE_REFLECT, STYPE_REFRACT):
            first_powered = surf
            break
    if first_powered is None:
        raise ValueError(
            'prescription contains no powered surfaces; FFL is undefined.'
        )
    n_ambient = lensdata_n_ambient(prescription, n_ambient)
    M, _ = system_matrix(prescription, wvl=wvl, n_ambient=n_ambient)
    C = M[1, 0]
    D = M[1, 1]
    if abs(C) < 1e-30:
        raise ValueError(
            'paraxial system has no net power; FFL is infinite.'
        )
    ffl_from_first_entry = -float(D) * float(n_ambient) / float(C)
    extra = float(first_powered.P[2]) - float(prescription[0].P[2])
    return ffl_from_first_entry + extra


def _matrix_to_plane(prescription, k, wvl, n_ambient):
    """ABCD from the first entry's vertex (with refraction) to the *plane* of prescription[k] — translation only, no refraction at k.

    Returns (M, n_at_plane).  Used for paraxial pupil location, where
    the stop is treated as an aperture in a plane rather than as a
    refracting surface.

    """
    return _walk_matrix(prescription, wvl, n_ambient,
                        end_index=k, include_end_surface=False)


def entrance_pupil_z(prescription, wvl=None, n_ambient=None, stop_index=None):
    """Lab-frame z of the paraxial entrance pupil.

    The entrance pupil is the image of the aperture stop in object space; its
    center is where an object-space chief ray crosses the axis.  This is the
    plane a collimated or finite-conjugate bundle must pass through for a
    field point to be sampled correctly, so launch() uses it to position
    off-axis bundles.

    Parameters
    ----------
    prescription : sequence of Surface
        when a LensData is passed, wvl, n_ambient, and stop_index each default
        to the corresponding system metadata it carries.
    wvl : float or str, optional
        wavelength in microns (or a LensData wavelength name).
    n_ambient : float, optional
        object-space index.
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
    wvl = lensdata_wavelength(prescription, wvl)
    n_ambient = lensdata_n_ambient(prescription, n_ambient)
    stop_index = lensdata_stop_index(prescription, stop_index)
    if stop_index is None:
        return None
    k = int(stop_index)
    if k < 0 or k >= len(prescription):
        return None
    M_to_stop, _ = _matrix_to_plane(prescription, k, wvl, n_ambient)
    A_b = float(M_to_stop[0, 0])
    B_b = float(M_to_stop[0, 1])
    if abs(A_b) < 1e-30:
        return None  # telecentric: entrance pupil at infinity
    ep_distance = B_b * float(n_ambient) / A_b
    return float(prescription[0].P[2]) + ep_distance


class FirstOrderProperties:
    """Paraxial first-order properties of a prescription.

    Populated by first_order.  Attributes that cannot be computed for
    the supplied inputs are left as None; the __repr__ skips those
    rows so the output is always a clean, ordered summary.

    Always set
    ----------
    wavelength : float
        wavelength used, microns.
    n_object, n_image : float
        signed indices in object and image space.  n_image is negative
        if the prescription contains an odd number of reflections.
    n_surfaces, n_refractive, n_reflective, n_eval : int
        prescription counts.
    total_track : float
        z of the last entry minus z of the first.

    Set when the system has net power
    ---------------------------------
    efl : float
        effective focal length, signed.
    bfl : float
        back focal length, last powered vertex to rear focal point.
    ffl : float
        front focal length, front focal point to first powered vertex.
    paraxial_image_distance : float
        signed distance from the last entry's vertex to the image plane.
    paraxial_image_z : float
        lab-frame z-position of the paraxial image plane.

    Set when epd is supplied
    ----------------------------
    epd : float
        echoed input.
    fno : float
        |efl| / epd.
    na_image : float
        paraxial image-space numerical aperture, |C| * epd / 2
        (assumes collimated object-space input).

    Set when stop_index is supplied
    -----------------------------------
    stop_index : int
        echoed input.
    ep_z, xp_z : float
        lab-frame z-positions of the entrance and exit pupils.  None
        if the corresponding pupil is at infinity (telecentric).
    ep_distance, xp_distance : float
        signed distances from the first and last entries' vertices to the
        EP and XP.

    Set when both epd and stop_index are supplied
    -----------------------------------------------------
    stop_diameter, ep_diameter, xp_diameter : float
        physical diameters.  ep_diameter == epd by definition;
        included for symmetry.

    """

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


def first_order(prescription, wvl=None, n_ambient=None, *,
                epd=None, stop_index=None):
    """Compute paraxial first-order properties of a prescription.

    Combines effective_focal_length, back_focal_length,
    front_focal_length, paraxial_image_distance, and optional
    pupil / F-number calculations into a single report.  Returns a
    FirstOrderProperties instance whose __repr__ is a multi-line
    summary suitable for printing.

    When a LensData is passed, wvl, epd, and stop_index each default to the
    corresponding system metadata it carries.

    Parameters
    ----------
    prescription : sequence of Surface
    wvl : float or str, optional
        wavelength in microns (or a LensData wavelength name).  None defaults
        to the LensData reference wavelength, else 0.6328.
    n_ambient : float
        object-space index.
    epd : float, optional
        entrance pupil diameter.  When supplied, F-number and image-space
        NA are computed; combined with stop_index, pupil diameters as
        well.  Defaults from the LensData epd when omitted.
    stop_index : int, optional
        index of the aperture stop within prescription.  When
        supplied, paraxial entrance and exit pupil z-positions are
        computed.  Defaults from the LensData stop_index when omitted.
        Convention: the stop is treated as an aperture in a
        plane (no refraction at the stop in the EP path); the stop's own
        refraction, if any, is carried by the post-stop matrix used for
        XP location.

    Returns
    -------
    FirstOrderProperties
        attributes are set per the input arguments; quantities that are
        not computable (afocal system, telecentric pupil, etc.) are
        None.

    """
    wvl = lensdata_wavelength(prescription, wvl)
    n_ambient = lensdata_n_ambient(prescription, n_ambient)
    epd = lensdata_epd(prescription, epd)
    stop_index = lensdata_stop_index(prescription, stop_index)
    out = FirstOrderProperties()
    n_surfaces = len(prescription)
    if n_surfaces == 0:
        raise ValueError('prescription is empty')

    out.wavelength = float(wvl)
    out.n_object = float(n_ambient)
    out.n_surfaces = n_surfaces
    out.n_refractive = sum(1 for s in prescription if s.typ == STYPE_REFRACT)
    out.n_reflective = sum(1 for s in prescription if s.typ == STYPE_REFLECT)
    out.n_eval = n_surfaces - out.n_refractive - out.n_reflective
    out.total_track = float(prescription[-1].P[2]) - float(prescription[0].P[2])

    M, n_image_signed = system_matrix(prescription, wvl=wvl, n_ambient=n_ambient)
    out.n_image = float(n_image_signed)
    A = float(M[0, 0])
    B = float(M[0, 1])
    C = float(M[1, 0])
    D = float(M[1, 1])

    has_power = abs(C) >= 1e-30
    if has_power:
        out.efl = -float(n_ambient) / C
        out.paraxial_image_distance = -A * out.n_image / C
        out.paraxial_image_z = (float(prescription[-1].P[2])
                                + out.paraxial_image_distance)
        first_powered = None
        last_powered = None
        for surf in prescription:
            if (_paraxial_curvature(surf) != 0.0
                    and surf.typ in (STYPE_REFLECT, STYPE_REFRACT)):
                if first_powered is None:
                    first_powered = surf
                last_powered = surf
        if last_powered is not None:
            out.bfl = (out.paraxial_image_distance
                       + float(prescription[-1].P[2])
                       - float(last_powered.P[2]))
        if first_powered is not None:
            ffl_from_first = -D * float(n_ambient) / C
            out.ffl = (ffl_from_first
                       + float(first_powered.P[2])
                       - float(prescription[0].P[2]))

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

        M_to_stop, n_at_stop = _matrix_to_plane(prescription, k, wvl,
                                                n_ambient)
        M_from_stop, _ = system_matrix(prescription[k:], wvl=wvl,
                                       n_ambient=n_at_stop)
        A_b = float(M_to_stop[0, 0])
        B_b = float(M_to_stop[0, 1])
        B_a = float(M_from_stop[0, 1])
        D_a = float(M_from_stop[1, 1])

        if abs(A_b) >= 1e-30:
            out.ep_distance = B_b * float(n_ambient) / A_b
            out.ep_z = float(prescription[0].P[2]) + out.ep_distance
        if abs(D_a) >= 1e-30:
            out.xp_distance = -B_a * out.n_image / D_a
            out.xp_z = float(prescription[-1].P[2]) + out.xp_distance

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
