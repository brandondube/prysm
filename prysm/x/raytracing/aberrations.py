"""Third-order Seidel and primary chromatic aberrations."""

from prysm.conf import config
from prysm.mathops import np

from .spencer_and_murty import STYPE_REFLECT, STYPE_REFRACT
from .paraxial import (
    _assert_first_order_geometry,
    _paraxial_curvature,
    entrance_pupil_z,
    local_vertex_curvatures,
)
from ._meta import (
    system_wavelength,
    system_epd,
    system_stop_index,
    object_space_index,
)

# microns of wavelength per one prescription length unit, used to express the
# wavefront coefficients in waves.  Unknown units fall back to mm (prysm's
# documented default, matching analysis.wavefront).
_MICRONS_PER_UNIT = {
    'm': 1.0e6, 'cm': 1.0e4, 'mm': 1.0e3, 'um': 1.0, 'nm': 1.0e-3,
    'micron': 1.0, 'microns': 1.0, 'in': 25400.0, 'inch': 25400.0,
}


class _ParaxialRecord:
    """Per-surface paraxial state for one ray (marginal or chief)."""

    __slots__ = ('y', 'theta_b', 'theta_a', 'n_b', 'n_a', 'c', 'shape', 'typ')

    def __init__(self, y, theta_b, theta_a, n_b, n_a, c, shape, typ):
        self.y = y
        self.theta_b = theta_b
        self.theta_a = theta_a
        self.n_b = n_b
        self.n_a = n_a
        self.c = c
        self.shape = shape
        self.typ = typ


def _surfaces_of(prescription):
    """Compiled Surface list for a LensData, or the sequence itself."""
    if hasattr(prescription, 'to_surfaces'):
        return prescription.to_surfaces()
    return list(prescription)


def paraxial_trace(prescription, y0, theta0, wvl, n_ambient):
    """Trace one paraxial ray in real-slope coordinates, recording each surface.

    Walks the prescription vertex-to-vertex: a transfer y += t*theta through
    each gap, then refraction (or reflection, n' = -n) at the surface.  theta
    is the real ray slope (not the reduced angle n*theta).

    Parameters
    ----------
    prescription : sequence of Surface or LensData
    y0 : float
        ray height at the first surface vertex.
    theta0 : float
        ray slope (object space) entering the first surface.
    wvl : float
        wavelength in microns.
    n_ambient : float
        object-space index.

    Returns
    -------
    list of _ParaxialRecord
        one record per surface, in order.

    """
    surfaces = _surfaces_of(prescription)
    _assert_first_order_geometry(surfaces)
    recs = []
    n = float(n_ambient)
    y = float(y0)
    theta = float(theta0)
    z_prev = float(surfaces[0].P[2])
    for k, surf in enumerate(surfaces):
        if k > 0:
            t = float(surf.P[2]) - z_prev
            y = y + t * theta
        c = _paraxial_curvature(surf)
        theta_b = theta
        n_b = n
        if surf.typ == STYPE_REFRACT:
            n_a = float(surf.material.n(wvl))
            theta_a = (n_b * theta_b - y * (n_a - n_b) * c) / n_a
        elif surf.typ == STYPE_REFLECT:
            n_a = -n_b
            theta_a = (n_b * theta_b - y * (n_a - n_b) * c) / n_a
        else:  # eval / dummy plane: no power, no index change
            n_a = n_b
            theta_a = theta_b
        recs.append(_ParaxialRecord(y, theta_b, theta_a, n_b, n_a, c,
                                    getattr(surf, 'shape', None), surf.typ))
        n = n_a
        theta = theta_a
        z_prev = float(surf.P[2])
    return recs


def _assert_rotational_third_order_geometry(surfaces):
    """Raise when Seidel sums are requested for non-rotational geometry."""
    _assert_first_order_geometry(surfaces)
    for idx, surf in enumerate(surfaces):
        if surf.typ not in (STYPE_REFLECT, STYPE_REFRACT):
            continue
        c_x, c_y = local_vertex_curvatures(surf)
        scale = max(1.0, abs(c_x), abs(c_y))
        if abs(c_x - c_y) > 1e-12 * scale:
            raise ValueError(
                'Seidel aberrations require centered rotational surfaces; '
                f'surface {idx} has different local x and y vertex '
                'curvatures.'
            )


def _signed_indices(surfaces, wvl, n_ambient):
    """Per-surface (n_before, n_after) lists at one wavelength.

    Reflections flip the sign of the running index (n' = -n), matching the
    signed-index convention used throughout this subpackage; eval planes
    leave it unchanged.

    """
    n_b = []
    n_a = []
    n = float(n_ambient)
    for surf in surfaces:
        n_b.append(n)
        if surf.typ == STYPE_REFRACT:
            n = float(surf.material.n(wvl))
        elif surf.typ == STYPE_REFLECT:
            n = -n
        n_a.append(n)
    return n_b, n_a


def _fourth_order_asphere_term(shape):
    """Fourth-order surface deformation G for the aspheric Seidel contribution.

    G is the coefficient of r^4 in the surface sag departure from a sphere of
    the same vertex curvature.  For a conic this is k c^3 / 8; an even asphere
    adds its first (r^4) coefficient.  Rotationally non-symmetric or
    higher-freeform shapes return 0 (their contribution is not third-order).

    """
    if shape is None:
        return 0.0
    name = type(shape).__name__
    params = getattr(shape, 'params', None) or {}
    c = float(params.get('c', 0.0))
    k = float(params.get('k', 0.0))
    if name in ('Sphere',):
        return 0.0
    if name in ('Conic', 'OffAxisConic'):
        return k * c ** 3 / 8.0
    if name == 'EvenAsphere':
        coefs = params.get('coefs', ()) or ()
        a4 = float(coefs[0]) if len(coefs) > 0 else 0.0
        return k * c ** 3 / 8.0 + a4
    return 0.0


def _reduce_field(field):
    """Reduce a Field to (object_z_or_None, slope_or_height, is_angle).

    For an angle field the second item is tan(field magnitude) (object at
    infinity); for a height field it is the object height magnitude and the
    first item is the object z.

    """
    if field.kind == 'angle':
        ax, ay = field.angle_radians()
        mag = float(np.hypot(np.tan(ax), np.tan(ay)))
        return None, mag, True
    h = float(np.hypot(field.hx, field.hy))
    return field.object_z, h, False


def _max_field(fields):
    """The largest-magnitude field of a sequence (the Seidel evaluation field)."""
    best = None
    best_mag = -1.0
    for f in fields:
        if f.kind == 'angle':
            ax, ay = f.angle_radians()
            mag = float(np.hypot(ax, ay))
        else:
            mag = float(np.hypot(f.hx, f.hy))
        if mag > best_mag:
            best_mag = mag
            best = f
    return best


def _marginal_chief_launch(prescription, field, wvl, n_ambient, epd,
                           stop_index):
    """Object-space (y, theta) launches for the marginal and chief rays.

    The marginal (axial) ray runs from the on-axis object point through the
    edge of the entrance pupil; the chief ray runs from the field point
    through the center of the entrance pupil.

    """
    z_ep = entrance_pupil_z(prescription, wvl, stop_index=stop_index)
    if z_ep is None:
        raise ValueError(
            'cannot locate the entrance pupil (no aperture stop, or the '
            'system is telecentric in object space); Seidel sums need a '
            'defined chief ray.  Set stop_index on the OpticalSystem.'
        )
    surfaces = _surfaces_of(prescription)
    z_s1 = float(surfaces[0].P[2])
    a = float(epd) / 2.0

    obj_z, fld, is_angle = _reduce_field(field)
    if is_angle:
        y0_m, theta0_m = a, 0.0
        theta0_c = fld
        y0_c = fld * (z_s1 - z_ep)
    else:
        span = z_ep - obj_z
        if abs(span) < 1e-30:
            raise ValueError(
                'object plane coincides with the entrance pupil; cannot '
                'build paraxial marginal/chief rays.'
            )
        theta0_m = a / span
        y0_m = theta0_m * (z_s1 - obj_z)
        theta0_c = -fld / span
        y0_c = fld + theta0_c * (z_s1 - obj_z)
    return (y0_m, theta0_m), (y0_c, theta0_c)


class SeidelResult:
    """Surface-by-surface third-order and primary chromatic aberration sums.

    Attributes
    ----------
    SI, SII, SIII, SIV, SV : ndarray
        per-surface Seidel sums (spherical, coma, astigmatism, Petzval,
        distortion), in prescription length units.  Aspheric fourth-order
        contributions are folded in.
    CI, CII : ndarray or None
        per-surface primary axial and lateral chromatic sums; None when fewer
        than two wavelengths were available.
    sums : dict
        column totals, keyed 'SI'..'SV' (and 'CI','CII' when present).
    optical_invariant : float
        the Lagrange (optical) invariant H of the marginal/chief pair.
    wavelength : float
        wavelength used, microns.
    unit : str
        prescription length unit used for the waves conversion.
    field : Field
        the field point the chief-ray terms were evaluated at.

    The Seidel coefficients are wavefront-aberration sums in length units;
    wavefront_coefficients() converts the totals to the classical W040.. set
    in waves, and transverse_aberrations() to ray-intercept errors.

    """

    __slots__ = ('SI', 'SII', 'SIII', 'SIV', 'SV', 'CI', 'CII',
                 'sums', 'optical_invariant', 'wavelength', 'unit', 'field',
                 'n_image')

    def __init__(self, SI, SII, SIII, SIV, SV, CI, CII, optical_invariant,
                 wavelength, unit, field, n_image):
        self.SI = SI
        self.SII = SII
        self.SIII = SIII
        self.SIV = SIV
        self.SV = SV
        self.CI = CI
        self.CII = CII
        self.optical_invariant = float(optical_invariant)
        self.wavelength = float(wavelength)
        self.unit = unit
        self.field = field
        self.n_image = float(n_image)
        sums = {
            'SI': float(SI.sum()), 'SII': float(SII.sum()),
            'SIII': float(SIII.sum()), 'SIV': float(SIV.sum()),
            'SV': float(SV.sum()),
        }
        if CI is not None:
            sums['CI'] = float(CI.sum())
            sums['CII'] = float(CII.sum())
        self.sums = sums

    def _wavelength_in_length(self):
        mpu = _MICRONS_PER_UNIT.get(self.unit, _MICRONS_PER_UNIT['mm'])
        return self.wavelength / mpu

    def wavefront_coefficients(self):
        """Total third-order wavefront coefficients in waves.

        Returns a dict with W040 (spherical), W131 (coma), W222
        (astigmatism), W220 (field curvature, Petzval + tangential), and
        W311 (distortion), following Welford's 1/8, 1/2, 1/2, 1/4, 1/2
        factors.  Waves are referenced to the analysis wavelength expressed
        in the prescription length unit.

        """
        wvl_len = self._wavelength_in_length()
        s = self.sums
        return {
            'W040': 0.125 * s['SI'] / wvl_len,
            'W131': 0.5 * s['SII'] / wvl_len,
            'W222': 0.5 * s['SIII'] / wvl_len,
            'W220': 0.25 * (s['SIV'] + s['SIII']) / wvl_len,
            'W311': 0.5 * s['SV'] / wvl_len,
        }

    def transverse_aberrations(self, n_image=None, image_slope=None):
        """Total third-order transverse ray aberrations in length units.

        Returns a dict with TSA (transverse spherical), TCO (tangential
        coma), TAS / SAS (tangential / sagittal astigmatism), PTB (Petzval),
        and DST (distortion).  The conversion factor is 1 / (2 n' u') where
        n' is the image-space index and u' the marginal-ray image-space
        slope; supply image_slope, or it is recovered from the optical
        invariant and the chief data is not needed.

        """
        if n_image is None:
            n_image = self.n_image
        if image_slope is None:
            raise ValueError(
                'transverse_aberrations needs the image-space marginal slope '
                "(image_slope=...); it is u' from the paraxial marginal ray."
            )
        cnvrt = 1.0 / (2.0 * n_image * image_slope)
        s = self.sums
        return {
            'TSA': cnvrt * s['SI'],
            'TCO': cnvrt * 3.0 * s['SII'],
            'TAS': cnvrt * (3.0 * s['SIII'] + s['SIV']),
            'SAS': cnvrt * (s['SIII'] + s['SIV']),
            'PTB': cnvrt * s['SIV'],
            'DST': cnvrt * s['SV'],
        }

    def __repr__(self):
        names = ['SI', 'SII', 'SIII', 'SIV', 'SV']
        arrs = [self.SI, self.SII, self.SIII, self.SIV, self.SV]
        if self.CI is not None:
            names += ['CI', 'CII']
            arrs += [self.CI, self.CII]
        nsurf = len(self.SI)
        header = '  surf | ' + ' '.join(f'{nm:>11s}' for nm in names)
        lines = ['SeidelResult', header, '  ' + '-' * (len(header) - 2)]
        for i in range(nsurf):
            row = ' '.join(f'{float(a[i]):11.4e}' for a in arrs)
            lines.append(f'  {i:>4d} | {row}')
        sums = ' '.join(f'{self.sums[nm]:11.4e}' for nm in names)
        lines.append('  ' + '-' * (len(header) - 2))
        lines.append(f'  {"sum":>4s} | {sums}')
        lines.append(f'  optical invariant: {self.optical_invariant:.6g}')
        return '\n'.join(lines)


def seidel_aberrations(prescription, field=None, wvl=None, *,
                       epd=None, stop_index=None,
                       wavelengths=None, unit=None):
    """Compute surface-by-surface Seidel and primary chromatic aberrations.

    Parameters
    ----------
    prescription : sequence of Surface or OpticalSystem
        the system.  An OpticalSystem supplies epd, stop_index, fields,
        reference wavelength, and unit when those arguments are omitted; the
        object-space index comes from the object surface material.
    field : Field, optional
        the field point at which the field-dependent terms (coma,
        astigmatism, distortion, lateral color) are evaluated.  Defaults to
        the largest-magnitude field of the system.
    wvl : float or str, optional
        wavelength in microns (or a system wavelength name).  Defaults to
        the reference wavelength.
    epd : float, optional
        entrance pupil diameter.
    stop_index : int, optional
        index of the aperture stop; required to locate the entrance pupil.
    wavelengths : sequence of float, optional
        two or more wavelengths (microns) for the primary chromatic terms.
        Defaults to the system wavelengths.  Chromatic terms are skipped
        when fewer than two are available.
    unit : str, optional
        prescription length unit, used only for the waves conversion in
        SeidelResult.wavefront_coefficients.  Defaults to a system unit,
        else 'mm'.

    Returns
    -------
    SeidelResult

    """
    wvl = system_wavelength(prescription, wvl)
    n_object = object_space_index(prescription, wvl)
    epd = system_epd(prescription, epd, wvl)
    stop_index = system_stop_index(prescription, stop_index)
    if epd is None:
        raise ValueError('an entrance pupil diameter is required (epd=...)')
    if field is None:
        fields = getattr(prescription, 'fields', None)
        if not fields:
            raise ValueError(
                'a field is required (field=...); the prescription carries '
                'no fields to default from.'
            )
        field = _max_field(fields)
    if unit is None:
        unit = getattr(prescription, 'unit', None) or 'mm'
    if wavelengths is None:
        wavelengths = getattr(prescription, 'wavelengths', None)
    if isinstance(wavelengths, dict):
        # LensData stores wavelengths as {name: microns}
        wavelengths = list(wavelengths.values())
    surfaces = _surfaces_of(prescription)
    _assert_rotational_third_order_geometry(surfaces)

    (y0_m, u0_m), (y0_c, u0_c) = _marginal_chief_launch(
        prescription, field, wvl, n_object, epd, stop_index)

    marg = paraxial_trace(surfaces, y0_m, u0_m, wvl, n_object)
    chief = paraxial_trace(surfaces, y0_c, u0_c, wvl, n_object)

    # Lagrange (optical) invariant, evaluated object-side; constant through
    # the system, so any surface would give the same value.
    H = float(n_object) * (marg[0].y * u0_c - chief[0].y * u0_m)

    nsurf = len(marg)
    SI = np.zeros(nsurf, dtype=config.precision)
    SII = np.zeros(nsurf, dtype=config.precision)
    SIII = np.zeros(nsurf, dtype=config.precision)
    SIV = np.zeros(nsurf, dtype=config.precision)
    SV = np.zeros(nsurf, dtype=config.precision)

    have_color = (wavelengths is not None
                  and len({float(w) for w in wavelengths}) >= 2)
    if have_color:
        wl_sorted = sorted(float(w) for w in wavelengths)
        wl_short, wl_long = wl_sorted[0], wl_sorted[-1]
        nb_s, na_s = _signed_indices(surfaces, wl_short, n_object)
        nb_l, na_l = _signed_indices(surfaces, wl_long, n_object)
        CI = np.zeros(nsurf, dtype=config.precision)
        CII = np.zeros(nsurf, dtype=config.precision)
    else:
        CI = CII = None

    for i in range(nsurf):
        m = marg[i]
        ch = chief[i]
        c = m.c
        n_b, n_a = m.n_b, m.n_a
        y = m.y
        ybar = ch.y
        # refraction invariants A = n' i' = n i (exact); i = theta + y c.
        A = n_a * (m.theta_a + y * c)
        Abar = n_a * (ch.theta_a + ybar * c)
        dun = m.theta_a / n_a - m.theta_b / n_b   # Delta(u/n) for the marginal
        P_pet = c * (1.0 / n_a - 1.0 / n_b)        # Petzval term
        dn2 = 1.0 / n_a ** 2 - 1.0 / n_b ** 2

        si = -A * A * y * dun
        sii = -A * Abar * y * dun
        siii = -Abar * Abar * y * dun
        siv = -H * H * P_pet
        sv = -Abar * (Abar * Abar * dn2 * y
                      - (H + Abar * y) * ybar * P_pet)

        # rotationally symmetric aspheric (fourth-order) contribution
        G = _fourth_order_asphere_term(m.shape)
        if G != 0.0 and y != 0.0:
            e = ybar / y
            si_star = 8.0 * G * (n_a - n_b) * y ** 4
            si += si_star
            sii += si_star * e
            siii += si_star * e * e
            sv += si_star * e * e * e

        SI[i] = si
        SII[i] = sii
        SIII[i] = siii
        SIV[i] = siv
        SV[i] = sv

        if have_color:
            # Delta(dn/n) across the surface, dn = n_short - n_long.  Mirrors
            # are non-dispersive (dn = 0 either side), so only refractions
            # contribute primary color.
            dn_b = nb_s[i] - nb_l[i]
            dn_a = na_s[i] - na_l[i]
            delta_rel_disp = dn_a / n_a - dn_b / n_b
            CI[i] = A * y * delta_rel_disp
            CII[i] = Abar * y * delta_rel_disp

    n_image = marg[-1].n_a
    return SeidelResult(SI, SII, SIII, SIV, SV, CI, CII, H, wvl, unit, field,
                        n_image)
