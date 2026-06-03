"""Compute primitives for geometric ray-trace analysis.

Each function here takes a prescription (or a trace) and returns numbers,
not plots.  The plotting routines in plotting.py compose with these.
Operands in design.py also wrap these for use in optimization problems.

Provided primitives:

- transverse_ray_aberration: per-ray image-plane offset vs pupil coordinate
  (the math behind plot_transverse_ray_aberration, exposed for testing /
  scripting / merit functions)
- wavefront: trace and compute OPD on the chief-ray-centered reference
  sphere (composes exit-pupil resolution + opt.opd_from_raytrace)
- wavefront_zernike_fit: project an OPD bundle onto a Zernike basis
- distortion: chief-ray image-plane error vs paraxial proxy, per field
- field_curvature: x- and y-fan best-focus shifts per field.  These are
  sagittal and tangential only for pure-y fields on axisymmetric systems.
- axial_color: paraxial image distance at multiple wavelengths
- chromatic_focal_shift: best-focus shift across wavelength
- lateral_color: chief-ray image-plane landing at multiple (field,
  wavelength) pairs

"""

from prysm.conf import config
from prysm.mathops import np

from prysm.polynomials import zernike_nm_seq, lstsq

from .spencer_and_murty import raytrace
from .opt import (
    xp_reference_sphere,
    opd_from_raytrace,
    opd_from_raytrace_eic,
    # _pupil_center_chief_index lives in opt (a lower layer) so the OPD
    # primitives, wavefront(), the differential trace, and the adjoint merit
    # heads all anchor the reference sphere on the same ray.  Re-exported here
    # for importers that reach it via analysis.
    _pupil_center_chief_index,  # noqa: F401
    _intersect_lines,
    _valid_mask,
)
from .paraxial import paraxial_image_distance, first_order
from .launch import Field, Sampling, launch
from ._meta import (
    system_wavelength, system_epd, object_space_index, image_space_index,
)
from .surfaces import Conic, EvenAsphere, Plane, Sphere


def _require_epd(prescription, epd, wvl=None):
    """Resolve epd from an explicit value or the system; error if neither."""
    epd = system_epd(prescription, epd, wvl)
    if epd is None:
        raise TypeError(
            'epd is required; pass epd=... or supply an OpticalSystem whose '
            'aperture spec resolves it.'
        )
    return epd


# ---------- transverse ray aberration --------------------------------------

def transverse_ray_aberration(P_hist, axis='y', chief_index=None, status=None,
                              reference='chief'):
    """Per-ray image-plane offset from a reference, vs pupil coordinate.

    Parameters
    ----------
    P_hist : ndarray
        position history from raytrace, shape (jj+1, N, 3).
    axis : str
        which axis to report: 'x' or 'y'.
    chief_index : int, optional
        row index of the chief ray; default: the ray nearest the pupil center
        (correct for any sampling, unlike a fixed N//2 which is the center only
        for fan/rect grids).  Used as the registration ray when reference is
        chief, and to center the pupil coordinate in both cases.
    status : ndarray, optional
        per-ray status from raytrace.  Invalid rays are excluded when
        provided.  If omitted, rays with non-finite image coordinates are
        excluded.
    reference : str, optional
        image-plane registration point that the error is measured from.
        chief (default) subtracts the chief ray's landing; centroid subtracts
        the mean landing of the surviving rays.  Centroid matches the option
        of the same name in Zemax and CODE V and is the natural choice when
        the chief ray does not survive -- e.g. it falls inside the central
        obscuration of a Cassegrain.

    Returns
    -------
    pupil : ndarray, shape (N,)
        pupil (x or y) coordinate for each ray, measured as the launch
        offset from the chief ray.  Reporting it chief-relative keeps the fan
        centered on the pupil even when the bundle was shifted laterally at
        the launch plane to route through an off-axis entrance pupil (the
        default for a system with an interior stop); on axis, or with the
        stop at the first surface, the chief launches on axis and this is
        just the launch coordinate.  Matches the convention used by
        wavefront().
    delta : ndarray, shape (N,)
        image-plane (x or y) offset from the reference for each ray.

    """
    P_hist = np.asarray(P_hist)
    if axis == 'x':
        ax = 0
    elif axis == 'y':
        ax = 1
    else:
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")
    if chief_index is None:
        chief_index = _pupil_center_chief_index(P_hist[0])
    launch = P_hist[0, :, ax]
    image = P_hist[-1, :, ax]

    valid = _valid_mask(status, P_hist[-1])
    if valid is not None:
        launch_v = launch[valid]
        image_v = image[valid]
    else:
        launch_v = launch
        image_v = image

    # the pupil coordinate is referenced the same way as the image error, so
    # the fan stays centered on the pupil.  Using chief_index (N//2) for the
    # pupil zero is only correct when that ray is the pupil center; an annular
    # (obscured) bundle has no center sample there, so the centroid reference
    # subtracts the launch centroid instead -- otherwise the pupil axis comes
    # out lopsided even though the system is on axis.
    if reference == 'chief':
        ref_pupil = launch[chief_index]
        ref_image = image[chief_index]
        if not np.isfinite(ref_image):
            raise ValueError(
                'chief ray image coordinate is not finite; pass '
                "reference='centroid' for an obscured or vignetted bundle"
            )
    elif reference == 'centroid':
        ref_pupil = np.mean(launch_v)
        ref_image = np.mean(image_v)
    else:
        raise ValueError(
            f"reference must be 'chief' or 'centroid', got {reference!r}"
        )
    return launch_v - ref_pupil, image_v - ref_image


# ---------- wavefront -------------------------------------------------------


def _filtered_chief_index(valid, chief_index):
    """Position of chief_index within the valid-only subset of rays."""
    valid_indices = np.nonzero(valid)[0]
    return int(np.nonzero(valid_indices == chief_index)[0][0])


def _apply_field_and_output(opd, x_pupil, y_pupil, field, output, wavelength):
    """Field-tilt removal + length/waves scaling shared by the wavefront paths.

    Adds the collimated launch-plane tilt back in (an angular field tilts the
    reference plane) and converts to the requested output.  Returns
    (opd_out, scale): scale is the factor opd was multiplied by -- 1.0 for
    length, -1/(wavelength*1e-3) for waves -- so a differential caller can scale
    its dOPD/dtau maps by the same factor.  The field tilt is tau-independent,
    so it shifts opd but contributes nothing to the tangent maps.
    """
    if field is not None:
        ax, ay = field.angle_radians()
        opd = opd + (np.sin(ax) * x_pupil + np.sin(ay) * y_pupil)
    if output == 'length':
        scale = 1.0
    elif output == 'waves':
        scale = -1.0 / (float(wavelength) * 1e-3)
    else:
        raise ValueError(f"output must be 'length' or 'waves', got {output!r}")
    return opd * scale, scale


def _first_order_geometry_failure(exc):
    """True when first_order failed because scalar ABCD geometry is invalid."""
    msg = str(exc)
    return ('centered axial geometry' in msg
            or 'vertex normal to be axial' in msg)


def wavefront(prescription, P, S, wavelength, *,
              chief_index=None,
              axis_point=None, axis_dir=None, P_xp=None,
              epd=None, stop_index=None,
              pupil_coords=None, field=None, output='length',
              method='sphere', reference='chief'):
    """Trace and compute OPD on the chief-ray-centered reference sphere.

    Composes spencer_and_murty.raytrace, paraxial or explicit exit-pupil
    resolution, and opt.opd_from_raytrace.  Prefer P_xp, or stop_index with
    paraxial exit-pupil resolution, for centered on-axis wavefronts.  The
    chief-ray/axis geometric estimate is a fallback for off-axis use only.

    Parameters
    ----------
    prescription : sequence of Surface
    P, S : ndarray, shape (N, 3)
        launch positions and direction cosines (typically from launch()).
    wavelength : float
        in microns.
    chief_index : int, optional
        row index of the chief ray.  Defaults to the launch ray nearest the
        pupil center, with invalid rays excluded when reference='centroid'.
    axis_point, axis_dir : iterable, optional
        point on, and direction of, the optical axis.  Defaults: origin
        and +z.
    P_xp : iterable, optional
        exit-pupil reference point.  If omitted, it is estimated from the
        system stop/paraxial exit pupil when stop_index is resolvable, else
        from the chief ray and optical axis for off-axis fields.  For
        lens-design OPD fans, pass the paraxial exit-pupil center explicitly,
        e.g. (0, 0, xp_z).
    epd : float, optional
        entrance-pupil diameter for paraxial stop-based P_xp resolution.
        Defaults from an OpticalSystem aperture when available.
    stop_index : int, optional
        aperture-stop surface index for paraxial exit-pupil resolution.
        Defaults from an OpticalSystem stop_index when available.
    pupil_coords : tuple of array_like, optional
        x and y pupil coordinates to return and to use for angular-field
        tilt correction.  This is useful when P has been propagated to a
        physical object-side plane for correct OPL accumulation but the
        desired pupil coordinate remains the entrance-pupil coordinate.
    field : Field, optional
        angular field used to remove the launch-plane tilt from collimated
        object-space bundles.  No tilt correction is applied when omitted.
    output : str, optional
        'length' returns OPD in prescription length units with prysm's
        sign convention (longer ray OPL is positive).  'waves' returns
        lens-design wavefront-error convention in waves, chief == 0,
        equivalent to -(OPD + launch_tilt) / wavelength.  The 'waves'
        conversion assumes the prescription is in millimeters and wavelength
        is in microns (waves = OPD_mm / (wavelength_um * 1e-3)).
    method : str, optional
        'sphere' (default) uses the legacy explicit reference-sphere
        intersection.  'eic' uses the cancellation-free Welford form for
        finite conjugates and falls back to a planar reference (Mikš limit)
        when the reference-sphere radius is effectively infinite -- bit
        identical to 'sphere' for benign systems, more robust for long /
        afocal conjugates.
    reference : str, optional
        'chief' (default) anchors the reference sphere on the chief ray and
        raises if that ray is obscured.  'centroid' anchors instead on the
        surviving ray nearest the pupil center, so OPD analysis works for an
        obscured or vignetted bundle (e.g. a Cassegrain, whose central
        obstruction removes the geometric chief).  When the chief ray is valid
        the two are identical.

    Returns
    -------
    opd : ndarray, shape (N,)
        OPD relative to chief, on the reference sphere centered at the
        chief-ray image point.  Units and sign are controlled by output.
    x_pupil, y_pupil : ndarray, shape (N,)
        launch (x, y) coordinates — the canonical pupil parameterization.

    """
    if reference not in ('chief', 'centroid'):
        raise ValueError(
            f"reference must be 'chief' or 'centroid', got {reference!r}"
        )
    P = np.asarray(P)
    n_object = object_space_index(prescription, wavelength)
    trace = raytrace(prescription, P, S, wavelength)
    valid = _valid_mask(trace.status, trace.P[-1])
    if chief_index is None:
        # for reference='centroid' restrict to surviving rays so an obscured
        # bundle resolves to the innermost valid ray instead of the clipped
        # geometric chief.
        mask = valid if reference == 'centroid' else None
        chief_index = _pupil_center_chief_index(P, mask)
    if not bool(valid[chief_index]):
        if reference == 'chief':
            raise ValueError(
                'chief ray is invalid; cannot define reference sphere.  Pass '
                "reference='centroid' for an obscured or vignetted bundle."
            )
        raise ValueError('no valid rays to anchor the reference sphere')

    P_chief_final = trace.P[-1, chief_index]
    S_chief_final = trace.S[-1, chief_index]
    if P_xp is None:
        resolved_stop = (stop_index if stop_index is not None
                         else getattr(prescription, 'stop_index', None))
        if resolved_stop is not None:
            try:
                fo = first_order(prescription, wvl=wavelength, epd=epd,
                                 stop_index=resolved_stop)
            except ValueError as exc:
                if ((axis_point is None and axis_dir is None)
                        or not _first_order_geometry_failure(exc)):
                    raise
            else:
                if fo.xp_z is None:
                    raise ValueError(
                        'paraxial exit pupil is at infinity; pass P_xp '
                        'explicitly for a planar or finite reference'
                    )
                P_xp = np.array([0.0, 0.0, float(fo.xp_z)], dtype=P.dtype)
        if P_xp is None:
            _, _, P_xp = xp_reference_sphere(P_chief_final, S_chief_final,
                                             axis_point=axis_point,
                                             axis_dir=axis_dir)
    else:
        P_xp = np.asarray(P_xp, dtype=P.dtype)
    filtered_chief = _filtered_chief_index(valid, chief_index)
    if method == 'eic':
        opd_fn = opd_from_raytrace_eic
    elif method == 'sphere':
        opd_fn = opd_from_raytrace
    else:
        raise ValueError(
            f"wavefront method must be 'sphere' or 'eic', got {method!r}"
        )
    n_image = image_space_index(prescription, wavelength, fallback=n_object)
    opd = opd_fn(trace.P[:, valid], trace.S[:, valid],
                 trace.OPL[:, valid],
                 P_img=P_chief_final, P_xp=P_xp,
                 n_image=n_image,
                 chief_index=filtered_chief)
    if pupil_coords is None:
        # pupil coordinate is the launch offset from the chief ray, so the
        # parameterization is correct even when the bundle was routed through
        # an off-axis entrance pupil (chief launch point != origin)
        x_pupil = P[valid, 0] - P[chief_index, 0]
        y_pupil = P[valid, 1] - P[chief_index, 1]
    else:
        x_pupil = np.asarray(pupil_coords[0])[valid]
        y_pupil = np.asarray(pupil_coords[1])[valid]

    opd, _ = _apply_field_and_output(opd, x_pupil, y_pupil, field, output,
                                     wavelength)
    return opd, x_pupil, y_pupil


def wavefront_zernike_fit(opd, x_pupil, y_pupil, nms, *,
                          normalization_radius=None, norm=True):
    """Least-squares fit a Zernike series to a wavefront sample bundle.

    Parameters
    ----------
    opd : ndarray, shape (N,)
        OPD samples, e.g. from wavefront().
    x_pupil, y_pupil : ndarray, shape (N,)
        pupil coordinates parallel to opd.
    nms : iterable of (int, int)
        Zernike (n, m) indices to fit.
    normalization_radius : float, optional
        radius by which to normalize pupil coords before fitting; default
        max(sqrt(x^2+y^2)) over the supplied samples.
    norm : bool, optional
        if True (default), use orthonormal (unit-RMS) Zernikes; if False,
        zero-to-peak normalization.

    Returns
    -------
    coefs : ndarray, shape (len(nms),)
        fit coefficients, parallel to nms.
    residual_rms : float
        RMS of opd - fit.

    """
    opd = np.asarray(opd)
    x_pupil = np.asarray(x_pupil)
    y_pupil = np.asarray(y_pupil)
    valid = np.isfinite(opd) & np.isfinite(x_pupil) & np.isfinite(y_pupil)
    if not valid.any():
        raise ValueError('at least one finite OPD sample is required')
    opd = opd[valid]
    x_pupil = x_pupil[valid]
    y_pupil = y_pupil[valid]
    rsq = x_pupil * x_pupil + y_pupil * y_pupil
    if normalization_radius is None:
        normalization_radius = float(np.sqrt(np.max(rsq)))
    if normalization_radius <= 0.0:
        raise ValueError(
            'normalization_radius must be positive; got '
            f'{normalization_radius}'
        )
    rho = np.sqrt(rsq) / normalization_radius
    theta = np.arctan2(y_pupil, x_pupil)
    basis = np.asarray(zernike_nm_seq(nms, rho, theta, norm=norm))
    coefs = lstsq(basis, opd)
    fit = np.tensordot(coefs, basis, axes=1)
    residual = opd - fit
    rms = float(np.sqrt(np.mean(residual * residual)))
    return coefs, rms


# ---------- distortion ------------------------------------------------------

def distortion(prescription, fields, wavelength=None, *, epd=None,
               paraxial_fraction=1e-4, distortion_type='f-tan',
               pupil_z=None):
    """Per-field image-plane error of the chief ray vs a paraxial proxy.

    For each field, traces the real chief ray and a paraxial proxy chief
    ray (same direction, scaled down by paraxial_fraction).  By default,
    distortion is measured against the lens-design f-tan ideal image
    height.  The scaled-angle proxy is available with
    distortion_type='linear-angle'.

    Parameters
    ----------
    prescription : sequence of Surface
    fields : iterable of Field
        field points to evaluate.  Must all be kind='angle'.
    wavelength : float
        in microns.
    epd : float
        entrance pupil diameter (only enters as the chief-ray pupil-z
        location is the EP; the chief is a single ray at pupil center so
        epd does not affect the trace, but launch() needs it nominally).
    paraxial_fraction : float
        scale factor for the paraxial-proxy field angles.  Default 1e-4.
    distortion_type : str, optional
        'f-tan' (default) or 'linear-angle'.  f-tan compares real chief-ray
        landing against a paraxial focal scale times tan(field angle).
        linear-angle scales the tiny-field chief-ray landing directly by
        1/paraxial_fraction.
    pupil_z : float, optional
        z position of the entrance pupil used for chief-ray launch.  If
        omitted, launch() defaults to the first surface vertex.

    Returns
    -------
    real_xy : ndarray, shape (n_fields, 2)
        actual chief-ray image-plane (x, y) per field.
    paraxial_xy : ndarray, shape (n_fields, 2)
        scaled-up paraxial chief-ray landing per field.
    percent : ndarray, shape (n_fields,)
        signed percent distortion, 100 * (h_real - h_ref) / h_ref, where
        h_ref is the ideal (paraxial) image height and h_real is the real
        chief-ray height projected onto the ideal direction.  Positive is
        pincushion, negative is barrel; 0 for the on-axis field.

    """
    wavelength = system_wavelength(prescription, wavelength)
    epd = _require_epd(prescription, epd, wavelength)
    fields = list(fields)
    n = len(fields)
    real_xy = np.zeros((n, 2), dtype=config.precision)
    paraxial_xy = np.zeros((n, 2), dtype=config.precision)
    percent = np.zeros(n, dtype=config.precision)
    chief = Sampling.chief()
    for i, field in enumerate(fields):
        P_r, S_r = launch(prescription, field, wavelength, chief,
                          epd=epd, pupil_z=pupil_z)
        tr_r = raytrace(prescription, P_r, S_r, wavelength)
        real_xy[i] = tr_r.P[-1, 0, :2]

        ax, ay = field.angle_radians()
        small = Field(
            float(np.degrees(ax * paraxial_fraction)),
            float(np.degrees(ay * paraxial_fraction)),
            kind='angle', unit='deg',
        )
        P_p, S_p = launch(prescription, small, wavelength, chief,
                          epd=epd, pupil_z=pupil_z)
        tr_p = raytrace(prescription, P_p, S_p, wavelength)
        if distortion_type == 'linear-angle':
            paraxial_xy[i] = tr_p.P[-1, 0, :2] / paraxial_fraction
        elif distortion_type == 'f-tan':
            small_slopes = np.array([
                np.tan(ax * paraxial_fraction),
                np.tan(ay * paraxial_fraction),
            ], dtype=config.precision)
            field_slopes = np.array([np.tan(ax), np.tan(ay)],
                                    dtype=config.precision)
            focal_scale = np.zeros(2, dtype=config.precision)
            nonzero = np.abs(small_slopes) > 0.0
            focal_scale[nonzero] = tr_p.P[-1, 0, :2][nonzero] / small_slopes[nonzero]
            paraxial_xy[i] = focal_scale * field_slopes
        else:
            raise ValueError(
                "distortion_type must be 'f-tan' or 'linear-angle', got "
                f"{distortion_type!r}"
            )

        denom = float(np.hypot(*paraxial_xy[i]))
        if denom > 0.0:
            # signed distortion: project the real chief-ray landing onto the
            # ideal (paraxial) image-height direction so the sign survives.  A
            # real height longer than ideal is pincushion (positive); shorter
            # is barrel (negative).  Taking |real - paraxial| would collapse
            # both to a positive magnitude and make them indistinguishable.
            real_height = float(np.dot(real_xy[i], paraxial_xy[i])) / denom
            percent[i] = 100.0 * (real_height - denom) / denom

    return real_xy, paraxial_xy, percent


# ---------- field curvature -------------------------------------------------

_AXISYMMETRIC_SHAPES = (Plane, Sphere, Conic, EvenAsphere)


def _field_is_pure_y(field):
    """True when a field lies in the classical y-z meridian."""
    return abs(float(getattr(field, 'hx', 0.0))) <= 1e-12


def _prescription_is_axisymmetric(prescription):
    """Conservative axisymmetry check for field-curvature labels."""
    surfaces = (prescription.to_surfaces()
                if hasattr(prescription, 'to_surfaces') else list(prescription))
    for surf in surfaces:
        if getattr(surf, 'R', None) is not None:
            return False
        P = np.asarray(getattr(surf, 'P', (0.0, 0.0, 0.0)))
        if np.any(np.abs(P[:2]) > 1e-12):
            return False
        if not isinstance(getattr(surf, 'shape', None), _AXISYMMETRIC_SHAPES):
            return False
    return True


def _field_curvature_labels(prescription, fields):
    """Curve labels for the two field_curvature output arrays."""
    fields = list(fields)
    if fields and all(_field_is_pure_y(field) for field in fields) \
            and _prescription_is_axisymmetric(prescription):
        return ('S', 'T'), ('sagittal', 'tangential')
    return ('X', 'Y'), ('x fan', 'y fan')

def _line_intersection_z(P0, S0, P1, S1):
    """Closest-approach z between two skew rays (P0, S0) and (P1, S1).

    Used for field-curvature focus location: with a chief-marginal pair, the
    common-perpendicular midpoint z is a good surrogate for the marginal
    ray's focus along the chief.

    """
    s = _intersect_lines(P0, S0, P1, S1)
    Q0 = P0 + s[0] * S0
    Q1 = P1 + s[1] * S1
    return 0.5 * (float(Q0[2]) + float(Q1[2]))


def field_curvature(prescription, fields, wavelength=None, *, epd=None,
                    marginal_fraction=1e-3):
    """X- and y-fan focus shifts per field point.

    For each field, traces a 3-ray chief+marginal bundle in x and y and
    locates the longitudinal focus by intersecting each marginal ray with the
    chief.  Result is z, in lab-frame, where each fan converges; subtract
    prescription[-1].P[2] to get a shift relative to the last surface vertex.

    For pure-y fields on axisymmetric systems these are the classical
    sagittal (x fan) and tangential (y fan) field curves.  For non-axisymmetric
    systems or fields with x and y components, interpret the returned arrays as
    x-fan and y-fan focus positions.

    Parameters
    ----------
    prescription : sequence of Surface
    fields : iterable of Field
        field points to evaluate.  Must all be kind='angle'.
    wavelength : float
        in microns.
    epd : float
        entrance pupil diameter.
    marginal_fraction : float
        radius used for the marginal ray, as a fraction of EPD/2.  Default
        1e-3, i.e. a near-chief ray, which returns the differential
        (Coddington) fan foci.  In the classical pure-y axisymmetric case,
        these are sagittal and tangential field curves.  A finite zone (e.g.
        0.7) instead reports
        where that real zonal fan focuses, which folds in coma and oblique
        spherical aberration and can differ from the differential foci by an
        order of magnitude at high aperture and field.

    Returns
    -------
    x_fan_z : ndarray, shape (n_fields,)
        z position where the x-fan marginal converges with the chief, in lab
        frame.  This is sagittal for pure-y fields on an axisymmetric system.
    y_fan_z : ndarray, shape (n_fields,)
        z position where the y-fan marginal converges with the chief.  This is
        tangential for pure-y fields on an axisymmetric system.

    """
    wavelength = system_wavelength(prescription, wavelength)
    epd = _require_epd(prescription, epd, wavelength)
    fields = list(fields)
    n = len(fields)
    x_fan_z = np.zeros(n, dtype=config.precision)
    y_fan_z = np.zeros(n, dtype=config.precision)
    chief = Sampling.chief()
    r_marg = float(marginal_fraction) * float(epd) / 2.0
    for i, field in enumerate(fields):
        # chief
        P_c, S_c = launch(prescription, field, wavelength, chief,
                          epd=epd)
        tr_c = raytrace(prescription, P_c, S_c, wavelength)
        Pc = tr_c.P[-1, 0]
        Sc = tr_c.S[-1, 0]
        # x-fan marginal: one ray offset +x_marg from the chief in the pupil;
        # += keeps any chief offset from entrance-pupil routing.
        P_sx = P_c.copy()
        P_sx[0, 0] = P_sx[0, 0] + r_marg
        tr_sx = raytrace(prescription, P_sx, S_c, wavelength)
        x_fan_z[i] = _line_intersection_z(Pc, Sc,
                                          tr_sx.P[-1, 0],
                                          tr_sx.S[-1, 0])
        # y-fan marginal: one ray offset +y_marg from the chief.
        P_ty = P_c.copy()
        P_ty[0, 1] = P_ty[0, 1] + r_marg
        tr_ty = raytrace(prescription, P_ty, S_c, wavelength)
        y_fan_z[i] = _line_intersection_z(Pc, Sc,
                                          tr_ty.P[-1, 0],
                                          tr_ty.S[-1, 0])
    return x_fan_z, y_fan_z


# ---------- color -----------------------------------------------------------

def axial_color(prescription, wavelengths):
    """Paraxial image distance at each of several wavelengths.

    Parameters
    ----------
    prescription : sequence of Surface
    wavelengths : iterable of float
        wavelengths in microns.

    Returns
    -------
    bfd : ndarray, shape (n_wavelengths,)
        signed paraxial image distance from the last surface vertex.

    """
    return np.array([
        paraxial_image_distance(prescription, wvl=float(w))
        for w in wavelengths
    ], dtype=config.precision)


def _surfaces_for_reference(prescription):
    return (prescription.to_surfaces()
            if hasattr(prescription, 'to_surfaces') else list(prescription))


def _system_wavelength_range(prescription):
    """Wavelength span from OpticalSystem metadata, or None."""
    wavelengths = getattr(prescription, 'wavelengths', None)
    if not wavelengths:
        return None
    if isinstance(wavelengths, dict):
        values = wavelengths.values()
    else:
        values = wavelengths
    values = [float(w) for w in values]
    return min(values), max(values)


def _chromatic_wavelength_samples(prescription, wavelengths, samples):
    if wavelengths is not None:
        return np.asarray([float(w) for w in wavelengths], dtype=config.precision)
    span = _system_wavelength_range(prescription)
    if span is None:
        raise TypeError(
            'wavelengths is required unless prescription carries system '
            'wavelength metadata'
        )
    return np.linspace(span[0], span[1], int(samples), dtype=config.precision)


def _best_focus_shift_from_trace(P_final, S_final, status=None):
    """Axial plane shift that minimizes centroid-referenced RMS spot radius."""
    P_final = np.asarray(P_final)
    S_final = np.asarray(S_final)
    valid = _valid_mask(status, P_final)
    valid = valid & np.isfinite(S_final).all(axis=1)
    valid = valid & (np.abs(S_final[:, 2]) > 1e-30)
    if not valid.any():
        raise ValueError('at least one valid ray is required for best focus')

    P = P_final[valid]
    S = S_final[valid]
    xy = P[:, :2]
    slopes = S[:, :2] / S[:, 2:3]
    xy = xy - np.mean(xy, axis=0)
    slopes = slopes - np.mean(slopes, axis=0)
    denom = float(np.sum(slopes * slopes))
    if denom <= 0.0:
        return 0.0
    return -float(np.sum(xy * slopes)) / denom


def _best_focus_z(prescription, wavelength, *, epd, field, sampling):
    """Lab-frame centroid-RMS best focus for a traced bundle."""
    epd = _require_epd(prescription, epd, wavelength)
    if field is None:
        field = Field(0.0, 0.0, unit='deg')
    if sampling is None:
        sampling = Sampling.hex(nrings=8)
    P, S = launch(prescription, field, wavelength, sampling, epd=epd)
    tr = raytrace(prescription, P, S, wavelength)
    dz = _best_focus_shift_from_trace(tr.P[-1], tr.S[-1], tr.status)
    return float(_surfaces_for_reference(prescription)[-1].P[2]) + dz


def _chromatic_focus_z(prescription, wavelength, focus, *, epd, field,
                       sampling):
    surfaces = _surfaces_for_reference(prescription)
    if focus == 'paraxial':
        return (float(surfaces[-1].P[2])
                + float(paraxial_image_distance(prescription, wvl=wavelength)))
    if focus == 'best':
        return _best_focus_z(
            prescription, wavelength, epd=epd, field=field,
            sampling=sampling,
        )
    raise ValueError(f"focus must be 'best' or 'paraxial', got {focus!r}")


def chromatic_focal_shift(prescription, wavelengths=None, *,
                          reference_wavelength=None, focus='best',
                          epd=None, field=None, sampling=None, samples=101):
    """Best-focus shift as a smooth function of wavelength.

    Focus is evaluated at each sampled wavelength, then the focus at the
    reference wavelength is subtracted.  By default the wavelength grid is a
    dense linear sweep spanning all wavelengths carried by an OpticalSystem.
    With focus='best' the focus depth is the axial plane that minimizes
    centroid-referenced RMS spot radius of the traced bundle.  With
    focus='paraxial' it reports the paraxial focal shift.

    Parameters
    ----------
    prescription : sequence of Surface
        optical system.
    wavelengths : iterable of float, optional
        wavelengths in microns.  If omitted, samples spans the full range of
        prescription.wavelengths.
    reference_wavelength : float, optional
        wavelength in microns whose focus is used as zero.  Defaults to the
        prescription reference wavelength when available.
    focus : str, optional
        'best' (default) or 'paraxial'.  The best-focus mode requires epd.
    epd : float, optional
        entrance pupil diameter for focus='best'.  Defaults from an
        OpticalSystem aperture spec when available.
    field : Field, optional
        field used for focus='best'.  Defaults to the on-axis angular field.
    sampling : Sampling, optional
        ray bundle used for focus='best'.  Defaults to an 8-ring hexapolar
        bundle.
    samples : int, optional
        number of wavelength samples used when wavelengths is omitted.

    Returns
    -------
    wavelengths : ndarray, shape (n_wavelengths,)
        wavelength grid in microns.
    shift : ndarray, shape (n_wavelengths,)
        focus shift in prescription length units.  Positive means downstream
        of the reference-wavelength best focus.

    """
    wavelengths = _chromatic_wavelength_samples(prescription, wavelengths,
                                                samples)
    if reference_wavelength is None:
        reference_wavelength = system_wavelength(prescription, None)
    reference_wavelength = float(reference_wavelength)
    focus = focus.lower()
    foci = np.array([
        _chromatic_focus_z(
            prescription, float(w), focus, epd=epd, field=field,
            sampling=sampling,
        )
        for w in wavelengths
    ], dtype=config.precision)

    ref = _chromatic_focus_z(
        prescription, reference_wavelength, focus, epd=epd, field=field,
        sampling=sampling,
    )
    return wavelengths, foci - ref


def lateral_color(prescription, fields, wavelengths, *, epd=None):
    """Chief-ray image-plane landing at every (field, wavelength) pair.

    The lateral chromatic aberration at field i is the difference between
    landing[i, j] across wavelengths j; users typically subtract the
    primary-wavelength row for a wavelength-difference plot.

    Parameters
    ----------
    prescription : sequence of Surface
    fields : iterable of Field
        field points (kind='angle').
    wavelengths : iterable of float
        wavelengths in microns.
    epd : float
        entrance pupil diameter.

    Returns
    -------
    landing : ndarray, shape (n_fields, n_wavelengths, 2)
        chief-ray (x, y) at the image plane.

    """
    epd = _require_epd(prescription, epd)
    fields = list(fields)
    wavelengths = list(wavelengths)
    n_f = len(fields)
    n_w = len(wavelengths)
    out = np.zeros((n_f, n_w, 2), dtype=config.precision)
    chief = Sampling.chief()
    for i, field in enumerate(fields):
        for j, w in enumerate(wavelengths):
            P, S = launch(prescription, field, float(w), chief, epd=epd)
            tr = raytrace(prescription, P, S, float(w))
            out[i, j] = tr.P[-1, 0, :2]
    return out
