"""Ray optics analysis.
"""

from collections import namedtuple

from prysm.conf import config
from prysm.mathops import np

from prysm.polynomials import zernike_nm_seq, lstsq

from .spencer_and_murty import raytrace, valid_mask
from .opt import (
    xp_reference_sphere,
    opd_from_raytrace_eic,
    _pupil_center_chief_index,
)
from ._line_math import line_intersection_params
from .paraxial import paraxial_image_distance, first_order
from .launch import Field, Sampling, launch
from ._meta import (
    system_wavelength, system_epd, object_space_index, image_space_index,
)
from .surfaces import Conic, EvenAsphere, Plane, Sphere


# ---------- result containers ----------------------------------------------
# Plain namedtuples: the analyses below are pure data producers, so the result
# is a labelled bundle of arrays, not an object with behaviour.  Every grid
# array is indexed [field_index, wavelength_index, sample_index]; the fields and
# wavelengths members map those leading indices back to the physical Field /
# wavelength so the indexing is never ambiguous.

DistortionResult = namedtuple('DistortionResult', ['real_xy', 'paraxial_xy', 'percent'])
FieldCurvatureResult = namedtuple('FieldCurvatureResult', ['x_fan_z', 'y_fan_z'])
RayFanGrid = namedtuple('RayFanGrid', ['fields', 'wavelengths', 'pupil', 'x', 'y'])
OPDFanGrid = namedtuple('OPDFanGrid', ['fields', 'wavelengths', 'pupil', 'x', 'y'])
SpotGrid = namedtuple('SpotGrid', ['fields', 'wavelengths', 'x', 'y', 'valid', 'reference'])


def _axis_index(axis):
    if axis == 'x':
        return 0
    if axis == 'y':
        return 1
    raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")


def _reference_value(values, valid, reference, chief_index, *, allow_none=False):
    """Reference point shared by fan and spot analyses."""
    values = np.asarray(values)
    if reference == 'centroid':
        return np.mean(values[valid], axis=0)
    if reference == 'chief':
        if not bool(valid[chief_index]):
            raise ValueError(
                'chief ray is invalid; pass reference="centroid" for an '
                'obscured or vignetted bundle'
            )
        return values[chief_index]
    if reference is None and allow_none:
        return np.zeros(values.shape[1:], dtype=values.dtype)
    choices = "'centroid', 'chief', or None" if allow_none else "'centroid' or 'chief'"
    raise ValueError(f'reference must be {choices}, got {reference!r}')


def _center_valid(values, valid, reference, chief_index, *, allow_none=False):
    """Reference-subtract values and NaN-out invalid rays."""
    values = np.asarray(values)
    ref = _reference_value(values, valid, reference, chief_index,
                           allow_none=allow_none)
    mask = valid.reshape(valid.shape + (1,) * (values.ndim - 1))
    return np.where(mask, values - ref, np.nan), ref


def _require_epd(prescription, epd, wvl=None):
    """Resolve epd from an explicit value or the system; error if neither."""
    epd = system_epd(prescription, epd, wvl)
    if epd is None:
        raise TypeError(
            'epd is required; pass epd=... or supply an OpticalSystem whose '
            'aperture spec resolves it.'
        )
    return epd


def _first_order_geometry_failure(exc):
    """True when first_order failed because scalar ABCD geometry is invalid."""
    msg = str(exc)
    return ('centered axial geometry' in msg
            or 'vertex normal to be axial' in msg)


def resolve_exit_pupil(prescription, wavelength, *, stop_index=None, epd=None,
                       field=None, chief=None, axis_point=None, axis_dir=None):
    """Locate the exit-pupil reference point P_xp for a wavefront evaluation.

    The explicit, side-effect-free resolution that wavefront() deliberately no
    longer does.  Two routes, in order:

    1. Paraxial: when an aperture stop is resolvable (stop_index, else the
       prescription's stop_index), the paraxial exit pupil is first_order's
       xp_z on the optical axis, (0, 0, xp_z).  This is field-independent and
       depends only on (lens, wavelength), so a grid analysis resolves it once
       per wavelength.  Raises when the paraxial exit pupil is at infinity
       (afocal) -- pass P_xp explicitly there.
    2. Geometric: with no resolvable stop (e.g. an off-axis field on a bare
       surface list), take a chief ray's closest approach to the optical axis
       (xp_reference_sphere).  The chief is the pre-traced (P, S) pair in chief
       when given (so a caller reuses a bundle it already traced), else the
       chief ray for field is traced here.

    Parameters
    ----------
    prescription : sequence of Surface or OpticalSystem
    wavelength : float
        in microns
    stop_index : int, optional
        aperture-stop surface index; defaults to the prescription stop_index.
    epd : float, optional
        entrance-pupil diameter for the paraxial solve; defaults from a system.
    field : Field, optional
        field whose chief ray seeds the geometric route.  Defaults to on-axis.
        Ignored when chief is given.
    chief : tuple of ndarray, optional
        (P_chief_final, S_chief_final) -- the post-trace chief position and
        direction for the geometric route, supplied by a caller that already
        traced the bundle (avoids re-launching and keeps the exit pupil
        consistent with that bundle's own chief).
    axis_point, axis_dir : iterable, optional
        point on, and direction of, the optical axis for the geometric route.
        Defaults: origin and +z.

    Returns
    -------
    P_xp : ndarray, shape (3,)
        exit-pupil reference point, ready to pass to wavefront(P_xp=...).

    """
    resolved_stop = (stop_index if stop_index is not None
                     else getattr(prescription, 'stop_index', None))
    if resolved_stop is not None:
        try:
            fo = first_order(prescription, wvl=wavelength, epd=epd,
                             stop_index=resolved_stop)
        except ValueError as exc:
            # a centered-ABCD geometry failure is only recoverable via the
            # geometric route when an explicit axis was supplied; otherwise
            # surface the first_order error.
            if ((axis_point is None and axis_dir is None)
                    or not _first_order_geometry_failure(exc)):
                raise
        else:
            if fo.xp_z is None:
                raise ValueError(
                    'paraxial exit pupil is at infinity; pass P_xp '
                    'explicitly for a planar or finite reference'
                )
            return np.array([0.0, 0.0, float(fo.xp_z)], dtype=config.precision)

    # geometric route: take the chief ray's axis closest-approach.
    if chief is not None:
        P_chief_final, S_chief_final = chief
    else:
        if field is None:
            field = Field(0.0, 0.0)
        epd_geo = system_epd(prescription, epd, wavelength)
        if epd_geo is None:
            epd_geo = 1.0  # chief is a single pupil-center ray; epd only nominal
        P, S = launch(prescription, field, wavelength, Sampling.chief(),
                      epd=epd_geo)
        tr = raytrace(prescription, P, S, wavelength)
        P_chief_final, S_chief_final = tr.P[-1, 0], tr.S[-1, 0]
    _, _, P_xp = xp_reference_sphere(P_chief_final, S_chief_final,
                                     axis_point=axis_point, axis_dir=axis_dir)
    return np.asarray(P_xp, dtype=config.precision)


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
    ax = _axis_index(axis)
    if chief_index is None:
        chief_index = _pupil_center_chief_index(P_hist[0])
    launch = P_hist[0, :, ax]
    image = P_hist[-1, :, ax]

    valid = valid_mask(status, P_hist[-1])

    # the pupil coordinate is referenced the same way as the image error, so
    # the fan stays centered on the pupil.  Using chief_index (N//2) for the
    # pupil zero is only correct when that ray is the pupil center; an annular
    # (obscured) bundle has no center sample there, so the centroid reference
    # subtracts the launch centroid instead -- otherwise the pupil axis comes
    # out lopsided even though the system is on axis.
    if reference == 'chief':
        ref_pupil = launch[chief_index]
    elif reference == 'centroid':
        ref_pupil = np.mean(launch[valid])
    else:
        ref_pupil = _reference_value(launch, valid, reference, chief_index)
    ref_image = _reference_value(image, valid, reference, chief_index)
    return launch[valid] - ref_pupil, image[valid] - ref_image


# ---------- wavefront -------------------------------------------------------


def _filtered_chief_index(valid, chief_index):
    """Position of chief_index within the valid-only subset of rays."""
    valid_indices = np.nonzero(valid)[0]
    return int(np.nonzero(valid_indices == chief_index)[0][0])


def _resolve_chief_index(P, valid, reference, chief_index):
    if chief_index is not None:
        return int(chief_index)
    mask = valid if reference == 'centroid' else None
    return _pupil_center_chief_index(P, mask)


def _wavefront_from_trace(prescription, P, wavelength, trace, *, P_xp,
                          chief_index=None, pupil_coords=None, field=None,
                          output='length', reference='chief'):
    """Wavefront kernel for callers that already have the raytrace result."""
    valid = valid_mask(trace.status, trace.P[-1])
    chief_index = _resolve_chief_index(P, valid, reference, chief_index)
    if not bool(valid[chief_index]):
        if reference == 'chief':
            raise ValueError(
                "chief ray is invalid; cannot define reference sphere.  Pass "
                "reference='centroid' for an obscured or vignetted bundle."
            )
        raise ValueError(
            f'anchor ray (chief_index={chief_index}) is invalid; pass a '
            'chief_index that survives the trace, or omit it to auto-select '
            'the surviving ray nearest the pupil center'
        )

    n_object = object_space_index(prescription, wavelength)
    n_image = image_space_index(prescription, wavelength, fallback=n_object)
    P_chief_final = trace.P[-1, chief_index]
    filtered_chief = _filtered_chief_index(valid, chief_index)
    opd = opd_from_raytrace_eic(trace.P[:, valid], trace.S[:, valid],
                                trace.OPL[:, valid],
                                P_img=P_chief_final,
                                P_xp=np.asarray(P_xp, dtype=P.dtype),
                                n_image=n_image,
                                chief_index=filtered_chief)
    if pupil_coords is None:
        x_pupil = P[valid, 0] - P[chief_index, 0]
        y_pupil = P[valid, 1] - P[chief_index, 1]
    else:
        x_pupil = np.asarray(pupil_coords[0])[valid]
        y_pupil = np.asarray(pupil_coords[1])[valid]

    opd, _ = _apply_field_and_output(opd, x_pupil, y_pupil, field, output,
                                     wavelength)
    return opd, x_pupil, y_pupil, valid


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


def wavefront(prescription, P, S, wavelength, *,
              P_xp,
              chief_index=None,
              pupil_coords=None, field=None, output='length',
              reference='chief'):
    """Trace and compute OPD on the chief-ray-centered reference sphere.

    A pure OPD kernel: it traces the bundle and evaluates the optical path
    difference on the exit-pupil reference sphere via the cancellation-free
    (Welford-rationalized) intersection in opt.opd_from_raytrace_eic.  It does
    no first-order / exit-pupil resolution -- pass a resolved P_xp.  Resolve it
    once with resolve_exit_pupil (or the cached OpticalSystem.exit_pupil) and
    feed it in; for a lens-design OPD fan that is the paraxial exit-pupil center
    (0, 0, xp_z).

    Parameters
    ----------
    prescription : sequence of Surface
        compiled optical prescription; lensdata.to_surfaces()
    P, S : ndarray, shape (N, 3)
        launch positions and direction cosines (typically from launch()).
    wavelength : float
        in microns.
    P_xp : iterable
        exit-pupil reference point (required).  Centers the reference sphere's
        radius R = |P_xp - P_img|.  Resolve it with resolve_exit_pupil or
        OpticalSystem.exit_pupil.
    chief_index : int, optional
        row index of the chief ray.  Defaults to the launch ray nearest the
        pupil center, with invalid rays excluded when reference='centroid'.
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
        raise ValueError(f"reference must be 'chief' or 'centroid', got {reference!r}")
    if P_xp is None:
        raise TypeError(
            'P_xp is required; resolve it with resolve_exit_pupil(...) or '
            'OpticalSystem.exit_pupil(...) and pass it in.  wavefront does no '
            'exit-pupil resolution of its own.'
        )
    trace = raytrace(prescription, P, S, wavelength)
    opd, x_pupil, y_pupil, _ = _wavefront_from_trace(
        prescription, P, wavelength, trace, P_xp=P_xp,
        chief_index=chief_index, pupil_coords=pupil_coords, field=field,
        output=output, reference=reference,
    )
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
    DistortionResult
        A namedtuple (real_xy, paraxial_xy, percent); element i of each array
        corresponds to fields[i].
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

    return DistortionResult(real_xy, paraxial_xy, percent)


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
    s = line_intersection_params(P0, S0, P1, S1)
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
    FieldCurvatureResult
        A namedtuple (x_fan_z, y_fan_z); element i of each array corresponds to
        fields[i].
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
        P0, S0 = launch(prescription, field, wavelength, chief, epd=epd)
        P = np.repeat(P0, 3, axis=0)
        S = np.repeat(S0, 3, axis=0)
        P[1, 0] = P[1, 0] + r_marg
        P[2, 1] = P[2, 1] + r_marg
        tr = raytrace(prescription, P, S, wavelength)
        P_final = tr.P[-1]
        S_final = tr.S[-1]
        x_fan_z[i] = _line_intersection_z(P_final[0], S_final[0],
                                          P_final[1], S_final[1])
        y_fan_z[i] = _line_intersection_z(P_final[0], S_final[0],
                                          P_final[2], S_final[2])
    return FieldCurvatureResult(x_fan_z, y_fan_z)


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
    valid = valid_mask(status, P_final)
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


# ---------- grid analyses (consistent sampling across field & wavelength) ---
# Commercial codes build a ray-fan / spot / OPD-fan plot for every field and
# wavelength from a single command.  These functions are the array-data half of
# that: they trace every (field, wavelength) with one fixed pupil sampling and
# return a stacked, labelled grid.  The plotting layer turns a grid into a
# figure; the same grid can be pickled, fed to a merit, or differenced.

def _resolve_fields(prescription, fields):
    """Fields to evaluate, defaulting to the system's FieldSet, else on-axis."""
    if fields is not None:
        return list(fields)
    sys_fields = getattr(prescription, 'fields', None)
    if sys_fields is not None and len(sys_fields) > 0:
        return list(sys_fields)
    return [Field(0.0, 0.0)]


def _resolve_wavelengths(prescription, wavelengths):
    """Wavelengths (microns) to evaluate, defaulting to the system's set."""
    if wavelengths is not None:
        return [float(w) for w in wavelengths]
    wv = getattr(prescription, 'wavelengths', None)
    if wv:
        values = wv.values() if hasattr(wv, 'values') else wv
        return [float(w) for w in values]
    return [system_wavelength(prescription, None)]


def _fan_grid_setup(prescription, fields, wavelengths, nrays, distribution):
    fields = _resolve_fields(prescription, fields)
    wavelengths = _resolve_wavelengths(prescription, wavelengths)
    x_fan = Sampling.fan(n=nrays, axis='x', distribution=distribution)
    y_fan = Sampling.fan(n=nrays, axis='y', distribution=distribution)
    pupil = np.asarray(x_fan.build(1.0)[:, 0], dtype=config.precision)
    shape = (len(fields), len(wavelengths), pupil.shape[0])
    x = np.full(shape, np.nan, dtype=config.precision)
    y = np.full(shape, np.nan, dtype=config.precision)
    return fields, wavelengths, x_fan, y_fan, pupil, x, y


def _fan_image_error(prescription, field, wavelength, axis, sampling, epd,
                     reference):
    """Reference-subtracted image error of one ray fan, full length with NaN.

    Returns one value per fan sample (NaN where the ray failed), the transverse
    image-plane error in the fan's own axis measured from the chief or centroid.
    Rays clipped by a real aperture during the trace carry a failure status, so
    the valid mask NaNs them and the fan stays full length.
    """
    ax = _axis_index(axis)
    P, S = launch(prescription, field, wavelength, sampling, epd=epd)
    tr = raytrace(prescription, P, S, wavelength)
    valid = valid_mask(tr.status, tr.P[-1])
    image = tr.P[-1, :, ax]
    ci = _pupil_center_chief_index(P)
    centered, _ = _center_valid(image, valid, reference, ci)
    return centered


def ray_aberration_fans(prescription, fields=None, wavelengths=None, *,
                        nrays=21, epd=None, distribution='uniform',
                        reference='chief'):
    """Transverse ray-aberration fans for every field and wavelength.

    Traces an x (sagittal) and a y (tangential) ray fan for each field and
    wavelength using one shared normalized pupil sampling, so the result stacks
    cleanly and a single pupil axis serves every curve.  This is the data step
    behind plotting.plot_ray_fans; the grid can be saved, differenced, or used
    as an optimizer merit without re-tracing.

    Parameters
    ----------
    prescription : sequence of Surface or OpticalSystem
        the optical system.
    fields : iterable of Field, optional
        fields to evaluate; defaults to the system FieldSet, else the on-axis
        field.
    wavelengths : iterable of float, optional
        wavelengths in microns; defaults to the system wavelengths, else the
        reference wavelength.
    nrays : int, optional
        number of rays per fan.  Odd values place a sample at pupil center.
    epd : float, optional
        entrance pupil diameter; defaults from a system aperture spec.
    distribution : str, optional
        radial spacing of the fan samples ('uniform', 'cheby', 'random').
    reference : str, optional
        image-plane registration: 'chief' (default) or 'centroid'.

    Returns
    -------
    RayFanGrid
        A namedtuple (fields, wavelengths, pupil, x, y).  fields is the tuple of
        Field evaluated; wavelengths is an array of microns; pupil is the shared
        normalized pupil coordinate (-1..1) along the fan.  x and y have shape
        (n_fields, n_wavelengths, nrays): x is the sagittal (X) fan image error
        in x, y is the tangential (Y) fan image error in y, both
        reference-subtracted and in prescription length units.  Index as
        x[field_index, wavelength_index, pupil_index]; failed rays are NaN.

    """
    fields, wavelengths, x_fan, y_fan, pupil, x, y = _fan_grid_setup(
        prescription, fields, wavelengths, nrays, distribution,
    )
    for i, field in enumerate(fields):
        for j, w in enumerate(wavelengths):
            epd_w = _require_epd(prescription, epd, w)
            x[i, j] = _fan_image_error(prescription, field, w, 'x', x_fan,
                                       epd_w, reference)
            y[i, j] = _fan_image_error(prescription, field, w, 'y', y_fan,
                                       epd_w, reference)
    return RayFanGrid(tuple(fields),
                      np.asarray(wavelengths, dtype=config.precision),
                      pupil, x, y)


def _exit_pupil_for(prescription, wavelength, *, field=None, stop_index=None,
                    epd=None):
    """Resolve P_xp, using the system's cached exit_pupil when available.

    An OpticalSystem memoizes the (field-independent paraxial) exit pupil per
    wavelength; a bare surface list / LensData resolves it directly.
    """
    if hasattr(prescription, 'exit_pupil') and hasattr(prescription, 'lens'):
        return prescription.exit_pupil(wavelength, field=field,
                                       stop_index=stop_index, epd=epd)
    return resolve_exit_pupil(prescription, wavelength, stop_index=stop_index,
                              epd=epd, field=field)


def _opd_fan(prescription, field, tilt_field, wavelength, sampling, epd,
             P_xp, output, n_pupil):
    """OPD of one ray fan, full length with NaN where rays failed."""
    P, S = launch(prescription, field, wavelength, sampling, epd=epd)
    tr = raytrace(prescription, P, S, wavelength)
    opd, _, _, valid = _wavefront_from_trace(
        prescription, P, wavelength, tr, P_xp=P_xp, field=tilt_field,
        output=output,
    )
    full = np.full(n_pupil, np.nan, dtype=config.precision)
    full[valid] = opd
    return full


def opd_fans(prescription, fields=None, wavelengths=None, *, nrays=21,
             epd=None, distribution='uniform', stop_index=None,
             output='waves'):
    """Wavefront (OPD) fans for every field and wavelength.

    The OPD analogue of ray_aberration_fans: for each field and wavelength it
    traces an x and y fan and evaluates the optical path difference on the
    chief-ray reference sphere (composing analysis.wavefront), sharing one
    normalized pupil sampling.  This is the data step behind
    plotting.plot_opd_fans.

    Parameters
    ----------
    prescription : sequence of Surface or OpticalSystem
        the optical system.
    fields : iterable of Field, optional
        fields to evaluate; defaults to the system FieldSet, else on-axis.
    wavelengths : iterable of float, optional
        wavelengths in microns; defaults to the system wavelengths.
    nrays : int, optional
        number of rays per fan.
    epd : float, optional
        entrance pupil diameter; defaults from a system aperture spec.
    distribution : str, optional
        radial spacing of the fan samples.
    stop_index : int, optional
        aperture-stop index for exit-pupil resolution; defaults from a system.
    output : str, optional
        'waves' (default) or 'length'; see analysis.wavefront.

    Returns
    -------
    OPDFanGrid
        A namedtuple (fields, wavelengths, pupil, x, y).  pupil is the shared
        normalized pupil coordinate (-1..1).  x and y have shape
        (n_fields, n_wavelengths, nrays): the OPD along the x fan and the y fan,
        chief-referenced (chief == 0).  Index as
        x[field_index, wavelength_index, pupil_index]; failed rays are NaN.

    """
    fields, wavelengths, x_fan, y_fan, pupil, x, y = _fan_grid_setup(
        prescription, fields, wavelengths, nrays, distribution,
    )
    n_pupil = pupil.shape[0]
    for i, field in enumerate(fields):
        # angular field tilt is removed inside wavefront; height fields carry
        # no launch-plane tilt to remove.
        tilt_field = field if getattr(field, 'kind', 'angle') == 'angle' else None
        for j, w in enumerate(wavelengths):
            epd_w = _require_epd(prescription, epd, w)
            P_xp = _exit_pupil_for(prescription, w, field=field,
                                   stop_index=stop_index, epd=epd_w)
            x[i, j] = _opd_fan(prescription, field, tilt_field, w, x_fan,
                               epd_w, P_xp, output, n_pupil)
            y[i, j] = _opd_fan(prescription, field, tilt_field, w, y_fan,
                               epd_w, P_xp, output, n_pupil)
    return OPDFanGrid(tuple(fields),
                      np.asarray(wavelengths, dtype=config.precision),
                      pupil, x, y)


def spot_diagrams(prescription, fields=None, wavelengths=None, *,
                  sampling=None, epd=None, reference='centroid'):
    """Image-plane spot data for every field and wavelength.

    Traces one fixed 2D pupil sampling for each field and wavelength and returns
    the image-plane landings, reference-subtracted so each (field, wavelength)
    bundle is centered.  This is the data step behind plotting.plot_spot_diagrams
    and feeds spot_rms_radius / spot_geometric_radius.

    Parameters
    ----------
    prescription : sequence of Surface or OpticalSystem
        the optical system.
    fields : iterable of Field, optional
        fields to evaluate; defaults to the system FieldSet, else on-axis.
    wavelengths : iterable of float, optional
        wavelengths in microns; defaults to the system wavelengths.
    sampling : Sampling, optional
        shared pupil sampling pattern.  Defaults to a 6-ring hexapolar grid.
    epd : float, optional
        entrance pupil diameter; defaults from a system aperture spec.
    reference : str or None, optional
        per-bundle center subtracted from the landings: 'centroid' (default),
        'chief', or None for raw image coordinates.

    Returns
    -------
    SpotGrid
        A namedtuple (fields, wavelengths, x, y, valid, reference).  x, y, and
        valid have shape (n_fields, n_wavelengths, n_samples): the centered
        image coordinates and a per-ray validity mask.  reference has shape
        (n_fields, n_wavelengths, 2): the absolute (x, y) center that was
        subtracted, so absolute landings are recoverable.  Index as
        x[field_index, wavelength_index, sample_index].

    """
    fields = _resolve_fields(prescription, fields)
    wavelengths = _resolve_wavelengths(prescription, wavelengths)
    if sampling is None:
        sampling = Sampling.hex(nrings=6)
    nf = len(fields)
    nw = len(wavelengths)
    n_samples = sampling.build(1.0).shape[0]
    x = np.full((nf, nw, n_samples), np.nan, dtype=config.precision)
    y = np.full((nf, nw, n_samples), np.nan, dtype=config.precision)
    valid = np.zeros((nf, nw, n_samples), dtype=bool)
    reference_xy = np.full((nf, nw, 2), np.nan, dtype=config.precision)
    for i, field in enumerate(fields):
        for j, w in enumerate(wavelengths):
            epd_w = _require_epd(prescription, epd, w)
            P, S = launch(prescription, field, w, sampling, epd=epd_w)
            tr = raytrace(prescription, P, S, w)
            # rays clipped by a real aperture during the trace are status-
            # flagged (not deleted), so the bundle stays full length and those
            # samples come out NaN via the valid mask.
            v = valid_mask(tr.status, tr.P[-1])
            xi = tr.P[-1, :, 0]
            yi = tr.P[-1, :, 1]
            image_xy = np.stack([xi, yi], axis=1)
            ci = _pupil_center_chief_index(P)
            centered, ref = _center_valid(image_xy, v, reference, ci,
                                          allow_none=True)
            x[i, j] = centered[:, 0]
            y[i, j] = centered[:, 1]
            valid[i, j] = v
            reference_xy[i, j] = ref
    return SpotGrid(tuple(fields),
                    np.asarray(wavelengths, dtype=config.precision),
                    x, y, valid, reference_xy)


def _spot_centered(spot_grid):
    """Centroid-referenced x, y of a SpotGrid, regardless of stored reference."""
    x = np.asarray(spot_grid.x)
    y = np.asarray(spot_grid.y)
    xc = x - np.nanmean(x, axis=2, keepdims=True)
    yc = y - np.nanmean(y, axis=2, keepdims=True)
    return xc, yc


def spot_rms_radius(spot_grid):
    """Centroid-referenced RMS spot radius per field and wavelength.

    Re-centers each bundle on its own centroid first, so the result is the true
    geometric RMS radius independent of the grid's stored reference.

    Returns
    -------
    ndarray, shape (n_fields, n_wavelengths)
        RMS radius in prescription length units; entry [i, j] is fields[i] at
        wavelengths[j].

    """
    xc, yc = _spot_centered(spot_grid)
    return np.sqrt(np.nanmean(xc * xc + yc * yc, axis=2))


def spot_geometric_radius(spot_grid):
    """Maximum (geometric) spot radius from the centroid per field/wavelength.

    Returns
    -------
    ndarray, shape (n_fields, n_wavelengths)
        the farthest valid ray from the centroid, in prescription length units;
        entry [i, j] is fields[i] at wavelengths[j].

    """
    xc, yc = _spot_centered(spot_grid)
    return np.sqrt(np.nanmax(xc * xc + yc * yc, axis=2))
