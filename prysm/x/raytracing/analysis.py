"""Ray optics analysis.
"""

from collections import namedtuple
from dataclasses import dataclass

from prysm.conf import config
from prysm.mathops import np

from prysm.polynomials import zernike_nm_seq, lstsq

from .spencer_and_murty import raytrace, valid_mask
from .opt import (
    xp_reference_sphere,
    hopkins_eic_closing,
    reference_sphere_curvature,
    centroid_referenced_rms,
    _pupil_center_chief_index,
)
from .paraxial import paraxial_image_distance
from .launch import Field, Sampling, launch, _apply_vignetting
from ._meta import (
    system_wavelength, system_epd, object_space_index, image_space_index,
    system_first_order,
)
from ._trace_grid import (
    TraceRecord, iter_trace_grid, trace_cell, _resolve_fields,
    _resolve_wavelengths, _require_epd, field_sweep,
)
from .surfaces import Conic, EvenAsphere, Plane, Sphere


# ---------- result containers ----------------------------------------------
# Grid arrays are indexed [field_index, wavelength_index, sample_index].


@dataclass(frozen=True, slots=True)
class DistortionResult:
    real_xy: object
    paraxial_xy: object
    percent: object


@dataclass(frozen=True, slots=True)
class FieldCurvatureResult:
    x_fan_z: object
    y_fan_z: object


RayFanGrid = namedtuple('RayFanGrid', ['fields', 'wavelengths', 'pupil_x', 'pupil_y', 'x', 'y'])
OPDFanGrid = namedtuple('OPDFanGrid', ['fields', 'wavelengths', 'pupil_x', 'pupil_y', 'x', 'y'])
SpotGrid = namedtuple('SpotGrid', ['fields', 'wavelengths', 'x', 'y', 'valid', 'reference'])
FullFieldGrid = namedtuple('FullFieldGrid', ['hx', 'hy', 'data', 'metric', 'kind', 'unit'])


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
    out = values - ref
    out[~valid] = np.nan
    return out, ref


def _first_order_geometry_failure(exc):
    """True when first_order failed because scalar ABCD geometry is invalid."""
    msg = str(exc)
    return ('centered axial geometry' in msg
            or 'vertex normal to be axial' in msg)


def resolve_exit_pupil(prescription, wavelength, *, stop_index=None, epd=None,
                       field=None, chief=None, axis_point=None, axis_dir=None):
    """Locate the exit-pupil reference point P_xp for a wavefront evaluation.

    Uses the paraxial exit pupil when a stop is resolvable.  Otherwise, uses
    the closest approach of a chief ray to the optical axis.

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
    P_xp : ndarray, shape (3,), or None
        exit-pupil reference point, ready to pass to wavefront(P_xp=...).  None
        when the exit pupil is at infinity (image-space telecentric); the EIC
        closing reads that as its curvature kappa = 0 limit.

    """
    resolved_stop = (stop_index if stop_index is not None
                     else getattr(prescription, 'stop_index', None))
    if resolved_stop is not None:
        try:
            fo = system_first_order(prescription, wvl=wavelength, epd=epd,
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
                # exit pupil at infinity (image-space telecentric): the EIC
                # closing takes this as its curvature kappa = 0 limit, so there
                # is nothing to anchor -- signal it with None.
                return None
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
        tr = trace_cell(prescription, field, wavelength, Sampling.chief(),
                        epd=epd_geo).trace
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
        row index of the chief ray.  Defaults to the ray nearest pupil center.
    status : ndarray, optional
        per-ray status from raytrace.  Invalid rays are excluded when
        provided.  If omitted, rays with non-finite image coordinates are
        excluded.
    reference : str, optional
        image-plane registration point: 'chief' or 'centroid'.

    Returns
    -------
    pupil : ndarray, shape (N,)
        pupil coordinate along the selected axis.
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

    # Use the same reference convention for pupil and image coordinates.
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


def _wavefront_from_trace(prescription, P, wavelength, trace, *, P_xp=None,
                          chief_index=None, pupil_coords=None, field=None,
                          output='length', reference='chief'):
    """Wavefront kernel for callers that already have the raytrace result.

    P_xp anchors the reference sphere through the exit-pupil point; pass None
    (the default) to resolve it from the prescription -- a resolvable stop gives
    the paraxial exit pupil, an off-axis chief its axis closest-approach, and an
    image-space-telecentric system the curvature kappa = 0 (exit pupil at
    infinity) the closing handles natively.
    """
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
    if P_xp is None:
        # resolve from the prescription; None back means the exit pupil is at
        # infinity (telecentric) -> curvature 0, handled below.
        P_xp = resolve_exit_pupil(
            prescription, wavelength, field=field,
            chief=(P_chief_final, trace.S[-1, chief_index]))
    if P_xp is not None:
        P_xp = np.asarray(P_xp, dtype=P.dtype)
    curvature = reference_sphere_curvature(P_xp, P_chief_final)
    filtered_chief = _filtered_chief_index(valid, chief_index)
    opd = hopkins_eic_closing(trace.P[:, valid], trace.S[:, valid],
                              trace.OPL[:, valid],
                              center=P_chief_final, curvature=curvature,
                              n_image=n_image, chief_index=filtered_chief)
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
    """Field-tilt removal and length/waves scaling for wavefront paths."""
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
              P_xp=None,
              chief_index=None,
              pupil_coords=None, field=None, output='length',
              reference='chief'):
    """Trace and compute OPD on the chief-ray-centered reference sphere.

    Parameters
    ----------
    prescription : sequence of Surface
        compiled optical prescription; lensdata.to_surfaces()
    P, S : ndarray, shape (N, 3)
        launch positions and direction cosines (typically from launch()).
    wavelength : float
        in microns.
    P_xp : iterable, optional
        exit-pupil reference point.  Default None resolves it from the
        prescription (paraxial exit pupil for a resolvable stop, the chief-ray
        axis closest-approach otherwise); an image-space-telecentric system
        resolves to the curvature kappa = 0 limit (exit pupil at infinity).
    chief_index : int, optional
        row index of the chief ray.  Defaults to the launch ray nearest the
        pupil center, with invalid rays excluded when reference='centroid'.
    pupil_coords : tuple of array_like, optional
        x and y pupil coordinates to return and use for tilt correction.
    field : Field, optional
        angular field used to remove the launch-plane tilt from collimated
        object-space bundles.  No tilt correction is applied when omitted.
    output : str, optional
        'length' or 'waves'.
    reference : str, optional
        'chief' or 'centroid'.

    Returns
    -------
    opd : ndarray, shape (N,)
        OPD relative to chief.  Units are controlled by output.
    x_pupil, y_pupil : ndarray, shape (N,)
        pupil coordinates.

    """
    if reference not in ('chief', 'centroid'):
        raise ValueError(f"reference must be 'chief' or 'centroid', got {reference!r}")
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

def distortion(prescription, fields=None, wavelength=None, *, epd=None,
               paraxial_fraction=1e-4, distortion_type='f-tan',
               pupil_z=None, samples=101):
    """Per-field image-plane error of the chief ray vs a paraxial proxy.

    Parameters
    ----------
    prescription : sequence of Surface
    fields : iterable of Field, optional
        field points to evaluate.  Must all be kind='angle'.  None sweeps
        the span of the system FieldSet densely; see field_sweep.
    wavelength : float
        in microns.
    epd : float
        entrance pupil diameter.
    paraxial_fraction : float
        scale factor for the paraxial-proxy field angles.  Default 1e-4.
    distortion_type : str, optional
        'f-tan' (default) or 'linear-angle'.
    pupil_z : float, optional
        z position of the entrance pupil used for chief-ray launch.  If
        omitted, launch() defaults to the first surface vertex.
    samples : int, optional
        number of sweep points when fields is None.

    Returns
    -------
    DistortionResult
        Object with real_xy, paraxial_xy, and percent attributes; element i of
        each array corresponds to fields[i].
    real_xy : ndarray, shape (n_fields, 2)
        actual chief-ray image-plane (x, y) per field.
    paraxial_xy : ndarray, shape (n_fields, 2)
        scaled-up paraxial chief-ray landing per field.
    percent : ndarray, shape (n_fields,)
        signed percent distortion.

    """
    wavelength = system_wavelength(prescription, wavelength)
    epd = _require_epd(prescription, epd, wavelength)
    fields = field_sweep(prescription, fields, samples)
    n = len(fields)
    real_xy = np.zeros((n, 2), dtype=config.precision)
    paraxial_xy = np.zeros((n, 2), dtype=config.precision)
    percent = np.zeros(n, dtype=config.precision)
    chief = Sampling.chief()

    # Compare the real chief ray to a tiny-field paraxial proxy.
    angles = [field.angle_radians() for field in fields]
    proxy_fields = [
        Field(float(np.degrees(ax * paraxial_fraction)),
              float(np.degrees(ay * paraxial_fraction)),
              kind='angle', unit='deg')
        for ax, ay in angles
    ]
    real = iter_trace_grid(prescription, fields, [wavelength], chief,
                           epd=epd, pupil_z=pupil_z)
    proxy = iter_trace_grid(prescription, proxy_fields, [wavelength], chief,
                            epd=epd, pupil_z=pupil_z)
    for rr, pr, (ax, ay) in zip(real, proxy, angles):
        i = rr.i
        real_xy[i] = rr.trace.P[-1, 0, :2]
        proxy_xy = pr.trace.P[-1, 0, :2]

        if distortion_type == 'linear-angle':
            paraxial_xy[i] = proxy_xy / paraxial_fraction
        elif distortion_type == 'f-tan':
            small_slopes = np.array([
                np.tan(ax * paraxial_fraction),
                np.tan(ay * paraxial_fraction),
            ], dtype=config.precision)
            field_slopes = np.array([np.tan(ax), np.tan(ay)],
                                    dtype=config.precision)
            focal_scale = np.zeros(2, dtype=config.precision)
            nonzero = np.abs(small_slopes) > 0.0
            focal_scale[nonzero] = proxy_xy[nonzero] / small_slopes[nonzero]
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

def field_curvature(prescription, fields=None, wavelength=None, *,
                    samples=101):
    """X- and y-section parabasal focus z per field point.

    Parameters
    ----------
    prescription : sequence of Surface
    fields : iterable of Field, optional
        field points to evaluate.  None sweeps the span of the system
        FieldSet densely; see field_sweep.
    wavelength : float
        in microns.
    samples : int, optional
        number of sweep points when fields is None.

    Returns
    -------
    FieldCurvatureResult
        Object with x_fan_z and y_fan_z attributes; element i of each array
        corresponds to fields[i].
    x_fan_z : ndarray, shape (n_fields,)
        z where the x-section pencil focuses.
    y_fan_z : ndarray, shape (n_fields,)
        z where the y-section pencil focuses.

    """
    from .parabasal import parabasal_foci  # local: avoid a circular import

    wavelength = system_wavelength(prescription, wavelength)
    fields = field_sweep(prescription, fields, samples)
    n = len(fields)
    x_fan_z = np.zeros(n, dtype=config.precision)
    y_fan_z = np.zeros(n, dtype=config.precision)
    for i, field in enumerate(fields):
        x_fan_z[i], y_fan_z[i] = parabasal_foci(prescription, field,
                                                wavelength)
    return FieldCurvatureResult(x_fan_z, y_fan_z)


# ---------- color -----------------------------------------------------------

def axial_color(prescription, wavelengths=None):
    """Paraxial image distance at each of several wavelengths.

    Parameters
    ----------
    prescription : sequence of Surface
    wavelengths : iterable of float, optional
        wavelengths in microns.  None defaults to the system set, else the
        reference wavelength.

    Returns
    -------
    bfd : ndarray, shape (n_wavelengths,)
        signed paraxial image distance from the last surface vertex.

    """
    wavelengths = _resolve_wavelengths(prescription, wavelengths)
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
    if wavelengths is None or len(wavelengths) == 0:
        return None
    values = [float(w) for w in wavelengths]
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
    if field is None:
        field = Field(0.0, 0.0, unit='deg')
    if sampling is None:
        sampling = Sampling.hex(nrings=8)
    r = trace_cell(prescription, field, wavelength, sampling, epd=epd)
    dz = _best_focus_shift_from_trace(r.trace.P[-1], r.trace.S[-1],
                                      r.trace.status)
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
        focus shift in prescription length units.

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


def lateral_color(prescription, fields=None, wavelengths=None, *, epd=None,
                  samples=101):
    """Chief-ray image-plane landing at every (field, wavelength) pair.

    The lateral chromatic aberration at field i is the difference between
    landing[i, j] across wavelengths j; users typically subtract the
    primary-wavelength row for a wavelength-difference plot.

    Parameters
    ----------
    prescription : sequence of Surface
    fields : iterable of Field, optional
        field points (kind='angle').  None sweeps the span of the system
        FieldSet densely; see field_sweep.
    wavelengths : iterable of float, optional
        wavelengths in microns.  None defaults to the system set, else the
        reference wavelength.
    epd : float
        entrance pupil diameter.
    samples : int, optional
        number of sweep points when fields is None.

    Returns
    -------
    landing : ndarray, shape (n_fields, n_wavelengths, 2)
        chief-ray (x, y) at the image plane.

    """
    # Use one pupil size for all wavelengths.
    epd = _require_epd(prescription, epd)
    fields = field_sweep(prescription, fields, samples)
    wavelengths = _resolve_wavelengths(prescription, wavelengths)
    out = np.zeros((len(fields), len(wavelengths), 2), dtype=config.precision)
    for r in iter_trace_grid(prescription, fields, wavelengths,
                             Sampling.chief(), epd=epd):
        out[r.i, r.j] = r.trace.P[-1, 0, :2]
    return out


# ---------- grid analyses (consistent sampling across field & wavelength) ---
# Stacked grid data for ray-fan, OPD-fan, and spot plots.

def _fan_grid_setup(prescription, fields, wavelengths, nrays, distribution):
    fields = _resolve_fields(prescription, fields)
    wavelengths = _resolve_wavelengths(prescription, wavelengths)
    x_fan = Sampling.fan(n=nrays, axis='x', distribution=distribution)
    y_fan = Sampling.fan(n=nrays, axis='y', distribution=distribution)
    # per-field abscissas: the canonical fan coordinates compressed onto each
    # field's vignetted pupil by the same map launch applies to the samples,
    # still normalized to the nominal (unvignetted) pupil.  A vignetted fan
    # therefore spans less than [-1, 1] -- it is never stretched back out.
    xy_x = x_fan.build(1.0)
    xy_y = y_fan.build(1.0)
    nrays = xy_x.shape[0]
    pupil_x = np.empty((len(fields), nrays), dtype=config.precision)
    pupil_y = np.empty((len(fields), nrays), dtype=config.precision)
    for i, field in enumerate(fields):
        pupil_x[i] = _apply_vignetting(xy_x, field)[:, 0]
        pupil_y[i] = _apply_vignetting(xy_y, field)[:, 1]
    shape = (len(fields), len(wavelengths), nrays)
    x = np.full(shape, np.nan, dtype=config.precision)
    y = np.full(shape, np.nan, dtype=config.precision)
    return fields, wavelengths, x_fan, y_fan, pupil_x, pupil_y, x, y


def _fan_image_error(record, axis, reference):
    """Reference-subtracted image error of one traced ray fan, NaN-padded.

    Returns one value per fan sample (NaN where the ray failed): the transverse
    image-plane error in the fan's own axis measured from the chief or centroid.
    Rays clipped by a real aperture during the trace carry a failure status, so
    the valid mask NaNs them and the fan stays full length.
    """
    ax = _axis_index(axis)
    image = record.trace.P[-1, :, ax]
    ci = _pupil_center_chief_index(record.P)
    centered, _ = _center_valid(image, record.valid, reference, ci)
    return centered


def ray_aberration_fans(prescription, fields=None, wavelengths=None, *,
                        nrays=21, epd=None, distribution='uniform',
                        reference='chief'):
    """Transverse ray-aberration fans for every field and wavelength.

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
        namedtuple (fields, wavelengths, pupil_x, pupil_y, x, y); the pupil
        abscissas are per-field, normalized to the nominal pupil, and span
        only the vignetted extent for a field with vignetting factors.

    """
    fields, wavelengths, x_fan, y_fan, pupil_x, pupil_y, x, y = _fan_grid_setup(
        prescription, fields, wavelengths, nrays, distribution,
    )
    # x/y fan iterators have matching cell order.
    for xr, yr in zip(
            iter_trace_grid(prescription, fields, wavelengths, x_fan, epd=epd),
            iter_trace_grid(prescription, fields, wavelengths, y_fan, epd=epd)):
        x[xr.i, xr.j] = _fan_image_error(xr, 'x', reference)
        y[yr.i, yr.j] = _fan_image_error(yr, 'y', reference)
    return RayFanGrid(tuple(fields),
                      np.asarray(wavelengths, dtype=config.precision),
                      pupil_x, pupil_y, x, y)


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


def _opd_fan(prescription, record, tilt_field, P_xp, output, n_pupil):
    """OPD of one traced ray fan, full length with NaN where rays failed."""
    opd, _, _, valid = _wavefront_from_trace(
        prescription, record.P, record.wvl, record.trace, P_xp=P_xp,
        field=tilt_field, output=output,
    )
    full = np.full(n_pupil, np.nan, dtype=config.precision)
    full[valid] = opd
    return full


def opd_fans(prescription, fields=None, wavelengths=None, *, nrays=21,
             epd=None, distribution='uniform', stop_index=None,
             output='waves'):
    """Wavefront (OPD) fans for every field and wavelength.

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
        namedtuple (fields, wavelengths, pupil_x, pupil_y, x, y); see
        ray_aberration_fans for the pupil abscissa convention.

    """
    fields, wavelengths, x_fan, y_fan, pupil_x, pupil_y, x, y = _fan_grid_setup(
        prescription, fields, wavelengths, nrays, distribution,
    )
    n_pupil = pupil_x.shape[-1]
    # x/y fan iterators have matching cell order.
    for xr, yr in zip(
            iter_trace_grid(prescription, fields, wavelengths, x_fan, epd=epd),
            iter_trace_grid(prescription, fields, wavelengths, y_fan, epd=epd)):
        field = yr.field
        # angular field tilt is removed inside wavefront; height fields carry
        # no launch-plane tilt to remove.
        tilt_field = field if getattr(field, 'kind', 'angle') == 'angle' else None
        P_xp = _exit_pupil_for(prescription, yr.wvl, field=field,
                               stop_index=stop_index, epd=yr.epd)
        x[xr.i, xr.j] = _opd_fan(prescription, xr, tilt_field, P_xp, output,
                                 n_pupil)
        y[yr.i, yr.j] = _opd_fan(prescription, yr, tilt_field, P_xp, output,
                                 n_pupil)
    return OPDFanGrid(tuple(fields),
                      np.asarray(wavelengths, dtype=config.precision),
                      pupil_x, pupil_y, x, y)


def spot_diagrams(prescription, fields=None, wavelengths=None, *,
                  sampling=None, epd=None, reference='centroid'):
    """Image-plane spot data for every field and wavelength.

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
        namedtuple (fields, wavelengths, x, y, valid, reference).

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
    for r in iter_trace_grid(prescription, fields, wavelengths, sampling,
                             epd=epd):
        # rays clipped by a real aperture during the trace are status-flagged
        # (not deleted), so the bundle stays full length and those samples come
        # out NaN via the valid mask.
        v = r.valid
        xi = r.trace.P[-1, :, 0]
        yi = r.trace.P[-1, :, 1]
        image_xy = np.stack([xi, yi], axis=1)
        ci = _pupil_center_chief_index(r.P)
        centered, ref = _center_valid(image_xy, v, reference, ci,
                                      allow_none=True)
        x[r.i, r.j] = centered[:, 0]
        y[r.i, r.j] = centered[:, 1]
        valid[r.i, r.j] = v
        reference_xy[r.i, r.j] = ref
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
    return centroid_referenced_rms(np.asarray(spot_grid.x),
                                   np.asarray(spot_grid.y), axis=2)


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


# ---------- full-field displays ----------------------------------------------
# A 2D map of a scalar image-quality metric over the field disc, in the spirit
# of Code V FMA / full-field displays.

def _full_field_template(prescription, max_field):
    """Field kind/unit/object_z and the field-disc radius for full_field."""
    base = _resolve_fields(prescription, None)
    kinds = {f.kind for f in base}
    if len(kinds) != 1:
        raise ValueError('full_field requires system fields of a single kind')
    kind = kinds.pop()
    if kind == 'angle':
        if len({f.unit for f in base}) != 1:
            raise ValueError(
                'full_field requires system fields with a single angular unit'
            )
        object_z = None
    else:
        if len({f.object_z for f in base}) != 1:
            raise ValueError(
                'full_field requires system fields with a single object plane'
            )
        object_z = base[0].object_z
    unit = base[0].unit
    if max_field is None:
        max_field = max(float(np.hypot(f.hx, f.hy)) for f in base)
    max_field = float(max_field)
    if max_field <= 0.0:
        raise ValueError(
            'full_field needs a nonzero field extent; define off-axis system '
            'fields or pass max_field'
        )
    return kind, unit, object_z, max_field


def _as_wavelength_list(wavelengths):
    """None passes through; a scalar becomes a one-element list."""
    if wavelengths is None:
        return None
    if np.ndim(wavelengths) == 0:
        return [float(wavelengths)]
    return [float(w) for w in wavelengths]


def _spectral_weights(prescription, wavelengths, resolved):
    """System spectral weights when the wavelength set defaulted, else ones."""
    if wavelengths is None:
        w = getattr(prescription, 'weights', None)
        if w is not None and len(w) == len(resolved):
            return [float(x) for x in w]
    return [1.0] * len(resolved)


def _full_field_rms_spot(prescription, fields, wavelengths, sampling, epd):
    """Polychromatic centroid-referenced RMS spot radius per field.

    Rays from all wavelengths pool into one weighted bundle per field, so the
    result includes lateral color blur, not just the per-wavelength average.
    """
    wvls = _resolve_wavelengths(prescription, wavelengths)
    weights = _spectral_weights(prescription, wavelengths, wvls)
    if sampling is None:
        sampling = Sampling.hex(nrings=6)
    n_samples = sampling.build(1.0).shape[0]
    shape = (len(fields), len(wvls), n_samples)
    x = np.full(shape, np.nan, dtype=config.precision)
    y = np.full(shape, np.nan, dtype=config.precision)
    for r in iter_trace_grid(prescription, fields, wvls, sampling, epd=epd):
        v = r.valid
        xi = np.full(n_samples, np.nan, dtype=config.precision)
        yi = np.full(n_samples, np.nan, dtype=config.precision)
        xi[v] = r.trace.P[-1, v, 0]
        yi[v] = r.trace.P[-1, v, 1]
        x[r.i, r.j] = xi
        y[r.i, r.j] = yi
    w = np.asarray(weights, dtype=config.precision)[None, :, None]
    m = np.isfinite(x)
    wm = np.where(m, w, 0.0)
    xw = np.where(m, x, 0.0)
    yw = np.where(m, y, 0.0)
    wsum = wm.sum(axis=(1, 2))
    safe = np.where(wsum > 0.0, wsum, 1.0)
    cx = (wm * xw).sum(axis=(1, 2)) / safe
    cy = (wm * yw).sum(axis=(1, 2)) / safe
    r2 = (xw - cx[:, None, None])**2 + (yw - cy[:, None, None])**2
    rms = np.sqrt((wm * r2).sum(axis=(1, 2)) / safe)
    rms[wsum == 0.0] = np.nan
    return rms


def _full_field_rms_wfe(prescription, fields, wavelength, sampling, epd,
                        stop_index):
    """Piston-removed RMS wavefront error in waves per field."""
    if sampling is None:
        sampling = Sampling.hex(nrings=6)
    out = np.full(len(fields), np.nan, dtype=config.precision)
    for i, field in enumerate(fields):
        r = trace_cell(prescription, field, wavelength, sampling, epd=epd)
        tilt_field = field if field.kind == 'angle' else None
        P_xp = _exit_pupil_for(prescription, wavelength, field=field,
                               stop_index=stop_index, epd=r.epd)
        try:
            opd, _, _, _ = _wavefront_from_trace(
                prescription, r.P, wavelength, r.trace, P_xp=P_xp,
                field=tilt_field, output='waves')
        except ValueError:
            # the chief ray was clipped; this field is a hole in the map
            continue
        if opd.size:
            resid = opd - np.mean(opd)
            out[i] = float(np.sqrt(np.mean(resid * resid)))
    return out


def full_field(prescription, metric='rms spot', *, samples=15, max_field=None,
               wavelengths=None, sampling=None, epd=None, stop_index=None):
    """Scalar image-quality metric over a 2D grid of field points.

    The full-field analog of Code V FMA / full-field displays: a samples x
    samples Cartesian grid spans the field square, and points inside the
    field disc (radius max_field) are evaluated; points outside are NaN.

    Parameters
    ----------
    prescription : sequence of Surface or OpticalSystem
        the optical system.
    metric : str, optional
        which scalar to evaluate per field point:

        - 'rms spot': polychromatic centroid-referenced RMS spot radius, in
          prescription length units.  Rays from every wavelength pool into
          one spectrally-weighted bundle.
        - 'rms wfe': piston-removed RMS wavefront error in waves, at the
          first given wavelength (default the reference wavelength).
        - 'distortion': signed percent distortion of the chief ray vs the
          paraxial proxy, at the first given wavelength.
        - 'lateral color': magnitude of the chief-ray landing separation
          between the shortest and longest wavelengths, in length units.

    samples : int, optional
        grid points per axis.
    max_field : float, optional
        field-disc radius, in the system field units; defaults to the
        largest system field magnitude.
    wavelengths : float or iterable of float, optional
        wavelengths in microns; defaults to the system set.  The
        monochromatic metrics use the first entry.
    sampling : Sampling, optional
        pupil sampling for the bundle metrics ('rms spot', 'rms wfe');
        defaults to a 6-ring hexapolar grid.
    epd : float, optional
        entrance pupil diameter; defaults from a system aperture spec.
    stop_index : int, optional
        aperture-stop index for exit-pupil resolution ('rms wfe' only).

    Returns
    -------
    FullFieldGrid
        namedtuple (hx, hy, data, metric, kind, unit); hx, hy, and data are
        (samples, samples) arrays, with data NaN outside the field disc.

    """
    kind, unit, object_z, radius = _full_field_template(prescription, max_field)
    wavelengths = _as_wavelength_list(wavelengths)
    coords = np.linspace(-radius, radius, int(samples))
    hx, hy = np.meshgrid(coords, coords)
    inside = np.hypot(hx, hy) <= radius * (1.0 + 1e-9)
    idx = np.nonzero(inside.ravel())[0]
    flat_fields = [
        Field(float(fx), float(fy), kind=kind, unit=unit, object_z=object_z)
        for fx, fy in zip(hx.ravel()[idx], hy.ravel()[idx])
    ]
    key = metric.lower().replace('-', ' ').replace('_', ' ')
    if key == 'rms spot':
        values = _full_field_rms_spot(prescription, flat_fields, wavelengths,
                                      sampling, epd)
    elif key == 'rms wfe':
        wvl = system_wavelength(
            prescription, None if wavelengths is None else wavelengths[0])
        values = _full_field_rms_wfe(prescription, flat_fields, wvl, sampling,
                                     epd, stop_index)
    elif key == 'distortion':
        wvl = None if wavelengths is None else wavelengths[0]
        values = distortion(prescription, flat_fields, wvl, epd=epd).percent
    elif key == 'lateral color':
        wvls = _resolve_wavelengths(prescription, wavelengths)
        if len(wvls) < 2:
            raise ValueError(
                "metric 'lateral color' needs at least two wavelengths"
            )
        landing = lateral_color(prescription, flat_fields, wvls, epd=epd)
        d = (landing[:, int(np.argmax(wvls))]
             - landing[:, int(np.argmin(wvls))])
        values = np.hypot(d[:, 0], d[:, 1])
    else:
        raise ValueError(
            "metric must be 'rms spot', 'rms wfe', 'distortion', or "
            f"'lateral color', got {metric!r}"
        )
    data = np.full(hx.size, np.nan, dtype=config.precision)
    data[idx] = np.asarray(values, dtype=config.precision)
    return FullFieldGrid(hx, hy, data.reshape(hx.shape), key, kind, unit)
