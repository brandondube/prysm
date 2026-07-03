"""Ray optics analysis.
"""

from collections import namedtuple
from dataclasses import dataclass

from prysm.conf import config
from prysm.mathops import np

from prysm.polynomials import zernike_nm_seq, lstsq

from .spencer_and_murty import raytrace, valid_mask, _is_measurement_surf
from .opt import (
    xp_reference_sphere,
    hopkins_eic_closing,
    reference_sphere_curvature,
    centroid_referenced_rms,
    centroid_referenced_max,
    _pupil_center_chief_index,
)
from .paraxial import paraxial_image_distance
from .launch import Field, Sampling, _apply_vignetting
from ._trace_grid import (
    TraceRecord, iter_trace_grid, trace_cell, _resolve_fields,  # NOQA: F401 re-export
    _resolve_wavelengths, _require_epd, field_sweep,
)
from ._resolve import compiled_surfaces, resolve_wavelength, trace_context
from .surfaces import Conic, EvenAsphere, Plane, Sphere


# ---------- result containers ----------------------------------------------
# Grid arrays are indexed [field_index, wavelength_index, sample_index].


@dataclass(frozen=True, slots=True)
class DistortionResult:
    real_xy: object
    paraxial_xy: object
    percent: object
    fields: object = None


@dataclass(frozen=True, slots=True)
class FieldCurvatureResult:
    x_fan_z: object
    y_fan_z: object
    fields: object = None
    labels: object = None
    image_z: object = None


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


def resolve_exit_pupil(system, wavelength, *, stop_index=None, epd=None,
                       field=None, chief=None, axis_point=None, axis_dir=None,
                       min_perp=1e-6, return_mode=False):
    """Locate the exit-pupil reference point P_xp for a wavefront evaluation.

    Uses a paraxial stop when available, otherwise the chief-axis closest
    approach.

    Parameters
    ----------
    system : sequence of Surface or OpticalSystem
    wavelength : float
        in microns
    stop_index : int, optional
        aperture-stop surface index; defaults to the system stop_index.
    epd : float, optional
        entrance-pupil diameter for the paraxial solve.
    field : Field, optional
        field for the geometric-route chief ray.
    chief : tuple of ndarray, optional
        post-trace chief position and direction for the geometric route.
    axis_point, axis_dir : iterable, optional
        point on, and direction of, the optical axis.
    min_perp : float, optional
        minimum chief slope to the axis for the geometric route.
    return_mode : bool, optional
        if True, return (P_xp, 'paraxial' or 'geometric').

    Returns
    -------
    P_xp : ndarray, shape (3,), or None
        exit-pupil reference point, or None for image-space telecentric.

    """
    def _ret(P_xp, mode):
        return (P_xp, mode) if return_mode else P_xp

    resolved_stop = (stop_index if stop_index is not None
                     else getattr(system, 'stop_index', None))
    if resolved_stop is not None:
        try:
            resolver = getattr(system, '_ynu_first_order', None)
            if callable(resolver):
                fo = resolver(wvl=wavelength, epd=epd,
                              stop_index=resolved_stop)
            else:
                from .paraxial import ynu_first_order
                fo = ynu_first_order(compiled_surfaces(system),
                                     wvl=wavelength, epd=epd,
                                     stop_index=resolved_stop)
        except ValueError as exc:
            # Only explicit-axis calls can fall back from centered ABCD errors.
            if ((axis_point is None and axis_dir is None)
                    or not _first_order_geometry_failure(exc)):
                raise
        else:
            if fo.xp_z is None:
                # Image-space telecentric: kappa = 0, no finite anchor.
                return _ret(None, 'paraxial')
            P_xp = np.array([0.0, 0.0, float(fo.xp_z)], dtype=config.precision)
            return _ret(P_xp, 'paraxial')

    # geometric route: take the chief ray's axis closest-approach.
    if chief is not None:
        P_chief_final, S_chief_final = chief
    else:
        if field is None:
            field = Field(0.0, 0.0)
        epd_geo = epd
        if epd_geo is None:
            resolver = getattr(system, 'entrance_pupil_diameter', None)
            if callable(resolver):
                epd_geo = resolver(wavelength)
        if epd_geo is None:
            epd_geo = 1.0  # chief is a single pupil-center ray; epd only nominal
        tr = trace_cell(system, field, wavelength, Sampling.chief(),
                        epd=epd_geo).trace
        P_chief_final, S_chief_final = tr.P[-1, 0], tr.S[-1, 0]
    _, _, P_xp = xp_reference_sphere(P_chief_final, S_chief_final,
                                     axis_point=axis_point, axis_dir=axis_dir,
                                     min_perp=min_perp)
    return _ret(np.asarray(P_xp, dtype=config.precision), 'geometric')


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
    pupil = P_hist[0, :, ax]
    image = P_hist[-1, :, ax]

    valid = valid_mask(status, P_hist[-1])

    # Use the same reference convention for pupil and image coordinates.
    if reference == 'chief':
        ref_pupil = pupil[chief_index]
    elif reference == 'centroid':
        ref_pupil = np.mean(pupil[valid])
    else:
        ref_pupil = _reference_value(pupil, valid, reference, chief_index)
    ref_image = _reference_value(image, valid, reference, chief_index)
    return pupil[valid] - ref_pupil, image[valid] - ref_image


def spot_positions(P_final, status=None, origin=None):
    """Valid image-plane spot landings, optionally re-centered.

    Parameters
    ----------
    P_final : ndarray, shape (N, 3)
        final ray positions (trace.P[-1]).
    status : ndarray, optional
        per-ray status from raytrace; invalid rays are dropped when given.
    origin : str or iterable, optional
        center to subtract: 'centroid' or an explicit (x, y).

    Returns
    -------
    x, y : ndarray
        image-plane coordinates of the surviving rays.

    """
    P_final = np.asarray(P_final)
    x = P_final[..., 0]
    y = P_final[..., 1]
    if status is not None:
        valid = valid_mask(status, P_final)
        x = x[valid]
        y = y[valid]
    if origin is not None:
        if isinstance(origin, str):
            if origin.lower() == 'centroid':
                origin = (np.nanmean(x), np.nanmean(y))
            else:
                raise ValueError("origin string must be 'centroid'")
        origin = np.asarray(origin)
        x = x - origin[0]
        y = y - origin[1]
    return x, y


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


def _require_valid_chief(valid, chief_index, reference='chief'):
    """Raise the canonical error when the anchor ray did not survive."""
    if bool(valid[chief_index]):
        return
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


class ReferenceSphereClosing:
    """Chief-zeroed OPD and reusable reference-sphere geometry."""

    __slots__ = ('opd', 'curvature', 'filtered_chief', 'R', 'delta')

    def __init__(self, opd, curvature, filtered_chief, R, delta):
        self.opd = opd
        self.curvature = curvature
        self.filtered_chief = filtered_chief
        self.R = R
        self.delta = delta


def close_on_reference_sphere(trace, valid, chief_index, *, center, P_xp,
                              n_image):
    """Close a traced bundle onto the chief-image reference sphere.

    P_xp may be None for the telecentric kappa = 0 limit.
    """
    center = np.asarray(center)
    curvature = reference_sphere_curvature(P_xp, center)
    if P_xp is None:
        delta = None
        R = np.inf
    else:
        delta = np.asarray(P_xp, dtype=center.dtype) - center
        R = float(np.sqrt(np.sum(delta * delta)))
    filtered_chief = _filtered_chief_index(valid, chief_index)
    opd = hopkins_eic_closing(trace.P[:, valid], trace.S[:, valid],
                              trace.OPL[:, valid], center=center,
                              curvature=curvature, n_image=n_image,
                              chief_index=filtered_chief)
    return ReferenceSphereClosing(opd, curvature, filtered_chief, R, delta)


class WavefrontClosing:
    """Closed wavefront of one traced bundle, with the geometry that made it."""

    __slots__ = ('opd', 'valid', 'chief_index', 'center', 'P_xp', 'xp_mode',
                 'curvature', 'R', 'delta', 'filtered_chief', 'n_image')

    def __init__(self, opd, valid, chief_index, center, P_xp, xp_mode,
                 curvature, R, delta, filtered_chief, n_image):
        self.opd = opd
        self.valid = valid
        self.chief_index = chief_index
        self.center = center
        self.P_xp = P_xp
        self.xp_mode = xp_mode
        self.curvature = curvature
        self.R = R
        self.delta = delta
        self.filtered_chief = filtered_chief
        self.n_image = n_image


def close_wavefront(system, trace, wavelength, chief_index, *, field=None,
                    center=None, P_xp=None, stop_index=None, epd=None,
                    axis_point=None, axis_dir=None, min_perp=1e-6, valid=None,
                    reference='chief', apply_field_tilt=True, ctx=None):
    """Close a traced bundle into a chief-referenced OPD, resolving as needed.

    Owns the full recipe: validity, medium indices, exit-pupil resolution,
    EIC closing, and the launch-plane field-tilt ramp.  Chief selection stays
    with the caller.  OPD is in length units; output scaling is presentation.

    Parameters
    ----------
    system : sequence of Surface or OpticalSystem
    trace : RayTraceResult
        traced bundle to close.
    wavelength : float
        in microns.
    chief_index : int
        row index of the anchor ray.
    field : Field, optional
        angular field whose launch-plane tilt ramp is removed when
        apply_field_tilt.
    center : ndarray, optional
        reference-sphere center; defaults to the chief image point.
    P_xp : iterable, optional
        exit-pupil point; default resolves through resolve_exit_pupil.
    stop_index, epd, axis_point, axis_dir, min_perp
        forwarded to resolve_exit_pupil when P_xp is None.
    valid : ndarray, optional
        bool valid-ray mask; defaults to the trace validity.
    reference : str, optional
        'chief' or 'centroid'; phrases the invalid-anchor error.
    apply_field_tilt : bool, optional
        apply the field ramp from trace.P[0] when field is given.
    ctx : TraceContext, optional
        resolved metadata; built from the system when omitted.

    Returns
    -------
    WavefrontClosing
        opd over the valid rays plus the reusable closing geometry.

    """
    if valid is None:
        valid = valid_mask(trace.status, trace.P[-1])
    chief_index = int(chief_index)
    _require_valid_chief(valid, chief_index, reference)
    if ctx is None:
        ctx = trace_context(system, wavelength)
    n_image = ctx.n_image
    P_chief = trace.P[-1, chief_index]
    if center is None:
        center = P_chief
    if P_xp is None:
        # None back means telecentric image space: curvature 0.
        P_xp, xp_mode = resolve_exit_pupil(
            system, wavelength, stop_index=stop_index, epd=epd,
            chief=(P_chief, trace.S[-1, chief_index]),
            axis_point=axis_point, axis_dir=axis_dir, min_perp=min_perp,
            return_mode=True)
    else:
        xp_mode = 'fixed'
    if P_xp is not None:
        P_xp = np.asarray(P_xp, dtype=config.precision)
    closing = close_on_reference_sphere(trace, valid, chief_index,
                                        center=center, P_xp=P_xp,
                                        n_image=n_image)
    opd = closing.opd
    if apply_field_tilt and field is not None:
        ax, ay = field.angle_radians()
        P0 = trace.P[0]
        x_pupil = P0[valid, 0] - P0[chief_index, 0]
        y_pupil = P0[valid, 1] - P0[chief_index, 1]
        opd = opd + (np.sin(ax) * x_pupil + np.sin(ay) * y_pupil)
    return WavefrontClosing(opd, valid, chief_index, center, P_xp, xp_mode,
                            closing.curvature, closing.R, closing.delta,
                            closing.filtered_chief, n_image)


def _wavefront_from_trace(system, P, wavelength, trace, *, P_xp=None,
                          chief_index=None, pupil_coords=None, field=None,
                          output='length', reference='chief'):
    """Wavefront kernel for callers that already have the raytrace result."""
    valid = valid_mask(trace.status, trace.P[-1])
    chief_index = _resolve_chief_index(P, valid, reference, chief_index)
    closing = close_wavefront(system, trace, wavelength, chief_index,
                              field=field, P_xp=P_xp, valid=valid,
                              reference=reference,
                              apply_field_tilt=pupil_coords is None)
    if pupil_coords is None:
        x_pupil = P[valid, 0] - P[chief_index, 0]
        y_pupil = P[valid, 1] - P[chief_index, 1]
        tilt_field = None   # ramp already applied by close_wavefront
    else:
        x_pupil = np.asarray(pupil_coords[0])[valid]
        y_pupil = np.asarray(pupil_coords[1])[valid]
        tilt_field = field  # override coordinates carry the ramp
    opd, _ = _apply_field_and_output(closing.opd, x_pupil, y_pupil, tilt_field,
                                     output, wavelength)
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


def wavefront(system, P, S, wavelength, *,
              P_xp=None,
              chief_index=None,
              pupil_coords=None, field=None, output='length',
              reference='chief'):
    """Trace and compute OPD on the chief-ray-centered reference sphere.

    Parameters
    ----------
    system : sequence of Surface
        compiled optical system; lensdata.to_surfaces()
    P, S : ndarray, shape (N, 3)
        launch positions and direction cosines (typically from launch()).
    wavelength : float
        in microns.
    P_xp : iterable, optional
        exit-pupil reference point.  Default None resolves it from the
        system (paraxial exit pupil for a resolvable stop, the chief-ray
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
    trace = raytrace(system, P, S, wavelength)
    opd, x_pupil, y_pupil, _ = _wavefront_from_trace(
        system, P, wavelength, trace, P_xp=P_xp,
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

def distortion(system, fields=None, wavelength=None, *, epd=None,
               paraxial_fraction=1e-4, distortion_type='f-tan',
               pupil_z=None, samples=101):
    """Per-field image-plane error of the chief ray vs a paraxial proxy.

    Parameters
    ----------
    system : sequence of Surface
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
        Object with real_xy, paraxial_xy, percent, and fields attributes;
        element i of each array corresponds to fields[i].
    real_xy : ndarray, shape (n_fields, 2)
        actual chief-ray image-plane (x, y) per field.
    paraxial_xy : ndarray, shape (n_fields, 2)
        scaled-up paraxial chief-ray landing per field.
    percent : ndarray, shape (n_fields,)
        signed percent distortion.

    """
    wavelength = resolve_wavelength(system, wavelength)
    epd = _require_epd(system, epd, wavelength)
    fields = field_sweep(system, fields, samples)
    n = len(fields)
    real_xy = np.zeros((n, 2), dtype=config.precision)
    paraxial_xy = np.zeros((n, 2), dtype=config.precision)
    percent = np.zeros(n, dtype=config.precision)
    chief = Sampling.chief()

    # Compare the real chief ray to a tiny-field paraxial proxy.
    for i, field in enumerate(fields):
        ax, ay = field.angle_radians()
        proxy_field = Field(float(np.degrees(ax * paraxial_fraction)),
                            float(np.degrees(ay * paraxial_fraction)),
                            kind='angle', unit='deg')
        real = trace_cell(system, field, wavelength, chief,
                          epd=epd, pupil_z=pupil_z)
        proxy = trace_cell(system, proxy_field, wavelength, chief,
                           epd=epd, pupil_z=pupil_z)
        real_xy[i] = real.trace.P[-1, 0, :2]
        proxy_xy = proxy.trace.P[-1, 0, :2]

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

    return DistortionResult(real_xy, paraxial_xy, percent, tuple(fields))


# ---------- field curvature -------------------------------------------------

_AXISYMMETRIC_SHAPES = (Plane, Sphere, Conic, EvenAsphere)


def _field_is_pure_y(field):
    """True when a field lies in the classical y-z meridian."""
    return abs(float(getattr(field, 'hx', 0.0))) <= 1e-12


def _system_is_axisymmetric(system):
    """Conservative axisymmetry check for field-curvature labels."""
    surfaces = (system.to_surfaces()
                if hasattr(system, 'to_surfaces') else list(system))
    for surf in surfaces:
        if getattr(surf, 'R', None) is not None:
            return False
        P = np.asarray(getattr(surf, 'P', (0.0, 0.0, 0.0)))
        if np.any(np.abs(P[:2]) > 1e-12):
            return False
        if not isinstance(getattr(surf, 'shape', None), _AXISYMMETRIC_SHAPES):
            return False
    return True


def _field_curvature_labels(system, fields):
    """Curve labels for the two field_curvature output arrays."""
    fields = list(fields)
    if fields and all(_field_is_pure_y(field) for field in fields) \
            and _system_is_axisymmetric(system):
        return ('S', 'T'), ('sagittal', 'tangential')
    return ('X', 'Y'), ('x fan', 'y fan')

def field_curvature(system, fields=None, wavelength=None, *,
                    samples=101):
    """X- and y-section parabasal focus z per field point.

    Parameters
    ----------
    system : sequence of Surface
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
        Object with x_fan_z, y_fan_z, fields, labels, and image_z attributes;
        element i of each array corresponds to fields[i].
    x_fan_z : ndarray, shape (n_fields,)
        z where the x-section pencil focuses.
    y_fan_z : ndarray, shape (n_fields,)
        z where the y-section pencil focuses.

    """
    from .parabasal import parabasal_foci  # local: avoid a circular import

    ctx = trace_context(system, wavelength)
    wavelength = ctx.wavelength
    fields = field_sweep(system, fields, samples)
    n = len(fields)
    x_fan_z = np.zeros(n, dtype=config.precision)
    y_fan_z = np.zeros(n, dtype=config.precision)
    for i, field in enumerate(fields):
        x_fan_z[i], y_fan_z[i] = parabasal_foci(system, field,
                                                wavelength)
    labels, _ = _field_curvature_labels(ctx.surfaces, fields)
    return FieldCurvatureResult(x_fan_z, y_fan_z, tuple(fields), labels,
                                float(ctx.surfaces[-1].P[2]))


# ---------- color -----------------------------------------------------------

def _system_wavelength_range(system):
    """Wavelength span from OpticalSystem metadata, or None."""
    wavelengths = getattr(system, 'wavelengths', None)
    if wavelengths is None or len(wavelengths) == 0:
        return None
    values = [float(w) for w in wavelengths]
    return min(values), max(values)


def _chromatic_wavelength_samples(system, wavelengths, samples):
    if wavelengths is not None:
        return np.asarray([float(w) for w in wavelengths], dtype=config.precision)
    span = _system_wavelength_range(system)
    if span is None:
        raise TypeError(
            'wavelengths is required unless system carries system '
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


def _best_focus_z(system, wavelength, *, epd, field, sampling):
    """Lab-frame centroid-RMS best focus for a traced bundle."""
    if field is None:
        field = Field(0.0, 0.0, unit='deg')
    if sampling is None:
        sampling = Sampling.hex(nrings=8)
    r = trace_cell(system, field, wavelength, sampling, epd=epd)
    dz = _best_focus_shift_from_trace(r.trace.P[-1], r.trace.S[-1],
                                      r.trace.status)
    return float(compiled_surfaces(system)[-1].P[2]) + dz


def _chromatic_focus_z(system, wavelength, focus, *, epd, field,
                       sampling):
    surfaces = compiled_surfaces(system)
    if focus == 'paraxial':
        # paraxial_image_distance references the last interacting vertex; add
        # it to that vertex's z for the absolute focus
        ref = surfaces
        while len(ref) > 1 and _is_measurement_surf(
                getattr(ref[-1], 'typ', None)):
            ref = ref[:-1]
        return (float(ref[-1].P[2])
                + float(paraxial_image_distance(surfaces, wvl=wavelength)))
    if focus == 'best':
        return _best_focus_z(
            system, wavelength, epd=epd, field=field,
            sampling=sampling,
        )
    raise ValueError(f"focus must be 'best' or 'paraxial', got {focus!r}")


def chromatic_focal_shift(system, wavelengths=None, *,
                          reference_wavelength=None, focus='best',
                          epd=None, field=None, sampling=None, samples=101):
    """Best-focus shift as a smooth function of wavelength.

    Parameters
    ----------
    system : sequence of Surface
        optical system.
    wavelengths : iterable of float, optional
        wavelengths in microns.  If omitted, samples spans the full range of
        system.wavelengths.
    reference_wavelength : float, optional
        wavelength in microns whose focus is used as zero.  Defaults to the
        system reference wavelength when available.
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
        focus shift in system length units.

    """
    wavelengths = _chromatic_wavelength_samples(system, wavelengths,
                                                samples)
    if reference_wavelength is None:
        reference_wavelength = resolve_wavelength(system, None)
    reference_wavelength = float(reference_wavelength)
    focus = focus.lower()
    foci = np.array([
        _chromatic_focus_z(
            system, float(w), focus, epd=epd, field=field,
            sampling=sampling,
        )
        for w in wavelengths
    ], dtype=config.precision)

    ref = _chromatic_focus_z(
        system, reference_wavelength, focus, epd=epd, field=field,
        sampling=sampling,
    )
    return wavelengths, foci - ref


def lateral_color(system, fields=None, wavelengths=None, *, epd=None,
                  samples=101):
    """Chief-ray image-plane landing at every (field, wavelength) pair.

    The lateral chromatic aberration at field i is the difference between
    landing[i, j] across wavelengths j; users typically subtract the
    primary-wavelength row for a wavelength-difference plot.

    Parameters
    ----------
    system : sequence of Surface
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
    epd = _require_epd(system, epd)
    fields = field_sweep(system, fields, samples)
    wavelengths = _resolve_wavelengths(system, wavelengths)
    out = np.zeros((len(fields), len(wavelengths), 2), dtype=config.precision)
    for r in iter_trace_grid(system, fields, wavelengths,
                             Sampling.chief(), epd=epd):
        out[r.i, r.j] = r.trace.P[-1, 0, :2]
    return out


# ---------- grid analyses (consistent sampling across field & wavelength) ---
# Stacked grid data for ray-fan, OPD-fan, and spot plots.

def _fan_grid_setup(system, fields, wavelengths, nrays, distribution):
    fields = _resolve_fields(system, fields)
    wavelengths = _resolve_wavelengths(system, wavelengths)
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


def ray_aberration_fans(system, fields=None, wavelengths=None, *,
                        nrays=21, epd=None, distribution='uniform',
                        reference='chief'):
    """Transverse ray-aberration fans for every field and wavelength.

    Parameters
    ----------
    system : sequence of Surface or OpticalSystem
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
        system, fields, wavelengths, nrays, distribution,
    )
    # x/y fan iterators have matching cell order.
    for xr, yr in zip(
            iter_trace_grid(system, fields, wavelengths, x_fan, epd=epd),
            iter_trace_grid(system, fields, wavelengths, y_fan, epd=epd)):
        x[xr.i, xr.j] = _fan_image_error(xr, 'x', reference)
        y[yr.i, yr.j] = _fan_image_error(yr, 'y', reference)
    return RayFanGrid(tuple(fields),
                      np.asarray(wavelengths, dtype=config.precision),
                      pupil_x, pupil_y, x, y)


def _exit_pupil_for(system, wavelength, *, field=None, stop_index=None,
                    epd=None):
    """Resolve P_xp, using the system's cached exit_pupil when available.

    An OpticalSystem memoizes the (field-independent paraxial) exit pupil per
    wavelength; a bare surface list / LensData resolves it directly.
    """
    if hasattr(system, 'exit_pupil') and hasattr(system, 'lens'):
        return system.exit_pupil(wavelength, field=field,
                                       stop_index=stop_index, epd=epd)
    return resolve_exit_pupil(system, wavelength, stop_index=stop_index,
                              epd=epd, field=field)


def _opd_fan(system, record, tilt_field, P_xp, output, n_pupil):
    """OPD of one traced ray fan, full length with NaN where rays failed."""
    opd, _, _, valid = _wavefront_from_trace(
        system, record.P, record.wvl, record.trace, P_xp=P_xp,
        field=tilt_field, output=output,
    )
    full = np.full(n_pupil, np.nan, dtype=config.precision)
    full[valid] = opd
    return full


def opd_fans(system, fields=None, wavelengths=None, *, nrays=21,
             epd=None, distribution='uniform', stop_index=None,
             output='waves'):
    """Wavefront (OPD) fans for every field and wavelength.

    Parameters
    ----------
    system : sequence of Surface or OpticalSystem
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
        system, fields, wavelengths, nrays, distribution,
    )
    n_pupil = pupil_x.shape[-1]
    # x/y fan iterators have matching cell order.
    for xr, yr in zip(
            iter_trace_grid(system, fields, wavelengths, x_fan, epd=epd),
            iter_trace_grid(system, fields, wavelengths, y_fan, epd=epd)):
        field = yr.field
        # angular field tilt is removed inside wavefront; height fields carry
        # no launch-plane tilt to remove.
        tilt_field = field if getattr(field, 'kind', 'angle') == 'angle' else None
        P_xp = _exit_pupil_for(system, yr.wvl, field=field,
                               stop_index=stop_index, epd=yr.epd)
        x[xr.i, xr.j] = _opd_fan(system, xr, tilt_field, P_xp, output,
                                 n_pupil)
        y[yr.i, yr.j] = _opd_fan(system, yr, tilt_field, P_xp, output,
                                 n_pupil)
    return OPDFanGrid(tuple(fields),
                      np.asarray(wavelengths, dtype=config.precision),
                      pupil_x, pupil_y, x, y)


def spot_diagrams(system, fields=None, wavelengths=None, *,
                  sampling=None, epd=None, reference='centroid'):
    """Image-plane spot data for every field and wavelength.

    Parameters
    ----------
    system : sequence of Surface or OpticalSystem
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
    fields = _resolve_fields(system, fields)
    wavelengths = _resolve_wavelengths(system, wavelengths)
    if sampling is None:
        sampling = Sampling.hex(nrings=6)
    nf = len(fields)
    nw = len(wavelengths)
    n_samples = sampling.build(1.0).shape[0]
    x = np.full((nf, nw, n_samples), np.nan, dtype=config.precision)
    y = np.full((nf, nw, n_samples), np.nan, dtype=config.precision)
    valid = np.zeros((nf, nw, n_samples), dtype=bool)
    reference_xy = np.full((nf, nw, 2), np.nan, dtype=config.precision)
    for r in iter_trace_grid(system, fields, wavelengths, sampling,
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


def spot_rms_radius(spot_grid):
    """Centroid-referenced RMS spot radius per field and wavelength.

    Re-centers each bundle on its own centroid first, so the result is the true
    geometric RMS radius independent of the grid's stored reference.

    Returns
    -------
    ndarray, shape (n_fields, n_wavelengths)
        RMS radius in system length units; entry [i, j] is fields[i] at
        wavelengths[j].

    """
    return centroid_referenced_rms(np.asarray(spot_grid.x),
                                   np.asarray(spot_grid.y), axis=2)


def spot_geometric_radius(spot_grid):
    """Maximum (geometric) spot radius from the centroid per field/wavelength.

    Returns
    -------
    ndarray, shape (n_fields, n_wavelengths)
        the farthest valid ray from the centroid, in system length units;
        entry [i, j] is fields[i] at wavelengths[j].

    """
    return centroid_referenced_max(np.asarray(spot_grid.x),
                                   np.asarray(spot_grid.y), axis=2)


# ---------- full-field displays ----------------------------------------------
# A 2D map of a scalar image-quality metric over the field disc, in the spirit
# of Code V FMA / full-field displays.

def _full_field_template(system, max_field):
    """Field kind/unit/object_z and the field-disc radius for full_field."""
    base = _resolve_fields(system, None)
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


def _spectral_weights(system, wavelengths, resolved):
    """System spectral weights when the wavelength set defaulted, else ones."""
    if wavelengths is None:
        w = getattr(system, 'weights', None)
        if w is not None and len(w) == len(resolved):
            return [float(x) for x in w]
    return [1.0] * len(resolved)


def _full_field_rms_spot(system, fields, wavelengths, sampling, epd):
    """Polychromatic centroid-referenced RMS spot radius per field.

    Rays from all wavelengths pool into one weighted bundle per field, so the
    result includes lateral color blur, not just the per-wavelength average.
    """
    wvls = _resolve_wavelengths(system, wavelengths)
    weights = _spectral_weights(system, wavelengths, wvls)
    if sampling is None:
        sampling = Sampling.hex(nrings=6)
    n_samples = sampling.build(1.0).shape[0]
    shape = (len(fields), len(wvls), n_samples)
    x = np.full(shape, np.nan, dtype=config.precision)
    y = np.full(shape, np.nan, dtype=config.precision)
    for r in iter_trace_grid(system, fields, wvls, sampling, epd=epd):
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


def _full_field_rms_wfe(system, fields, wavelength, sampling, epd,
                        stop_index):
    """Piston-removed RMS wavefront error in waves per field."""
    if sampling is None:
        sampling = Sampling.hex(nrings=6)
    out = np.full(len(fields), np.nan, dtype=config.precision)
    for i, field in enumerate(fields):
        r = trace_cell(system, field, wavelength, sampling, epd=epd)
        tilt_field = field if field.kind == 'angle' else None
        P_xp = _exit_pupil_for(system, wavelength, field=field,
                               stop_index=stop_index, epd=r.epd)
        try:
            opd, _, _, _ = _wavefront_from_trace(
                system, r.P, wavelength, r.trace, P_xp=P_xp,
                field=tilt_field, output='waves')
        except ValueError:
            # the chief ray was clipped; this field is a hole in the map
            continue
        if opd.size:
            resid = opd - np.mean(opd)
            out[i] = float(np.sqrt(np.mean(resid * resid)))
    return out


def full_field(system, metric='rms spot', *, samples=15, max_field=None,
               wavelengths=None, sampling=None, epd=None, stop_index=None):
    """Scalar image-quality metric over a 2D grid of field points.

    The full-field analog of Code V FMA / full-field displays: a samples x
    samples Cartesian grid spans the field square, and points inside the
    field disc (radius max_field) are evaluated; points outside are NaN.

    Parameters
    ----------
    system : sequence of Surface or OpticalSystem
        the optical system.
    metric : str, optional
        which scalar to evaluate per field point:

        - 'rms spot': polychromatic centroid-referenced RMS spot radius, in
          system length units.  Rays from every wavelength pool into
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
    kind, unit, object_z, radius = _full_field_template(system, max_field)
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
        values = _full_field_rms_spot(system, flat_fields, wavelengths,
                                      sampling, epd)
    elif key == 'rms wfe':
        wvl = resolve_wavelength(
            system, None if wavelengths is None else wavelengths[0])
        values = _full_field_rms_wfe(system, flat_fields, wvl, sampling,
                                     epd, stop_index)
    elif key == 'distortion':
        wvl = None if wavelengths is None else wavelengths[0]
        values = distortion(system, flat_fields, wvl, epd=epd).percent
    elif key == 'lateral color':
        wvls = _resolve_wavelengths(system, wavelengths)
        if len(wvls) < 2:
            raise ValueError(
                "metric 'lateral color' needs at least two wavelengths"
            )
        landing = lateral_color(system, flat_fields, wvls, epd=epd)
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
