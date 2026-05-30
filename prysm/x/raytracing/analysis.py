"""Compute primitives for geometric ray-trace analysis.

Each function here takes a prescription (or a trace) and returns numbers,
not plots.  The plotting routines in plotting.py compose with these.
Operands in design.py also wrap these for use in optimization problems.

Provided primitives:

- transverse_ray_aberration: per-ray image-plane offset vs pupil coordinate
  (the math behind plot_transverse_ray_aberration, exposed for testing /
  scripting / merit functions)
- wavefront: trace and compute OPD on the chief-ray-centered reference
  sphere (composes opt.xp_reference_sphere + opt.opd_from_raytrace)
- wavefront_zernike_fit: project an OPD bundle onto a Zernike basis
- distortion: chief-ray image-plane error vs paraxial proxy, per field
- field_curvature: sagittal and tangential best-focus shifts per field
- axial_color: paraxial image distance at multiple wavelengths
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
from .paraxial import paraxial_image_distance
from .launch import Field, Sampling, launch
from ._meta import (
    lensdata_wavelength, lensdata_epd, object_space_index, image_space_index,
)


def _require_epd(prescription, epd):
    """Resolve epd from an explicit value or the LensData; error if neither."""
    epd = lensdata_epd(prescription, epd)
    if epd is None:
        raise TypeError(
            'epd is required; pass epd=... or supply a LensData carrying it.'
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


def wavefront(prescription, P, S, wavelength, *,
              n_ambient=None, chief_index=None,
              axis_point=None, axis_dir=None, P_xp=None,
              pupil_coords=None, field=None, output='length',
              method='sphere', reference='chief'):
    """Trace and compute OPD on the chief-ray-centered reference sphere.

    Composes spencer_and_murty.raytrace, opt.xp_reference_sphere, and
    opt.opd_from_raytrace.  For a centered system, axis defaults pin the
    optical axis to z through the origin.  For tilted / decentered
    systems, supply axis_point and axis_dir.  To use a lens-design reference
    sphere, pass P_xp as the paraxial exit-pupil center.

    Parameters
    ----------
    prescription : sequence of Surface
    P, S : ndarray, shape (N, 3)
        launch positions and direction cosines (typically from launch()).
    wavelength : float
        in microns.
    n_ambient : float
        ambient index of refraction.
    chief_index : int, optional
        row index of the chief ray.  Default N//2 (matches raygen
        convention).
    axis_point, axis_dir : iterable, optional
        point on, and direction of, the optical axis.  Defaults: origin
        and +z.
    P_xp : iterable, optional
        exit-pupil reference point.  If omitted, it is estimated from the
        chief ray and optical axis.  For lens-design OPD fans, use the
        paraxial exit-pupil center, e.g. (0, 0, xp_z).
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
    n_object = object_space_index(prescription, wavelength, n_ambient)
    trace = raytrace(prescription, P, S, wavelength, n_ambient=n_object)
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
               n_ambient=1.0, paraxial_fraction=1e-4, distortion_type='f-tan',
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
    n_ambient : float
        ambient index.
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
    wavelength = lensdata_wavelength(prescription, wavelength)
    epd = _require_epd(prescription, epd)
    fields = list(fields)
    n = len(fields)
    real_xy = np.zeros((n, 2), dtype=config.precision)
    paraxial_xy = np.zeros((n, 2), dtype=config.precision)
    percent = np.zeros(n, dtype=config.precision)
    chief = Sampling.chief()
    for i, field in enumerate(fields):
        P_r, S_r = launch(prescription, field, wavelength, chief,
                          epd=epd, n_ambient=n_ambient, pupil_z=pupil_z)
        tr_r = raytrace(prescription, P_r, S_r, wavelength,
                        n_ambient=n_ambient)
        real_xy[i] = tr_r.P[-1, 0, :2]

        ax, ay = field.angle_radians()
        small = Field(
            float(np.degrees(ax * paraxial_fraction)),
            float(np.degrees(ay * paraxial_fraction)),
            kind='angle', unit='deg',
        )
        P_p, S_p = launch(prescription, small, wavelength, chief,
                          epd=epd, n_ambient=n_ambient, pupil_z=pupil_z)
        tr_p = raytrace(prescription, P_p, S_p, wavelength,
                        n_ambient=n_ambient)
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
                    n_ambient=1.0, marginal_fraction=1e-3):
    """Sagittal and tangential focus shifts per field point.

    For each field, traces a 3-ray chief+marginal bundle in both x (sagittal)
    and y (tangential) and locates the longitudinal focus by intersecting
    each marginal ray with the chief.  Result is z, in lab-frame, where
    each fan converges; subtract prescription[-1].P[2] to get a shift
    relative to the last surface vertex.

    Parameters
    ----------
    prescription : sequence of Surface
    fields : iterable of Field
        field points to evaluate.  Must all be kind='angle'.
    wavelength : float
        in microns.
    epd : float
        entrance pupil diameter.
    n_ambient : float
        ambient index.
    marginal_fraction : float
        radius used for the marginal ray, as a fraction of EPD/2.  Default
        1e-3, i.e. a near-chief ray, which returns the differential
        (Coddington) sagittal and tangential foci -- the classical
        astigmatic field curves.  A finite zone (e.g. 0.7) instead reports
        where that real zonal fan focuses, which folds in coma and oblique
        spherical aberration and can differ from the differential foci by an
        order of magnitude at high aperture and field.

    Returns
    -------
    sagittal_z : ndarray, shape (n_fields,)
        z position where the sagittal (x-fan) marginal converges with the
        chief, in lab frame.
    tangential_z : ndarray, shape (n_fields,)
        z position where the tangential (y-fan) marginal converges with
        the chief.

    """
    wavelength = lensdata_wavelength(prescription, wavelength)
    epd = _require_epd(prescription, epd)
    fields = list(fields)
    n = len(fields)
    sagittal_z = np.zeros(n, dtype=config.precision)
    tangential_z = np.zeros(n, dtype=config.precision)
    chief = Sampling.chief()
    r_marg = float(marginal_fraction) * float(epd) / 2.0
    for i, field in enumerate(fields):
        # chief
        P_c, S_c = launch(prescription, field, wavelength, chief,
                          epd=epd, n_ambient=n_ambient)
        tr_c = raytrace(prescription, P_c, S_c, wavelength,
                        n_ambient=n_ambient)
        Pc = tr_c.P[-1, 0]
        Sc = tr_c.S[-1, 0]
        # sagittal marginal (one ray offset +x_marg from the chief in the
        # pupil; += keeps any chief offset from entrance-pupil routing)
        P_sx = P_c.copy()
        P_sx[0, 0] = P_sx[0, 0] + r_marg
        tr_sx = raytrace(prescription, P_sx, S_c, wavelength,
                         n_ambient=n_ambient)
        sagittal_z[i] = _line_intersection_z(Pc, Sc,
                                             tr_sx.P[-1, 0],
                                             tr_sx.S[-1, 0])
        # tangential marginal (one ray offset +y_marg from the chief)
        P_ty = P_c.copy()
        P_ty[0, 1] = P_ty[0, 1] + r_marg
        tr_ty = raytrace(prescription, P_ty, S_c, wavelength,
                         n_ambient=n_ambient)
        tangential_z[i] = _line_intersection_z(Pc, Sc,
                                               tr_ty.P[-1, 0],
                                               tr_ty.S[-1, 0])
    return sagittal_z, tangential_z


# ---------- color -----------------------------------------------------------

def axial_color(prescription, wavelengths, *, n_ambient=1.0):
    """Paraxial image distance at each of several wavelengths.

    Parameters
    ----------
    prescription : sequence of Surface
    wavelengths : iterable of float
        wavelengths in microns.
    n_ambient : float
        object-space index.

    Returns
    -------
    bfd : ndarray, shape (n_wavelengths,)
        signed paraxial image distance from the last surface vertex.

    """
    return np.array([
        paraxial_image_distance(prescription, wvl=float(w), n_ambient=n_ambient)
        for w in wavelengths
    ], dtype=config.precision)


def lateral_color(prescription, fields, wavelengths, *, epd=None, n_ambient=1.0):
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
    n_ambient : float
        ambient index.

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
            P, S = launch(prescription, field, float(w), chief,
                          epd=epd, n_ambient=n_ambient)
            tr = raytrace(prescription, P, S, float(w),
                          n_ambient=n_ambient)
            out[i, j] = tr.P[-1, 0, :2]
    return out
