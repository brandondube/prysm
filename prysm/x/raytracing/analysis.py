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
    _intersect_lines,
    _valid_mask,
)
from .paraxial import paraxial_image_distance
from .launch import Field, Sampling, launch


# ---------- transverse ray aberration --------------------------------------

def transverse_ray_aberration(P_hist, axis='y', chief_index=None, status=None):
    """Per-ray image-plane offset from chief, vs pupil coordinate.

    Parameters
    ----------
    P_hist : ndarray
        position history from raytrace, shape (jj+1, N, 3).
    axis : str
        which axis to report: 'x' or 'y'.
    chief_index : int, optional
        row index of the chief ray; default N//2.
    status : ndarray, optional
        per-ray status from raytrace.  Invalid rays are excluded when
        provided.  If omitted, rays with non-finite image coordinates are
        excluded.

    Returns
    -------
    pupil : ndarray, shape (N,)
        launch (x or y) coordinate for each ray.
    delta : ndarray, shape (N,)
        image-plane (x or y) offset from chief for each ray.

    """
    P_hist = np.asarray(P_hist)
    if axis == 'x':
        ax = 0
    elif axis == 'y':
        ax = 1
    else:
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")
    if chief_index is None:
        chief_index = P_hist.shape[1] // 2
    pupil = P_hist[0, :, ax]
    image = P_hist[-1, :, ax]
    chief_image = P_hist[-1, chief_index, ax]
    if not np.isfinite(chief_image):
        raise ValueError('chief ray image coordinate is not finite')

    valid = _valid_mask(status, P_hist[-1])
    if valid is not None:
        pupil = pupil[valid]
        image = image[valid]
    return pupil, image - chief_image


# ---------- wavefront -------------------------------------------------------

def wavefront(prescription, P, S, wavelength, *,
              n_ambient=1.0, chief_index=None,
              axis_point=None, axis_dir=None):
    """Trace and compute OPD on the chief-ray-centered reference sphere.

    Composes spencer_and_murty.raytrace, opt.xp_reference_sphere, and
    opt.opd_from_raytrace.  For a centered system, axis defaults pin the
    optical axis to z through the origin.  For tilted / decentered
    systems, supply axis_point and axis_dir.

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

    Returns
    -------
    opd : ndarray, shape (N,)
        OPD relative to chief, on the reference sphere centered at the
        chief-ray image point.
    x_pupil, y_pupil : ndarray, shape (N,)
        launch (x, y) coordinates — the canonical pupil parameterization.

    """
    P = np.asarray(P)
    trace = raytrace(prescription, P, S, wavelength, n_ambient=n_ambient)
    if chief_index is None:
        chief_index = trace.P.shape[1] // 2
    valid = _valid_mask(trace.status, trace.P[-1])
    if not valid[chief_index]:
        raise ValueError('chief ray is invalid; cannot define reference sphere')

    P_chief_final = trace.P[-1, chief_index]
    S_chief_final = trace.S[-1, chief_index]
    _, _, P_xp = xp_reference_sphere(P_chief_final, S_chief_final,
                                     axis_point=axis_point,
                                     axis_dir=axis_dir)
    valid_indices = np.nonzero(valid)[0]
    filtered_chief = int(np.nonzero(valid_indices == chief_index)[0][0])
    opd = opd_from_raytrace(trace.P[:, valid], trace.S[:, valid],
                            trace.OPL[:, valid],
                            P_img=P_chief_final, P_xp=P_xp,
                            n_image=n_ambient,
                            chief_index=filtered_chief)
    return opd, P[valid, 0], P[valid, 1]


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

def distortion(prescription, fields, wavelength, *, epd, n_ambient=1.0,
               paraxial_fraction=1e-4):
    """Per-field image-plane error of the chief ray vs a paraxial proxy.

    For each field, traces the real chief ray and a paraxial proxy chief
    ray (same direction, scaled down by paraxial_fraction).  Distortion is
    the deviation of the real chief landing from the linear (scaled)
    paraxial landing.

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

    Returns
    -------
    real_xy : ndarray, shape (n_fields, 2)
        actual chief-ray image-plane (x, y) per field.
    paraxial_xy : ndarray, shape (n_fields, 2)
        scaled-up paraxial chief-ray landing per field.
    percent : ndarray, shape (n_fields,)
        100 * |real - paraxial| / |paraxial|.  Returns 0 for the on-axis
        field where paraxial landing is the origin.

    """
    fields = list(fields)
    n = len(fields)
    real_xy = np.zeros((n, 2), dtype=config.precision)
    paraxial_xy = np.zeros((n, 2), dtype=config.precision)
    percent = np.zeros(n, dtype=config.precision)
    chief = Sampling.chief()
    for i, field in enumerate(fields):
        P_r, S_r = launch(prescription, field, wavelength, chief,
                          epd=epd, n_ambient=n_ambient)
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
                          epd=epd, n_ambient=n_ambient)
        tr_p = raytrace(prescription, P_p, S_p, wavelength,
                        n_ambient=n_ambient)
        paraxial_xy[i] = tr_p.P[-1, 0, :2] / paraxial_fraction

        denom = float(np.hypot(*paraxial_xy[i]))
        if denom > 0.0:
            num = float(np.hypot(*(real_xy[i] - paraxial_xy[i])))
            percent[i] = 100.0 * num / denom

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


def field_curvature(prescription, fields, wavelength, *, epd,
                    n_ambient=1.0, marginal_fraction=0.7):
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
        0.7 (the classical lens-design "0.7-zone" sample).

    Returns
    -------
    sagittal_z : ndarray, shape (n_fields,)
        z position where the sagittal (x-fan) marginal converges with the
        chief, in lab frame.
    tangential_z : ndarray, shape (n_fields,)
        z position where the tangential (y-fan) marginal converges with
        the chief.

    """
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
        # sagittal marginal (one ray at +x_marg)
        P_sx = P_c.copy()
        P_sx[0, 0] = r_marg
        tr_sx = raytrace(prescription, P_sx, S_c, wavelength,
                         n_ambient=n_ambient)
        sagittal_z[i] = _line_intersection_z(Pc, Sc,
                                             tr_sx.P[-1, 0],
                                             tr_sx.S[-1, 0])
        # tangential marginal (one ray at +y_marg)
        P_ty = P_c.copy()
        P_ty[0, 1] = r_marg
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


def lateral_color(prescription, fields, wavelengths, *, epd, n_ambient=1.0):
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
