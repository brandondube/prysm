"""Complex pupil field from a ray trace, for physical-optics propagation.

The geometric trace in spencer_and_murty is intensity-blind: it carries
positions, directions, and optical path length, but no amplitude.  This module
realizes the complex electric field A*exp(i*k*W) on the exit-pupil reference
sphere and resamples it onto a regular grid so it can be handed to
prysm.propagation.

The diffraction model is Hopkins 1981 (Optica Acta 28, no. 5, 667-714),
section 10: the scalar point-spread function is the 2D Fourier transform of the
pupil function f = a * exp(i*k*W) taken over the exit-pupil reference sphere,
and that transform is only valid when the pupil coordinates are the coordinates
of points on the reference sphere -- i.e. proportional to the ray direction
cosines (sine space), not the tan-space crossing of a flat exit-pupil plane.
The real amplitude a over the sphere is where energy conservation lives; it is
the square root of the local areal magnification of the ray bundle.

Amplitude is built from composable factors:

- amplitude_apodization: geometric energy conservation (areal Jacobian of the
  entrance-pupil to reference-sphere mapping).  Always meaningful; needs no
  coating model and no kernel change.  This is Hopkins' a(X', Y').
- a coating factor from interface_coefficients (the shared seam): the scalar,
  unpolarized default multiplies sqrt((T_s + T_p) / 2) per interface.
- the polarization ray-trace matrix (see raytrace_prt), which replaces the
  scalar coating factor with the full vector treatment.

This module sits above analysis in the layering and is the one raytracing
module besides plotting that reaches outside the package -- it imports
prysm.propagation to build the output wavefront.
"""

from prysm.conf import config
from prysm.mathops import np, row_dot, interpolate
from prysm.coordinates import make_xy_grid
from prysm.propagation import Wavefront, phase_prefix

from prysm import thinfilm

from . import spencer_and_murty as sm
from .spencer_and_murty import (
    STYPE_REFLECT, STYPE_REFRACT, raytrace,
)
from .launch import Sampling
from .paraxial import first_order
from .opt import (
    _pupil_center_chief_index,
    opd_from_raytrace_eic, xp_reference_sphere,
)
from .analysis import _apply_field_and_output, _filtered_chief_index
from ._trace_grid import trace_cell
from ._meta import (
    system_epd, system_wavelength, object_space_index, image_space_index,
)


def _complex_sqrt(x):
    """Square root with negative reals sent to the imaginary axis.

    Replaces np.lib.scimath.sqrt (a numpy-only construct) with a cast to the
    configured complex precision ahead of the root.  The behavior that matters
    is preserved: a beyond-critical-angle discriminant (1 - sin^2 < 0) becomes
    an imaginary cosine, which callers detect with np.imag(...) != 0 to flag
    total internal reflection.  Casting up front also honors config.precision
    (complex64 under a 32-bit precision) and stays portable across backends.
    """
    return np.sqrt(np.asarray(x, dtype=config.precision_complex))


class FieldTraceResult:
    """A geometric trace plus a per-ray scalar amplitude.

    The intensity-aware companion to RayTraceResult: it carries the same
    trace, plus amplitude, the real per-ray amplitude transmittance through the
    system (1 == lossless).  The geometric trace is left untouched, so the
    intensity-blind fast path is unaffected.
    """

    __slots__ = ('trace', 'amplitude')

    def __init__(self, trace, amplitude):
        self.trace = trace
        self.amplitude = amplitude

    # forward the common trace attributes so a FieldTraceResult can stand in
    # for a RayTraceResult in the analysis layer
    @property
    def P(self):
        return self.trace.P

    @property
    def S(self):
        return self.trace.S

    @property
    def OPL(self):
        return self.trace.OPL

    @property
    def status(self):
        return self.trace.status


def surface_normals_from_trace(prescription, trace, wavelength):
    """Recompute per-surface normals, incidence cosines, and indices.

    The kernel computes the surface normal and the incidence angle internally
    while bending each ray, but does not store them.  Here they are recovered
    from the recorded intersection points and directions by transforming back
    into each surface's local frame and re-evaluating sag_and_normal -- the
    exact path the kernel walked, so no change to the hot trace is needed.

    Parameters
    ----------
    prescription : sequence of Surface
        the traced system.
    trace : RayTraceResult
        result of spencer_and_murty.raytrace through prescription.
    wavelength : float
        wavelength in microns, used to evaluate surface indices.

    Returns
    -------
    cosI : ndarray, shape (jj, N)
        cosine of the angle of incidence at each surface for each ray, signed
        as the kernel's refract convention (normal dotted with the incident
        direction).  The angle of incidence is arccos(abs(cosI)).
    n0 : ndarray, shape (jj,)
        index of refraction preceding each surface.
    n1 : ndarray, shape (jj,)
        index of refraction following each surface.  For a reflecting surface
        n1 == n0 (the medium is unchanged); the sign convention of the kernel
        (signed index for mirrors) is not applied here.
    typ : ndarray, shape (jj,)
        surface interaction code (STYPE_REFLECT or STYPE_REFRACT) per surface.

    """
    P_hist = np.asarray(trace.P)
    S_hist = np.asarray(trace.S)
    surfaces = list(prescription)
    jj = len(surfaces)
    n_rays = P_hist.shape[1]
    cosI = np.empty((jj, n_rays), dtype=P_hist.dtype)
    n0 = np.empty(jj, dtype=config.precision)
    n1 = np.empty(jj, dtype=config.precision)
    typ = np.empty(jj, dtype=int)

    nj = object_space_index(prescription, wavelength)
    for j, surf in enumerate(surfaces):
        # incident direction on surface j is S_hist[j]; the intersection is
        # P_hist[j+1].  Transform both into the surface's local frame.
        XYZloc, Sloc = sm.transform_to_local_coords(
            P_hist[j + 1], surf.P, S_hist[j], surf.R)
        x = XYZloc[..., 0]
        y = XYZloc[..., 1]
        _, n_hat = surf.sag_and_normal(x, y)
        cosI[j] = row_dot(n_hat, Sloc)

        n0[j] = nj
        typ[j] = surf.typ
        if surf.typ == STYPE_REFRACT:
            nprime = float(surf.n(wavelength))
            n1[j] = nprime
            nj = nprime
        else:
            # reflection (or eval) leaves the propagation index unchanged
            n1[j] = nj
    return cosI, n0, n1, typ


def interface_coefficients(n0, n1, cosI, *, coating=None, wavelength=None):
    """Complex s- and p-amplitude coefficients at one interface.

    The shared seam consumed by both the scalar coating-aware path and the
    polarization ray trace.  With coating None the bare-interface Fresnel
    coefficients are returned; a coating spec routes through a thin-film stack.

    Parameters
    ----------
    n0 : float
        index preceding the interface.
    n1 : float
        index following the interface.  For a reflection at a bare mirror pass
        n1 == n0; the reflection coefficients are then +/-1 and the s/p
        transmission is zero.
    cosI : ndarray
        cosine of the angle of incidence (may be signed); abs is used.
    coating : object, optional
        thin-film stack specification.  Reserved for multilayer support via
        thinfilm.multilayer_stack_rt; None gives the bare Fresnel interface.
    wavelength : float, optional
        wavelength in microns; required for a multilayer coating.

    Returns
    -------
    r_s, r_p, t_s, t_p : ndarray
        complex amplitude reflection and transmission coefficients for s- and
        p-polarized light.  Shapes match cosI.

    """
    if coating is not None:
        raise NotImplementedError(
            'multilayer coatings are not wired up yet; pass coating=None for '
            'the bare Fresnel interface'
        )
    cosI = np.abs(np.asarray(cosI))
    theta0 = np.arccos(np.clip(cosI, 0.0, 1.0))
    # Snell, with a complex sqrt so total internal reflection produces an
    # imaginary cos(theta1); the resulting |r| == 1 and |t| handling fall out.
    sint1 = (n0 / n1) * np.sin(theta0)
    cost1 = _complex_sqrt(1.0 - sint1 * sint1)
    theta1 = np.arccos(cost1)
    r_s = thinfilm.fresnel_rs(n0, n1, theta0, theta1)
    r_p = thinfilm.fresnel_rp(n0, n1, theta0, theta1)
    t_s = thinfilm.fresnel_ts(n0, n1, theta0, theta1)
    t_p = thinfilm.fresnel_tp(n0, n1, theta0, theta1)
    return r_s, r_p, t_s, t_p


def _power_transmittance(n0, n1, cosI):
    """Unpolarized intensity transmittance (T_s + T_p) / 2 at an interface.

    Uses the energy-correct obliquity factor so the returned value is a true
    power transmittance, not |t|**2 alone.
    """
    cosI = np.abs(np.asarray(cosI))
    theta0 = np.arccos(np.clip(cosI, 0.0, 1.0))
    sint1 = (n0 / n1) * np.sin(theta0)
    cost1 = _complex_sqrt(1.0 - sint1 * sint1)
    theta1 = np.arccos(cost1)
    t_s = thinfilm.fresnel_ts(n0, n1, theta0, theta1)
    t_p = thinfilm.fresnel_tp(n0, n1, theta0, theta1)
    # power transmittance carries (n1 cos theta1) / (n0 cos theta0)
    oblique = np.real((n1 * cost1) / (n0 * np.cos(theta0)))
    T_s = oblique * np.abs(t_s) ** 2
    T_p = oblique * np.abs(t_p) ** 2
    # beyond the critical angle cost1 is imaginary -> no transmitted power
    tir = np.imag(cost1) != 0
    T = 0.5 * (T_s + T_p)
    return np.where(tir, 0.0, T)


def unpolarized_amplitude(prescription, trace, wavelength, *,
                          coatings=None):
    """Per-ray scalar amplitude transmittance through the system.

    Accumulates sqrt of the unpolarized power transmittance (T_s + T_p)/2 at
    each refracting interface; reflecting surfaces are treated as lossless
    (bare-mirror amplitude 1) unless a coating is supplied.  This is the
    amplitude-only, unpolarized coating factor -- the wavefront phase is left
    to the geometric optical path difference.

    Parameters
    ----------
    prescription : sequence of Surface
        the traced system.
    trace : RayTraceResult
        result of raytrace through prescription.
    wavelength : float
        wavelength in microns.
    coatings : sequence, optional
        per-surface coating specs; None (default) gives bare interfaces.

    Returns
    -------
    amplitude : ndarray, shape (N,)
        real, nonnegative per-ray amplitude factor (1 == no loss).

    """
    cosI, n0, n1, typ = surface_normals_from_trace(
        prescription, trace, wavelength)
    jj, n_rays = cosI.shape
    amp = np.ones(n_rays, dtype=config.precision)
    for j in range(jj):
        coating = None if coatings is None else coatings[j]
        if coating is not None:
            raise NotImplementedError(
                'multilayer coatings are not wired up yet')
        if typ[j] == STYPE_REFRACT:
            T = _power_transmittance(n0[j], n1[j], cosI[j])
            amp = amp * np.sqrt(np.clip(T, 0.0, None))
        # bare reflection: lossless, amplitude unchanged
    return amp


def raytrace_field(prescription, P, S, wavelength, *,
                   coatings=None):
    """Intensity-aware trace: geometry plus a scalar amplitude.

    Wraps spencer_and_murty.raytrace and accumulates the unpolarized,
    amplitude-only coating factor (the wavefront phase comes from the optical
    path difference, computed downstream).  The kernel is not modified; this is
    the separate, intensity-aware entry point.

    Parameters
    ----------
    prescription : sequence of Surface
        the system to trace.
    P, S : ndarray, shape (N, 3)
        launch positions and direction cosines.
    wavelength : float
        wavelength in microns.
    coatings : sequence, optional
        per-surface coating specs; None (default) gives bare Fresnel interfaces
        and lossless mirrors.

    Returns
    -------
    FieldTraceResult
        carrying the geometric trace and the per-ray amplitude.

    """
    trace = raytrace(prescription, P, S, wavelength)
    amplitude = unpolarized_amplitude(prescription, trace, wavelength,
                                      coatings=coatings)
    return FieldTraceResult(trace, amplitude)


def reference_sphere_coords(Q, P_xp, axis_dir=None):
    """Sine-space pupil coordinates of reference-sphere landing points.

    Projects each ray's reference-sphere intersection Q onto the plane through
    the exit-pupil center P_xp perpendicular to the optical axis, giving the
    transverse coordinates of points on the sphere.  These are Hopkins'
    (X', Y') and are proportional to the ray direction cosines (sine space) --
    the only parameterization for which the diffraction PSF is the plain
    Fourier transform of the pupil function.

    Parameters
    ----------
    Q : ndarray, shape (N, 3)
        reference-sphere intersection points, e.g. from
        spencer_and_murty.intersect_reference_sphere.
    P_xp : ndarray, shape (3,)
        exit-pupil center (a point on the optical axis).
    axis_dir : iterable, optional
        optical-axis direction; default +z.

    Returns
    -------
    X, Y : ndarray, shape (N,)
        transverse (sine-space) pupil coordinates, referenced to P_xp, in the
        same length units as the prescription.

    """
    Q = np.asarray(Q)
    P_xp = np.asarray(P_xp, dtype=Q.dtype)
    if axis_dir is None:
        w = np.array([0.0, 0.0, 1.0], dtype=Q.dtype)
    else:
        w = np.asarray(axis_dir, dtype=Q.dtype)
        w = w / np.sqrt(np.sum(w * w))
    # build an orthonormal in-plane basis (u, v) perpendicular to the axis w
    helper = np.array([1.0, 0.0, 0.0], dtype=Q.dtype)
    if abs(float(np.sum(helper * w))) > 0.9:
        helper = np.array([0.0, 1.0, 0.0], dtype=Q.dtype)
    u = helper - np.sum(helper * w) * w
    u = u / np.sqrt(np.sum(u * u))
    v = np.cross(w, u)
    d = Q - P_xp
    return d @ u, d @ v


def _inpaint_nan(arr):
    """Fill non-finite samples from finite neighbors (Jacobi/harmonic fill).

    amplitude_apodization differentiates the reference-sphere coordinates with
    np.gradient; a single missed or clipped ray (NaN) would otherwise have its
    central difference spread the NaN onto its valid grid neighbors and zero
    their amplitude.  A few neighbor-averaging passes give the holes a smooth
    finite fill so the surviving rim rays keep a sensible Jacobian.  The holes
    themselves are masked out of the returned amplitude regardless.
    """
    arr = np.asarray(arr, dtype=config.precision).copy()
    hole = ~np.isfinite(arr)
    if not np.any(hole):
        return arr
    arr = np.where(hole, 0.0, arr)
    for _ in range(int(max(arr.shape))):
        acc = np.zeros_like(arr)
        cnt = np.zeros_like(arr)
        acc[1:] += arr[:-1]
        cnt[1:] += 1.0
        acc[:-1] += arr[1:]
        cnt[:-1] += 1.0
        acc[:, 1:] += arr[:, :-1]
        cnt[:, 1:] += 1.0
        acc[:, :-1] += arr[:, 1:]
        cnt[:, :-1] += 1.0
        arr = np.where(hole, acc / cnt, arr)
    return arr


def amplitude_apodization(entrance_xy, sphere_xy, *, valid=None):
    """Energy-conservation amplitude over the reference sphere.

    Computes sqrt(dA_entrance / dA_sphere), the square root of the local areal
    magnification of the ray bundle as it maps from the (uniform) entrance
    pupil to its transverse footprint on the reference sphere.  This is the
    real amplitude a(X', Y') of Hopkins' pupil function for a uniformly
    illuminated input: rays that crowd together on the sphere carry higher
    irradiance.

    Both inputs are structured grids so the areal Jacobian is a clean finite
    difference; flatten afterwards for the scattered-to-regular resample.

    Parameters
    ----------
    entrance_xy : ndarray, shape (M, M, 2)
        entrance-pupil sample coordinates on a structured grid.
    sphere_xy : ndarray, shape (M, M, 2)
        the corresponding reference-sphere (sine-space) coordinates from
        reference_sphere_coords, reshaped to the same grid.
    valid : ndarray, shape (M, M), optional
        boolean mask of surviving rays; masked entries return amplitude 0.

    Returns
    -------
    amplitude : ndarray, shape (M, M)
        nonnegative per-ray amplitude factor, unnormalized (relative).

    """
    entrance_xy = np.asarray(entrance_xy)
    sphere_xy = np.asarray(sphere_xy)
    a = entrance_xy[..., 0]
    b = entrance_xy[..., 1]
    X = sphere_xy[..., 0]
    Y = sphere_xy[..., 1]
    # a missed/clipped ray lands NaN on the reference sphere; np.gradient's
    # central difference would otherwise spread that NaN onto its valid
    # neighbors and zero their amplitude.  Fill the holes from finite neighbors
    # so the surviving rim keeps a sensible Jacobian (holes are masked below).
    X = _inpaint_nan(X)
    Y = _inpaint_nan(Y)
    # 1D entrance axes (the rect grid is separable in a, b)
    a_axis = a[0, :]
    b_axis = b[:, 0]
    dX_da = np.gradient(X, a_axis, axis=1)
    dX_db = np.gradient(X, b_axis, axis=0)
    dY_da = np.gradient(Y, a_axis, axis=1)
    dY_db = np.gradient(Y, b_axis, axis=0)
    detJ = dX_da * dY_db - dX_db * dY_da   # dA_sphere / dA_entrance
    mag = np.abs(detJ)
    with np.errstate(divide='ignore', invalid='ignore'):
        amp = 1.0 / np.sqrt(mag)
    amp = np.where(np.isfinite(amp), amp, 0.0)
    if valid is not None:
        amp = np.where(valid, amp, 0.0)
    return amp


# ---------- orchestration: pupil field + propagation bridge ----------------


class PupilField:
    """Complex pupil field samples on the exit-pupil reference sphere.

    Scattered, sine-space samples ready to be resampled onto a regular grid
    (pupil_field_to_wavefront).  X, Y are Hopkins' (X', Y') -- transverse
    coordinates of the ray landings on the reference sphere, proportional to
    the ray direction cosines.  amplitude is the real pupil amplitude
    a(X', Y') (geometric apodization times any coating factor); opd is the
    wavefront error referred to the chief, in prescription length units.
    """

    __slots__ = ('X', 'Y', 'amplitude', 'opd', 'wavelength', 'efl',
                 'n_image', 'P_xp', 'P_img', 'P_matrix')

    def __init__(self, X, Y, amplitude, opd, wavelength, efl, n_image,
                 P_xp, P_img, P_matrix=None):
        self.X = X
        self.Y = Y
        # for a scalar field, amplitude is the geometric apodization times the
        # unpolarized coating factor.  For a polarized field, amplitude is the
        # geometric apodization only and P_matrix carries the (vector) coating.
        self.amplitude = amplitude
        self.opd = opd
        self.wavelength = wavelength
        self.efl = efl
        self.n_image = n_image
        self.P_xp = P_xp
        self.P_img = P_img
        # per-ray (N, 3, 3) polarization ray-trace matrix, or None (scalar)
        self.P_matrix = P_matrix

    @property
    def polarized(self):
        return self.P_matrix is not None


def _first_order_geometry_failure(exc):
    """True when first_order failed because scalar ABCD geometry is invalid."""
    msg = str(exc)
    return ('centered axial geometry' in msg
            or 'vertex normal to be axial' in msg)


def _resolve_exit_pupil(prescription, field, wavelength, epd,
                        stop_index, P_xp, P_img, chief_P, chief_S,
                        axis_dir=None):
    """Resolve the reference-sphere center (P_img) and exit-pupil point (P_xp).

    P_img defaults to the real chief-ray image landing.  P_xp is taken from an
    explicit argument, else the paraxial exit pupil (needs stop_index), else
    the chief-ray closest approach to the axis (off-axis fields only).
    """
    if P_img is None:
        P_img = np.asarray(chief_P)
    else:
        P_img = np.asarray(P_img)

    if P_xp is not None:
        return np.asarray(P_xp), P_img

    if stop_index is not None:
        try:
            fo = first_order(prescription, wvl=wavelength,
                             epd=epd, stop_index=stop_index)
        except ValueError as exc:
            if axis_dir is None or not _first_order_geometry_failure(exc):
                raise
        else:
            if fo.xp_z is None:
                raise ValueError(
                    'paraxial exit pupil is at infinity (telecentric); pass P_xp '
                    'explicitly for a planar reference'
                )
            P_xp = np.array([0.0, 0.0, float(fo.xp_z)], dtype=P_img.dtype)
            return P_xp, P_img

    # last resort: chief-ray geometric estimate.  This triangulates the exit
    # pupil from where the chief ray crosses the axis, which is ill-conditioned
    # when the chief is (nearly) parallel to and on the axis -- the on-axis
    # field, and also an even npupil grid whose center sample is half a step
    # off axis.  Guard on the chief's perpendicular slope to the axis.
    chief_S = np.asarray(chief_S)
    if axis_dir is None:
        w = np.array([0.0, 0.0, 1.0], dtype=chief_S.dtype)
    else:
        w = np.asarray(axis_dir, dtype=chief_S.dtype)
        w = w / np.sqrt(np.sum(w * w))
    s_perp = chief_S - np.sum(chief_S * w) * w
    if float(np.sqrt(np.sum(s_perp * s_perp))) < 1e-3:
        raise ValueError(
            'cannot locate the exit pupil from a near-axial chief ray; pass '
            'stop_index=... or P_xp=... to anchor the reference sphere'
        )
    _, R, P_xp = xp_reference_sphere(chief_P, chief_S, axis_dir=w)
    return P_xp, P_img


def pupil_field(prescription, field, wavelength=None, *, epd=None, npupil=64,
                stop_index=None, P_xp=None, P_img=None, axis_dir=None,
                pupil_z=None, coatings=None,
                output='length', reference='chief', polarized=False):
    """Realize the complex pupil field on the exit-pupil reference sphere.

    Traces a structured (rect) entrance-pupil grid, computes the wavefront
    error and the physical amplitude (geometric energy apodization times the
    unpolarized coating factor) at each ray's reference-sphere landing, and
    returns the scattered sine-space samples.  The result is the input to
    pupil_field_to_wavefront for physical-optics propagation.

    Parameters
    ----------
    prescription : sequence of Surface or LensData
    field : Field
        the field point to evaluate.
    wavelength : float, optional
        wavelength in microns; defaults from LensData metadata if available.
    epd : float, optional
        entrance pupil diameter; defaults from LensData metadata.
    npupil : int
        entrance-pupil grid is npupil by npupil.
    stop_index : int, optional
        aperture-stop surface index, used to locate the paraxial exit pupil
        when P_xp is not given.
    P_xp : iterable, optional
        exit-pupil center.  Overrides the paraxial estimate; required for an
        on-axis field unless stop_index is supplied.
    P_img : iterable, optional
        reference-sphere center; defaults to the real chief-ray image point.
    axis_dir : iterable, optional
        optical-axis direction; default +z.
    pupil_z : float, optional
        launch-plane z; passed to launch().
    coatings : sequence, optional
        per-surface coating specs; None gives bare interfaces.
    output : str
        'length' (default) or 'waves' for the returned opd units.
    reference : str
        'chief' (default) anchors the reference sphere on the pupil-center
        chief ray and raises if it is obscured; 'centroid' falls back to the
        surviving ray nearest the pupil center for an obscured or vignetted
        bundle (e.g. a Cassegrain), mirroring analysis.wavefront.

    Returns
    -------
    PupilField
        scattered sine-space samples (valid rays only) plus the efl (R'),
        image-space index, and pupil/image anchors.

    """
    wavelength = system_wavelength(prescription, wavelength)
    epd = system_epd(prescription, epd, wavelength)
    if epd is None:
        raise TypeError(
            'epd is required; pass epd=... or an OpticalSystem whose aperture '
            'spec resolves it.')
    if reference not in ('chief', 'centroid'):
        raise ValueError(
            f"reference must be 'chief' or 'centroid', got {reference!r}")
    n_object = object_space_index(prescription, wavelength)

    sampling = Sampling.rect(n=npupil)

    # launch + trace + valid-mask through the shared per-cell tracer; the
    # intensity-aware trace_fn carries the scalar coating amplitude or the
    # polarization ray-trace matrix, both forwarding the geometric trace.
    def _trace_fn(presc, P, S, w):
        if polarized:
            return raytrace_prt(presc, P, S, w, coatings=coatings)
        return raytrace_field(presc, P, S, w, coatings=coatings)

    record = trace_cell(prescription, field, wavelength, sampling,
                        epd=epd, pupil_z=pupil_z, trace_fn=_trace_fn)
    valid = record.valid
    result = record.trace
    if polarized:
        trace = result.trace
        coating_amp = None
        P_matrix_all = result.P_matrix
    else:
        trace = result.trace
        coating_amp = result.amplitude
        P_matrix_all = None

    # the uniform entrance-pupil sample grid -- the pupil coordinate of every
    # ray.  Use this, not the launch positions P (which collapse onto the
    # object point for a finite-conjugate / object-height field), for the
    # apodization area, the circular mask, the chief, and the field tilt.
    pupil_xy = sampling.build(0.5 * epd)

    # chief ray anchoring the reference sphere, the OPD, and the circular pupil.
    # reference='chief' (default) takes the pupil-center ray and raises if it
    # is obscured; reference='centroid' falls back to the surviving ray nearest
    # the pupil center, so an obscured/vignetted bundle (e.g. a Cassegrain)
    # still resolves -- mirroring analysis.wavefront.  Resolved on the trace
    # validity (before the circular mask it centers) so the chief is never
    # excluded by its own aperture.
    mask = valid if reference == 'centroid' else None
    chief_index = _pupil_center_chief_index(pupil_xy, mask)
    if not bool(valid[chief_index]):
        if reference == 'chief':
            raise ValueError(
                'chief ray is obscured or vignetted; cannot anchor the '
                "reference sphere.  Pass reference='centroid' for an "
                'obscured or vignetted bundle.'
            )
        if not bool(np.any(valid)):
            raise ValueError('no valid rays to anchor the reference sphere')
        raise ValueError(
            f'anchor ray (chief_index={chief_index}) is invalid; pass a '
            'chief_index that survives the trace, or omit it to auto-select '
            'the surviving ray nearest the pupil center'
        )

    # the rect sampling fills a square; the entrance pupil of diameter epd is
    # the inscribed circle.  Referenced to the chief so it tracks the bundle
    # (matching analysis.wavefront's pupil coordinate).  Mask the corner rays
    # so the pupil -- and hence the diffraction PSF -- is circular.
    r_entrance = np.hypot(pupil_xy[:, 0] - pupil_xy[chief_index, 0],
                          pupil_xy[:, 1] - pupil_xy[chief_index, 1])
    circ = r_entrance <= (0.5 * epd) * (1.0 + 1e-9)
    valid = valid & circ

    P_xp, P_img = _resolve_exit_pupil(
        prescription, field, wavelength, epd, stop_index,
        P_xp, P_img, trace.P[-1, chief_index], trace.S[-1, chief_index],
        axis_dir=axis_dir)
    R = float(np.sqrt(np.sum((np.asarray(P_xp) - P_img) ** 2)))

    # OPD on the reference sphere (valid rays), mirroring analysis.wavefront
    filtered_chief = _filtered_chief_index(valid, chief_index)
    n_image = image_space_index(prescription, wavelength, fallback=n_object)
    opd = opd_from_raytrace_eic(trace.P[:, valid], trace.S[:, valid],
                                trace.OPL[:, valid],
                                P_img=P_img, P_xp=P_xp, n_image=n_image,
                                chief_index=filtered_chief)

    # sine-space sphere coordinates (all rays, then mask)
    Q, _ = sm.intersect_reference_sphere(trace.P[-1], trace.S[-1], P_img, R)
    X_all, Y_all = reference_sphere_coords(Q, P_xp, axis_dir)

    # amplitude: geometric apodization (structured grid) times coating factor
    entrance_xy = np.ascontiguousarray(pupil_xy).reshape(npupil, npupil, 2)
    sphere_xy = np.stack([X_all, Y_all], axis=-1).reshape(npupil, npupil, 2)
    valid_grid = valid.reshape(npupil, npupil)
    amp_geo = amplitude_apodization(entrance_xy, sphere_xy,
                                    valid=valid_grid).reshape(-1)
    # scalar field folds the unpolarized coating loss into the amplitude;
    # the polarized field keeps amplitude geometric and lets P_matrix carry it
    if coating_amp is None:
        amplitude_all = amp_geo
    else:
        amplitude_all = amp_geo * coating_amp

    # field-tilt removal + output scaling on the OPD, using the entrance-pupil
    # coordinate referred to the chief (matches analysis.wavefront's
    # convention).  The launch-plane tilt is an angular-field concept; a
    # finite-conjugate (height) field has no such tilt to remove.
    x_pupil = pupil_xy[valid, 0] - pupil_xy[chief_index, 0]
    y_pupil = pupil_xy[valid, 1] - pupil_xy[chief_index, 1]
    tilt_field = field if field.kind == 'angle' else None
    opd, _ = _apply_field_and_output(opd, x_pupil, y_pupil, tilt_field, output,
                                     wavelength)

    n_image = abs(float(n_image))
    P_matrix = None if P_matrix_all is None else P_matrix_all[valid]
    return PupilField(
        X=X_all[valid], Y=Y_all[valid], amplitude=amplitude_all[valid],
        opd=opd, wavelength=wavelength, efl=R / n_image, n_image=n_image,
        P_xp=np.asarray(P_xp), P_img=P_img, P_matrix=P_matrix)


def _resample_grid(pf, npix, margin):
    """Shared scatter-to-regular-grid setup for the wavefront bridge.

    Returns the regular-grid scattered-data interpolation points, the grid
    sample spacing, and the resampled wavefront phase (nm), zeroed outside the
    traced support.
    """
    x = np.asarray(pf.X)
    y = np.asarray(pf.Y)
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(pf.opd)
    x = x[finite]
    y = y[finite]
    opd = np.asarray(pf.opd)[finite]
    r = float(np.max(np.hypot(x, y)))
    diameter = 2.0 * r * float(margin)
    xg, yg = make_xy_grid(npix, diameter=diameter)
    dx = diameter / npix
    pts = np.stack([x, y], axis=-1)
    opd_grid = interpolate.griddata(pts, opd, (xg, yg), method='cubic',
                                    fill_value=0.0)
    opd_grid = np.where(np.isfinite(opd_grid), opd_grid, 0.0)
    phase_nm = opd_grid * 1.0e6   # length (mm) -> nm
    return finite, pts, (xg, yg), dx, phase_nm


def _griddata_complex(pts, values, grid_pts):
    """Cubic griddata for complex values (real and imaginary parts), zero-fill.

    Suitable only for slowly varying complex fields (e.g. geometric amplitude
    times the polarization Jones weight) -- never wrapped phase, which is
    carried separately as the wavefront term.
    """
    re = interpolate.griddata(pts, np.real(values), grid_pts, method='cubic',
                              fill_value=0.0)
    im = interpolate.griddata(pts, np.imag(values), grid_pts, method='cubic',
                              fill_value=0.0)
    g = np.where(np.isfinite(re), re, 0.0) + 1j * np.where(np.isfinite(im),
                                                           im, 0.0)
    return g


def pupil_field_to_wavefront(pf, *, npix=256, margin=1.05,
                             input_polarization=None):
    """Resample scattered pupil-field samples onto a regular grid wavefront.

    Cubic-interpolates the smooth amplitude and OPD fields (real, low-order --
    never wrapped phase) from the scattered sine-space samples onto a regular
    cartesian grid, zeroing outside the traced support, and builds a
    prysm.propagation.Wavefront ready for .focus(efl=pf.efl).

    For a polarized PupilField, the transverse field components (Ex, Ey) of
    P_matrix @ input_polarization are each resampled and returned as a list of
    component Wavefronts; the longitudinal Ez is neglected (the one
    approximation -- valid except at extreme numerical aperture).

    Parameters
    ----------
    pf : PupilField
        scattered samples from pupil_field (opd must be in length units).
    npix : int
        output grid is npix by npix.
    margin : float
        the grid half-width is margin times the largest sample radius.
    input_polarization : iterable, optional
        incident Jones/field vector (global x, y[, z]); required for a
        polarized PupilField, ignored for a scalar one.

    Returns
    -------
    Wavefront or list of Wavefront
        a single pupil-space wavefront for a scalar field; the [Ex, Ey]
        component wavefronts for a polarized field.  Propagate to the PSF with
        .focus(efl=pf.efl).

    """
    finite, pts, grid_pts, dx, phase_nm = _resample_grid(pf, npix, margin)
    phase_term = np.exp(phase_prefix(pf.wavelength) * phase_nm)

    if not pf.polarized:
        amp = np.asarray(pf.amplitude)[finite]
        amp_grid = interpolate.griddata(pts, amp, grid_pts, method='cubic',
                                        fill_value=0.0)
        amp_grid = np.where(np.isfinite(amp_grid), amp_grid, 0.0)
        return Wavefront(amp_grid * phase_term, pf.wavelength, dx)

    if input_polarization is None:
        raise TypeError(
            'input_polarization is required for a polarized PupilField')
    e_in = np.zeros(3, dtype=config.precision_complex)
    e_in[:len(input_polarization)] = np.asarray(input_polarization,
                                                dtype=config.precision_complex)
    amp = np.asarray(pf.amplitude)[finite]
    # E_vec = P @ E_in (N, 3); fold the geometric amplitude in, then resample
    e_vec = np.einsum('nij,j->ni', pf.P_matrix[finite], e_in)
    wavefronts = []
    for c in (0, 1):     # transverse x, y components; Ez neglected
        g = _griddata_complex(pts, amp * e_vec[:, c], grid_pts)
        wavefronts.append(Wavefront(g * phase_term, pf.wavelength, dx))
    return wavefronts


def pupil_field_psf(pf, *, npix=256, margin=1.05, Q=2,
                    input_polarization='unpolarized'):
    """Intensity point-spread function from a pupil field.

    Resamples, focuses each scalar/vector component, and incoherently combines.
    A polarized pupil field is illuminated with the requested input state;
    'unpolarized' incoherently averages two orthogonal input polarizations.

    Parameters
    ----------
    pf : PupilField
        from pupil_field.
    npix : int
        pupil grid size.
    margin : float
        pupil grid half-width factor.
    Q : float
        focusing pad factor.
    input_polarization : str or iterable
        'unpolarized' (default) or an incident Jones/field vector.

    Returns
    -------
    psf : ndarray
        intensity PSF (not normalized).
    dx : float
        PSF sample spacing in microns.

    """
    if not pf.polarized:
        wf = pupil_field_to_wavefront(pf, npix=npix, margin=margin)
        psf = wf.focus(pf.efl, Q=Q)
        return np.abs(psf.data) ** 2, psf.dx

    if isinstance(input_polarization, str):
        if input_polarization != 'unpolarized':
            raise ValueError("string input_polarization must be 'unpolarized'")
        inputs = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
        weight = 0.5
    else:
        inputs = [input_polarization]
        weight = 1.0

    total = None
    last_dx = None
    for e_in in inputs:
        comps = pupil_field_to_wavefront(pf, npix=npix, margin=margin,
                                         input_polarization=e_in)
        for wf in comps:
            psf = wf.focus(pf.efl, Q=Q)
            last_dx = psf.dx
            contribution = weight * np.abs(psf.data) ** 2
            total = contribution if total is None else total + contribution
    return total, last_dx


# ---------- Phase 3: polarization ray tracing (PRT) ------------------------


class PRTResult:
    """A geometric trace plus a per-ray 3x3 polarization ray-trace matrix.

    P_matrix maps an incident 3D electric field vector to the exitant 3D field
    after the whole system (Yun/Chipman polarization ray tracing).  It composes
    the per-interface Jones matrices in their local s-p-k bases, carried in the
    global frame, so the geometric rotation of the polarization basis between
    surfaces is included.  The transmission coefficients are energy-normalized,
    so the unpolarized intensity (mean over two orthogonal inputs) reduces to
    the scalar (T_s + T_p) / 2 of the coating-aware path.
    """

    __slots__ = ('trace', 'P_matrix')

    def __init__(self, trace, P_matrix):
        self.trace = trace
        self.P_matrix = P_matrix

    @property
    def P(self):
        return self.trace.P

    @property
    def S(self):
        return self.trace.S

    @property
    def OPL(self):
        return self.trace.OPL

    @property
    def status(self):
        return self.trace.status


def _global_normal_and_cosI(surf, P_int_global, S_in_global):
    """Global-frame surface normal and incidence cosine at an intersection."""
    XYZloc, Sloc = sm.transform_to_local_coords(
        P_int_global, surf.P, S_in_global, surf.R)
    _, n_local = surf.sag_and_normal(XYZloc[..., 0], XYZloc[..., 1])
    if surf.R is None:
        n_global = n_local
    else:
        n_global = np.matmul(surf.R.T, n_local[..., np.newaxis]).squeeze(-1)
    return n_global, row_dot(n_local, Sloc)


def _unit(v):
    n = np.sqrt(np.sum(v * v, axis=-1, keepdims=True))
    return v / n


def _interface_jones(n0, n1, cosI, typ, *, coating=None):
    """Energy-normalized s/p amplitude pair (a_s, a_p) for one interface.

    For refraction the transmission coefficients carry the sqrt of the index/
    obliquity factor, so |a|**2 is the power transmittance.  A bare reflection
    is modelled as a lossless ideal mirror with a_s = 1, a_p = -1.  The opposite
    sign is required by the s-p-k basis convention (p_out = k_out x s flips
    relative to p_in on reflection): it makes s(x)s + p(x)p collapse to the
    transverse projector, so the reflection is the azimuth-independent
    diag(1, 1, -1) and introduces no diattenuation or retardance.  A real metal
    or coated mirror would supply complex s/p coefficients via coating.
    """
    if coating is not None:
        raise NotImplementedError('multilayer coatings are not wired up yet')
    cosI = np.abs(np.asarray(cosI))
    theta0 = np.arccos(np.clip(cosI, 0.0, 1.0))
    if typ == STYPE_REFRACT:
        sint1 = (n0 / n1) * np.sin(theta0)
        cost1 = _complex_sqrt(1.0 - sint1 * sint1)
        theta1 = np.arccos(cost1)
        t_s = thinfilm.fresnel_ts(n0, n1, theta0, theta1)
        t_p = thinfilm.fresnel_tp(n0, n1, theta0, theta1)
        oblique = _complex_sqrt((n1 * cost1) / (n0 * np.cos(theta0)))
        a_s = t_s * oblique
        a_p = t_p * oblique
        # beyond the critical angle no power is transmitted
        tir = np.imag(cost1) != 0
        a_s = np.where(tir, 0.0, a_s)
        a_p = np.where(tir, 0.0, a_p)
        return (a_s.astype(config.precision_complex),
                a_p.astype(config.precision_complex))
    ones = np.ones_like(cosI, dtype=config.precision_complex)
    if typ == STYPE_REFLECT:
        # bare reflection: lossless ideal mirror, p flips sign in s-p-k basis
        return ones, -ones
    # eval / non-interacting surface: identity (the ray passes straight
    # through, k_out == k_in, so the per-surface matrix is the identity)
    return ones, ones


def raytrace_prt(prescription, P, S, wavelength, *,
                 coatings=None):
    """Polarization ray trace: geometry plus a per-ray 3x3 P matrix.

    Wraps spencer_and_murty.raytrace and accumulates the Yun/Chipman
    polarization ray-tracing matrix for each ray.  The hot kernel is not
    modified.

    Parameters
    ----------
    prescription : sequence of Surface
    P, S : ndarray, shape (N, 3)
        launch positions and direction cosines.
    wavelength : float
        wavelength in microns.
    coatings : sequence, optional
        per-surface coating specs; None gives bare interfaces.

    Returns
    -------
    PRTResult
        the geometric trace and the per-ray (N, 3, 3) complex P matrix.

    """
    trace = raytrace(prescription, P, S, wavelength)
    surfaces = list(prescription)
    P_hist = trace.P
    S_hist = trace.S
    n_rays = P_hist.shape[1]
    Pmat = np.broadcast_to(np.eye(3, dtype=config.precision_complex),
                           (n_rays, 3, 3)).copy()

    nj = object_space_index(prescription, wavelength)
    for j, surf in enumerate(surfaces):
        coating = None if coatings is None else coatings[j]
        k_in = _unit(S_hist[j])
        k_out = _unit(S_hist[j + 1])
        n_g, cosI = _global_normal_and_cosI(surf, P_hist[j + 1], S_hist[j])

        s = np.cross(k_in, n_g)
        s_norm = np.sqrt(np.sum(s * s, axis=-1, keepdims=True))
        # normal incidence: plane of incidence is undefined; any perpendicular
        # to k_in works because a_s == a_p there.
        degen = (s_norm[..., 0] < 1e-12)
        fallback = np.cross(k_in, np.array([1.0, 0.0, 0.0]))
        fb_norm = np.sqrt(np.sum(fallback * fallback, axis=-1, keepdims=True))
        small = fb_norm[..., 0] < 1e-12
        if np.any(small):
            fallback = np.where(small[:, None],
                                np.cross(k_in, np.array([0.0, 1.0, 0.0])),
                                fallback)
            fb_norm = np.sqrt(np.sum(fallback * fallback, axis=-1,
                                     keepdims=True))
        s_norm_safe = np.where(s_norm > 0, s_norm, 1.0)
        s = np.where(degen[:, None], fallback / fb_norm, s / s_norm_safe)
        p_in = np.cross(k_in, s)
        p_out = np.cross(k_out, s)

        if surf.typ == STYPE_REFRACT:
            n1 = float(surf.n(wavelength))
        else:
            n1 = nj
        a_s, a_p = _interface_jones(nj, n1, cosI, surf.typ, coating=coating)
        if surf.typ == STYPE_REFRACT:
            nj = n1

        O_in = np.stack([s, p_in, k_in], axis=-1)         # (N,3,3) columns
        O_out = np.stack([s, p_out, k_out], axis=-1)
        J = np.zeros((n_rays, 3, 3), dtype=config.precision_complex)
        J[:, 0, 0] = a_s
        J[:, 1, 1] = a_p
        J[:, 2, 2] = 1.0
        Pj = O_out @ J @ np.swapaxes(
            O_in, -1, -2).astype(config.precision_complex)
        Pmat = Pj @ Pmat
    return PRTResult(trace, Pmat)
