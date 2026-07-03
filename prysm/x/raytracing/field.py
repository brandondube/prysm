"""Complex pupil fields from ray traces."""

from prysm.conf import config
from prysm.mathops import np, row_dot, interpolate
from prysm.coordinates import make_xy_grid
from prysm.propagation import Wavefront, phase_prefix

from prysm import thinfilm
from prysm.x.coatings.stack import Stack, stack_rt

from . import spencer_and_murty as sm
from .spencer_and_murty import (
    STYPE_REFLECT, STYPE_REFRACT, raytrace,
)
from .launch import Sampling, _apply_vignetting
from .paraxial import effective_focal_length
from .opt import (
    _pupil_center_chief_index,
)
from .analysis import _apply_field_and_output, close_wavefront
from ._resolve import resolve_wavelength, compiled_surfaces
from ._trace_grid import trace_cell
from ._meta import object_space_index


def _complex_sqrt(x):
    """Square root in configured complex precision."""
    return np.sqrt(np.asarray(x, dtype=config.precision_complex))


class FieldTraceResult:
    """A geometric trace plus per-ray scalar amplitude."""

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


def surface_normals_from_trace(system, trace, wavelength):
    """Recompute per-surface normals, incidence cosines, and indices.

    The kernel computes the surface normal and the incidence angle internally
    while bending each ray, but does not store them.  Here they are recovered
    from the recorded intersection points and directions by transforming back
    into each surface's local frame and re-evaluating sag_and_normal -- the
    exact path the kernel walked, so no change to the hot trace is needed.

    Parameters
    ----------
    system : sequence of Surface
        the traced system.
    trace : RayTraceResult
        result of spencer_and_murty.raytrace through system.
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
    surfaces = list(system)
    jj = len(surfaces)
    n_rays = P_hist.shape[1]
    cosI = np.empty((jj, n_rays), dtype=P_hist.dtype)
    n0 = np.empty(jj, dtype=config.precision)
    n1 = np.empty(jj, dtype=config.precision)
    typ = np.empty(jj, dtype=int)

    nj = object_space_index(surfaces, wavelength)
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
            nprime = float(surf.material.n(wavelength))
            n1[j] = nprime
            nj = nprime
        else:
            # reflection (or eval) leaves the propagation index unchanged
            n1[j] = nj
    return cosI, n0, n1, typ


def _transmission_energy_norm(n0, n1, theta0, pol):
    """Obliquity factor from field transmission to sqrt(power)."""
    cost0 = np.cos(theta0)
    cost1 = _complex_sqrt(1.0 - ((n0 / n1) * np.sin(theta0)) ** 2)
    # The caller zeroes grazing and critical-angle rays.
    with np.errstate(divide='ignore', invalid='ignore'):
        if pol == 's':
            ratio = (n1 * cost1) / (n0 * cost0)
        else:
            ratio = (n1 * cost0) / (n0 * cost1)
    return _complex_sqrt(np.real(ratio))


def _coating_coefficients(coating, n0, n1, cosI, theta0, typ, wavelength):
    """Thin-film stack s/p amplitudes for one traced interface."""
    if wavelength is None:
        raise TypeError('a coated surface requires a wavelength')
    if typ == STYPE_REFRACT:
        stack = Stack(coating.indices, coating.thicknesses,
                      substrate_index=n1, ambient_index=n0)
        _, t_s = stack_rt(stack, wavelength, theta0, 's')
        _, t_p = stack_rt(stack, wavelength, theta0, 'p')
        a_s = (t_s * _transmission_energy_norm(n0, n1, theta0, 's')).astype(
            config.precision_complex)
        a_p = (t_p * _transmission_energy_norm(n0, n1, theta0, 'p')).astype(
            config.precision_complex)
        # TIR and grazing incidence transmit no power.
        cost1 = _complex_sqrt(1.0 - ((n0 / n1) * np.sin(theta0)) ** 2)
        dead = (np.imag(cost1) != 0) | ~np.isfinite(a_s) | ~np.isfinite(a_p)
        a_s[dead] = 0.0
        a_p[dead] = 0.0
        return a_s, a_p
    if typ == STYPE_REFLECT:
        stack = Stack(coating.indices, coating.thicknesses,
                      substrate_index=coating.substrate_index, ambient_index=n0)
        r_s, _ = stack_rt(stack, wavelength, theta0, 's')
        r_p, _ = stack_rt(stack, wavelength, theta0, 'p')
        # s-p-k basis signs match the bare ideal mirror limit: (1, -1).
        return ((-r_s).astype(config.precision_complex),
                r_p.astype(config.precision_complex))
    ones = np.ones_like(cosI, dtype=config.precision_complex)
    return ones, ones


def interface_coefficients(n0, n1, cosI, typ, *, coating=None, wavelength=None):
    """Energy-normalized s/p amplitude coefficients (a_s, a_p) for one interface.

    Parameters
    ----------
    n0 : float
        index preceding the interface (the incidence medium).
    n1 : float
        index following the interface.  Equal to n0 at a reflection.
    cosI : ndarray
        cosine of the angle of incidence (may be signed); its abs is used.
    typ : int
        surface interaction code (STYPE_REFRACT, STYPE_REFLECT, or eval).
    coating : coatings.Stack, optional
        thin-film stack; None gives a bare interface.
    wavelength : float, optional
        wavelength in microns; required when a coating is present.

    Returns
    -------
    a_s, a_p : ndarray
        complex s- and p-amplitude coefficients, shaped like cosI.

    Notes
    -----
    TIR returns zero.  Bare reflection returns the ideal mirror (1, -1).

    """
    cosI = np.abs(np.asarray(cosI))
    theta0 = np.arccos(np.clip(cosI, 0.0, 1.0))
    if coating is not None:
        return _coating_coefficients(coating, n0, n1, cosI, theta0, typ,
                                     wavelength)
    if typ == STYPE_REFRACT:
        # Complex cos(theta1) carries TIR.
        sint1 = (n0 / n1) * np.sin(theta0)
        cost1 = _complex_sqrt(1.0 - sint1 * sint1)
        # Zero non-transmitting rays after the divisions.
        with np.errstate(divide='ignore', invalid='ignore'):
            theta1 = np.arccos(cost1)
            t_s = thinfilm.fresnel_ts(n0, n1, theta0, theta1)
            t_p = thinfilm.fresnel_tp(n0, n1, theta0, theta1)
            oblique = _complex_sqrt((n1 * cost1) / (n0 * np.cos(theta0)))
            a_s = (t_s * oblique).astype(config.precision_complex)
            a_p = (t_p * oblique).astype(config.precision_complex)
        dead = (np.imag(cost1) != 0) | ~np.isfinite(a_s) | ~np.isfinite(a_p)
        a_s[dead] = 0.0
        a_p[dead] = 0.0
        return a_s, a_p
    ones = np.ones_like(cosI, dtype=config.precision_complex)
    if typ == STYPE_REFLECT:
        return ones, -ones
    return ones, ones


def _power_coefficient(a_s, a_p):
    """Unpolarized power coefficient (|a_s|**2 + |a_p|**2) / 2."""
    return 0.5 * (np.abs(a_s) ** 2 + np.abs(a_p) ** 2)


def unpolarized_amplitude(system, trace, wavelength):
    """Per-ray scalar amplitude transmittance through the system.

    Uses sqrt((|a_s|**2 + |a_p|**2) / 2) at each surface.

    Parameters
    ----------
    system : sequence of Surface
        the traced system.
    trace : RayTraceResult
        result of raytrace through system.
    wavelength : float
        wavelength in microns.

    Returns
    -------
    amplitude : ndarray, shape (N,)
        real, nonnegative per-ray amplitude factor (1 == no loss).

    """
    cosI, n0, n1, typ = surface_normals_from_trace(
        system, trace, wavelength)
    surfaces = list(system)
    jj, n_rays = cosI.shape
    amp = np.ones(n_rays, dtype=config.precision)
    for j in range(jj):
        coating = surfaces[j].coating
        if coating is None and typ[j] != STYPE_REFRACT:
            continue
        a_s, a_p = interface_coefficients(
            n0[j], n1[j], cosI[j], typ[j], coating=coating,
            wavelength=wavelength)
        amp = amp * np.sqrt(np.clip(_power_coefficient(a_s, a_p), 0.0, None))
    return amp


def raytrace_field(system, P, S, wavelength):
    """Intensity-aware trace: geometry plus a scalar amplitude.

    Parameters
    ----------
    system : sequence of Surface
        the system to trace.
    P, S : ndarray, shape (N, 3)
        launch positions and direction cosines.
    wavelength : float
        wavelength in microns.

    Returns
    -------
    FieldTraceResult
        carrying the geometric trace and the per-ray amplitude.

    """
    trace = raytrace(system, P, S, wavelength)
    amplitude = unpolarized_amplitude(system, trace, wavelength)
    return FieldTraceResult(trace, amplitude)


def _axis_perp_basis(axis_dir, dtype):
    """Orthonormal (u, v) spanning the plane perpendicular to the optical axis."""
    if axis_dir is None:
        w = np.array([0.0, 0.0, 1.0], dtype=dtype)
    else:
        w = np.asarray(axis_dir, dtype=dtype)
        w = w / np.sqrt(np.sum(w * w))
    helper = np.array([1.0, 0.0, 0.0], dtype=dtype)
    if abs(float(np.sum(helper * w))) > 0.9:
        helper = np.array([0.0, 1.0, 0.0], dtype=dtype)
    u = helper - np.sum(helper * w) * w
    u = u / np.sqrt(np.sum(u * u))
    v = np.cross(w, u)
    return u, v


def sine_space_coords(S_last, S_chief, scale, axis_dir=None):
    """Sine-space (direction-cosine) pupil coordinates of a ray bundle.

    The scale maps direction-cosine differences to length units.

    Parameters
    ----------
    S_last : ndarray, shape (N, 3)
        post-final-surface ray direction cosines.
    S_chief : ndarray, shape (3,)
        chief-ray direction cosines (the pupil-coordinate origin).
    scale : float
        length scale (|EFL|, or the reference-sphere radius for a tilted system
        whose paraxial EFL is undefined).
    axis_dir : iterable, optional
        optical-axis direction; default +z.

    Returns
    -------
    X, Y : ndarray, shape (N,)
        sine-space pupil coordinates, referenced to the chief, in the same
        length units as the system.

    """
    S_last = np.asarray(S_last)
    S_chief = np.asarray(S_chief, dtype=S_last.dtype)
    u, v = _axis_perp_basis(axis_dir, S_last.dtype)
    # chief minus ray (not ray minus chief): the reference-sphere landing sits
    # downstream of the exit-pupil plane.
    d = float(scale) * (S_chief[None, :] - S_last)
    return d @ u, d @ v


def _inpaint_nan(arr):
    """Fill non-finite samples from finite neighbors."""
    arr = np.asarray(arr, dtype=config.precision).copy()
    hole = ~np.isfinite(arr)
    if not np.any(hole):
        return arr
    arr[hole] = 0.0
    # neighbor count depends only on grid shape
    cnt = np.zeros_like(arr)
    cnt[1:] += 1.0
    cnt[:-1] += 1.0
    cnt[:, 1:] += 1.0
    cnt[:, :-1] += 1.0
    for _ in range(int(max(arr.shape))):
        acc = np.zeros_like(arr)
        acc[1:] += arr[:-1]
        acc[:-1] += arr[1:]
        acc[:, 1:] += arr[:, :-1]
        acc[:, :-1] += arr[:, 1:]
        arr[hole] = acc[hole] / cnt[hole]
    return arr


def amplitude_apodization(entrance_xy, sphere_xy, *, valid=None):
    """Energy-conservation amplitude over the reference sphere.

    Computes sqrt(dA_entrance / dA_sphere), the square root of the local areal
    magnification of the ray bundle.

    Parameters
    ----------
    entrance_xy : ndarray, shape (M, M, 2)
        entrance-pupil sample coordinates on a structured grid.
    sphere_xy : ndarray, shape (M, M, 2)
        the corresponding sine-space pupil coordinates from sine_space_coords,
        reshaped to the same grid.
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
    # Fill holes before np.gradient spreads them to valid neighbors.
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
    amp[~np.isfinite(amp)] = 0.0
    if valid is not None:
        amp[~valid] = 0.0
    return amp


# ---------- orchestration: pupil field + propagation bridge ----------------


class PupilField:
    """Complex pupil field samples on the exit-pupil reference sphere.

    Scattered sine-space samples ready for pupil_field_to_wavefront.
    """

    __slots__ = ('X', 'Y', 'amplitude', 'opd', 'wavelength', 'efl',
                 'n_image', 'P_xp', 'P_img', 'P_matrix')

    def __init__(self, X, Y, amplitude, opd, wavelength, efl, n_image,
                 P_xp, P_img, P_matrix=None):
        self.X = X
        self.Y = Y
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


def _pupil_coordinate_scale(system, wavelength, P_xp, center):
    """Length scale for the sine-space pupil coordinate.

    Uses |EFL| when available, otherwise the reference-sphere radius.
    """
    surfaces = compiled_surfaces(system)
    try:
        return abs(float(effective_focal_length(surfaces, wvl=wavelength)))
    except ValueError:
        if P_xp is None:
            raise
        return float(np.sqrt(np.sum((np.asarray(P_xp)
                                     - np.asarray(center)) ** 2)))


def pupil_field(system, field, wavelength=None, *, epd=None, npupil=64,
                stop_index=None, P_xp=None, P_img=None, axis_dir=None,
                pupil_z=None,
                output='length', reference='chief', polarized=False):
    """Realize the complex pupil field on the exit-pupil reference sphere.

    Parameters
    ----------
    system : sequence of Surface or LensData
    field : Field
        the field point to evaluate.
    wavelength : float, optional
        wavelength in microns; defaults from LensData metadata if available.
    epd : float, optional
        entrance pupil diameter; defaults from LensData metadata.
    npupil : int
        entrance-pupil grid is npupil by npupil.
    stop_index : int, optional
        aperture-stop surface index.
    P_xp : iterable, optional
        exit-pupil center.
    P_img : iterable, optional
        reference-sphere center; defaults to the real chief-ray image point.
    axis_dir : iterable, optional
        optical-axis direction; default +z.
    pupil_z : float, optional
        launch-plane z; passed to launch().  Per-surface coatings are read from
        each Surface.coating attribute.
    output : str
        'length' (default) or 'waves' for the returned opd units.
    reference : str
        'chief' or 'centroid'.

    Returns
    -------
    PupilField
        scattered sine-space samples.

    """
    wavelength = resolve_wavelength(system, wavelength)
    if epd is None:
        resolver = getattr(system, 'entrance_pupil_diameter', None)
        if callable(resolver):
            epd = resolver(wavelength)
    if epd is None:
        raise TypeError(
            'epd is required; pass epd=... or an OpticalSystem whose aperture '
            'spec resolves it.')
    if reference not in ('chief', 'centroid'):
        raise ValueError(
            f"reference must be 'chief' or 'centroid', got {reference!r}")
    sampling = Sampling.rect(n=npupil)

    def _trace_fn(presc, P, S, w):
        if polarized:
            return raytrace_prt(presc, P, S, w)
        return raytrace_field(presc, P, S, w)

    record = trace_cell(system, field, wavelength, sampling,
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

    # Nominal coordinates define the circle; vignetted coordinates match rays.
    nominal_pupil_xy = sampling.build(0.5 * epd)
    pupil_xy = _apply_vignetting(nominal_pupil_xy, field)

    # Anchor the reference sphere before applying the circular pupil mask.
    mask = valid if reference == 'centroid' else None
    chief_index = _pupil_center_chief_index(pupil_xy, mask)

    # Rect sampling fills a square; the entrance pupil is the inscribed circle.
    r_entrance = np.hypot(
        nominal_pupil_xy[:, 0] - nominal_pupil_xy[chief_index, 0],
        nominal_pupil_xy[:, 1] - nominal_pupil_xy[chief_index, 1],
    )
    circ = r_entrance <= (0.5 * epd) * (1.0 + 1e-9)
    valid = valid & circ

    # Reference-sphere center defaults to the real chief-ray landing (P_img
    # override); the ramp stays local -- it rides the vignetting-compressed
    # grid, not the launch positions.  min_perp raised: even rect grids put
    # the auto-chief half a step off axis.
    P_img = None if P_img is None else np.asarray(P_img)
    closing = close_wavefront(system, trace, wavelength, chief_index,
                              center=P_img, P_xp=P_xp, stop_index=stop_index,
                              epd=epd, axis_dir=axis_dir, min_perp=1e-3,
                              valid=valid, reference=reference,
                              apply_field_tilt=False)
    P_img = closing.center
    P_xp = closing.P_xp
    n_image = closing.n_image
    opd = closing.opd

    # Sine-space pupil coordinates from ray direction cosines.
    scale = _pupil_coordinate_scale(system, wavelength, P_xp, P_img)
    X_all, Y_all = sine_space_coords(trace.S[-1], trace.S[-1, chief_index],
                                     scale, axis_dir)

    # amplitude: geometric apodization (structured grid) times coating factor
    entrance_xy = np.ascontiguousarray(pupil_xy).reshape(npupil, npupil, 2)
    sphere_xy = np.stack([X_all, Y_all], axis=-1).reshape(npupil, npupil, 2)
    valid_grid = valid.reshape(npupil, npupil)
    amp_geo = amplitude_apodization(entrance_xy, sphere_xy,
                                    valid=valid_grid).reshape(-1)
    # Scalar fields fold coating loss into amplitude; PRT carries it in P_matrix.
    if coating_amp is None:
        amplitude_all = amp_geo
    else:
        amplitude_all = amp_geo * coating_amp

    # local tilt removal on the compressed grid, per the closing call above
    x_pupil = pupil_xy[valid, 0] - pupil_xy[chief_index, 0]
    y_pupil = pupil_xy[valid, 1] - pupil_xy[chief_index, 1]
    tilt_field = field if field.kind == 'angle' else None
    opd, _ = _apply_field_and_output(opd, x_pupil, y_pupil, tilt_field, output,
                                     wavelength)

    n_image = abs(float(n_image))
    P_matrix = None if P_matrix_all is None else P_matrix_all[valid]
    return PupilField(
        X=X_all[valid], Y=Y_all[valid], amplitude=amplitude_all[valid],
        opd=opd, wavelength=wavelength, efl=scale / n_image, n_image=n_image,
        P_xp=(None if P_xp is None else np.asarray(P_xp)),
        P_img=P_img, P_matrix=P_matrix)


def _resample_grid(pf, npix, margin):
    """Shared scatter-to-regular-grid setup for the wavefront bridge.

    Returns interpolation points, grid spacing, and phase in nm.
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
    opd_grid[~np.isfinite(opd_grid)] = 0.0
    phase_nm = opd_grid * 1.0e6   # length (mm) -> nm
    return finite, pts, (xg, yg), dx, phase_nm


def _griddata_complex(pts, values, grid_pts):
    """Cubic griddata for complex values (real and imaginary parts), zero-fill.

    Phase is carried separately as the wavefront term.
    """
    re = interpolate.griddata(pts, np.real(values), grid_pts, method='cubic',
                              fill_value=0.0)
    im = interpolate.griddata(pts, np.imag(values), grid_pts, method='cubic',
                              fill_value=0.0)
    re[~np.isfinite(re)] = 0.0
    im[~np.isfinite(im)] = 0.0
    return re + 1j * im


def pupil_field_to_wavefront(pf, *, npix=256, margin=1.05,
                             input_polarization=None):
    """Resample scattered pupil-field samples onto a regular grid wavefront.

    Polarized fields return the transverse Ex and Ey component wavefronts.

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
        amp_grid[~np.isfinite(amp_grid)] = 0.0
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

    P_matrix maps incident 3D electric field to exitant field.
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


def raytrace_prt(system, P, S, wavelength):
    """Polarization ray trace: geometry plus a per-ray 3x3 P matrix.

    Parameters
    ----------
    system : sequence of Surface
    P, S : ndarray, shape (N, 3)
        launch positions and direction cosines.
    wavelength : float
        wavelength in microns.

    Returns
    -------
    PRTResult
        the geometric trace and the per-ray (N, 3, 3) complex P matrix.

    """
    trace = raytrace(system, P, S, wavelength)
    surfaces = list(system)
    P_hist = trace.P
    S_hist = trace.S
    n_rays = P_hist.shape[1]
    Pmat = np.broadcast_to(np.eye(3, dtype=config.precision_complex),
                           (n_rays, 3, 3)).copy()

    nj = object_space_index(surfaces, wavelength)
    for j, surf in enumerate(surfaces):
        coating = surf.coating
        k_in = _unit(S_hist[j])
        k_out = _unit(S_hist[j + 1])
        n_g, cosI = _global_normal_and_cosI(surf, P_hist[j + 1], S_hist[j])

        s = np.cross(k_in, n_g)
        s_norm = np.sqrt(np.sum(s * s, axis=-1, keepdims=True))
        # normal incidence: plane of incidence is undefined; any perpendicular
        # to k_in works because a_s == a_p there.
        degen = (s_norm[..., 0] < 1e-12)
        xhat = np.array([1.0, 0.0, 0.0], dtype=k_in.dtype)
        fallback = np.cross(k_in, xhat)
        fb_norm = np.sqrt(np.sum(fallback * fallback, axis=-1, keepdims=True))
        small = fb_norm[..., 0] < 1e-12
        if np.any(small):
            yhat = np.array([0.0, 1.0, 0.0], dtype=k_in.dtype)
            fallback[small] = np.cross(k_in[small], yhat)
            fb_norm = np.sqrt(np.sum(fallback * fallback, axis=-1,
                                     keepdims=True))
        s_norm_safe = np.where(s_norm > 0, s_norm, 1.0)
        s = np.where(degen[:, None], fallback / fb_norm, s / s_norm_safe)
        p_in = np.cross(k_in, s)
        p_out = np.cross(k_out, s)

        if surf.typ == STYPE_REFRACT:
            n1 = float(surf.material.n(wavelength))
        else:
            n1 = nj
        a_s, a_p = interface_coefficients(nj, n1, cosI, surf.typ,
                                          coating=coating, wavelength=wavelength)
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
