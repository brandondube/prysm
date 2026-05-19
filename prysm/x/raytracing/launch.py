"""Field / Sampling / launch / stop-aim ergonomics.

Collapses the standard "build a launch bundle for one (field, wavelength,
pupil sampling) combination" recipe into a single call.

- Field: a vanilla (hx, hy, kind, unit) record.  kind='angle' for collimated
  systems; kind='height' for finite-conjugate object planes.
- Sampling: pupil sampling pattern (chief / fan / cross / rect / hex /
  spiral); built from classmethod factories that return a self-describing
  object whose .build(extent) returns pupil-relative (P, S).
- launch: combines a Field, a Sampling, and a prescription to produce
  absolute (P, S) ready for raytrace.  Optional aim_to triggers per-ray
  stop aiming via aim_bundle_to_surface.
- aim_bundle_to_surface: vectorized wrapper around opt.ray_aim that
  preserves the launch z (which the legacy ray_aim resets to zero).

"""

from prysm.conf import config
from prysm.mathops import np

from .raygen import (
    generate_collimated_ray_fan,
    generate_collimated_rect_ray_grid,
    generate_collimated_hex_ray_grid,
    generate_collimated_radial_spiral_ray_grid,
    concat_rayfans,
)
from .opt import ray_aim


# ---------- Field -----------------------------------------------------------

class Field:
    """A field point: (hx, hy) angular or height pair.

    For collimated (object at infinity) systems use kind='angle' and
    pass (hx, hy) in `unit` ('deg' or 'rad').  For finite-conjugate
    object planes use kind='height' and pass (hx, hy) in length units.

    """

    __slots__ = ('hx', 'hy', 'kind', 'unit')

    def __init__(self, hx=0.0, hy=0.0, kind='angle', unit='deg'):
        if kind not in ('angle', 'height'):
            raise ValueError(
                f"Field kind must be 'angle' or 'height', got {kind!r}"
            )
        if kind == 'angle' and unit not in ('deg', 'rad'):
            raise ValueError(
                f"Field unit must be 'deg' or 'rad' when kind='angle', got {unit!r}"
            )
        self.hx = float(hx)
        self.hy = float(hy)
        self.kind = kind
        self.unit = unit

    def angle_radians(self):
        """Return (hx, hy) in radians.  Only valid for kind='angle'."""
        if self.kind != 'angle':
            raise ValueError(
                "Field.angle_radians: kind must be 'angle', got "
                f"{self.kind!r}"
            )
        if self.unit == 'rad':
            return self.hx, self.hy
        return float(np.deg2rad(self.hx)), float(np.deg2rad(self.hy))

    def __repr__(self):
        if self.kind == 'angle':
            return f'Field(hx={self.hx}, hy={self.hy}, unit={self.unit!r})'
        return f'Field(hx={self.hx}, hy={self.hy}, kind=height)'


# ---------- Sampling --------------------------------------------------------

class Sampling:
    """Pupil sampling pattern.

    Use the classmethod factories (Sampling.fan, .cross, .rect, .hex,
    .spiral, .chief) to construct.  build(extent) returns (P, S) in
    pupil-relative coordinates: P has shape (N, 3) with z=0 and (x, y)
    sampling the requested pattern; S has shape (N, 3) of unit +z
    direction cosines (pre-tilt; the launcher applies the field tilt).

    The `extent` argument to build() is the pattern's outer half-extent
    (typically EPD/2 for collimated systems).

    """

    __slots__ = ('kind', 'opts')

    def __init__(self, kind, **opts):
        self.kind = kind
        self.opts = opts

    def build(self, extent):
        """Generate (P, S) at pupil z=0, scaled to the given extent."""
        kind = self.kind
        if kind == 'chief':
            P = np.zeros((1, 3), dtype=config.precision)
            S = np.zeros((1, 3), dtype=config.precision)
            S[0, 2] = 1.0
            return P, S
        if kind == 'fan':
            n = self.opts['n']
            azimuth = self.opts.get('azimuth', 90)
            return generate_collimated_ray_fan(n, maxr=extent, azimuth=azimuth)
        if kind == 'cross':
            n = self.opts['n']
            fx = generate_collimated_ray_fan(n, maxr=extent, azimuth=0)
            fy = generate_collimated_ray_fan(n, maxr=extent, azimuth=90)
            return concat_rayfans(fx, fy)
        if kind == 'rect':
            n = self.opts['n']
            return generate_collimated_rect_ray_grid(n, maxx=extent)
        if kind == 'hex':
            nrings = self.opts['nrings']
            spacing = self.opts.get('spacing')
            if spacing is None:
                spacing = extent / nrings
            return generate_collimated_hex_ray_grid(nrings, spacing)
        if kind == 'spiral':
            nrings = self.opts['nrings']
            return generate_collimated_radial_spiral_ray_grid(
                nrings, maxr=extent,
                samples_per_ring=self.opts.get('samples_per_ring'),
                radial_distribution=self.opts.get('radial_distribution', 'cheby'),
                include_center=self.opts.get('include_center', True),
            )
        raise ValueError(f'unknown sampling kind {kind!r}')

    @classmethod
    def chief(cls):
        """A single chief ray (1 ray at the pupil origin)."""
        return cls('chief')

    @classmethod
    def fan(cls, n=11, axis='y'):
        """A 1D fan of n rays along the chosen axis ('x' or 'y')."""
        if axis == 'y':
            azi = 90
        elif axis == 'x':
            azi = 0
        else:
            raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")
        return cls('fan', n=int(n), azimuth=azi)

    @classmethod
    def cross(cls, n=11):
        """An X+Y fan, 2*n total rays (the central ray appears twice)."""
        return cls('cross', n=int(n))

    @classmethod
    def rect(cls, n=21):
        """A rectangular n*n grid."""
        return cls('rect', n=int(n))

    @classmethod
    def hex(cls, nrings=5, spacing=None):
        """A hexapolar grid of nrings concentric rings (1+3*nrings*(nrings+1) rays)."""
        return cls('hex', nrings=int(nrings), spacing=spacing)

    @classmethod
    def spiral(cls, nrings=5, samples_per_ring=None,
               radial_distribution='cheby', include_center=True):
        """A radial-azimuthal spiral grid (Forbes-style Q-poly fitting grid)."""
        return cls('spiral', nrings=int(nrings),
                   samples_per_ring=samples_per_ring,
                   radial_distribution=radial_distribution,
                   include_center=bool(include_center))

    def __repr__(self):
        opts = ', '.join(f'{k}={v!r}' for k, v in self.opts.items())
        return f'Sampling({self.kind!r}{", " + opts if opts else ""})'


# ---------- launch ----------------------------------------------------------

def launch(prescription, field, wavelength, sampling, *,
           epd=None, n_ambient=1.0, pupil_z=None,
           aim_to=None, aim_target=(0.0, 0.0)):
    """Build (P, S) for one (field, wavelength) launch bundle.

    Currently supports collimated (kind='angle') Field only; finite-
    conjugate launches will follow.

    Parameters
    ----------
    prescription : sequence of Surface
    field : Field
        the field point.  kind='angle' is required in this version.
    wavelength : float
        wavelength in microns.  Only consumed when aim_to is set (the
        aim trace needs it); has no effect on the launch geometry itself.
    sampling : Sampling
        pupil sampling pattern.
    epd : float
        entrance pupil diameter; sampling is built with extent = epd/2.
        Required for non-chief samplings.
    n_ambient : float
        ambient index of refraction (only used during aim).
    pupil_z : float, optional
        z position of the entrance pupil; rays start at this z.  Default:
        the first surface's vertex z.  For systems with an EP not at the
        first surface (most common: pupil image-side of a mirror), supply
        pupil_z explicitly.
    aim_to : int, optional
        if given, run vectorized stop aiming so each ray hits surface
        prescription[aim_to] at aim_target.  Useful for chief / marginal
        / vignetted ray studies.
    aim_target : (float, float), optional
        target xy at the aimed surface.  Default (0, 0) (chief-ray aim).

    Returns
    -------
    P, S : ndarray, ndarray
        shape (N, 3) launch positions and direction cosines.

    """
    if field.kind != 'angle':
        raise NotImplementedError(
            "launch() currently supports only collimated kind='angle' "
            "fields; finite-conjugate (kind='height') launches will be "
            "added later"
        )
    if sampling.kind != 'chief' and epd is None:
        raise ValueError(
            f'sampling kind {sampling.kind!r} needs an entrance pupil '
            'diameter; pass epd=...'
        )

    extent = 0.0 if epd is None else float(epd) / 2.0
    Pp, _Sp = sampling.build(extent)

    ax, ay = field.angle_radians()
    Sx = float(np.sin(ax))
    Sy = float(np.sin(ay))
    Sz_sq = 1.0 - Sx * Sx - Sy * Sy
    if Sz_sq < 0.0:
        raise ValueError(
            f'field angles ({ax}, {ay}) rad have sin^2 sum > 1; '
            'beam direction is not physical'
        )
    Sz = float(np.sqrt(Sz_sq))
    S = np.broadcast_to(np.array([Sx, Sy, Sz], dtype=Pp.dtype),
                        Pp.shape).copy()

    if pupil_z is None:
        pupil_z = float(prescription[0].P[2])
    P = Pp.astype(Pp.dtype, copy=True)
    P[:, 2] = pupil_z

    if aim_to is not None:
        P = aim_bundle_to_surface(
            P, S, prescription, aim_to,
            target_xy=aim_target,
            wavelength=wavelength,
            n_ambient=n_ambient,
        )

    return P, S


# ---------- aim -------------------------------------------------------------

def aim_bundle_to_surface(P, S, prescription, surface_index, *,
                          target_xy=(0.0, 0.0), wavelength,
                          n_ambient=1.0, tol=1e-12, maxiter=200,
                          strict=True):
    """Per-ray stop aim: adjust each ray's launch (Px, Py) so it lands at
    target_xy on prescription[surface_index].

    Each ray is solved independently via a scipy L-BFGS-B in opt.ray_aim;
    the launch z is preserved (the legacy ray_aim resets z to 0; we
    restore it after each call).  This O(N) loop is the simple and robust
    implementation; a vectorized version is possible but rarely
    bottlenecks design problems compared to merit evaluation cost.

    Parameters
    ----------
    P, S : ndarray, shape (N, 3)
        nominal launch positions and direction cosines (e.g. from
        Sampling.build + a launch() pre-aim).
    prescription : sequence of Surface
    surface_index : int
        which surface to aim at; rays land at target_xy on this surface.
    target_xy : (float, float), optional
        target landing XY in the surface's own local frame.  (0, 0) is the
        chief-ray aim (centered on the stop / aimed surface).
    wavelength : float
        wavelength for the aim trace, in microns.
    n_ambient : float
        ambient index of refraction.
    tol, maxiter : float, int
        L-BFGS-B convergence settings; forwarded to opt.ray_aim.
    strict : bool, optional
        if True (default), raise on per-ray aim failure; if False, leave
        the failed ray at its nominal P.

    Returns
    -------
    P : ndarray, shape (N, 3)
        launch positions with (x, y) adjusted.  z is preserved.

    """
    P = np.asarray(P).copy()
    S = np.asarray(S)
    target = (float(target_xy[0]), float(target_xy[1]), float('nan'))
    n_rays = P.shape[0]
    for i in range(n_rays):
        z_i = float(P[i, 2])
        new_P = ray_aim(
            P[i], S[i], prescription, surface_index, wavelength,
            target=target, tol=tol, maxiter=maxiter, strict=strict,
            n_ambient=n_ambient,
        )
        new_P[2] = z_i
        P[i] = new_P
    return P
