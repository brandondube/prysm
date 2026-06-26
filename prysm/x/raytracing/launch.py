"""Field, Sampling, launch, and stop-aim ergonomics."""

import warnings

from prysm.conf import config
from prysm.mathops import np, array_to_true_numpy

from . import raygen
from .opt import aim_rays
from .paraxial import entrance_pupil_z, NonAxialSystemError
from .spencer_and_murty import raytrace, valid_mask
from ._meta import object_space_index
from ._resolve import compiled_surfaces


def _entrance_pupil_z(system, wavelength):
    """Entrance-pupil z, using a system cache when present."""
    f = getattr(system, 'entrance_pupil_z', None)
    if not callable(f):
        surfaces = (system.to_surfaces()
                    if hasattr(system, 'to_surfaces') else system)
        stop_index = getattr(system, 'stop_index', None)
        f = lambda wvl: entrance_pupil_z(  # noqa: E731
            surfaces, wvl, stop_index=stop_index)
    try:
        return f(wavelength)
    except NonAxialSystemError:
        # Decentered/tilted geometry has no paraxial entrance pupil; the caller
        # launches (warned) instead of failing.
        return None


class Field:
    """A field point.

    kind='angle' describes a collimated source.  kind='height' describes a
    finite-conjugate object point and requires object_z.

    """

    __slots__ = ('hx', 'hy', 'kind', 'unit', 'object_z', 'vignetting')

    def __init__(self, hx=0.0, hy=0.0, kind='angle', unit='deg',
                 object_z=None, vignetting=None):
        """Initialize a field point.

        Parameters
        ----------
        hx, hy : float, optional
            field coordinates: ray angles for kind='angle' (in unit), or object
            heights in length units for kind='height'.
        kind : str, optional
            'angle' for a collimated source (default) or 'height' for a
            finite-conjugate object point.
        unit : str, optional
            angular unit for kind='angle', 'deg' (default) or 'rad'.
        object_z : float, optional
            absolute z of the object plane; required for kind='height'.
        vignetting : mapping, optional
            per-field Code V side-vignetting factors vux, vlx, vuy, vly.  Each
            scales its half of the launched pupil by (1 - factor): positive
            factors compress that side, negative factors (pupil expansion)
            grow it, and a factor of 1 or more is degenerate (rejected).  An
            all-zero mapping is treated as no vignetting.

        """
        if kind not in ('angle', 'height'):
            raise ValueError(
                f"Field kind must be 'angle' or 'height', got {kind!r}"
            )
        if kind == 'angle' and unit not in ('deg', 'rad'):
            raise ValueError(
                f"Field unit must be 'deg' or 'rad' for kind='angle', "
                f'got {unit!r}'
            )
        if kind == 'height' and object_z is None:
            raise ValueError(
                "Field kind='height' requires object_z (absolute z of "
                'the object plane)'
            )
        self.hx = float(hx)
        self.hy = float(hy)
        self.kind = kind
        self.unit = unit
        self.object_z = None if object_z is None else float(object_z)
        self.vignetting = _normalize_vignetting(vignetting)

    def angle_radians(self):
        """Return (hx, hy) in radians.  Only valid for kind='angle'."""
        if self.kind != 'angle':
            raise ValueError(
                "Field.angle_radians: kind must be 'angle', got "
                f'{self.kind!r}'
            )
        if self.unit == 'rad':
            return self.hx, self.hy
        return float(np.deg2rad(self.hx)), float(np.deg2rad(self.hy))

    def __repr__(self):
        if self.kind == 'angle':
            return (f'Field(hx={self.hx}, hy={self.hy}, '
                    f'unit={self.unit!r})')
        return (f'Field(hx={self.hx}, hy={self.hy}, kind=height, '
                f'object_z={self.object_z})')


def _normalize_vignetting(vignetting):
    """Normalize per-field Code V vignetting factors."""
    if vignetting is None:
        return None
    keys = ('vux', 'vlx', 'vuy', 'vly')
    out = {}
    for key in keys:
        value = float(vignetting.get(key, 0.0))
        if value >= 1.0:
            raise ValueError(
                f'vignetting factor {key.upper()}={value:g} collapses its '
                'side of the pupil; factors must be < 1'
            )
        out[key] = value
    if not any(out.values()):
        return None
    return out


class Sampling:
    """Pupil sampling pattern.

    Construct via the classmethod factories.  build(extent) returns a
    shape (N, 2) array of pupil xy coordinates.

    """

    __slots__ = ('kind', 'opts')

    def __init__(self, kind, **opts):
        """Initialize a pupil sampling pattern."""
        self.kind = kind
        self.opts = opts

    def build(self, extent):
        """Pupil sample coordinates scaled to the given extent."""
        kind = self.kind
        if kind == 'chief':
            return np.zeros((1, 2), dtype=config.precision)
        elif kind == 'points':
            xy = np.asarray(self.opts['xy'], dtype=config.precision) * extent
        elif kind == 'fan':
            P, _ = raygen.generate_collimated_ray_fan(
                self.opts['n'], maxr=extent,
                azimuth=self.opts.get('azimuth', 90),
                distribution=self.opts.get('distribution', 'uniform'))
            xy = P[:, :2]
        elif kind == 'cross':
            n = self.opts['n']
            dist = self.opts.get('distribution', 'uniform')
            Px, _ = raygen.generate_collimated_ray_fan(
                n, maxr=extent, azimuth=0, distribution=dist)
            Py, _ = raygen.generate_collimated_ray_fan(
                n, maxr=extent, azimuth=90, distribution=dist)
            xy = np.concatenate([Px[:, :2], Py[:, :2]], axis=0)
        elif kind == 'rect':
            P, _ = raygen.generate_collimated_rect_ray_grid(
                self.opts['n'], maxx=extent,
                distribution=self.opts.get('distribution', 'uniform'))
            xy = P[:, :2]
        elif kind == 'hex':
            nrings = self.opts['nrings']
            spacing = self.opts.get('spacing')
            if spacing is None:
                spacing = extent / nrings if nrings > 0 else 0.0
            P, _ = raygen.generate_collimated_hex_ray_grid(nrings, spacing)
            xy = P[:, :2]
        elif kind == 'spiral':
            P, _ = raygen.generate_collimated_radial_spiral_ray_grid(
                self.opts['nrings'], maxr=extent,
                samples_per_ring=self.opts.get('samples_per_ring'),
                radial_distribution=self.opts.get(
                    'radial_distribution', 'cheby'),
                include_center=self.opts.get('include_center', True))
            xy = P[:, :2]
        else:
            raise ValueError(f'unknown sampling kind {kind!r}')

        # Obscuration is the linear inner-radius ratio.
        obscuration = self.opts.get('obscuration')
        if obscuration:
            r = np.hypot(xy[:, 0], xy[:, 1])
            xy = xy[r >= float(obscuration) * extent]
        return xy

    @classmethod
    def chief(cls):
        """A single chief ray at the pupil origin."""
        return cls('chief')

    @classmethod
    def points(cls, xy):
        """Explicit normalized pupil samples."""
        return cls('points', xy=xy)

    @classmethod
    def fan(cls, n=11, axis='y', distribution='uniform', obscuration=None):
        """A 1D fan of n rays along axis ('x' or 'y')."""
        if axis == 'y':
            azi = 90
        elif axis == 'x':
            azi = 0
        else:
            raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")
        return cls('fan', n=int(n), azimuth=azi, distribution=distribution,
                   obscuration=obscuration)

    @classmethod
    def cross(cls, n=11, distribution='uniform', obscuration=None):
        """An x and y fan, 2*n rays total."""
        return cls('cross', n=int(n), distribution=distribution,
                   obscuration=obscuration)

    @classmethod
    def rect(cls, n=21, distribution='uniform', obscuration=None):
        """A rectangular n by n grid of rays."""
        return cls('rect', n=int(n), distribution=distribution,
                   obscuration=obscuration)

    @classmethod
    def hex(cls, nrings=5, spacing=None, obscuration=None):
        """A hexapolar grid of nrings concentric rings."""
        return cls('hex', nrings=int(nrings), spacing=spacing,
                   obscuration=obscuration)

    @classmethod
    def spiral(cls, nrings=5, samples_per_ring=None,
               radial_distribution='cheby', include_center=True,
               obscuration=None):
        """A radial-azimuthal spiral grid."""
        return cls('spiral', nrings=int(nrings),
                   samples_per_ring=samples_per_ring,
                   radial_distribution=radial_distribution,
                   include_center=bool(include_center),
                   obscuration=obscuration)

    def __repr__(self):
        opts = ', '.join(f'{k}={v!r}' for k, v in self.opts.items())
        sep = ', ' if opts else ''
        return f'Sampling({self.kind!r}{sep}{opts})'


def _collimated_PS(pupil_xy, pupil_z, field):
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
    n_rays = pupil_xy.shape[0]
    P = np.empty((n_rays, 3), dtype=pupil_xy.dtype)
    P[:, :2] = pupil_xy
    P[:, 2] = pupil_z
    S = np.broadcast_to(
        np.array([Sx, Sy, Sz], dtype=pupil_xy.dtype),
        (n_rays, 3),
    ).copy()
    return P, S


def _finite_PS(pupil_xy, pupil_z, field):
    n_rays = pupil_xy.shape[0]
    obj = np.array([field.hx, field.hy, field.object_z],
                   dtype=pupil_xy.dtype)
    P = np.broadcast_to(obj, (n_rays, 3)).copy()
    target = np.empty((n_rays, 3), dtype=pupil_xy.dtype)
    target[:, :2] = pupil_xy
    target[:, 2] = pupil_z
    direction = target - P
    norm = np.sqrt(np.sum(direction * direction, axis=-1, keepdims=True))
    if not np.all(norm > 0):
        raise ValueError(
            'one or more pupil samples coincide with the object point; '
            'cannot build a finite-conjugate direction'
        )
    return P, direction / norm


def _perp_basis(w):
    """Meridional T/S basis perpendicular to unit vector w."""
    st = float(np.sqrt(w[0] * w[0] + w[1] * w[1]))
    if st < 1e-12:
        e1 = np.array([1.0, 0.0, 0.0], dtype=w.dtype)
        e2 = np.array([0.0, float(np.sign(w[2])), 0.0], dtype=w.dtype)
        return e1, e2
    e1 = np.array([float(w[1]), -float(w[0]), 0.0], dtype=w.dtype) / st
    if float(e1[0]) < 0.0 or (float(e1[0]) == 0.0 and float(e1[1]) < 0.0):
        e1 = -e1
    e2 = np.cross(w, e1)
    return e1, e2


def _object_space_cone_PS(system, field, wavelength, sampling, na,
                          ep_z='paraxial'):
    """Sine-condition object cone for an object-space NA / F-number aperture.

    ep_z selects the chief-ray pivot: 'paraxial' (default) reads the paraxial
    entrance pupil, a float overrides it (the field-continuation ladder feeds
    the parabasal field-dependent EP here), and None pivots straight ahead.
    """
    if field.kind != 'height':
        raise ValueError(
            'an object-space NA / F-number aperture requires a finite-'
            "conjugate (kind='height') field")
    n_obj = object_space_index(compiled_surfaces(system), wavelength)
    sinU = float(na) / float(n_obj)
    if not (0.0 < sinU < 1.0):
        raise ValueError(
            f'object-space NA {na:g} over index {n_obj:g} gives sin(U)='
            f'{sinU:g}, which is not a physical cone half-angle')

    pupil_xy = sampling.build(1.0)  # normalized: rim at radius 1
    pupil_xy = _apply_vignetting(pupil_xy, field)
    if pupil_xy.dtype != config.precision:
        pupil_xy = pupil_xy.astype(config.precision)
    n_rays = pupil_xy.shape[0]

    obj = np.array([field.hx, field.hy, field.object_z], dtype=config.precision)

    if ep_z == 'paraxial':
        ep_z = _entrance_pupil_z(system, wavelength)
    if ep_z is not None:
        axis_pt = np.array([0.0, 0.0, float(ep_z)], dtype=config.precision)
        chief = axis_pt - obj
    else:
        chief = np.array([0.0, 0.0, 1.0], dtype=config.precision)
    chief = chief / np.sqrt(np.sum(chief * chief))

    e1, e2 = _perp_basis(chief)
    # S = sqrt(1 - sinU^2 |rho|^2) chief + sinU (rho_x e1 + rho_y e2)
    rho = pupil_xy
    trans = sinU * (rho[:, 0:1] * e1[np.newaxis, :]
                    + rho[:, 1:2] * e2[np.newaxis, :])
    axial_sq = 1.0 - sinU * sinU * np.sum(rho * rho, axis=1)
    axial = np.sqrt(np.clip(axial_sq, 0.0, None))
    S = axial[:, np.newaxis] * chief[np.newaxis, :] + trans
    P = np.broadcast_to(obj, (n_rays, 3)).copy()
    return P, S, rho


def _apply_vignetting(pupil_xy, field):
    """Scale pupil samples by per-field side-vignetting factors."""
    vignetting = getattr(field, 'vignetting', None)
    if not vignetting:
        return pupil_xy
    x = pupil_xy[:, 0]
    y = pupil_xy[:, 1]
    x = x * np.where(x >= 0.0,
                     1.0 - vignetting.get('vux', 0.0),
                     1.0 - vignetting.get('vlx', 0.0))
    y = y * np.where(y >= 0.0,
                     1.0 - vignetting.get('vuy', 0.0),
                     1.0 - vignetting.get('vly', 0.0))
    return np.stack([x, y], axis=1)


def _has_decentered_geometry(system):
    """True when any surface carries a decentered vertex or a rotation."""
    for surf in system:
        P = np.asarray(getattr(surf, 'P', (0.0, 0.0, 0.0)))
        if P.shape[0] >= 2 and bool(np.any(np.abs(P[:2]) > 1e-12)):
            return True
        R = getattr(surf, 'R', None)
        if R is not None and bool(np.any(np.abs(np.asarray(R) - np.eye(3)) > 1e-12)):
            return True
    return False


def _warn_paraxial_aiming(system, ray_aiming):
    """Warn when paraxial aiming is used with decentered geometry."""
    if ray_aiming != 'paraxial':
        return
    if _has_decentered_geometry(system):
        warnings.warn(
            'launch: the system carries tilts/decenters but '
            "ray_aiming is 'paraxial'; the paraxial entrance pupil ignores "
            "them and bundles may miss the stop.  Consider ray_aiming='real' "
            'or an explicit aim_to=stop.',
            stacklevel=3,
        )


def _real_aim_to_stop(P, S, rho, system, stop_index, wavelength, finite):
    """Aim a normalized pupil grid linearly onto the real stop.

    Returns (P, S, converged); converged is the per-ray aim_rays mask.
    """
    trace_path = system[:stop_index + 1]
    tr = raytrace(trace_path, P, S, wavelength)
    L = tr.P[-1, :, :2]
    valid = np.isfinite(L).all(axis=1)

    def _scale(rk, lk):
        # Pupil-to-stop secant through the two rim rays.
        rk = rk[valid]
        lk = lk[valid]
        if rk.size < 2:
            return 0.0
        imax = int(np.argmax(rk))
        imin = int(np.argmin(rk))
        drho = float(rk[imax] - rk[imin])
        return float(lk[imax] - lk[imin]) / drho if abs(drho) > 1e-12 else 0.0

    sx = _scale(rho[:, 0], L[:, 0])
    sy = _scale(rho[:, 1], L[:, 1])
    target = np.stack([rho[:, 0] * sx, rho[:, 1] * sy], axis=1)
    vary = 'direction' if finite else 'position'
    P, S, converged = aim_rays(P, S, system, stop_index, target,
                               wavelength, vary=vary, strict=False)
    return P, S, converged


# Adaptive field-continuation homotopy: start step, growth on success, and the
# step floor (and an iteration backstop) below which a field is taken as
# untransmittable rather than subdivided further.
_LADDER_STEP0 = 0.25
_LADDER_GROW = 1.6
_LADDER_MIN_STEP = 1.0 / 128
_LADDER_MAXITER = 200


def _scaled_field(field, frac):
    """Field with hx, hy scaled by frac (the continuation homotopy parameter)."""
    return Field(hx=field.hx * frac, hy=field.hy * frac, kind=field.kind,
                 unit=field.unit, object_z=field.object_z,
                 vignetting=field.vignetting)


def _parabasal_ep_z(system, field, wavelength):
    """Field-dependent entrance-pupil z, with paraxial fallback."""
    from .parabasal import first_order  # lazy: parabasal imports this module
    try:
        ep = first_order(system, field, wavelength).ep_z
    except (ValueError, IndexError, ArithmeticError, np.linalg.LinAlgError):
        # Degenerate parabasal chiefs fall back to the paraxial pupil.
        ep = None
    if ep is None:
        return _entrance_pupil_z(system, wavelength)
    if hasattr(ep, '__len__'):
        ep = float(np.mean(ep))
    return float(ep)


def _warm_start_bundle(P, S, seedP, seedS, finite):
    """Seed the varied transverse component from the previous ladder rung."""
    if finite:
        # direction-varied: seed the transverse direction cosines, then
        # renormalize so the trace that sets the secant scale stays unit-length.
        S[:, 0] = seedS[:, 0]
        S[:, 1] = seedS[:, 1]
        S /= np.sqrt(np.sum(S * S, axis=1, keepdims=True))
    else:
        # position-varied: seed the transverse launch positions.
        P[:, 0] = seedP[:, 0]
        P[:, 1] = seedP[:, 1]


def _aim_to_stop_with_ladder(P, S, rho, build_bundle, field, system,
                             stop_index, wavelength, finite, drop_unaimed=False):
    """Real aiming with an adaptive field-and-pupil continuation fallback.

    Walks both the field and the pupil from on-axis up to the target, warm-
    starting each rung from the previous one's solution and bisecting the step
    whenever a rung loses the chief, so the seed stays inside the next rung's
    Newton basin -- a wide-field system's valid launch window walks too fast for
    a fixed schedule, and its near-paraxial pupil places a grazing-field bundle
    metres off axis.  Scaling the pupil with the field walks the rim rays out
    gradually rather than aiming a full bundle through a violently aberrated
    pupil in one shot.  Acceptance follows the chief alone so a genuinely
    vignetted rim ray cannot stall the walk (it is dropped at the end).
    """
    P, S, conv = _real_aim_to_stop(P, S, rho, system, stop_index,
                                   wavelength, finite)
    if bool(np.all(conv)):
        return P, S

    chief = int(np.argmin(rho[:, 0] ** 2 + rho[:, 1] ** 2))
    seedP = seedS = None
    convfull = np.zeros(rho.shape[0], dtype=bool)
    Pfull = Sfull = None
    frac = 0.0
    step = _LADDER_STEP0
    for _ in range(_LADDER_MAXITER):
        if frac >= 1.0:
            break
        nxt = min(1.0, frac + step)
        fld_k = _scaled_field(field, nxt)
        ep_k = _parabasal_ep_z(system, fld_k, wavelength)
        Pk, Sk, rho_k = build_bundle(fld_k, ep_k, escale=nxt)
        if seedP is not None:
            _warm_start_bundle(Pk, Sk, seedP, seedS, finite)
        Pk, Sk, convk = _real_aim_to_stop(Pk, Sk, rho_k, system, stop_index,
                                          wavelength, finite)
        if bool(convk[chief]):
            seedP, seedS = Pk, Sk
            frac = nxt
            step = min(step * _LADDER_GROW, 1.0)
            if frac >= 1.0:
                convfull, Pfull, Sfull = convk, Pk, Sk
        else:
            step *= 0.5
            if step < _LADDER_MIN_STEP:
                break

    # Adopt only the full-field rays the primary missed.
    rescued = convfull & ~conv
    if bool(np.any(rescued)):
        P = P.copy()
        S = S.copy()
        P[rescued] = Pfull[rescued]
        S[rescued] = Sfull[rescued]

    if drop_unaimed:
        aimed = conv | convfull
        if not bool(np.all(aimed)):
            # Keep launch positions finite for chief selection.
            S = np.array(S, copy=True)
            S[~aimed] = np.nan
    return P, S


def launch(system, field, wavelength, sampling, *,
           epd=None, pupil_extent=None, pupil_z=None,
           aim_to=None, aim_target=(0.0, 0.0), aim_strict=True,
           drop_unaimed=True):
    """Build (P, S) for one field, wavelength, and pupil sampling.

    Parameters
    ----------
    system : sequence of Surface
        the system to launch into.
    field : Field
        the field point.  kind='angle' or kind='height'.
    wavelength : float
        wavelength in microns.
    sampling : Sampling
        pupil sampling pattern.
    epd : float, optional
        entrance pupil diameter.
    pupil_extent : float, optional
        pattern outer half-extent, used in place of epd/2 when given.
    pupil_z : float, optional
        z position of the collimated launch plane.
    aim_to : int, optional
        aimed surface index.
    aim_target : (float, float), optional
        target xy at the aimed surface.  Default (0, 0) (chief-ray aim).
    aim_strict : bool, optional
        forwarded to aim_rays on the explicit aim_to path.
    drop_unaimed : bool, optional
        under real aiming, NaN the S of rays that cannot be aimed onto the stop
        so a vignetted ray cannot masquerade as a valid bundle sample.  Default
        True (ADR-0009); solves that must probe the rim pass False.

    Returns
    -------
    P, S : ndarray, ndarray
        shape (N, 3) launch positions and direction cosines.

    """
    ray_aiming = str(getattr(system, 'ray_aiming', 'paraxial')).lower()
    real_aiming = (ray_aiming == 'real' and aim_to is None
                   and sampling.kind != 'chief')
    stop_index = getattr(system, 'stop_index', None)
    if aim_to is None:
        _warn_paraxial_aiming(system, ray_aiming)

    # Object-space aperture modes launch from an object-space cone.
    object_mode = False
    na = None
    if epd is None and pupil_extent is None:
        aperture = getattr(system, 'aperture', None)
        bc = aperture.resolve(system, wavelength) if aperture is not None else None
        object_mode = bc is not None and bc[0] in ('NA_OBJECT', 'FNO_OBJECT')
        if object_mode:
            na = bc[1] if bc[0] == 'NA_OBJECT' else 1.0 / (2.0 * bc[1])

    finite = object_mode or field.kind != 'angle'

    if not object_mode:
        if epd is None and pupil_extent is None:
            # default from the system aperture spec if any
            resolver = getattr(system, 'entrance_pupil_diameter', None)
            if callable(resolver):
                epd = resolver(wavelength)
        if sampling.kind != 'chief' and epd is None and pupil_extent is None:
            raise ValueError(
                f'sampling kind {sampling.kind!r} needs an entrance pupil '
                'size; pass epd=... or pupil_extent=...'
            )
        if pupil_extent is not None:
            extent = float(pupil_extent)
        elif epd is not None:
            extent = float(epd) / 2.0
        else:
            extent = 0.0
        if pupil_z is None:
            pupil_z = float(system[0].P[2])
        pupil_z = float(pupil_z)

    def _build(fld, ep_z, escale=1.0):
        """Bundle (P, S, rho) for one field, seeded onto entrance pupil ep_z.

        ep_z = 'paraxial' reads the paraxial pupil, a float overrides it (the
        ladder feeds the parabasal field-dependent EP), and None means no seed.
        escale shrinks the pupil extent for the continuation homotopy (rho stays
        normalized), so a wide-field bundle's rim rays walk out gradually.
        """
        if object_mode:
            return _object_space_cone_PS(system, fld, wavelength,
                                         sampling, na, ep_z=ep_z)
        e = (_entrance_pupil_z(system, wavelength)
             if ep_z == 'paraxial' else ep_z)
        ext = extent * escale
        pupil_xy = sampling.build(ext)
        pupil_xy = _apply_vignetting(pupil_xy, fld)
        if pupil_xy.dtype != config.precision:
            pupil_xy = pupil_xy.astype(config.precision)
        if fld.kind == 'angle':
            P, S = _collimated_PS(pupil_xy, pupil_z, fld)
            if e is not None:
                # Slide the collimated bundle to the entrance-pupil plane.
                S0 = S[0]
                shift = (pupil_z - e) / S0[2]
                P = P + np.stack([shift * S0[0], shift * S0[1],
                                  np.zeros_like(shift)])
        else:
            target_z = float(e) if e is not None else pupil_z
            P, S = _finite_PS(pupil_xy, target_z, fld)
        rho = pupil_xy / ext if ext > 0.0 else np.zeros_like(pupil_xy)
        return P, S, rho

    # Primary bundle: paraxial-EP seed (no seed when explicitly aiming).
    P, S, rho = _build(field, None if aim_to is not None else 'paraxial')

    # position the bundle relative to the stop
    if aim_to is not None:
        vary = 'direction' if finite else 'position'
        P, S, _ = aim_rays(
            P, S, system, aim_to, aim_target, wavelength,
            strict=aim_strict, vary=vary,
        )
    elif real_aiming and stop_index is not None:
        P, S = _aim_to_stop_with_ladder(
            P, S, rho, _build, field, system, stop_index, wavelength,
            finite, drop_unaimed=drop_unaimed)

    return P, S


def solve_vignetting(system, field, wavelength, *, tol=1e-3, maxiter=20):
    """Solve Code V-style real-ray vignetting factors for one field."""
    # Solve on an unvignetted clone.
    bare = Field(field.hx, field.hy, kind=field.kind, unit=field.unit,
                 object_z=field.object_z)
    # Row 0 is chief; rows 1-4 match keys below.
    edges = np.asarray([
        [0.0, 0.0],
        [1.0, 0.0],
        [-1.0, 0.0],
        [0.0, 1.0],
        [0.0, -1.0],
    ], dtype=config.precision)
    keys = ('vux', 'vlx', 'vuy', 'vly')

    def transmits(scales):
        s = np.asarray([1.0, *scales], dtype=config.precision)
        xy = edges * s[:, np.newaxis]
        # Probe the rim to find where vignetting begins -- keep best-effort
        # un-aimed rays rather than NaN-ing them (ADR-0009).
        P, S = launch(system, bare, wavelength, Sampling.points(xy),
                      drop_unaimed=False)
        result = raytrace(system, P, S, wavelength)
        return array_to_true_numpy(valid_mask(result.status))

    valid = transmits([1.0] * 4)
    if not bool(valid[0]):
        raise ValueError(
            'solve_vignetting: the chief ray does not transmit; vignetting '
            'factors are referenced to it')
    lo = [1.0 if bool(v) else 0.0 for v in valid[1:]]
    hi = [1.0] * 4
    active = [not bool(v) for v in valid[1:]]
    for _ in range(maxiter):
        gaps = [h - l for h, l, a in zip(hi, lo, active) if a]
        if not gaps or max(gaps) <= tol:
            break
        mid = [(l + h) / 2.0 if a else 1.0
               for l, h, a in zip(lo, hi, active)]
        vm = transmits(mid)
        for i in range(4):
            if active[i]:
                if bool(vm[i + 1]):
                    lo[i] = mid[i]
                else:
                    hi[i] = mid[i]
    for key, l, a in zip(keys, lo, active):
        if a and l == 0.0:
            raise ValueError(
                f'solve_vignetting: the {key} edge ray fails at every probed '
                'pupil scale; the side appears fully vignetted')
    return {key: 1.0 - l for key, l in zip(keys, lo)}
