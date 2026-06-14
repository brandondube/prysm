"""Design operands and optimization problems for raytracing systems."""

import inspect
import math
import warnings

from prysm.conf import config
from prysm.mathops import np
from prysm.x.optym.least_squares import (  # NOQA - re-export for users
    DampedLeastSquares,
    DampedLeastSquaresResult,
    damped_least_squares,
)

from .launch import Field, Sampling, launch as _launch
from .spencer_and_murty import raytrace, valid_mask, STYPE_EVAL
from .surfaces import _map_stype
from .sensitivity import merit_jacobian_free as _merit_jacobian_free
from .opt import (
    rms_spot_radius, _pupil_center_chief_index,
    hopkins_eic_closing, reference_sphere_curvature,
)
from .paraxial import (
    effective_focal_length,
    back_focal_length,
    paraxial_image_distance,
)
from . import analysis as _analysis
from ._meta import object_space_index, image_space_index, system_first_order

_CACHE_MISS = object()


# ---------- Trace cache ------------------------------------------------------

class _TraceCache:
    """Per-merit-call raytrace cache keyed by `(id(P), id(S), wvl)`."""

    __slots__ = ('_prescription', '_cache', '_n_traces', '_xp_cache',
                 '_launch_cache')

    def __init__(self, prescription):
        self._prescription = prescription
        self._cache = {}
        self._xp_cache = {}
        self._launch_cache = {}
        self._n_traces = 0

    def launch(self, field, wavelength, sampling, *, epd=None):
        """Launch bundle (P, S) for a recipe, memoized for this merit call."""
        key = (None if field is None else id(field),
               None if sampling is None else id(sampling),
               float(wavelength), epd)
        cached = self._launch_cache.get(key)
        if cached is None:
            f = Field() if field is None else field
            s = Sampling.hex(nrings=4) if sampling is None else sampling
            cached = _launch(self._prescription, f, wavelength, s, epd=epd)
            self._launch_cache[key] = cached
        return cached

    def trace(self, P, S, wavelength):
        key = (id(P), id(S), float(wavelength))
        cached = self._cache.get(key)
        if cached is None:
            cached = raytrace(self._prescription, P, S, wavelength)
            self._cache[key] = cached
            self._n_traces += 1
        return cached

    def exit_pupil(self, P, S, wavelength, *, P_xp=None, chief_index=None,
                   stop_index=None, epd=None, axis_point=None, axis_dir=None):
        """Exit-pupil reference point for an operand bundle, resolved once."""
        if P_xp is not None:
            return np.asarray(P_xp)
        key = (id(P), id(S), float(wavelength), stop_index)
        cached = self._xp_cache.get(key, _CACHE_MISS)
        if cached is _CACHE_MISS:
            resolved_stop = (stop_index if stop_index is not None
                             else getattr(self._prescription, 'stop_index', None))
            chief = None
            if resolved_stop is None:
                tr = self.trace(P, S, wavelength)
                ci = (chief_index if chief_index is not None
                      else _pupil_center_chief_index(P))
                chief = (tr.P[-1, ci], tr.S[-1, ci])
            cached = _analysis.resolve_exit_pupil(
                self._prescription, wavelength, stop_index=stop_index, epd=epd,
                chief=chief, axis_point=axis_point, axis_dir=axis_dir)
            self._xp_cache[key] = cached
        return cached

    @property
    def n_traces(self):
        """Number of underlying `raytrace` calls made (cache misses)."""
        return self._n_traces


# ---------- Operands ---------------------------------------------------------

def _resolve_wavelength(prescription, wavelength):
    """Resolve an operand wavelength, deferring None to the prescription."""
    if wavelength is not None:
        return float(wavelength)
    resolve = getattr(prescription, 'wavelength', None)
    if callable(resolve):
        return float(resolve(None))
    raise ValueError(
        'operand wavelength=None resolves the reference wavelength of an '
        'OpticalSystem; this prescription carries no wavelengths -- pass '
        'wavelength= explicitly'
    )


def _class_accepts_kw(cls, name):
    """True when cls can be called with keyword `name`."""
    params = inspect.signature(cls).parameters
    if name in params:
        return True
    return any(p.kind == inspect.Parameter.VAR_KEYWORD
               for p in params.values())


class Merit:
    """Target/weight plumbing and adjoint contract for merit terms."""

    name = 'merit'

    def __init__(self, target=None, weight=1.0, *, min=None, max=None):
        self.target = 0.0 if target is None else float(target)
        self.weight = float(weight)
        self.min = None if min is None else float(min)
        self.max = None if max is None else float(max)
        self._target_set = target is not None

    def _bundle(self, prescription, cache):
        """Resolved (P, S, wavelength) for ray-based merits; None otherwise."""
        return None

    def __call__(self, prescription, cache):
        raise NotImplementedError(
            f'{type(self).__name__} provides no optimizer value')

    def value(self, trace, prescription, wavelength):
        raise NotImplementedError(
            f'{type(self).__name__} provides no trace-based value')

    def seed(self, trace, prescription, wavelength):
        raise NotImplementedError(
            f'{type(self).__name__} provides no adjoint seed')

    def direct_gradient(self, trace, prescription, wavelength, seeds):
        """Optional direct d merit / d seed terms outside the ray-state sweep."""
        return None

    @property
    def seedable(self):
        """True when this merit overrides seed (drives the adjoint sweep)."""
        return type(self).seed is not Merit.seed

    @property
    def has_value(self):
        """True when this merit overrides value for a traced bundle."""
        return type(self).value is not Merit.value


def _zeros_like_trace_state(trace):
    n = trace.P[-1].shape[0]
    P_bar = np.zeros((n, 3), dtype=config.precision)
    S_bar = np.zeros((n, 3), dtype=config.precision)
    L_bar = np.zeros(n, dtype=config.precision)
    return P_bar, S_bar, L_bar


class _RayMerit(Merit):
    """Merit over one launch recipe (field, wavelength, sampling).

    field=None is the on-axis Field(); wavelength=None is the prescription
    reference wavelength; sampling=None is Sampling.hex(nrings=4) -- all
    resolved at call time, not construction.  epd overrides the launch pupil
    size (defaulting to the prescription aperture).

    """

    def __init__(self, field=None, wavelength=None, sampling=None, *,
                 target=None, weight=1.0, min=None, max=None, epd=None):
        super().__init__(target=target, weight=weight, min=min, max=max)
        self.field = field
        self.wavelength = None if wavelength is None else float(wavelength)
        self.sampling = sampling
        self.epd = epd

    def _bundle(self, prescription, cache):
        wvl = _resolve_wavelength(prescription, self.wavelength)
        P, S = cache.launch(self.field, wvl, self.sampling, epd=self.epd)
        return P, S, wvl


class RmsSpotRadius(_RayMerit):
    """Weighted RMS spot radius at the image plane for one launch recipe."""

    name = 'rms_spot_radius'

    def __call__(self, prescription, cache):
        P, S, wvl = self._bundle(prescription, cache)
        trace = cache.trace(P, S, wvl)
        return self.value(trace, prescription, wvl)

    def value(self, trace, prescription, wavelength):
        return float(rms_spot_radius(trace.P[-1], status=trace.status))

    def seed(self, trace, prescription, wavelength):
        P_bar, S_bar, L_bar = _zeros_like_trace_state(trace)
        valid = valid_mask(trace.status, trace.P[-1])
        xy = trace.P[-1][valid, :2]
        nv = xy.shape[0]
        if nv == 0:
            return P_bar, S_bar, L_bar
        centroid = xy.mean(axis=0)
        delta = xy - centroid
        rms = float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))
        if rms <= 1e-300:
            return P_bar, S_bar, L_bar
        P_bar[valid, 0] = delta[:, 0] / (nv * rms)
        P_bar[valid, 1] = delta[:, 1] / (nv * rms)
        return P_bar, S_bar, L_bar


class RayHeightAt(_RayMerit):
    """Position of one ray along one Cartesian axis at one prescription
    surface.

    Useful for chief / marginal ray boundary conditions and for
    constraining specific image-plane points (e.g.,
    surface_index=-1, axis=1 is the y-position at the image plane).
    ray_index selects the ray within the launched sampling pattern.

    """

    def __init__(self, field=None, wavelength=None, sampling=None, *,
                 surface_index, axis, target=None, weight=1.0,
                 min=None, max=None, ray_index=0, epd=None):
        super().__init__(field, wavelength, sampling, target=target,
                         weight=weight, min=min, max=max, epd=epd)
        self.surface_index = int(surface_index)
        self.axis = int(axis)
        self.ray_index = int(ray_index)

    def __call__(self, prescription, cache):
        P, S, wvl = self._bundle(prescription, cache)
        trace = cache.trace(P, S, wvl)
        return float(trace.P[self.surface_index, self.ray_index, self.axis])


class Boresight(_RayMerit):
    """Centroid distance from a target point at the final surface, for
    one launch recipe.  Use to enforce a chief-ray landing point.

    """

    def __init__(self, field=None, wavelength=None, sampling=None, *,
                 target_xy=(0.0, 0.0), weight=1.0, min=None, max=None,
                 epd=None):
        # boresight residual is the distance to target_xy; target stays 0
        super().__init__(field, wavelength, sampling, weight=weight,
                         min=min, max=max, epd=epd)
        self.target_xy = (float(target_xy[0]), float(target_xy[1]))

    def __call__(self, prescription, cache):
        P, S, wvl = self._bundle(prescription, cache)
        trace = cache.trace(P, S, wvl)
        valid = valid_mask(trace.status, trace.P[-1])
        Pf = trace.P[-1]
        if valid.any():
            mean = Pf[valid, :2].mean(axis=0)
        else:
            mean = Pf[:, :2].mean(axis=0)
        dx = mean[0] - self.target_xy[0]
        dy = mean[1] - self.target_xy[1]
        return float(np.sqrt(dx * dx + dy * dy))


class EFL(Merit):
    """Effective focal length (paraxial ABCD).

    wavelength=None resolves the prescription reference wavelength at call
    time, so EFL(target=100) works as a constraint on an OpticalSystem.

    """

    name = 'efl'

    def __init__(self, wavelength=None, target=None, weight=1.0, *,
                 min=None, max=None):
        super().__init__(target=target, weight=weight, min=min, max=max)
        self.wavelength = None if wavelength is None else float(wavelength)

    def __call__(self, prescription, cache):
        wvl = _resolve_wavelength(prescription, self.wavelength)
        return float(effective_focal_length(prescription, wvl=wvl))


class BFL(Merit):
    """Back focal length (last powered surface vertex to rear focal point)."""

    name = 'bfl'

    def __init__(self, wavelength=None, target=None, weight=1.0, *,
                 min=None, max=None):
        super().__init__(target=target, weight=weight, min=min, max=max)
        self.wavelength = None if wavelength is None else float(wavelength)

    def __call__(self, prescription, cache):
        wvl = _resolve_wavelength(prescription, self.wavelength)
        return float(back_focal_length(prescription, wvl=wvl))


class ParaxialImageDistance(Merit):
    """Signed distance from the last surface vertex to the paraxial image
    plane (collimated on-axis input)."""

    name = 'paraxial_image_distance'

    def __init__(self, wavelength=None, target=None, weight=1.0, *,
                 min=None, max=None):
        super().__init__(target=target, weight=weight, min=min, max=max)
        self.wavelength = None if wavelength is None else float(wavelength)

    def __call__(self, prescription, cache):
        wvl = _resolve_wavelength(prescription, self.wavelength)
        return float(paraxial_image_distance(prescription, wvl=wvl))


class TotalTrack(Merit):
    """Axial length from the first non-object row through the image surface.

    The Code V TTL: the sum of finite row gaps, skipping the leading object
    row's gap (finite or infinite) so only the glass-to-image track counts.
    Reads the prescription rows; no rays are traced.

    """

    name = 'total_track'

    def __init__(self, target=None, weight=1.0, *, min=None, max=None):
        super().__init__(target=target, weight=weight, min=min, max=max)

    def __call__(self, prescription, cache):
        rows = prescription.rows
        start = 0
        if rows:
            typ = getattr(rows[0], 'typ', None)
            if typ is not None and _map_stype(typ) == STYPE_EVAL:
                start = 1  # leading object row; its gap is object distance
        total = 0.0
        for row in rows[start:]:
            t = float(getattr(row, 'thickness', 0.0))
            if math.isfinite(t):
                total += t
        return float(total)


class Thickness(Merit):
    """One prescription row's axial gap, by row index.

    The edge-guard constraint: Thickness(3, min=0.5) keeps row 3's gap
    manufacturable while other DOFs move.

    """

    name = 'thickness'

    def __init__(self, surface, target=None, weight=1.0, *,
                 min=None, max=None):
        super().__init__(target=target, weight=weight, min=min, max=max)
        self.surface = int(surface)

    def __call__(self, prescription, cache):
        return float(prescription.rows[self.surface].thickness)


class _CallableMerit(Merit):
    """Adapter giving a bare f(prescription, cache) -> float the Merit
    protocol, with target=0, weight=1, and the callable's name."""

    def __init__(self, fn, target=None, weight=1.0, *, min=None, max=None):
        super().__init__(target=target, weight=weight, min=min, max=max)
        self.fn = fn
        self.name = getattr(fn, '__name__', 'callable')

    def __call__(self, prescription, cache):
        return float(self.fn(prescription, cache))


class WavefrontRMS(_RayMerit):
    """RMS of OPD on the chief-ray reference sphere.

    epd forwards both to the launch and to the exit-pupil resolution.

    """

    name = 'rms_wfe'

    def __init__(self, field=None, wavelength=None, sampling=None, *,
                 target=None, weight=1.0, min=None, max=None,
                 chief_index=None,
                 axis_point=None, axis_dir=None, P_xp=None,
                 epd=None, stop_index=None):
        super().__init__(field, wavelength, sampling, target=target,
                         weight=weight, min=min, max=max, epd=epd)
        self.chief_index = chief_index
        self.axis_point = axis_point
        self.axis_dir = axis_dir
        self.P_xp = P_xp
        self.stop_index = stop_index

    def _geometry(self, trace, prescription, wavelength, *, P_xp_override=None):
        """Reference-sphere geometry + EIC OPD shared by value and seed."""
        valid = valid_mask(trace.status, trace.P[-1])
        chief = self.chief_index
        if chief is None:
            chief = _pupil_center_chief_index(trace.P[0])
        if not bool(valid[chief]):
            raise ValueError('chief ray is invalid; cannot define reference sphere')

        P_last = trace.P[-1]
        S_last = trace.S[-1]
        C = P_last[chief]
        n_object = object_space_index(prescription, wavelength)
        n_image = image_space_index(prescription, wavelength, fallback=n_object)

        if P_xp_override is not None:
            P_xp = np.asarray(P_xp_override, dtype=config.precision)
            xp_mode = 'fixed'
        elif self.P_xp is not None:
            P_xp = np.asarray(self.P_xp, dtype=config.precision)
            xp_mode = 'fixed'
        else:
            resolved_stop = (self.stop_index if self.stop_index is not None
                             else getattr(prescription, 'stop_index', None))
            if resolved_stop is not None:
                try:
                    fo = system_first_order(
                        prescription, wvl=wavelength, epd=self.epd,
                        stop_index=resolved_stop)
                except ValueError as exc:
                    if ((self.axis_point is None and self.axis_dir is None)
                            or not _analysis._first_order_geometry_failure(exc)):
                        raise
                    P_xp = _analysis.resolve_exit_pupil(
                        prescription, wavelength, stop_index=self.stop_index,
                        epd=self.epd, chief=(C, S_last[chief]),
                        axis_point=self.axis_point, axis_dir=self.axis_dir)
                    xp_mode = 'geometric'
                else:
                    if fo.xp_z is None:
                        P_xp = None
                    else:
                        P_xp = np.array([0.0, 0.0, float(fo.xp_z)],
                                        dtype=config.precision)
                    xp_mode = 'paraxial'
            else:
                P_xp = _analysis.resolve_exit_pupil(
                    prescription, wavelength, stop_index=self.stop_index,
                    epd=self.epd, chief=(C, S_last[chief]),
                    axis_point=self.axis_point, axis_dir=self.axis_dir)
                xp_mode = 'geometric'
            if P_xp is not None:
                P_xp = np.asarray(P_xp, dtype=config.precision)

        curvature = reference_sphere_curvature(P_xp, C)
        if P_xp is None:
            delta = None
            R = np.inf
        else:
            delta = P_xp - C
            R = float(np.sqrt(np.sum(delta * delta)))
        chief_v = _analysis._filtered_chief_index(valid, chief)
        opd = hopkins_eic_closing(
            trace.P[:, valid], trace.S[:, valid], trace.OPL[:, valid],
            center=C, curvature=curvature, n_image=n_image,
            chief_index=chief_v)
        rms = float(np.sqrt(np.mean(opd * opd)))
        return dict(valid=valid, chief=chief, chief_v=chief_v, C=C, R=R,
                    curvature=curvature, delta=delta, P_xp=P_xp,
                    xp_mode=xp_mode, opd=opd, rms=rms,
                    n_image=n_image, P_last=P_last, S_last=S_last)

    def __call__(self, prescription, cache, *, return_seed=False):
        P, S, wvl = self._bundle(prescription, cache)
        trace = cache.trace(P, S, wvl)
        if return_seed:
            g = self._geometry(trace, prescription, wvl)
            return g['rms'], self._seed_from_geometry(trace, g)
        # reuse the cache's memoized exit pupil for the value path; the geometry
        # helper resolves the identical point on the adjoint path.
        P_xp = cache.exit_pupil(
            P, S, wvl, P_xp=self.P_xp,
            chief_index=self.chief_index, stop_index=self.stop_index,
            epd=self.epd, axis_point=self.axis_point, axis_dir=self.axis_dir)
        g = self._geometry(trace, prescription, wvl,
                           P_xp_override=P_xp)
        return g['rms']

    def value(self, trace, prescription, wavelength):
        return self._geometry(trace, prescription, wavelength)['rms']

    def seed(self, trace, prescription, wavelength):
        g = self._geometry(trace, prescription, wavelength)
        return self._seed_from_geometry(trace, g)

    def _seed_components_from_geometry(self, trace, g):
        # local import keeps the optimizer/design import surface free of the
        # adjoint subpackage (and dodges any import-time cycle).
        from .adjoint.primitives import (
            adj_eic_closing_full,
            adj_closest_point_on_axis,
        )

        valid = g['valid']
        chief = g['chief']
        chief_v = g['chief_v']
        C = g['C']
        R = g['R']
        curvature = g['curvature']
        delta = g['delta']
        xp_mode = g['xp_mode']
        opd = g['opd']
        rms = g['rms']
        n_image = g['n_image']
        P_last = g['P_last']
        S_last = g['S_last']

        P_bar, S_bar, L_bar = _zeros_like_trace_state(trace)
        if rms <= 1e-300:
            return P_bar, S_bar, L_bar, np.zeros(3, dtype=config.precision)

        nv = opd.shape[0]
        opd_bar = opd / (nv * rms)
        opl_total_bar = opd_bar.copy()
        opl_total_bar[chief_v] = opl_total_bar[chief_v] - opd_bar.sum()
        L_bar[valid] = opl_total_bar
        s_bar = n_image * opl_total_bar

        P_rs, S_rs, C_bar, kappa_bar = adj_eic_closing_full(
            P_last[valid], S_last[valid], C, curvature, s_bar)
        P_bar[valid] = P_bar[valid] + P_rs
        S_bar[valid] = S_bar[valid] + S_rs

        if delta is None:
            P_xp_bar = np.zeros(3, dtype=config.precision)
        else:
            # kappa = 1/R  ->  R_bar = -kappa_bar / R^2
            R_bar = -kappa_bar / (R * R)
            delta_bar = R_bar * delta / R
            C_bar = C_bar - delta_bar        # delta = P_xp - C
            P_xp_bar = delta_bar
        P_bar[chief] = P_bar[chief] + C_bar
        if xp_mode == 'geometric':
            axis_point = (np.zeros(3, dtype=config.precision)
                          if self.axis_point is None
                          else np.asarray(self.axis_point, dtype=config.precision))
            axis_dir = (np.array([0., 0., 1.], dtype=config.precision)
                        if self.axis_dir is None
                        else np.asarray(self.axis_dir, dtype=config.precision))
            P_c_bar, S_c_bar = adj_closest_point_on_axis(
                C, S_last[chief], axis_point, axis_dir, P_xp_bar)
            P_bar[chief] = P_bar[chief] + P_c_bar
            S_bar[chief] = S_bar[chief] + S_c_bar
        return P_bar, S_bar, L_bar, P_xp_bar

    def _seed_from_geometry(self, trace, g):
        return self._seed_components_from_geometry(trace, g)[:3]

    def direct_gradient(self, trace, prescription, wavelength, seeds):
        if self.P_xp is not None:
            return None
        resolved_stop = (self.stop_index if self.stop_index is not None
                         else getattr(prescription, 'stop_index', None))
        if resolved_stop is None:
            return None
        g = self._geometry(trace, prescription, wavelength)
        if g['xp_mode'] != 'paraxial':
            return None
        _, _, _, P_xp_bar = self._seed_components_from_geometry(trace, g)
        if P_xp_bar[2] == 0.0:
            return np.zeros(len(seeds), dtype=config.precision)
        from ._diff_raytrace import paraxial_exit_pupil_z_tangents

        xp_z_dot = paraxial_exit_pupil_z_tangents(
            prescription, wavelength, seeds, stop_index=resolved_stop)
        return P_xp_bar[2] * xp_z_dot


class ZernikeCoefficient(_RayMerit):
    """One coefficient of a Zernike fit to the OPD across one launch recipe.

    Useful for driving a single aberration term (e.g. drive primary
    spherical Z(4, 0) -> 0 while leaving the others free).

    """

    name = 'zernike_coefficient'

    def __init__(self, field=None, wavelength=None, sampling=None, *,
                 n, m, nms_basis, target=None, weight=1.0,
                 min=None, max=None,
                 chief_index=None,
                 axis_point=None, axis_dir=None,
                 P_xp=None, epd=None, stop_index=None,
                 normalization_radius=None, norm=True):
        super().__init__(field, wavelength, sampling, target=target,
                         weight=weight, min=min, max=max, epd=epd)
        self.n = int(n)
        self.m = int(m)
        nms_basis = [(int(nn), int(mm)) for nn, mm in nms_basis]
        if (self.n, self.m) not in nms_basis:
            raise ValueError(
                f'(n, m)=({self.n}, {self.m}) must appear in nms_basis '
                f'{nms_basis!r}; the basis sets which modes are jointly fit'
            )
        self.nms_basis = tuple(nms_basis)
        self._idx = nms_basis.index((self.n, self.m))
        self.chief_index = chief_index
        self.axis_point = axis_point
        self.axis_dir = axis_dir
        self.P_xp = P_xp
        self.stop_index = stop_index
        self.normalization_radius = normalization_radius
        self.norm = bool(norm)

    def __call__(self, prescription, cache):
        P, S, wvl = self._bundle(prescription, cache)
        P_xp = cache.exit_pupil(
            P, S, wvl, P_xp=self.P_xp,
            chief_index=self.chief_index, stop_index=self.stop_index,
            epd=self.epd, axis_point=self.axis_point, axis_dir=self.axis_dir)
        opd, x_pup, y_pup = _analysis.wavefront(
            prescription, P, S, wvl,
            chief_index=self.chief_index, P_xp=P_xp,
        )
        coefs, _ = _analysis.wavefront_zernike_fit(
            opd, x_pup, y_pup, self.nms_basis,
            normalization_radius=self.normalization_radius,
            norm=self.norm,
        )
        return float(coefs[self._idx])


class Distortion(Merit):
    """Percent distortion at one off-axis field, vs paraxial proxy."""

    name = 'distortion'

    def __init__(self, field, wavelength=None, *, epd, target=None,
                 weight=1.0, min=None, max=None, paraxial_fraction=1e-4):
        super().__init__(target=target, weight=weight, min=min, max=max)
        self.field = field
        self.wavelength = None if wavelength is None else float(wavelength)
        self.epd = float(epd)
        self.paraxial_fraction = float(paraxial_fraction)

    def __call__(self, prescription, cache):
        wvl = _resolve_wavelength(prescription, self.wavelength)
        result = _analysis.distortion(
            prescription, [self.field], wvl,
            epd=self.epd,
            paraxial_fraction=self.paraxial_fraction,
        )
        return float(result.percent[0])


class FieldCurvature(Merit):
    """X/y section focus separation at one off-axis field.

    Returns abs(x_fan_z - y_fan_z) at the requested field.

    """

    name = 'field_curvature'

    def __init__(self, field, wavelength=None, *, target=None,
                 weight=1.0, min=None, max=None):
        super().__init__(target=target, weight=weight, min=min, max=max)
        self.field = field
        self.wavelength = None if wavelength is None else float(wavelength)

    def __call__(self, prescription, cache):
        from .parabasal import parabasal_foci  # local: avoid a circular import

        wvl = _resolve_wavelength(prescription, self.wavelength)
        x_z, y_z = parabasal_foci(prescription, self.field, wvl)
        # parabasal_foci returns nan when the chief ray fails to trace; surface
        # that as a clear error (as WavefrontRMS does) rather than feeding a nan
        # residual into the optimizer, where it silently stalls the solve.
        if not (math.isfinite(x_z) and math.isfinite(y_z)):
            raise ValueError(
                'field_curvature operand: the chief ray failed to trace at '
                f'field {self.field!r}; cannot evaluate field curvature (check '
                'the starting geometry or constrain the variables).')
        return float(abs(x_z - y_z))


# ---------- Problem ----------------------------------------------------------

def _is_lensdata(model):
    """A LensData duck-types as a system with a free vector + compiler."""
    return (hasattr(model, 'pack') and hasattr(model, 'update')
            and hasattr(model, 'to_surfaces'))


class Problem:
    """Design optimization over a LensData free vector.

    Operands produce weighted residuals.  Constraints are operands with
    target/min/max bounds.  gradient='auto' uses adjoint residual Jacobians
    when available; gradient='fd' forces finite differences.

    """

    def __init__(self, lensdata, operands=None, *,
                 constraints=None, gradient='auto'):
        # OpticalSystem supplies metadata; LensData owns the free vector.
        prescription = lensdata
        if not _is_lensdata(lensdata) and _is_lensdata(getattr(lensdata,
                                                              'lens', None)):
            prescription = lensdata
            lensdata = lensdata.lens
        elif not _is_lensdata(lensdata):
            raise TypeError(
                'Problem requires a LensData or OpticalSystem (it needs '
                'pack/update/to_surfaces to own the free vector); got '
                f'{type(lensdata).__name__}.'
            )
        if gradient not in ('auto', 'fd'):
            raise ValueError(
                f"gradient must be 'auto' or 'fd', got {gradient!r}")
        self.lensdata = lensdata
        self.prescription = prescription  # duck-types as a surface sequence
        self.operands = list(operands or [])
        eqs, ineqs = _route_constraints(constraints)
        self.equality_constraints = eqs
        # list of (operand, kind, bound) with kind 'min' or 'max'
        self.inequality_constraints = ineqs
        self.gradient = gradient

    def x0(self):
        """Initial free vector from LensData.pack()."""
        return self.lensdata.pack()

    def _set_x(self, x):
        self.lensdata.update(x)

    def _operand_vector(self, operands, *, weighted):
        cache = _TraceCache(self.prescription)
        out = np.empty(len(operands), dtype=config.precision)
        for i, op in enumerate(operands):
            v = op(self.prescription, cache)
            r = v - op.target
            if weighted:
                r = op.weight * r
            out[i] = r
        return out, cache

    def residuals(self, x, return_cache=False):
        """Per-operand weighted residual vector [w_i * (op_i - target_i)]."""
        self._set_x(x)
        out, cache = self._operand_vector(self.operands, weighted=True)
        if return_cache:
            return out, cache
        return out

    def equalities(self, x, return_cache=False):
        """Unweighted equality constraint vector, op_i - target_i == 0."""
        self._set_x(x)
        out, cache = self._operand_vector(
            self.equality_constraints, weighted=False,
        )
        if return_cache:
            return out, cache
        return out

    def inequalities(self, x, return_cache=False):
        """Unweighted inequality constraint vector, g_i(x) >= 0.

        A min-bounded constraint contributes value - min; a max-bounded one
        contributes max - value.

        """
        self._set_x(x)
        cache = _TraceCache(self.prescription)
        out = np.empty(len(self.inequality_constraints),
                       dtype=config.precision)
        for i, (op, kind, bound) in enumerate(self.inequality_constraints):
            v = op(self.prescription, cache)
            out[i] = (v - bound) if kind == 'min' else (bound - v)
        if return_cache:
            return out, cache
        return out

    def solve(self, x0=None, **kwargs):
        """Run constrained damped least squares and update LensData to result.

        Extra equality/inequality kwargs are combined with stored constraints.
        A failed solver result still updates the lens and emits a warning.

        """
        eq = _combine_constraints(
            self.equalities,
            kwargs.pop('equality_constraints', None),
        )
        ineq = _combine_constraints(
            self.inequalities,
            kwargs.pop('inequality_constraints', None),
        )
        result = damped_least_squares(
            self,
            x0=x0,
            equality_constraints=eq,
            inequality_constraints=ineq,
            **kwargs,
        )
        self._set_x(result.x)
        if not result.success:
            warnings.warn(
                f'optimization did not converge: {result.message}; the lens '
                'was updated to the best iterate anyway',
                stacklevel=2,
            )
        return result

    def _eval_merit(self, prescription):
        """Sum of squared weighted residuals on the given prescription."""
        cache = _TraceCache(prescription)
        total = 0.0
        for op in self.operands:
            v = op(prescription, cache)
            r = op.weight * (v - op.target)
            total = total + r * r
        return total

    def merit(self, x):
        """Scalar sum of squared weighted residuals."""
        self._set_x(x)
        return float(self._eval_merit(self.prescription))

    def _free_slot_seeds(self):
        """DiffSeed objects for current free DOFs."""
        from .tolerance import Perturbation
        from ._diff_raytrace import seeds_from_perturbations

        ld = self.lensdata
        perturbations = []
        for slot in ld.spec.free_slots():
            group, r, off = slot
            nominal = float(ld.spec.get_value(slot))
            perturbations.append(Perturbation(
                ld, slot, None, nominal, step=0.0, name=f'{group}{r}.{off}'))
        return seeds_from_perturbations(perturbations)

    def residual_jacobian(self, x):
        """Adjoint Jacobian of the weighted residual vector at x, or None.

        Operand launch recipes are evaluated at x, then differentiated as fixed
        ray bundles.

        """
        if self.gradient != 'auto':
            return None
        ops = self.operands
        if not ops:
            return None
        for op in ops:
            if not op.seedable:
                return None
        self._set_x(x)
        try:
            seeds = self._free_slot_seeds()
        except NotImplementedError:
            return None
        if not seeds:
            return None
        # local import: design stays import-light, mirroring the seed path
        from .adjoint.tolerance_analysis import multi_objective_sensitivity

        # Group operands that share the same launch bundle.
        cache = _TraceCache(self.prescription)
        bundles = []
        groups = {}
        for i, op in enumerate(ops):
            bundle = op._bundle(self.prescription, cache)
            if bundle is None:
                return None
            bundles.append(bundle)
            P, S, wvl = bundle
            groups.setdefault((id(P), id(S), float(wvl)), []).append(i)
        J = np.zeros((len(ops), len(seeds)), dtype=config.precision)
        for idxs in groups.values():
            P, S, wvl = bundles[idxs[0]]
            result = multi_objective_sensitivity(
                self.prescription, P, S, wvl,
                seeds, [ops[i] for i in idxs])
            for row, i in zip(result.jacobian, idxs):
                J[i] = ops[i].weight * row
        return J

    def jacobian(self, x, method='auto', step=1e-6):
        """Gradient of scalar merit with respect to x."""
        if method == 'auto':
            J = self.residual_jacobian(x)
            if J is not None:
                r, _ = self._operand_vector(self.operands, weighted=True)
                return 2.0 * (J.T @ r)
            method = 'fd'
        self._set_x(x)
        return _merit_jacobian_free(
            self.lensdata, lambda: self._eval_merit(self.prescription),
            method=method, step=step)


def _as_operand_list(operands):
    if operands is None:
        return []
    if isinstance(operands, Merit):
        return [operands]
    return list(operands)


def _route_constraints(constraints):
    """Split a constraints list into equality operands and inequality terms.

    target= (or no bound at all) makes an equality; min= / max= make
    inequality terms (operand, kind, bound) in the g(x) >= 0 convention.
    target together with min/max raises.

    """
    eqs = []
    ineqs = []
    for op in _as_operand_list(constraints):
        mn = getattr(op, 'min', None)
        mx = getattr(op, 'max', None)
        if mn is None and mx is None:
            eqs.append(op)
            continue
        if getattr(op, '_target_set', False):
            raise ValueError(
                f'constraint {getattr(op, "name", type(op).__name__)} mixes '
                'target= with min=/max=; use target= alone for an equality '
                'or min=/max= alone for inequalities'
            )
        if mn is not None:
            ineqs.append((op, 'min', float(mn)))
        if mx is not None:
            ineqs.append((op, 'max', float(mx)))
    return eqs, ineqs


def _combine_constraints(primary, extra):
    if extra is None:
        return primary
    if callable(extra):
        return (primary, extra)
    return (primary, *tuple(extra))


# ---------- Goal factory -----------------------------------------------------

_GOAL_OPERANDS = {
    'spot': RmsSpotRadius,
    'wavefront': WavefrontRMS,
}


def build_problem(system, goal='spot', *, sampling=None, fields=None,
                  wavelengths=None, constraints=None):
    """Assemble a Problem from goal items fanned over fields and wavelengths."""
    items = list(goal) if isinstance(goal, (list, tuple)) else [goal]

    resolve_field = getattr(system, 'field', None)
    if fields is not None:
        flds = [resolve_field(f) if callable(resolve_field) else f
                for f in fields]
    else:
        flds = list(getattr(system, 'fields', None) or [])
    if not flds:
        flds = [None]

    if wavelengths is not None:
        wvls = [float(w) for w in wavelengths]
        wts = [1.0] * len(wvls)
    else:
        wvls = [float(w) for w in getattr(system, 'wavelengths', [])]
        wts = [float(w) for w in getattr(system, 'weights', [])]
        if len(wts) != len(wvls):
            wts = [1.0] * len(wvls)
    if not wvls:
        wvls = [None]
        wts = [1.0]

    ops = []
    for item in items:
        if isinstance(item, str):
            cls = _GOAL_OPERANDS.get(item)
            if cls is None:
                raise ValueError(
                    f'unknown goal {item!r}; known goals: '
                    f'{sorted(_GOAL_OPERANDS)}'
                )
        elif isinstance(item, type) and issubclass(item, Merit):
            cls = item
        elif isinstance(item, Merit):
            ops.append(item)
            continue
        elif callable(item):
            ops.append(_CallableMerit(item))
            continue
        else:
            raise TypeError(
                'goal items must be a string, a Merit subclass or instance, '
                f'or a callable; got {type(item).__name__}'
            )
        recipe_class = (_class_accepts_kw(cls, 'field')
                        or _class_accepts_kw(cls, 'sampling'))
        wavelength_class = _class_accepts_kw(cls, 'wavelength')
        weight_class = _class_accepts_kw(cls, 'weight')
        if recipe_class:
            for f in flds:
                for w, wt in zip(wvls, wts):
                    kwargs = {}
                    if _class_accepts_kw(cls, 'field'):
                        kwargs['field'] = f
                    if wavelength_class:
                        kwargs['wavelength'] = w
                    if _class_accepts_kw(cls, 'sampling'):
                        kwargs['sampling'] = sampling
                    if weight_class:
                        kwargs['weight'] = wt
                    ops.append(cls(**kwargs))
        elif wavelength_class:
            for w, wt in zip(wvls, wts):
                kwargs = {'wavelength': w}
                if weight_class:
                    kwargs['weight'] = wt
                ops.append(cls(**kwargs))
        else:
            kwargs = {}
            if weight_class:
                kwargs['weight'] = 1.0
            ops.append(cls(**kwargs))
    return Problem(system, ops, constraints=constraints)
