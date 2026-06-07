"""Design operands and optimization problems for raytracing systems."""

from prysm.conf import config
from prysm.mathops import np
from prysm.x.optym.least_squares import (  # NOQA - re-export for users
    DampedLeastSquares,
    DampedLeastSquaresResult,
    damped_least_squares,
)

from .spencer_and_murty import raytrace, valid_mask
from .sensitivity import merit_jacobian_free as _merit_jacobian_free
from .opt import rms_spot_radius, _pupil_center_chief_index, opd_from_raytrace_eic
from .paraxial import (
    effective_focal_length,
    back_focal_length,
    paraxial_image_distance,
    first_order,
)
from . import analysis as _analysis
from ._meta import object_space_index, image_space_index


# ---------- Trace cache ------------------------------------------------------

class _TraceCache:
    """Per-merit-call raytrace cache keyed by `(id(P), id(S), wvl)`."""

    __slots__ = ('_prescription', '_cache', '_n_traces', '_xp_cache')

    def __init__(self, prescription):
        self._prescription = prescription
        self._cache = {}
        self._xp_cache = {}
        self._n_traces = 0

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
        """Exit-pupil reference point for an operand bundle, resolved once.

        Honors an explicit P_xp; otherwise resolves via analysis.resolve_exit_pupil
        and memoizes per (bundle, wavelength, stop) for the merit call.  When no
        stop is resolvable the geometric route reuses this bundle's own cached
        chief ray, matching the differential trace (wavefront_with_tangents).
        """
        if P_xp is not None:
            return np.asarray(P_xp)
        key = (id(P), id(S), float(wavelength), stop_index)
        cached = self._xp_cache.get(key)
        if cached is None:
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

class Merit:
    """Shared target/weight plumbing and adjoint contract for merit terms.

    Subclasses call super().__init__(target, weight) and then set their own
    task-specific attributes.  Object-space media come from the prescription's
    object surface material, resolved per merit call.

    """

    name = 'merit'

    def __init__(self, target=0.0, weight=1.0):
        self.target = float(target)
        self.weight = float(weight)

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


class RmsSpotRadius(Merit):
    """Weighted RMS spot radius at the image plane for one launch bundle."""

    name = 'rms_spot_radius'

    def __init__(self, P, S, wavelength, target=0.0, weight=1.0):
        super().__init__(target=target, weight=weight)
        self.P = P
        self.S = S
        self.wavelength = float(wavelength)

    def __call__(self, prescription, cache):
        trace = cache.trace(self.P, self.S, self.wavelength)
        return self.value(trace, prescription, self.wavelength)

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


class RayHeightAt(Merit):
    """Position of one ray along one Cartesian axis at one prescription
    surface.

    Useful for chief / marginal ray boundary conditions and for
    constraining specific image-plane points (e.g.,
    surface_index=-1, axis=1 is the y-position at the image plane).

    """

    def __init__(self, P, S, wavelength, surface_index, axis,
                 target=0.0, weight=1.0, ray_index=0):
        super().__init__(target=target, weight=weight)
        self.P = P
        self.S = S
        self.wavelength = float(wavelength)
        self.surface_index = int(surface_index)
        self.axis = int(axis)
        self.ray_index = int(ray_index)

    def __call__(self, prescription, cache):
        trace = cache.trace(self.P, self.S, self.wavelength)
        return float(trace.P[self.surface_index, self.ray_index, self.axis])


class Boresight(Merit):
    """Centroid distance from a target point at the final surface, for
    one launch bundle.  Use to enforce a chief-ray landing point.

    """

    def __init__(self, P, S, wavelength, target_xy=(0.0, 0.0), weight=1.0):
        # boresight residual is the distance to target_xy; target stays 0
        super().__init__(target=0.0, weight=weight)
        self.P = P
        self.S = S
        self.wavelength = float(wavelength)
        self.target_xy = (float(target_xy[0]), float(target_xy[1]))

    def __call__(self, prescription, cache):
        trace = cache.trace(self.P, self.S, self.wavelength)
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
    """Effective focal length (paraxial ABCD)."""

    def __init__(self, wavelength, target=0.0, weight=1.0):
        super().__init__(target=target, weight=weight)
        self.wavelength = float(wavelength)

    def __call__(self, prescription, cache):
        return float(effective_focal_length(prescription, wvl=self.wavelength))


class BFL(Merit):
    """Back focal length (last powered surface vertex to rear focal point)."""

    def __init__(self, wavelength, target=0.0, weight=1.0):
        super().__init__(target=target, weight=weight)
        self.wavelength = float(wavelength)

    def __call__(self, prescription, cache):
        return float(back_focal_length(prescription, wvl=self.wavelength))


class ParaxialImageDistance(Merit):
    """Signed distance from the last surface vertex to the paraxial image
    plane (collimated on-axis input)."""

    def __init__(self, wavelength, target=0.0, weight=1.0):
        super().__init__(target=target, weight=weight)
        self.wavelength = float(wavelength)

    def __call__(self, prescription, cache):
        return float(paraxial_image_distance(prescription, wvl=self.wavelength))


class WavefrontRMS(Merit):
    """RMS of OPD on the chief-ray reference sphere."""

    name = 'rms_wfe'

    def __init__(self, P, S, wavelength, target=0.0, weight=1.0,
                 chief_index=None,
                 axis_point=None, axis_dir=None, P_xp=None,
                 epd=None, stop_index=None):
        super().__init__(target=target, weight=weight)
        self.P = P
        self.S = S
        self.wavelength = float(wavelength)
        self.chief_index = chief_index
        self.axis_point = axis_point
        self.axis_dir = axis_dir
        self.P_xp = P_xp
        self.epd = epd
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
                    fo = first_order(prescription, wvl=wavelength, epd=self.epd,
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
                        raise ValueError(
                            'paraxial exit pupil is at infinity; pass P_xp '
                            'explicitly for a planar or finite reference'
                        )
                    P_xp = np.array([0.0, 0.0, float(fo.xp_z)],
                                    dtype=config.precision)
                    xp_mode = 'paraxial'
            else:
                P_xp = _analysis.resolve_exit_pupil(
                    prescription, wavelength, stop_index=self.stop_index,
                    epd=self.epd, chief=(C, S_last[chief]),
                    axis_point=self.axis_point, axis_dir=self.axis_dir)
                xp_mode = 'geometric'
            P_xp = np.asarray(P_xp, dtype=config.precision)

        delta = P_xp - C
        R = float(np.sqrt(np.sum(delta * delta)))
        if R <= 1e-12:
            raise ValueError(
                'reference-sphere radius is degenerate; pass a nondegenerate P_xp'
            )
        chief_v = _analysis._filtered_chief_index(valid, chief)
        opd = opd_from_raytrace_eic(
            trace.P[:, valid], trace.S[:, valid], trace.OPL[:, valid],
            P_img=C, P_xp=P_xp, n_image=n_image, chief_index=chief_v,
            infinite_threshold=np.inf)
        rms = float(np.sqrt(np.mean(opd * opd)))
        return dict(valid=valid, chief=chief, chief_v=chief_v, C=C, R=R,
                    delta=delta, P_xp=P_xp, xp_mode=xp_mode, opd=opd, rms=rms,
                    n_image=n_image, P_last=P_last, S_last=S_last)

    def __call__(self, prescription, cache, *, return_seed=False):
        trace = cache.trace(self.P, self.S, self.wavelength)
        if return_seed:
            g = self._geometry(trace, prescription, self.wavelength)
            return g['rms'], self._seed_from_geometry(trace, g)
        # reuse the cache's memoized exit pupil for the value path; the geometry
        # helper resolves the identical point on the adjoint path.
        P_xp = cache.exit_pupil(
            self.P, self.S, self.wavelength, P_xp=self.P_xp,
            chief_index=self.chief_index, stop_index=self.stop_index,
            epd=self.epd, axis_point=self.axis_point, axis_dir=self.axis_dir)
        g = self._geometry(trace, prescription, self.wavelength,
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
            adj_intersect_reference_sphere_full,
            adj_closest_point_on_axis,
        )

        valid = g['valid']
        chief = g['chief']
        chief_v = g['chief_v']
        C = g['C']
        R = g['R']
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
        t_bar = n_image * opl_total_bar

        P_rs, S_rs, C_bar, R_bar = adj_intersect_reference_sphere_full(
            P_last[valid], S_last[valid], C, R, t_bar)
        P_bar[valid] = P_bar[valid] + P_rs
        S_bar[valid] = S_bar[valid] + S_rs

        delta_bar = R_bar * delta / R
        C_bar = C_bar - delta_bar            # delta = P_xp - C
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


class ZernikeCoefficient(Merit):
    """One coefficient of a Zernike fit to the OPD across one launch bundle.

    Useful for driving a single aberration term (e.g. drive primary
    spherical Z(4, 0) -> 0 while leaving the others free).

    """

    def __init__(self, P, S, wavelength, n, m, *,
                 nms_basis, target=0.0, weight=1.0,
                 chief_index=None,
                 axis_point=None, axis_dir=None,
                 P_xp=None, epd=None, stop_index=None,
                 normalization_radius=None, norm=True):
        super().__init__(target=target, weight=weight)
        self.P = P
        self.S = S
        self.wavelength = float(wavelength)
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
        self.epd = epd
        self.stop_index = stop_index
        self.normalization_radius = normalization_radius
        self.norm = bool(norm)

    def __call__(self, prescription, cache):
        P_xp = cache.exit_pupil(
            self.P, self.S, self.wavelength, P_xp=self.P_xp,
            chief_index=self.chief_index, stop_index=self.stop_index,
            epd=self.epd, axis_point=self.axis_point, axis_dir=self.axis_dir)
        opd, x_pup, y_pup = _analysis.wavefront(
            prescription, self.P, self.S, self.wavelength,
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

    def __init__(self, field, wavelength, *, epd, target=0.0, weight=1.0,
                 paraxial_fraction=1e-4):
        super().__init__(target=target, weight=weight)
        self.field = field
        self.wavelength = float(wavelength)
        self.epd = float(epd)
        self.paraxial_fraction = float(paraxial_fraction)

    def __call__(self, prescription, cache):
        _, _, percent = _analysis.distortion(
            prescription, [self.field], self.wavelength,
            epd=self.epd,
            paraxial_fraction=self.paraxial_fraction,
        )
        return float(percent[0])


class FieldCurvature(Merit):
    """X/y fan focus separation at one off-axis field.

    Returns abs(x_fan_z - y_fan_z) at the requested field.  For pure-y fields
    on an axisymmetric system this is the classical sagittal-tangential focus
    separation; otherwise it is the local x/y fan separation.  For a full
    Petzval-flat optimization, sum these across multiple fields.

    """

    def __init__(self, field, wavelength, *, epd, target=0.0, weight=1.0,
                 marginal_fraction=0.7):
        super().__init__(target=target, weight=weight)
        self.field = field
        self.wavelength = float(wavelength)
        self.epd = float(epd)
        self.marginal_fraction = float(marginal_fraction)

    def __call__(self, prescription, cache):
        sag_z, tan_z = _analysis.field_curvature(
            prescription, [self.field], self.wavelength,
            epd=self.epd,
            marginal_fraction=self.marginal_fraction,
        )
        return float(abs(sag_z[0] - tan_z[0]))


# ---------- Problem ----------------------------------------------------------

def _is_lensdata(model):
    """A LensData duck-types as a system with a free vector + compiler."""
    return (hasattr(model, 'pack') and hasattr(model, 'update')
            and hasattr(model, 'to_surfaces'))


class Problem:
    """A design-optimization problem over a LensData's free vector.

    `Problem(lensdata, operands, equality_constraints=None,
    inequality_constraints=None)`.  The free vector is the LensData's packed
    DOFs (mark them with `lensdata.vary(...)`); `x` is scattered back with
    `lensdata.update`, the system recompiled, and the operands evaluated
    against the compiled surfaces.

    Objective operands are weighted residual terms.  Equality and inequality
    operands are hard constraints evaluated as `operand - target`, ignoring
    `weight`; inequalities use the optym convention g(x) >= 0.

    Methods: x0, residuals, equalities,
    inequalities, solve, merit, jacobian.

    """

    def __init__(self, lensdata, operands=None, *,
                 equality_constraints=None, inequality_constraints=None,
                 constraints=None):
        # accept either a LensData (the free-vector owner) or an OpticalSystem
        # wrapping one; the system is used as the prescription so operands see
        # its aperture / object medium, while pack/update act on the lens.
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
        if constraints is not None:
            if equality_constraints is not None:
                raise ValueError(
                    'use either constraints or equality_constraints, not both'
                )
            equality_constraints = constraints
        self.lensdata = lensdata
        self.prescription = prescription  # duck-types as a surface sequence
        self.operands = list(operands or [])
        self.equality_constraints = _as_operand_list(equality_constraints)
        self.inequality_constraints = _as_operand_list(inequality_constraints)

    def x0(self):
        """Initial parameter vector — the LensData's packed free vector."""
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
        """Per-operand weighted residual vector [w_i * (op_i - target_i)].

        This is the least-squares objective vector.  Hard constraints live in
        equalities() and inequalities(), so use solve() or
        damped_least_squares(..., equality_constraints=prob.equalities) for
        exact constraint solves.  When return_cache=True also returns the
        _TraceCache used (for introspection — e.g., counting trace calls in
        tests).

        """
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
        """Unweighted inequality constraint vector, op_i - target_i >= 0."""
        self._set_x(x)
        out, cache = self._operand_vector(
            self.inequality_constraints, weighted=False,
        )
        if return_cache:
            return out, cache
        return out

    def solve(self, x0=None, **kwargs):
        """Run constrained damped least squares and update LensData to result.

        Keyword arguments are forwarded to damped_least_squares.  Explicit
        equality_constraints or inequality_constraints keywords are combined
        with the hard operands stored on this Problem.

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
        return result

    def _eval_merit(self, prescription):
        """Sum of squared weighted residuals on the given prescription.

        Does not set parameters; callers responsible for that.  Shared
        by merit() (which sets x first) and jacobian() (which delegates
        parameter setting to merit_jacobian_free).

        """
        cache = _TraceCache(prescription)
        total = 0.0
        for op in self.operands:
            v = op(prescription, cache)
            r = op.weight * (v - op.target)
            total = total + r * r
        return total

    def merit(self, x):
        """Scalar sum of squared weighted residuals.

        Suitable for scipy.optimize.minimize.

        """
        self._set_x(x)
        return float(self._eval_merit(self.prescription))

    def jacobian(self, x, method='fd', step=1e-6):
        """Gradient of the scalar merit w.r.t. x (length n_params).

        method='fd' (default) uses central differences; method='autograd'
        requires the prysm backend to be torch.

        """
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


def _combine_constraints(primary, extra):
    if extra is None:
        return primary
    if callable(extra):
        return (primary, extra)
    return (primary, *tuple(extra))
