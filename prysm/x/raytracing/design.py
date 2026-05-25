"""Design layer: Variables, Operands, and the ``Problem`` class.

A thin layer over the kernel that turns a sequential prescription into a
design loop:

- **Variables** are ``(getter, setter)`` callable pairs.  The typed
  factories below (``curvature_of``, ``radius_of``, ``kappa_of``,
  ``coef_of``, ``position_of``, ``thickness_after``) cover the common
  cases by name; arbitrary scalars can always be wrapped manually.
  Variables mutate the prescription in place — the sag_and_normal closures on
  each surface read ``params`` per call so perturbations are picked up
  on the next trace without rebuilding.
- **Operands** are vanilla classes with
  ``__call__(prescription, cache) -> scalar`` plus a ``target`` and
  ``weight``.  Trace-based operands ask the per-call cache for their
  ray-trace; paraxial operands compute directly from the prescription.
- **Problem** wraps a prescription + variables + operands.
  ``residuals(x)`` returns the per-operand weighted residual vector for
  ``scipy.optimize.least_squares``; ``merit(x)`` returns the scalar sum
  of squared residuals for ``scipy.optimize.minimize``; ``jacobian(x)``
  reuses ``sensitivity.merit_jacobian`` to compute the gradient of the
  scalar merit (FD or torch autograd).

A trace cache is created fresh on each ``residuals`` / ``merit`` call so
multiple operands sharing the same launch bundle and wavelength evaluate
the trace once per call.  The cache lifetime is one merit call, not
global; cache hits are based on object identity (``id(P)``, ``id(S)``,
and the float wavelength), so operands should hold references to the
same launch arrays rather than copies.

"""

from prysm.conf import config
from prysm.mathops import np
from prysm.x.optym.least_squares import (  # NOQA - re-export for users
    DampedLeastSquares,
    DampedLeastSquaresResult,
    damped_least_squares,
)

from .spencer_and_murty import raytrace
from .sensitivity import merit_jacobian_free as _merit_jacobian_free
from .opt import rms_spot_radius, opd_from_raytrace, xp_reference_sphere
from .paraxial import (
    effective_focal_length,
    back_focal_length,
    paraxial_image_distance,
)
from . import analysis as _analysis


# ---------- Trace cache ------------------------------------------------------

class _TraceCache:
    """Per-merit-call ray-trace cache keyed by ``(id(P), id(S), wvl)``.

    Two operands sharing the same launch bundle and wavelength evaluate
    the trace once per merit call.  Identity-based keying assumes the
    user retains the same ``P``, ``S`` array objects across operands
    (the operand constructors capture references).

    Not thread-safe; not reusable across merit calls.

    """

    __slots__ = ('_prescription', '_cache', '_n_traces')

    def __init__(self, prescription):
        self._prescription = prescription
        self._cache = {}
        self._n_traces = 0

    def trace(self, P, S, wavelength, n_ambient=1.0):
        key = (id(P), id(S), float(wavelength), float(n_ambient))
        cached = self._cache.get(key)
        if cached is None:
            cached = raytrace(self._prescription, P, S, wavelength,
                              n_ambient=n_ambient)
            self._cache[key] = cached
            self._n_traces += 1
        return cached

    @property
    def n_traces(self):
        """Number of underlying ``raytrace`` calls made (cache misses)."""
        return self._n_traces


# ---------- Operands ---------------------------------------------------------

class _OperandBase:
    """Shared target/weight/n_ambient plumbing for operand classes.

    Subclasses call super().__init__(target, weight, n_ambient) and then
    set their own task-specific attributes.

    """

    def __init__(self, target=0.0, weight=1.0, n_ambient=1.0):
        self.target = float(target)
        self.weight = float(weight)
        self.n_ambient = float(n_ambient)


class RmsSpotRadius(_OperandBase):
    """Weighted RMS spot radius at the image plane for one launch bundle."""

    def __init__(self, P, S, wavelength, target=0.0, weight=1.0,
                 n_ambient=1.0):
        super().__init__(target=target, weight=weight, n_ambient=n_ambient)
        self.P = P
        self.S = S
        self.wavelength = float(wavelength)

    def __call__(self, prescription, cache):
        trace = cache.trace(self.P, self.S, self.wavelength,
                            n_ambient=self.n_ambient)
        return rms_spot_radius(trace.P[-1], status=trace.status)


class RayHeightAt(_OperandBase):
    """Position of one ray along one Cartesian axis at one prescription
    surface.

    Useful for chief / marginal ray boundary conditions and for
    constraining specific image-plane points (e.g.,
    surface_index=-1, axis=1 is the y-position at the image plane).

    """

    def __init__(self, P, S, wavelength, surface_index, axis,
                 target=0.0, weight=1.0, ray_index=0, n_ambient=1.0):
        super().__init__(target=target, weight=weight, n_ambient=n_ambient)
        self.P = P
        self.S = S
        self.wavelength = float(wavelength)
        self.surface_index = int(surface_index)
        self.axis = int(axis)
        self.ray_index = int(ray_index)

    def __call__(self, prescription, cache):
        trace = cache.trace(self.P, self.S, self.wavelength,
                            n_ambient=self.n_ambient)
        return float(trace.P[self.surface_index, self.ray_index, self.axis])


class Boresight(_OperandBase):
    """Centroid distance from a target point at the final surface, for
    one launch bundle.  Use to enforce a chief-ray landing point.

    """

    def __init__(self, P, S, wavelength, target_xy=(0.0, 0.0), weight=1.0,
                 n_ambient=1.0):
        # boresight residual is the distance to target_xy; target stays 0
        super().__init__(target=0.0, weight=weight, n_ambient=n_ambient)
        self.P = P
        self.S = S
        self.wavelength = float(wavelength)
        self.target_xy = (float(target_xy[0]), float(target_xy[1]))

    def __call__(self, prescription, cache):
        trace = cache.trace(self.P, self.S, self.wavelength,
                            n_ambient=self.n_ambient)
        valid = trace.status.imag == 0
        Pf = trace.P[-1]
        if valid.any():
            mean = Pf[valid, :2].mean(axis=0)
        else:
            mean = Pf[:, :2].mean(axis=0)
        dx = mean[0] - self.target_xy[0]
        dy = mean[1] - self.target_xy[1]
        return float(np.sqrt(dx * dx + dy * dy))


class EFL(_OperandBase):
    """Effective focal length (paraxial ABCD)."""

    def __init__(self, wavelength, target=0.0, weight=1.0, n_ambient=1.0):
        super().__init__(target=target, weight=weight, n_ambient=n_ambient)
        self.wavelength = float(wavelength)

    def __call__(self, prescription, cache):
        return float(effective_focal_length(prescription, wvl=self.wavelength,
                                            n_ambient=self.n_ambient))


class BFL(_OperandBase):
    """Back focal length (last powered surface vertex to rear focal point)."""

    def __init__(self, wavelength, target=0.0, weight=1.0, n_ambient=1.0):
        super().__init__(target=target, weight=weight, n_ambient=n_ambient)
        self.wavelength = float(wavelength)

    def __call__(self, prescription, cache):
        return float(back_focal_length(prescription, wvl=self.wavelength,
                                       n_ambient=self.n_ambient))


class ParaxialImageDistance(_OperandBase):
    """Signed distance from the last surface vertex to the paraxial image
    plane (collimated on-axis input)."""

    def __init__(self, wavelength, target=0.0, weight=1.0, n_ambient=1.0):
        super().__init__(target=target, weight=weight, n_ambient=n_ambient)
        self.wavelength = float(wavelength)

    def __call__(self, prescription, cache):
        return float(paraxial_image_distance(prescription, wvl=self.wavelength,
                                             n_ambient=self.n_ambient))


class WavefrontRMS(_OperandBase):
    """RMS of the OPD across one launch bundle.

    Composes analysis.wavefront — traces, computes OPD on the chief-ray
    reference sphere, returns sqrt(mean(opd**2)) of valid rays.  When the
    chief-ray is included in the bundle (the standard convention), its
    contribution is identically zero and does not bias the RMS.

    """

    def __init__(self, P, S, wavelength, target=0.0, weight=1.0,
                 n_ambient=1.0, chief_index=None,
                 axis_point=None, axis_dir=None):
        super().__init__(target=target, weight=weight, n_ambient=n_ambient)
        self.P = P
        self.S = S
        self.wavelength = float(wavelength)
        self.chief_index = chief_index
        self.axis_point = axis_point
        self.axis_dir = axis_dir

    def __call__(self, prescription, cache):
        opd, _, _ = _analysis.wavefront(
            prescription, self.P, self.S, self.wavelength,
            n_ambient=self.n_ambient, chief_index=self.chief_index,
            axis_point=self.axis_point, axis_dir=self.axis_dir,
        )
        return float(np.sqrt(np.mean(opd * opd)))


class ZernikeCoefficient(_OperandBase):
    """One coefficient of a Zernike fit to the OPD across one launch bundle.

    Useful for driving a single aberration term (e.g. drive primary
    spherical Z(4, 0) -> 0 while leaving the others free).

    """

    def __init__(self, P, S, wavelength, n, m, *,
                 nms_basis, target=0.0, weight=1.0,
                 n_ambient=1.0, chief_index=None,
                 axis_point=None, axis_dir=None,
                 normalization_radius=None, norm=True):
        super().__init__(target=target, weight=weight, n_ambient=n_ambient)
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
        self.normalization_radius = normalization_radius
        self.norm = bool(norm)

    def __call__(self, prescription, cache):
        opd, x_pup, y_pup = _analysis.wavefront(
            prescription, self.P, self.S, self.wavelength,
            n_ambient=self.n_ambient, chief_index=self.chief_index,
            axis_point=self.axis_point, axis_dir=self.axis_dir,
        )
        coefs, _ = _analysis.wavefront_zernike_fit(
            opd, x_pup, y_pup, self.nms_basis,
            normalization_radius=self.normalization_radius,
            norm=self.norm,
        )
        return float(coefs[self._idx])


class Distortion(_OperandBase):
    """Percent distortion at one off-axis field, vs paraxial proxy."""

    def __init__(self, field, wavelength, *, epd, target=0.0, weight=1.0,
                 n_ambient=1.0, paraxial_fraction=1e-4):
        super().__init__(target=target, weight=weight, n_ambient=n_ambient)
        self.field = field
        self.wavelength = float(wavelength)
        self.epd = float(epd)
        self.paraxial_fraction = float(paraxial_fraction)

    def __call__(self, prescription, cache):
        _, _, percent = _analysis.distortion(
            prescription, [self.field], self.wavelength,
            epd=self.epd, n_ambient=self.n_ambient,
            paraxial_fraction=self.paraxial_fraction,
        )
        return float(percent[0])


class FieldCurvature(_OperandBase):
    """Sagittal-tangential focus separation at one off-axis field.

    Returns abs(sagittal_z - tangential_z) at the requested field; driving
    this toward zero flattens astigmatic field curvature locally.  For a
    full Petzval-flat optimization, sum these across multiple fields.

    """

    def __init__(self, field, wavelength, *, epd, target=0.0, weight=1.0,
                 n_ambient=1.0, marginal_fraction=0.7):
        super().__init__(target=target, weight=weight, n_ambient=n_ambient)
        self.field = field
        self.wavelength = float(wavelength)
        self.epd = float(epd)
        self.marginal_fraction = float(marginal_fraction)

    def __call__(self, prescription, cache):
        sag_z, tan_z = _analysis.field_curvature(
            prescription, [self.field], self.wavelength,
            epd=self.epd, n_ambient=self.n_ambient,
            marginal_fraction=self.marginal_fraction,
        )
        return float(abs(sag_z[0] - tan_z[0]))


# ---------- Problem ----------------------------------------------------------

def _is_lensdata(model):
    """A LensData duck-types as a system with a free vector + compiler."""
    return (hasattr(model, 'pack') and hasattr(model, 'update')
            and hasattr(model, 'to_surfaces'))


class Problem:
    """A design-optimization problem.

    Two modes, selected by the first argument:

    - **LensData** — ``Problem(lensdata, operands)``.  The free vector is the
      LensData's packed DOFs; ``x`` is scattered back with ``lensdata.update``,
      the system recompiled, and the operands evaluated against the compiled
      surfaces.  This is the canonical path.
    - **Legacy surface list** — ``Problem(surfaces, variables, operands)``.
      ``variables`` are ``(getter, setter)`` pairs that mutate the surfaces in
      place.  Retained for the (deprecated) Prescription design session.

    Methods: :meth:`x0`, :meth:`residuals`, :meth:`merit`, :meth:`jacobian`.

    """

    def __init__(self, model, variables=None, operands=None):
        if _is_lensdata(model):
            self.lensdata = model
            self.prescription = model  # duck-types as a surface sequence
            ops = operands if operands is not None else variables
            self.operands = list(ops or [])
            self.variables = None
        else:
            self.lensdata = None
            self.prescription = model
            self.variables = list(variables or [])
            self.operands = list(operands or [])

    def x0(self):
        """Initial parameter vector from the current system state."""
        if self.lensdata is not None:
            return self.lensdata.pack()
        return np.array([float(g()) for g, _ in self.variables])

    def _set_x(self, x):
        if self.lensdata is not None:
            self.lensdata.update(x)
            return
        for (_, s), v in zip(self.variables, x):
            s(float(v))

    def residuals(self, x, return_cache=False):
        """Per-operand weighted residual vector [w_i * (op_i - target_i)].

        Suitable for scipy.optimize.least_squares.  When return_cache=True
        also returns the _TraceCache used (for introspection — e.g.,
        counting trace calls in tests).

        """
        self._set_x(x)
        cache = _TraceCache(self.prescription)
        out = np.empty(len(self.operands), dtype=config.precision)
        for i, op in enumerate(self.operands):
            v = op(self.prescription, cache)
            out[i] = op.weight * (v - op.target)
        if return_cache:
            return out, cache
        return out

    def _eval_merit(self, prescription):
        """Sum of squared weighted residuals on the given prescription.

        Does not set parameters; callers responsible for that.  Shared
        by merit() (which sets x first) and jacobian() (which delegates
        parameter setting to merit_jacobian).

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
        if self.lensdata is None:
            raise TypeError(
                'Problem.jacobian requires a LensData model (the free vector); '
                'build the problem as Problem(lensdata, operands) and mark DOFs '
                'with lensdata.vary(...)'
            )
        self._set_x(x)
        return _merit_jacobian_free(
            self.lensdata, lambda: self._eval_merit(self.prescription),
            method=method, step=step)
