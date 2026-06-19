"""System-level metadata wrapper for LensData."""

import math
import warnings

from prysm.conf import config
from prysm.mathops import np

from .paraxial import (
    effective_focal_length,
    entrance_pupil_z,
    system_matrix,
)
from ._cache import StateCache


# aperture modes
EPD = 'EPD'
FNO_IMAGE = 'FNO_IMAGE'
FNO_OBJECT = 'FNO_OBJECT'
NA_IMAGE = 'NA_IMAGE'
NA_OBJECT = 'NA_OBJECT'

_APERTURE_MODES = (EPD, FNO_IMAGE, FNO_OBJECT, NA_IMAGE, NA_OBJECT)
_OBJECT_SPACE_MODES = (FNO_OBJECT, NA_OBJECT)
_POWER_EPS = 1e-30


def _tuple_or_none(value):
    if value is None:
        return None
    arr = np.asarray(value, dtype=config.precision).ravel()
    return tuple(float(v) for v in arr)


def _field_key(field):
    if field is None:
        return None
    vignetting = getattr(field, 'vignetting', None)
    vignetting = (None if vignetting is None
                  else tuple((k, float(v)) for k, v in sorted(vignetting.items())))
    return (
        getattr(field, 'hx', None),
        getattr(field, 'hy', None),
        getattr(field, 'kind', None),
        getattr(field, 'unit', None),
        getattr(field, 'object_z', None),
        vignetting,
    )


def _aperture_key(aperture):
    if aperture is None:
        return None
    return (aperture.mode, float(aperture.value))


class ApertureSpec:
    """The aperture of an optical system: a mode plus a value.

    mode is one of EPD, FNO_IMAGE, FNO_OBJECT, NA_IMAGE, NA_OBJECT (module-level
    string constants).  F-number and numerical-aperture modes resolve to an
    equivalent entrance-pupil diameter.

    """

    __slots__ = ('mode', 'value')

    def __init__(self, value, mode=EPD):
        """Initialize an aperture specification.

        Parameters
        ----------
        value : float
            Aperture value in the units selected by mode (diameter for EPD,
            a dimensionless F-number, or a numerical aperture).
        mode : str, optional
            One of the module-level aperture-mode constants.  Defaults to EPD.

        """
        mode = str(mode).upper()
        if mode not in _APERTURE_MODES:
            raise ValueError(
                f'aperture mode {mode!r} must be one of {_APERTURE_MODES}'
            )
        self.mode = mode
        self.value = float(value)

    @classmethod
    def epd(cls, value):
        """An entrance-pupil-diameter aperture."""
        return cls(value, EPD)

    @classmethod
    def fno(cls, value, *, object_space=False):
        """An image-space (default) or object-space F-number aperture."""
        return cls(value, FNO_OBJECT if object_space else FNO_IMAGE)

    @classmethod
    def na(cls, value, *, object_space=False):
        """An image-space (default) or object-space numerical-aperture aperture."""
        return cls(value, NA_OBJECT if object_space else NA_IMAGE)

    def validate(self, object_at_infinity, *, has_power=True):
        """Raise if this spec is illegal for the conjugate / power it sees.

        Parameters
        ----------
        object_at_infinity : bool
            True when the object is at infinity (no finite object surface).
            Object-space modes are illegal in that case.
        has_power : bool, optional
            False for an afocal system; focusing modes (F-number, NA) are then
            illegal.

        """
        if object_at_infinity and self.mode in _OBJECT_SPACE_MODES:
            raise ValueError(
                f'aperture mode {self.mode!r} is object-space and requires a '
                'finite-conjugate object; this system images from infinity'
            )
        if not has_power and self.mode != EPD:
            raise ValueError(
                f'aperture mode {self.mode!r} needs a focusing system; this '
                'system has no net power (afocal) -- specify an EPD instead'
            )

    def _validate_for_system(self, system, wvl=None):
        """Raise if this spec is illegal for system at wavelength wvl."""
        object_at_infinity = bool(getattr(system, 'object_at_infinity', True))
        self.validate(object_at_infinity, has_power=True)
        if self.mode == EPD:
            return
        M, _ = system_matrix(system, wvl=wvl)
        C = float(M[1, 0])
        self.validate(object_at_infinity, has_power=abs(C) >= _POWER_EPS)

    def resolve(self, system, wvl=None):
        """Resolve this spec to a tagged launch boundary condition.

        Returns a (kind, value) pair naming the aperture definition and its
        numeric value: ('EPD', diameter), ('NA_OBJECT', na), etc.  launch
        consumes the tag directly -- it sets up the launch positions and
        direction cosines from the definition and aims the resulting rays onto
        the stop, so no privileged scalar pre-conversion happens here.  The
        equivalent entrance-pupil diameter (a first-order readout) is available
        separately via entrance_pupil_diameter.

        """
        self._validate_for_system(system, wvl)
        return (self.mode, self.value)

    def entrance_pupil_diameter(self, system, wvl=None):
        """Equivalent paraxial entrance-pupil diameter for this spec.

        A first-order readout: EPD returns its value directly; the image-space
        modes invert the paraxial relations EFL = -n_object / C and
        NA_image = |C| * EPD / 2; the object-space modes propagate a paraxial
        marginal ray of the requested object-space NA from the object surface
        to the entrance pupil.  Used for reporting (system.epd) and to size the
        pupil-sampling pattern in the analysis layer; the launch geometry is
        driven by the tagged definition from resolve, not by this scalar.

        Returns the entrance-pupil diameter in system length units.

        """
        object_at_infinity = bool(getattr(system, 'object_at_infinity', True))
        self.validate(object_at_infinity, has_power=True)
        if self.mode == EPD:
            return self.value

        wvl = system.wavelength(wvl)
        M, _ = system_matrix(system, wvl=wvl)
        C = float(M[1, 0])
        self.validate(object_at_infinity, has_power=abs(C) >= _POWER_EPS)

        if self.mode == NA_IMAGE:
            # NA_image = |C| * EPD / 2  ->  EPD = 2 * NA / |C|
            return 2.0 * self.value / abs(C)
        if self.mode == FNO_IMAGE:
            # working F/# at infinity = |EFL| / EPD
            efl = effective_focal_length(system, wvl=wvl)
            return abs(efl) / self.value

        # object-space modes: marginal ray of object-space NA from the object
        # surface to the entrance pupil.  height_at_EP = u_obj * |z_EP - z_obj|,
        # u_obj = NA_obj / n_obj; EPD = 2 * height_at_EP.
        n_obj = _object_index(system, wvl)
        if self.mode == FNO_OBJECT:
            # object-space F/# = 1 / (2 * NA_object)
            na_obj = 1.0 / (2.0 * self.value)
        else:  # NA_OBJECT
            na_obj = self.value
        u_obj = na_obj / n_obj
        z_obj = float(system[0].P[2])
        z_ep = entrance_pupil_z(system, wvl=wvl)
        if z_ep is None:
            raise ValueError(
                'cannot resolve an object-space aperture: the entrance pupil '
                'is at infinity (object-space telecentric) or the stop is '
                'unknown'
            )
        return 2.0 * u_obj * abs(z_ep - z_obj)

    def __repr__(self):
        if self.mode == EPD:
            return f'ApertureSpec(EPD={self.value:g})'
        return f'ApertureSpec({self.mode}={self.value:g})'


def _object_index(system, wvl):
    """Object-space index from the object surface material (1.0 if absent)."""
    from ._meta import object_space_index
    return object_space_index(system, wvl)


class FieldSet:
    """Ordered field points with a tabular repr."""

    __slots__ = ('fields',)

    def __init__(self, fields=None):
        """Initialize a field set."""
        self.fields = _coerce_fields(fields)

    def __len__(self):
        return len(self.fields)

    def __iter__(self):
        return iter(self.fields)

    def __getitem__(self, item):
        return self.fields[item]

    def __repr__(self):
        if not self.fields:
            return 'FieldSet (empty)'
        lines = ['FieldSet']
        kind = self.fields[0].kind
        if kind == 'angle':
            lines.append(f'  {"#":>3s}  {"hx":>10s}  {"hy":>10s}  unit')
            for i, f in enumerate(self.fields):
                lines.append(
                    f'  {i:>3d}  {f.hx:>10.4g}  {f.hy:>10.4g}  {f.unit}')
        else:
            lines.append(
                f'  {"#":>3s}  {"hx":>10s}  {"hy":>10s}  {"object_z":>10s}')
            for i, f in enumerate(self.fields):
                lines.append(
                    f'  {i:>3d}  {f.hx:>10.4g}  {f.hy:>10.4g}  '
                    f'{f.object_z:>10.4g}')
        return '\n'.join(lines)


class OpticalSystem:
    """System metadata around a LensData surface spine."""

    __slots__ = ('lens', 'aperture', 'fields', 'wavelengths', 'weights',
                 'reference', 'unit', 'title', 'stop_index',
                 'ray_aiming', 'source_path', 'source_format', 'extras',
                 '_derived', '_trace_cache')

    def __init__(self, lens, *, aperture=None, fields=None, wavelengths=None,
                 weights=None, reference=None, unit=None, title=None,
                 stop_index=None, ray_aiming='paraxial', source_path=None,
                 source_format=None, extras=None):
        """Initialize a system over a lens."""
        self.lens = lens
        if aperture is not None and not isinstance(aperture, ApertureSpec):
            aperture = ApertureSpec.epd(aperture)
        self.aperture = aperture
        self.fields = fields if isinstance(fields, FieldSet) else FieldSet(fields)
        self.wavelengths = _coerce_wavelengths(wavelengths)
        self.weights = _coerce_weights(weights, self.wavelengths)
        if len(self.wavelengths) and float(np.max(self.wavelengths)) >= 200.0:
            offender = float(np.max(self.wavelengths))
            warnings.warn(
                f'wavelengths are micrometers; {offender:g} looks like '
                'nanometers',
                stacklevel=2,
            )
        self.reference = 0 if reference is None else int(reference)
        self.unit = unit
        self.title = title
        self.stop_index = stop_index
        ray_aiming = str(ray_aiming).lower()
        if ray_aiming not in ('paraxial', 'real'):
            raise ValueError(
                f"ray_aiming must be 'paraxial' or 'real', got {ray_aiming!r}")
        self.ray_aiming = ray_aiming
        self.source_path = source_path
        self.source_format = source_format
        self.extras = dict(extras) if extras else {}
        # Version-keyed derived quantities.
        self._derived = StateCache()
        # Analysis grids for convenience plot methods.
        self._trace_cache = StateCache()

    # -- surface-sequence delegation (duck-type as the compiled surfaces) --
    def to_surfaces(self):
        """Compiled surface list of the underlying lens."""
        return self.lens.to_surfaces()

    @property
    def surfaces(self):
        """Compiled surface list of the underlying lens."""
        return self.lens.surfaces

    @property
    def rows(self):
        """Editable rows of the underlying lens."""
        return self.lens.rows

    def __len__(self):
        return len(self.lens)

    def __iter__(self):
        return iter(self.lens)

    def __getitem__(self, item):
        return self.lens[item]

    # Row editing and the optimizer free vector are a LensData concern: edit
    # surfaces and drive the optimizer through os.lens, not the system.  The
    # system intentionally does NOT delegate add / vary / pack / update / etc.

    # -- metadata resolvers (moved off LensData) --
    @property
    def reference_wavelength(self):
        """Resolved reference wavelength in microns, or None."""
        if len(self.wavelengths) == 0:
            return None
        return float(self.wavelengths[self.reference])

    def wavelength(self, wavelength=None):
        """Resolve a wavelength to microns; None selects the reference."""
        if wavelength is None:
            ref = self.reference_wavelength
            return 0.6328 if ref is None else ref
        return float(wavelength)

    def field(self, field=None):
        """Resolve a field index, scalar y angle, tuple, or Field."""
        if field is None:
            if not self.fields:
                return Field(0.0, 0.0)
            return self.fields[0]
        if isinstance(field, int):
            return self.fields[field]
        return _coerce_field(field)

    @property
    def epd(self):
        """Equivalent entrance-pupil diameter, or None."""
        return self.entrance_pupil_diameter()

    def entrance_pupil_diameter(self, wvl=None):
        """Equivalent entrance-pupil diameter at wvl, cached on the system."""
        if self.aperture is None:
            return None
        wvl = self.wavelength(wvl)
        key = ('epd', self.lens._version, float(wvl),
               self.aperture.mode, self.aperture.value)
        return self._derived.get_or_compute(
            key, lambda: float(self.aperture.entrance_pupil_diameter(self, wvl)))

    @property
    def object_at_infinity(self):
        """True when there is no finite-conjugate object surface."""
        rows = self.lens.rows
        if not rows:
            return True
        first = rows[0]
        # Leading eval row with finite thickness is a finite-conjugate object.
        thi = getattr(first, 'thickness', None)
        from .spencer_and_murty import STYPE_EVAL
        from .surfaces import _map_stype
        typ = getattr(first, 'typ', None)
        is_eval = typ is not None and _map_stype(typ) == STYPE_EVAL
        if not is_eval:
            return True
        return not math.isfinite(float(thi))

    # -- first-order + solves --
    def first_order(self, field=0, wavelength=None, *, epd=None,
                    stop_index=None, force_sym=False):
        """Cached parabasal first-order properties about a chief ray."""
        from .parabasal import first_order, _resolve_field
        wvl = self.wavelength(wavelength)
        resolved_stop = stop_index if stop_index is not None else self.stop_index
        key = ('fo', self.lens._version, _field_key(_resolve_field(self, field)),
               float(wvl), epd, resolved_stop, bool(force_sym))
        return self._derived.get_or_compute(
            key, lambda: first_order(self, field=field, wavelength=wvl,
                                     epd=epd, stop_index=stop_index,
                                     force_sym=force_sym))

    def _ynu_first_order(self, wvl=None, *, epd=None, stop_index=None):
        """Internal YNU first-order properties, cached on the system."""
        from .paraxial import ynu_first_order
        wvl = self.wavelength(wvl)
        resolved_stop = stop_index if stop_index is not None else self.stop_index
        key = ('ynu_fo', self.lens._version, float(wvl), epd, resolved_stop)
        return self._derived.get_or_compute(
            key, lambda: ynu_first_order(self, wvl=wvl, epd=epd,
                                         stop_index=stop_index))

    def entrance_pupil_z(self, wvl=None, stop_index=None):
        """Lab-frame z of the paraxial entrance pupil, cached on the system."""
        from .paraxial import entrance_pupil_z
        wvl = self.wavelength(wvl)
        resolved_stop = stop_index if stop_index is not None else self.stop_index
        key = ('ep_z', self.lens._version, float(wvl), resolved_stop)
        return self._derived.get_or_compute(
            key, lambda: entrance_pupil_z(self, wvl, stop_index=stop_index))

    def exit_pupil(self, wvl=None, field=None, *, stop_index=None, epd=None,
                   axis_point=None, axis_dir=None):
        """Resolved exit-pupil reference point P_xp, cached on the system."""
        from .analysis import resolve_exit_pupil
        wvl = self.wavelength(wvl)
        resolved_stop = stop_index if stop_index is not None else self.stop_index
        field_key = _field_key(field)
        key = (
            'exit_pupil', self.lens._version, float(wvl), field_key,
            resolved_stop, None if epd is None else float(epd),
            _tuple_or_none(axis_point), _tuple_or_none(axis_dir),
            _aperture_key(self.aperture), self.ray_aiming,
        )
        return self._derived.get_or_compute(
            key, lambda: resolve_exit_pupil(
                self, wvl, stop_index=resolved_stop, epd=epd, field=field,
                axis_point=axis_point, axis_dir=axis_dir))

    def solve_image_distance(self, surface=None, *, wavelength=None):
        """Seed the lens image-distance solve with a resolved wavelength."""
        wvl = self.wavelength(wavelength)
        self.lens.solve_image_distance(surface, wavelength=wvl)
        return self

    def clear_image_distance_solve(self):
        """Disable the underlying lens image-distance solve, if any."""
        self.lens.clear_image_distance_solve()
        return self

    def reset_raytrace_cache(self):
        """Clear cached raytrace data and reset the lens edit version."""
        self._trace_cache.clear()
        self._derived.clear()
        self.lens._surfaces_cache = None
        self.lens._version = 0
        return self

    def set_vignetting(self, fields=None, wavelength=None, *, tol=1e-3):
        """Solve per-field vignetting factors against the real apertures."""
        from .launch import solve_vignetting, _normalize_vignetting
        wvl = self.wavelength(wavelength)
        if fields is None:
            fields = self.fields
        for field in fields:
            if isinstance(field, int):
                field = self.fields[field]
            factors = solve_vignetting(self, field, wvl, tol=tol)
            field.vignetting = _normalize_vignetting(factors)
        return self

    # -- convenience plotting (lazy plotting import; cached grids) --
    def _fingerprint(self):
        """Hashable snapshot of metadata that drives a grid trace."""
        aperture = self.aperture
        ap = None if aperture is None else (aperture.mode, aperture.value)
        fields = tuple(
            (f.kind, f.hx, f.hy, f.unit, f.object_z,
             None if f.vignetting is None
             else tuple(sorted(f.vignetting.items())))
            for f in self.fields)
        return (self.lens._version, ap, fields,
                tuple(float(w) for w in self.wavelengths),
                tuple(float(w) for w in self.weights),
                self.reference, self.stop_index, self.ray_aiming)

    def _cached_grid(self, kind, fn, kwargs):
        """Return fn(self, **kwargs), memoized on the live fingerprint."""
        # Settle lazy lens dependencies before fingerprinting.
        self.lens.to_surfaces()
        key = (self._fingerprint(), kind, repr(tuple(sorted(kwargs.items()))))
        return self._trace_cache.get_or_compute(key, lambda: fn(self, **kwargs))

    def layout_2d(self, **kwargs):
        """Draw a 2D layout of the optics and rays."""
        from . import plotting
        return plotting.layout(self, **kwargs)

    def plot_spots(self, *, fields=None, wavelengths=None, sampling=None,
                   epd=None, reference='centroid', **plot_kwargs):
        """Spot diagrams for the system fields and wavelengths."""
        from . import plotting
        from .analysis import spot_diagrams
        grid = self._cached_grid('spots', spot_diagrams, dict(
            fields=fields, wavelengths=wavelengths, sampling=sampling,
            epd=epd, reference=reference))
        return plotting.plot_spot_diagrams(grid, **plot_kwargs)

    def plot_ray_fans(self, *, fields=None, wavelengths=None, nrays=21,
                      epd=None, distribution='uniform', reference='chief',
                      **plot_kwargs):
        """Transverse ray-aberration fans for the system."""
        from . import plotting
        from .analysis import ray_aberration_fans
        grid = self._cached_grid('ray_fans', ray_aberration_fans, dict(
            fields=fields, wavelengths=wavelengths, nrays=nrays, epd=epd,
            distribution=distribution, reference=reference))
        return plotting.plot_ray_fans(grid, **plot_kwargs)

    def plot_opd_fans(self, *, fields=None, wavelengths=None, nrays=21,
                      epd=None, distribution='uniform', stop_index=None,
                      output='waves', **plot_kwargs):
        """Wavefront OPD fans for the system."""
        from . import plotting
        from .analysis import opd_fans
        grid = self._cached_grid('opd_fans', opd_fans, dict(
            fields=fields, wavelengths=wavelengths, nrays=nrays, epd=epd,
            distribution=distribution, stop_index=stop_index, output=output))
        return plotting.plot_opd_fans(grid, **plot_kwargs)

    def plot_field_curvature(self, *, fields=None, wavelength=None,
                             samples=101, **plot_kwargs):
        """Field-curvature curves for the system."""
        from . import plotting
        from .analysis import field_curvature
        grid = self._cached_grid('field_curvature', field_curvature, dict(
            fields=fields, wavelength=wavelength, samples=samples))
        return plotting.plot_field_curvature(
            self, fields, result=grid, samples=samples, **plot_kwargs)

    def plot_distortion(self, *, fields=None, wavelength=None, epd=None,
                        distortion_type='f-tan', pupil_z=None, samples=101,
                        **plot_kwargs):
        """Percent-distortion curve for the system."""
        from . import plotting
        from .analysis import distortion
        grid = self._cached_grid('distortion', distortion, dict(
            fields=fields, wavelength=wavelength, epd=epd,
            distortion_type=distortion_type, pupil_z=pupil_z,
            samples=samples))
        return plotting.plot_distortion(
            self, fields, result=grid, samples=samples, **plot_kwargs)

    def plot_chromatic_focal_shift(self, *, wavelengths=None,
                                   reference_wavelength=None, focus='best',
                                   epd=None, field=None, sampling=None,
                                   samples=101, **plot_kwargs):
        """Chromatic focal-shift curve for the system."""
        from . import plotting
        from .analysis import chromatic_focal_shift
        data = self._cached_grid(
            'chromatic_focal_shift', chromatic_focal_shift, dict(
                wavelengths=wavelengths,
                reference_wavelength=reference_wavelength, focus=focus,
                epd=epd, field=field, sampling=sampling, samples=samples))
        return plotting.plot_chromatic_focal_shift(
            self, result=data, **plot_kwargs)

    def plot_axial_color(self, *, wavelengths=None, **plot_kwargs):
        """Paraxial focus shift over the wavelength set."""
        from . import plotting
        from .analysis import axial_color
        data = self._cached_grid('axial_color', axial_color, dict(
            wavelengths=wavelengths))
        return plotting.plot_axial_color(
            self, wavelengths, result=data, **plot_kwargs)

    def plot_lateral_color(self, *, fields=None, wavelengths=None, epd=None,
                           samples=101, **plot_kwargs):
        """Lateral-color curves for non-reference wavelengths."""
        from . import plotting
        from .analysis import lateral_color
        data = self._cached_grid('lateral_color', lateral_color, dict(
            fields=fields, wavelengths=wavelengths, epd=epd, samples=samples))
        return plotting.plot_lateral_color(
            self, fields, wavelengths, result=data, samples=samples,
            **plot_kwargs)

    def plot_full_field(self, *, metric='rms spot', samples=15,
                        max_field=None, wavelengths=None, sampling=None,
                        epd=None, stop_index=None, **plot_kwargs):
        """2D metric map over the system field disc."""
        from . import plotting
        from .analysis import full_field
        grid = self._cached_grid('full_field', full_field, dict(
            metric=metric, samples=samples, max_field=max_field,
            wavelengths=wavelengths, sampling=sampling, epd=epd,
            stop_index=stop_index))
        return plotting.plot_full_field(grid, **plot_kwargs)

    # -- optimization sugar --
    def problem(self, goal='spot', *, sampling=None, fields=None,
                wavelengths=None, constraints=None):
        """Assemble a design.Problem over this system's free vector."""
        from .design import build_problem
        return build_problem(self, goal, sampling=sampling, fields=fields,
                             wavelengths=wavelengths, constraints=constraints)

    def optimize(self, goal='spot', *, sampling=None, fields=None,
                 wavelengths=None, constraints=None, **solve_kwargs):
        """Build and solve an optimization problem in one shot."""
        prob = self.problem(goal, sampling=sampling, fields=fields,
                            wavelengths=wavelengths, constraints=constraints)
        return prob.solve(**solve_kwargs)

    # -- listings delegate to the lens --
    def list_surfaces(self, *, unit=None):
        """Tabular surface listing (lens data editor)."""
        return self.lens.list_surfaces(
            stop_index=self.stop_index, unit=unit or self.unit)

    def list_apertures(self):
        """Tabular per-surface clear-aperture listing."""
        return self.lens.list_apertures()

    def list_decenters(self):
        """Tabular coordinate-break decenter / tilt listing."""
        return self.lens.list_decenters()

    def copy(self):
        """Return a copy: the lens is copied; metadata containers are copied."""
        return OpticalSystem(
            self.lens.copy(), aperture=self.aperture,
            fields=list(self.fields), wavelengths=self.wavelengths,
            weights=self.weights, reference=self.reference, unit=self.unit,
            title=self.title, stop_index=self.stop_index,
            ray_aiming=self.ray_aiming,
            source_path=self.source_path, source_format=self.source_format,
            extras=dict(self.extras),
        )

    def __repr__(self):
        ap = repr(self.aperture) if self.aperture is not None else 'None'
        return (f'OpticalSystem(rows={len(self.lens.rows)}, aperture={ap}, '
                f'fields={len(self.fields)}, wavelengths={len(self.wavelengths)}, '
                f'stop_index={self.stop_index})')


# ---------------------------------------------------------------------------
# metadata coercion (field + wavelength specs)
# ---------------------------------------------------------------------------

def _coerce_field(field):
    """Coerce an input field specification to a Field.

    Scalars are interpreted as y field values; sequences provide x and y.

    """
    if isinstance(field, Field):
        return field
    if np.isscalar(field):
        return Field(0.0, float(field))
    return Field(float(field[0]), float(field[1]))


def _coerce_fields(fields):
    """Coerce a sequence of field specifications.  None becomes an empty list."""
    if fields is None:
        return []
    if isinstance(fields, FieldSet):
        return list(fields.fields)
    return [_coerce_field(field) for field in fields]


def _coerce_wavelengths(wavelengths):
    """Coerce wavelength metadata to a 1-D micron array.

    None becomes an empty array.

    """
    if wavelengths is None:
        return np.asarray([], dtype=config.precision)
    if hasattr(wavelengths, 'keys'):
        raise TypeError(
            'wavelengths must be a sequence of micron floats, not a mapping; '
            'pass e.g. list(FRAUNHOFER_LINES_UM.values()) and select the '
            'reference by integer index'
        )
    return np.asarray([float(w) for w in wavelengths], dtype=config.precision)


def _coerce_weights(weights, wavelengths):
    """Coerce spectral weights to an array parallel to wavelengths.

    None defaults to all ones; an explicit sequence must match the wavelength
    count.

    """
    n = len(wavelengths)
    if weights is None:
        return np.ones(n, dtype=config.precision)
    weights = np.asarray([float(w) for w in weights], dtype=config.precision)
    if len(weights) != n:
        raise ValueError(
            f'weights length {len(weights)} does not match the {n} '
            'wavelengths'
        )
    return weights


# imported at module end to avoid a circular import at package load time
from .launch import Field  # noqa: E402


__all__ = ['OpticalSystem', 'ApertureSpec', 'FieldSet',
           'EPD', 'FNO_IMAGE', 'FNO_OBJECT', 'NA_IMAGE', 'NA_OBJECT']
