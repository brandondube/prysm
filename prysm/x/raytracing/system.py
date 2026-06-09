"""System-level metadata wrapper for LensData."""

import math

from prysm.mathops import np

from .paraxial import (
    effective_focal_length,
    entrance_pupil_z,
    system_matrix,
)


# aperture modes
EPD = 'EPD'
FNO_IMAGE = 'FNO_IMAGE'
FNO_OBJECT = 'FNO_OBJECT'
NA_IMAGE = 'NA_IMAGE'
NA_OBJECT = 'NA_OBJECT'

_APERTURE_MODES = (EPD, FNO_IMAGE, FNO_OBJECT, NA_IMAGE, NA_OBJECT)
_OBJECT_SPACE_MODES = (FNO_OBJECT, NA_OBJECT)
_POWER_EPS = 1e-30

# cache sentinel: distinguishes an unresolved key from a resolved None (an exit
# pupil at infinity / telecentric, which the EIC closing reads as curvature 0).
_EXIT_PUPIL_MISS = object()


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
    """An ordered set of field points with a tabular repr.

    Wraps a list of launch.Field; iterable, sized, and indexable so it behaves
    like the bare field list the analysis layer iterates.

    """

    __slots__ = ('fields',)

    def __init__(self, fields=None):
        """Initialize from a sequence of field specs (coerced to Field)."""
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
    """System-level handle wrapping a LensData spine.

    Owns the aperture (an ApertureSpec), the fields (a FieldSet), the
    wavelengths and reference wavelength, the length unit, provenance, and the
    integer stop index, and holds the LensData that carries the surfaces.

    Duck-types as the compiled surface sequence: len/iter/getitem, to_surfaces,
    rows, and surfaces all delegate to the lens, so the paraxial / analysis /
    launch layer consumes an OpticalSystem exactly as a surface sequence while
    reading system metadata off it.  Row editing and the optimizer free vector
    are NOT delegated -- edit surfaces / drive the optimizer through os.lens.

    """

    __slots__ = ('lens', 'aperture', 'fields', 'wavelengths',
                 'reference_wavelength', 'unit', 'title', 'stop_index',
                 'ray_aiming', 'source_path', 'source_format', 'extras',
                 '_derived')

    def __init__(self, lens, *, aperture=None, fields=None, wavelengths=None,
                 reference_wavelength=None, unit=None, title=None,
                 stop_index=None, ray_aiming='paraxial', source_path=None,
                 source_format=None, extras=None):
        """Initialize a system over a lens."""
        self.lens = lens
        if aperture is not None and not isinstance(aperture, ApertureSpec):
            aperture = ApertureSpec.epd(aperture)
        self.aperture = aperture
        self.fields = fields if isinstance(fields, FieldSet) else FieldSet(fields)
        self.wavelengths = _coerce_wavelengths(wavelengths)
        if reference_wavelength is None and self.wavelengths:
            reference_wavelength = next(iter(self.wavelengths))
        self.reference_wavelength = reference_wavelength
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
        # version-keyed cache of derived first-order quantities (e.g. the exit
        # pupil); invalidated implicitly via self.lens._version in the key.
        self._derived = {}

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
    def wavelength(self, wavelength=None):
        """Resolve a wavelength name or scalar to microns."""
        if wavelength is None:
            wavelength = self.reference_wavelength
        if wavelength is None:
            return 0.6328
        if isinstance(wavelength, str):
            return float(self.wavelengths[wavelength])
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
        """Equivalent entrance-pupil diameter, or None when no aperture is set.

        A first-order readout of the aperture definition (see
        ApertureSpec.entrance_pupil_diameter).
        """
        if self.aperture is None:
            return None
        return float(self.aperture.entrance_pupil_diameter(self))

    @property
    def object_at_infinity(self):
        """True when there is no finite-conjugate object surface."""
        rows = self.lens.rows
        if not rows:
            return True
        first = rows[0]
        # an object surface is the leading eval row carrying object-space gap;
        # a finite object has finite leading thickness.
        thi = getattr(first, 'thickness', None)
        from .spencer_and_murty import STYPE_EVAL
        from .surfaces import _map_stype
        typ = getattr(first, 'typ', None)
        is_eval = typ is not None and _map_stype(typ) == STYPE_EVAL
        if not is_eval:
            return True
        return not math.isfinite(float(thi))

    # -- first-order + solves --
    def first_order(self, wvl=None, *, epd=None, stop_index=None):
        """Paraxial first-order properties (delegates to paraxial.first_order)."""
        from .paraxial import first_order
        return first_order(self, wvl=wvl, epd=epd, stop_index=stop_index)

    def exit_pupil(self, wvl=None, field=None, *, stop_index=None, epd=None,
                   axis_point=None, axis_dir=None):
        """Resolved exit-pupil reference point P_xp, cached on the system.

        Wraps analysis.resolve_exit_pupil and memoizes the result keyed by the
        lens edit version, wavelength, field, and stop index, so a grid
        analysis resolves the (field-independent, paraxial) exit pupil once per
        wavelength instead of per field/wavelength/axis.  The cache is lazy and
        explicit -- it computes on first access and never recomputes silently;
        an edit through self.lens bumps lens._version (a cache miss), and
        stop_index is part of the key so reassigning self.stop_index can't
        return a stale pupil.

        Parameters mirror analysis.resolve_exit_pupil; wvl and field default to
        the system reference wavelength and on-axis field.

        Returns
        -------
        P_xp : ndarray, shape (3,), or None
            exit-pupil reference point, ready to pass to wavefront(P_xp=...).
            None when the exit pupil is at infinity (image-space telecentric),
            which the EIC closing reads as its curvature kappa = 0 limit.

        """
        from .analysis import resolve_exit_pupil
        wvl = self.wavelength(wvl)
        resolved_stop = stop_index if stop_index is not None else self.stop_index
        # field enters the key only via the geometric route; identify it by a
        # hashable surrogate (its angle/height tuple) so equal fields share.
        field_key = None if field is None else (
            getattr(field, 'hx', None), getattr(field, 'hy', None),
            getattr(field, 'kind', None))
        key = (self.lens._version, float(wvl), field_key, resolved_stop)
        cached = self._derived.get(key, _EXIT_PUPIL_MISS)
        if cached is _EXIT_PUPIL_MISS:
            cached = resolve_exit_pupil(
                self, wvl, stop_index=resolved_stop, epd=epd, field=field,
                axis_point=axis_point, axis_dir=axis_dir)
            self._derived[key] = cached
        return cached

    def solve_image_distance(self, surface=None, *, wavelength=None):
        """Seed the lens image-distance solve with a resolved wavelength."""
        wvl = self.wavelength(wavelength)
        self.lens.solve_image_distance(surface, wavelength=wvl)
        return self

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
            fields=list(self.fields), wavelengths=dict(self.wavelengths),
            reference_wavelength=self.reference_wavelength, unit=self.unit,
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
    """Coerce wavelength metadata to a dictionary.

    Sequence inputs are keyed by stringified integer index.

    """
    if wavelengths is None:
        return {}
    if hasattr(wavelengths, 'items'):
        return dict(wavelengths)
    return {str(i): float(w) for i, w in enumerate(wavelengths)}


# imported at module end to avoid a circular import at package load time
from .launch import Field  # noqa: E402


__all__ = ['OpticalSystem', 'ApertureSpec', 'FieldSet',
           'EPD', 'FNO_IMAGE', 'FNO_OBJECT', 'NA_IMAGE', 'NA_OBJECT']
