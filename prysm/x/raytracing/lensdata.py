"""Editable LensData rows and surface compilation."""

import math
import warnings

from prysm.conf import config
from prysm.mathops import np

from ..materials import MIRROR
from .surfaces import (
    Biconic,
    Chebyshev,
    Conic,
    EvenAsphere,
    Jacobi,
    OffAxisConic,
    Plane,
    Q2D,
    Sphere,
    Surface,
    Toroid,
    XY,
    Zernike,
    circular_aperture,
    _map_stype,
)
from .paraxial import paraxial_image_distance
from .spencer_and_murty import STYPE_EVAL, STYPE_REFLECT


_DEG2RAD = math.pi / 180.0


def R_rh(rz, ry, rx, radians=False):
    """Right-handed ZYX rotation matrix, backend-pure."""
    k = 1.0 if radians else _DEG2RAD
    alpha = rx * k
    beta = ry * k
    gamma = rz * k
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    cb = np.cos(beta)
    sb = np.sin(beta)
    cg = np.cos(gamma)
    sg = np.sin(gamma)
    zero = ca * 0.0
    one = zero + 1.0
    Rx = np.stack([
        np.stack([one, zero, zero]),
        np.stack([zero, ca, -sa]),
        np.stack([zero, sa, ca]),
    ])
    Ry = np.stack([
        np.stack([cb, zero, sb]),
        np.stack([zero, one, zero]),
        np.stack([-sb, zero, cb]),
    ])
    Rz = np.stack([
        np.stack([cg, -sg, zero]),
        np.stack([sg, cg, zero]),
        np.stack([zero, zero, one]),
    ])
    return Rx @ Ry @ Rz


# 180-degree x rotation for normal-incidence mirror folds.
_FLIP_Z = np.asarray([[1.0, 0.0, 0.0],
                      [0.0, -1.0, 0.0],
                      [0.0, 0.0, -1.0]], dtype=config.precision)


def _ben_auto_gamma(alpha_deg, beta_deg):
    """Bend auto-roll gamma (degrees) that keeps the folded axis level.

    gamma = atan2(-sin(alpha) sin(beta), cos(alpha) + cos(beta)); zero when
    either alpha or beta is zero.  This is the roll correction used by
    lens-design bend coordinate breaks.

    """
    a = alpha_deg * _DEG2RAD
    b = beta_deg * _DEG2RAD
    num = -np.sin(a) * np.sin(b)
    den = np.cos(a) + np.cos(b)
    return np.arctan2(num, den) / _DEG2RAD


def _as_mat(R):
    """Return a concrete 3x3 matrix (identity if R is None)."""
    if R is None:
        return np.eye(3, dtype=config.precision)
    return R


def _local_to_global(Rgl):
    """Local->global rotation from a global->local matrix (or None)."""
    return _as_mat(Rgl).T


def _compose_global_to_local(Rgl, local_rot):
    """Compose an additional rotation expressed in the current local frame."""
    return _as_mat(local_rot) @ _as_mat(Rgl)


def _axial_step(thickness):
    """A length-3 step of thickness along the local +z axis."""
    step = np.zeros(3, dtype=config.precision)
    step[2] = thickness
    return step


def _none_if_identity(Rgl):
    """Collapse a near-identity rotation to None so the kernel skips it."""
    if Rgl is None:
        return None
    if np.allclose(np.asarray(Rgl), np.eye(3)):
        return None
    return Rgl


def _apply_decenter_tilt(o, Rgl, decenter, tilt):
    """Apply coordinate-break decenter then tilt to a running frame."""
    o = o + _local_to_global(Rgl) @ decenter
    Rt = R_rh(tilt[0], tilt[1], tilt[2])
    return o, _compose_global_to_local(Rgl, Rt)


class _FrameState:
    """Mutable frame state for coordinate-break layout scans."""

    __slots__ = ('o', 'Rgl', 'frames', 'pending_pose', 'pending_fold')

    def __init__(self):
        self.o = np.zeros(3, dtype=config.precision)
        self.Rgl = None
        self.frames = {}
        self.pending_pose = None
        self.pending_fold = None

    def advance(self, thickness):
        self.o = self.o + _local_to_global(self.Rgl) @ _axial_step(thickness)


# ---------------------------------------------------------------------------
# Shape adapters: how each Shape decomposes into numeric DOFs + static metadata
# ---------------------------------------------------------------------------

class _ShapeDescriptor:
    """Shape-declared DOF layout used by SurfaceRow."""

    __slots__ = ('cls', 'scalar_dofs', 'vector_dofs', 'meta_keys',
                 'categories', 'build')

    def __init__(self, cls):
        """Read the DOF descriptor off a registered Shape class."""
        self.cls = cls
        self.scalar_dofs = tuple(cls.SCALAR_DOFS)
        self.vector_dofs = tuple(cls.VECTOR_DOFS)
        self.meta_keys = tuple(cls.META_KEYS)
        self.categories = {k: list(v) for k, v in cls.CATEGORIES.items()}
        self.build = cls.from_params


def _adapter_for(shape):
    """Return the DOF descriptor for a registered shape."""
    cls = type(shape)
    if not hasattr(cls, 'from_params'):
        raise TypeError(
            f'shape {cls.__name__} is not registered with LensData; declare '
            f'SCALAR_DOFS / VECTOR_DOFS / META_KEYS / CATEGORIES and a '
            f'from_params classmethod on the shape class'
        )
    return _ShapeDescriptor(cls)


def _bounds_for_dof(nominal, lo, hi, relative, is_radius):
    """Ordered (lo, hi) bounds for one constrained DOF slot."""
    if is_radius:
        if nominal == 0.0:
            if relative is not None:
                warnings.warn(
                    'relative radius bound on a flat (c=0) surface is '
                    'degenerate; leaving it unbounded', stacklevel=3)
            return None
        quantity = 1.0 / nominal
    else:
        quantity = nominal

    if relative is not None:
        if quantity == 0.0:
            warnings.warn(
                'relative bound on a zero nominal is degenerate; leaving it '
                'unbounded', stacklevel=3)
            return None
        qlo = quantity * (1.0 - relative)
        qhi = quantity * (1.0 + relative)
    else:
        qlo = -np.inf if lo is None else float(lo)
        qhi = np.inf if hi is None else float(hi)

    if is_radius:
        # map radius bounds back to curvature (1/R); the reciprocal flips order
        clo = 0.0 if np.isinf(qhi) else 1.0 / qhi
        chi = 0.0 if np.isinf(qlo) else 1.0 / qlo
        blo, bhi = clo, chi
    else:
        blo, bhi = qlo, qhi

    if blo > bhi:
        blo, bhi = bhi, blo
    return (blo, bhi)


def _invalidate_row_owner(row):
    owner = getattr(row, '_owner', None)
    if owner is not None:
        owner._invalidate()


class _InvalidatingArray(np.ndarray):
    """ndarray view that clears a row owner's surface cache when edited."""

    def __new__(cls, values, row, dtype=None):
        arr = np.asarray(values, dtype=dtype).view(cls)
        arr._row = row
        return arr

    def __array_finalize__(self, obj):
        self._row = getattr(obj, '_row', None)

    def __setitem__(self, item, value):
        super().__setitem__(item, value)
        row = getattr(self, '_row', None)
        if row is not None:
            _invalidate_row_owner(row)


def _invalidating_array(values, row, dtype=None):
    arr = np.asarray(values, dtype=dtype)
    if hasattr(np, 'ndarray') and isinstance(arr, np.ndarray):
        out = arr.view(_InvalidatingArray)
        out._row = row
        return out
    return arr


_MISSING = object()


class _InvalidatingDict(dict):
    """dict that clears a row owner's surface cache on mutation."""

    def __init__(self, *args, row=None, **kwargs):
        self._row = row
        super().__init__(*args, **kwargs)

    def _invalidate(self):
        row = getattr(self, '_row', None)
        if row is not None:
            _invalidate_row_owner(row)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self._invalidate()

    def __delitem__(self, key):
        super().__delitem__(key)
        self._invalidate()

    def clear(self):
        super().clear()
        self._invalidate()

    def pop(self, key, default=_MISSING):
        if default is _MISSING:
            value = super().pop(key)
        else:
            if key not in self:
                return default
            value = super().pop(key)
        self._invalidate()
        return value

    def popitem(self):
        value = super().popitem()
        self._invalidate()
        return value

    def setdefault(self, key, default=None):
        if key in self:
            return self[key]
        value = super().setdefault(key, default)
        self._invalidate()
        return value

    def update(self, *args, **kwargs):
        super().update(*args, **kwargs)
        self._invalidate()


def _invalidating_dict(value, row):
    if value is None:
        return None
    if isinstance(value, _InvalidatingDict):
        return _InvalidatingDict(dict(value), row=row)
    if not isinstance(value, dict):
        return value
    return _InvalidatingDict(value, row=row)


class _RowList(list):
    """Owned row container that invalidates LensData on structural edits."""

    def __init__(self, owner, rows=()):
        self._owner = owner
        super().__init__()
        self.extend(rows)

    def _own(self, row):
        if hasattr(row, '_owner'):
            object.__setattr__(row, '_owner', self._owner)
        return row

    def _invalidate(self):
        self._owner._invalidate()

    def append(self, row):
        super().append(self._own(row))
        self._invalidate()

    def extend(self, rows):
        changed = False
        for row in rows:
            super().append(self._own(row))
            changed = True
        if changed:
            self._invalidate()

    def insert(self, index, row):
        super().insert(index, self._own(row))
        self._invalidate()

    def __setitem__(self, item, value):
        if isinstance(item, slice):
            value = [self._own(row) for row in value]
        else:
            value = self._own(value)
        super().__setitem__(item, value)
        self._invalidate()

    def __delitem__(self, item):
        super().__delitem__(item)
        self._invalidate()

    def clear(self):
        super().clear()
        self._invalidate()

    def pop(self, index=-1):
        value = super().pop(index)
        self._invalidate()
        return value

    def remove(self, value):
        super().remove(value)
        self._invalidate()


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------

class SurfaceRow:
    """One sequential optical surface in a LensData system."""

    _INVALIDATING_ATTRS = {
        'params', 'meta', 'thickness', 'material', 'typ', 'semidiameter',
        'aperture', 'bounding', 'grating', 'edge', 'coating',
    }

    def __setattr__(self, name, value):
        if name == 'params':
            value = _invalidating_array(value, self, dtype=config.precision)
        elif name in ('meta', 'bounding'):
            value = _invalidating_dict(value, self)
        object.__setattr__(self, name, value)
        if name in self._INVALIDATING_ATTRS:
            _invalidate_row_owner(self)

    def __init__(self, shape, *, thickness=0.0, material=None, typ='refr',
                 semidiameter=None, aperture=None, bounding=None, grating=None,
                 edge=None, coating=None):
        """Initialize a surface row from a shape."""
        object.__setattr__(self, '_owner', None)
        adapter = _adapter_for(shape)
        params = []
        key_offsets = {}
        sp = shape.params or {}
        for key in adapter.scalar_dofs:
            key_offsets[key] = (len(params), 1)
            params.append(sp[key])
        for key in adapter.vector_dofs:
            vals = list(sp[key])
            key_offsets[key] = (len(params), len(vals))
            params.extend(vals)

        self.shape_kind = type(shape)
        self.adapter = adapter
        self.params = (np.asarray(params, dtype=config.precision)
                       if params else np.zeros(0, dtype=config.precision))
        self.key_offsets = key_offsets
        self.meta = {key: sp[key] for key in adapter.meta_keys}

        categories = {}
        for cat, keys in adapter.categories.items():
            offs = []
            for key in keys:
                start, length = key_offsets[key]
                offs.extend(range(start, start + length))
            categories[cat] = offs
        self.categories = categories

        self.thickness = thickness
        self.material = material
        self.typ = typ
        self.semidiameter = semidiameter
        if aperture is None and semidiameter is not None:
            aperture = circular_aperture(semidiameter)
        self.aperture = aperture
        if bounding is None and semidiameter is not None:
            bounding = {'outer_radius': float(semidiameter)}
        self.bounding = bounding
        self.grating = grating
        self.edge = edge
        self.coating = coating

    @property
    def is_reflective(self):
        """True if this surface reflects (folds the layout frame)."""
        return _map_stype(self.typ) == STYPE_REFLECT

    def build_shape(self):
        """Rebuild the Shape object from the current parameter array + meta."""
        p = dict(self.meta)
        scalar = set(self.adapter.scalar_dofs)
        for key, (start, length) in self.key_offsets.items():
            # A length-1 vector DOF must stay a length-1 block, not collapse to
            # a scalar; distinguish by role, not by length.
            if key in scalar:
                p[key] = self.params[start]
            else:
                p[key] = self.params[start:start + length]
        return self.adapter.build(p)

    def dof_slots(self, row_index):
        """Yield ('shape'/'thickness', row_index, offset) for every scalar DOF."""
        for off in range(len(self.params)):
            yield ('shape', row_index, off)
        yield ('thickness', row_index, 0)

    def copy(self):
        """Return a copy of the row."""
        new = object.__new__(SurfaceRow)
        object.__setattr__(new, '_owner', None)
        new.shape_kind = self.shape_kind
        new.adapter = self.adapter
        new.params = np.array(self.params, copy=True)
        new.key_offsets = dict(self.key_offsets)
        new.meta = dict(self.meta)
        new.categories = {k: list(v) for k, v in self.categories.items()}
        new.thickness = self.thickness
        new.material = self.material
        new.typ = self.typ
        new.semidiameter = self.semidiameter
        new.aperture = self.aperture
        new.bounding = self.bounding
        new.grating = self.grating
        new.edge = self.edge
        new.coating = self.coating
        return new


class CoordBreak:
    """A right-handed coordinate break."""

    _INVALIDATING_ATTRS = {
        'decenter', 'tilt', 'kind', 'ret_target', 'thickness',
    }

    def __setattr__(self, name, value):
        if name in ('decenter', 'tilt'):
            value = _invalidating_array(value, self, dtype=config.precision)
        object.__setattr__(self, name, value)
        if name in self._INVALIDATING_ATTRS:
            _invalidate_row_owner(self)

    def __init__(self, *, decenter=(0.0, 0.0, 0.0), tilt=(0.0, 0.0, 0.0),
                 kind='basic', ret_target=None, thickness=0.0):
        """Initialize a coordinate break row."""
        object.__setattr__(self, '_owner', None)
        self.decenter = np.asarray(decenter, dtype=config.precision)
        self.tilt = np.asarray(tilt, dtype=config.precision)
        self.kind = kind
        self.ret_target = ret_target
        self.thickness = thickness

    def dof_slots(self, row_index):
        """Yield decenter / tilt / thickness DOF slots for this break."""
        for off in range(3):
            yield ('decenter', row_index, off)
        for off in range(3):
            yield ('tilt', row_index, off)
        yield ('thickness', row_index, 0)

    def copy(self):
        """Return a copy of the coordinate break."""
        new = object.__new__(CoordBreak)
        object.__setattr__(new, '_owner', None)
        new.decenter = np.array(self.decenter, copy=True)
        new.tilt = np.array(self.tilt, copy=True)
        new.kind = self.kind
        new.ret_target = self.ret_target
        new.thickness = self.thickness
        return new


# ---------------------------------------------------------------------------
# Parameter specification: named slots <-> dense free vector
# ---------------------------------------------------------------------------

class ParamSpec:
    """Free-vector slots, flags, and bounds for LensData rows."""

    def __init__(self, lensdata):
        self._ld = lensdata
        self._free = {}     # slot -> True
        self._bounds = {}   # slot -> (lo, hi)

    # -- slot enumeration --
    def slots(self):
        """Ordered list of every scalar DOF slot, row by row."""
        out = []
        for r, row in enumerate(self._ld.rows):
            out.extend(row.dof_slots(r))
        return out

    def free_slots(self):
        """Ordered list of the slots currently marked free."""
        return [s for s in self.slots() if self._free.get(s, False)]

    # -- value access --
    def get_value(self, slot):
        """Return the scalar value addressed by a slot."""
        group, r, off = slot
        row = self._ld.rows[r]
        if group == 'shape':
            return row.params[off]
        if group == 'thickness':
            return row.thickness
        if group == 'decenter':
            return row.decenter[off]
        if group == 'tilt':
            return row.tilt[off]
        raise KeyError(group)

    def set_value(self, slot, value):
        """Set the scalar value addressed by a slot."""
        group, r, off = slot
        row = self._ld.rows[r]
        if group == 'shape':
            row.params[off] = value
        elif group == 'thickness':
            row.thickness = value
        elif group == 'decenter':
            row.decenter[off] = value
        elif group == 'tilt':
            row.tilt[off] = value
        else:
            raise KeyError(group)

    # -- optimizer surface --
    def pack(self):
        """Gather the free DOFs into a dense contiguous vector."""
        free = self.free_slots()
        out = np.empty(len(free), dtype=config.precision)
        for i, slot in enumerate(free):
            out[i] = self.get_value(slot)
        return out

    def scatter(self, x):
        """Write a dense free vector back into the rows."""
        free = self.free_slots()
        if len(x) != len(free):
            raise ValueError(
                f'expected {len(free)} free DOFs, got {len(x)}'
            )
        if np.__name__ == 'numpy':
            for slot, value in zip(free, x):
                self.set_value(slot, value)
            return

        # tensor backend: functional per-row reconstruction
        by_row_group = {}
        for i, slot in enumerate(free):
            group, r, off = slot
            by_row_group.setdefault((r, group), []).append((off, x[i]))
        for (r, group), items in by_row_group.items():
            row = self._ld.rows[r]
            if group == 'thickness':
                row.thickness = items[0][1]
            elif group == 'shape':
                vals = [row.params[j] for j in range(len(row.params))]
                for off, v in items:
                    vals[off] = v
                row.params = np.stack(vals)
            elif group == 'decenter':
                vals = [row.decenter[j] for j in range(3)]
                for off, v in items:
                    vals[off] = v
                row.decenter = np.stack(vals)
            elif group == 'tilt':
                vals = [row.tilt[j] for j in range(3)]
                for off, v in items:
                    vals[off] = v
                row.tilt = np.stack(vals)

    def bounds(self):
        """Return (lo, hi) arrays parallel to the free vector."""
        free = self.free_slots()
        lo = np.empty(len(free), dtype=config.precision)
        hi = np.empty(len(free), dtype=config.precision)
        for i, slot in enumerate(free):
            blo, bhi = self._bounds.get(slot, (-np.inf, np.inf))
            lo[i] = blo
            hi[i] = bhi
        return lo, hi


# ---------------------------------------------------------------------------
# LensData
# ---------------------------------------------------------------------------

class LensData:
    """Editable sequential optical system."""

    def __init__(self):
        """Initialize an empty lens."""
        self.rows = _RowList(self)
        self.spec = ParamSpec(self)
        self._pickups = []      # (target_slots, source_slots, scale, offset)
        self._image_solve = None  # (surface_row_index, wavelength)
        self._dependent = set()  # slots driven by a pickup/solve (never free)
        self._surfaces_cache = None
        self._version = 0  # bumped on every edit; keys system-side derived caches
        self._resolving = False  # True while solves/pickups write derived DOFs

    # -- construction --
    def add(self, shape, *, thickness=0.0, material=None, typ='refr',
            semidiameter=None, aperture=None, bounding=None, grating=None,
            edge=None, coating=None):
        """Append a surface row and return self."""
        self.rows.append(SurfaceRow(
            shape, thickness=thickness, material=material, typ=typ,
            semidiameter=semidiameter, aperture=aperture, bounding=bounding,
            grating=grating, edge=edge, coating=coating,
        ))
        self._invalidate()
        return self

    def add_coordbreak(self, *, decenter=(0.0, 0.0, 0.0), tilt=(0.0, 0.0, 0.0),
                       kind='basic', ret_target=None, thickness=0.0):
        """Append a coordinate break."""
        self.rows.append(CoordBreak(
            decenter=decenter, tilt=tilt, kind=kind, ret_target=ret_target,
            thickness=thickness,
        ))
        self._invalidate()
        return self

    def _invalidate(self):
        """Clear compiled surfaces and bump the edit version."""
        if self._resolving:
            return
        self._surfaces_cache = None
        self._version += 1

    # -- compilation --
    def to_surfaces(self):
        """Compile rows into posed Surface objects."""
        if self._surfaces_cache is not None:
            return self._surfaces_cache

        self._resolve_dependencies()
        surfaces = self._compile_surfaces()
        self._surfaces_cache = surfaces
        return surfaces

    def _compile_surfaces(self):
        """Compile rows -> surfaces with no caching or dependency resolution."""
        if any(isinstance(row, CoordBreak) for row in self.rows):
            return self._to_surfaces_general()
        return self._to_surfaces_axial()

    def _resolve_dependencies(self):
        """Apply pickups then solves before compilation."""
        self._resolving = True
        try:
            for targets, sources, scale, offset in self._pickups:
                for t, s in zip(targets, sources):
                    self.spec.set_value(
                        t, scale * self.spec.get_value(s) + offset)
            if self._image_solve is not None:
                surf_idx, wvl = self._image_solve
                surfaces = self._compile_surfaces()
                surface_rows = [i for i, row in enumerate(self.rows)
                                if isinstance(row, SurfaceRow)]
                try:
                    solved_surface = surface_rows.index(surf_idx)
                except ValueError as e:
                    raise ValueError(
                        'image-distance solve target must be a surface row'
                    ) from e
                image_surface = solved_surface + 1
                if image_surface >= len(surface_rows):
                    raise ValueError(
                        'image-distance solve target must be the gap before a '
                        'trailing eval image plane'
                    )
                image_row_idx = surface_rows[image_surface]
                image_row = self.rows[image_row_idx]
                if (image_surface != len(surface_rows) - 1
                        or _map_stype(image_row.typ) != STYPE_EVAL):
                    raise ValueError(
                        'image-distance solve target must be the gap before a '
                        'trailing eval image plane'
                    )
                lens = surfaces[:image_surface]
                pid = paraxial_image_distance(lens, wvl=wvl)
                self.rows[surf_idx].thickness = pid
        finally:
            self._resolving = False

    def _build_surface(self, row, P, R=None):
        """Build a posed Surface from a LensData row."""
        return Surface(
            shape=row.build_shape(), interaction=row.typ, P=P, R=R,
            material=None if row.material is MIRROR else row.material,
            bounding=row.bounding, aperture=row.aperture, grating=row.grating,
            edge=getattr(row, 'edge', None),
            coating=getattr(row, 'coating', None),
        )

    def _to_surfaces_axial(self):
        """Compile rows for a system without coordinate breaks."""
        surfaces = []
        z = 0.0
        sign = 1.0
        for row in self.rows:
            surfaces.append(self._build_surface(row, P=[0.0, 0.0, z]))
            if row.is_reflective:
                sign = -sign
            z = z + sign * row.thickness
        return surfaces

    def _to_surfaces_general(self):
        """Compile rows for a system containing coordinate breaks."""
        surfaces = []
        state = _FrameState()

        for idx, row in enumerate(self.rows):
            if isinstance(row, CoordBreak):
                self._apply_coordbreak(row, state)
                continue

            # surface placement (honor a one-shot DAR pose without perturbing
            # the running axis)
            if state.pending_pose is not None:
                o_s, Rgl_s = _apply_decenter_tilt(state.o, state.Rgl,
                                                  *state.pending_pose)
                state.pending_pose = None
            else:
                o_s, Rgl_s = state.o, state.Rgl
            surfaces.append(self._build_surface(
                row, P=o_s, R=_none_if_identity(Rgl_s)))
            state.frames[idx] = (o_s, Rgl_s)

            # fold the running frame at a reflecting surface so downstream rows
            # lie on the reflected beam.  A pending BEN fold (re-apply tilt +
            # auto-gamma) bends the axis along the reflected beam; otherwise the
            # normal-incidence fold reverses the local z axis.
            if row.is_reflective:
                if state.pending_fold is not None:
                    state.Rgl = _compose_global_to_local(
                        state.Rgl, R_rh(*state.pending_fold))
                    state.pending_fold = None
                else:
                    state.Rgl = _compose_global_to_local(state.Rgl, _FLIP_Z)
            # advance the gap along the (folded) local +z
            state.o = (state.o
                       + _local_to_global(state.Rgl)
                       @ _axial_step(row.thickness))

        return surfaces

    def _apply_coordbreak(self, cb, state):
        """Apply a coordinate break to the running layout state."""
        kind = cb.kind
        decenter = cb.decenter
        tilt = cb.tilt

        if kind == 'dar':
            # decenter-and-return: the decenter/tilt apply only to the next
            # surface; the running axis is untouched.  Advance the break gap
            # along the unperturbed axis.
            state.pending_pose = (decenter, tilt)
            state.advance(cb.thickness)
            return

        if kind == 'ret':
            # return-to-surface: restore the recorded frame of a prior row,
            # undoing intervening tilts/decenters/thicknesses.
            if cb.ret_target is None or cb.ret_target not in state.frames:
                raise ValueError(
                    f'RET coordinate break targets row {cb.ret_target!r}, '
                    'which has not been placed yet'
                )
            state.o, state.Rgl = state.frames[cb.ret_target]
            state.advance(cb.thickness)
            return

        if kind == 'rev':
            # reverse: inverse of a matching basic break (negated tilt applied
            # first, then the decenter subtracted), for coupling before/after.
            Rt = R_rh(tilt[0], tilt[1], tilt[2])
            state.Rgl = _compose_global_to_local(state.Rgl, _as_mat(Rt).T)
            state.o = state.o - _local_to_global(state.Rgl) @ decenter
            state.advance(cb.thickness)
            return

        if kind == 'ben':
            # decenter-and-bend: orient the mirror by the tilt now; register a
            # fold (re-apply alpha, beta + the auto-computed gamma) consumed at
            # the next reflecting surface to bend the axis by twice the tilt.
            state.o, state.Rgl = _apply_decenter_tilt(state.o, state.Rgl,
                                                      decenter, tilt)
            gamma = _ben_auto_gamma(tilt[2], tilt[1])
            state.pending_fold = (gamma, tilt[1], tilt[2])
            state.advance(cb.thickness)
            return

        if kind != 'basic':
            raise ValueError(
                f"unknown coordinate-break kind {kind!r}; expected one of "
                "'basic', 'dar', 'ret', 'rev', 'ben'"
            )

        # basic: cumulative decenter + tilt, persists for all succeeding rows
        state.o, state.Rgl = _apply_decenter_tilt(state.o, state.Rgl,
                                                  decenter, tilt)
        state.advance(cb.thickness)

    @property
    def surfaces(self):
        """The compiled surface list (cached; invalidated on edits)."""
        return self.to_surfaces()

    # -- sequence protocol (duck-type as a surface list) --
    def __len__(self):
        """Return the number of compiled surfaces."""
        return len(self.to_surfaces())

    def __iter__(self):
        """Iterate over compiled surfaces."""
        return iter(self.to_surfaces())

    def __getitem__(self, item):
        """Return one or more compiled surfaces by index."""
        return self.to_surfaces()[item]

    # -- optimizer surface --
    def pack(self):
        """Dense contiguous vector of the free DOFs."""
        return self.spec.pack()

    def update(self, x):
        """Scatter a free vector into the rows, resolve dependents, invalidate."""
        self.spec.scatter(x)
        self._resolve_dependencies()
        self._invalidate()
        return self

    def bounds(self):
        """(lo, hi) arrays parallel to the free vector."""
        return self.spec.bounds()

    # -- variable selection (category x surface-range) --
    def vary(self, category, surfaces='all'):
        """Mark a category of DOFs free over a range of surfaces."""
        slots = self._category_slots(category, surfaces)
        if category == 'thickness':
            self._clear_image_distance_solve_if_selected(slots)
        for slot in slots:
            if slot not in self._dependent:
                self.spec._free[slot] = True
        return self

    def freeze(self, category, surfaces='all'):
        """Inverse of vary."""
        for slot in self._category_slots(category, surfaces):
            self.spec._free.pop(slot, None)
        return self

    def vary_all(self):
        """Mark every scalar DOF free (except pickup/solve dependents)."""
        for slot in self.spec.slots():
            if slot not in self._dependent:
                self.spec._free[slot] = True
        return self

    def freeze_all(self):
        """Mark every scalar DOF fixed."""
        self.spec._free.clear()
        return self

    def constrain(self, category, *, lo=None, hi=None, relative=None,
                  surfaces='all'):
        """Set box bounds on a category of DOFs over a range of surfaces.

        Radius bounds are in radius and converted to curvature.  Relative
        bounds are anchored at the current nominal value.

        """
        if relative is None and lo is None and hi is None:
            raise ValueError('constrain needs lo/hi (absolute) or relative')
        is_radius = category == 'radius'
        for slot in self._category_slots(category, surfaces):
            nominal = float(self.spec.get_value(slot))
            bounds = _bounds_for_dof(nominal, lo, hi, relative, is_radius)
            if bounds is None:
                self.spec._bounds.pop(slot, None)
            else:
                self.spec._bounds[slot] = bounds
        return self

    # -- pickups and solves (dependent DOFs, resolved on compile) --
    def pickup(self, category, surface, *, from_surface, from_category=None,
               scale=1.0, offset=0.0):
        """Make a DOF a pickup of another: dependent = scale*source + offset."""
        from_category = from_category or category
        targets = self._category_slots(category, surface)
        sources = self._category_slots(from_category, from_surface)
        if not targets or not sources:
            raise ValueError(
                f'pickup found no {category!r}/{from_category!r} DOFs on the '
                'requested surfaces'
            )
        if len(targets) != len(sources):
            raise ValueError(
                f'pickup target ({len(targets)} DOFs) and source '
                f'({len(sources)} DOFs) must have equal length'
            )
        for t in targets:
            self.spec._free.pop(t, None)
            self._dependent.add(t)
        self._pickups.append((targets, sources, float(scale), float(offset)))
        self._invalidate()
        return self

    def solve_image_distance(self, surface=None, *, wavelength=None):
        """Solve a gap so the image plane sits at the paraxial image.

        The solved thickness is frozen until clear_image_distance_solve() or
        vary('thickness', ...) selects it.

        """
        if surface is None:
            evals = [i for i, r in enumerate(self.rows)
                     if isinstance(r, SurfaceRow)
                     and _map_stype(r.typ) == STYPE_EVAL]
            if not evals:
                raise ValueError(
                    'solve_image_distance needs an eval (image) plane to place'
                )
            eval_idx = max(evals)
            lens_rows = [i for i in range(eval_idx)
                         if isinstance(self.rows[i], SurfaceRow)]
            if not lens_rows:
                raise ValueError('no powered surface precedes the image plane')
            surface = max(lens_rows)
        else:
            surface = surface % len(self.rows)
        self._image_solve = (surface, wavelength)
        slot = ('thickness', surface, 0)
        self.spec._free.pop(slot, None)
        self._dependent.add(slot)
        self._invalidate()
        return self

    def clear_image_distance_solve(self):
        """Disable the active paraxial image-distance solve, if any."""
        if self._image_solve is None:
            return self
        surface, _ = self._image_solve
        slot = ('thickness', surface, 0)
        self._image_solve = None
        if slot not in self._pickup_target_slots():
            self._dependent.discard(slot)
        self._invalidate()
        return self

    def _pickup_target_slots(self):
        """Return slots currently driven by pickup dependencies."""
        out = set()
        for targets, _, _, _ in self._pickups:
            out.update(targets)
        return out

    def _clear_image_distance_solve_if_selected(self, slots):
        """Clear the image-distance solve when its thickness is selected."""
        if self._image_solve is None:
            return
        surface, _ = self._image_solve
        if ('thickness', surface, 0) in slots:
            self.clear_image_distance_solve()

    def _select_rows(self, surfaces):
        """Resolve a row selector to concrete row indices."""
        n = len(self.rows)
        if surfaces == 'all' or surfaces is None:
            return list(range(n))
        if isinstance(surfaces, slice):
            return list(range(*surfaces.indices(n)))
        if isinstance(surfaces, int):
            return [surfaces % n]
        return [int(s) % n for s in surfaces]

    def _category_slots(self, category, surfaces):
        """Return all slots selected by a category and row selector."""
        slots = []
        for r in self._select_rows(surfaces):
            row = self.rows[r]
            if category == 'thickness':
                slots.append(('thickness', r, 0))
            elif category in ('tilt', 'decenter'):
                if isinstance(row, CoordBreak):
                    for off in range(3):
                        slots.append((category, r, off))
            else:  # shape category
                if isinstance(row, SurfaceRow):
                    for off in row.categories.get(category, ()):
                        slots.append(('shape', r, off))
        return slots

    # -- listings (tabular inspection of the rows) --
    def list_surfaces(self, *, stop_index=None, unit=None):
        """Lens-data-editor surface table (see listings.surface_table)."""
        from .listings import surface_table
        return surface_table(self, stop_index=stop_index, unit=unit)

    def list_apertures(self):
        """Per-surface clear-aperture table (see listings.aperture_table)."""
        from .listings import aperture_table
        return aperture_table(self)

    def list_decenters(self):
        """Coordinate-break decenter / tilt table (see listings.decenter_table)."""
        from .listings import decenter_table
        return decenter_table(self)

    # -- copy --
    def copy(self):
        """Return a copy with cloned rows and preserved DOF selections."""
        new = LensData()
        new.rows = _RowList(new, [row.copy() for row in self.rows])
        new.spec._free = dict(self.spec._free)
        new.spec._bounds = dict(self.spec._bounds)
        new._pickups = [(list(t), list(s), sc, off)
                        for t, s, sc, off in self._pickups]
        new._image_solve = self._image_solve
        new._dependent = set(self._dependent)
        return new

    def __repr__(self):
        """Return a compact representation of the lens."""
        return (f'LensData(n_rows={len(self.rows)}, '
                f'n_free={len(self.spec.free_slots())})')


__all__ = ['LensData', 'SurfaceRow', 'CoordBreak', 'ParamSpec', 'R_rh']
