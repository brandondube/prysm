"""Tabular listings of a LensData: surfaces, clear apertures, coordinate breaks.

Numbers-only table builders that mirror the lens-data-editor views of
commercial codes.  Each returns a vanilla __slots__ table object whose
__repr__ is a clean aligned table (modeled on paraxial.FirstOrderProperties and
aberrations.SeidelResult).  LensData exposes thin delegators (list_surfaces /
list_apertures / list_decenters); OpticalSystem forwards its stop_index and unit
to the surface table.

No plotting and no matplotlib here -- these are text tables for inspection.
"""

from .spencer_and_murty import STYPE_EVAL, STYPE_REFLECT, STYPE_REFRACT
from .surfaces import _map_stype
from ..materials import MIRROR, air, vacuum
from .lensdata import CoordBreak, SurfaceRow


def _radius_str(c):
    """Radius 1/c as a string, 'inf' for a flat (c == 0) surface."""
    c = float(c)
    if c == 0.0:
        return 'inf'
    return f'{1.0 / c:.6g}'


def _type_str(typ):
    """Short refr/refl/eval label for a surface interaction type."""
    s = _map_stype(typ)
    if s == STYPE_REFRACT:
        return 'refr'
    if s == STYPE_REFLECT:
        return 'refl'
    if s == STYPE_EVAL:
        return 'eval'
    return str(typ)


def material_str(material, typ):
    """Display string for a row material.

    MIRROR (or a reflective surface) shows as MIRROR; a constant index shows as
    its numeric value; a named glass callable shows its name; air / None is
    blank.
    """
    if _map_stype(typ) == STYPE_REFLECT or material is MIRROR \
            or material == MIRROR:
        return 'MIRROR'
    if material is None or material is air or material is vacuum:
        return ''
    name = getattr(material, 'name', None)
    if name:
        return str(name)
    if callable(material):
        return 'n(wvl)'
    try:
        return f'{float(material):.5g}'
    except (TypeError, ValueError):
        return str(material)


def surface_row_mappings(lensdata):
    """Map raw rows to compiled surface and exported Zemax surface indices."""
    records = []
    surface_index = 0
    zemax_surface_number = 1
    for row_index, row in enumerate(lensdata.rows):
        if isinstance(row, SurfaceRow):
            compiled = surface_index
            surface_index += 1
        else:
            compiled = None
        records.append({
            'row_index': row_index,
            'surface_index': compiled,
            'zemax_surface_number': zemax_surface_number,
        })
        zemax_surface_number += 1
    return records


class SurfaceTable:
    """Lens-data-editor table: index, type, radius, conic, thickness, material, semidiameter."""

    __slots__ = ('records', 'unit', 'stop_index')

    def __init__(self, records, unit=None, stop_index=None):
        self.records = records
        self.unit = unit
        self.stop_index = stop_index

    def __repr__(self):
        unit = f' [{self.unit}]' if self.unit else ''
        header = (f'  {"#":>3s} {"":>1s} {"type":>6s} {"radius":>12s} '
                  f'{"conic":>10s} {"thickness":>12s} {"material":>10s} '
                  f'{"semidia":>10s} {"coat":>5s}')
        lines = [f'SurfaceTable{unit}', header,
                 '  ' + '-' * (len(header) - 2)]
        for r in self.records:
            mark = '*' if r['stop'] else ' '
            sd = '' if r['semidiameter'] is None else f'{r["semidiameter"]:.6g}'
            coat = 'Y' if r.get('coating') else ''
            lines.append(
                f'  {r["index"]:>3d} {mark:>1s} {r["type"]:>6s} '
                f'{r["radius"]:>12s} {r["conic"]:>10s} '
                f'{r["thickness"]:>12.6g} {r["material"]:>10s} {sd:>10s} '
                f'{coat:>5s}')
        return '\n'.join(lines)


class ApertureTable:
    """Per-surface clear-aperture table: semidiameter, aperture kind, bounding outer radius."""

    __slots__ = ('records',)

    def __init__(self, records):
        self.records = records

    def __repr__(self):
        header = (f'  {"#":>3s} {"semidia":>12s} {"aperture":>16s} '
                  f'{"outer_radius":>14s}')
        lines = ['ApertureTable', header, '  ' + '-' * (len(header) - 2)]
        for r in self.records:
            sd = '' if r['semidiameter'] is None else f'{r["semidiameter"]:.6g}'
            outer = ('' if r['outer_radius'] is None
                     else f'{r["outer_radius"]:.6g}')
            lines.append(
                f'  {r["index"]:>3d} {sd:>12s} {r["aperture"]:>16s} '
                f'{outer:>14s}')
        return '\n'.join(lines)


class DecenterTable:
    """Coordinate-break table: index, dx/dy/dz, rz/ry/rx, kind."""

    __slots__ = ('records',)

    def __init__(self, records):
        self.records = records

    def __repr__(self):
        if not self.records:
            return 'DecenterTable (no coordinate breaks)'
        header = (f'  {"#":>3s} {"dx":>9s} {"dy":>9s} {"dz":>9s} '
                  f'{"rz":>9s} {"ry":>9s} {"rx":>9s} {"kind":>7s}')
        lines = ['DecenterTable', header, '  ' + '-' * (len(header) - 2)]
        for r in self.records:
            lines.append(
                f'  {r["index"]:>3d} {r["dx"]:>9.4g} {r["dy"]:>9.4g} '
                f'{r["dz"]:>9.4g} {r["rz"]:>9.4g} {r["ry"]:>9.4g} '
                f'{r["rx"]:>9.4g} {r["kind"]:>7s}')
        return '\n'.join(lines)


def surface_table(lensdata, *, stop_index=None, unit=None):
    """Build the lens-data-editor surface table for a LensData."""
    records = []
    mappings = surface_row_mappings(lensdata)
    for mapping, row in zip(mappings, lensdata.rows):
        i = mapping['row_index']
        surface_index = mapping['surface_index']
        is_stop = surface_index == stop_index
        if isinstance(row, CoordBreak):
            records.append({
                'index': i, 'type': f'CB:{row.kind}', 'radius': '',
                'conic': '', 'thickness': float(row.thickness),
                'material': '', 'semidiameter': None, 'coating': False,
                'surface_index': surface_index, 'stop': is_stop,
            })
            continue
        shape = row.build_shape()
        params = shape.params or {}
        # Ask the shape's descriptor for its canonical radius/conic DOF instead
        # of guessing c vs c_y (deepening 02).
        cats = type(shape).CATEGORIES
        radius_keys = cats.get('radius') or cats.get('curvature') or ()
        conic_keys = cats.get('conic') or ()
        c = params.get(radius_keys[-1], 0.0) if radius_keys else 0.0
        k = params.get(conic_keys[-1], 0.0) if conic_keys else 0.0
        records.append({
            'index': i, 'type': _type_str(row.typ),
            'radius': _radius_str(c),
            'conic': f'{float(k):.6g}',
            'thickness': float(row.thickness),
            'material': material_str(row.material, row.typ),
            'semidiameter': (None if row.semidiameter is None
                             else float(row.semidiameter)),
            'coating': getattr(row, 'coating', None) is not None,
            'surface_index': surface_index, 'stop': is_stop,
        })
    return SurfaceTable(records, unit=unit, stop_index=stop_index)


def aperture_table(lensdata):
    """Build the per-surface clear-aperture table for a LensData."""
    records = []
    for i, row in enumerate(lensdata.rows):
        if isinstance(row, CoordBreak):
            continue
        aperture = row.aperture
        kind = getattr(aperture, '__name__', None) or (
            type(aperture).__name__ if aperture is not None else '')
        bounding = row.bounding or {}
        records.append({
            'index': i,
            'semidiameter': (None if row.semidiameter is None
                             else float(row.semidiameter)),
            'aperture': kind,
            'outer_radius': bounding.get('outer_radius'),
        })
    return ApertureTable(records)


def decenter_table(lensdata):
    """Build the coordinate-break decenter / tilt table for a LensData."""
    records = []
    for i, row in enumerate(lensdata.rows):
        if not isinstance(row, CoordBreak):
            continue
        dx, dy, dz = (float(v) for v in row.decenter)
        rz, ry, rx = (float(v) for v in row.tilt)
        records.append({
            'index': i, 'dx': dx, 'dy': dy, 'dz': dz,
            'rz': rz, 'ry': ry, 'rx': rx, 'kind': row.kind,
        })
    return DecenterTable(records)


__all__ = ['surface_table', 'aperture_table', 'decenter_table',
           'SurfaceTable', 'ApertureTable', 'DecenterTable', 'material_str']
