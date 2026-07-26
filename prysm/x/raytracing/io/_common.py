"""Shared helpers for the raytracing IO parsers (Zemax, Code V, ...)."""

import math
import re
import warnings


# vignetting-factor tokens of the supported decks.  Code V: VUX/VUY/VLX/VLY.
# Zemax: VDX/VDY/VCX/VCY/VAN (with a trailing surface index).
_VIGNETTING_RE = re.compile(
    r'\b(VUX|VUY|VLX|VLY|VDX|VDY|VCX|VCY|VAN)\b', re.IGNORECASE)


def warn_vignetting_ignored(text, format_name):
    """Warn once if a prescription declares vignetting factors.

    prysm models real vignetting by clipping at the per-surface clear
    apertures, so the affine vignetting factors (decenter/scale of the
    normalized pupil) of Code V / Zemax are intentionally ignored; this emits a
    one-time note so the user knows they were dropped.
    """
    if _VIGNETTING_RE.search(text or ''):
        warnings.warn(
            f'{format_name} vignetting factors were found and ignored; prysm '
            'models vignetting by clipping at the per-surface clear apertures '
            'rather than via affine pupil-scaling factors.',
            stacklevel=3)


def read_text_or_path(path_or_text, is_text=False):
    """Return text and source path metadata for parser entry points."""
    if is_text:
        return path_or_text, None
    with open(path_or_text, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()
    return text, str(path_or_text)


def fields_from_xy(x_values, y_values, kind='angle', unit='deg',
                   object_z=None, length_scale=1.0, vignetting=None):
    """Build Field records from possibly uneven x/y field lists."""
    from ..launch import Field

    x_values = list(x_values)
    y_values = list(y_values)
    if not x_values and not y_values:
        return []
    n = max(len(x_values), len(y_values))
    if not x_values:
        x_values = [0.0] * n
    if not y_values:
        y_values = [0.0] * n
    if len(x_values) < n:
        x_values += [0.0] * (n - len(x_values))
    if len(y_values) < n:
        y_values += [0.0] * (n - len(y_values))
    if vignetting is None:
        vignetting = [None] * n
    else:
        vignetting = list(vignetting)
        if len(vignetting) < n:
            vignetting += [None] * (n - len(vignetting))
    if kind == 'angle':
        return [Field(hx, hy, kind='angle', unit=unit, vignetting=vig)
                for hx, hy, vig in zip(x_values, y_values, vignetting)]
    object_z = scale_length_to_mm(object_z, length_scale)
    return [Field(scale_length_to_mm(hx, length_scale),
                  scale_length_to_mm(hy, length_scale),
                  kind=kind, object_z=object_z, vignetting=vig)
            for hx, hy, vig in zip(x_values, y_values, vignetting)]


_UNIT_TO_MM = {
    'mm': 1.0,
    'millimeter': 1.0,
    'millimeters': 1.0,
    'cm': 10.0,
    'centimeter': 10.0,
    'centimeters': 10.0,
    'm': 1000.0,
    'meter': 1000.0,
    'meters': 1000.0,
    'in': 25.4,
    'inch': 25.4,
    'inches': 25.4,
    'ft': 304.8,
    'foot': 304.8,
    'feet': 304.8,
}


def length_scale_to_mm(unit):
    """Return the factor that converts one source length unit to millimeters."""
    if unit is None:
        return 1.0
    key = str(unit).strip().lower()
    try:
        return _UNIT_TO_MM[key]
    except KeyError as e:
        raise ValueError(
            f'unsupported prescription length unit {unit!r}; supported units '
            'are mm, cm, m, in, and ft'
        ) from e


def scale_length_to_mm(value, scale):
    """Scale a finite length-like value to millimeters."""
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        return value
    return value * scale


def _scale_curvature(value, scale):
    """Scale inverse-length curvature into inverse millimeters."""
    return float(value) / scale


def _scale_even_asphere_coefs(coefs, scale):
    """Scale even-asphere coefficients from source units to millimeters."""
    scaled = []
    for i, coef in enumerate(coefs, start=1):
        power = 2 * (i + 1)  # i=1 is rho**4
        scaled.append(float(coef) / (scale ** (power - 1)))
    return tuple(scaled)


def scale_surface_params_to_mm(kind, params, scale):
    """Scale normalized SurfaceSpec shape params from source units to mm."""
    if scale == 1.0:
        return dict(params)
    out = dict(params)
    if kind in ('conic', 'even_asphere', 'zernike', 'xy'):
        out['c'] = _scale_curvature(out.get('c', 0.0), scale)
    if kind == 'even_asphere':
        out['coefs'] = _scale_even_asphere_coefs(out.get('coefs', ()), scale)
    elif kind == 'toroid':
        out['c_x'] = _scale_curvature(out['c_x'], scale)
        out['c_y'] = _scale_curvature(out['c_y'], scale)
        out['coefs_y'] = _scale_even_asphere_coefs(
            out.get('coefs_y', ()), scale)
    elif kind == 'biconic':
        out['c_x'] = _scale_curvature(out['c_x'], scale)
        out['c_y'] = _scale_curvature(out['c_y'], scale)
    elif kind in ('zernike', 'xy'):
        out['normalization_radius'] = scale_length_to_mm(
            out['normalization_radius'], scale)
        out['coefs'] = tuple(float(c) * scale for c in out.get('coefs', ()))
    return out


def aperture_kwargs_from_radii(outer_radius, scale, inner_radius=None):
    """LensData.add keyword args for a circular or annular clear aperture.

    A semi-diameter becomes a circular clip (drawn at clip x oversize); an
    annular pair becomes an annular clip with an explicit annular extent so the
    bore draws.
    """
    outer = scale_length_to_mm(outer_radius, scale)
    if outer is None:
        return {}
    inner = scale_length_to_mm(inner_radius, scale)
    from ..aperture import Aperture, annular_aperture, CircularExtent
    if inner is None:
        return {'aperture': Aperture(clip=float(outer))}
    if inner < 0 or outer <= 0 or inner >= outer:
        raise ValueError(
            'clear-aperture radii must satisfy 0 <= inner < outer'
        )
    return {'aperture': Aperture(
        clip=annular_aperture(inner, outer),
        extent=CircularExtent(float(outer), inner_radius=float(inner)),
    )}


def fold_sign(n_refl):
    """Gap sign given the number of preceding reflections.

    Zemax and Code V encode post-mirror gaps as negative thicknesses on an
    unfolded axis; LensData folds the frame at each reflection and keeps
    thickness positive.  The conversion negates the gap once per preceding
    reflection, so the sign alternates with the parity of n_refl.  Shared by
    both readers (decode) and both writers (encode), which are inverses of one
    another, so the fold convention lives in exactly one place.
    """
    return -1.0 if (n_refl % 2) else 1.0


def writable_shape_or_raise(shape_kind, is_eval, writer):
    """Reject surface rows a prescription writer would serialize lossily.

    Only Conic, Sphere, and Plane round-trip losslessly through the
    rotationally symmetric Zemax / Code V writers; eval (image-plane) rows
    carry no shape and are always allowed.  writer is the calling function
    name, interpolated into the error message.
    """
    if is_eval:
        return
    from ..surfaces import Conic, Plane, Sphere
    if shape_kind in (Conic, Plane, Sphere):
        return
    raise NotImplementedError(
        f'{writer} cannot export {shape_kind.__name__} without losing shape '
        'data; supported writer shapes are Conic, Sphere, and Plane.'
    )


def aperture_export_radii(aperture, *, allow_annular):
    """Return strict `(outer, inner)` clip radii for a supported aperture."""
    from ..aperture import AnnularClip, CircularClip
    clip = aperture.clip
    if clip is None:
        if (aperture.extent is not None or aperture.substrate is not None
                or aperture.features):
            raise ValueError('cosmetic extent/substrate/features are unsupported')
        return None, None
    if isinstance(clip, CircularClip):
        if clip.x0 != 0.0 or clip.y0 != 0.0:
            raise ValueError('decentered circular clips are unsupported')
        inner = None
        outer = clip.radius
    elif isinstance(clip, AnnularClip) and allow_annular:
        if clip.x0 != 0.0 or clip.y0 != 0.0:
            raise ValueError('decentered annular clips are unsupported')
        inner = clip.inner_radius
        outer = clip.outer_radius
    else:
        raise ValueError(
            f'{type(clip).__name__} clips are unsupported by this writer')
    if aperture.substrate is not None or aperture.features:
        raise ValueError('substrate and edge features are unsupported')
    extent = aperture.extent
    if extent is not None:
        if (float(extent.outer_radius) != float(outer)
                or float(extent.inner_radius) != float(inner or 0.0)):
            raise ValueError('drawn extent differs from the exported clip')
    return float(outer), None if inner is None else float(inner)


def preflight_export(system, writer):
    """Aggregate every semantic feature a strict writer cannot represent."""
    from ..lensdata import CoordBreak, SurfaceRow
    from ..spencer_and_murty import STYPE_REFLECT, _is_measurement_surf
    from ..surfaces import Conic, Plane, Sphere, _map_stype
    from ... import materials

    if writer not in ('write_zmx', 'write_seq'):
        raise ValueError(f'unknown writer {writer!r}')
    allow_annular = writer == 'write_seq'
    problems = []
    lens = getattr(system, 'lens', system)
    rows = getattr(lens, 'rows', None)
    if rows is None:
        raise TypeError(f'{writer} requires LensData or OpticalSystem')

    for row_index, row in enumerate(rows):
        if isinstance(row, CoordBreak):
            allowed = ('basic',) if writer == 'write_zmx' else ('basic', 'dar')
            if row.kind not in allowed:
                problems.append(
                    f'row {row_index}: CoordBreak kind {row.kind!r}')
            if row.ret_target is not None:
                problems.append(f'row {row_index}: CoordBreak ret_target')
            continue
        if not isinstance(row, SurfaceRow):
            problems.append(f'row {row_index}: unknown row type')
            continue
        stype = _map_stype(row.typ)
        if (not _is_measurement_surf(stype)
                and row.shape_kind not in (Conic, Plane, Sphere)):
            problems.append(
                f'row {row_index}: shape {row.shape_kind.__name__}')
        if row.grating is not None:
            problems.append(f'row {row_index}: OPLFunc/grating')
        if row.coating is not None:
            problems.append(f'row {row_index}: coating stack')
        try:
            aperture_export_radii(row.aperture,
                                  allow_annular=allow_annular)
        except ValueError as exc:
            problems.append(f'row {row_index}: aperture ({exc})')
        if stype != STYPE_REFLECT and row.material not in (
                None, materials.air, materials.vacuum):
            page = getattr(row.material, 'page_info', None)
            if not page or not page.get('page'):
                problems.append(
                    f'row {row_index}: material has no external catalog name')

    aperture = getattr(system, 'aperture', None)
    if aperture is not None and getattr(aperture, 'mode', None) != 'EPD':
        problems.append(
            f'system aperture mode {getattr(aperture, "mode", None)!r}')
    fields = list(getattr(system, 'fields', ()) or ())
    for i, field in enumerate(fields):
        if field.kind == 'angle' and field.unit != 'deg':
            problems.append(f'field {i}: angular unit {field.unit!r}')
        if writer == 'write_seq' and field.kind != 'angle':
            problems.append(f'field {i}: object-height field')
        if writer == 'write_zmx' and field.vignetting is not None:
            problems.append(f'field {i}: vignetting factors')
    extras = getattr(system, 'extras', None) or {}
    unsupported_extras = sorted(set(extras) - {'VERS', 'MODE'})
    if unsupported_extras:
        problems.append('system extras: ' + ', '.join(unsupported_extras))
    if problems:
        raise NotImplementedError(
            f'{writer} cannot losslessly export: ' + '; '.join(problems))


def parse_float(token):
    """Parse a numeric token from a prescription file.

    Accepts `INF` or `INFINITY` (case-insensitive) for +∞ — both Zemax
    and Code V emit these for unbounded curvature radii.

    """
    t = token.strip()
    if t.upper() in ('INF', 'INFINITY'):
        return float('inf')
    return float(t)
