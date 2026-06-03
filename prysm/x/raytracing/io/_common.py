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
    """LensData.add keyword args for a circular or annular clear aperture."""
    outer = scale_length_to_mm(outer_radius, scale)
    if outer is None:
        return {}
    inner = scale_length_to_mm(inner_radius, scale)
    if inner is None:
        return {'semidiameter': outer}
    if inner < 0 or outer <= 0 or inner >= outer:
        raise ValueError(
            'clear-aperture radii must satisfy 0 <= inner < outer'
        )
    from ..surfaces import annular_aperture
    return {
        'semidiameter': outer,
        'aperture': annular_aperture(inner, outer),
        'bounding': {'inner_radius': float(inner), 'outer_radius': float(outer)},
    }


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


def parse_float(token):
    """Parse a numeric token from a prescription file.

    Accepts `INF` or `INFINITY` (case-insensitive) for +∞ — both Zemax
    and Code V emit these for unbounded curvature radii.

    """
    t = token.strip()
    if t.upper() in ('INF', 'INFINITY'):
        return float('inf')
    return float(t)
