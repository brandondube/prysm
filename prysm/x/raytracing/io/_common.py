"""Shared helpers for the raytracing IO parsers (Zemax, Code V, ...)."""

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


def fields_from_xy(x_values, y_values, kind='angle', unit='deg'):
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
    if kind == 'angle':
        return [Field(hx, hy, kind='angle', unit=unit)
                for hx, hy in zip(x_values, y_values)]
    return [Field(hx, hy, kind=kind) for hx, hy in zip(x_values, y_values)]


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
