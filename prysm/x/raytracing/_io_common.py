"""Shared helpers for the raytracing IO parsers (Zemax, Code V, ...)."""


def read_text_or_path(path_or_text, is_text=False):
    """Return text and source path metadata for parser entry points."""
    if is_text:
        return path_or_text, None
    with open(path_or_text, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()
    return text, str(path_or_text)


def fields_from_xy(x_values, y_values, kind='angle', unit='deg'):
    """Build Field records from possibly uneven x/y field lists."""
    from .launch import Field

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


def parse_float(token):
    """Parse a numeric token from a prescription file.

    Accepts ``INF`` or ``INFINITY`` (case-insensitive) for +∞ — both Zemax
    and Code V emit these for unbounded curvature radii.

    """
    t = token.strip()
    if t.upper() in ('INF', 'INFINITY'):
        return float('inf')
    return float(t)
