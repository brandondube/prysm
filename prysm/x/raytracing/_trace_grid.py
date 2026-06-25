"""Shared field/wavelength trace-grid helpers."""

import math

from .spencer_and_murty import raytrace, valid_mask
from .launch import Field, launch


def _resolve_fields(system, fields):
    """Fields to evaluate, defaulting to the system's FieldSet, else on-axis."""
    if fields is not None:
        return list(fields)
    sys_fields = getattr(system, 'fields', None)
    if sys_fields is not None and len(sys_fields) > 0:
        return list(sys_fields)
    return [Field(0.0, 0.0)]


def field_sweep(system, fields=None, samples=101):
    """Dense field samples spanning the system field set.

    A drop-in upgrade of _resolve_fields for analyses that are smooth
    functions of field (field curvature, distortion, lateral color):
    explicitly given fields pass through verbatim, but when fields is None
    the defaulted system FieldSet is replaced by samples points spaced
    linearly from the smallest to the largest field magnitude, along the
    direction of the largest field.  The dense points carry no per-field
    vignetting; the chief-ray analyses this serves launch at the pupil
    center, which vignetting factors do not move.

    Falls back to the discrete _resolve_fields result whenever a sweep is
    not well defined: a heterogeneous field set (mixed kind, angular unit,
    or object plane) or an all-on-axis one.  A field set with a single
    distinct magnitude sweeps from on axis out to it.

    Parameters
    ----------
    system : sequence of Surface or OpticalSystem
        the optical system.
    fields : iterable of Field, optional
        explicit field points, returned verbatim; None triggers the sweep.
    samples : int, optional
        number of sweep points when fields is None.

    Returns
    -------
    list of Field
        the fields to evaluate.

    """
    base = _resolve_fields(system, fields)
    if fields is not None or len(base) == 0:
        return base
    kinds = {f.kind for f in base}
    if len(kinds) != 1:
        return base
    kind = kinds.pop()
    if kind == 'angle':
        if len({f.unit for f in base}) != 1:
            return base
        object_z = None
    else:
        if len({f.object_z for f in base}) != 1:
            return base
        object_z = base[0].object_z
    unit = base[0].unit
    mags = [math.hypot(f.hx, f.hy) for f in base]
    mmax = max(mags)
    if mmax <= 0.0:
        return base
    outer = base[mags.index(mmax)]
    ux = outer.hx / mmax
    uy = outer.hy / mmax
    mmin = min(mags)
    if mmin >= mmax:
        mmin = 0.0
    samples = max(int(samples), 2)
    step = (mmax - mmin) / (samples - 1)
    return [
        Field(ux * (mmin + step * i), uy * (mmin + step * i),
              kind=kind, unit=unit, object_z=object_z)
        for i in range(samples)
    ]


def _resolve_wavelengths(system, wavelengths):
    """Wavelengths (microns) to evaluate, defaulting to the system's set."""
    if wavelengths is not None:
        return [float(w) for w in wavelengths]
    wv = getattr(system, 'wavelengths', None)
    if wv is not None and len(wv):
        return [float(w) for w in wv]
    resolver = getattr(system, 'wavelength', None)
    if callable(resolver):
        return [float(resolver(None))]
    raise TypeError(
        'wavelengths is required for a bare surface sequence; only an '
        'OpticalSystem defaults the wavelength set.'
    )


def _require_epd(system, epd, wvl=None):
    """Resolve epd from an explicit value or the system; error if neither."""
    if epd is None:
        resolver = getattr(system, 'entrance_pupil_diameter', None)
        if callable(resolver):
            epd = resolver(wvl)
    if epd is None:
        raise TypeError(
            'epd is required; pass epd=... or supply an OpticalSystem whose '
            'aperture spec resolves it.'
        )
    return float(epd)


class TraceRecord:
    """One traced (field, wavelength) cell.

    Attributes
    ----------
    i, j : int
        field index and wavelength index within the grid.
    field : Field
        the field point traced.
    wvl : float
        wavelength in microns.
    epd : float
        the entrance-pupil diameter the bundle was sized with.
    P, S : ndarray, shape (N, 3)
        launch positions and direction cosines.
    trace : RayTraceResult
        the raytrace result.
    valid : ndarray, shape (N,)
        per-ray validity mask.

    """

    __slots__ = ('i', 'j', 'field', 'wvl', 'epd', 'P', 'S', 'trace', 'valid')

    def __init__(self, i, j, field, wvl, epd, P, S, trace, valid):
        self.i = i
        self.j = j
        self.field = field
        self.wvl = wvl
        self.epd = epd
        self.P = P
        self.S = S
        self.trace = trace
        self.valid = valid


def _launch_trace(system, field, wvl, sampling, *, epd, pupil_z, aim_to,
                  trace_fn):
    """Resolve epd, launch one sampling, trace it, and mask the invalid rays."""
    epd = _require_epd(system, epd, wvl)
    # NaN unaimed real-aiming rays so fans/spots truncate at vignetting.
    P, S = launch(system, field, wvl, sampling, epd=epd, pupil_z=pupil_z,
                  aim_to=aim_to, drop_unaimed=True)
    trace = trace_fn(system, P, S, wvl)
    valid = valid_mask(trace.status, trace.P[-1])
    return epd, P, S, trace, valid


def trace_cell(system, field, wvl, sampling, *, epd=None, pupil_z=None,
               aim_to=None, trace_fn=raytrace):
    """Launch and trace one (field, wavelength) bundle.

    Returns
    -------
    TraceRecord
        with i = j = 0.

    """
    epd, P, S, trace, valid = _launch_trace(
        system, field, wvl, sampling, epd=epd, pupil_z=pupil_z,
        aim_to=aim_to, trace_fn=trace_fn)
    return TraceRecord(0, 0, field, wvl, epd, P, S, trace, valid)


def iter_trace_grid(system, fields, wavelengths, sampling, *,
                    epd=None, pupil_z=None, aim_to=None, trace_fn=raytrace):
    """Trace one pupil sampling over every field and wavelength.

    Parameters
    ----------
    system : sequence of Surface or OpticalSystem
        the optical system.
    fields : iterable of Field or None
        evaluated fields; None defaults to the system FieldSet, else on-axis.
    wavelengths : iterable of float or None
        wavelengths in microns; None defaults to the system set, else the
        reference wavelength.
    sampling : Sampling
        the shared pupil sampling launched in every cell.
    epd : float, optional
        entrance-pupil diameter; resolved per wavelength via _require_epd.
    pupil_z : float, optional
        launch-plane z; forwarded to launch.
    aim_to : int, optional
        explicit per-ray stop-aiming surface; forwarded to launch.
    trace_fn : callable, optional
        trace step; defaults to the geometric raytrace kernel.

    Yields
    ------
    TraceRecord
        one per cell, in row-major order.

    """
    fields = _resolve_fields(system, fields)
    wavelengths = _resolve_wavelengths(system, wavelengths)
    for i, field in enumerate(fields):
        for j, wvl in enumerate(wavelengths):
            epd_w, P, S, trace, valid = _launch_trace(
                system, field, wvl, sampling, epd=epd, pupil_z=pupil_z,
                aim_to=aim_to, trace_fn=trace_fn)
            yield TraceRecord(i, j, field, wvl, epd_w, P, S, trace, valid)
