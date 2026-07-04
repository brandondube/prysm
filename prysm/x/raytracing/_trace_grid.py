"""Shared field/wavelength trace-grid helpers."""

import math

from prysm.mathops import np

from .spencer_and_murty import raytrace, valid_mask
from .launch import Field, Sampling, launch
from ._resolve import resolve_wavelength, trace_context


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

    When fields is None and the system fields are homogeneous, return
    samples along the largest field's direction.  Otherwise return the
    resolved fields.

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
    try:
        return [resolve_wavelength(system, None)]
    except ValueError:
        raise TypeError(
            'wavelengths is required for a bare surface sequence; only an '
            'OpticalSystem defaults the wavelength set.'
        ) from None


def _require_epd(system, epd, wvl=None):
    """Resolve epd from an explicit value or the system; error if neither."""
    if epd is None:
        epd = trace_context(system, wvl, chief=True).epd
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


class LayoutRecord:
    """One traced layout fan: the field, its trace, and the valid mask."""

    __slots__ = ('field', 'trace', 'valid')

    def __init__(self, field, trace, valid):
        self.field = field
        self.trace = trace
        self.valid = valid


class _OutlineTrace:
    """Minimal P/S carrier so plot_optics can size glass over many fields."""

    __slots__ = ('P', 'S')

    def __init__(self, P, S):
        self.P = P
        self.S = S


def _valid_only_positions(trace):
    """Position history with clipped/missed rays NaN'd out (for glass sizing)."""
    P = np.array(trace.P)
    mask = valid_mask(trace.status, P[-1])
    if mask is not None:
        P[:, ~mask, :] = np.nan
    return P


def layout_records(system, fields=None, wavelength=None, sampling=None,
                   axis='y'):
    """Trace one pupil fan per field for a 2D layout drawing.

    Parameters
    ----------
    system : sequence of Surface or OpticalSystem
        the optical system.
    fields : iterable of Field, optional
        fields to trace; None defaults to the system FieldSet, else on-axis.
    wavelength : float, optional
        in microns; None resolves to the system reference.
    sampling : Sampling or int, optional
        pupil sampling per fan; an int is shorthand for a fan of that many
        rays along axis, None a 3-ray fan.
    axis : str, optional
        fan axis 'y' or 'x', used when sampling is None or an int.

    Returns
    -------
    records : list of LayoutRecord
        one traced fan per field.
    outline : _OutlineTrace
        every field's footprint concatenated, for glass sizing.

    """
    wvl = resolve_wavelength(system, wavelength)
    fields = _resolve_fields(system, fields)
    if sampling is None:
        sampling = Sampling.fan(n=3, axis=axis)
    elif isinstance(sampling, int):
        sampling = Sampling.fan(n=int(sampling), axis=axis)
    records = []
    for field in fields:
        trace = raytrace(system, *launch(system, field, wvl, sampling,
                                         drop_unaimed=True), wvl)
        records.append(LayoutRecord(field, trace,
                                    valid_mask(trace.status, trace.P[-1])))
    # Size glass from every valid field footprint; clipped rays are NaN'd out.
    outline = _OutlineTrace(
        np.concatenate([_valid_only_positions(r.trace) for r in records],
                       axis=1),
        np.concatenate([np.asarray(r.trace.S) for r in records], axis=1),
    )
    return records, outline
