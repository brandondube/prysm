"""Shared field/wavelength trace-grid helpers."""

from .spencer_and_murty import raytrace, valid_mask
from .launch import Field, launch
from ._meta import system_wavelength, system_epd


def _resolve_fields(prescription, fields):
    """Fields to evaluate, defaulting to the system's FieldSet, else on-axis."""
    if fields is not None:
        return list(fields)
    sys_fields = getattr(prescription, 'fields', None)
    if sys_fields is not None and len(sys_fields) > 0:
        return list(sys_fields)
    return [Field(0.0, 0.0)]


def _resolve_wavelengths(prescription, wavelengths):
    """Wavelengths (microns) to evaluate, defaulting to the system's set."""
    if wavelengths is not None:
        return [float(w) for w in wavelengths]
    wv = getattr(prescription, 'wavelengths', None)
    if wv:
        values = wv.values() if hasattr(wv, 'values') else wv
        return [float(w) for w in values]
    return [system_wavelength(prescription, None)]


def _require_epd(prescription, epd, wvl=None):
    """Resolve epd from an explicit value or the system; error if neither."""
    epd = system_epd(prescription, epd, wvl)
    if epd is None:
        raise TypeError(
            'epd is required; pass epd=... or supply an OpticalSystem whose '
            'aperture spec resolves it.'
        )
    return epd


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


def _launch_trace(prescription, field, wvl, sampling, *, epd, pupil_z, aim_to,
                  trace_fn):
    """Resolve epd, launch one sampling, trace it, and mask the invalid rays."""
    epd = _require_epd(prescription, epd, wvl)
    P, S = launch(prescription, field, wvl, sampling, epd=epd, pupil_z=pupil_z,
                  aim_to=aim_to)
    trace = trace_fn(prescription, P, S, wvl)
    valid = valid_mask(trace.status, trace.P[-1])
    return epd, P, S, trace, valid


def trace_cell(prescription, field, wvl, sampling, *, epd=None, pupil_z=None,
               aim_to=None, trace_fn=raytrace):
    """Launch and trace one (field, wavelength) bundle.

    Returns
    -------
    TraceRecord
        with i = j = 0.

    """
    epd, P, S, trace, valid = _launch_trace(
        prescription, field, wvl, sampling, epd=epd, pupil_z=pupil_z,
        aim_to=aim_to, trace_fn=trace_fn)
    return TraceRecord(0, 0, field, wvl, epd, P, S, trace, valid)


def iter_trace_grid(prescription, fields, wavelengths, sampling, *,
                    epd=None, pupil_z=None, aim_to=None, trace_fn=raytrace):
    """Trace one pupil sampling over every field and wavelength.

    Parameters
    ----------
    prescription : sequence of Surface or OpticalSystem
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
    fields = _resolve_fields(prescription, fields)
    wavelengths = _resolve_wavelengths(prescription, wavelengths)
    for i, field in enumerate(fields):
        for j, wvl in enumerate(wavelengths):
            epd_w, P, S, trace, valid = _launch_trace(
                prescription, field, wvl, sampling, epd=epd, pupil_z=pupil_z,
                aim_to=aim_to, trace_fn=trace_fn)
            yield TraceRecord(i, j, field, wvl, epd_w, P, S, trace, valid)
