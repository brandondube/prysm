"""Direct tests for the traced (field x wavelength) grid iterator.

The grid analyses (ray fans, OPD fans, spot diagrams, lateral color, distortion)
are extractors over iter_trace_grid; this file is the iterator's own test
surface, so those analyses only need to test their extractor.
"""
import numpy as np
import pytest

from tests.x.raytracing.surface_helpers import conic, plane

from prysm.x.raytracing import LensData, OpticalSystem, ApertureSpec, Field
from prysm.x.raytracing.surfaces import Conic, Plane, circular_aperture
from prysm.x.raytracing.launch import Sampling, launch
from prysm.x.raytracing.spencer_and_murty import raytrace
from prysm.x.raytracing.opt import _valid_mask
from prysm.x.raytracing._trace_grid import (
    TraceRecord,
    trace_cell,
    iter_trace_grid,
    _resolve_fields,
    _resolve_wavelengths,
    _require_epd,
)


def _singlet_system(fields=None, wavelengths=None, ref='d'):
    """Sphere/sphere singlet with system metadata (stop at the first surface)."""
    lens = LensData()
    (lens.add(Conic(1 / 50.0, 0.0), typ='refr', material=lambda w: 1.5168,
              thickness=5.0)
         .add(Conic(-1 / 50.0, 0.0), typ='refr', material=lambda w: 1.0,
              thickness=95.0)
         .add(Plane(), typ='eval'))
    if fields is None:
        fields = [Field(0, 0), Field(0, 3)]
    if wavelengths is None:
        wavelengths = {'F': 0.4861, 'd': 0.5876, 'C': 0.6563}
    return OpticalSystem(lens, aperture=ApertureSpec.epd(10.0), fields=fields,
                         wavelengths=wavelengths, reference_wavelength=ref,
                         stop_index=0)


def _bare_singlet():
    """A plain surface list (no system metadata)."""
    return [
        conic(c=1 / 50.0, k=0.0, interaction='refr', P=[0, 0, 0],
              material=lambda w: 1.5),
        conic(c=-1 / 50.0, k=0.0, interaction='refr', P=[0, 0, 5.0],
              material=lambda w: 1.0),
        plane(interaction='eval', P=[0, 0, 100.0]),
    ]


# ---------- grid shape and indexing -----------------------------------------

def test_grid_row_major_indices_and_count():
    sys = _singlet_system()
    records = list(iter_trace_grid(sys, None, None, Sampling.hex(nrings=2)))
    assert len(records) == 2 * 3
    # row-major: field outer, wavelength inner
    seen = [(r.i, r.j) for r in records]
    assert seen == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    for r in records:
        assert isinstance(r, TraceRecord)


def test_grid_records_carry_physical_field_and_wavelength():
    fields = [Field(0, 0), Field(0, 3)]
    sys = _singlet_system(fields=fields)
    records = list(iter_trace_grid(sys, fields, [0.5876], Sampling.chief()))
    assert [r.wvl for r in records] == [0.5876, 0.5876]
    assert records[0].field is fields[0]
    assert records[1].field is fields[1]


# ---------- defaulting ------------------------------------------------------

def test_grid_defaults_fields_and_wavelengths_from_system():
    sys = _singlet_system()
    records = list(iter_trace_grid(sys, None, None, Sampling.chief()))
    # 2 system fields x 3 system wavelengths
    assert len({r.i for r in records}) == 2
    np.testing.assert_allclose(sorted({r.wvl for r in records}),
                               sorted([0.4861, 0.5876, 0.6563]))


def test_grid_defaults_epd_from_aperture_spec():
    sys = _singlet_system()
    r = next(iter_trace_grid(sys, [Field(0, 0)], [0.5876], Sampling.chief()))
    assert r.epd == pytest.approx(10.0)


def test_grid_bare_prescription_defaults_to_on_axis_and_reference_wvl():
    presc = _bare_singlet()
    records = list(iter_trace_grid(presc, None, None, Sampling.chief(),
                                   epd=4.0))
    assert len(records) == 1
    r = records[0]
    assert (r.field.hx, r.field.hy) == (0.0, 0.0)
    assert r.wvl == pytest.approx(0.6328)  # kernel default reference


def test_require_epd_raises_without_epd_or_system():
    presc = _bare_singlet()
    with pytest.raises(TypeError, match='epd is required'):
        list(iter_trace_grid(presc, [Field(0, 0)], [0.5876],
                             Sampling.hex(nrings=2)))


# ---------- validity masking ------------------------------------------------

def test_grid_valid_mask_all_true_for_clean_trace():
    sys = _singlet_system()
    r = next(iter_trace_grid(sys, [Field(0, 0)], [0.5876],
                             Sampling.hex(nrings=3)))
    assert r.valid.dtype == bool
    assert r.valid.all()
    assert r.valid.shape[0] == r.P.shape[0]


def test_grid_valid_mask_flags_clipped_rays():
    presc = _bare_singlet()
    presc[0].aperture = circular_aperture(1.5)  # clips the bundle rim
    r = next(iter_trace_grid(presc, [Field(0, 0)], [0.55],
                             Sampling.hex(nrings=4), epd=8.0))
    assert not r.valid.all()
    assert r.valid.any()
    # the mask agrees with a hand-rolled trace
    expected = _valid_mask(r.trace.status, r.trace.P[-1])
    np.testing.assert_array_equal(r.valid, expected)


# ---------- bit-identical to the open-coded path ----------------------------

def test_grid_cell_matches_open_coded_launch_and_trace():
    sys = _singlet_system()
    field = Field(0, 3)
    wvl = 0.5876
    sampling = Sampling.fan(n=11, axis='y')
    epd = _require_epd(sys, None, wvl)
    P_ref, S_ref = launch(sys, field, wvl, sampling, epd=epd)
    tr_ref = raytrace(sys, P_ref, S_ref, wvl)

    r = next(iter_trace_grid(sys, [field], [wvl], sampling))
    np.testing.assert_array_equal(r.P, P_ref)
    np.testing.assert_array_equal(r.S, S_ref)
    np.testing.assert_array_equal(r.trace.P, tr_ref.P)
    np.testing.assert_array_equal(r.trace.S, tr_ref.S)


# ---------- trace_cell ------------------------------------------------------

def test_trace_cell_single_bundle():
    sys = _singlet_system()
    r = trace_cell(sys, Field(0, 0), 0.5876, Sampling.hex(nrings=2))
    assert isinstance(r, TraceRecord)
    assert (r.i, r.j) == (0, 0)
    assert r.valid.all()


def test_trace_cell_custom_trace_fn_is_used():
    sys = _singlet_system()
    calls = {'n': 0}

    def counting_trace(presc, P, S, wvl):
        calls['n'] += 1
        return raytrace(presc, P, S, wvl)

    trace_cell(sys, Field(0, 0), 0.5876, Sampling.chief(),
               trace_fn=counting_trace)
    assert calls['n'] == 1


def test_explicit_epd_overrides_system_per_cell():
    sys = _singlet_system()
    r = next(iter_trace_grid(sys, [Field(0, 0)], [0.5876],
                             Sampling.hex(nrings=2), epd=6.0))
    assert r.epd == pytest.approx(6.0)


# ---------- resolution helpers ----------------------------------------------

def test_resolve_fields_is_idempotent_on_a_list():
    fields = [Field(0, 0), Field(0, 2)]
    assert _resolve_fields(None, fields) == fields


def test_resolve_wavelengths_casts_to_float():
    out = _resolve_wavelengths(None, [1, 2])
    assert out == [1.0, 2.0]
    assert all(isinstance(w, float) for w in out)
