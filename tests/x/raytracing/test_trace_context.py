"""Tests for trace_context, the resolved-metadata seam."""
import pytest

from prysm.x.raytracing import OpticalSystem, sample_rx
from prysm.x.raytracing._resolve import TraceContext, trace_context
from prysm.x.raytracing._meta import object_image_indices

WVL = 0.5875618


def _doublet_system():
    return OpticalSystem(sample_rx.doublet_conic(), aperture=15.0,
                         fields=[0.0, 3.0], wavelengths=[WVL], reference=0,
                         stop_index=2)


def test_system_fills_wavelength_and_surfaces():
    sys = _doublet_system()
    ctx = trace_context(sys)
    assert ctx.wavelength == pytest.approx(WVL)
    assert ctx.surfaces == sys.to_surfaces()
    assert ctx.epd is None
    assert ctx.stop_index is None


def test_chief_fills_epd_and_stop():
    sys = _doublet_system()
    ctx = trace_context(sys, chief=True)
    assert ctx.epd == pytest.approx(sys.entrance_pupil_diameter(WVL))
    assert ctx.stop_index == sys.stop_index


def test_explicit_scalars_win_over_system():
    sys = _doublet_system()
    ctx = trace_context(sys, 0.5, chief=True, epd=3.0, stop_index=1)
    assert ctx.wavelength == 0.5
    assert ctx.epd == 3.0
    assert ctx.stop_index == 1


def test_media_match_meta_helpers():
    sys = _doublet_system()
    ctx = trace_context(sys)
    n_object, n_image = object_image_indices(ctx.surfaces, ctx.wavelength)
    assert ctx.n_object == n_object
    assert ctx.n_image == n_image


def test_bare_sequence_requires_wavelength():
    surfaces = _doublet_system().to_surfaces()
    with pytest.raises(ValueError, match='wavelength must be given'):
        trace_context(surfaces)


def test_bare_sequence_leaves_chief_scalars_as_passed():
    surfaces = _doublet_system().to_surfaces()
    ctx = trace_context(surfaces, WVL, chief=True)
    assert ctx.epd is None
    assert ctx.stop_index is None
    ctx = trace_context(surfaces, WVL, chief=True, epd=4.0, stop_index=2)
    assert ctx.epd == 4.0
    assert ctx.stop_index == 2


def test_n_image_lazy_raise_without_image_surface():
    # media resolve on access, so a truncated sequence is usable until then
    surfaces = _doublet_system().to_surfaces()[:-1]
    ctx = TraceContext(surfaces, WVL)
    assert ctx.n_object == 1.0
    with pytest.raises(ValueError, match='image-space index'):
        ctx.n_image
