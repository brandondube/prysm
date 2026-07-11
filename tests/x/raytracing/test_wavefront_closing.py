"""Tests for close_wavefront, the system-level owner of the OPD recipe."""
import numpy as np
import pytest

from prysm.x.raytracing import OpticalSystem, sample_rx
from prysm.x.raytracing.spencer_and_murty import raytrace, valid_mask
from prysm.x.raytracing.launch import Field, Sampling, launch
from prysm.x.raytracing.opt import _pupil_center_chief_index
from prysm.x.raytracing.analysis import (
    close_wavefront,
    close_on_reference_sphere,
    wavefront,
)
from prysm.x.raytracing._meta import object_image_indices
from prysm.x.raytracing._resolve import compiled_surfaces

from tests.x.raytracing.test_eic_closing import _telecentric

WVL = 0.5875618


def _doublet_system():
    sys = OpticalSystem(sample_rx.doublet_conic(), aperture=15.0,
                        fields=[0.0, 3.0], wavelengths=[WVL], reference=0,
                        stop_index=2)   # the front stop plane (0 is OBJECT)
    sys.solve.image_distance()
    return sys


def _traced_bundle(sys, field, sampling=None):
    if sampling is None:
        sampling = Sampling.fan(n=21, axis='y')
    P, S = launch(sys, field, WVL, sampling, epd=sys.epd)
    trace = raytrace(sys, P, S, WVL)
    return P, S, trace


def test_close_wavefront_matches_wavefront_resolved_xp():
    sys = _doublet_system()
    P, S, trace = _traced_bundle(sys, Field(0.0, 0.0))
    chief = _pupil_center_chief_index(P)
    wc = close_wavefront(sys, trace, WVL, chief)
    opd, _, _ = wavefront(sys, P, S, WVL, output='length')
    np.testing.assert_array_equal(wc.opd, opd)
    assert wc.xp_mode == 'paraxial'
    assert wc.P_xp is not None
    assert wc.chief_index == chief
    np.testing.assert_array_equal(wc.center, trace.P[-1, chief])


def test_close_wavefront_matches_wavefront_fixed_xp():
    sys = _doublet_system()
    P, S, trace = _traced_bundle(sys, Field(0.0, 0.0))
    chief = _pupil_center_chief_index(P)
    P_xp = np.asarray(sys.exit_pupil(WVL))
    wc = close_wavefront(sys, trace, WVL, chief, P_xp=P_xp)
    opd, _, _ = wavefront(sys, P, S, WVL, P_xp=P_xp, output='length')
    np.testing.assert_array_equal(wc.opd, opd)
    assert wc.xp_mode == 'fixed'
    np.testing.assert_array_equal(wc.P_xp, P_xp)


def test_close_wavefront_telecentric_resolves_kappa_zero():
    sys = _telecentric()
    wvl = sys.wavelength()
    fld = Field(3.0, 0.0)
    P, S = launch(sys, fld, wvl, Sampling.fan(n=21, axis='y'), epd=sys.epd)
    trace = raytrace(sys, P, S, wvl)
    chief = _pupil_center_chief_index(P)
    wc = close_wavefront(sys, trace, wvl, chief)
    assert wc.P_xp is None
    assert wc.xp_mode == 'paraxial'
    assert wc.curvature == 0.0
    assert wc.R == np.inf
    assert wc.delta is None
    assert np.all(np.isfinite(wc.opd))


def test_close_wavefront_center_override():
    sys = _doublet_system()
    P, S, trace = _traced_bundle(sys, Field(0.0, 0.0))
    chief = _pupil_center_chief_index(P)
    center = trace.P[-1, chief] + np.array([0.0, 0.0, 0.5])
    P_xp = np.asarray(sys.exit_pupil(WVL))
    wc = close_wavefront(sys, trace, WVL, chief, center=center, P_xp=P_xp)
    valid = valid_mask(trace.status, trace.P[-1])
    _, n_image = object_image_indices(compiled_surfaces(sys), WVL)
    expected = close_on_reference_sphere(trace, valid, chief, center=center,
                                         P_xp=P_xp, n_image=n_image)
    np.testing.assert_array_equal(wc.opd, expected.opd)
    default = close_wavefront(sys, trace, WVL, chief, P_xp=P_xp)
    assert not np.array_equal(wc.opd, default.opd)


def test_close_wavefront_field_tilt_ramp():
    sys = _doublet_system()
    fld = Field(0.0, 3.0)
    P, S, trace = _traced_bundle(sys, fld)
    chief = _pupil_center_chief_index(P)
    on = close_wavefront(sys, trace, WVL, chief, field=fld)
    off = close_wavefront(sys, trace, WVL, chief, field=fld,
                          apply_field_tilt=False)
    valid = on.valid
    ax, ay = fld.angle_radians()
    ramp = (np.sin(ax) * (P[valid, 0] - P[chief, 0])
            + np.sin(ay) * (P[valid, 1] - P[chief, 1]))
    np.testing.assert_array_equal(on.opd, off.opd + ramp)
    opd, _, _ = wavefront(sys, P, S, WVL, field=fld, output='length')
    np.testing.assert_array_equal(on.opd, opd)


def test_close_wavefront_invalid_chief_errors():
    sys = _doublet_system()
    P, S, trace = _traced_bundle(sys, Field(0.0, 0.0))
    chief = _pupil_center_chief_index(P)
    dead = np.zeros(P.shape[0], dtype=bool)
    with pytest.raises(ValueError, match='chief ray is invalid'):
        close_wavefront(sys, trace, WVL, chief, valid=dead)
    with pytest.raises(ValueError, match='anchor ray'):
        close_wavefront(sys, trace, WVL, chief, valid=dead,
                        reference='centroid')


def test_close_wavefront_off_axis_geometric_fallback():
    """A decentered system resolves its exit pupil geometrically, not by raising.

    An off-axis parabola imaging collimated light to its parent focus has
    OPD identically zero; the centered-ABCD route is unavailable, and the
    closing must fall back to the chief-axis geometric route.
    """
    from prysm.x.raytracing import LensData
    from prysm.x.raytracing.surfaces import Conic, Plane

    lens = (LensData()
            .add(Plane(), typ='eval', thickness=50.0)
            .add_coordbreak(decenter=(0.0, -30.0, 0.0))
            .add(Conic(-1.0 / 200.0, -1.0), typ='refl', thickness=100.0))
    sys = OpticalSystem(lens, aperture=16.0, fields=[0.0],
                        wavelengths=[WVL], reference=0, stop_index=2)
    fld = Field(0.0, 0.0)
    P, S = launch(sys, fld, WVL, Sampling.rect(n=11))
    opd, xp, yp = wavefront(sys, P, S, WVL, field=fld, output='length')
    assert np.nanmax(np.abs(opd)) < 1e-9
    from prysm.x.raytracing.analysis import resolve_exit_pupil
    P_xp, mode = resolve_exit_pupil(sys, WVL, return_mode=True)
    assert mode == 'geometric'
    assert np.isfinite(np.asarray(P_xp)).all()
