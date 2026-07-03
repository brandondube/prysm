"""Tests that analysis / launch / paraxial / plotting default system metadata
from an OpticalSystem when omitted, and require explicit metadata otherwise.
"""
import numpy as np
import pytest

from prysm.x.raytracing import (
    LensData, OpticalSystem, FRAUNHOFER_LINES_UM,
    Field, Sampling, launch,
)
from prysm.x import materials
from prysm.x.raytracing.surfaces import Conic, Plane
from prysm.x.raytracing.paraxial import effective_focal_length
from prysm.x.raytracing.analysis import distortion, field_curvature


_n_bk7 = materials.ConstantMaterial(1.5168, name='N-BK7')


# toy dispersion so wavelength selection actually moves the answer
_dispersive = materials.FormulaMaterial('DISP', lambda wvl: 1.5 + 0.01 / wvl)


def _singlet(material=_n_bk7):
    lens = LensData()
    (lens.add(Conic(1 / 102.0, 0.0), thickness=6.0, material=material,
              aperture=12.0)
         .add(Conic(-1 / 102.0, 0.0), thickness=95.0,
              material=materials.air, aperture=12.0)
         .add(Plane(), typ='eval', material=materials.air,
              aperture=12.0))
    return OpticalSystem(lens, aperture=20.0, fields=[0, 1.0],
                         wavelengths=list(FRAUNHOFER_LINES_UM.values()),
                         reference=1, stop_index=0)


# ---------- system-level metadata resolution (ADR-0001) --------------------

def test_system_wavelength_defaults_and_resolves():
    # Wavelength resolution lives on OpticalSystem: None -> reference.
    sys = _singlet()
    assert sys.wavelength(None) == pytest.approx(sys.reference_wavelength)
    assert sys.wavelength(0.5) == pytest.approx(0.5)


def test_system_without_wavelengths_uses_kernel_default():
    # A system carrying no wavelengths resolves None to the kernel default
    # (0.6328); this default no longer lives in the free primitives.
    lens = LensData().add(Conic(1 / 102.0, 0.0), thickness=6.0,
                          material=_n_bk7, aperture=12.0)
    sys = OpticalSystem(lens)
    assert sys.wavelength(None) == pytest.approx(0.6328)
    assert sys.wavelength(0.5) == pytest.approx(0.5)


def test_system_epd_and_stop_defaults():
    sys = _singlet()
    assert sys.epd == pytest.approx(20.0)
    assert sys.entrance_pupil_diameter(0.5) == pytest.approx(20.0)
    assert sys.stop_index == 0


# ---------- paraxial --------------------------------------------------------

def test_efl_defaults_wavelength_to_reference():
    # Default-wavelength resolution lives on OpticalSystem (ADR-0001); the
    # system resolves None to the reference before calling the primitive.
    sys = _singlet()
    assert sys._ynu_first_order().efl == pytest.approx(
        effective_focal_length(sys.to_surfaces(),
                               wvl=sys.reference_wavelength))


def test_efl_resolves_wavelength_with_dispersion():
    sys = _singlet(material=_dispersive)
    surfaces = sys.to_surfaces()
    f_val = FRAUNHOFER_LINES_UM['F']
    c_val = FRAUNHOFER_LINES_UM['C']
    # dispersion makes F differ from C
    assert (effective_focal_length(surfaces, wvl=f_val)
            != pytest.approx(effective_focal_length(surfaces, wvl=c_val)))


def test_first_order_defaults_wavelength_epd_stop():
    # The system resolves wvl/epd/stop and hands the scalars to the primitive.
    sys = _singlet()
    fo = sys._ynu_first_order()
    assert fo.wavelength == pytest.approx(sys.wavelength())
    assert fo.epd == pytest.approx(20.0)   # epd defaulted -> fno computed
    assert fo.fno is not None
    assert fo.stop_index == 0              # stop defaulted -> pupils computed


# ---------- launch ----------------------------------------------------------

def test_launch_defaults_epd_from_system():
    sys = _singlet()
    wvl = sys.wavelength()
    P1, S1 = launch(sys, sys.field(0), wvl, Sampling.hex(nrings=2))
    P2, S2 = launch(sys, sys.field(0), wvl, Sampling.hex(nrings=2), epd=sys.epd)
    np.testing.assert_allclose(P1, P2)
    np.testing.assert_allclose(S1, S2)


def test_launch_surface_list_requires_epd():
    surfs = list(_singlet().surfaces)
    with pytest.raises(ValueError, match='entrance pupil'):
        launch(surfs, Field(0, 0), 0.55, Sampling.hex(nrings=2))


# ---------- analysis --------------------------------------------------------

def test_distortion_defaults_epd_and_wavelength():
    sys = _singlet()
    fields = [Field(0, 0), Field(0, 1.0)]
    a = distortion(sys, fields)
    b = distortion(sys, fields, sys.wavelength(), epd=sys.epd)
    np.testing.assert_allclose(a.real_xy, b.real_xy)
    np.testing.assert_allclose(a.paraxial_xy, b.paraxial_xy)
    np.testing.assert_allclose(a.percent, b.percent)


def test_field_curvature_defaults_wavelength():
    sys = _singlet()
    fields = [Field(0, 0), Field(0, 1.0)]
    a = field_curvature(sys, fields)
    b = field_curvature(sys, fields, sys.wavelength())
    np.testing.assert_allclose(a.x_fan_z, b.x_fan_z)
    np.testing.assert_allclose(a.y_fan_z, b.y_fan_z)


def test_analysis_surface_list_without_epd_raises():
    surfs = list(_singlet().surfaces)
    with pytest.raises(TypeError, match='epd is required'):
        distortion(surfs, [Field(0, 0)], 0.55)
