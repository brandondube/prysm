"""Tests that analysis / launch / paraxial / plotting default system metadata
from an OpticalSystem when omitted, and require explicit metadata otherwise.
"""
import numpy as np
import pytest

from prysm.x.raytracing import (
    LensData, OpticalSystem, ApertureSpec, FRAUNHOFER_LINES_UM,
    Field, Sampling, launch,
)
from prysm.x import materials
from prysm.x.raytracing.surfaces import Conic, Plane
from prysm.x.raytracing.paraxial import effective_focal_length, first_order
from prysm.x.raytracing.analysis import distortion, field_curvature
from prysm.x.raytracing._meta import (
    system_wavelength, system_epd, system_stop_index,
)


def _n_bk7(wvl):
    return 1.5168


def _dispersive(wvl):
    # toy dispersion so wavelength selection actually moves the answer
    return 1.5 + 0.01 / wvl


def _singlet(material=_n_bk7):
    lens = LensData()
    (lens.add(Conic(1 / 102.0, 0.0), thickness=6.0, material=material,
              semidiameter=12.0)
         .add(Conic(-1 / 102.0, 0.0), thickness=95.0,
              material=materials.air, semidiameter=12.0)
         .add(Plane(), typ='eval', material=materials.air,
              semidiameter=12.0))
    return OpticalSystem(lens, aperture=20.0, fields=[0, 1.0],
                         wavelengths=FRAUNHOFER_LINES_UM,
                         reference_wavelength='d', stop_index=0)


# ---------- the _meta helpers ----------------------------------------------

def test_system_wavelength_defaults_and_resolves_names():
    sys = _singlet()
    assert system_wavelength(sys, None) == pytest.approx(sys.wavelength('d'))
    assert system_wavelength(sys, 'F') == pytest.approx(sys.wavelength('F'))
    assert system_wavelength(sys, 0.5) == pytest.approx(0.5)


def test_surface_list_defaults_wavelength_to_kernel_default():
    surfs = list(_singlet().surfaces)
    # a bare surface sequence carries no metadata: None -> kernel default
    assert system_wavelength(surfs, None) == pytest.approx(0.6328)
    assert system_wavelength(surfs, 0.5) == pytest.approx(0.5)


def test_system_epd_and_stop_defaults():
    sys = _singlet()
    assert system_epd(sys, None) == pytest.approx(20.0)
    assert system_epd(sys, 7.0) == pytest.approx(7.0)        # explicit wins
    assert system_epd(list(sys.surfaces), None) is None       # no metadata
    assert system_stop_index(sys, None) == 0
    assert system_stop_index(sys, 2) == 2


# ---------- paraxial --------------------------------------------------------

def test_efl_defaults_wavelength_to_reference():
    sys = _singlet()
    assert effective_focal_length(sys) == pytest.approx(
        effective_focal_length(sys, wvl=sys.wavelength('d')))


def test_efl_resolves_wavelength_name_with_dispersion():
    sys = _singlet(material=_dispersive)
    by_name = effective_focal_length(sys, wvl='F')
    by_value = effective_focal_length(sys, wvl=sys.wavelength('F'))
    assert by_name == pytest.approx(by_value)
    # dispersion makes F differ from C
    assert by_name != pytest.approx(effective_focal_length(sys, wvl='C'))


def test_first_order_defaults_wavelength_epd_stop():
    sys = _singlet()
    fo = first_order(sys)
    assert fo.wavelength == pytest.approx(sys.wavelength('d'))
    assert fo.epd == pytest.approx(20.0)   # epd defaulted -> fno computed
    assert fo.fno is not None
    assert fo.stop_index == 0              # stop defaulted -> pupils computed


# ---------- launch ----------------------------------------------------------

def test_launch_defaults_epd_from_system():
    sys = _singlet()
    wvl = sys.wavelength('d')
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
    b = distortion(sys, fields, sys.wavelength('d'), epd=sys.epd)
    for x, y in zip(a, b):
        np.testing.assert_allclose(x, y)


def test_field_curvature_defaults_epd_and_wavelength():
    sys = _singlet()
    fields = [Field(0, 0), Field(0, 1.0)]
    a = field_curvature(sys, fields)
    b = field_curvature(sys, fields, sys.wavelength('d'), epd=sys.epd)
    np.testing.assert_allclose(a[0], b[0])
    np.testing.assert_allclose(a[1], b[1])


def test_analysis_surface_list_without_epd_raises():
    surfs = list(_singlet().surfaces)
    with pytest.raises(TypeError, match='epd is required'):
        distortion(surfs, [Field(0, 0)], 0.55)
