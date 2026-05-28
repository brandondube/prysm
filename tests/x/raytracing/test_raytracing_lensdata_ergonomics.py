"""Tests that analysis / launch / paraxial / plotting default system metadata
from a LensData when omitted, and require explicit metadata otherwise.
"""
import numpy as np
import pytest

from prysm.x.raytracing import LensData, FRAUNHOFER_LINES_UM, Field, Sampling, launch
from prysm.x.raytracing import materials
from prysm.x.raytracing.surfaces import Conic, Plane
from prysm.x.raytracing.paraxial import effective_focal_length, first_order
from prysm.x.raytracing.analysis import distortion, field_curvature
from prysm.x.raytracing._meta import (
    lensdata_wavelength, lensdata_epd, lensdata_stop_index,
)


def _n_bk7(wvl):
    return 1.5168


def _dispersive(wvl):
    # toy dispersion so wavelength selection actually moves the answer
    return 1.5 + 0.01 / wvl


def _singlet(material=_n_bk7):
    return (LensData(epd=20.0, fields=[0, 1.0], wavelengths=FRAUNHOFER_LINES_UM,
                     reference_wavelength='d', stop_index=0)
            .add(Conic(1 / 102.0, 0.0), thickness=6.0, material=material,
                 semidiameter=12.0)
            .add(Conic(-1 / 102.0, 0.0), thickness=95.0,
                 material=materials.air, semidiameter=12.0)
            .add(Plane(), typ='eval', material=materials.air,
                 semidiameter=12.0))


# ---------- the _meta helpers ----------------------------------------------

def test_lensdata_wavelength_defaults_and_resolves_names():
    ld = _singlet()
    assert lensdata_wavelength(ld, None) == pytest.approx(ld.wavelength('d'))
    assert lensdata_wavelength(ld, 'F') == pytest.approx(ld.wavelength('F'))
    assert lensdata_wavelength(ld, 0.5) == pytest.approx(0.5)


def test_surface_list_requires_explicit_wavelength():
    surfs = list(_singlet().surfaces)
    with pytest.raises(TypeError, match='wavelength is required'):
        lensdata_wavelength(surfs, None)
    assert lensdata_wavelength(surfs, 0.5) == pytest.approx(0.5)


def test_lensdata_epd_and_stop_defaults():
    ld = _singlet()
    assert lensdata_epd(ld, None) == pytest.approx(20.0)
    assert lensdata_epd(ld, 7.0) == pytest.approx(7.0)       # explicit wins
    assert lensdata_epd(list(ld.surfaces), None) is None      # no metadata
    assert lensdata_stop_index(ld, None) == 0
    assert lensdata_stop_index(ld, 2) == 2


# ---------- paraxial --------------------------------------------------------

def test_efl_defaults_wavelength_to_reference():
    ld = _singlet()
    assert effective_focal_length(ld) == pytest.approx(
        effective_focal_length(ld, wvl=ld.wavelength('d')))


def test_efl_resolves_wavelength_name_with_dispersion():
    ld = _singlet(material=_dispersive)
    by_name = effective_focal_length(ld, wvl='F')
    by_value = effective_focal_length(ld, wvl=ld.wavelength('F'))
    assert by_name == pytest.approx(by_value)
    # dispersion makes F differ from C
    assert by_name != pytest.approx(effective_focal_length(ld, wvl='C'))


def test_first_order_defaults_wavelength_epd_stop():
    ld = _singlet()
    fo = first_order(ld)
    assert fo.wavelength == pytest.approx(ld.wavelength('d'))
    assert fo.epd == pytest.approx(20.0)   # epd defaulted -> fno computed
    assert fo.fno is not None
    assert fo.stop_index == 0              # stop defaulted -> pupils computed


# ---------- launch ----------------------------------------------------------

def test_launch_defaults_epd_from_lensdata():
    ld = _singlet()
    wvl = ld.wavelength('d')
    P1, S1 = launch(ld, ld.field(0), wvl, Sampling.hex(nrings=2))
    P2, S2 = launch(ld, ld.field(0), wvl, Sampling.hex(nrings=2), epd=ld.epd)
    np.testing.assert_allclose(P1, P2)
    np.testing.assert_allclose(S1, S2)


def test_launch_surface_list_requires_epd():
    surfs = list(_singlet().surfaces)
    with pytest.raises(ValueError, match='entrance pupil'):
        launch(surfs, Field(0, 0), 0.55, Sampling.hex(nrings=2))


# ---------- analysis --------------------------------------------------------

def test_distortion_defaults_epd_and_wavelength():
    ld = _singlet()
    fields = [Field(0, 0), Field(0, 1.0)]
    a = distortion(ld, fields)
    b = distortion(ld, fields, ld.wavelength('d'), epd=ld.epd)
    for x, y in zip(a, b):
        np.testing.assert_allclose(x, y)


def test_field_curvature_defaults_epd_and_wavelength():
    ld = _singlet()
    fields = [Field(0, 0), Field(0, 1.0)]
    a = field_curvature(ld, fields)
    b = field_curvature(ld, fields, ld.wavelength('d'), epd=ld.epd)
    np.testing.assert_allclose(a[0], b[0])
    np.testing.assert_allclose(a[1], b[1])


def test_analysis_surface_list_without_epd_raises():
    surfs = list(_singlet().surfaces)
    with pytest.raises(TypeError, match='epd is required'):
        distortion(surfs, [Field(0, 0)], 0.55)
