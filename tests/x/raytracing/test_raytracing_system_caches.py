"""Version-stamped first-order caches on OpticalSystem (first_order,
entrance_pupil_z, entrance_pupil_diameter) and their launch/merit consults."""
import numpy as np

import pytest

from prysm.x import materials
from prysm.x.raytracing import (
    OpticalSystem, ApertureSpec, LensData, Sphere, Plane, Field, Sampling,
)
from prysm.x.raytracing import paraxial
from prysm.x.raytracing import parabasal
from prysm.x.raytracing.launch import launch
from prysm.x.raytracing.system import ApertureSpec as _ApertureSpec


def _doublet(aperture=None):
    # first powered surface is row 1
    ld = (LensData()
          .add(Sphere(1 / 61.47), thickness=6.0,
               material=materials.ConstantMaterial(1.5168), aperture=12.0)
          .add(Sphere(-1 / 44.64), thickness=2.5,
               material=materials.ConstantMaterial(1.673), aperture=12.0)
          .add(Sphere(-1 / 129.94), thickness=0.0,
               material=materials.air, aperture=12.0))
    sys = OpticalSystem(ld, aperture=aperture or ApertureSpec.epd(22.0),
                        fields=[Field(0, 0), Field(0, 0.7), Field(0, 1.0)],
                        wavelengths=[0.486, 0.587, 0.656], reference=1,
                        stop_index=1)
    sys.solve.image_distance()
    return sys


def _count_calls(monkeypatch, module, name):
    counter = {'n': 0}
    inner = getattr(module, name)

    def spy(*args, **kwargs):
        counter['n'] += 1
        return inner(*args, **kwargs)

    monkeypatch.setattr(module, name, spy)
    return counter


def test_first_order_cached_per_version_and_wavelength(monkeypatch):
    sys = _doublet()
    calls = _count_calls(monkeypatch, parabasal, 'first_order')
    fo1 = sys.first_order(wavelength=0.587)
    fo2 = sys.first_order(wavelength=0.587)
    assert calls['n'] == 1
    assert fo2 is fo1
    sys.first_order(wavelength=0.486)
    assert calls['n'] == 2
    # Lens edit -> recompute.
    sys.lens.rows[2].thickness = 2.6
    sys.lens._invalidate()
    fo3 = sys.first_order(wavelength=0.587)
    assert calls['n'] == 3
    assert fo3 is not fo1


def test_entrance_pupil_z_cached_and_correct(monkeypatch):
    sys = _doublet()
    direct = paraxial.entrance_pupil_z(sys.to_surfaces(), 0.587,
                                       stop_index=sys.stop_index)
    calls = _count_calls(monkeypatch, paraxial, 'entrance_pupil_z')
    z1 = sys.entrance_pupil_z(0.587)
    z2 = sys.entrance_pupil_z(0.587)
    assert calls['n'] == 1
    assert z1 == z2 == direct
    sys.lens.rows[2].thickness = 2.6
    sys.lens._invalidate()
    sys.entrance_pupil_z(0.587)
    assert calls['n'] == 2


def test_launch_consults_system_entrance_pupil_cache(monkeypatch):
    sys = _doublet()
    calls = _count_calls(monkeypatch, paraxial, 'entrance_pupil_z')
    for f in sys.fields:
        launch(sys, f, 0.587, Sampling.hex(3))
    # One paraxial pupil solve across the field grid.
    assert calls['n'] == 1


def test_launch_on_bare_lensdata_unchanged():
    # Bare LensData has no stop metadata.
    sys = _doublet()
    P_sys, S_sys = launch(sys, Field(0, 0.7), 0.587, Sampling.hex(3))
    P_ld, S_ld = launch(sys.lens, Field(0, 0.7), 0.587, Sampling.hex(3),
                        epd=22.0)
    assert np.allclose(S_ld, S_sys)
    assert P_ld.shape == P_sys.shape
    assert np.all(np.isfinite(P_ld))


def test_dependency_resolution_does_not_bump_version():
    # Solves/pickups during compile do not bump the edit version.
    sys = _doublet()
    ld = sys.lens
    ld._invalidate()  # force a cold compile with the image solve active
    v0 = ld._version
    ld.to_surfaces()
    assert ld._version == v0
    ld.to_surfaces()
    assert ld._version == v0


def test_entrance_pupil_diameter_cached_and_aperture_keyed(monkeypatch):
    sys = _doublet(aperture=ApertureSpec.fno(5.0))
    calls = _count_calls(monkeypatch, _ApertureSpec, 'entrance_pupil_diameter')
    d1 = sys.epd
    d2 = sys.epd
    assert calls['n'] == 1
    assert d1 == d2
    # Aperture assignment changes the cache key.
    sys.aperture = ApertureSpec.fno(10.0)
    d3 = sys.epd
    assert calls['n'] == 2
    assert d3 == pytest.approx(d1 / 2.0, rel=1e-12)


def test_resolve_exit_pupil_consults_system_first_order(monkeypatch):
    from prysm.x.raytracing.analysis import resolve_exit_pupil

    sys = _doublet()
    calls = _count_calls(monkeypatch, paraxial, 'ynu_first_order')
    p1 = resolve_exit_pupil(sys, 0.587)
    p2 = resolve_exit_pupil(sys, 0.587)
    assert calls['n'] == 1
    assert np.allclose(p1, p2)
