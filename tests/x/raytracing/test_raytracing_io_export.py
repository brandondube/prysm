"""Export tests: LensData -> .seq / .zmx round-trips on the
rotationally symmetric subset, including the post-mirror sign convention."""

import numpy as np
import pytest

from prysm.x.raytracing import OpticalSystem
from prysm.x.raytracing import LensData
from prysm.x import materials
from prysm.x.raytracing.io import read_seq, write_seq, read_zmx, write_zmx
from prysm.x.raytracing.surfaces import Conic, EvenAsphere, Plane


def make_refractive():
    # OBJECT/IMAGE endpoints are implicit.
    lens = LensData()
    (lens.add(Conic(1 / 50.0, 0.0), thickness=5.0, material=materials.air)
         .add(Conic(-1 / 50.0, -0.5), thickness=95.0, material=materials.air))
    return OpticalSystem(lens, aperture=10.0, wavelengths=[0.55])


def make_mirror():
    lens = LensData()
    lens.add(Conic(1 / 200.0, -1.0), typ='refl', thickness=50.0)
    return OpticalSystem(lens, aperture=10.0, wavelengths=[0.55])


def _assert_geometry_round_trips(a, b):
    sa, sb = a.surfaces, b.surfaces
    assert len(sa) == len(sb)
    for x, y in zip(sa, sb):
        assert x.typ == y.typ
        np.testing.assert_allclose(np.asarray(x.P), np.asarray(y.P), atol=1e-9)
        pa, pb = x.params or {}, y.params or {}
        for key in ('c', 'k'):
            if key in pa or key in pb:
                np.testing.assert_allclose(pa.get(key, 0.0), pb.get(key, 0.0))


def test_seq_round_trip_refractive():
    ld = make_refractive()
    back = read_seq(write_seq(ld), _is_text=True)
    _assert_geometry_round_trips(ld, back)
    assert back.epd == 10.0


def test_seq_round_trip_mirror_sign_convention():
    ld = make_mirror()
    text = write_seq(ld)
    assert 'REFL' in text
    back = read_seq(text, _is_text=True)
    _assert_geometry_round_trips(ld, back)
    # image lands at negative z (folded), same as the original
    assert float(np.asarray(back.surfaces[-1].P)[2]) == pytest.approx(-50.0)


def test_seq_round_trip_codev_alpha_beta_signs():
    text = """\
LEN
CUM
SO ; THI 1E10
S ; CUY 0 ; THI 0 ; ADE 5 ; BDE -2 ; CDE 3
SI
GO
"""
    back = read_seq(write_seq(read_seq(text, _is_text=True)), _is_text=True)
    cb = back.rows[1]               # rows[0] is the OBJECT endpoint
    np.testing.assert_allclose(np.asarray(cb.tilt), [3.0, 2.0, -5.0])


def test_seq_export_rejects_unsupported_shape_without_loss():
    ld = LensData().add(EvenAsphere(0.01, 0.0, (1e-4,)), thickness=1.0)
    with pytest.raises(NotImplementedError, match='EvenAsphere'):
        write_seq(ld)


def test_zmx_round_trip_refractive():
    ld = make_refractive()
    back = read_zmx(write_zmx(ld), _is_text=True)
    _assert_geometry_round_trips(ld, back)
    assert back.epd == 10.0


def test_zmx_round_trip_mirror_sign_convention():
    ld = make_mirror()
    text = write_zmx(ld)
    assert 'MIRROR' in text
    back = read_zmx(text, _is_text=True)
    _assert_geometry_round_trips(ld, back)
    assert float(np.asarray(back.surfaces[-1].P)[2]) == pytest.approx(-50.0)


def test_zmx_export_carries_stop_index():
    ld = make_refractive()
    ld.stop_index = 1
    back = read_zmx(write_zmx(ld), _is_text=True)
    assert back.stop_index == 1


def test_zmx_export_maps_stop_index_past_coordbreak():
    lens = LensData()
    lens.add_coordbreak(decenter=(1.0, 0.0, 0.0), thickness=0.0)  # rows[1]
    lens.add(Plane(), typ='eval')
    sys = OpticalSystem(lens, stop_index=1)
    text = write_zmx(sys)
    assert 'STOP 2\n' in text
    back = read_zmx(text, _is_text=True)
    assert back.stop_index == 1


def test_zmx_export_rejects_unsupported_shape_without_loss():
    ld = LensData().add(EvenAsphere(0.01, 0.0, (1e-4,)), thickness=1.0)
    with pytest.raises(NotImplementedError, match='EvenAsphere'):
        write_zmx(ld)


class _StubGlassDB:
    """Minimal material catalog resolving one fixed name to a ConstantMaterial."""

    def __init__(self, name, n):
        self._name = name
        self.material = materials.ConstantMaterial(n, name=name)

    def material_for_name(self, name, **kwargs):
        if name == self._name:
            return self.material
        raise KeyError(name)


def _finite_conjugate_system(object_medium=None):
    lens = LensData()
    lens.object_row.thickness = 50.0
    if object_medium is not None:
        lens.object_row.material = object_medium
    (lens.add(Conic(1 / 50.0, 0.0), thickness=5.0, material=materials.air)
         .add(Conic(-1 / 50.0, -0.5), thickness=95.0, material=materials.air))
    return OpticalSystem(lens, aperture=10.0, wavelengths=[0.55])


def test_zmx_round_trip_finite_object_conjugate():
    db = _StubGlassDB('BK7', 1.6)
    ld = _finite_conjugate_system(object_medium=db.material)
    back = read_zmx(write_zmx(ld), _is_text=True, database=db)
    assert back.rows[0].thickness == pytest.approx(50.0)
    assert back.rows[0].material is not materials.air
    assert back.rows[0].material is not materials.vacuum


def test_seq_round_trip_finite_object_conjugate():
    db = _StubGlassDB('BK7', 1.6)
    ld = _finite_conjugate_system(object_medium=db.material)
    back = read_seq(write_seq(ld), _is_text=True, database=db)
    assert back.rows[0].thickness == pytest.approx(50.0)
    assert back.rows[0].material is not materials.air
    assert back.rows[0].material is not materials.vacuum
