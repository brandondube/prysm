"""Export tests: LensData -> .seq / .zmx round-trips on the
rotationally symmetric subset, including the post-mirror sign convention."""

import numpy as np
import pytest

from prysm.x.raytracing import LensData
from prysm.x.raytracing import materials
from prysm.x.raytracing.io_codev import read_seq, write_seq
from prysm.x.raytracing.io_zemax import read_zmx, write_zmx
from prysm.x.raytracing.surfaces import Conic, EvenAsphere, Plane


def _air(wvl):
    return 1.0


def make_refractive():
    return (LensData(epd=10.0, wavelengths=[0.55])
            .add(Conic(1 / 50.0, 0.0), thickness=5.0, material=_air)
            .add(Conic(-1 / 50.0, -0.5), thickness=95.0, material=_air)
            .add(Plane(), typ='eval'))


def make_mirror():
    return (LensData(epd=10.0, wavelengths=[0.55])
            .add(Conic(1 / 200.0, -1.0), typ='refl', thickness=50.0)
            .add(Plane(), typ='eval'))


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
    cb = back.rows[0]
    np.testing.assert_allclose(np.asarray(cb.tilt), [3.0, 2.0, -5.0])


def test_seq_export_rejects_unsupported_shape_without_loss():
    ld = (LensData()
          .add(EvenAsphere(0.01, 0.0, (1e-4,)), thickness=1.0)
          .add(Plane(), typ='eval'))
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


def test_zmx_export_rejects_unsupported_shape_without_loss():
    ld = (LensData()
          .add(EvenAsphere(0.01, 0.0, (1e-4,)), thickness=1.0)
          .add(Plane(), typ='eval'))
    with pytest.raises(NotImplementedError, match='EvenAsphere'):
        write_zmx(ld)
