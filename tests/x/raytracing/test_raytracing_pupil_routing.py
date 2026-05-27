"""Entrance-pupil routing in launch() and its analysis consequences.

launch() positions off-axis bundles so the pupil sampling lands on the
paraxial entrance pupil when the stop is known (a LensData carries
stop_index).  Before this, bundles were launched centered on the axis at the
first surface, which is only correct when the stop is the first surface; for a
stop in the middle of the system (the common case) off-axis spot/distortion/
field-curvature/wavefront were wrong.

Expected values marked "optiland" were produced by optiland 0.6.0 on the same
constant-index Cooke triplet (so glass databases cannot confound the check);
they are hard-coded here to avoid a test dependency on optiland.
"""
import numpy as np
import pytest

from prysm.x.raytracing import LensData, launch, raytrace, Sampling, Field
from prysm.x.raytracing.surfaces import ConicSag, PlaneSag
from prysm.x.raytracing import materials as pmat
from prysm.x.raytracing.paraxial import first_order, entrance_pupil_z
from prysm.x.raytracing import analysis as pa
from prysm.x.raytracing.opt import rms_spot_radius

# constant indices @ 0.55 um (SK16, F2) so the system matches optiland exactly
N_SK16 = 1.62260856
N_F2 = 1.62365512
WVL = 0.55
EPD = 10.0
STOP_INDEX = 3  # 4th optical surface

# (radius, thickness, index); stop on the 4th surface
_COOKE = [
    (22.01359,   3.25896, N_SK16),
    (-435.76044, 6.00755, 1.0),
    (-22.21328,  0.99997, N_F2),
    (20.29192,   4.75041, 1.0),
    (79.68360,   2.95208, N_SK16),
    (-18.39533,  42.20778, 1.0),
]


def cooke():
    ld = LensData(epd=EPD, fields=[0.0, 14.0, 20.0], wavelengths={'w': WVL},
                  reference_wavelength='w', stop_index=STOP_INDEX)
    for R, t, n in _COOKE:
        mat = float(n) if n != 1.0 else pmat.air
        ld.add(ConicSag(1.0 / R, 0.0), thickness=t, material=mat)
    ld.add(PlaneSag(), typ='eval', material=pmat.air, semidiameter=1e3)
    return ld


def biconvex_stop_first():
    """Stop at the first surface -> entrance pupil at the first surface."""
    ld = LensData(epd=20.0, fields=[0.0, 10.0], wavelengths={'w': WVL},
                  reference_wavelength='w', stop_index=0)
    ld.add(ConicSag(1 / 50.0, 0.0), thickness=6.0, material=1.5)
    ld.add(ConicSag(-1 / 50.0, 0.0), thickness=46.0, material=pmat.air)
    ld.add(PlaneSag(), typ='eval', material=pmat.air, semidiameter=1e3)
    return ld


# ---------- entrance_pupil_z ------------------------------------------------

def test_entrance_pupil_z_matches_first_order():
    ld = cooke()
    ep = entrance_pupil_z(ld)
    assert ep == pytest.approx(first_order(ld).ep_z)
    # optiland EPL for this system is 11.51216 mm from the first surface (z=0)
    assert ep == pytest.approx(11.51216, abs=1e-4)


def test_entrance_pupil_z_none_without_stop():
    # a bare Surface sequence carries no stop_index
    presc = list(cooke().to_surfaces())
    assert entrance_pupil_z(presc) is None


def test_entrance_pupil_z_at_first_surface_when_stop_is_first():
    ld = biconvex_stop_first()
    assert entrance_pupil_z(ld) == pytest.approx(0.0, abs=1e-9)


# ---------- routing geometry ------------------------------------------------

def test_routed_chief_passes_through_stop_center_paraxially():
    """At a tiny field the routed chief lands on the stop center."""
    ld = cooke()
    P, S = launch(ld, Field(0.0, 1e-3, unit='deg'), WVL, Sampling.chief())
    tr = raytrace(ld, P, S, WVL)
    # P history: row k+1 is the intersection at surface index k
    xy_at_stop = tr.P[STOP_INDEX + 1, 0, :2]
    np.testing.assert_allclose(xy_at_stop, 0.0, atol=1e-5)


def test_aim_pupil_false_is_legacy_axis_launch():
    """Opt-out reproduces the on-axis launch at the first surface."""
    ld = cooke()
    f = Field(0.0, 14.0, unit='deg')
    P, _ = launch(ld, f, WVL, Sampling.chief(), aim_pupil=False)
    np.testing.assert_allclose(P[0, :2], 0.0, atol=1e-12)
    np.testing.assert_allclose(P[0, 2], 0.0, atol=1e-12)  # first surface vertex
    # routed launch is shifted off axis at the launch plane
    P2, _ = launch(ld, f, WVL, Sampling.chief())
    assert abs(float(P2[0, 1])) > 1e-3


def test_routing_noop_for_stop_at_first_surface():
    """With the stop at surface 1 the EP is the first surface: no shift."""
    ld = biconvex_stop_first()
    f = Field(0.0, 10.0, unit='deg')
    P_routed, _ = launch(ld, f, WVL, Sampling.chief())
    P_legacy, _ = launch(ld, f, WVL, Sampling.chief(), aim_pupil=False)
    np.testing.assert_allclose(P_routed, P_legacy, atol=1e-12)


# ---------- analysis consequences vs optiland -------------------------------

def test_chief_landing_matches_optiland():
    ld = cooke()
    # optiland chief-ray image y at fields 14 and 20 deg
    expected = {14.0: 12.419795, 20.0: 18.136026}
    for fd, want in expected.items():
        P, S = launch(ld, Field(0.0, fd, unit='deg'), WVL, Sampling.chief())
        tr = raytrace(ld, P, S, WVL)
        assert float(tr.P[-1, 0, 1]) == pytest.approx(want, abs=2e-5)


def test_distortion_matches_optiland():
    ld = cooke()
    # optiland Distortion at 20 deg (f-tan): 0.06202477 %
    _, _, pct = pa.distortion(ld, [Field(0.0, 20.0, unit='deg')], WVL, epd=EPD)
    assert float(pct[0]) == pytest.approx(0.06202477, abs=1e-6)


def test_rms_spot_matches_optiland():
    ld = cooke()
    # optiland SpotDiagram rms radius (mm), hexapolar: ~0.012117 @ 20 deg
    P, S = launch(ld, Field(0.0, 20.0, unit='deg'), WVL, Sampling.hex(nrings=6))
    tr = raytrace(ld, P, S, WVL)
    r = float(rms_spot_radius(tr.P[-1], status=tr.status))
    assert r == pytest.approx(0.012117, rel=0.03)
    # and the un-routed launch is grossly inflated (the bug this guards)
    P0, S0 = launch(ld, Field(0.0, 20.0, unit='deg'), WVL,
                    Sampling.hex(nrings=6), aim_pupil=False)
    tr0 = raytrace(ld, P0, S0, WVL)
    r0 = float(rms_spot_radius(tr0.P[-1], status=tr0.status))
    assert r0 > 10 * r


def test_wavefront_rms_matches_optiland():
    ld = cooke()
    xp_z = first_order(ld).xp_z
    # optiland OPD rms (waves), hexapolar
    expected = {0.0: 0.17864, 20.0: 0.43099}
    for fd, want in expected.items():
        f = Field(0.0, fd, unit='deg')
        P, S = launch(ld, f, WVL, Sampling.hex(nrings=10))
        opd, _, _ = pa.wavefront(ld, P, S, WVL, P_xp=(0, 0, xp_z),
                                 field=f, output='waves')
        opd = np.asarray(opd)
        opd = opd[np.isfinite(opd)]
        rms = float(np.sqrt(np.mean((opd - opd.mean()) ** 2)))
        assert rms == pytest.approx(want, rel=0.05)


def test_wavefront_default_chief_is_hex_center():
    """The hex center ray is index 0, not N//2; wavefront must pick it.

    With the correct (center) chief the reference sphere is centered on the
    true image point and the OPD is the system wavefront error; picking the
    historical N//2 ray (off-center for hex) centers the sphere on the wrong
    point and inflates the RMS.  The default must match the index-0 chief and
    differ markedly from N//2.
    """
    ld = cooke()
    P, S = launch(ld, Field(0.0, 0.0, unit='deg'), WVL, Sampling.hex(nrings=6))
    n = P.shape[0]

    def rms(chief_index):
        opd, _, _ = pa.wavefront(ld, P, S, WVL, chief_index=chief_index)
        opd = np.asarray(opd)
        return float(np.sqrt(np.mean(opd ** 2)))

    rms_default = rms(None)
    rms_center = rms(0)          # the actual hex pupil-center ray
    rms_nhalf = rms(n // 2)      # the historical (wrong-for-hex) default
    assert rms_default == pytest.approx(rms_center, rel=1e-9)
    assert rms_nhalf > 1.2 * rms_center
