"""Entrance-pupil routing in launch() and its analysis consequences.

launch() positions off-axis bundles so the pupil sampling lands on the
paraxial entrance pupil when the stop is known (a LensData carries
stop_index).  Before this, bundles were launched centered on the axis at the
first surface, which is only correct when the stop is the first surface; for a
stop in the middle of the system (the common case) off-axis spot/distortion/
field-curvature/wavefront were wrong.

"""
import numpy as np
import pytest

from prysm.x.raytracing import OpticalSystem
from prysm.x.raytracing import LensData, launch, raytrace, Sampling, Field
from prysm.x.raytracing.surfaces import Conic, Plane
from prysm.x import materials as pmat
from prysm.x.raytracing.paraxial import ynu_first_order, entrance_pupil_z
from prysm.x.raytracing import analysis as pa

# constant indices @ 0.55 um for a compact Cooke-style routing fixture
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
    lens = LensData()
    for R, t, n in _COOKE:
        mat = pmat.ConstantMaterial(n) if n != 1.0 else pmat.air
        lens.add(Conic(1.0 / R, 0.0), thickness=t, material=mat)
    lens.add(Plane(), typ='eval', material=pmat.air, semidiameter=1e3)
    return OpticalSystem(lens, aperture=EPD, fields=[0.0, 14.0, 20.0],
                         wavelengths=[WVL], reference=0,
                         stop_index=STOP_INDEX)


def biconvex_stop_first():
    """Stop at the first surface -> entrance pupil at the first surface."""
    lens = LensData()
    lens.add(Conic(1 / 50.0, 0.0), thickness=6.0, material=pmat.ConstantMaterial(1.5))
    lens.add(Conic(-1 / 50.0, 0.0), thickness=46.0, material=pmat.air)
    lens.add(Plane(), typ='eval', material=pmat.air, semidiameter=1e3)
    return OpticalSystem(lens, aperture=20.0, fields=[0.0, 10.0],
                         wavelengths=[WVL], reference=0,
                         stop_index=0)


# ---------- entrance_pupil_z ------------------------------------------------

def test_entrance_pupil_z_matches_first_order():
    ld = cooke()
    ep = entrance_pupil_z(ld)
    assert ep == pytest.approx(ynu_first_order(ld).ep_z)


def test_entrance_pupil_z_none_without_stop():
    # a bare Surface sequence carries no stop_index
    presc = list(cooke().to_surfaces())
    assert entrance_pupil_z(presc, wvl=WVL) is None


def test_entrance_pupil_z_at_first_surface_when_stop_is_first():
    ld = biconvex_stop_first()
    assert entrance_pupil_z(ld) == pytest.approx(0.0, abs=1e-9)


# ---------- ray aiming mode (paraxial vs real) ------------------------------

def _y_at_stop(sys, field):
    P, S = launch(sys, field, WVL, Sampling.fan(n=11, axis='y'))
    tr = raytrace(sys, P, S, WVL)
    return tr.P[STOP_INDEX + 1, :, 1]


def test_real_ray_aiming_lands_chief_on_stop_center():
    """ray_aiming='real' drives the chief exactly onto the stop center at a
    wide field where paraxial entrance-pupil routing leaves a residual (the
    real chief ray's pupil aberration)."""
    fld = Field(0.0, 20.0, unit='deg')
    chief_par = abs(_y_at_stop(cooke(), fld)[5])         # center sample == chief
    real_sys = cooke()
    real_sys.ray_aiming = 'real'
    chief_real = abs(_y_at_stop(real_sys, fld)[5])
    assert chief_par > 1e-4          # paraxial routing misses the stop center
    assert chief_real < 1e-9         # real aiming nails it


def test_real_ray_aiming_linearizes_pupil_to_stop_map():
    """Real aiming maps the normalized pupil linearly onto the stop (the
    pupil-distortion correction), holding the rim marginal; paraxial routing
    does not."""
    fld = Field(0.0, 20.0, unit='deg')
    rho = np.linspace(-1.0, 1.0, 11)             # fan(n=11) normalized pupil
    real_sys = cooke()
    real_sys.ray_aiming = 'real'
    y_real = _y_at_stop(real_sys, fld)
    y_par = _y_at_stop(cooke(), fld)
    # y_stop / rho is constant under real aiming (linear map), not under paraxial
    nz = rho != 0.0
    ratio_real = y_real[nz] / rho[nz]
    ratio_par = y_par[nz] / rho[nz]
    assert np.std(ratio_real) < 1e-6
    assert np.std(ratio_par) > 1e-3
    # the aperture (rim-to-rim span at the stop) is held by the secant scale
    np.testing.assert_allclose(y_real[-1] - y_real[0], y_par[-1] - y_par[0],
                               rtol=1e-6)


# ---------- Phase 5: real-aiming field-continuation ladder ------------------

from prysm.x.raytracing.launch import (
    _collimated_PS, _apply_vignetting, _entrance_pupil_z, _real_aim_to_stop,
    _scaled_field, _parabasal_ep_z,
)

_FISHEYE_STOP = 2


def fisheye(epd, ray_aiming='real'):
    """Wide-angle retrofocus with a strong negative front group."""
    NG = pmat.ConstantMaterial(1.6)
    ld = LensData()
    ld.add(Conic(1 / 40.0, 0.0), thickness=3.0, material=NG, semidiameter=14.0)
    ld.add(Conic(1 / 9.0, 0.0), thickness=22.0, material=pmat.air,
           semidiameter=9.0)
    ld.add(Conic(1 / 16.0, 0.0), thickness=4.0, material=NG, semidiameter=6.0)
    ld.add(Conic(-1 / 16.0, 0.0), thickness=45.0, material=pmat.air,
           semidiameter=6.0)
    ld.add(Plane(), typ='eval', material=pmat.air, semidiameter=1e4)
    sys = OpticalSystem(ld, aperture=epd, fields=[0.0], wavelengths=[WVL],
                        reference=0, stop_index=_FISHEYE_STOP)
    sys.ray_aiming = ray_aiming
    return sys


def _primary_only(sys, angle, epd, n=15):
    """Real aiming without the continuation ladder."""
    ep = _entrance_pupil_z(sys, WVL)
    pupil_z = float(list(sys.to_surfaces())[0].P[2])
    fld = Field(0.0, float(angle), unit='deg')
    pupil_xy = _apply_vignetting(
        Sampling.fan(n=n, axis='y').build(epd / 2), fld).astype(float)
    P, S = _collimated_PS(pupil_xy, pupil_z, fld)
    S0 = S[0]
    shift = (pupil_z - ep) / S0[2]
    P = P + np.stack([shift * S0[0], shift * S0[1], np.zeros_like(shift)])
    return _real_aim_to_stop(P, S, pupil_xy / (epd / 2), sys, _FISHEYE_STOP,
                             WVL, False)


def test_ladder_rescues_wide_field_marginal_rays():
    """At wide field the ladder converges more rays than primary aiming."""
    epd, ang, n = 6.0, 50.0, 15
    sys = fisheye(epd)
    _, _, conv_primary = _primary_only(sys, ang, epd, n=n)
    P, S = launch(sys, Field(0.0, ang, unit='deg'), WVL,
                  Sampling.fan(n=n, axis='y'))
    finite_ladder = np.isfinite(
        raytrace(sys, P, S, WVL).P[_FISHEYE_STOP + 1, :, 1]).sum()
    assert int(conv_primary.sum()) < 13          # paraxial seed leaves rays out
    assert int(finite_ladder) > int(conv_primary.sum())  # ladder rescues them


def test_ladder_holds_chief_on_stop_center_at_wide_field():
    """The ladder keeps the chief on the stop center while rescuing marginals."""
    sys = fisheye(6.0)
    P, S = launch(sys, Field(0.0, 50.0, unit='deg'), WVL,
                  Sampling.fan(n=15, axis='y'))
    y = raytrace(sys, P, S, WVL).P[_FISHEYE_STOP + 1, :, 1]
    assert abs(y[7]) < 1e-9                       # center sample == chief (n=15)


def test_ladder_is_dormant_when_primary_converges():
    """When primary aiming converges, the ladder returns it unchanged."""
    epd, ang, n = 4.0, 20.0, 15
    sys = fisheye(epd)
    Pp, Sp, conv = _primary_only(sys, ang, epd, n=n)
    assert bool(np.all(conv))                     # primary fully converges here
    P, S = launch(sys, Field(0.0, ang, unit='deg'), WVL,
                  Sampling.fan(n=n, axis='y'))
    np.testing.assert_array_equal(P, Pp)
    np.testing.assert_array_equal(S, Sp)


def test_ladder_is_best_effort_and_never_raises():
    """Inaccessible fields return best-effort without raising."""
    sys = fisheye(6.0)
    # 70 deg is past the real aperture.
    P, S = launch(sys, Field(0.0, 70.0, unit='deg'), WVL,
                  Sampling.fan(n=15, axis='y'))   # must not raise
    assert P.shape == (15, 3) and S.shape == (15, 3)


def test_ladder_never_worse_than_primary_across_fields():
    """The per-ray merge guarantees the ladder is never worse than the primary."""
    epd, n = 7.0, 15
    for ang in (44.0, 48.0, 52.0):
        sys = fisheye(epd)
        _, _, conv_primary = _primary_only(sys, ang, epd, n=n)
        P, S = launch(sys, Field(0.0, ang, unit='deg'), WVL,
                      Sampling.fan(n=n, axis='y'))
        finite_ladder = np.isfinite(
            raytrace(sys, P, S, WVL).P[_FISHEYE_STOP + 1, :, 1]).sum()
        assert int(finite_ladder) >= int(conv_primary.sum())


def test_drop_unaimed_nans_unaimable_ray_directions():
    """drop_unaimed NaNs only unaimable ray directions."""
    sys = fisheye(6.0)
    fld = Field(0.0, 30.0, unit='deg')             # the +y rim is unaimable here
    samp = Sampling.fan(n=15, axis='y')
    P_be, S_be = launch(sys, fld, WVL, samp)                      # best-effort
    P_dr, S_dr = launch(sys, fld, WVL, samp, drop_unaimed=True)   # dropped

    unaimable = ~np.isfinite(S_dr).all(axis=1)
    assert unaimable.any() and not unaimable.all()   # partial: chief/core aim
    assert np.isfinite(S_be).all()                   # default keeps them finite
    np.testing.assert_array_equal(P_be, P_dr)        # positions untouched
    np.testing.assert_array_equal(S_dr[~unaimable], S_be[~unaimable])
    # Best-effort launch would still trace the dropped ray.
    img_be = raytrace(sys, P_be, S_be, WVL).P[-1]
    assert np.isfinite(img_be[unaimable]).all()


def test_ray_fans_truncate_unaimable_rim_without_kink():
    """Wide-field fans NaN-truncate unaimable rim rays."""
    sys = fisheye(6.0)
    grid = pa.ray_aberration_fans(sys, fields=[Field(0.0, 30.0, unit='deg')],
                                  wavelengths=[WVL], nrays=15)
    yfan = grid.y[0, 0]
    assert np.isnan(yfan).any()                      # unaimable rim truncated
    finite_idx = np.flatnonzero(np.isfinite(yfan))
    assert finite_idx.size >= 12 and 7 in finite_idx  # chief (center) survives
    # surviving samples remain contiguous
    assert np.array_equal(finite_idx,
                          np.arange(finite_idx[0], finite_idx[-1] + 1))


def test_scaled_field_scales_field_coordinates():
    f = Field(3.0, -4.0, unit='deg')
    h = _scaled_field(f, 0.25)
    assert (h.hx, h.hy) == pytest.approx((0.75, -1.0))
    assert h.kind == 'angle' and h.unit == 'deg'


def test_parabasal_ep_z_is_field_dependent():
    """The parabasal EP walks with field; it differs from the paraxial pupil."""
    sys = fisheye(6.0)
    ep_paraxial = _entrance_pupil_z(sys, WVL)
    ep_wide = _parabasal_ep_z(sys, Field(0.0, 50.0, unit='deg'), WVL)
    assert isinstance(ep_wide, float)
    assert abs(ep_wide - ep_paraxial) > 0.5      # genuine field-dependent walk


def test_ray_aiming_paraxial_is_the_default():
    assert cooke().ray_aiming == 'paraxial'


# ---------- routing geometry ------------------------------------------------

def test_routed_chief_passes_through_stop_center_paraxially():
    """At a tiny field the routed chief lands on the stop center."""
    ld = cooke()
    P, S = launch(ld, Field(0.0, 1e-3, unit='deg'), WVL, Sampling.chief())
    tr = raytrace(ld, P, S, WVL)
    # P history: row k+1 is the intersection at surface index k
    xy_at_stop = tr.P[STOP_INDEX + 1, 0, :2]
    np.testing.assert_allclose(xy_at_stop, 0.0, atol=1e-5)


def test_routing_noop_for_stop_at_first_surface():
    """With the stop at surface 1 the EP is the first surface: no shift."""
    ld = biconvex_stop_first()
    f = Field(0.0, 10.0, unit='deg')
    P, _ = launch(ld, f, WVL, Sampling.chief())
    np.testing.assert_allclose(P[0, :2], 0.0, atol=1e-12)
    np.testing.assert_allclose(P[0, 2], 0.0, atol=1e-12)


def test_wavefront_default_chief_is_hex_center():
    """The hex center ray is index 0, not N//2; wavefront must pick it.

    With the correct (center) chief the reference sphere is centered on the
    true image point and the OPD is the system wavefront error; picking the
    N//2 ray (off-center for hex) subtracts the wrong reference ray.  The
    default must match the index-0 chief and differ markedly from N//2.
    """
    ld = cooke()
    xp_z = ynu_first_order(ld).xp_z
    P, S = launch(ld, Field(0.0, 0.0, unit='deg'), WVL, Sampling.hex(nrings=6))
    n = P.shape[0]

    def rms(chief_index):
        opd, _, _ = pa.wavefront(ld, P, S, WVL, chief_index=chief_index,
                                 P_xp=(0, 0, xp_z))
        opd = np.asarray(opd)
        return float(np.sqrt(np.mean(opd ** 2)))

    rms_default = rms(None)
    rms_center = rms(0)          # the actual hex pupil-center ray
    rms_nhalf = rms(n // 2)      # an off-center hex ray
    assert rms_default == pytest.approx(rms_center, rel=1e-9)
    assert abs(rms_nhalf - rms_center) > 0.2 * rms_center


def test_launch_threads_aim_strict_to_aim_rays(monkeypatch):
    """launch(aim_to=...) must forward aim_strict to aim_rays.

    Without an opt-out a vignetting study (some rays unaimable) aborts the whole
    launch; aim_strict=False routes through to aim_rays(strict=False) so those
    rays return best-effort instead.  The same explicit aiming branch selects
    position variation for collimated bundles and direction variation for
    finite-conjugate bundles.
    """
    import importlib
    launch_mod = importlib.import_module('prysm.x.raytracing.launch')

    captured = []

    def fake_aim(P, S, prescription, surface_index, target_xy, wvl, **kw):
        captured.append((kw.get('strict'), kw.get('vary')))
        return P, S, np.ones(P.shape[0], dtype=bool)

    monkeypatch.setattr(launch_mod, 'aim_rays', fake_aim)
    ld = cooke()
    angle_fld = Field(0.0, 14.0, unit='deg')
    height_fld = Field(0.0, 1.0, kind='height', object_z=-100.0)
    launch(ld, angle_fld, WVL, Sampling.fan(n=5), aim_to=STOP_INDEX)
    launch(ld, height_fld, WVL, Sampling.fan(n=5), aim_to=STOP_INDEX,
           aim_strict=False)
    assert captured == [(True, 'position'), (False, 'direction')]
