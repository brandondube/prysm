"""Tests for prysm.x.raytracing.aberrations (Seidel + primary chromatic)."""

import numpy as np
import pytest

from prysm.x.raytracing import OpticalSystem
from prysm.x.raytracing import FRAUNHOFER_LINES_UM, LensData
from prysm.x import materials
from prysm.x.raytracing.surfaces import Conic, Plane
from prysm.x.raytracing.launch import Field, Sampling, launch
from prysm.x.raytracing import analysis
from prysm.x.raytracing.aberrations import (
    seidel_aberrations,
    paraxial_trace,
    _marginal_chief_launch,
)
from prysm.x.raytracing.paraxial import paraxial_image_distance
from prysm.x.raytracing._resolve import trace_context


def _n_const(value):
    return materials.ConstantMaterial(value)


# Sellmeier-ish at 3 prysm-relevant wavelengths; enough dispersion to make
# CI/CII nonzero (representative N-BK7-ish values).
_bk7_dispersive = materials.FormulaMaterial(
    'N-BK7',
    lambda wvl: {0.4861327: 1.5224, 0.5875618: 1.5168, 0.6562725: 1.5143}[float(wvl)],
)


def _singlet(epd=8.0, c1=1 / 61.0, gap=None, material=None, dispersive=False):
    """Build a singlet with stop at S0 and an eval plane at paraxial focus.

    Caller may pass an explicit gap; otherwise it is solved to put the eval
    plane on the paraxial focus.
    """
    mat = material or _n_const(1.5168)
    probe_lens = LensData()
    (probe_lens.add(Conic(c1, 0.0), thickness=6.0, material=mat,
                    aperture=10.0)
               .add(Conic(-c1, 0.0), thickness=50.0,
                    material=materials.air, aperture=10.0))
    ld_probe = OpticalSystem(
        probe_lens, aperture=epd, fields=[Field(0, 0.0, kind='angle')],
        wavelengths=list(FRAUNHOFER_LINES_UM.values()), reference=1,
        stop_index=1)   # first powered surface (index 0 is OBJECT)
    wvl = ld_probe.wavelength()
    if gap is None:
        # measure from the last powered surface (exclude the IMAGE plane)
        gap = paraxial_image_distance(ld_probe.surfaces[:-1], wvl)
    lens = LensData()
    (lens.add(Conic(c1, 0.0), thickness=6.0, material=mat,
              aperture=10.0)
         .add(Conic(-c1, 0.0), thickness=gap, material=materials.air,
              aperture=10.0))
    ld = OpticalSystem(
        lens, aperture=epd, fields=[Field(0, 0.0, kind='angle')],
        wavelengths=(list(FRAUNHOFER_LINES_UM.values()) if dispersive
                     else [0.5875618]),
        reference=(1 if dispersive else 0), stop_index=1)
    return ld


def test_optical_invariant_constant_across_surfaces():
    """H is the Lagrange invariant -- conserved by paraxial refraction."""
    ld = _singlet()
    wvl = ld.wavelength()
    field = Field(0.0, 2.0, kind='angle')
    ctx = trace_context(ld, wvl, chief=True, epd=ld.epd, stop_index=1)
    (y0m, u0m), (y0c, u0c) = _marginal_chief_launch(ctx, field)
    marg = paraxial_trace(ld, y0m, u0m, wvl, 1.0)
    chief = paraxial_trace(ld, y0c, u0c, wvl, 1.0)
    # H = n (u y_bar - u_bar y); using before-surface quantities at each surface
    H = [m.n_b * (m.theta_b * c.y - c.theta_b * m.y)
         for m, c in zip(marg, chief)]
    np.testing.assert_allclose(H, H[0], rtol=0.0, atol=1e-12)


def test_petzval_matches_analytic_sum():
    """Total S-IV equals -H^2 * sum_i c_i (1/n'_i - 1/n_i)."""
    ld = _singlet()
    field = Field(0.0, 2.0, kind='angle')
    res = seidel_aberrations(ld, field=field)
    wvl = ld.wavelength()
    ctx = trace_context(ld, wvl, chief=True, epd=ld.epd, stop_index=1)
    (y0m, u0m), _ = _marginal_chief_launch(ctx, field)
    marg = paraxial_trace(ld, y0m, u0m, wvl, 1.0)
    P_petz = sum(m.c * (1.0 / m.n_a - 1.0 / m.n_b) for m in marg)
    expected = -res.optical_invariant ** 2 * P_petz
    np.testing.assert_allclose(res.sums['SIV'], expected,
                               rtol=1e-12, atol=1e-14)


def test_W040_matches_real_ray_rho4_coefficient():
    """Seidel W040 must match the rho^4 term of the on-axis real-ray OPD.

    Sign convention: prysm's analysis.wavefront 'length' output reports OPL
    differences (longer ray OPL -> positive), while the Seidel sign convention
    is the classical wavefront-error W (undercorrected spherical -> SI > 0).
    The two differ by an overall sign; here we compare magnitudes and require
    opposite signs.
    """
    ld = _singlet(epd=8.0)
    wvl = ld.wavelength()
    field = Field(0.0, 0.0, kind='angle')
    res = seidel_aberrations(ld, field=field)
    W040_len = res.sums['SI'] / 8.0

    P, S = launch(ld, field, wvl, Sampling.fan(n=61, axis='y'), epd=ld.epd)
    opd, _, yp = analysis.wavefront(ld, P, S, wvl, P_xp=ld.exit_pupil(wvl),
                                    output='length')
    rho = yp / (ld.epd / 2.0)
    A = np.vstack([np.ones_like(rho), rho ** 2, rho ** 4]).T
    coef, *_ = np.linalg.lstsq(A, opd, rcond=None)
    real_rho4 = float(coef[2])
    # signs must oppose (W vs OPL convention) and magnitudes must agree
    # within third-order validity at modest aperture
    assert real_rho4 * W040_len < 0
    np.testing.assert_allclose(abs(real_rho4), abs(W040_len), rtol=0.05)


def test_chromatic_terms_zero_for_nondispersive_glass():
    """A constant-n stub material -> dn = 0 at every surface -> CI = CII = 0."""
    ld = _singlet(material=_n_const(1.5168), dispersive=False)
    field = Field(0.0, 2.0, kind='angle')
    res = seidel_aberrations(ld, field=field,
                             wavelengths=[0.486, 0.588, 0.656])
    assert res.CI is not None and res.CII is not None
    np.testing.assert_allclose(res.CI, 0.0, atol=0.0)
    np.testing.assert_allclose(res.CII, 0.0, atol=0.0)


def test_chromatic_terms_nonzero_for_real_glass():
    """A dispersive material produces nonzero primary axial / lateral color."""
    ld = _singlet(material=_bk7_dispersive, dispersive=True)
    field = Field(0.0, 2.0, kind='angle')
    res = seidel_aberrations(ld, field=field)
    assert abs(res.sums['CI']) > 0
    assert abs(res.sums['CII']) > 0


def test_wavefront_coefficients_apply_classical_factors():
    """W040 = SI/8, W131 = SII/2, W222 = SIII/2, W220 = (SIII + SIV)/4,
    W311 = SV/2, all divided by wavelength in length units."""
    ld = _singlet()
    field = Field(0.0, 2.0, kind='angle')
    res = seidel_aberrations(ld, field=field)
    W = res.wavefront_coefficients()
    wvl_len = res.wavelength * 1e-3   # um -> mm
    np.testing.assert_allclose(W['W040'], 0.125 * res.sums['SI'] / wvl_len)
    np.testing.assert_allclose(W['W131'], 0.5 * res.sums['SII'] / wvl_len)
    np.testing.assert_allclose(W['W222'], 0.5 * res.sums['SIII'] / wvl_len)
    np.testing.assert_allclose(
        W['W220'], 0.25 * (res.sums['SIV'] + res.sums['SIII']) / wvl_len)
    np.testing.assert_allclose(W['W311'], 0.5 * res.sums['SV'] / wvl_len)


def test_eval_plane_contributes_zero_to_seidel():
    """A typ='eval' Plane has no power, no index change -- pure transfer."""
    ld = _singlet()
    field = Field(0.0, 2.0, kind='angle')
    res = seidel_aberrations(ld, field=field)
    last = len(res.SI) - 1
    np.testing.assert_allclose(res.SI[last], 0.0)
    np.testing.assert_allclose(res.SII[last], 0.0)
    np.testing.assert_allclose(res.SIII[last], 0.0)
    np.testing.assert_allclose(res.SIV[last], 0.0)
    np.testing.assert_allclose(res.SV[last], 0.0)


def test_seidel_requires_stop_index():
    """Without a stop the chief ray is undefined -- raise informatively."""
    ld = _singlet()
    ld.stop_index = None
    with pytest.raises(ValueError, match='entrance pupil'):
        seidel_aberrations(ld, field=Field(0.0, 2.0, kind='angle'))
