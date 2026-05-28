"""Tests for the EIC (Welford-rationalized) OPD path."""

import numpy as np

from prysm.x.raytracing import FRAUNHOFER_LINES_UM, LensData
from prysm.x.raytracing import materials
from prysm.x.raytracing.surfaces import Conic, Plane
from prysm.x.raytracing.launch import Field, Sampling, launch
from prysm.x.raytracing import analysis
from prysm.x.raytracing.opt import (
    opd_from_raytrace, opd_from_raytrace_eic, xp_reference_sphere,
    eic_distance,
)
from prysm.x.raytracing.spencer_and_murty import raytrace
from prysm.x.raytracing.paraxial import paraxial_image_distance


def _n_const(value):
    def n(wvl):
        return value
    return n


def _singlet(epd=8.0):
    mat = _n_const(1.5168)
    ld_probe = (LensData(epd=epd, fields=[Field(0, 0.0, kind='angle')],
                         wavelengths={'d': 0.5875618},
                         reference_wavelength='d', stop_index=0, unit='mm')
                .add(Conic(1 / 61.0, 0.0), thickness=6.0, material=mat,
                     semidiameter=10.0)
                .add(Conic(-1 / 61.0, 0.0), thickness=50.0,
                     material=materials.air, semidiameter=10.0))
    wvl = ld_probe.wavelength('d')
    foc = paraxial_image_distance(ld_probe, wvl)
    ld = (LensData(epd=epd, fields=[Field(0, 0.0, kind='angle')],
                   wavelengths={'d': 0.5875618},
                   reference_wavelength='d', stop_index=0, unit='mm')
          .add(Conic(1 / 61.0, 0.0), thickness=6.0, material=mat,
               semidiameter=10.0)
          .add(Conic(-1 / 61.0, 0.0), thickness=foc, material=materials.air,
               semidiameter=10.0)
          .add(Plane(), typ='eval', material=materials.air,
               semidiameter=12.0))
    return ld


def test_eic_matches_sphere_on_finite_conjugates():
    """For a normal finite-conjugate system the rationalized form gives a
    bit-identical OPD to the legacy explicit-sphere intersection (the
    cancelling branch is the converging-beam case but at modest aperture
    the loss is well below 1 ULP)."""
    ld = _singlet()
    wvl = ld.wavelength('d')
    field = Field(0.0, 0.0, kind='angle')
    P, S = launch(ld, field, wvl, Sampling.fan(n=41, axis='y'), epd=ld.epd)
    opd_sphere, _, _ = analysis.wavefront(ld, P, S, wvl, method='sphere',
                                           output='length')
    opd_eic, _, _ = analysis.wavefront(ld, P, S, wvl, method='eic',
                                        output='length')
    np.testing.assert_allclose(opd_eic, opd_sphere, rtol=0.0, atol=1e-12)


def test_eic_plane_fallback_triggers_for_infinite_reference():
    """When P_xp is far enough that R exceeds infinite_threshold the EIC
    path uses the planar reference and still returns finite, chief-zeroed
    OPD (the legacy sphere path can't even define R for true infinity)."""
    ld = _singlet()
    wvl = ld.wavelength('d')
    field = Field(0.0, 0.0, kind='angle')
    P, S = launch(ld, field, wvl, Sampling.fan(n=21, axis='y'), epd=ld.epd)
    trace = raytrace(ld, P, S, wvl, n_ambient=1.0)
    chief = P.shape[0] // 2
    P_img = trace.P[-1, chief]
    # push P_xp far behind the image so R is enormous -> plane fallback
    P_xp = P_img + np.array([0.0, 0.0, -1.0e12])
    opd = opd_from_raytrace_eic(trace.P, trace.S, trace.OPL, P_img, P_xp,
                                n_image=1.0, chief_index=chief,
                                infinite_threshold=1.0e6)
    assert np.all(np.isfinite(opd))
    np.testing.assert_allclose(opd[chief], 0.0, atol=0.0)
    # off-chief rays carry non-trivial wavefront content
    assert np.max(np.abs(opd)) > 0


def test_eic_sphere_preserves_aberration_content_at_very_large_R():
    """At R far beyond a normal lens-design radius the spherical leg
    n_image * s is dominated by the bulk -R term but EIC's rationalized
    form preserves the small (aberration-carrying) beta = d.(P_e - P_img)
    correction.  The aberration spread of the OPD must therefore stay
    close to the small-R value, not collapse to zero.
    """
    ld = _singlet()
    wvl = ld.wavelength('d')
    field = Field(0.0, 0.0, kind='angle')
    P, S = launch(ld, field, wvl, Sampling.fan(n=21, axis='y'), epd=ld.epd)
    trace = raytrace(ld, P, S, wvl, n_ambient=1.0)
    chief = P.shape[0] // 2
    P_img = trace.P[-1, chief]

    P_xp_small = P_img + np.array([0.0, 0.0, -100.0])
    opd_ref = opd_from_raytrace_eic(trace.P, trace.S, trace.OPL,
                                    P_img, P_xp_small, chief_index=chief,
                                    infinite_threshold=np.inf)
    span_ref = float(opd_ref.max() - opd_ref.min())

    P_xp_big = P_img + np.array([0.0, 0.0, -1.0e10])
    opd_big = opd_from_raytrace_eic(trace.P, trace.S, trace.OPL,
                                    P_img, P_xp_big, chief_index=chief,
                                    infinite_threshold=np.inf)
    span_big = float(opd_big.max() - opd_big.min())

    # the aberration spread must be preserved (within a few percent) at
    # large R; legacy form preserves it here too -- the headline value of
    # EIC's rationalized form is freedom from worrying about cancellation
    # at all, plus the plane fallback when R is genuinely infinite.
    np.testing.assert_allclose(span_big, span_ref, rtol=0.05)


def test_eic_distance_matches_definition():
    """Sanity-check the eic_distance primitive against its definition."""
    rng = np.random.default_rng(0)
    P_a = rng.normal(size=(5, 3))
    P_b = rng.normal(size=(5, 3))
    d_a = rng.normal(size=(5, 3))
    d_a /= np.linalg.norm(d_a, axis=-1, keepdims=True)
    d_b = rng.normal(size=(5, 3))
    d_b /= np.linalg.norm(d_b, axis=-1, keepdims=True)
    e = eic_distance(P_a, d_a, P_b, d_b)
    expected = (((d_a + d_b) * (P_a - P_b)).sum(-1)
                / (1.0 + (d_a * d_b).sum(-1)))
    np.testing.assert_allclose(e, expected, rtol=1e-14)
    # symmetry: e(a,b) and e(b,a) are signed mirrors when ends are swapped --
    # the relation is e(a,b) - e(b,a) = (d_a - d_b)·(P_a - P_b) / (1 + d_a·d_b)
    # in the equal-direction limit d_a == d_b: e(a,b) = -e(b,a)
    e2 = eic_distance(P_a, d_a, P_b, d_a)
    e3 = eic_distance(P_b, d_a, P_a, d_a)
    np.testing.assert_allclose(e2, -e3, rtol=1e-14)
