"""Tests for the determinate equally-inclined-chord (EIC) wavefront closing.

hopkins_eic_closing references every ray to the chief-ray image point through
the reference sphere, parametrized by the sphere's finite center and its
curvature kappa = 1/R.  The branch-free rational form must (1) reproduce the
explicit reference-sphere root to machine precision for a finite exit pupil,
and (2) stay finite and correctly-signed as the exit pupil recedes to infinity
(kappa -> 0, image-space telecentric), where the radius itself diverges.
"""

import numpy as np
import pytest

from prysm.x.raytracing import OpticalSystem, LensData
from prysm.x import materials
from prysm.x.raytracing.surfaces import Conic, Plane
from prysm.x.raytracing.launch import Field, Sampling, launch
from prysm.x.raytracing.spencer_and_murty import raytrace
from prysm.x.raytracing.opt import (
    hopkins_eic_closing, reference_sphere_curvature, eic_distance,
)
from prysm.x.raytracing.paraxial import (
    paraxial_image_distance, first_order,
)


def _singlet(epd=8.0):
    mat = materials.ConstantMaterial(1.5168)
    probe = LensData()
    (probe.add(Conic(1 / 61.0, 0.0), thickness=6.0, material=mat, semidiameter=10.0)
          .add(Conic(-1 / 61.0, 0.0), thickness=50.0, material=materials.air,
               semidiameter=10.0))
    sysp = OpticalSystem(probe, aperture=epd, fields=[Field(0, 0.0, kind='angle')],
                         wavelengths={'d': 0.5875618}, reference_wavelength='d',
                         stop_index=0, unit='mm')
    wvl = sysp.wavelength('d')
    foc = paraxial_image_distance(sysp, wvl)
    lens = LensData()
    (lens.add(Conic(1 / 61.0, 0.0), thickness=6.0, material=mat, semidiameter=10.0)
         .add(Conic(-1 / 61.0, 0.0), thickness=foc, material=materials.air,
              semidiameter=10.0)
         .add(Plane(), typ='eval', material=materials.air, semidiameter=12.0))
    return OpticalSystem(lens, aperture=epd, fields=[Field(0, 0.0, kind='angle')],
                         wavelengths={'d': 0.5875618}, reference_wavelength='d',
                         stop_index=0, unit='mm')


def _telecentric(epd=6.0):
    """Image-space-telecentric build: stop one front-focal-distance ahead of a
    lens, so the chief exits parallel to the axis and the exit pupil is at
    infinity (first_order().xp_z is None)."""
    mat = materials.ConstantMaterial(1.5168)
    c = 1.0 / 40.0
    probe = LensData()
    (probe.add(Conic(c, 0.0), thickness=3.0, material=mat, semidiameter=14.0)
          .add(Conic(-c, 0.0), thickness=60.0, material=materials.air,
               semidiameter=14.0))
    sp = OpticalSystem(probe, aperture=epd, fields=[Field(3, 0.0, kind='angle')],
                       wavelengths={'d': 0.5875618}, reference_wavelength='d',
                       stop_index=0, unit='mm')
    ffl = first_order(sp, 'd', stop_index=0).ffl
    lens = LensData()
    (lens.add(Plane(), typ='eval', material=materials.air, semidiameter=epd / 2)
         .add(Conic(c, 0.0), thickness=3.0, material=mat, semidiameter=20.0)
         .add(Conic(-c, 0.0), thickness=60.0, material=materials.air, semidiameter=20.0)
         .add(Plane(), typ='eval', material=materials.air, semidiameter=30.0))
    lens.rows[0].thickness = abs(ffl)
    sysT = OpticalSystem(lens, aperture=epd, fields=[Field(3, 0.0, kind='angle')],
                         wavelengths={'d': 0.5875618}, reference_wavelength='d',
                         stop_index=0, unit='mm')
    wvl = sysT.wavelength('d')
    lens.rows[2].thickness = paraxial_image_distance(sysT, wvl)
    return sysT


def _sphere_root_opd(trace, C, R, n_image, chief):
    """Explicit reference-sphere OPD oracle (t = -b - sqrt, the deleted path)."""
    P_last = trace.P[-1]
    S_last = trace.S[-1]
    d = P_last - C
    b = np.sum(S_last * d, axis=-1)
    cc = np.sum(d * d, axis=-1) - R * R
    t = -b - np.sqrt(b * b - cc)
    total = trace.OPL.sum(axis=0) + n_image * t
    return total - total[chief]


def test_closing_matches_reference_sphere_root_to_machine_precision():
    ld = _singlet()
    wvl = ld.wavelength('d')
    P, S = launch(ld, Field(0.0, 0.0, kind='angle'), wvl,
                  Sampling.fan(n=41, axis='y'), epd=ld.epd)
    trace = raytrace(ld, P, S, wvl)
    chief = P.shape[0] // 2
    C = trace.P[-1, chief]
    P_xp = np.asarray(ld.exit_pupil(wvl))
    R = float(np.sqrt(np.sum((P_xp - C) ** 2)))

    opd_oracle = _sphere_root_opd(trace, C, R, 1.0, chief)
    kappa = reference_sphere_curvature(P_xp, C)
    opd_eic = hopkins_eic_closing(trace.P, trace.S, trace.OPL,
                                  center=C, curvature=kappa,
                                  n_image=1.0, chief_index=chief)
    # the rationalized form avoids the converging-beam cancellation, so it is
    # at least as accurate as the explicit root; they agree to ~machine eps.
    np.testing.assert_allclose(opd_eic, opd_oracle, rtol=0.0, atol=1e-11)
    assert opd_eic[chief] == 0.0
    # undercorrected spherical aberration -> edge focuses short -> W040 < 0
    assert opd_eic[-1] < 0.0


def test_closing_is_finite_and_signed_at_telecentric_kappa_zero():
    ld = _telecentric()
    wvl = ld.wavelength('d')
    fo = first_order(ld, wvl, stop_index=0)
    assert fo.xp_z is None  # exit pupil genuinely at infinity
    kappa = reference_sphere_curvature(None, np.zeros(3))
    assert kappa == 0.0

    fld = Field(3.0, 0.0, kind='angle')
    P, S = launch(ld, fld, wvl, Sampling.fan(n=31, axis='y'), epd=ld.epd)
    trace = raytrace(ld, P, S, wvl)
    chief = P.shape[0] // 2
    C = trace.P[-1, chief]
    opd = hopkins_eic_closing(trace.P, trace.S, trace.OPL,
                              center=C, curvature=kappa,
                              n_image=1.0, chief_index=chief)
    assert np.all(np.isfinite(opd))
    assert opd[chief] == 0.0
    assert float(opd.max() - opd.min()) > 0.0


def test_closing_kappa_zero_is_limit_of_small_curvature():
    """kappa=0 (telecentric) is the continuous limit of a tiny finite curvature,
    not a separate branch."""
    ld = _singlet()
    wvl = ld.wavelength('d')
    P, S = launch(ld, Field(0.0, 0.0, kind='angle'), wvl,
                  Sampling.fan(n=21, axis='y'), epd=ld.epd)
    trace = raytrace(ld, P, S, wvl)
    chief = P.shape[0] // 2
    C = trace.P[-1, chief]
    opd0 = hopkins_eic_closing(trace.P, trace.S, trace.OPL, center=C,
                               curvature=0.0, n_image=1.0, chief_index=chief)
    opd_eps = hopkins_eic_closing(trace.P, trace.S, trace.OPL, center=C,
                                  curvature=1e-9, n_image=1.0, chief_index=chief)
    np.testing.assert_allclose(opd_eps, opd0, rtol=0.0, atol=1e-9)


def test_reference_sphere_curvature():
    assert reference_sphere_curvature(None, np.zeros(3)) == 0.0
    C = np.array([0.0, 0.0, 10.0])
    P_xp = np.array([0.0, 0.0, -52.0])
    assert reference_sphere_curvature(P_xp, C) == pytest.approx(1.0 / 62.0)
    with pytest.raises(ValueError, match='degenerate'):
        reference_sphere_curvature(C, C)


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
    # equal-direction limit: e(a,b) = -e(b,a) when ends are swapped
    e2 = eic_distance(P_a, d_a, P_b, d_a)
    e3 = eic_distance(P_b, d_a, P_a, d_a)
    np.testing.assert_allclose(e2, -e3, rtol=1e-14)
