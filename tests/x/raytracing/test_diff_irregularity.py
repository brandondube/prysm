"""Phase 6: surface irregularity (CYN/CYD) tangents + Zernike-coef sensitivity.

Two stretch deliverables of the wavefront-differential roadmap:

- seed_irregularity models a Zernike surface departure delta z = a Z_n^m(x/R,y/R)
  as an added sag term with analytic partials (sags.zernike_irregularity_partials).
  Its dOPD/da map is validated against central FD of analysis.wavefront with the
  perturbed surface built as an actual Zernike shape carrying +/- h of that mode
  -- so the analytic seed is checked against the real kernel, no special casing.
- WavefrontDifferential.zernike_sensitivity fits the nominal wavefront and each
  per-tolerance map onto a Zernike basis (reusing wavefront_zernike_fit); the
  linear least-squares fit makes the fit of dW_p exactly dc/dtau_p, validated
  against central FD of wavefront_zernike_fit(wavefront(perturbed)).
"""
import numpy as np
import pytest

from prysm.x.raytracing import LensData
from prysm.x.raytracing.launch import Field, Sampling, launch
from prysm.x.raytracing.surfaces import Conic, Plane
from prysm.x.raytracing.spencer_and_murty import STYPE_EVAL
from prysm.x.raytracing.paraxial import paraxial_image_distance
from prysm.x.raytracing.analysis import wavefront, wavefront_zernike_fit
from prysm.x.raytracing.sags import zernike_irregularity_partials
from prysm.x.raytracing._diff_raytrace import (
    wavefront_with_tangents,
    seed_irregularity,
    seed_curvature,
)
from prysm.x.raytracing.tolerance import Perturbation
from prysm.x.raytracing.wavefront_differential import (
    wavefront_differential, WavefrontDifferential,
)
from prysm.x.raytracing.surfaces import Zernike
from tests.x.raytracing.surface_helpers import conic, zernike, plane


# ---------- kernel-level: seed_irregularity dW vs FD ------------------------

NG = 1.62
RN = 8.0          # irregularity normalization radius
WVL = 0.55
# base (un-irregular) surface-0 conic; the perturbed system swaps in a Zernike
C0, K0 = 1 / 40.0, -0.6


def make_system(irr=None):
    """Two-surface refractor.  irr=((n, m), amp) puts that Zernike mode on s0."""
    n_glass = lambda w: NG
    if irr is None:
        s0 = conic(c=C0, k=K0, typ='refr', P=[0, 0, 0.0], n=n_glass)
    else:
        (n, m), amp = irr
        s0 = zernike(c=C0, k=K0, normalization_radius=RN, nms=[(n, m)],
                     coefs=[amp], typ='refr', P=[0, 0, 0.0], n=n_glass)
    s1 = conic(c=-1 / 55.0, k=0.2, typ='refr', P=[0, 0, 6.0], n=lambda w: 1.0)
    img = plane(typ='eval', P=[0, 0, 56.0])
    return [s0, s1, img]


def ray_bundle():
    ax, ay = 0.04, 0.06
    Sx, Sy = np.sin(ax), np.sin(ay)
    Sz = np.sqrt(1.0 - Sx * Sx - Sy * Sy)
    xs = np.linspace(-7, 7, 5)
    ys = np.linspace(-7, 7, 5)
    XX, YY = np.meshgrid(xs, ys)
    pupil = np.stack([XX.ravel(), YY.ravel()], axis=-1)
    n = pupil.shape[0]
    P = np.empty((n, 3))
    P[:, :2] = pupil
    P[:, 2] = -12.0
    S = np.broadcast_to(np.array([Sx, Sy, Sz]), (n, 3)).copy()
    return P, S


# CYN = Z(2,2) cylinder along axes; CYD = Z(2,-2) 45-degree cylinder; plus a
# power, a spherical, and a coma term to exercise general (n, m).
@pytest.mark.parametrize('mode', [(2, 2), (2, -2), (2, 0), (4, 0), (3, 1)])
def test_irregularity_dW_matches_fd(mode):
    P, S = ray_bundle()
    seed = seed_irregularity(0, mode[0], mode[1], RN)
    _, _, _, dW = wavefront_with_tangents(make_system(), P, S, WVL, [seed])
    h = 1e-6
    op, _, _ = wavefront(make_system((mode, +h)), P, S, WVL)
    om, _, _ = wavefront(make_system((mode, -h)), P, S, WVL)
    dW_fd = (op - om) / (2 * h)
    # FD of the composed trace->OPD pipeline is truncation-limited; the analytic
    # tangent is the more accurate of the two.
    np.testing.assert_allclose(dW[:, 0], dW_fd, rtol=1e-5, atol=1e-7)


def test_irregularity_waves_output_scales():
    P, S = ray_bundle()
    mode = (2, 2)
    seed = seed_irregularity(0, *mode, RN)
    _, _, _, dW = wavefront_with_tangents(make_system(), P, S, WVL, [seed],
                                          output='waves')
    h = 1e-6
    op, _, _ = wavefront(make_system((mode, +h)), P, S, WVL, output='waves')
    om, _, _ = wavefront(make_system((mode, -h)), P, S, WVL, output='waves')
    np.testing.assert_allclose(dW[:, 0], (op - om) / (2 * h),
                               rtol=1e-5, atol=1e-6)


def test_chief_irregularity_tangent_is_zero():
    """The chief OPD is identically 0, so every dW column vanishes there."""
    P, S = ray_bundle()
    center = np.mean(P[:, :2], axis=0)
    chief = int(np.argmin(np.sum((P[:, :2] - center) ** 2, axis=1)))
    _, _, _, dW = wavefront_with_tangents(
        make_system(), P, S, WVL,
        [seed_irregularity(0, 2, 2, RN), seed_irregularity(1, 2, -2, RN)])
    np.testing.assert_allclose(dW[chief], 0.0, atol=1e-12)


def test_multiple_irregularity_seeds_one_trace():
    P, S = ray_bundle()
    seeds = [seed_irregularity(0, 2, 2, RN, name='CYN'),
             seed_irregularity(0, 2, -2, RN, name='CYD'),
             seed_curvature(1)]
    _, _, _, dW = wavefront_with_tangents(make_system(), P, S, WVL, seeds)
    assert dW.shape[1] == 3
    h = 1e-6
    for p, mode in enumerate([(2, 2), (2, -2)]):
        op, _, _ = wavefront(make_system((mode, +h)), P, S, WVL)
        om, _, _ = wavefront(make_system((mode, -h)), P, S, WVL)
        np.testing.assert_allclose(dW[:, p], (op - om) / (2 * h),
                                   rtol=1e-5, atol=1e-7)


def test_irregularity_partials_value_matches_zernike_surface():
    """The seed's sag partial is the Zernike surface's per-unit-amplitude sag."""
    x = np.linspace(-6, 6, 9)
    y = np.linspace(-5, 5, 9)
    sag_t, gx_t, gy_t = zernike_irregularity_partials(2, 2, x, y, RN)
    # a unit-amplitude Zernike-only sag is exactly the amplitude partial
    shape = Zernike(0.0, 0.0, RN, [(2, 2)], [1.0])
    np.testing.assert_allclose(sag_t, shape.sag(x, y), rtol=1e-12, atol=1e-12)


# ---------- front-end: Zernike-coefficient sensitivity vs FD ----------------

NG_LD = 1.6


def _glass(w):
    return NG_LD


def _air(w):
    return 1.0


def singlet():
    ld = (LensData(epd=10.0, wavelengths=[0.5], n_ambient=1.0)
          .add(Conic(1 / 30.0, 0.0), typ='refr', thickness=4.0, material=_glass)
          .add(Conic(-1 / 30.0, 0.0), typ='refr', thickness=20.0, material=_air)
          .add(Plane(), typ='eval'))
    lens = [s for s in ld.to_surfaces() if s.typ != STYPE_EVAL]
    bfd = float(paraxial_image_distance(lens, wvl=0.5, n_ambient=1.0))
    ld.rows[1].thickness = bfd
    ld._invalidate()
    return ld


def _bundle(ld):
    return launch(ld, Field(2.5, 2.5), 0.5, Sampling.rect(n=7),
                  epd=10.0, pupil_z=-5.0, aim_pupil=False)


def _perts(ld):
    return [
        Perturbation.normal(ld, 'curvature', 0, 1e-5, name='c1'),
        Perturbation.normal(ld, 'conic', 0, 1e-4, name='k1'),
        Perturbation.normal(ld, 'thickness', 0, 5e-4, name='t0'),
    ]


NMS = [(2, 0), (2, 2), (2, -2), (3, 1), (3, -1), (4, 0)]


def test_zernike_sensitivity_nominal_matches_direct_fit():
    ld = singlet()
    P, S = _bundle(ld)
    wd = wavefront_differential(ld, _perts(ld), P, S, 0.5)
    R = float(np.sqrt(np.max(wd.x_pupil ** 2 + wd.y_pupil ** 2)))
    nom, _ = wd.zernike_sensitivity(NMS, normalization_radius=R)
    direct, _ = wavefront_zernike_fit(wd.W0, wd.x_pupil, wd.y_pupil, NMS,
                                      normalization_radius=R)
    np.testing.assert_allclose(nom, direct, rtol=1e-12, atol=1e-14)


def test_zernike_sensitivity_matches_fd():
    ld = singlet()
    P, S = _bundle(ld)
    perts = _perts(ld)
    wd = wavefront_differential(ld, perts, P, S, 0.5)
    R = float(np.sqrt(np.max(wd.x_pupil ** 2 + wd.y_pupil ** 2)))
    _, dc = wd.zernike_sensitivity(NMS, normalization_radius=R)

    def fit_perturbed(pert, T):
        try:
            pert.set(pert.nominal + T)
            opd, x, y = wavefront(ld, P, S, 0.5, output='length')
            c, _ = wavefront_zernike_fit(opd, x, y, NMS,
                                         normalization_radius=R)
        finally:
            pert.reset()
        return np.asarray(c)

    for p, pert in enumerate(perts):
        h = pert.step
        dc_fd = (fit_perturbed(pert, +h) - fit_perturbed(pert, -h)) / (2 * h)
        np.testing.assert_allclose(dc[:, p], dc_fd, rtol=1e-4, atol=1e-7)


def test_zernike_sensitivity_requires_pupil_coords():
    # a bare constructor with no recorded pupil coordinates
    wd = WavefrontDifferential(np.zeros(5), np.zeros((5, 1)))
    with pytest.raises(ValueError, match='pupil coordinates'):
        wd.zernike_sensitivity([(2, 0)])


# ---------- front-end: irregularity as an extra-seed tolerance column -------

def test_extra_seeds_irregularity_is_a_tolerance_column():
    ld = singlet()
    P, S = _bundle(ld)
    perts = [Perturbation.normal(ld, 'curvature', 0, 1e-5, name='c1')]
    irr = [seed_irregularity(0, 2, 2, 5.0, name='CYN'),
           seed_irregularity(1, 2, -2, 5.0, name='CYD')]
    wd = wavefront_differential(ld, perts, P, S, 0.5, extra_seeds=irr,
                                extra_steps=[0.1, 0.1])
    assert wd.n_params == 3
    assert wd.names == ['c1', 'CYN', 'CYD']
    # the irregularity columns get the full TOR treatment (finite self terms)
    assert wd.A[1] > 0 and wd.A[2] > 0
    rows = wd.rows()
    assert rows[1]['scale'] == 0.1 and rows[2]['scale'] == 0.1


def test_extra_seeds_column_equals_standalone_seed():
    """The extra_seeds irregularity column is the same map a standalone trace
    yields -- so it inherits the kernel-level FD validation above."""
    ld = singlet()
    P, S = _bundle(ld)
    perts = [Perturbation.normal(ld, 'curvature', 0, 1e-5, name='c1')]
    irr = seed_irregularity(0, 2, 2, 5.0, name='CYN')
    wd = wavefront_differential(ld, perts, P, S, 0.5, extra_seeds=[irr])
    _, _, _, dW = wavefront_with_tangents(ld.to_surfaces(), P, S, 0.5, [irr])
    np.testing.assert_allclose(wd.dW[:, 1], dW[:, 0], rtol=1e-10, atol=1e-12)


def test_extra_seeds_compose_with_compensators():
    ld = singlet()
    P, S = _bundle(ld)
    perts = [Perturbation.normal(ld, 'curvature', 0, 1e-5, name='c1')]
    irr = [seed_irregularity(0, 2, 2, 5.0, name='CYN')]
    comp = [Perturbation.normal(ld, 'thickness', 1, 1e-3, name='focus')]
    wd = wavefront_differential(ld, perts, P, S, 0.5, extra_seeds=irr,
                                compensators=comp)
    assert wd.is_compensated
    assert wd.n_params == 2                 # c1 + CYN, comp is not a column
    assert wd.compensator_motions().shape == (1, 2)
