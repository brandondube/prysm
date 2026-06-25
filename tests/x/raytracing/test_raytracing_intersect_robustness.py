"""First-root intersection tests for departure-banded Newton."""

import warnings

import pytest

import numpy as np

from numpy.polynomial import polynomial as npoly

from prysm.x.raytracing.surfaces import EvenAsphere, Sphere, Zernike, Surface
from prysm.x.raytracing.intersections import (
    bracketed_newton_solve_s,
    newton_raphson_solve_s,
    ray_conic_intersect,
)
from prysm.x.raytracing.spencer_and_murty import (
    raytrace,
    STATUS_OK,
    SURFACE_INTERSECTION_DEFAULT_MAXITER,
)


# gull-wing even asphere with two forward crossings for many rays.
GULL_C = 1 / 30.0
GULL_COEFS = (-2e-5, 1e-9)


def polynomial_first_root(P, S, c, coefs):
    """All forward intersections of a ray with a k=-1 even asphere, sorted."""
    Px, Py, Pz = P
    Sx, Sy, Sz = S
    rsq = np.array([Px * Px + Py * Py, 2 * (Px * Sx + Py * Sy), Sx * Sx + Sy * Sy])
    sag = npoly.polymul(np.array([c / 2]), rsq)
    p = rsq.copy()
    for a in coefs:
        p = npoly.polymul(p, rsq)
        sag = npoly.polyadd(sag, a * p)
    F = npoly.polysub(np.array([Pz, Sz]), sag)
    r = npoly.polyroots(F)
    real = r[np.abs(r.imag) < 1e-9].real
    return np.sort(real[real >= -1e-12])


def gull_wing_surface(outer_radius=30.0):
    shape = EvenAsphere(c=GULL_C, k=-1.0, coefs=GULL_COEFS)
    with warnings.catch_warnings():
        # this surface legitimately trips the multiple-crossing setup warning
        warnings.simplefilter('ignore')
        surf = Surface(shape=shape, interaction='refl', P=[0, 0, 0],
                       bounding={'outer_radius': outer_radius})
        surf.departure_band()
    return surf


# tighter fold where both crossings can land inside the domain.
FOLD_C = 1 / 40.0
FOLD_COEFS = (-1e-4, 1e-8)
FOLD_R = 20.0


def in_domain_fold_surface():
    shape = EvenAsphere(c=FOLD_C, k=-1.0, coefs=FOLD_COEFS)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        surf = Surface(shape=shape, interaction='refl', P=[0, 0, 0],
                       bounding={'outer_radius': FOLD_R})
        surf.departure_band()
    return surf


def test_gull_wing_matches_polynomial_oracle():
    """Production intersect returns the exact first root on a folded asphere."""
    surf = gull_wing_surface()
    checked = 0
    for h in np.linspace(0, 14, 8):
        for ang in np.linspace(-60, 60, 17):
            a = np.radians(ang)
            P = np.array([[0.0, h, -10.0]])
            S = np.array([[0.0, np.sin(a), np.cos(a)]])
            roots = polynomial_first_root(P[0], S[0], GULL_C, GULL_COEFS)
            Q, n, v = surf.intersect(P, S, forward_only=True)
            if len(roots) == 0:
                assert not v[0], f'traced a ray with no real intersection (h={h}, ang={ang})'
                continue
            assert v[0], f'failed to trace a ray with a real intersection (h={h}, ang={ang})'
            s_found = float(np.sum((Q[0] - P[0]) * S[0]))
            assert s_found == pytest.approx(roots[0], abs=1e-6), \
                f'wrong sheet at h={h}, ang={ang}'
            checked += 1
    # the sweep must actually exercise multi-root geometry, not vacuously pass
    assert checked > 100


def test_known_wrong_sheet_rays_are_fixed():
    """Embedded cases where the unguarded Newton converges to the far sheet."""
    surf = gull_wing_surface()
    shape = surf.shape
    # (height, angle deg): unguarded Newton lands at s=161.7 / s=-159.2; the
    # true first roots are near s=27.
    cases = [(1.0, -60.0), (0.5, 60.0), (8.0, 51.0)]
    for h, ang in cases:
        a = np.radians(ang)
        P = np.array([[0.0, h, -10.0]])
        S = np.array([[0.0, np.sin(a), np.cos(a)]])
        roots = polynomial_first_root(P[0], S[0], GULL_C, GULL_COEFS)
        assert len(roots) >= 2

        # demonstrate the unguarded path is actually wrong on this ray
        Sz = S[..., 2]
        s0 = -P[..., 2] / Sz
        P1 = P + s0[..., np.newaxis] * S
        Qc, _, _ = ray_conic_intersect(P1, S, GULL_C, -1.0)
        s1 = Qc[..., 2] / Sz
        Qn, _, vn = newton_raphson_solve_s(P1, S, shape.sag_and_normal, s1=s1)
        s_raw = float(np.sum((Qn[0] - P[0]) * S[0]))
        assert vn[0]
        assert abs(s_raw - roots[0]) > 1.0

        # the guarded production stack lands on the first root
        Q, n, v = surf.intersect(P, S, forward_only=True)
        assert v[0]
        s_found = float(np.sum((Q[0] - P[0]) * S[0]))
        assert s_found == pytest.approx(roots[0], abs=1e-6)


def test_mild_asphere_roots_unchanged_by_guard():
    """The acceptance band is transparent on well-behaved surfaces."""
    shape = EvenAsphere(c=1 / 50.0, k=0.0, coefs=(1e-7, 1e-10))
    surf = Surface(shape=shape, interaction='refl', P=[0, 0, 0],
                   bounding={'outer_radius': 15.0})
    h = np.linspace(-14, 14, 23)
    P = np.zeros((h.size, 3))
    P[:, 1] = h
    P[:, 2] = -5.0
    S = np.zeros((h.size, 3))
    S[:, 2] = 1.0
    Qg, ng, vg = surf.intersect(P, S, forward_only=True)
    # unguarded reference
    Qc, _, _ = ray_conic_intersect(P, S, 1 / 50.0, 0.0)
    s1 = Qc[..., 2]
    Qr, nr, vr = newton_raphson_solve_s(P + np.array([0, 0, 5.0]), S,
                                        shape.sag_and_normal, s1=s1)
    assert vg.all() and vr.all()
    assert np.allclose(Qg, Qr, atol=1e-10)
    assert np.allclose(ng, nr, atol=1e-10)


def test_bracketed_newton_finds_first_root_in_band():
    """Direct unit test: several crossings inside the band resolve to the first."""
    surf = gull_wing_surface()
    shape = surf.shape
    a = np.radians(-60)
    P1 = np.array([[0.0, -17.32050808, 0.0]])  # vertex-plane point of the h=0 ray
    S = np.array([[0.0, np.sin(a), np.cos(a)]])
    # band straddling both crossings of this ray (s ~ 6.7 and ~141.7 from P1);
    # the domain corridor clips the far crossing, the Lipschitz march finds the
    # near one.
    lo = np.array([-30.0])
    hi = np.array([160.0])
    Q, n, v = bracketed_newton_solve_s(P1, S, shape.sag_and_normal, lo, hi,
                                       lipschitz=surf.departure_band().lipschitz,
                                       domain_radius=30.0)
    assert v[0]
    s_found = float(np.sum((Q[0] - P1[0]) * S[0]))
    roots = polynomial_first_root(P1[0], S[0], GULL_C, GULL_COEFS)
    assert s_found == pytest.approx(roots[0], abs=1e-6)


def test_bracketed_newton_requires_lipschitz():
    """The march needs its first-root-guaranteeing bound; None is an error."""
    shape = Sphere(c=1 / 100.0)
    P1 = np.array([[0.0, 0.0, -5.0]])
    S = np.array([[0.0, 0.0, 1.0]])
    with pytest.raises(ValueError):
        bracketed_newton_solve_s(P1, S, shape.sag_and_normal,
                                 np.array([0.0]), np.array([2.0]))


def test_bracketed_newton_rejects_no_sign_change():
    """A band that never crosses the surface returns invalid, not garbage."""
    shape = Sphere(c=1 / 100.0)
    P1 = np.array([[0.0, 0.0, -5.0]])
    S = np.array([[0.0, 0.0, 1.0]])
    # band entirely before the surface: residual sign never changes
    lo = np.array([0.0])
    hi = np.array([2.0])
    Q, n, v = bracketed_newton_solve_s(P1, S, shape.sag_and_normal, lo, hi,
                                       lipschitz=1.0)
    assert not v[0]
    assert np.isnan(Q[0]).all()


def test_forward_only_rejects_root_behind_ray():
    """A surface behind the ray is a virtual intersection at reflect/refract."""
    shape = EvenAsphere(c=1 / 50.0, k=0.0, coefs=(1e-7,))
    surf = Surface(shape=shape, interaction='refl', P=[0, 0, 0],
                   bounding={'outer_radius': 15.0})
    # ray starts past the surface travelling away from it
    P = np.array([[0.0, 2.0, 5.0]])
    S = np.array([[0.0, 0.0, 1.0]])
    Q, n, v = surf.intersect(P, S, forward_only=False)
    assert v[0]
    assert float(np.sum((Q[0] - P[0]) * S[0])) < 0
    Q, n, v = surf.intersect(P, S, forward_only=True)
    assert not v[0]


def test_first_segment_exempt_from_forward_acceptance():
    """Concave-front systems launch from the vertex plane: signed first segment."""
    # concave asphere: marginal rays at the vertex plane sit past the surface
    shape = EvenAsphere(c=-1 / 40.0, k=0.0, coefs=(1e-8,))
    surf = Surface(shape=shape, interaction='refl', P=[0, 0, 0],
                   bounding={'outer_radius': 12.0})
    h = np.linspace(-10, 10, 11)
    P = np.zeros((h.size, 3))
    P[:, 1] = h
    # launch exactly at the vertex plane; concave sag puts the intersection
    # behind every off-axis ray
    S = np.zeros((h.size, 3))
    S[:, 2] = 1.0
    res = raytrace([surf], P, S, 0.5876)
    assert (res.status.imag == STATUS_OK).all()

    # the exemption is only for the first surface: launch past the asphere
    # so its intersection is behind every ray.  Alone (first surface) the
    # virtual intersection traces; behind an eval surface it is rejected.
    P_past = P.copy()
    P_past[:, 2] = 5.0
    res = raytrace([surf], P_past, S, 0.5876)
    assert (res.status.imag == STATUS_OK).all()

    eval_surf = Surface(shape=Sphere(c=0.0), interaction='eval', P=[0, 0, 5.0])
    res = raytrace([eval_surf, surf], P_past, S, 0.5876)
    assert (res.status.imag != STATUS_OK).all()
    assert (res.status.real == 2).all()


def test_departure_band_domain_resolution():
    """Domain radius: bounding, else normalization radius, else conic limit."""
    asph = EvenAsphere(c=1 / 50.0, k=0.0, coefs=(1e-7,))
    s = Surface(shape=asph, interaction='refl', P=[0, 0, 0],
                bounding={'outer_radius': 9.0})
    band = s.departure_band()
    assert band.bounded
    assert band.domain_radius == 9.0
    assert band.max_departure > 0

    zern = Zernike(c=1 / 50.0, k=0.0, normalization_radius=7.0,
                   nms=[(4, 0)], coefs=[1e-4])
    s = Surface(shape=zern, interaction='refl', P=[0, 0, 0])
    assert s.departure_band().domain_radius == 7.0

    # spherical base, no bounding, no normalization radius: the conic's own
    # domain limit 1/(|c| sqrt(1+k))
    s = Surface(shape=asph, interaction='refl', P=[0, 0, 0])
    assert s.departure_band().domain_radius == pytest.approx(0.999 * 50.0, rel=1e-12)

    # parabolic base is unbounded: no resolvable domain, guard disabled
    para = EvenAsphere(c=1 / 50.0, k=-1.0, coefs=(1e-7,))
    s = Surface(shape=para, interaction='refl', P=[0, 0, 0])
    assert not s.departure_band().bounded

    # analytic shapes carry no conic seed and no band
    s = Surface(shape=Sphere(c=1 / 50.0), interaction='refl', P=[0, 0, 0])
    assert not s.departure_band().bounded


def test_multiple_crossing_setup_warning():
    """Surfaces whose departure slope admits several crossings warn at setup."""
    shape = EvenAsphere(c=GULL_C, k=-1.0, coefs=GULL_COEFS)
    surf = Surface(shape=shape, interaction='refl', P=[0, 0, 0],
                   bounding={'outer_radius': 30.0})
    with pytest.warns(UserWarning, match='multiple ray crossings'):
        surf.departure_band()

    # mild surfaces stay silent
    mild = EvenAsphere(c=1 / 50.0, k=0.0, coefs=(1e-8,))
    surf = Surface(shape=mild, interaction='refl', P=[0, 0, 0],
                   bounding={'outer_radius': 10.0})
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        surf.departure_band()


def test_rim_grazer_oracle_agreement():
    """Near-tangent rays at the fold either match the oracle or status out."""
    surf = gull_wing_surface()
    wrong = 0
    for h in np.linspace(15, 25, 6):
        for ang in (75.0, 80.0, 85.0, -75.0, -80.0):
            a = np.radians(ang)
            P = np.array([[0.0, h, -2.0]])
            S = np.array([[0.0, np.sin(a), np.cos(a)]])
            roots = polynomial_first_root(P[0], S[0], GULL_C, GULL_COEFS)
            Q, n, v = surf.intersect(P, S, forward_only=True)
            if not v[0]:
                # deterministic rejection is an acceptable grazer outcome
                continue
            s_found = float(np.sum((Q[0] - P[0]) * S[0]))
            if len(roots) == 0 or abs(s_found - roots[0]) > 1e-6:
                wrong += 1
    assert wrong == 0


def test_in_domain_far_crossing_policed_hole():
    """A far in-domain crossing admitted by the band is corrected."""
    surf = in_domain_fold_surface()
    a = np.radians(72.0)
    P = np.array([[0.0, -18.0, -3.0]])
    S = np.array([[0.0, np.sin(a), np.cos(a)]])
    roots = polynomial_first_root(P[0], S[0], FOLD_C, FOLD_COEFS)
    assert len(roots) >= 2

    # unguarded conic-seeded Newton lands on an in-domain far crossing
    Sz = S[..., 2]
    s0 = -P[..., 2] / Sz
    P1 = P + s0[..., np.newaxis] * S
    Qc, _, hit = ray_conic_intersect(P1, S, FOLD_C, -1.0)
    assert hit[0]  # this ray IS policed (the seed conic is hit)
    s1 = Qc[..., 2] / Sz
    Qn, _, vn = newton_raphson_solve_s(P1, S, surf.shape.sag_and_normal, s1=s1)
    assert vn[0]
    s_newton = float(np.sum((Qn[0] - P1[0]) * S[0])) + s0[0]
    assert abs(s_newton - roots[1]) < 1e-4
    assert np.hypot(Qn[0, 0], Qn[0, 1]) < FOLD_R

    # guarded path returns the first crossing
    Q, n, v = surf.intersect(P, S, forward_only=True)
    assert v[0]
    s_found = float(np.sum((Q[0] - P[0]) * S[0]))
    assert s_found == pytest.approx(roots[0], abs=1e-6)


def test_in_domain_far_crossing_seed_missed_hole():
    """A seed-missed in-domain far crossing is corrected."""
    surf = in_domain_fold_surface()
    a = np.radians(82.0)
    P = np.array([[0.0, -18.0, -3.0]])
    S = np.array([[0.0, np.sin(a), np.cos(a)]])
    roots = polynomial_first_root(P[0], S[0], FOLD_C, FOLD_COEFS)
    assert len(roots) >= 2

    Sz = S[..., 2]
    s0 = -P[..., 2] / Sz
    P1 = P + s0[..., np.newaxis] * S
    Qc, _, hit = ray_conic_intersect(P1, S, FOLD_C, -1.0)
    assert not hit[0]  # the seed conic misses this grazing ray
    s1 = Qc[..., 2] / Sz
    Qn, _, vn = newton_raphson_solve_s(P1, S, surf.shape.sag_and_normal, s1=s1)
    assert vn[0]
    s_newton = float(np.sum((Qn[0] - P1[0]) * S[0])) + s0[0]
    assert abs(s_newton - roots[1]) < 1e-4
    assert np.hypot(Qn[0, 0], Qn[0, 1]) < FOLD_R

    Q, n, v = surf.intersect(P, S, forward_only=True)
    assert v[0]
    s_found = float(np.sum((Q[0] - P[0]) * S[0]))
    assert s_found == pytest.approx(roots[0], abs=1e-6)


def test_in_domain_fold_oracle_sweep():
    """Sweep the in-domain fold against the polynomial oracle."""
    surf = in_domain_fold_surface()
    checked = 0
    for Pz in (-3.0, -6.0, -12.0):
        for h in np.linspace(-18, 18, 25):
            for ang in np.linspace(-84, 84, 43):
                a = np.radians(ang)
                P = np.array([[0.0, h, Pz]])
                S = np.array([[0.0, np.sin(a), np.cos(a)]])
                roots = polynomial_first_root(P[0], S[0], FOLD_C, FOLD_COEFS)
                in_dom = [r for r in roots
                          if r > 1e-9 and abs(P[0, 1] + r * S[0, 1]) < FOLD_R]
                Q, n, v = surf.intersect(P, S, forward_only=True)
                if not v[0]:
                    continue  # deterministic rejection is acceptable
                s_found = float(np.sum((Q[0] - P[0]) * S[0]))
                # traced rays land on the first in-domain crossing
                target = in_dom[0] if in_dom else roots[0]
                assert s_found == pytest.approx(target, abs=1e-5), \
                    f'wrong sheet at Pz={Pz}, h={h}, ang={ang}'
                if in_dom:
                    checked += 1
    assert checked > 500


def test_lipschitz_march_first_root_with_far_in_domain_crossing():
    """The Lipschitz march returns the first in-domain crossing."""
    surf = in_domain_fold_surface()
    L = surf.departure_band().lipschitz
    a = np.radians(72.0)
    P = np.array([[0.0, -18.0, -3.0]])
    S = np.array([[0.0, np.sin(a), np.cos(a)]])
    Sz = S[..., 2]
    s0 = -P[..., 2] / Sz
    P1 = P + s0[..., np.newaxis] * S
    roots = polynomial_first_root(P[0], S[0], FOLD_C, FOLD_COEFS)
    first_from_P1 = roots[0] - s0[0]
    second_from_P1 = roots[1] - s0[0]
    lo = np.array([min(first_from_P1, second_from_P1) - 5.0])
    hi = np.array([max(first_from_P1, second_from_P1) + 5.0])
    Q, n, v = bracketed_newton_solve_s(P1, S, surf.shape.sag_and_normal,
                                       lo, hi, lipschitz=L,
                                       domain_radius=FOLD_R)
    assert v[0]
    s_found = float(np.sum((Q[0] - P1[0]) * S[0]))
    assert s_found == pytest.approx(first_from_P1, abs=1e-6)


def _converging_asphere_rays():
    """A mild even asphere and a fan of rays that all converge from the seed."""
    shape = EvenAsphere(c=1 / 50.0, k=0.0, coefs=(1e-7, 1e-10))
    h = np.linspace(-12, 12, 25)
    P = np.zeros((h.size, 3))
    P[:, 1] = h
    P[:, 2] = -5.0
    S = np.zeros((h.size, 3))
    S[:, 2] = 1.0
    Sz = S[..., 2]
    P1 = P + (-P[..., 2] / Sz)[..., np.newaxis] * S
    Qc, _, _ = ray_conic_intersect(P1, S, 1 / 50.0, 0.0)
    s1 = Qc[..., 2] / Sz
    return shape, P1, S, s1


def _sag_call_counter(shape):
    """Wrap a shape's sag_and_normal to count calls (one per Newton iter)."""
    calls = [0]

    def wrapped(x, y):
        calls[0] += 1
        return shape.sag_and_normal(x, y)

    return wrapped, calls


def test_nonfinite_rays_dropped_at_entry_cost_no_iterations():
    """Non-finite rays interleaved in a batch add zero Newton iterations.

    The raytrace kernel forwards clipped/failed rays to later surfaces as NaN.
    Newton cannot advance from a non-finite state, so left in the active set
    those rays would iterate all the way to maxiter on every surface.  They are
    dropped at entry instead, so a batch with NaN rays costs exactly what its
    finite subset costs, and the finite rays trace identically.
    """
    shape, P1, S, s1 = _converging_asphere_rays()

    sag_f, calls_f = _sag_call_counter(shape)
    Qf, nf, vf = newton_raphson_solve_s(P1, S, sag_f, s1=s1)
    assert vf.all()
    assert calls_f[0] < SURFACE_INTERSECTION_DEFAULT_MAXITER  # converged early

    # Interleave 40 NaN rays among the finite ones (P, S, and seed all NaN).
    nan_rows = np.full((40, 3), np.nan)
    nan_s = np.full(40, np.nan)
    Pm = np.concatenate([P1[:5], nan_rows, P1[5:]], axis=0)
    Sm = np.concatenate([S[:5], nan_rows, S[5:]], axis=0)
    s1m = np.concatenate([s1[:5], nan_s, s1[5:]], axis=0)
    fin = np.isfinite(Pm).all(axis=1)

    sag_m, calls_m = _sag_call_counter(shape)
    Qm, nm, vm = newton_raphson_solve_s(Pm, Sm, sag_m, s1=s1m)

    # The NaN rays cost no extra iterations (the maxiter spin is gone).
    assert calls_m[0] == calls_f[0]
    # Finite rays are valid and unchanged; NaN rays come back invalid + NaN.
    assert vm[fin].all()
    assert not vm[~fin].any()
    assert np.allclose(Qm[fin], Qf, atol=1e-12)
    assert np.isnan(Qm[~fin]).all()


def test_all_nonfinite_returns_without_iterating():
    """An all-NaN batch returns immediately, invalid, with no sag evaluations."""
    shape = EvenAsphere(c=1 / 50.0, k=0.0, coefs=(1e-7,))
    P = np.full((16, 3), np.nan)
    S = np.full((16, 3), np.nan)
    sag, calls = _sag_call_counter(shape)
    Q, n, v = newton_raphson_solve_s(P, S, sag, s1=0.0)
    assert not v.any()
    assert np.isnan(Q).all()
    assert calls[0] == 0
