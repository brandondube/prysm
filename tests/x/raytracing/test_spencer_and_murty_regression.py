"""Bit-equality snapshot regression for the Newton-Raphson ray-surface solver.

Locks down the f64 output of `newton_raphson_solve_s` and the conic-seeded
`EvenAsphere.intersect` path against a hand-built ray batch.  The snapshot
hashes are computed from the *current* implementation; a refactor that is
truly arithmetic-identical (same operations in the same order) will leave
them untouched.  A change that perturbs the loop's floating-point trace —
even within Newton tolerance — will flip the hash and force a deliberate
re-snapshot.

If you intentionally change the algorithm (different convergence test,
different ordering, different dtype handling), re-run with `-s` and update
the constants below from the printed values.
"""
import hashlib

import numpy as np

from tests.x.raytracing.surface_helpers import (
    plane, sphere, conic, off_axis_conic, even_asphere, q2d, zernike, xy,
    chebyshev, jacobi, toroid, biconic,
)

from prysm.x.raytracing.surfaces import Surface
from prysm.x.raytracing.spencer_and_murty import newton_raphson_solve_s


def _ray_batch():
    n = 17
    xs = np.linspace(-4.0, 4.0, n)
    X, Y = np.meshgrid(xs, xs, indexing='xy')
    Sx = 0.01 * np.sin(np.linspace(0, 3.14159, X.size))
    Sy = 0.01 * np.cos(np.linspace(0, 6.28318, X.size))
    Sz = np.sqrt(1 - Sx * Sx - Sy * Sy)
    P = np.stack([X.ravel(), Y.ravel(), np.full(X.size, -50.0)], axis=-1).astype(np.float64)
    S = np.stack([Sx, Sy, Sz], axis=-1).astype(np.float64)
    return P, S


def _asphere():
    return even_asphere(1 / 10.0, -1.0, (1e-4, 1e-6), 'refr',
                                np.array([0.0, 0.0, 0.0]), n=lambda w: 1.5)


def _md5(arr):
    return hashlib.md5(np.ascontiguousarray(arr).tobytes()).hexdigest()


def test_bare_newton_snapshot():
    """Cold-start Newton from s1=0 on an EvenAsphere FFp.  Exercises the
    'all rays start active' code path through `newton_raphson_solve_s`.
    """
    P, S = _ray_batch()
    surf = _asphere()
    Z0 = P[..., 2]
    m = S[..., 2]
    P1 = P + (-Z0 / m)[:, None] * S

    Pj, r, valid = newton_raphson_solve_s(P1, S, surf.sag_normal,
                                          s1=0.0, return_valid=True)

    assert int(valid.sum()) == 289
    assert _md5(Pj) == '18ce403132974180314561a5395e221e'
    assert _md5(r) == 'c31c2ef4e5ec3a2dc181418ba6a2e2ce'


def test_conic_seeded_newton_snapshot():
    """EvenAsphere.intersect — conic-seeded Newton.  Per-ray s1 array."""
    P, S = _ray_batch()
    surf = _asphere()

    Pj, r, valid = surf.intersect(P, S, return_valid=True)

    assert int(valid.sum()) == 289
    assert _md5(Pj) == 'd4faa6259ab9629dba7c2f6a8080e418'
    assert _md5(r) == '56e79c39e16cdb097261137f8f449fe6'


def test_partial_nonconvergence_snapshot():
    """Cap maxiter=2 so most rays fail to converge to tolerance.  Pins the
    NaN-propagation path (`Pj_out[mask] = nan`) and the `valid` mask
    construction.
    """
    P, S = _ray_batch()
    surf = _asphere()
    Z0 = P[..., 2]
    m = S[..., 2]
    P1 = P + (-Z0 / m)[:, None] * S

    Pj, r, valid = newton_raphson_solve_s(P1, S, surf.sag_normal,
                                          s1=0.0, maxiter=2,
                                          return_valid=True)

    # snapshot: 1 ray converges in 2 iters, 288 hit maxiter
    assert int(valid.sum()) == 1
    assert int(np.isnan(Pj[:, 0]).sum()) == 288
    # the one survivor's coordinates are pinned for arithmetic-identity
    assert _md5(Pj[valid]) == 'ee8eddadc83566bf5f561abe01e47986'


def test_oblique_mixed_convergence_snapshot():
    """Tiny 2-ray batch where one ray hits cleanly and one diverges.
    Locks down the "some-converge-some-don't" branch in one place.
    """
    surf = _asphere()
    P = np.array([[0.0, 0.0, -50.0], [3.0, 3.0, -50.0]], dtype=np.float64)
    S = np.array([[0.0, 0.0, 1.0], [0.9, 0.0, np.sqrt(1 - 0.81)]], dtype=np.float64)
    P1 = P + (-P[:, 2] / S[:, 2])[:, None] * S

    Pj, r, valid = newton_raphson_solve_s(P1, S, surf.sag_normal,
                                          s1=0.0, maxiter=5,
                                          return_valid=True)
    assert valid.tolist() == [True, False]
    assert Pj[0, 0] == 0.0 and Pj[0, 1] == 0.0 and Pj[0, 2] == 0.0
    assert np.isnan(Pj[1]).all()
    assert np.isnan(r[1]).all()


def test_all_converge_first_iter_snapshot():
    """Single axial ray on a Plane via the bare Newton path.  Tests the
    fast-exit branch where every ray converges before maxiter completes.
    """
    surf = plane('refl', np.array([0.0, 0.0, 0.0]))
    P = np.array([[0.0, 0.0, -10.0]], dtype=np.float64)
    S = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)
    P1 = P + (-P[:, 2] / S[:, 2])[:, None] * S

    Pj, r, valid = newton_raphson_solve_s(P1, S, surf.sag_normal,
                                          s1=0.0, maxiter=10,
                                          return_valid=True)
    assert valid.tolist() == [True]
    np.testing.assert_array_equal(Pj, np.array([[0.0, 0.0, 0.0]]))
    np.testing.assert_array_equal(r, np.array([[0.0, 0.0, 1.0]]))
