"""Snapshot regression for the Newton-Raphson ray-surface solver.

Locks down the f64 output of `newton_raphson_solve_s` and the conic-seeded
`EvenAsphere.intersect` path against a hand-built ray batch, comparing to
reference arrays committed under `_snapshots/`.  The references are compared
with `np.allclose` rather than byte-equality: off-axis rays converge to
values that differ in the last few ULPs across platforms (libm sin/cos/sqrt
and FMA contraction differ between macOS-arm and Linux-x86), so a bit-exact
hash is not portable.  The tolerance is tight enough to catch a real
algorithmic regression while tolerating last-bit platform noise.

If you intentionally change the algorithm, regenerate the `_snapshots/*.npy`
files from the current implementation.
"""
import os

import numpy as np

from prysm.x import materials
from tests.x.raytracing.surface_helpers import (
    plane, sphere, conic, off_axis_conic, even_asphere, q2d, zernike, xy,
    chebyshev, jacobi, toroid, biconic,
)

from prysm.x.raytracing.surfaces import Surface
from prysm.x.raytracing.spencer_and_murty import newton_raphson_solve_s

_SNAP_DIR = os.path.join(os.path.dirname(__file__), '_snapshots')


def _ref(name):
    return np.load(os.path.join(_SNAP_DIR, name))


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
                                np.array([0.0, 0.0, 0.0]), material=materials.ConstantMaterial(1.5))


def test_bare_newton_snapshot():
    """Cold-start Newton from s1=0 on an EvenAsphere sag_and_normal.  Exercises the
    'all rays start active' code path through `newton_raphson_solve_s`.
    """
    P, S = _ray_batch()
    surf = _asphere()
    Z0 = P[..., 2]
    m = S[..., 2]
    P1 = P + (-Z0 / m)[:, None] * S

    Pj, r, valid = newton_raphson_solve_s(P1, S, surf.sag_and_normal,
                                          s1=0.0)

    assert int(valid.sum()) == 289
    assert np.allclose(Pj, _ref('bare_newton_Pj.npy'), rtol=0, atol=1e-10)
    assert np.allclose(r, _ref('bare_newton_r.npy'), rtol=0, atol=1e-10)


def test_conic_seeded_newton_snapshot():
    """EvenAsphere.intersect — conic-seeded Newton.  Per-ray s1 array."""
    P, S = _ray_batch()
    surf = _asphere()

    Pj, r, valid = surf.intersect(P, S)

    assert int(valid.sum()) == 289
    assert np.allclose(Pj, _ref('conic_seeded_Pj.npy'), rtol=0, atol=1e-10)
    assert np.allclose(r, _ref('conic_seeded_r.npy'), rtol=0, atol=1e-10)


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

    Pj, r, valid = newton_raphson_solve_s(P1, S, surf.sag_and_normal,
                                          s1=0.0, maxiter=2)

    # snapshot: 1 ray converges in 2 iters, 288 hit maxiter
    assert int(valid.sum()) == 1
    assert int(np.isnan(Pj[:, 0]).sum()) == 288
    # the one survivor's coordinates are pinned
    assert np.allclose(Pj[valid], _ref('partial_survivor_Pj.npy'), rtol=0, atol=1e-10)


def test_oblique_mixed_convergence_snapshot():
    """Tiny 2-ray batch where one ray hits cleanly and one diverges.
    Locks down the "some-converge-some-don't" branch in one place.
    """
    surf = _asphere()
    P = np.array([[0.0, 0.0, -50.0], [3.0, 3.0, -50.0]], dtype=np.float64)
    S = np.array([[0.0, 0.0, 1.0], [0.9, 0.0, np.sqrt(1 - 0.81)]], dtype=np.float64)
    P1 = P + (-P[:, 2] / S[:, 2])[:, None] * S

    Pj, r, valid = newton_raphson_solve_s(P1, S, surf.sag_and_normal,
                                          s1=0.0, maxiter=5)
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

    Pj, r, valid = newton_raphson_solve_s(P1, S, surf.sag_and_normal,
                                          s1=0.0, maxiter=10)
    assert valid.tolist() == [True]
    np.testing.assert_array_equal(Pj, np.array([[0.0, 0.0, 0.0]]))
    np.testing.assert_array_equal(r, np.array([[0.0, 0.0, 1.0]]))
