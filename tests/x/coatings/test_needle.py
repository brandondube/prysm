"""Tests for needle optimization (Phase 3).

P(z) is checked against the finite difference of actually inserting a tiny layer;
the needle function vanishes (to tolerance) at a true optimum; synthesize grows a
one/two-layer start into a broadband AR; and the prune/merge housekeeping behaves.
"""
import numpy as np
import pytest

from prysm.x.coatings import Stack, RTA
from prysm.x.coatings.merit import Reflectance, MeritFunction
from prysm.x.coatings.needle import (
    needle_function, insert_needle, cleanup, synthesize,
)

SUB = 1.52


def _broadband_merit(npts=9):
    return MeritFunction([Reflectance(np.linspace(0.45, 0.65, npts),
                                      pol='s', target=0.0)])


# ----------------------------------------------------------- P(z) vs FD

@pytest.mark.parametrize('material', [2.05, 1.9])
@pytest.mark.parametrize('z', [0.05, 0.10, 0.22, 0.28])
def test_needle_function_matches_fd(material, z):
    stack = Stack([1.46, 2.2, 1.46], [0.10, 0.07, 0.12], SUB)
    merit = _broadband_merit()
    P = float(needle_function(stack, merit, material, z)[0])

    dn = 1e-7
    base = merit.value(stack)
    inserted = insert_needle(stack, z, material, thickness=dn)
    fd = (merit.value(inserted) - base) / dn
    assert np.isclose(P, fd, rtol=3e-3, atol=1e-6)


def test_needle_function_host_material_equals_thickness_gradient():
    # inserting the host's own material at depth z just thickens that layer, so
    # P(z) there equals the analytic thickness gradient of the containing layer.
    stack = Stack([1.46, 2.2, 1.46], [0.10, 0.07, 0.12], SUB)
    merit = _broadband_merit()
    _, grad = merit.value_and_grad(stack)
    for k, (z_mid, mat) in enumerate([(0.05, 1.46), (0.135, 2.2), (0.23, 1.46)]):
        P = float(needle_function(stack, merit, mat, z_mid)[0])
        assert P == pytest.approx(grad[k], rel=1e-9)


# ----------------------------------------------------------- stationarity

@pytest.mark.parametrize('material', [1.38, 2.05])
def test_stationarity_at_optimum(material):
    # the exact single-layer QWOT AR is a global optimum (R == 0 at lambda0), so
    # no infinitesimal needle of any material can lower the merit: P(z) >= -tol.
    n_ar = np.sqrt(SUB)
    wvl = 0.55
    ar = Stack([n_ar], [wvl / (4 * n_ar)], SUB)
    merit = MeritFunction([Reflectance(wvl, pol='s', target=0.0)])
    z = np.linspace(0, float(np.sum(np.asarray(ar.thicknesses))), 200)
    P = needle_function(ar, merit, material, z)
    assert P.min() >= -1e-9


# ----------------------------------------------------------- synthesize

def test_synthesize_grows_broadband_ar():
    merit = _broadband_merit(npts=7)
    materials = [1.38, 2.05]
    start = Stack([1.38, 2.05], [0.10, 0.10], SUB)
    start_merit = merit.value(start)

    result = synthesize(start, merit, materials, z_samples=120, max_iters=8,
                        max_layers=16)
    # the layer count grew and the merit dropped by orders of magnitude
    assert result.n_layers > len(start)
    assert result.merit < start_merit / 100
    wvls = np.linspace(0.45, 0.65, 7)
    R, _, _ = RTA(result.stack, wvls, 0.0, 's')
    assert np.max(R) < 5e-3


# ----------------------------------------------------------- prune / merge

def test_cleanup_prunes_thin_layer_and_merges_neighbours():
    # a vanishing needle between two equal-index hosts: drop it, merge the hosts.
    stack = Stack([1.4, 2.0, 1.4], [0.10, 5e-4, 0.15], SUB)
    cleaned = cleanup(stack, prune_tol=2e-3)
    assert len(cleaned) == 1
    assert cleaned.thicknesses[0] == pytest.approx(0.25)
    assert cleaned.indices[0] == pytest.approx(1.4)


def test_cleanup_merges_adjacent_same_index():
    stack = Stack([1.4, 1.4, 2.0], [0.10, 0.20, 0.05], SUB)
    cleaned = cleanup(stack)
    assert len(cleaned) == 2
    assert cleaned.thicknesses[0] == pytest.approx(0.30)
    assert cleaned.thicknesses[1] == pytest.approx(0.05)


def test_cleanup_keeps_distinct_materials():
    stack = Stack([1.4, 2.0, 1.4], [0.10, 0.08, 0.12], SUB)
    cleaned = cleanup(stack)
    assert len(cleaned) == 3


# ----------------------------------------------------------- insert geometry

def test_insert_needle_splits_host_preserving_thickness():
    stack = Stack([1.46, 2.2], [0.10, 0.20], SUB)
    inserted = insert_needle(stack, 0.04, 1.9, thickness=0.001)
    # host layer 0 split into 0.04 | needle | 0.06; layer 1 untouched
    assert len(inserted) == 4
    assert list(inserted.indices) == [1.46, 1.9, 1.46, 2.2]
    th = np.asarray(inserted.thicknesses)
    assert th[0] == pytest.approx(0.04)
    assert th[1] == pytest.approx(0.001)
    assert th[2] == pytest.approx(0.06)
    assert th[3] == pytest.approx(0.20)
