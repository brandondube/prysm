"""Tests for the surfaces= selector forms and constrain() box bounds
(absolute + relative, radius-vs-curvature quantity, and edge cases)."""

import numpy as np
import pytest

from prysm.x.raytracing import LensData
from prysm.x import materials
from prysm.x.raytracing.surfaces import Conic, EvenAsphere, Plane


_air = materials.air


def make_triplet():
    ld = LensData()
    radii = [50.0, -80.0, 40.0, -40.0, 80.0, -50.0]
    for i, r in enumerate(radii):
        ld.add(Conic(1.0 / r, 0.0), thickness=3.0 + i,
               material=_air, semidiameter=8.0)
    ld.add(Plane(), typ='eval', material=_air, semidiameter=8.0)
    return ld


# ---------------------------------------------------------------------------
# surfaces= selector forms
# ---------------------------------------------------------------------------

def test_selector_int():
    ld = make_triplet()
    ld.vary('curvature', surfaces=2)
    assert len(ld.pack()) == 1


def test_selector_list():
    ld = make_triplet()
    ld.vary('curvature', surfaces=[0, 2, 4])
    assert len(ld.pack()) == 3


def test_selector_slice():
    ld = make_triplet()
    ld.vary('curvature', surfaces=slice(1, 4))
    assert len(ld.pack()) == 3


def test_selector_all_skips_planes():
    ld = make_triplet()
    ld.vary('curvature', surfaces='all')
    # 6 conics have curvature; the eval plane does not
    assert len(ld.pack()) == 6


def test_selector_negative_index():
    ld = make_triplet()
    ld.vary('curvature', surfaces=-2)  # last conic
    np.testing.assert_allclose(ld.pack(), [1.0 / -50.0])


# ---------------------------------------------------------------------------
# absolute bounds
# ---------------------------------------------------------------------------

def test_absolute_thickness_bounds():
    ld = make_triplet()
    ld.vary('thickness', surfaces=[0, 1])
    ld.constrain('thickness', lo=0.0, hi=20.0, surfaces=[0, 1])
    lo, hi = ld.bounds()
    np.testing.assert_allclose(lo, [0.0, 0.0])
    np.testing.assert_allclose(hi, [20.0, 20.0])


def test_absolute_one_sided_bound_leaves_other_infinite():
    ld = make_triplet()
    ld.vary('thickness', surfaces=0)
    ld.constrain('thickness', lo=0.0, surfaces=0)
    lo, hi = ld.bounds()
    assert lo[0] == 0.0
    assert np.isinf(hi[0])


def test_unconstrained_free_dof_is_infinite():
    ld = make_triplet()
    ld.vary('curvature', surfaces=0)
    lo, hi = ld.bounds()
    assert np.isinf(lo[0]) and lo[0] < 0
    assert np.isinf(hi[0])


# ---------------------------------------------------------------------------
# relative bounds + radius-vs-curvature quantity
# ---------------------------------------------------------------------------

def test_relative_curvature_bound_is_pct_of_curvature():
    ld = make_triplet()
    ld.vary('curvature', surfaces=0)
    ld.constrain('curvature', relative=0.1, surfaces=0)
    c0 = 1.0 / 50.0
    lo, hi = ld.bounds()
    np.testing.assert_allclose([lo[0], hi[0]], [c0 * 0.9, c0 * 1.1])


def test_relative_radius_bound_is_pct_of_radius():
    ld = make_triplet()
    ld.vary('radius', surfaces=0)
    ld.constrain('radius', relative=0.1, surfaces=0)
    lo, hi = ld.bounds()
    # the bound is +/-10% of the RADIUS (50), converted back to curvature
    np.testing.assert_allclose(1.0 / hi[0], 50.0 * 0.9)
    np.testing.assert_allclose(1.0 / lo[0], 50.0 * 1.1)


def test_relative_radius_bound_orders_negative_nominal():
    ld = make_triplet()
    ld.vary('radius', surfaces=1)  # radius -80
    ld.constrain('radius', relative=0.1, surfaces=1)
    lo, hi = ld.bounds()
    assert lo[0] < hi[0]
    # radius window [-88, -72]
    radii = sorted([1.0 / lo[0], 1.0 / hi[0]])
    np.testing.assert_allclose(radii, [-88.0, -72.0])


def test_relative_bound_on_zero_curvature_is_unbounded_with_warning():
    ld = LensData().add(Conic(0.0, 0.0), thickness=1.0, material=_air,
                        semidiameter=5.0)
    ld.vary('curvature', surfaces=0)
    with pytest.warns(UserWarning):
        ld.constrain('curvature', relative=0.1, surfaces=0)
    lo, hi = ld.bounds()
    assert np.isinf(lo[0]) and np.isinf(hi[0])


def test_relative_radius_bound_on_flat_surface_is_unbounded_with_warning():
    ld = LensData().add(Conic(0.0, 0.0), thickness=1.0, material=_air,
                        semidiameter=5.0)
    ld.vary('radius', surfaces=0)
    with pytest.warns(UserWarning):
        ld.constrain('radius', relative=0.1, surfaces=0)
    lo, hi = ld.bounds()
    assert np.isinf(lo[0]) and np.isinf(hi[0])


# ---------------------------------------------------------------------------
# misc
# ---------------------------------------------------------------------------

def test_constrain_missing_category_is_silent_no_op():
    ld = make_triplet()
    ld.vary('curvature', surfaces='all')
    # the eval plane (last surface) has no curvature; constraining 'radius'
    # over 'all' must apply to the six conics and silently skip the plane
    ld.constrain('radius', relative=0.1, surfaces='all')
    assert len(ld.bounds()[0]) == 6


def test_constrain_requires_a_bound_spec():
    ld = make_triplet()
    with pytest.raises(ValueError):
        ld.constrain('thickness', surfaces=0)


def test_bounds_only_returned_for_free_slots():
    ld = make_triplet()
    ld.constrain('thickness', lo=0.0, hi=10.0, surfaces='all')  # set, not freed
    ld.vary('thickness', surfaces=2)
    lo, hi = ld.bounds()
    assert len(lo) == 1  # only the one freed thickness
    assert lo[0] == 0.0 and hi[0] == 10.0


def test_coefs_relative_bound_per_coefficient():
    coefs = (1e-4, -2e-6, 3e-9)
    ld = LensData().add(EvenAsphere(1 / 50.0, 0.0, coefs), thickness=2.0,
                        material=_air, semidiameter=8.0)
    ld.vary('coefs', surfaces=0)
    ld.constrain('coefs', relative=0.5, surfaces=0)
    lo, hi = ld.bounds()
    assert len(lo) == 3
    for c, l, h in zip(coefs, lo, hi):
        window = sorted([c * 0.5, c * 1.5])
        np.testing.assert_allclose([l, h], window)
