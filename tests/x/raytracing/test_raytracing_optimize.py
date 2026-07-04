"""High-level optimization ergonomics: build_problem fan-out, constraint
routing, lazy recipe re-launch, and the OpticalSystem.problem/.optimize sugar.
"""
import warnings

import numpy as np
import pytest

from prysm.x import materials
from prysm.x.raytracing import LensData, OpticalSystem
from prysm.x.raytracing.surfaces import Conic, Plane
from prysm.x.raytracing.launch import Field, Sampling
from prysm.x.raytracing.paraxial import effective_focal_length
from prysm.x.raytracing.design import (
    EFL, Merit, Problem, RmsSpotRadius, Thickness, TotalTrack, WavefrontRMS,
    build_problem, _CallableMerit, _TraceCache,
)


_glass = materials.ConstantMaterial(1.5168, name='N-BK7')


def make_singlet(image_gap=96.0, fields=(0,), wavelengths=(0.55,),
                 weights=None, stop_index=None):
    # conics are rows 1 and 2
    lens = LensData()
    (lens.add(Conic(1 / 102.0, 0.0), thickness=6.0, material=_glass,
              aperture=12.0)
         .add(Conic(-1 / 102.0, 0.0), thickness=image_gap,
              material=materials.air, aperture=12.0))
    return OpticalSystem(lens, aperture=20.0, fields=list(fields),
                         wavelengths=list(wavelengths), weights=weights,
                         stop_index=stop_index)


# ---------- the motivating example -------------------------------------------

def test_optimize_spot_focuses_singlet():
    """sys.opt.optimize('spot') with a free back gap collapses the spot."""
    sys_ = make_singlet(image_gap=80.0)
    sys_.opt.vary('thickness', surfaces=2)     # the back gap (conic2 row)
    prob = sys_.opt.problem('spot')
    spot0 = abs(prob.residuals(prob.x0())[0])
    res = sys_.opt.optimize('spot', maxiter=20)
    spot1 = abs(prob.residuals(res.x)[0])
    assert spot1 < 0.1 * spot0
    # the solve scattered the result into the lens in place
    np.testing.assert_allclose(sys_.lens.rows[2].thickness, res.x[0])


def test_constrained_optimize_hits_efl():
    """spot objective + EFL equality lands on the constraint."""
    sys_ = make_singlet()
    sys_.opt.vary('curvature', surfaces=1).vary('thickness', surfaces=2)
    with warnings.catch_warnings():
        # the line search exhausts at the constrained spot floor; the lens
        # still lands on the best iterate (and the constraint)
        warnings.simplefilter('ignore', UserWarning)
        sys_.opt.optimize('spot', constraints=[EFL(target=100.0)],
                      maxiter=30, damping=1e-8)
    assert effective_focal_length(sys_.to_surfaces(), wvl=0.55) == pytest.approx(
        100.0, rel=1e-5)


def test_problem_returns_inspectable_extendable_problem():
    sys_ = make_singlet()
    prob = sys_.opt.problem('spot')
    assert isinstance(prob, Problem)
    n = len(prob.operands)
    prob.operands.append(EFL(target=100.0, weight=0.1))
    assert len(prob.operands) == n + 1


# ---------- fan-out -----------------------------------------------------------

def test_fanout_counts_and_weights():
    sys_ = make_singlet(fields=(0.0, 1.0), wavelengths=(0.48, 0.55, 0.65),
                        weights=(1.0, 2.0, 1.0))
    prob = sys_.opt.problem('spot')
    assert len(prob.operands) == 6  # 2 fields x 3 wavelengths
    assert all(isinstance(op, RmsSpotRadius) for op in prob.operands)
    assert [op.weight for op in prob.operands] == [1.0, 2.0, 1.0, 1.0, 2.0, 1.0]
    assert [op.wavelength for op in prob.operands] == [0.48, 0.55, 0.65] * 2


def test_fanout_explicit_wavelengths_weight_uniformly():
    sys_ = make_singlet(wavelengths=(0.48, 0.55, 0.65), weights=(1., 2., 1.))
    prob = sys_.opt.problem('spot', wavelengths=[0.5])
    assert [op.wavelength for op in prob.operands] == [0.5]
    assert [op.weight for op in prob.operands] == [1.0]


def test_scalar_merit_class_fans_out_over_wavelengths_only():
    sys_ = make_singlet(fields=(0.0, 1.0), wavelengths=(0.48, 0.65),
                        weights=(1.0, 2.0))
    prob = sys_.opt.problem(EFL)
    assert len(prob.operands) == 2
    assert all(isinstance(op, EFL) for op in prob.operands)
    assert [op.wavelength for op in prob.operands] == [0.48, 0.65]
    assert [op.weight for op in prob.operands] == [1.0, 2.0]


class _FanMerit(Merit):
    """Custom merit accepting the fan-out recipe kwargs."""

    def __init__(self, field=None, wavelength=None, sampling=None, *,
                 weight=1.0):
        super().__init__(weight=weight)
        self.field = field
        self.wavelength = wavelength
        self.sampling = sampling

    def __call__(self, prescription, cache):
        return 0.0


def test_fanout_mixes_string_subclass_instance_and_callable():
    sys_ = make_singlet(fields=(0.0, 1.0), wavelengths=(0.48, 0.65))

    def my_merit(prescription, cache):
        return 1.0

    inst = EFL(target=100.0)
    prob = sys_.opt.problem(['spot', _FanMerit, inst, my_merit])
    # 4 + 4 fanned out, instance and callable pass through singly
    assert len(prob.operands) == 10
    fanned = [op for op in prob.operands if isinstance(op, _FanMerit)]
    assert len(fanned) == 4
    assert prob.operands[8] is inst
    wrapped = prob.operands[9]
    assert isinstance(wrapped, _CallableMerit)
    assert wrapped.name == 'my_merit'
    assert wrapped(sys_, _TraceCache(sys_)) == 1.0


def test_fanout_rejects_unknown_goal_and_bad_type():
    sys_ = make_singlet()
    with pytest.raises(ValueError, match='unknown goal'):
        sys_.opt.problem('sharpness')
    with pytest.raises(TypeError, match='goal items'):
        build_problem(sys_, 3.14)


# ---------- constraint routing ------------------------------------------------

def test_constraint_routing_targets_and_bounds():
    sys_ = make_singlet()
    prob = sys_.opt.problem('spot', constraints=[
        EFL(target=100.0),
        TotalTrack(max=110.0),
        Thickness(2, min=5.0, max=120.0),
    ])
    assert len(prob.equality_constraints) == 1
    assert isinstance(prob.equality_constraints[0], EFL)
    # min and max both set makes two inequality rows
    assert len(prob.inequality_constraints) == 3
    x0 = prob.x0()
    eq = prob.equalities(x0)
    ineq = prob.inequalities(x0)
    cache = _TraceCache(sys_)
    efl = EFL()(sys_, cache)
    ttl = TotalTrack()(sys_, cache)
    thk = Thickness(2)(sys_, cache)
    np.testing.assert_allclose(eq, [efl - 100.0])
    np.testing.assert_allclose(ineq, [110.0 - ttl, thk - 5.0, 120.0 - thk])


def test_constraint_target_with_bound_raises():
    sys_ = make_singlet()
    with pytest.raises(ValueError, match='mixes'):
        Problem(sys_, [], constraints=[EFL(target=100.0, min=90.0)])


# ---------- geometry operands ---------------------------------------------------

def test_total_track_and_thickness_match_hand_sums():
    sys_ = make_singlet(image_gap=96.0)
    cache = _TraceCache(sys_)
    assert TotalTrack()(sys_, cache) == pytest.approx(6.0 + 96.0)
    assert Thickness(2)(sys_, cache) == pytest.approx(96.0)   # conic2 back gap


def test_total_track_skips_object_row():
    # the OBJECT endpoint (row 0) carries the object distance, not track
    lens = LensData()
    (lens.add(Conic(1 / 102.0, 0.0), thickness=6.0, material=_glass)
         .add(Conic(-1 / 102.0, 0.0), thickness=96.0, material=materials.air))
    sys_ = OpticalSystem(lens, aperture=20.0, wavelengths=[0.55])
    assert TotalTrack()(sys_, _TraceCache(sys_)) == pytest.approx(102.0)
    # a finite object gap is object distance, not track, and is skipped too
    lens.rows[0].thickness = 50.0
    assert TotalTrack()(sys_, _TraceCache(sys_)) == pytest.approx(102.0)


# ---------- recipe defaults -----------------------------------------------------

def test_recipe_none_defaults_resolve_on_system():
    """RmsSpotRadius() means on-axis, reference wavelength, hex(4)."""
    sys_ = make_singlet(fields=(1.0,), wavelengths=(0.48, 0.55), weights=None)
    sys_.reference = 1
    bare = RmsSpotRadius()(sys_, _TraceCache(sys_))
    explicit = RmsSpotRadius(Field(0., 0.), 0.55, Sampling.hex(nrings=4))(
        sys_, _TraceCache(sys_))
    np.testing.assert_allclose(bare, explicit)


def test_recipe_wavelength_none_raises_on_bare_lensdata():
    sys_ = make_singlet()
    op = RmsSpotRadius()
    with pytest.raises(ValueError, match='wavelength'):
        op(sys_.lens, _TraceCache(sys_.lens))


# ---------- lazy re-launch ------------------------------------------------------

def test_bundle_relaunches_as_the_design_moves():
    """The launched bundle at iterate x differs from the one at x0: with the
    stop on the rear surface, the entrance pupil moves with the front
    curvature, and the off-axis bundle re-aims through it."""
    sys_ = make_singlet(fields=(3.0,), stop_index=2)   # stop on the rear conic
    sys_.opt.vary('curvature', surfaces=1)            # front conic curvature
    prob = sys_.opt.problem('spot')
    x0 = prob.x0()
    _, c0 = prob.residuals(x0, return_cache=True)
    _, c1 = prob.residuals(x0 + 0.002, return_cache=True)
    (P0, _), = c0._launch_cache.values()
    (P1, _), = c1._launch_cache.values()
    assert not np.allclose(P0, P1)
    sys_.opt.update(x0)
