"""differential kernel tangent state vs central FD of raytrace.

Validates raytrace_with_tangents' propagated (Pdot_last, Sdot_last, Ldot) on a
multi-surface refracting system against central finite differences of the real
spencer_and_murty.raytrace, for every supported perturbation type:
curvature, conic, despace, thickness (downstream fan-out), decenter, tilt, and
index.  Also exercises the d_* primitives end to end and the all-seeds-at-once
trailing parameter axis.
"""
import numpy as np
import pytest

from prysm.x import materials
from prysm.x.raytracing.spencer_and_murty import raytrace, valid_mask
from prysm.x.raytracing._diff_raytrace import (
    raytrace_with_tangents,
    seed_curvature,
    seed_conic,
    seed_decenter,
    seed_despace,
    seed_tilt,
    seed_index,
)
from prysm.x.raytracing.phase import PhaseFunction, LinearGrating
from prysm.x.raytracing.adjoint.backward_sweep import adjoint_gradient
from tests.x.raytracing.surface_helpers import conic, plane, even_asphere


# ---------- system + bundle -------------------------------------------------

NG = 1.62
BASE = dict(c0=1 / 40.0, k0=-0.6, c1=-1 / 55.0, k1=0.2,
            z0=0.0, z1=6.0, zimg=56.0, x1=0.0, y1=0.0, tiltx1=0.0, ng=NG)


def make_system(**over):
    p = dict(BASE, **over)
    n_glass = materials.ConstantMaterial(p['ng'])
    s0 = conic(c=p['c0'], k=p['k0'], interaction='refr', P=[0, 0, p['z0']], material=n_glass)
    kw1 = {}
    if p['tiltx1'] != 0.0:
        kw1 = dict(tilt=(0.0, 0.0, p['tiltx1']), tilt_radians=True)
    s1 = conic(c=p['c1'], k=p['k1'], interaction='refr',
               P=[p['x1'], p['y1'], p['z1']], material=materials.air, **kw1)
    img = plane(interaction='eval', P=[0, 0, p['zimg']])
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


WVL = 0.55


def fd_state(over_plus, over_minus, P, S, h):
    """Central FD of (P_last, S_last, OPL_total) over a parameter override."""
    def state(over):
        tr = raytrace(make_system(**over), P, S, WVL)
        return tr.P[-1], tr.S[-1], tr.OPL.sum(axis=0)
    Pp, Sp, Lp = state(over_plus)
    Pm, Sm, Lm = state(over_minus)
    return (Pp - Pm) / (2 * h), (Sp - Sm) / (2 * h), (Lp - Lm) / (2 * h)


def check(seed, over_plus, over_minus, h, rtol=1e-6, atol_P=1e-7,
          atol_S=1e-9, atol_L=1e-7):
    P, S = ray_bundle()
    res = raytrace_with_tangents(make_system(), P, S, WVL, [seed])
    dP = res.Pdot[-1][:, :, 0]
    dS = res.Sdot[-1][:, :, 0]
    dL = res.Ldot.sum(axis=0)[:, 0]
    dP_fd, dS_fd, dL_fd = fd_state(over_plus, over_minus, P, S, h)
    np.testing.assert_allclose(dP, dP_fd, rtol=rtol, atol=atol_P)
    np.testing.assert_allclose(dS, dS_fd, rtol=rtol, atol=atol_S)
    np.testing.assert_allclose(dL, dL_fd, rtol=rtol, atol=atol_L)


# ---------- per-perturbation validation -------------------------------------

def test_curvature_surface0():
    h = 1e-6
    check(seed_curvature(0), dict(c0=BASE['c0'] + h), dict(c0=BASE['c0'] - h), h)


def test_curvature_surface1():
    h = 1e-6
    check(seed_curvature(1), dict(c1=BASE['c1'] + h), dict(c1=BASE['c1'] - h), h)


def test_conic_surface0():
    h = 1e-5
    check(seed_conic(0), dict(k0=BASE['k0'] + h), dict(k0=BASE['k0'] - h), h)


def test_conic_surface1():
    h = 1e-5
    check(seed_conic(1), dict(k1=BASE['k1'] + h), dict(k1=BASE['k1'] - h), h)


def test_despace_surface1_only():
    """Despace: surface 1 moves in +z, image plane fixed."""
    h = 1e-6
    check(seed_despace([(1, +1)]),
          dict(z1=BASE['z1'] + h), dict(z1=BASE['z1'] - h), h)


def test_thickness_fan_out():
    """Thickness t(0->1): surface 1 AND the downstream image plane move."""
    h = 1e-6
    check(seed_despace([(1, +1), (2, +1)]),
          dict(z1=BASE['z1'] + h, zimg=BASE['zimg'] + h),
          dict(z1=BASE['z1'] - h, zimg=BASE['zimg'] - h), h)


def test_decenter_surface1_x():
    h = 1e-6
    check(seed_decenter(1, 'x'), dict(x1=h), dict(x1=-h), h)


def test_decenter_surface1_y():
    h = 1e-6
    check(seed_decenter(1, 'y'), dict(y1=h), dict(y1=-h), h)


def test_tilt_surface1_x():
    h = 1e-6
    check(seed_tilt(1, 'x'), dict(tiltx1=h), dict(tiltx1=-h), h,
          rtol=1e-5, atol_P=1e-6, atol_S=1e-8, atol_L=1e-6)


def test_index_glass():
    h = 1e-6
    check(seed_index(0), dict(ng=NG + h), dict(ng=NG - h), h)


# ---------- local-FD fallback on base Shape ---------------------------------

def test_fd_fallback_freeform_curvature():
    """A shape with no analytic Hessian (EvenAsphere) uses the base-Shape FD
    fallback for sag_hessian / sag_param_partials; the propagated tangent must
    still match FD of the kernel.
    """
    P, S = ray_bundle()
    coefs = (1e-7,)  # a4; nonzero so the surface is a genuine asphere
    c0, k0 = 1 / 40.0, -0.6

    def system(c):
        s0 = even_asphere(c=c, k=k0, coefs=coefs, interaction='refr',
                          P=[0, 0, 0], material=materials.ConstantMaterial(NG))
        img = plane(interaction='eval', P=[0, 0, 56.0])
        return [s0, img]

    res = raytrace_with_tangents(system(c0), P, S, WVL, [seed_curvature(0)])
    dP = res.Pdot[-1][:, :, 0]

    h = 1e-6
    trp = raytrace(system(c0 + h), P, S, WVL)
    trm = raytrace(system(c0 - h), P, S, WVL)
    dP_fd = (trp.P[-1] - trm.P[-1]) / (2 * h)
    # FD-of-FD fallback is looser than the analytic path
    np.testing.assert_allclose(dP, dP_fd, rtol=1e-4, atol=1e-5)


# ---------- all seeds at once (trailing parameter axis) ---------------------

def test_all_seeds_simultaneously():
    """One nominal trace, every tolerance's tangent extracted at once."""
    P, S = ray_bundle()
    seeds = [
        seed_curvature(0),
        seed_conic(1),
        seed_despace([(1, +1)]),
        seed_decenter(1, 'y'),
        seed_index(0),
    ]
    res = raytrace_with_tangents(make_system(), P, S, WVL, seeds)
    assert res.n_params == 5

    h = 1e-6
    overs = [
        (dict(c0=BASE['c0'] + h), dict(c0=BASE['c0'] - h)),
        (dict(k1=BASE['k1'] + h), dict(k1=BASE['k1'] - h)),
        (dict(z1=BASE['z1'] + h), dict(z1=BASE['z1'] - h)),
        (dict(y1=h), dict(y1=-h)),
        (dict(ng=NG + h), dict(ng=NG - h)),
    ]
    for p, (op, om) in enumerate(overs):
        dP = res.Pdot[-1][:, :, p]
        dP_fd, _, _ = fd_state(op, om, P, S, h)
        np.testing.assert_allclose(dP, dP_fd, rtol=1e-6, atol=1e-7)


# ---------- diffractive (grating) surfaces in the AD stacks -----------------


class _RadialPhase(PhaseFunction):
    """Nonlinear diffractive phase with nonzero Hessian."""

    def __init__(self, a, bx=0.0):
        self.a = a
        self.bx = bx

    def phase(self, x, y):
        return 0.5 * self.a * (x * x + y * y) + self.bx * x

    def phase_and_gradient(self, x, y):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        return self.phase(x, y), self.a * x + self.bx, self.a * y

    def phase_hessian(self, x, y):
        x = np.asarray(x, float)
        o = np.full(x.shape, self.a)
        z = np.zeros_like(x)
        return o, z, o


def grating_system(phase, *, interaction='refr', **over):
    """make_system with a phase function on surface 0."""
    p = dict(BASE, **over)
    n_glass = materials.ConstantMaterial(p['ng'])
    mat = n_glass if interaction == 'refr' else None
    s0 = conic(c=p['c0'], k=p['k0'], interaction=interaction,
               P=[0, 0, p['z0']], material=mat)
    s0.grating = phase
    s1 = conic(c=p['c1'], k=p['k1'], interaction='refr',
               P=[p['x1'], p['y1'], p['z1']], material=materials.air)
    img = plane(interaction='eval', P=[0, 0, p['zimg']])
    return [s0, s1, img]


def grating_fd(phase, over_plus, over_minus, P, S, h, **mk):
    def state(over):
        tr = raytrace(grating_system(phase, **mk, **over), P, S, WVL)
        return tr.P[-1], tr.S[-1], tr.OPL.sum(axis=0)
    Pp, Sp, Lp = state(over_plus)
    Pm, Sm, Lm = state(over_minus)
    return (Pp - Pm) / (2 * h), (Sp - Sm) / (2 * h), (Lp - Lm) / (2 * h)


# linear and radial phase fixtures
_LINEAR = LinearGrating(5.0, [1.0, 0.0], 1)
_RADIAL = _RadialPhase(2e-4, bx=3e-4)


@pytest.mark.parametrize('phase', [_LINEAR, _RADIAL])
@pytest.mark.parametrize('interaction', ['refr', 'refl'])
def test_grating_forward_tangents_match_fd(phase, interaction):
    """Diffractive forward tangents match central finite differences."""
    P, S = ray_bundle()
    h = 1e-6
    cases = [
        (seed_curvature(0), dict(c0=BASE['c0'] + h), dict(c0=BASE['c0'] - h)),
        (seed_decenter(1, 'y'), dict(y1=h), dict(y1=-h)),
        (seed_index(0), dict(ng=NG + h), dict(ng=NG - h)),
    ]
    for seed, op, om in cases:
        res = raytrace_with_tangents(
            grating_system(phase, interaction=interaction), P, S, WVL, [seed])
        dP = res.Pdot[-1][:, :, 0]
        dS = res.Sdot[-1][:, :, 0]
        dL = res.Ldot.sum(axis=0)[:, 0]
        dPf, dSf, dLf = grating_fd(phase, op, om, P, S, h,
                                   interaction=interaction)
        np.testing.assert_allclose(dP, dPf, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(dS, dSf, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(dL, dLf, rtol=1e-6, atol=1e-6)


class _SumComponentHead:
    """Scalar merit summing one final ray quantity over valid rays."""

    def __init__(self, component):
        self.component = component
        self.name = component

    def seed(self, trace, prescription, wavelength):
        v = valid_mask(trace.status, trace.P[-1])
        P_bar = np.zeros_like(trace.P[-1])
        S_bar = np.zeros_like(trace.S[-1])
        L_bar = np.zeros(trace.P[-1].shape[0], dtype=trace.P[-1].dtype)
        if self.component == 'Py':
            P_bar[v, 1] = 1.0
        else:
            L_bar[v] = 1.0
        return P_bar, S_bar, L_bar


@pytest.mark.parametrize('component', ['Py', 'OPL'])
def test_grating_adjoint_matches_forward(component):
    """Diffractive adjoint gradients match the forward JVP."""
    P, S = ray_bundle()
    seeds = [seed_curvature(0), seed_conic(1), seed_decenter(1, 'y'),
             seed_index(0), seed_despace([(1, +1), (2, +1)])]
    res = raytrace_with_tangents(grating_system(_RADIAL), P, S, WVL, seeds)
    tr = raytrace(grating_system(_RADIAL), P, S, WVL)
    v = valid_mask(tr.status, tr.P[-1])
    if component == 'Py':
        fwd = np.nansum(np.where(v[:, None], res.Pdot[-1][:, 1, :], 0.0), axis=0)
    else:
        fwd = np.nansum(np.where(v[:, None], res.Ldot.sum(axis=0), 0.0), axis=0)
    g = adjoint_gradient(grating_system(_RADIAL), P, S, WVL, seeds,
                         _SumComponentHead(component))
    np.testing.assert_allclose(g, fwd, rtol=1e-9, atol=1e-10)
