"""Shared fixtures: a refracting doublet-like system and a pupil ray bundle.

Mirrors the system used by prysm's forward-mode diff-raytrace tests so the
adjoint Jacobian can be checked against the validated forward tangents.
"""
import numpy as np
import pytest

from prysm.x import materials
from tests.x.raytracing.surface_helpers import conic, plane

NG = 1.62
WVL = 0.55

BASE = dict(c0=1 / 40.0, k0=-0.6, c1=-1 / 55.0, k1=0.2,
            z0=0.0, z1=6.0, zimg=56.0, x1=0.0, y1=0.0, ng=NG)


def make_system(**over):
    p = dict(BASE, **over)
    n_glass = materials.ConstantMaterial(p['ng'])
    s0 = conic(c=p['c0'], k=p['k0'], interaction='refr',
               P=[0, 0, p['z0']], material=n_glass)
    s1 = conic(c=p['c1'], k=p['k1'], interaction='refr',
               P=[p['x1'], p['y1'], p['z1']], material=materials.air)
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


@pytest.fixture
def system():
    return make_system()


@pytest.fixture
def bundle():
    return ray_bundle()
