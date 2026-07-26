"""Shared systems and ray bundles for differential raytracing tests."""

import numpy as np

from prysm.x import materials
from tests.x.raytracing.surface_helpers import conic, plane


NG = 1.62
WVL = 0.55
BASE = dict(
    c0=1 / 40.0,
    k0=-0.6,
    c1=-1 / 55.0,
    k1=0.2,
    z0=0.0,
    z1=6.0,
    zimg=56.0,
    x1=0.0,
    y1=0.0,
    tiltx1=0.0,
    ng=NG,
)


def make_system(**overrides):
    """Build the common two-surface differential-test refractor."""
    params = dict(BASE, **overrides)
    glass = materials.ConstantMaterial(params['ng'])
    first = conic(
        c=params['c0'], k=params['k0'], interaction='refr',
        P=[0, 0, params['z0']], material=glass,
    )
    second_kwargs = {}
    if params['tiltx1'] != 0.0:
        second_kwargs = dict(
            tilt=(0.0, 0.0, params['tiltx1']), tilt_radians=True)
    second = conic(
        c=params['c1'], k=params['k1'], interaction='refr',
        P=[params['x1'], params['y1'], params['z1']],
        material=materials.air, **second_kwargs,
    )
    image = plane(interaction='eval', P=[0, 0, params['zimg']])
    return [first, second, image]


def ray_bundle():
    """Return the common tilted 5x5 pupil bundle."""
    ax, ay = 0.04, 0.06
    sx, sy = np.sin(ax), np.sin(ay)
    sz = np.sqrt(1.0 - sx * sx - sy * sy)
    samples = np.linspace(-7, 7, 5)
    xx, yy = np.meshgrid(samples, samples)
    pupil = np.stack([xx.ravel(), yy.ravel()], axis=-1)
    positions = np.empty((pupil.shape[0], 3))
    positions[:, :2] = pupil
    positions[:, 2] = -12.0
    directions = np.broadcast_to(
        np.array([sx, sy, sz]), positions.shape).copy()
    return positions, directions
