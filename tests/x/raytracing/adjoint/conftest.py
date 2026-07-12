"""Shared fixtures: a refracting doublet-like system and a pupil ray bundle.

Mirrors the system used by prysm's forward-mode diff-raytrace tests so the
adjoint Jacobian can be checked against the validated forward tangents.
"""
import pytest

from tests.x.raytracing.differential_helpers import (
    BASE, WVL, make_system, ray_bundle,
)

__all__ = ['BASE', 'WVL', 'make_system', 'ray_bundle']


@pytest.fixture
def system():
    return make_system()


@pytest.fixture
def bundle():
    return ray_bundle()
