"""Tests for coating merit term input shape handling."""
import numpy as np
import pytest

from prysm.x.coatings.merit import Reflectance


def test_merit_rejects_ambiguous_1d_wavelength_angle_grid():
    wvl = np.linspace(0.45, 0.65, 4)
    theta = np.linspace(0.0, 0.2, 3)
    with pytest.raises(ValueError, match='both 1-D'):
        Reflectance(wvl, theta=theta, target=0.0)


def test_merit_rejects_nonbroadcast_target():
    wvl = np.linspace(0.45, 0.65, 4)[:, None]
    theta = np.linspace(0.0, 0.2, 3)[None, :]
    target = np.zeros(4)
    with pytest.raises(ValueError, match='broadcast-compatible'):
        Reflectance(wvl, theta=theta, target=target)
