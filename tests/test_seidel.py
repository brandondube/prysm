''' Unit tests for Seidel objects.
'''
import pytest

import numpy as np

from prysm import Seidel


@pytest.mark.parametrize('aberrations', [
    {'W020': 1},  # single field constant aberration
    {'W131': 1},  # single field dependent aberration
    {'W040': 1, 'W222': 1}])  # two aberrations
def test_seidel_produces_nonzero_opd(aberrations):
    p = Seidel(**aberrations)
    z = np.zeros(p.phase.shape)
    assert not np.allclose(p.phase, z)


def test_seidel_respects_field_dependence():
    aberrations = {
        'W171': 1
    }
    p = Seidel(**aberrations, field=0)
    phase = p.phase[np.isfinite(p.phase)]
    z = np.zeros(phase.shape)
    assert np.allclose(phase, z)


def test_seidel_repr():
    p = Seidel()
    assert type(repr(p)) is str
