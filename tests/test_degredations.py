"""Tests for image-chain degradation models."""

import pytest

import numpy as np

from prysm import degredations
from prysm.conf import config


def test_smear_ft_width_only_matches_sinc_x():
    fx = np.asarray([-0.5, 0, 0.5])
    fy = np.asarray([-0.25, 0, 0.25])

    out = degredations.smear_ft(fx, fy, width=2, height=0)

    np.testing.assert_allclose(out, np.sinc(fx * 2))
    assert out.dtype == config.precision


def test_smear_ft_height_only_matches_sinc_y():
    fx = np.asarray([-0.5, 0, 0.5])
    fy = np.asarray([-0.25, 0, 0.25])

    out = degredations.smear_ft(fx, fy, width=0, height=4)

    np.testing.assert_allclose(out, np.sinc(fy * 4))
    assert out.dtype == config.precision


def test_smear_ft_requires_nonzero_extent():
    with pytest.raises(AssertionError, match='one of width or height must be nonzero'):
        degredations.smear_ft(np.asarray([0]), np.asarray([0]), width=0, height=0)


def test_jitter_ft_zero_scale_is_unity():
    fr = np.asarray([0, 0.25, 0.5, 1])

    out = degredations.jitter_ft(fr, scale=0)

    np.testing.assert_allclose(out, np.ones_like(fr))
