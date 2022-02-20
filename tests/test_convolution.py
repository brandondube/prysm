"""Tests for convolution routines."""
from functools import partial

import pytest

import numpy as np

from prysm import convolution, degredations

def test_conv_functions():
    a = np.random.rand(100, 100)
    b = np.random.rand(100, 100)
    c = convolution.conv(a, b)
    assert c.shape == a.shape
    assert c.dtype == a.dtype


def test_apply_tf_functions():
    sm = partial(degredations.smear_ft, width=1, angle=123)
    ji = partial(degredations.jitter_ft, scale=1)
    a = np.random.rand(100, 100)
    aprime = convolution.apply_transfer_functions(a, 1, [sm, ji])
    assert aprime.shape == a.shape
