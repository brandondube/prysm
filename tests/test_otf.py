"""Optical Transfer Function (OTF) unit tests."""
import pytest

import numpy as np

from prysm import otf
from prysm.fttools import forward_ft_unit

import matplotlib
matplotlib.use('Agg')

SAMPLES = 32
LIM = 1e3


def test_mtf_calc_correct():
    x, y = forward_ft_unit(1/1e3, 128), forward_ft_unit(1/1e3, 128)
    xx, yy = np.meshgrid(x, y)
    dat = np.sin(xx)
    mtf = otf.mtf_from_psf(dat, x[1]-x[0])
    center = tuple(s//2 for s in mtf.shape)
    assert mtf.data[center] == 1


def test_ptf_calc_correct():
    x, y = forward_ft_unit(1/1e3, 128), forward_ft_unit(1/1e3, 128)
    xx, yy = np.meshgrid(x, y)
    dat = np.sin(xx)
    ptf = otf.ptf_from_psf(dat, x[1]-x[0])
    center = tuple(s//2 for s in ptf.shape)
    assert ptf.data[center] == 0


def test_otf_calc_correct():
    x, y = forward_ft_unit(1/1e3, 128), forward_ft_unit(1/1e3, 128)
    xx, yy = np.meshgrid(x, y)
    dat = np.sin(xx)
    otf_ = otf.otf_from_psf(dat, x[1]-x[0])
    center = tuple(s//2 for s in otf_.shape)
    assert otf_.data[center] == 1+0j
