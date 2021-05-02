"""Tests for object (target) synthesis routines."""
import pytest

import numpy as np

from prysm import objects, coordinates


@pytest.fixture
def xy():
    x, y = coordinates.make_xy_grid(32, diameter=1)
    return x, y


@pytest.fixture
def rt(xy):
    x, y = xy
    return coordinates.cart_to_polar(x, y)


@pytest.mark.parametrize(['wx', 'wy'], [
    [None, .05],
    [.05, None],
    [.05, .05]])
def test_slit(xy, wx, wy):
    x, y = xy
    ary = objects.slit(x, y, wx, wy)
    assert ary.any()  # at least something white


def test_pinhole(rt):
    r, _ = rt
    assert objects.pinhole(1, r).any()


@pytest.mark.parametrize('bg', ['w', 'b'])
def test_siemensstar(rt, bg):
    star = objects.siemensstar(*rt, 80, background=bg)
    assert star.any()


@pytest.mark.parametrize('bg', ['w', 'b'])
def test_tiltedsquare(xy, bg):
    sq = objects.tiltedsquare(*xy, background=bg)
    assert sq.any()


@pytest.mark.parametrize('crossed', [True, False])
def test_slantededge(xy, crossed):
    se = objects.slantededge(*xy, crossed=crossed)
    assert se.any()


def test_pinhole_ft_functional(rt):
    r, _ = rt
    assert objects.pinhole_ft(1., r).any()


@pytest.mark.parametrize(['wx', 'wy'], [
    [None, .05],
    [.05, None],
    [.05, .05]])
def test_slit_ft_functional(xy, wx, wy):
    r, _ = xy
    assert objects.slit_ft(wx, wy, *xy).any()
