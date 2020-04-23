"""Tests for basic geometry."""
import pytest

import numpy as np

from prysm import geometry

SHAPES = [
    geometry.square,
    geometry.pentagon,
    geometry.hexagon,
    geometry.heptagon,
    geometry.octagon,
    geometry.nonagon,
    geometry.decagon,
    geometry.hendecagon,
    geometry.dodecagon,
    geometry.trisdecagon]


def test_all_predefined_shapes():  # TODO: test more than just that these are ndarrays
    for shape in SHAPES:
        assert type(shape()) is np.ndarray


@pytest.mark.parametrize('sides, samples', [
    [5,  128],
    [10, 128],
    [25, 128],
    [5,  256],
    [25, 68]])
def test_regular_polygon(sides, samples):  # TODO: test more than just that these are ndarrays
    assert type(geometry.regular_polygon(sides, samples)) is np.ndarray


@pytest.mark.parametrize('sigma, samples', [
    [0.5, 128],
    [5,   256]])
def test_gaussian(sigma, samples):
    assert type(geometry.gaussian(sigma, samples)) is np.ndarray


def test_rotated_ellipse_fails_if_minor_is_bigger_than_major():
    minor = 1
    major = 0.5
    with pytest.raises(ValueError):
        geometry.rotated_ellipse(width_major=major, width_minor=minor)


@pytest.mark.parametrize('maj, min, majang', [
    [1, 0.5, 0],
    [1, 1, 5],
    [0.8, 0.1, 90]])
def test_rotated_ellipse(maj, min, majang):
    assert type(geometry.rotated_ellipse(width_major=maj,
                                         width_minor=min,
                                         major_axis_angle=majang)) is np.ndarray


def test_allcircles_zeros():
    funcs = ['circle', 'truecircle', 'inverted_circle']
    for func in funcs:
        assert (getattr(geometry, func)(32, 0) == 0).all()


def test_mask_cleaner_with_tuple():
    type_radius = ('circle', 1)
    assert type(geometry.mask_cleaner(type_radius, 64)) is np.ndarray


def test_truecircle_doesnt_error():
    circ = geometry.truecircle()
    assert type(circ) is np.ndarray


def test_inverted_circle_doesnt_error():
    icirc = geometry.inverted_circle()
    assert type(icirc) is np.ndarray


@pytest.mark.parametrize('vanes', [2, 3, 5, 6, 10])
def test_generate_spider_doesnt_error(vanes):
    mask = geometry.generate_spider(vanes, 1, 0, 25, 128)
    assert type(mask) is np.ndarray


def test_rectangle_duplicates_y_from_x():
    mask = geometry.rectangle(1)
    assert (mask == 1).all()


def test_rectangle_doesnt_break_angle_90():
    mask = geometry.rectangle(1, angle=90)
    assert mask.any()


def test_rectangle_doesnt_break_angle_not_0_or_90():
    mask = geometry.rectangle(1, angle=45)
    assert mask.any()
