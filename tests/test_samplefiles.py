"""Tests for samplefiles."""
import pytest

from prysm import sample_files


def test_barbara():
    assert sample_files('barbara.png')


def test_boat():
    assert sample_files('boat.png')


def test_goldhill():
    assert sample_files('goldhill.png')


def test_mountain():
    assert sample_files('mountain.png')
