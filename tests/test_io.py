"""Tests the io functions of prysm."""
import numpy as np

from prysm import io, sample_data

sample_files = sample_data.sample_files


def test_read_zygodat():
    p = sample_files('dat')
    result = io.read_zygo_dat(p)
    assert 'phase' in result
    assert 'intensity' in result
    assert 'lateral_resolution' in result['meta']


# def test_read_zygodat():
#     p = sample_files('dat')
#     result = io.read_zygo_dat(p)
#     assert 'phase' in result
#     assert 'intensity' in result
#     assert 'lateral_resolution' in result['meta']
