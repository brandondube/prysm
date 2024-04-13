"""Tests the io functions of prysm."""
import os
import tempfile

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


def test_write_zygodat_functions():
    p = sample_files('dat')
    dct = io.read_zygo_dat(p)
    tf = tempfile.TemporaryFile('wb')
    io.write_zygo_dat(tf, dct['phase'], dct['meta']['lateral_resolution'])


def test_codev_gridint_roundtrip():
    # units are nm and grid int has severe problems with resolution
    arr = np.random.rand(32, 32)*100
    with tempfile.NamedTemporaryFile(delete=False) as tf:
        tf.close()
        io.write_codev_gridint(arr, tf.name)
        arr2, _ = io.read_codev_gridint(tf.name)
        os.unlink(tf.name)

    # super wide tol because rounding is egregious
    assert np.allclose(arr, arr2, atol=1)


def test_write_codev_zfr_int_functions():
    coefs = np.random.rand(16)
    with tempfile.NamedTemporaryFile('w', delete=False) as tf:
        tf.close()
        io.write_codev_zfr_int(coefs, tf.name)
        os.unlink(tf.name)
