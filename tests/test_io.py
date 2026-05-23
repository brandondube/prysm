"""Tests the io functions of prysm."""
import os
import tempfile

import numpy as np
import pytest

from prysm import io, sample_data

sample_files = sample_data.sample_files


def test_read_zygodat():
    p = sample_files('dat')
    result = io.read_zygo_dat(p)
    assert 'phase' in result
    assert 'intensity' in result
    assert 'lateral_resolution' in result['meta']


def test_write_zygodat_roundtrips_phase_and_lateral_resolution(tmp_path):
    p = sample_files('dat')
    dct = io.read_zygo_dat(p)
    path = tmp_path / 'roundtrip.dat'

    with open(path, 'wb') as fid:
        io.write_zygo_dat(fid, dct['phase'], dct['meta']['lateral_resolution'])

    reread = io.read_zygo_dat(path)
    assert reread['meta']['lateral_resolution'] == pytest.approx(
        dct['meta']['lateral_resolution'] * 1e-3
    )
    assert reread['phase'].shape == dct['phase'].shape
#     assert 'lateral_resolution' in result['meta']


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


def test_codev_gridint_read_from_fixture():
    """Read a hand-written minimal CV INT file string.

    Independent of write_codev_gridint so that a future writer regression
    can't mask a reader regression. Also covers the np.fromstring →
    np.array(split()) migration in io.py:read_codev_gridint.
    """
    # 2x2 grid; SSZ=1 int/wvl, WVL=1um -> int*1000 nm
    # NDA sentinel -32768 marks NaN pixels
    body = "100 200\n300 -32768\n"
    text = (
        "test fixture comment\n"
        "GRD 2 2 SUR WVL 1.0 SSZ 1 NDA -32768\n"
        + body
    )
    with tempfile.NamedTemporaryFile('w', delete=False, suffix='.int') as tf:
        tf.write(text)
        tf.close()
        arr, meta = io.read_codev_gridint(tf.name)
        os.unlink(tf.name)

    # reader does reshape((n, m)) then flipud, so the file's bottom row is
    # the returned top row
    assert arr.shape == (2, 2)
    assert np.isnan(arr[0, 1])
    np.testing.assert_allclose(arr[0, 0], 300 * 1000)
    np.testing.assert_allclose(arr[1, :], [100 * 1000, 200 * 1000])
    assert meta['wavelength'] == 1.0


def test_write_codev_zfr_int_writes_nonzero_coefficients(tmp_path):
    path = tmp_path / 'coefs.int'
    coefs = np.zeros(16)
    coefs[3] = 1.25
    coefs[7] = -2.5

    io.write_codev_zfr_int(coefs, path)

    text = path.read_text()
    assert '1.250000000' in text
    assert '-2.500000000' in text
