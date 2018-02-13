''' Tests the io functions of prysm.
'''
from pathlib import Path

import pytest
import numpy as np

from prysm import io

data_root = Path(__file__).parent / 'io_files'


def test_read_oceanoptics_functions():
    # file, has 2048 pixels
    p = data_root / 'valid_sample_oceanoptics.txt'
    data = io.read_oceanoptics(p)

    # returns dict with wvl, value keys
    assert 'wvl' in data
    assert 'values' in data

    # data is of the proper length
    assert len(data['wvl'] == 2048)
    assert len(data['values'] == 2048)

    # data begins and ends with correct values
    assert data['wvl'][0] == 178.179
    assert data['values'][0] == 556.52

    assert data['wvl'][-1] == 871.906
    assert data['values'][-1] == 84.35


def test_read_oceanoptics_raises_for_invalid():
    p = data_root / 'invalid_sample_oceanoptics.txt'
    with pytest.raises(IOError):
        io.read_oceanoptics(p)


def test_read_mtfvfvf_functions():
    p = data_root / 'valid_sample_MTFvFvF_Sag.txt'
    result = io.read_trioptics_MTFvFvF(p)
    assert result


def test_read_mtf_vs_field():
    p = data_root / 'valid_sample_trioptics_mtf_vs_field.mht'
    result = io.read_trioptics_mtf_vs_field(p)
    assert 'sag' in result
    assert 'tan' in result
    assert 'freq' in result
    assert 'field' in result
    real_fields = np.asarray([20.0, 18.0, 16.0, 14.0, 12.0, 10.0, 8.0, 6.0, 4.0, 2.0, 0.0,
                              -2.0, -4.0, -6.0, -8.0, -10.0, -12.0, -14.0, -16.0, -18.0, -20.0])
    real_freqs = np.asarray([100, 200, 300, 400, 500], dtype=np.float64)
    assert np.allclose(real_fields, result['field'])
    assert np.allclose(real_freqs, result['freq'])
