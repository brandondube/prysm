''' Tests the io functions of prysm.
'''
from pathlib import Path

import numpy as np

from prysm import io

data_root = Path(__file__).parent / 'io_files'


def test_read_mtfvfvf_functions():
    p = data_root / 'valid_sample_MTFvFvF_Sag.txt'
    result = io.read_trioptics_mtfvfvf(p)
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
