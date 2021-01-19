"""Tests the io functions of prysm."""
import numpy as np

from prysm import io, sample_data

sample_files = sample_data.sample_files


def test_read_mtfvfvf_functions():
    p = sample_files('mtfvfvf')
    result = io.read_trioptics_mtfvfvf(p)
    assert result


def test_read_mtf_vs_field():
    p = sample_files('mtfvf')
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


def test_read_mtf_and_meta():
    p = sample_files('mtf')
    result = io.read_trioptics_mtf(p, metadata=True)
    assert result['focus'] == 2.8484
    assert max(result['freq']) == 900
    assert result['wavelength'] == 0.56
    assert result['efl'] == 97.4
    assert result['tan'][-1] == 0.007
    assert result['sag'][-1] == 0.001


def test_read_zygodat():
    p = sample_files('dat')
    result = io.read_zygo_dat(p)
    assert 'phase' in result
    assert 'intensity' in result
    assert 'lateral_resolution' in result['meta']
