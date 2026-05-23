"""Tests for convolution routines."""
from functools import partial

import numpy as np

from prysm import convolution, degredations


def test_conv_with_centered_delta_psf_is_identity():
    obj = np.arange(25, dtype=float).reshape(5, 5)
    psf = np.zeros_like(obj)
    psf[2, 2] = 1

    out = convolution.conv(obj, psf)

    np.testing.assert_allclose(out, obj, atol=1e-12)


def test_apply_transfer_functions_uses_callable_frequency_arguments():
    obj = np.arange(16, dtype=float).reshape(4, 4)

    def zero_lowpass(fx, fy, fr):
        assert fx.shape == (1, obj.shape[1])
        assert fy.shape == (obj.shape[0], 1)
        assert fr.shape == obj.shape
        return np.zeros_like(fr)

    out = convolution.apply_transfer_functions(obj, 1, [zero_lowpass])

    np.testing.assert_allclose(out, 0, atol=1e-12)


def test_apply_transfer_functions_with_shift_preserves_identity_tf():
    obj = np.arange(16, dtype=float).reshape(4, 4)

    out = convolution.apply_transfer_functions(obj, 1, [np.ones_like(obj)], shift=True)

    np.testing.assert_allclose(out, obj, atol=1e-12)


def test_apply_transfer_functions_composes_smear_and_jitter():
    sm = partial(degredations.smear_ft, width=1, height=1)
    ji = partial(degredations.jitter_ft, scale=1)
    obj = np.ones((8, 8), dtype=float)

    out = convolution.apply_transfer_functions(obj, 1, [sm, ji])

    assert out.shape == obj.shape
    assert np.isfinite(out).all()
