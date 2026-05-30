"""Low-level padding and adjoint kernels shared by the propagation routines.
"""
from ..mathops import np
from ..fttools import pad2d, crop_center


def _maybe_pad(wavefunction, Q):
    """Symmetric-pad by factor Q, or pass through if Q == 1."""
    if Q != 1:
        return pad2d(wavefunction, Q)
    return wavefunction


def _shape_before_pad(padded_shape, Q):
    """Infer the input shape from the padded shape and padding factor."""
    if Q == 1:
        return padded_shape
    return tuple(int(s // Q) for s in padded_shape)


def _adjoint_pad2d(array, Q):
    """Apply the adjoint of _maybe_pad(array, Q)."""
    out_shape = _shape_before_pad(array.shape, Q)
    if out_shape != array.shape:
        return crop_center(array, out_shape)
    return array


def _adjoint_multiply(grad, factor, real=False):
    """Adjoint with respect to x for y = x * factor."""
    if np.iscomplexobj(factor):
        out = grad * np.conj(factor)
    else:
        out = grad * factor
    if real:
        return np.real(out)
    return out


def _phase_prefix(wavelength):
    """Phase prefix or scale factor such that mul w/ OPD in nm produces radians.
    """
    return 1j * 2 * np.pi / wavelength / 1e3
