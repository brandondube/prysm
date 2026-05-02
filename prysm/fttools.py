"""Supplimental tools for computing fourier transforms."""
import math

import numpy as truenp

from .mathops import np, fft
from .conf import config


def fftrange(n, dtype=None):
    """FFT-aligned coordinate grid for n samples."""
    return np.arange(-(n//2), -(n//2)+n, dtype=dtype)


def _next_power_of_2(n):
    # 2 ** k == 1 << k
    return 1 << math.ceil(math.log2(n))


def next_fast_len(n):
    """The next fast FFT size.

    Defaults to powers of two if the FFT backend does not provide a function of the same name.
    """
    try:
        return fft.next_fast_len(n)
    except:  # NOQA -- cannot predict arbitrary library error types
        return _next_power_of_2(n)


def fftfreq(n, d=1.0):
    """Fast Fourier Transform frequency vector."""
    try:
        return fft.fftfreq(n, d).astype(config.precision)
    except:  # NOQA -- cannot predict arbitrary library error types
        out = truenp.fft.fftfreq(n, d).astype(config.precision)
        return np.asarray(out)


def pad2d(array, Q=2, value=0, mode='constant', out_shape=None):
    """Symmetrically pads a 2D array with a value.

    Parameters
    ----------
    array : ndarray
        source array
    Q : float, optional
        oversampling factor; ratio of input to output array widths
    value : float, optioanl
        value with which to pad the array
    mode : str, optional
        mode, passed directly to np.pad
    out_shape : tuple
        output shape for the array.  Overrides Q if given.
        in_shape * Q ~= out_shape (up to integer rounding)

    Returns
    -------
    ndarray
        padded array, may share memory with input array

    Notes
    -----
    padding will be symmetric.

    """
    if Q == 1 and out_shape is None:
        return array
    else:
        in_shape = array.shape
        if out_shape is None:
            out_shape = [math.ceil(s*Q) for s in in_shape]
        else:
            if isinstance(out_shape, int):
                out_shape = [out_shape]*array.ndim

        shape_diff = [o-i for o, i in zip(out_shape, in_shape)]
        pad_shape = []
        for d in shape_diff:
            divby2 = d//2
            lcl = (d-divby2, divby2)  # 13 => 6; (7,6) correct; 12 => 6; (6,6) correct
            pad_shape.append(lcl)

        if mode == 'constant':
            dbytwo = [math.ceil(d/2) for d in shape_diff]
            slcs = tuple((slice(d, d+s) for d, s in zip(dbytwo, in_shape)))
            out = np.zeros(out_shape, dtype=array.dtype)
            if value != 0:
                out += value

            out[slcs] = array

        else:
            kwargs = {'mode': mode}
            out = np.pad(array, pad_shape, **kwargs)

        return out


def crop_center(img, out_shape):
    """Crop the central (out_shape) of an image, with FFT alignment.

    As an example, if img=512x512 and out_shape=200
    out_shape => 200x200 and the returned array is 200x200, 156 elements from the [0,0]th pixel

    This function is the adjoint of pad2d.

    Parameters
    ----------
    img : ndarray
        ndarray of shape (m, n)
    out_shape : int or iterable of int
        shape to crop out, either a scalar or pair of values

    """
    if isinstance(out_shape, int):
        out_shape = (out_shape, out_shape)

    padding = [i-o for i, o in zip(img.shape, out_shape)]
    left = [math.ceil(p/2) for p in padding]
    slcs = tuple((slice(l, l+o) for l, o in zip(left, out_shape)))  # NOQA -- l ambiguous
    return img[slcs]


def forward_ft_unit(dx, samples, shift=True):
    """Compute the units resulting from a fourier transform.

    Parameters
    ----------
    dx : float
        center-to-center spacing of samples in an array
    samples : int
        number of samples in the data
    shift : bool, optional
        whether to shift the output.  If True, first element is a negative freq
        if False, first element is 0 freq.

    Returns
    -------
    ndarray
        array of sample frequencies in the output of an fft

    """
    unit = fftfreq(samples, dx)

    if shift:
        return fft.fftshift(unit)
    else:
        return unit


class MDFT:
    """Matrix DFT parameterized by input coordinates and output frequencies.

    Computes ``out[i, j] = sum_{k, l} ary[k, l] * exp(sign * 2j*pi * (y[k]*fy[i] + x[l]*fx[j]))``
    by precomputing two 1D basis matrices and applying them as ``Ey @ ary @ Ex.T``.

    The class is a pure DFT — it carries no notion of optics, oversampling, or
    normalization. Callers multiply the result by whatever scalar is appropriate
    for their problem.

    Parameters
    ----------
    x, y : ndarray
        1D arrays of input-plane coordinates along the second and first axis
        of the input array, respectively. Lengths must match ``ary.shape[1]``
        and ``ary.shape[0]``.
    fx, fy : ndarray
        1D arrays of output-plane frequencies along the second and first axis
        of the output array, respectively. Output shape will be
        ``(len(fy), len(fx))``.
    sign : int, optional
        Sign of the kernel exponent. ``-1`` (default) is the forward DFT
        convention; ``+1`` is the inverse DFT convention.

    Notes
    -----
    The same instance can be reused on different input arrays of matching
    shape — basis matrices are precomputed once at construction. Holding
    an instance is the caching mechanism; there is no global cache.

    """

    def __init__(self, x, y, fx, fy, sign=-1):
        """Build the basis matrices."""
        prefix = sign * 2j * np.pi
        self.Ex = np.exp(prefix * np.outer(fx, x))  # shape (len(fx), len(x))
        self.Ey = np.exp(prefix * np.outer(fy, y))  # shape (len(fy), len(y))

    def __call__(self, ary):
        """Apply the forward DFT to ``ary``."""
        return self.Ey @ ary @ self.Ex.T

    def adjoint(self, grad):
        """Apply the adjoint (conjugate transpose) of the forward DFT.

        For a real scalar normalization, this is the gradient backpropagation
        operator for ``__call__``. It also coincides with the inverse DFT
        (i.e. an ``MDFT`` of opposite sign that maps the *output* shape back
        to the *input* shape) up to scaling.
        """
        return self.Ey.conj().T @ grad @ self.Ex.conj()

    def nbytes(self):
        """Total size in memory of the basis matrices, bytes."""
        return self.Ex.nbytes + self.Ey.nbytes


class CZT:
    """Chirp-Z transform with the same external API as :class:`MDFT`.

    Internally uses the Bluestein/Jurling factorization for ``O(N log N)``
    cost per axis. Requires ``fx`` and ``fy`` to be uniformly spaced; ``x``
    and ``y`` are also assumed uniformly spaced.

    Parameters mirror :class:`MDFT`. ``sign=+1`` is implemented by complex
    conjugation of the input and output (matches the prior ``iczt2`` behavior).

    """

    def __init__(self, x, y, fx, fy, sign=-1):
        """Precompute CZT components for both axes."""
        if sign not in (-1, 1):
            raise ValueError(f'sign must be -1 or +1, got {sign}')
        self.sign = sign

        Nx = len(x)
        Mx = len(fx)
        Ny = len(y)
        My = len(fy)

        dx = float(x[1] - x[0])
        dfx = float(fx[1] - fx[0])
        dy = float(y[1] - y[0])
        dfy = float(fy[1] - fy[0])

        alpha_x = dx * dfx
        alpha_y = dy * dfy

        # CZT internal coordinate convention is fftrange + shift; recover
        # the per-axis shift (in samples) by reading the value at the
        # centered index.
        shift_x = float(fx[Mx//2]) / dfx
        shift_y = float(fy[My//2]) / dfy

        Kx = next_fast_len(Nx + Mx - 1)
        Ky = next_fast_len(Ny + My - 1)

        dtype = config.precision_complex
        Hx, bx, ax = _prepare_czt_basis(Nx, Mx, Kx, shift_x, alpha_x, dtype)
        Hy, by, ay = _prepare_czt_basis(Ny, My, Ky, shift_y, alpha_y, dtype)

        # column vectors broadcast along axis 0
        self._brow = by[:, np.newaxis]
        self._Hrow = Hy[:, np.newaxis]
        self._arow = ay[:, np.newaxis]
        self._bcol = bx
        self._Hcol = Hx
        self._acol = ax
        self._Mx, self._My = Mx, My
        self._Kx, self._Ky = Kx, Ky

    def __call__(self, ary):
        """Apply the CZT to ``ary``."""
        if self.sign == 1:
            ary = np.conj(ary) if np.iscomplexobj(ary) else ary
        gb = ary * self._bcol
        gb = gb * self._brow
        GB = fft.fft2(gb, (self._Ky, self._Kx))
        GB = GB * self._Hcol
        GB = GB * self._Hrow
        out = fft.ifft2(GB)
        out = out[:self._My, :self._Mx]
        out = out * self._acol
        out = out * self._arow
        if self.sign == 1:
            out = np.conj(out)
        return out

    def adjoint(self, grad):
        """Adjoint not implemented for CZT (matches prior behavior)."""
        raise NotImplementedError('gradient backpropagation not yet implemented for CZT')

    def nbytes(self):
        """Total size in memory of the cached components, bytes."""
        total = 0
        for arr in (self._brow, self._bcol, self._Hrow, self._Hcol,
                    self._arow, self._acol):
            total += arr.nbytes
        return total


def _prepare_czt_basis(N, M, K, shift, alpha, dtype):
    m = fftrange(M, dtype=config.precision)
    if shift != 0:
        m = m + shift

    prefix = -1j * np.pi
    a = np.exp(prefix * m*m * alpha)

    n = fftrange(N, dtype=config.precision)
    b = np.exp(prefix * n*n * alpha)

    h = np.zeros(K, dtype=dtype)

    # populate h piecewise, see Jurling2014 48c, 48d
    start = -((N - M) // 2) + shift
    j = np.arange(-start, -start+M, dtype=config.precision)
    h_left = np.pi * (j * j)

    j = np.arange(-start-N+1, -start, dtype=config.precision)
    h_right = np.pi * (j * j)

    h[:M] = np.exp(1j * alpha * h_left)
    h[K-N+1:K] = np.exp(1j * alpha * h_right)
    h[M:K-N+1] = 0
    H = fft.fft(h)

    return H, b, a


def fourier_resample(f, zoom):
    """Resample f via Fourier methods (truncated sinc interpolation).

    Parameters
    ----------
    f : ndarray
        ndim 2 ndarray, floating point dtype
    zoom : float
        zoom factor to apply
        out.shape == f.shape*zoom

    Returns
    -------
    ndarray
        zoomed f

    Notes
    -----
    Assumes F is (reasonably) bandlimited

    Energy will be deleted, not aliased, if zoom results in the output domain
    being smaller than the Fourier support of f

    """
    if zoom == 1:
        return f

    if isinstance(zoom, (float, int)):
        zoom = (zoom, zoom)
    elif not isinstance(zoom, tuple):
        zoom = tuple(float(z) for z in zoom)

    m, n = f.shape
    M = int(m*zoom[0])
    N = int(n*zoom[1])

    F = fft.fftshift(fft.fft2(fft.ifftshift(f)))

    # Build coordinates for an MDFT that maps the (m, n) Fourier-plane samples
    # to (M, N) spatial samples, mimicking what idft2 used to do with Q=zoom.
    # Input-plane samples (the F array) live on integer indices fftrange(m), fftrange(n).
    # Output-plane samples should live on (1/zoom[i]) * fftrange(M_i).
    x = fftrange(n, dtype=config.precision)
    y = fftrange(m, dtype=config.precision)
    fx = fftrange(N, dtype=config.precision) * (1.0/zoom[1]/n)
    fy = fftrange(M, dtype=config.precision) * (1.0/zoom[0]/m)

    fprime = MDFT(x, y, fx, fy, sign=+1)(F).real
    # match the prior implementation, which combined the executor's internal
    # sqrt(1/(m*n*zoom[0]*zoom[1])) normalization with an external
    # (zoom[0]*zoom[1])/sqrt(m*n) scaling — the net is sqrt(zoom)/(m*n).
    fprime *= math.sqrt(zoom[0]*zoom[1]) / (m*n)
    return fprime
