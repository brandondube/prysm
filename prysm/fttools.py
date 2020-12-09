"""Supplimental tools for computing fourier transforms."""
from collections.abc import Iterable

from .mathops import engine as e
from .conf import config


def fftrange(n):
    """FFT-aligned coordinate grid for n samples."""
    return e.arange(-n//2, -n//2+n)


def pad2d(array, Q=2, value=0, mode='constant'):
    """Symmetrically pads a 2D array with a value.

    Parameters
    ----------
    array : `numpy.ndarray`
        source array
    Q : `float`, optional
        oversampling factor; ratio of input to output array widths
    value : `float`, optioanl
        value with which to pad the array
    mode : `str`, optional
        mode, passed directly to np.pad

    Returns
    -------
    `numpy.ndarray`
        padded array

    Notes
    -----
    padding will be symmetric.

    """
    if Q == 1:
        return array
    else:
        if mode == 'constant':
            pad_shape, out_x, out_y = _padshape(array, Q)
            y, x = array.shape
            if value == 0:
                out = e.zeros((out_y, out_x), dtype=array.dtype)
            else:
                out = e.zeros((out_y, out_x), dtype=array.dtype) + value
            yy, xx = pad_shape
            out[yy[0]:yy[0] + y, xx[0]:xx[0] + x] = array
            return out
        else:
            pad_shape, *_ = _padshape(array, Q)

            if mode == 'constant':
                kwargs = {'constant_values': value, 'mode': mode}
            else:
                kwargs = {'mode': mode}
            return e.pad(array, pad_shape, **kwargs)


def _padshape(array, Q):
    y, x = array.shape
    out_x = int(e.ceil(x * Q))
    out_y = int(e.ceil(y * Q))
    factor_x = (out_x - x) / 2
    factor_y = (out_y - y) / 2
    return (
        (int(e.floor(factor_y)), int(e.ceil(factor_y))),
        (int(e.floor(factor_x)), int(e.ceil(factor_x)))), out_x, out_y


def forward_ft_unit(sample_spacing, samples, shift=True):
    """Compute the units resulting from a fourier transform.

    Parameters
    ----------
    sample_spacing : `float`
        center-to-center spacing of samples in an array
    samples : `int`
        number of samples in the data
    shift : `bool`, optional
        whether to shift the output.  If True, first element is a negative freq
        if False, first element is 0 freq.

    Returns
    -------
    `numpy.ndarray`
        array of sample frequencies in the output of an fft

    """
    unit = e.fft.fftfreq(samples, sample_spacing)

    if shift:
        return e.fft.fftshift(unit)
    else:
        return unit


class MatrixDFTExecutor:
    """MatrixDFTExecutor is an engine for performing matrix triple product DFTs as fast as possible."""

    def __init__(self):
        """Create a new MatrixDFTExecutor instance."""
        # Eq. (10-11) page 8 from R. Soumer (2007) oe-15--24-15935
        self.Ein_fwd = {}
        self.Eout_fwd = {}
        self.Ein_rev = {}
        self.Eout_rev = {}

    def _key(self, ary, Q, samples, shift):
        """Key to X, Y, U, V dicts."""
        if not isinstance(samples, Iterable):
            samples = (samples, samples)

        if not isinstance(shift, Iterable):
            shift = (shift, shift)

        return (Q, ary.shape, samples, shift)

    def dft2(self, ary, Q, samples, shift=None, norm=True):
        """Compute the two dimensional Discrete Fourier Transform of a matrix.

        Parameters
        ----------
        ary : `numpy.ndarray`
            an array, 2D, real or complex.  Not fftshifted.
        Q : `float`
            oversampling / padding factor to mimic an FFT.  If Q=2, Nyquist sampled
        samples : `int` or `Iterable`
            number of samples in the output plane.
            If an int, used for both dimensions.  If an iterable, used for each dim
        shift : `float`, optional
            shift of the output domain, as a frequency.  Same broadcast
            rules apply as with samples.
        norm : `bool`, optional
            if True, normalize the computation such that Parseval's theorm
            is not violated

        Returns
        -------
        `numpy.ndarray`
            2D array containing the shifted transform.
            Equivalent to ifftshift(fft2(fftshift(ary))) modulo output
            sampling/grid differences

        """
        self._setup_bases(ary=ary, Q=Q, samples=samples, shift=shift)
        key = self._key(ary=ary, Q=Q, samples=samples, shift=shift)
        Eout, Ein = self.Eout_fwd[key], self.Ein_fwd[key]

        out = Eout @ ary @ Ein
        if norm:
            coef = self._norm(ary=ary, Q=Q, samples=samples)
            out *= (1/coef)

        return out

    def idft2(self, ary, Q, samples, shift=None, norm=True):
        """Compute the two dimensional inverse Discrete Fourier Transform of a matrix.

        Parameters
        ----------
        ary : `numpy.ndarray`
            an array, 2D, real or complex.  Not fftshifted.
        Q : `float`
            oversampling / padding factor to mimic an FFT.  If Q=2, Nyquist sampled
        samples : `int` or `Iterable`
            number of samples in the output plane.
            If an int, used for both dimensions.  If an iterable, used for each dim
        shift : `float`, optional
            shift of the output domain, as a frequency.  Same broadcast
            rules apply as with samples.
        norm : `bool`, optional
            if True, normalize the computation such that Parseval's theorm
            is not violated

        Returns
        -------
        `numpy.ndarray`
            2D array containing the shifted transform.
            Equivalent to ifftshift(ifft2(fftshift(ary))) modulo output
            sampling/grid differences

        """
        self._setup_bases(ary=ary, Q=Q, samples=samples, shift=shift)
        key = self._key(ary=ary, Q=Q, samples=samples, shift=shift)
        Eout, Ein = self.Eout_rev[key], self.Ein_rev[key]
        out = Eout @ ary @ Ein
        if norm:
            coef = self._norm(ary=ary, Q=Q, samples=samples)
            out *= (1/coef)

        return out

    def _norm(self, ary, Q, samples):
        """Coefficient associated with a given propagation."""
        if not isinstance(samples, Iterable):
            samples = (samples, samples)

        # commenting out this warning
        # strictly true in the one-way case
        # but a 128 => 256, Q=2 fwd followed
        # by 256 => 128 Q=1 rev produces ~size*eps
        # max error, so this warning is overzealous
        # if samples[0]/Q < ary.shape[0]:
            # warn('mdft: computing normalization for output condition which contains Dirichlet clones, normalization cannot be accurate')

        n, m = ary.shape
        N, M = samples
        sz_i = n * m
        sz_o = N * M
        return e.sqrt(sz_i) * Q * e.sqrt(sz_i/sz_o)

    def _setup_bases(self, ary, Q, samples, shift):
        """Set up the basis matricies for given sampling parameters."""
        # broadcast sampling and shifts
        if not isinstance(samples, Iterable):
            samples = (samples, samples)

        if not isinstance(shift, Iterable):
            shift = (shift, shift)

        key = self._key(Q=Q, ary=ary, samples=samples, shift=shift)

        n, m = ary.shape
        N, M = samples

        try:
            # assume all arrays for the input are made together
            self.Ein_fwd[key]
        except KeyError:
            # X is the second dimension in C (numpy) array ordering convention
            X = e.arange(m, dtype=config.precision) - m//2
            Y = e.arange(n, dtype=config.precision) - n//2
            U = e.arange(M, dtype=config.precision) - M//2
            V = e.arange(N, dtype=config.precision) - N//2

            # do not even perform an op if shift is nothing
            if shift[0] is not None:
                Y -= shift[0]
                X -= shift[1]
                V -= shift[0]
                U -= shift[1]

            a = 1 / Q
            Eout_fwd = e.exp(-1j * 2 * e.pi * a / n * e.outer(Y, V).T)
            Ein_fwd = e.exp(-1j * 2 * e.pi * a / m * e.outer(X, U))
            Eout_rev = e.exp(1j * 2 * e.pi * a / n * e.outer(Y, V).T)
            Ein_rev = e.exp(1j * 2 * e.pi * a / m * e.outer(X, U))
            self.Ein_fwd[key] = Ein_fwd
            self.Eout_fwd[key] = Eout_fwd
            self.Eout_rev[key] = Eout_rev
            self.Ein_rev[key] = Ein_rev

    def clear(self):
        """Empty the internal caches to release memory."""
        self.Ein_fwd = {}
        self.Eout_fwd = {}
        self.Ein_rev = {}
        self.Eout_rev = {}

    def nbytes(self):
        """Total size in memory of the cache in bytes."""
        total = 0
        for dict_ in (self.Ein_fwd, self.Eout_fwd, self.Ein_rev, self.Eout_rev):
            for key in dict_:
                total += dict_[key].nbytes

        return total


mdft = MatrixDFTExecutor()
