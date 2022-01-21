"""Supplimental tools for computing fourier transforms."""
import math
from collections.abc import Iterable

from .mathops import np, fft
from .conf import config


def fftrange(n, dtype=None):
    """FFT-aligned coordinate grid for n samples."""
    return np.arange(-n//2, -n//2+n, dtype=dtype)


def pad2d(array, Q=2, value=0, mode='constant', out_shape=None):
    """Symmetrically pads a 2D array with a value.

    Parameters
    ----------
    array : numpy.ndarray
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
    numpy.ndarray
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
            slcs = tuple((slice(p[0], -p[1]) for p in pad_shape))
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
    img : numpy.ndarray
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
    numpy.ndarray
        array of sample frequencies in the output of an fft

    """
    unit = fft.fftfreq(samples, dx)

    if shift:
        return fft.fftshift(unit)
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
        Q = float(Q)
        if not isinstance(samples, Iterable):
            samples = (samples, samples)

        if not isinstance(shift, Iterable):
            shift = (shift, shift)

        return (Q, ary.shape, samples, shift)

    def dft2(self, ary, Q, samples, shift=None):
        """Compute the two dimensional Discrete Fourier Transform of a matrix.

        Parameters
        ----------
        ary : numpy.ndarray
            an array, 2D, real or complex.  Not fftshifted.
        Q : float
            oversampling / padding factor to mimic an FFT.  If Q=2, Nyquist sampled
        samples : int or Iterable
            number of samples in the output plane.
            If an int, used for both dimensions.  If an iterable, used for each dim
        shift : float, optional
            shift of the output domain, as a frequency.  Same broadcast
            rules apply as with samples.

        Returns
        -------
        numpy.ndarray
            2D array containing the shifted transform.
            Equivalent to ifftshift(fft2(fftshift(ary))) modulo output
            sampling/grid differences

        """
        self._setup_bases(ary=ary, Q=Q, samples=samples, shift=shift)
        key = self._key(ary=ary, Q=Q, samples=samples, shift=shift)
        Eout, Ein = self.Eout_fwd[key], self.Ein_fwd[key]

        out = Eout @ ary @ Ein

        return out

    def idft2(self, ary, Q, samples, shift=None):
        """Compute the two dimensional inverse Discrete Fourier Transform of a matrix.

        Parameters
        ----------
        ary : numpy.ndarray
            an array, 2D, real or complex.  Not fftshifted.
        Q : float
            oversampling / padding factor to mimic an FFT.  If Q=2, Nyquist sampled
        samples : int or Iterable
            number of samples in the output plane.
            If an int, used for both dimensions.  If an iterable, used for each dim
        shift : float, optional
            shift of the output domain, as a frequency.  Same broadcast
            rules apply as with samples.

        Returns
        -------
        numpy.ndarray
            2D array containing the shifted transform.
            Equivalent to ifftshift(ifft2(fftshift(ary))) modulo output
            sampling/grid differences

        """
        self._setup_bases(ary=ary, Q=Q, samples=samples, shift=shift)
        key = self._key(ary=ary, Q=Q, samples=samples, shift=shift)
        Eout, Ein = self.Eout_rev[key], self.Ein_rev[key]
        out = Eout @ ary @ Ein

        return out

    def _setup_bases(self, ary, Q, samples, shift):
        """Set up the basis matricies for given sampling parameters."""
        # broadcast sampling and shifts
        if not isinstance(samples, Iterable):
            samples = (samples, samples)

        if not isinstance(shift, Iterable):
            shift = (shift, shift)

        # this is for dtype stabilization
        Q = float(Q)

        key = self._key(Q=Q, ary=ary, samples=samples, shift=shift)

        n, m = ary.shape
        N, M = samples

        try:
            # assume all arrays for the input are made together
            self.Ein_fwd[key]
        except KeyError:
            # X is the second dimension in C (numpy) array ordering convention

            X, Y, U, V = (fftrange(n, dtype=config.precision) for n in (m, n, M, N))

            # do not even perform an op if shift is nothing
            if shift[0] is not None:
                Y -= shift[0]
                X -= shift[1]
                V -= shift[0]
                U -= shift[1]

            nm = n*m
            NM = N*M
            r = NM/nm
            a = 1 / Q
            Eout_fwd = np.exp(-1j * 2 * np.pi * a / n * np.outer(Y, V).T)
            Ein_fwd = np.exp(-1j * 2 * np.pi * a / m * np.outer(X, U))
            Eout_rev = np.exp(1j * 2 * np.pi * a / n * np.outer(Y, V).T) * (1/r)
            Ein_rev = np.exp(1j * 2 * np.pi * a / m * np.outer(X, U)) * (1/nm)
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
