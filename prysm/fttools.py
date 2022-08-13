"""Supplimental tools for computing fourier transforms."""
import math
from collections.abc import Iterable

import numpy as truenp

from .mathops import np, fft
from .conf import config


def fftrange(n, dtype=None):
    """FFT-aligned coordinate grid for n samples."""
    # return np.arange(-n//2, -n//2+n, dtype=dtype)
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
        return fft.fftfreq(n, d)
    except:  # NOQA -- cannot predict arbitrary library error types
        # if the FFT backend does not have fftfreq, use numpy's.  Then, cast
        # the data to the current numpy backend's data type
        # for example, if fft = cupy fft and it doesn't have FFTfreq,
        # use numpy's fftfreq, then turn that into a CuPy array
        out = truenp.fft.fftfreq(n, d)
        return np.asarray(out)


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
            # TODO: clean this garbage up, the code here shouldn't be completely
            # non common mode the way it is

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
    unit = fftfreq(samples, dx)

    if shift:
        return fft.fftshift(unit)
    else:
        return unit


def fourier_resample(f, zoom):
    """Resample f via Fourier methods (truncated sinc interpolation).

    Parameters
    ----------
    f : numpy.ndarray
        ndim 2 ndarray, floating point dtype
    zoom : float
        zoom factor to apply
        out.shape == f.shape*zoom

    Returns
    -------
    numpy.ndarray
        zoomed f

    Notes
    -----
    Assumes F is (reasonably) bandlimited

    Energy will be deleted, not aliased, if zoom results in the output domain
    being smaller than the Fourier support of f

    """
    # performance: not pre-shifting f introduces a linear phase term to the FFT
    # but we do the opposite "mistake" on the way out and they cancel.
    if zoom == 1:
        return f

    if isinstance(zoom, (float, int)):
        zoom = (zoom, zoom)
    elif not isinstance(zoom, tuple):
        zoom = tuple(float(zoom) for zoom in zoom)  # float for dtype stabilization: cupy

    m, n = f.shape
    M = int(m*zoom[0])
    N = int(n*zoom[1])

    F = fft.fftshift(fft.fft2(fft.ifftshift(f)))
    fprime = mdft.idft2(F, zoom, (M, N)).real
    fprime *= (zoom[0]*zoom[1])/(np.sqrt(f.size))
    return fprime
    # the below code is not commented out but is unreachable, it is an
    # alternative way, however it will produce a rounding error in the scaling
    # when m*zoom is not an integer
    F = fft.fftshift(fft.fft2(fft.ifftshift(f)))
    if zoom < 1:
        F = crop_center(F, (M, N))
    else:
        F = pad2d(F, out_shape=(M, N), value=0, mode='constant')

    # ifftshift divides by m*n
    # the scaling is wrong by the ratio F.size/f.size ~= zoom^2 (integer rounding)
    # real before shift, cheaper to shift f64 than c128
    fprime = fft.fftshift(fft.ifft2(fft.ifftshift(F)).real)
    fprime *= (F.size/f.size)
    return fprime


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
        if isinstance(Q, (float, int)):
            Q = (Q, Q)
        elif not isinstance(Q, tuple):
            Q = tuple(float(q) for q in Q)  # float for dtype stabilization: cupy

        if not isinstance(samples, Iterable):
            samples = (samples, samples)

        if not isinstance(shift, Iterable):
            shift = (shift, shift)

        return (Q, ary.shape, samples, shift)

    def dft2(self, ary, Q, samples, shift=(0, 0)):
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
        key = self._key(ary=ary, Q=Q, samples=samples, shift=shift)
        self._setup_bases(key)
        Eout, Ein = self.Eout_fwd[key], self.Ein_fwd[key]

        out = Eout @ ary @ Ein

        return out

    def idft2(self, ary, Q, samples, shift=(0, 0)):
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
        key = self._key(ary=ary, Q=Q, samples=samples, shift=shift)
        self._setup_bases(key)

        Eout, Ein = self.Eout_rev[key], self.Ein_rev[key]
        out = Eout @ ary @ Ein

        return out

    def _setup_bases(self, key):
        """Set up the basis matricies for given sampling parameters."""
        # broadcast sampling and shifts

        Q, shp, samples, shift = key

        Qn, Qm = Q
        # conversion here to Soummer's notation
        # still have N, M for dimensionality but
        # use lowercase m for "zoom" factor...
        mn, mm = 1 / Qn, 1 / Qm
        Na, Ma = shp
        Nb, Mb = samples

        try:
            # assume all arrays for the input are made together
            self.Ein_fwd[key]
        except KeyError:
            # X is the second dimension in C (numpy) array ordering convention

            X, Y, U, V = (fftrange(n, dtype=config.precision) for n in (Ma, Na, Mb, Nb))

            # do not even perform an op if shift is nothing
            if shift[1] != 0:
                Y -= shift[1]
                V -= shift[1]

            if shift[0] != 0:
                X -= shift[0]
                U -= shift[0]

            Eout_fwd = np.exp(-2j * np.pi / Na * mn * np.outer(Y, V).T)
            Ein_fwd = np.exp(-2j * np.pi / Ma * mm * np.outer(X, U))
            Eout_rev = np.exp(2j * np.pi / Na * mn * np.outer(Y, V).T)
            Ein_rev = np.exp(2j * np.pi / Ma * mm * np.outer(X, U))
            Ein_fwd *= (1/(Na*Qn))
            Ein_rev *= (1/(Nb*Qm))
            # scaling = np.sqrt(dx * dy * dxi * deta) / (wvl * fn)
            # observe
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


class ChirpZTransformExecutor:
    """Type which executes Chirp Z Transforms on 2D data, aka zoom FFTs."""
    def __init__(self):
        """Create a new Chirp Z Transform Executor."""
        self.components = {}

    def czt2(self, ary, Q, samples, shift=(0, 0)):
        """Compute the two dimensional Chirp Z Transform of a matrix.

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
            shift of the output domain, as a number of samples at the output
            sample rate.  I.e., if ary is 256x256, Q=2, and samples=512, then
            the output is identical to a padded FFT.  If shift=256, the DC frequency
            will be at the edge of the array; shift=(-256,256) would produce the
            same result as a padded FFT without shifts.

        Returns
        -------
        numpy.ndarray
            2D array containing the shifted transform.
            Equivalent to ifftshift(fft2(fftshift(ary))) modulo output
            sampling/grid differences

        """
        if not isinstance(samples, Iterable):
            samples = (samples, samples)

        if not isinstance(shift, Iterable):
            shift = (shift, shift)

        if not isinstance(Q, Iterable):
            Q = (Q, Q)

        dtype = ary.dtype

        m, n = ary.shape
        M, N = samples
        alphay = 1/(m*Q[0])
        alphax = 1/(n*Q[1])
        # alphay, alphax = Q

        # slightly different notation to Jurling
        # in Jurling, M = unpadded size of input domain
        #             R = unpadded size of output domain
        # we have     m = unpadded size of input domain
        #             M = unpadded size of output domain
        # the constraint is >= M+R - 1 -> m+M-1 (and #cols analogs)
        K = next_fast_len(m+M-1)
        L = next_fast_len(n+N-1)  # -                    norm = False
        key = (m, n, M, N, K, L, alphay, alphax, *shift, dtype, True)
        self._setup_bases(key)
        # b, H, a are the variables from Jurling (where they have hats)
        brow, bcol, Hrow, Hcol, arow, acol = self.components[key]

        # in our case, the dense 2D arrays are stored as vectors, which
        # dramatically reduces static memory usage.
        # Runtime is very slightly slower.

        # now do the transform, written out just like Jurling
        gb = ary * bcol
        gb *= brow  # faster in-place (minutely...)

        # K, L = size; pad if need be internally
        # benchmarked, and found 256 -> 512 w/ fft2:
        # pad2d+fft2 = 4.34 ms
        # fft2 w/ internal padding = 4.2 ms
        # 1024 -> 2048 = 112, 113 (same order)
        # --> marginal improvement internal to FFT for small data, who cares
        # for big; let FFT do it
        GBhat = fft.fft2(gb, (K, L))
        GBhat *= Hcol
        GBhat *= Hrow
        gxformed = fft.ifft2(GBhat)  # transformed g
        gxformed = gxformed[:M, :N]
        gxformed *= acol
        gxformed *= arow
        return gxformed

    def iczt2(self, ary, Q, samples, shift=(0, 0)):
        """Compute the two dimensional inverse Chirp Z Transform of a matrix.

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
            shift of the output domain, as a number of samples at the output
            sample rate.  I.e., if ary is 256x256, Q=2, and samples=512, then
            the output is identical to a padded FFT.  If shift=256, the DC frequency
            will be at the edge of the array; shift=(-256,256) would produce the
            same result as a padded FFT without shifts.

        Returns
        -------
        numpy.ndarray
            2D array containing the shifted transform.
            Equivalent to ifftshift(fft2(fftshift(ary))) modulo output
            sampling/grid differences

        """
        # notice: chirp z transform is fwd/reverse based only on +i vs -i in the
        # complex exponents
        # we can save a whole ton of memory and code dup by just using the
        # forward transform on the complex conjugate of the input.  Generally
        # arrays are complex for optics since we want to handle having OPD,
        # but np.conj copies real inputs, so we optimize for that.
        if np.iscomplexobj(ary):
            ary = np.conj(ary)

        xformed = self.czt2(ary, Q, samples, shift)
        return xformed

    def _setup_bases(self, key):
        try:
            # probe the cache to see if the key exists, else generate
            self.components[key]
        except KeyError:
            m, n, M, N, K, L, alphay, alphax, shifty, shiftx, dtype, norm = key
            Hrow, brow, arow = _prepare_czt_basis(m, M, K, shiftx, alphax, dtype, norm)
            Hcol, bcol, acol = _prepare_czt_basis(n, N, L, shifty, alphay, dtype, norm)
            # those are all vectors, now add singleton dimensions for numpy
            # to broadcast correctly in the following steps
            brow = brow[:, np.newaxis]
            Hrow = Hrow[:, np.newaxis]
            arow = arow[:, np.newaxis]
            self.components[key] = (brow, bcol, Hrow, Hcol, arow, acol)
            # benchmarked a version which turns these into 2D arrays at this step,
            # instead of doing two multiplies in the main czt function.
            # it is about 2% faster to compute the products up front here, in
            # exchange for squaring the memory use -> leave the caches as vectors

    def clear(self):
        """Empty the cache."""
        self.components = {}

    def nbytes(self):
        """Total size in memory of the cache in bytes."""
        total = 0
        for key in self.components:
            arrays = self.components[key]
            for array in arrays:
                total += array.nbytes

        return total


def _prepare_czt_basis(N, M, K, shift, alpha, dtype, norm=False):
    m = fftrange(M, dtype=dtype)
    if shift != 0:
        m += shift

    prefix = -1j * np.pi
    a = np.exp(prefix * m*m * alpha)

    n = fftrange(N, dtype=dtype)
    b = np.exp(prefix * n*n * alpha)

    # maybe can replace with empty for minor performance gains?
    h = np.zeros(K, dtype=dtype)

    # need to populate h piecewise, see Jurling2014 48c, 48d
    start = -((N - M) // 2) + shift
    j = np.arange(-start, -start+M, dtype=dtype)  # do not need a "-1" because arange is naturally end-exclusive
    # j is an index variable
    h[:M] = np.pi * (j * j)

    # check for off-by-1 bug
    j = np.arange(-start-N+1, -start, dtype=dtype)
    h[K-N+1:K] = np.pi * (j * j)

    # order matters, scalar * scalar * array avoids operations on whole array over and over again
    h = np.exp(1j * alpha * h)
    h[M:K-N+1] = 0
    H = fft.fft(h)
    if norm:
        b *= (alpha / np.sqrt(alpha))  # mul cheaper than div; div a single scalar instead of M elements
    return H, b, a


mdft = MatrixDFTExecutor()  # NOQA
czt = ChirpZTransformExecutor()  # NOQA
