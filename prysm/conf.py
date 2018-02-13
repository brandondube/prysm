"""Configuration for this instance of prysm."""
import numpy as np

_precision = 64
_precision_complex = 128
_parallel_rgb = True
_backend = 'np'
_zernike_base = 1


class Config(object):
    """Global configuration of prysm."""
    def __init__(self, precision=_precision, parallel_rgb=_parallel_rgb, backend=_backend, zernike_base=_zernike_base):
        """Create a new `Config` object.

        Parameters
        ----------
        precision : `int`
            32 or 64, number of bits of precision
        parallel_rgb : `bool`
            whether to parallelize RGB computations or not.  This improves
            performance for large arrays, but may slow things down if arrays are
            relatively small due to the spinup time of new processes

        backend : `str`, {'np'}
            a supported backend.  Current options are only "np" for numpy

        zernike_base : `int`, {0, 1}
            base for zernikes; start at 0 or 1

        """
        global _precision
        global _precision_complex
        global _parallel_rgb
        global _backend
        global _zernike_base

        self.set_precision(precision)
        self.set_parallel_rgb(parallel_rgb)
        self.set_backend(backend)
        self.set_zernike_base(zernike_base)

    def set_precision(self, precision):
        """Adjust precision used by prysm.

        Parameters
        ----------
        precision : `int`, {32, 64}
            what precision to use; either 32 or 64 bits

        Raises
        ------
        ValueError
            if precision is not a valid option

        """
        global _precision
        global _precision_complex

        if precision not in (32, 64):
            raise ValueError('invalid precision.  Precision should be 32 or 64.')

        if precision == 32:
            _precision = np.float32
            _precision_complex = np.complex64
        else:
            _precision = np.float64
            _precision_complex = np.complex128

    def set_parallel_rgb(self, parallel):
        """Adjust if prysm parallelizes RGB computations.

        Parameters
        ----------
        parallel : `bool`
            true or false

        """
        global _parallel_rgb
        _parallel_rgb = parallel

    def set_backend(self, backend):
        """Set the backend used by prysm.

        Parameters
        ----------
        backend : `str`, {'np'}
            backend used for computations

        Raises
        ------
        ValueError
            invalid backend

        """
        if backend.lower() not in ('np', 'numpy'):
            raise ValueError('Backend must be numpy')

        global _backend
        _backend = 'np'

    def set_zernike_base(self, base):
        """Zernike base; base-0 or base-1.

        Parameters
        ----------
        base : `int`, {0, 1}
            first index of zernike polynomials

        Raises
        ------
        ValueError
            invalid base given

        """
        if base not in (0, 1):
            raise ValueError('By convention zernike base must be 0 or 1.')

        global _zernike_base
        _zernike_base = base

    @property
    def precision(self):
        """Precision used for computations.

        Returns
        -------
        `object`: `numpy.float32` or `numpy.float64`
            precision used

        """
        global _precision
        return _precision

    @property
    def precision_complex(self):
        """Precision used for complex array computations.

        Returns
        -------
        `object`: `numpy.complex64` or `numpy.complex128`
            precision used for complex arrays

        """
        global _precision_complex
        return _precision_complex

    @property
    def parallel_rgb(self):
        """Whether RGB computations are parallelized.

        Returns
        -------
        `bool`
            If RGB computations are parallelized

        """
        global _parallel_rgb
        return _parallel_rgb

    @property
    def backend(self):
        """Backend used.

        Returns
        -------
        `str`
            {'np'} only

        """
        global _backend
        return _backend

    @property
    def zernike_base(self):
        """Zernike base.

        Returns
        -------
        `int`
            {0, 1}

        """
        global _zernike_base
        return _zernike_base


config = Config()
