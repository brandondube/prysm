"""Configuration for this instance of prysm."""
import numpy as np


class Config(object):
    """Global configuration of prysm."""
    def __init__(self,
                 precision=64,
                 backend=np,
                 zernike_base=1,
                 Q=2,
                 phase_colormap='inferno',
                 image_colormap='Greys_r',
                 lw=3,
                 zorder=3):
        """Create a new `Config` object.

        Parameters
        ----------
        precision : `int`
            32 or 64, number of bits of precision
        backend : `str`, {'np'}
            a supported backend.  Current options are only "np" for numpy
        zernike_base : `int`, {0, 1}
            base for zernikes; start at 0 or 1
        Q : `float`
            oversampling parameter for numerical propagations
        phase_colormap : `str`
            colormap used for plotting optical phases
        image_colormap: `str`
            colormap used for plotting greyscale images
        lw : `float`
            linewidth
        `zorder` : int
            z order used for line plots

        """
        self.initialized = False
        self.precision = precision
        self.backend = backend
        self.zernike_base = zernike_base
        self.chbackend_observers = []
        self.Q = Q
        self.phase_colormap = phase_colormap
        self.image_colormap = image_colormap
        self.lw = lw
        self.zorder = zorder
        self.initialized = True

    @property
    def precision(self):
        """Precision used for computations.

        Returns
        -------
        `object` : `numpy.float32` or `numpy.float64`
            precision used

        """
        return self._precision

    @property
    def precision_complex(self):
        """Precision used for complex array computations.

        Returns
        -------
        `object` : `numpy.complex64` or `numpy.complex128`
            precision used for complex arrays

        """
        return self._precision_complex

    @precision.setter
    def precision(self, precision):
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
        if precision not in (32, 64):
            raise ValueError('invalid precision.  Precision should be 32 or 64.')

        if precision == 32:
            self._precision = np.float32
            self._precision_complex = np.complex64
        else:
            self._precision = np.float64
            self._precision_complex = np.complex128

    @property
    def backend(self):
        """Backend used.

        Returns
        -------
        `str`
            {'np'} only

        """
        return self._backend

    @backend.setter
    def backend(self, backend):
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
        if isinstance(backend, str):
            if backend.lower() in ('np', 'numpy'):
                backend = 'numpy'
            elif backend.lower() in ('cp', 'cu', 'cuda'):
                backend = 'cupy'

            exec(f'import {backend}')
            self._backend = eval(backend)
        else:
            self._backend = backend

        if self.initialized:
            for obs in self.chbackend_observers:
                obs(self._backend)

    @property
    def zernike_base(self):
        """Zernike base.

        Returns
        -------
        `int`
            {0, 1}

        """
        return self._zernike_base

    @zernike_base.setter
    def zernike_base(self, base):
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

        self._zernike_base = base


config = Config()
