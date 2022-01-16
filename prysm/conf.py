"""Configuration for this instance of prysm."""
from .mathops import np


class Config(object):
    """Global configuration of prysm."""
    def __init__(self,
                 precision=64,
                 phase_cmap='inferno',
                 image_cmap='Greys_r',
                 lw=3,
                 zorder=3,
                 alpha=1,
                 interpolation='lanczos'):
        """Create a new Config object.

        Parameters
        ----------
        precision : int
            32 or 64, number of bits of precision
        phase_cmap : str
            colormap used for plotting optical phases
        image_cmap : str
            colormap used for plotting greyscale images
        lw : float
            linewidth
        zorder : int, optional
            zorder used for graphics made with matplotlib
        alpha : float
            transparency of lines (1=opaque) for graphics made with matplotlib
        interpolation : str
            interpolation type for 2D plots

        """
        self.chbackend_observers = []
        self.precision = precision
        self.phase_cmap = phase_cmap
        self.image_cmap = image_cmap
        self.lw = lw
        self.zorder = zorder
        self.alpha = alpha
        self.interpolation = interpolation

    @property
    def precision(self):
        """Precision used for computations.

        Returns
        -------
        object : numpy.float32 or numpy.float64
            precision used

        """
        return self._precision

    @property
    def precision_complex(self):
        """Precision used for complex array computations.

        Returns
        -------
        object : numpy.complex64 or numpy.complex128
            precision used for complex arrays

        """
        return self._precision_complex

    @precision.setter
    def precision(self, precision):
        """Adjust precision used by prysm.

        Parameters
        ----------
        precision : int, {32, 64}
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


config = Config()
