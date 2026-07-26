"""Configuration for this instance of prysm."""
from numbers import Integral

from .mathops import np


def _coerce_real_dtype(precision):
    """Return a concrete real floating dtype from a dtype-like value."""
    if isinstance(precision, Integral) and not isinstance(precision, bool):
        precision = f'float{precision}'

    try:
        dtype = np.dtype(precision)
    except (TypeError, ValueError) as exc:
        raise ValueError('precision should be a real floating dtype.') from exc

    if dtype.kind != 'f':
        raise ValueError('precision should be a real floating dtype.')

    return dtype.type


def _complex_dtype_for(real_dtype):
    """Return the complex dtype associated with a real floating dtype."""
    return np.result_type(real_dtype, 1j).type


class Config(object):
    """Global configuration of prysm."""
    def __init__(self,
                 precision=64,
                 *,
                 lw=3,
                 zorder=3):
        """Create a new Config object.

        Parameters
        ----------
        precision : dtype-like
            Real floating dtype, or integer bit-depth such as 16, 32, or 64
        lw : float
            linewidth
        zorder : int, optional
            zorder used for graphics made with matplotlib

        """
        self.precision = precision
        self.lw = lw
        self.zorder = zorder

    @property
    def precision(self):
        """Precision used for computations.

        Returns
        -------
        object
            Real floating dtype
            precision used

        """
        return self._precision

    @property
    def precision_complex(self):
        """Precision used for complex array computations.

        Returns
        -------
        object
            Complex dtype associated with precision
            precision used for complex arrays

        """
        return self._precision_complex

    @precision.setter
    def precision(self, precision):
        """Adjust precision used by prysm.

        Parameters
        ----------
        precision : dtype-like
            Real floating dtype, or integer bit-depth such as 16, 32, or 64

        Raises
        ------
        ValueError
            if precision is not a real floating dtype

        """
        self._precision = _coerce_real_dtype(precision)
        self._precision_complex = _complex_dtype_for(self._precision)


config = Config()
