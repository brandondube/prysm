"""Model surface finish errors of optical systems (low-amplitude, random phase errors)."""

import numpy as np

from .pupil import Pupil


class SurfaceFinish(Pupil):
    """A class for adding OPD to a pupil to represent surface finish errors.

    Attributes
    ----------
    amplitude : `float`
        amplitude associated with the surface finish
    fcn : `numpy.ndarray`
        wavefunction of the surface finish
    normalize : `bool`
        whether to normalize amplitude to unit RMS value
    phase : `numpy.ndarray`
        phase error of the surface finish

    """

    def __init__(self, *args, **kwargs):
        """Create a new SurfaceFinish instance.

        Parameters
        ----------
        amplitude : `float`
            amplitude of the surface finish error

        """
        self.normalize = False
        pass_args = {}
        if kwargs is not None:
            for key, value in kwargs.items():
                if key.lower() in ('amplitude', 'amp'):
                    self.amplitude = value
                else:
                    pass_args[key] = value

        super().__init__(**pass_args)

    def build(self):
        """Use the wavefront coefficients stored in this class instance to build a wavefront model.

        Parameters
        ----------
        none

        Returns
        -------
        phase : `numpy.ndarray`
            arrays containing the phase, of the pupil
        fcn : `numpy.ndarray`
            wavefunction for the pupil

        """
        self._gengrid()

        # fill the phase with random, normally distributed values,
        # normalize to unit PV, and scale to appropriate amplitude
        self.phase = np.random.randn(self.samples, self.samples)
        self.phase /= ((self.phase.max() - self.phase.min()) / self.amplitude)

        # convert to units of nm, um, etc
        self._correct_phase_units()
        self.fcn = np.exp(1j * 2 * np.pi / self.wavelength * self.phase)
        return self.phase, self.fcn
