"""A base pupil interface for different aberration models."""

from .conf import config
from .mathops import engine as e
from ._phase import OpticalPhase
from .geometry import mask_cleaner
from .util import std


class Pupil(OpticalPhase):
    """Pupil of an optical system."""
    def __init__(self, samples=128, dia=1, units=None, labels=None,
                 phase_mask='circle', transmission='circle', ux=None, uy=None, phase=None):
        """Create a new `Pupil` instance.

        Parameters
        ----------
        samples : `int`, optional
            number of samples across the pupil interior
        dia : `float`, optional
            diameter of the pupil, units.x
        units : `Units`
            units for the data
        labels : `Labels`
            labels used for plots
        phase_mask : `numpy.ndarray` or `str` or `tuple`
            Mask used to modify phase.
            If array, used directly.
            If str, assumed to be known mask type for geometry submodule.
            If tuple, assumed to be known mask type, with optional radius.
        transmission : `numpy.ndarray` or `str` or `tuple`
            Mask used to modify `self.fcn`.
            If array, used directly.
            If str, assumed to be known mask type for geometry submodule.
            If tuple, assumed to be known mask type, with optional radius.
        ux : `np.ndarray`
            x axis units
        uy : `np.ndarray`
            y axis units
        phase : `np.ndarray`
            phase data

        Notes
        -----
        If ux give, assume uy and phase also given; skip much of the pupil building process
        and simply copy values.

        """
        if ux is None:
            # must build a pupil
            ux = e.linspace(-dia / 2, dia / 2, samples)
            uy = e.linspace(-dia / 2, dia / 2, samples)
            self.samples = samples
            need_to_build = True
        else:
            # data already known
            need_to_build = False

        if units is None:
            units = config.phase_units

        if labels is None:
            labels = config.pupil_labels

        super().__init__(x=ux, y=uy, phase=phase, units=units, labels=labels)

        phase_mask = mask_cleaner(phase_mask, samples)

        if need_to_build:
            self.build()
            if phase_mask is not None:
                self.phase = self.phase * phase_mask
                self.phase[phase_mask == 0] = e.nan

            transmission = mask_cleaner(transmission, samples)
            self.transmission = transmission
            self.phase_mask = phase_mask
        else:
            holes = e.isnan(phase)
            transmission = e.ones(holes.shape)
            transmission[holes] = 0
            self.transmission = transmission
            self.phase_mask = phase_mask

    @property
    def strehl(self):
        """Strehl ratio of the pupil."""
        phase = self.change_z_unit(to='um', inplace=False)
        return e.exp(-4 * e.pi / self.wavelength / self.wavelength * std(phase) ** 2)

    @property
    def fcn(self):
        """Complex wavefunction associated with the pupil."""
        phase = self.change_z_unit(to='waves', inplace=False)

        fcn = e.exp(1j * 2 * e.pi * phase)  # phase implicitly in units of waves, no 2pi/l
        # guard against nans in phase
        if self.phase_mask is not None:
            fcn[e.isnan(phase)] = 0

        if self.transmission is not None:
            fcn *= self.transmission

        return fcn

    def build(self):
        """Construct a numerical model of a `Pupil`.

        The method should be overloaded by all subclasses to impart their unique
        mathematical models to the simulation.

        Returns
        -------
        `Pupil`
            this pupil instance

        """
        # build up the pupil
        self._gengrid()

        # fill in the phase of the pupil
        self.phase = e.zeros((self.samples, self.samples), dtype=config.precision)

        return self

    def __add__(self, other):
        """Sum the phase of two pupils.

        Parameters
        ----------
        other : `Pupil`
            pupil to add to this one

        Returns
        -------
        `Pupil`
            new Pupil object

        Raises
        ------
        ValueError
            if the two pupils are not identically sampled

        """
        if self.sample_spacing != other.sample_spacing or self.samples != other.samples:
            raise ValueError('Pupils must be identically sampled')

        result = self.copy()
        result.phase = self.phase + other.phase
        result._mask = self._mask * other._mask
        result.mask(result._mask, result.mask_target)
        return result

    def __sub__(self, other):
        """Compute the phase difference of two pupils.

        Parameters
        ----------
        other : `Pupil`
            pupil to add to this one

        Returns
        -------
        `Pupil`
            new Pupil object

        Raises
        ------
        ValueError
            if the two pupils are not identically sampled

        """
        if self.sample_spacing != other.sample_spacing or self.samples != other.samples:
            raise ValueError('Pupils must be identically sampled')

        result = self.copy()
        result.phase = self.phase - other.phase
        result._mask = self._mask * other._mask
        result.mask(result._mask, result.mask_target)
        return result

    @staticmethod
    def from_interferogram(interferogram, wvl=None):
        """Create a new Pupil instance from an interferogram.

        Parameters
        ----------
        interferogram : `Interferogram`
            an interferogram object
        wvl : `float`, optional
            wavelength of light, in micrometers, if not present in interferogram.meta

        Returns
        -------
        `Pupil`
            new Pupil instance

        Raises
        ------
        ValueError
            wavelength not present

        """
        if wvl is None:  # not user specified
            wvl = interferogram.wavelength

        return Pupil(wavelength=wvl, phase=interferogram.phase,
                     opd_unit=interferogram.phase_unit,
                     ux=interferogram.x, uy=interferogram.y,
                     mask=~(interferogram.phase == e.nan))
