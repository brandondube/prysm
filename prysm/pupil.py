"""A base pupil interface for different aberration models."""

from .conf import config
from .mathops import engine as e
from ._phase import OpticalPhase
from .coordinates import make_rho_phi_grid
from .geometry import mcache
from .util import std


class Pupil(OpticalPhase):
    """Pupil of an optical system."""
    def __init__(self, samples=128, dia=1.0, wavelength=0.55, opd_unit='waves',
                 mask='circle', mask_target='both', ux=None, uy=None, phase=None):
        """Create a new `Pupil` instance.

        Parameters
        ----------
        samples : `int`, optional
            number of samples across the pupil interior
        dia : `float`, optional
            diameter of the pupil, mm
        wavelength : `float`, optional
            wavelength of light, um
        opd_unit : `str`, optional, {'waves', 'um', 'nm'}
            unit used to m.express the OPD.  Equivalent strings may be used to the
            valid options, e.g. 'microns', or 'nanometers'
        mask : `str` or `numpy.ndarray`
            mask used to define the amplitude and boundary of the pupil; any
            regular polygon from `prysm.geometry` as a string, e.g. 'circle' is
            valid.  A user-provided ndarray can also be used.
        mask_target : `str`, {'phase', 'fcn', 'both', None}
            which array to mask during pupil creation; only masking fcn is
            faster for numerical propagations but will make plot2d() and the
            phase array not be truncated properly.  Note that if the mask is not
            binary and `phase` or `both` is used, phase plots will also not be
            correct, as they will be attenuated by the mask.
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

        Raises
        ------
        ValueError
            if the OPD unit given is invalid

        """
        if ux is None:
            # must build a pupil
            self.dia = dia
            ux = e.linspace(-dia / 2, dia / 2, samples)
            uy = e.linspace(-dia / 2, dia / 2, samples)
            self.samples = samples
            need_to_build = True
        else:
            # data already known
            need_to_build = False
        super().__init__(x=ux, y=uy, phase=phase,
                         wavelength=wavelength, phase_unit=opd_unit, spatial_unit='mm')
        self.xaxis_label = 'Pupil ξ'
        self.yaxis_label = 'Pupil η'
        self.zaxis_label = 'OPD'
        self.rho = self.phi = None

        if need_to_build:
            if type(mask) is not e.ndarray:
                mask = mcache(mask, self.samples)

            self._mask = mask
            self.mask_target = mask_target
            self.build()
            self.mask(self._mask, self.mask_target)
        else:
            protomask = e.isnan(phase)
            mask = e.ones(protomask.shape)
            mask[protomask] = 0
            self._mask = mask
            self.mask_target = 'fcn'

    @property
    def strehl(self):
        """Strehl ratio of the pupil."""
        phase = self.change_phase_unit(to='um', inplace=False)
        return e.exp(-4 * e.pi / self.wavelength / self.wavelength * std(phase) ** 2)

    @property
    def fcn(self):
        """Complex wavefunction associated with the pupil."""
        phase = self.change_phase_unit(to='waves', inplace=False)

        fcn = e.exp(1j * 2 * e.pi * phase)  # phase implicitly in units of waves, no 2pi/l
        # guard against nans in phase
        fcn[e.isnan(phase)] = 0

        if self.mask_target in ('fcn', 'both'):
            fcn *= self._mask

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

    def mask(self, mask, target, nanify=True):
        """Apply a mask to the pupil.

        Used to implement vignetting, chief ray angles, etc.

        Parameters
        ----------
        mask : `str` or `numpy.ndarray`
            if a string, uses geometry.mcache for high speed access to a mask with a given shape,
            e.g. mask='circle' or mask='hexagon'.  If an ndarray, directly use the mask.
        target : `str`, {'phase', 'fcn', 'both'}
            which array to mask
        nanify: `bool`, optional
            if True, make (target) equal to NaN where the mask is zero.

        Returns
        -------
        `Pupil`
            self, the pupil instance

        """
        if target in ('phase', 'both'):
            self.phase *= mask
            if nanify:
                nans = mask == 0
                self.phase[nans] = e.nan

        self._mask = mask
        return self

    def _gengrid(self):
        """Generate a uniform (x,y) grid and maps it to (rho,phi) coordinates for radial polynomials.

        Note
        ----
        angle is done via cart_to_polar(yv, xv) which yields angles w.r.t.
        the y axis.  This is the convention of optics and not a typo.

        Returns
        -------
        self.rho : `numpy.ndarray`
            the radial coordinate of the pupil coordinate grid
        self.phi : `numpy.ndarray`
            the azimuthal coordinate of the pupil coordinate grid

        """
        self.rho, self.phi = make_rho_phi_grid(self.samples, aligned='y')
        return self.rho, self.phi

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
