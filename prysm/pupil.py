'''A base pupil interface for different aberration models.
'''
from copy import deepcopy

from .conf import config
from .util import share_fig_ax, rms
from .coordinates import make_rho_phi_grid
from .geometry import mcache

from prysm import mathops as m


class Pupil(object):
    """Pupil of an optical system.

    Properties
    ----------
    slice_x: slice through the x axis of the pupil.  Returns (x,y)
             data where x is the sample coordinate and y is the phase.
    slice_y: slice through the y axis of the pupil.  Returns (x,y)
             data where x is the sample coordinate and y is the phase.

    pv: Peak-To-Valley wavefront error.

    rms: Root Mean Square wavefront error.

    Notes
    -----
    subclasses should implement a build() method and their own way of
    expressing OPD.

    Attributes
    ----------
    center : `int`
        index of the center sample, may be sheared by 1/2 for even sample counts
    epd : `float`
        entrance pupil diameter, mm
    fcn : `numpy.ndarray`
        wavefunction, complex 2D array
    opd_unit : `str`
        unit used to m.express phase errors
    phase : `numpy.ndarray`
        phase, real 2D array
    rho : `numpy.ndarray`
        radial ordinate axis, normalized units
    sample_spacing : `float`
        spacing of samples, mm
    samples : `int`
        number of samples across the pupil diameter
    unit : `numpy.ndarray`
        1D array which gives the sample locations across the 2D pupil region
    wavelength : `float`
        wavelength of light, um
    _mask : `numpy.ndarray`
        mask used to define pupil extent and amplitude
    mask_target : `str`
        target for automatic masking on pupil creation

    """
    def __init__(self, samples=128, epd=1.0, wavelength=0.55, opd_unit=r'$\lambda$',
                 mask='circle', mask_target='both'):
        """Create a new `Pupil` instance.

        Parameters
        ----------
        samples : int, optional
            number of samples across the pupil interior
        epd : float, optional
            diameter of the pupil, mm
        wavelength : float, optional
            wavelength of light, um
        opd_unit : str, optional, {'waves', 'um', 'nm'}
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

        Raises
        ------
        ValueError
            if the OPD unit given is invalid

        """
        self.samples = samples
        self.epd = epd
        self.wavelength = wavelength
        self.opd_unit = opd_unit
        self.phase = self.fcn = m.empty((samples, samples), dtype=config.precision)
        self.unit = m.linspace(-epd / 2, epd / 2, samples, dtype=config.precision)
        self.sample_spacing = self.unit[-1] - self.unit[-2]
        self.rho = self.phi = m.empty((samples, samples), dtype=config.precision)
        self.center = samples // 2

        if opd_unit.lower() in ('$\lambda$', 'waves'):
            self._opd_unit = 'waves'
            self._opd_str = '$\lambda$'
        elif opd_unit.lower() in ('$\mu m$', 'microns', 'micrometers', 'um'):
            self._opd_unit = 'microns'
            self._opd_str = '$\mu m$'
        elif opd_unit.lower() in ('nm', 'nanometers'):
            self._opd_unit = 'nanometers'
            self._opd_str = 'nm'
        else:
            raise ValueError('OPD must be m.expressed in waves, microns, or nm')

        if type(mask) is not m.ndarray:
            mask = mcache.get_mask(self.samples, mask)

        self._mask = mask
        self.mask_target = mask_target

        self.build()
        self.mask(self._mask, self.mask_target)

    # quick-access slices, properties ------------------------------------------

    @property
    def slice_x(self):
        """Retrieve a slice through the X axis of the `Pupil`.

        Returns
        -------
        self.unit : `numpy.ndarray`
            ordinate axis
        slice of self.phase : `numpy.ndarray`

        """
        return self.unit, self.phase[self.center]

    @property
    def slice_y(self):
        """Retrieve a slice through the Y axis of the `Pupil`.

        Returns
        -------
        self.unit : `numpy.ndarray`
            ordinate axis
        slice of self.phase : `numpy.ndarray`

        """
        return self.unit, self.phase[:, self.center]

    @property
    def pv(self):
        """Return the peak-to-valley wavefront error as a `float`.

        """
        non_nan = m.isfinite(self.phase)
        return self.phase[non_nan].max() - self.phase[non_nan].min()

    @property
    def rms(self):
        """Return the RMS wavefront error in the given OPD units as a `float`.

        """
        return rms(self.phase)

    # quick-access slices, properties ------------------------------------------

    # plotting -----------------------------------------------------------------

    def plot2d(self, fig=None, ax=None):
        """Create a 2D plot of the phase error of the pupil.

        Parameters
        ----------
        fig : `matplotlib.figure.Figure`, optional:
            Figure to draw plot in
        ax : `matplotlib.axes.Axis`
            Axis to draw plot in

        Returns
        -------
        fig : `matplotlib.figure.Figure`
            Figure containing the plot
        ax : `matplotlib.axes.Axis`
            Axis containing the plot

        """
        epd = self.epd

        fig, ax = share_fig_ax(fig, ax)
        im = ax.imshow(self.phase,
                       extent=[-epd / 2, epd / 2, -epd / 2, epd / 2],
                       cmap='RdYlBu',
                       interpolation='lanczos',
                       origin='lower')
        cb = fig.colorbar(im, label=f'OPD [{self._opd_str}]', ax=ax, fraction=0.046)
        cb.outline.set_edgecolor('k')
        cb.outline.set_linewidth(0.5)
        ax.set(xlabel=r'Pupil $\xi$ [mm]',
               ylabel=r'Pupil $\eta$ [mm]')
        return fig, ax

    def plot_slice_xy(self, fig=None, ax=None):
        """Create a plot of slices through the X and Y axes of the `Pupil`.

        Parameters
        ----------
        fig : `matplotlib.figure.Figure`, optional
            Figure to draw plot in
        ax : `matplotlib.axes.Axis`
            Axis to draw plot in

        Returns
        -------
        fig : `matplotlib.figure.Figure`, optional
            Figure containing the plot
        ax : `matplotlib.axes.Axis`, optional:
            Axis containing the plot

        """
        u, x = self.slice_x
        _, y = self.slice_y

        fig, ax = share_fig_ax(fig, ax)

        x = convert_phase(x, self)
        y = convert_phase(y, self)

        ax.plot(u, x, lw=3, label='Slice X')
        ax.plot(u, y, lw=3, label='Slice Y')
        ax.set(xlabel=r'Pupil $\rho$ [mm]',
               ylabel=f'OPD [{self._opd_str}]')
        ax.legend()
        return fig, ax

    def interferogram(self, visibility=1, passes=2, fig=None, ax=None):
        """Create an interferogram of the `Pupil`.

        Parameters
        ----------
        visibility : `float`
            Visibility of the interferogram
        passes : `float`
            Number of passes (double-pass, quadra-pass, etc.)
        fig : `matplotlib.figure.Figure`, optional
            Figure to draw plot in
        ax : `matplotlib.axes.Axis`
            Axis to draw plot in

        Returns
        -------
        fig : `matplotlib.figure.Figure`, optional
            Figure containing the plot
        ax : `matplotlib.axes.Axis`, optional:
            Axis containing the plot

        """
        epd = self.epd

        fig, ax = share_fig_ax(fig, ax)
        plotdata = (visibility * m.sin(2 * m.pi * passes * self.phase))
        im = ax.imshow(plotdata,
                       extent=[-epd / 2, epd / 2, -epd / 2, epd / 2],
                       cmap='Greys_r',
                       interpolation='lanczos',
                       clim=(-1, 1),
                       origin='lower')
        fig.colorbar(im, label=r'Wrapped Phase [$\lambda$]', ax=ax, fraction=0.046)
        ax.set(xlabel=r'Pupil $\xi$ [mm]',
               ylabel=r'Pupil $\eta$ [mm]')
        return fig, ax

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
        self.phase = zeros((self.samples, self.samples), dtype=config.precision)
        self._phase_to_wavefunction()

        return self

    def _phase_to_wavefunction(self):
        """Compute the wavefunction from the phase.

        Returns
        -------
        `Pupil`
            this pupil instance

        """
        phase = convert_phase(self.phase, self)

        # guard against nans in phase
        nans = ~m.isfinite(phase)
        if m.any(nans):
            phase[nans] = 0

        self.fcn = m.exp(1j * 2 * m.pi * phase)  # phase implicitly in units of waves, no 2pi/l
        return self

    def mask(self, mask, target):
        """Apply a mask to the pupil.

        Used to implement vignetting, chief ray angles, etc.

        Parameters
        ----------
        mask : `str` or `numpy.ndarray`
            if a string, uses geometry.mcache for high speed access to a mask with a given shape,
            e.g. mask='circle' or mask='hexagon'.  If an ndarray, directly use the mask.
        target : `str`, {'phase', 'fcn', 'both'}
            which array to mask;

        Returns
        -------
        `Pupil`
            self, the pupil instance

        """
        tl = target.lower()
        if tl == 'both':
            idx = mask == 0
            self.phase[idx] = m.nan
            self.fcn *= mask
        elif tl == 'phase':
            idx = mask == 0
            self.phase[idx] = m.nan
        else:
            self.fcn *= mask

        return self

    def clone(self):
        """Create a copy of this pupil.

        Returns
        -------
        `Pupil`
            a deep copy duplicate of this pupil

        """
        props = deepcopy(self.__dict__)
        retpupil = Pupil()
        retpupil.__dict__ = props
        return retpupil

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

    def _correct_phase_units(self):
        """Convert an expression of OPD in a unit to waves.

        Returns
        -------
        `Pupil`
            this pupil instance

        """
        return convert_phase(self.phase, self)

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

        result = self.clone()
        result.phase = self.phase + other.phase
        result._mask = self._mask * other._mask
        result = result._phase_to_wavefunction()
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

        result = self.clone()
        result.phase = self.phase - other.phase
        result._mask = self._mask * other._mask
        result = result._phase_to_wavefunction()
        result.mask(result._mask, result.mask_target)
        return result


def convert_phase(array, pupil):
    """Convert an OPD/phase map to have the same unit of expression as a pupil.

    Parameters
    ----------
    array : `numpy.ndarray` or `float`
        array of phase data
    pupil : `Pupil`
        a pupil to match the phase units to

    Returns
    -------
    `numpy.ndarray`
        phase-corrected array.

    """
    if pupil._opd_unit == 'microns':
        return array / pupil.wavelength
    elif pupil._opd_unit == 'nanometers':
        return array / pupil.wavelength / 1e3
    else:
        return array
