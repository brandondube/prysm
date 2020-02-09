"""phase basics."""

from .conf import config
from .mathops import engine as e
from ._richdata import RichData
from .plotting import share_fig_ax
from .util import pv, rms, Sa, std


class OpticalPhase(RichData):
    """Phase of an optical field."""
    _data_type = 'phase'

    def __init__(self, x, y, phase, labels, xy_unit=None, z_unit=None, wavelength=None):
        """Create a new instance of an OpticalPhase.

        Note that this class is not intended to be used directly, and is meant
        to allow shared functionality and interchange between the `Pupil` and
        `Interferogram` classes.

        Parameters
        ----------
        x : `numpy.ndarray`
            x unit axis
        y : `numpy.ndarray`
            y unit axis
        phase : `numpy.ndarray`
            phase data
        xlabel : `str`, optional
            x label used on plots
        ylabel : `str`, optional
            y label used on plots
        zlabel : `str`, optional
            z label used on plots
        xyunit : `str`, optional
            unit used for the XY axes
        zunit : `str`, optional
            unit used for the Z (data) axis
        wavelength : `float`, optional
            wavelength of light, in microns

        """

        super().__init__(x=x, y=y, data=phase, labels=labels,
                         xy_unit=xy_unit or config.phase_xy_unit,
                         z_unit=z_unit or config.phase_z_unit,
                         wavelength=wavelength)

    @property
    def pv(self):
        """Peak-to-Valley phase error.  DIN/ISO St."""
        return pv(self.phase)

    @property
    def rms(self):
        """RMS phase error.  DIN/ISO Sq."""
        return rms(self.phase)

    @property
    def Sa(self):
        """Sa phase error.  DIN/ISO Sa."""
        return Sa(self.phase)

    @property
    def std(self):
        """Standard deviation of phase error."""
        return std(self.phase)

    @property
    def diameter_x(self):
        """Diameter of the data in x."""
        return self.x[-1] - self.x[0]

    @property
    def diameter_y(self):
        """Diameter of the data in y."""
        return self.y[-1] - self.x[0]

    @property
    def diameter(self):
        """Greater of (self.diameter_x, self.diameter_y)."""
        return max((self.diameter_x, self.diameter_y))

    @property
    def semidiameter(self):
        """Half of self.diameter."""
        return self.diameter / 2

    @property
    def phase(self):
        """phase is the Z ("height" or "opd") data."""
        return self.data

    @phase.setter
    def phase(self, ary):
        """Set the phase."""
        self.data = ary

    def interferogram(self, visibility=1, passes=2, interpolation=config.interpolation, fig=None, ax=None):
        """Create an interferogram of the `Pupil`.

        Parameters
        ----------
        visibility : `float`
            Visibility of the interferogram
        passes : `float`
            Number of passes (double-pass, quadra-pass, etc.)
        interp_method : `str`, optional
            interpolation method, passed directly to matplotlib
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
        epd = self.diameter
        phase = self.change_z_unit(to='waves', inplace=False)

        fig, ax = share_fig_ax(fig, ax)
        plotdata = visibility * e.cos(2 * e.pi * passes * phase)
        im = ax.imshow(plotdata,
                       extent=[-epd / 2, epd / 2, -epd / 2, epd / 2],
                       cmap='Greys_r',
                       interpolation=interpolation,
                       clim=(-1, 1),
                       origin='lower')
        fig.colorbar(im, label=r'Wrapped Phase [$\lambda$]', ax=ax, fraction=0.046)
        ax.set(xlabel=self.labels.x(self.xy_unit, self.z_unit),
               ylabel=self.labels.y(self.xy_unit, self.z_unit))
        return fig, ax
