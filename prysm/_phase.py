"""phase basics."""

from .conf import config
from .mathops import engine as e
from ._basicdata import BasicData
from .util import share_fig_ax, pv, rms, Sa, std


class OpticalPhase(BasicData):
    """Phase of an optical field."""
    _data_attr = 'phase'
    units = {
        'm': 'm',
        'meter': 'm',
        'mm': 'mm',
        'millimeter': 'mm',
        'μm': 'μm',
        'um': 'μm',
        'micron': 'μm',
        'micrometer': 'μm',
        'nm': 'nm',
        'nanometer': 'nm',
        'Å': 'Å',
        'aa': 'Å',
        'angstrom': 'Å',
        'λ': 'λ',
        'waves': 'λ',
        'lambda': 'λ',
        'px': 'px',
        'pixel': 'px',
    }
    unit_scales = {
        'm': 1,
        'mm': 1e-3,
        'μm': 1e-6,
        'nm': 1e-9,
        'Å': 1e-10,
    }
    unit_changes = {
        'm_m': lambda x: 1,
        'm_mm': lambda x: 1e-3,
        'm_μm': lambda x: 1e-6,
        'm_nm': lambda x: 1e-9,
        'm_Å': lambda x: 1e-10,
        'm_λ': lambda x: 1e-6 * x,
        'mm_mm': lambda x: 1,
        'mm_m': lambda x: 1e3,
        'mm_μm': lambda x: 1e-3,
        'mm_nm': lambda x: 1e-6,
        'mm_Å': lambda x: 1e-7,
        'mm_λ': lambda x: 1e-3 * x,
        'μm_μm': lambda x: 1,
        'μm_m': lambda x: 1e6,
        'μm_mm': lambda x: 1e3,
        'μm_nm': lambda x: 1e-3,
        'μm_Å': lambda x: 1e-4,
        'μm_λ': lambda x: 1 * x,
        'nm_nm': lambda x: 1,
        'nm_m': lambda x: 1e9,
        'nm_mm': lambda x: 1e6,
        'nm_μm': lambda x: 1e3,
        'nm_Å': lambda x: 1e-1,
        'nm_λ': lambda x: 1e3 * x,
        'Å_Å': lambda x: 1,
        'Å_m': lambda x: 1e10,
        'Å_mm': lambda x: 1e7,
        'Å_μm': lambda x: 1e4,
        'Å_nm': lambda x: 10,
        'Å_λ': lambda x: 1e4 * x,
        'λ_λ': lambda x: 1,
        'λ_m': lambda x: 1e6 / x,
        'λ_mm': lambda x: 1e3 / x,
        'λ_μm': lambda x: x,
        'λ_nm': lambda x: 1e-3 / x,
        'λ_Å': lambda x: 1e-4 / x,
        'px_px': lambda x: 1,  # beware changing pixels to other units
        'px_m': lambda x: 1,
        'px_mm': lambda x: 1,
        'px_μm': lambda x: 1,
        'px_nm': lambda x: 1,
        'px_Å': lambda x: 1,
        'px_λ': lambda x: 1,
    }

    def __init__(self, x, y, phase, phase_unit, spatial_unit, wavelength=None):
        """Create a new instance of an OpticalPhase.

        Note that this class is not intended to be used directly, and is meant
        to allow shared functionality and interchange between the `Pupil` and
        `Interferogram` classes.

        Parameters
        ----------
        x : `np.ndarray`
            x spatial units
        y : `np.ndarray`
            y spatial units
        phase : `np.ndarray`
            phase/height/opd data
        phase_unit : `str`
            unit used to describe the phase, see `OpticalPhase`.units
        spatial_unit : `str`
            unit used to describe x and y, see `OpticalPhase`.units
        wavelength : `float`, optional
            wavelength of light, in microns

        """
        self.x = x
        self.y = y
        self.phase = phase
        self.wavelength = wavelength
        self.phase_unit = phase_unit
        self.spatial_unit = spatial_unit
        self.xaxis_label = 'x'  # these should be overriden by subclasses or instances
        self.yaxis_label = 'y'
        self.zaxis_label = 'z'

    @property
    def phase_unit(self):
        """Unit used to describe the optical phase."""
        return self._phase_unit

    @phase_unit.setter
    def phase_unit(self, unit):
        unit = unit.lower()
        if unit == 'å':
            self._phase_unit = unit.upper()
        else:
            if unit not in self.units:
                raise ValueError(f'{unit} not a valid unit, must be in {set(self.units.keys())}')
            self._phase_unit = self.units[unit]

    @property
    def spatial_unit(self):
        """Unit used to describe the spatial phase."""
        return self._spatial_unit

    @spatial_unit.setter
    def spatial_unit(self, unit):
        unit = unit.lower()
        if unit not in self.units:
            raise ValueError(f'{unit} not a valid unit, must be in {set(self.units.keys())}')

        self._spatial_unit = self.units[unit]

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

    def change_phase_unit(self, to, inplace=True):
        """Change the units used to describe the phase.

        Parameters
        ----------
        to : `str`
            new unit, a member of `OpticalPhase`.units.keys()
        inplace : `bool`, optional
            whether to change self.phase, if False, returns updated phase, if True, returns self.

        Returns
        -------
        `new_phase` : `np.ndarray`
            new phase data
        OR
        `self` : `OpticalPhase`
            self

        """
        fctr = self.unit_changes['_'.join([self.phase_unit, self.units[to]])](self.wavelength)
        new_phase = self.phase / fctr
        if inplace:
            self.phase = new_phase
            self.phase_unit = to
            return self
        else:
            return new_phase

    def change_spatial_unit(self, to, inplace=True):
        """Change the units used to describe the spatial dimensions.

        Parameters
        ----------
        to : `str`
            new unit, a member of `OpticalPhase`.units.keys()
        inplace : `bool`, optional
            whether to change self.x and self.y.
            If False, returns updated phase, if True, returns self

        Returns
        -------
        `new_ux` : `np.ndarray`
            new ordinate x axis
        `new_uy` : `np.ndarray`
            new ordinate y axis
        OR
        `self` : `OpticalPhase`
            self

        """
        if to.lower() != 'px':
            fctr = self.unit_changes['_'.join([self.spatial_unit, self.units[to]])](self.wavelength)
            new_ux = self.x / fctr
            new_uy = self.y / fctr
        else:
            sy, sx = self.shape
            new_ux = e.arange(sx, dtype=config.precision)
            new_uy = e.arange(sy, dtype=config.precision)
        if inplace:
            self.x = new_ux
            self.y = new_uy
            self.spatial_unit = to
            return self
        else:
            return new_ux, new_uy

    def plot2d(self,
               cmap=config.phase_colormap,
               clim=(None, None),
               interp_method='lanczos',
               show_colorbar=True,
               fig=None,
               ax=None):
        """Plot the phase in 2D.

        Parameters
        ----------
        cmap : `str`
            colormap to use, passed directly to matplotlib
        interp_method : `str`, optional
            interpolation method to use, passed directly to matplotlib
        show_colorbar : `bool`, optional
            whether to draw the colorbar
        fig : `matplotlib.figure.Figure`
            Figure containing the plot
        ax : `matplotlib.axes.Axis`
            Axis containing the plot

        Returns
        -------
        fig : `matplotlib.figure.Figure`
            Figure containing the plot
        ax : `matplotlib.axes.Axis`
            Axis containing the plot

        """
        fig, ax = share_fig_ax(fig, ax)

        if clim and not hasattr(clim, '__iter__'):
            clim = (-clim, clim)

        im = ax.imshow(self.phase,
                       extent=[self.x[0], self.x[-1], self.y[0], self.y[-1]],
                       cmap=cmap,
                       clim=clim,
                       origin='lower',
                       interpolation=interp_method)

        if show_colorbar:
            fig.colorbar(im, label=f'{self.zaxis_label} [{self.phase_unit}]', ax=ax, fraction=0.046)

        xlab = f'{self.xaxis_label} [{self.spatial_unit}]'
        ylab = f'{self.yaxis_label} [{self.spatial_unit}]'
        ax.set(xlabel=xlab, ylabel=ylab)
        return fig, ax

    def plot_slice_xy(self, lw=config.lw, zorder=config.zorder, fig=None, ax=None):
        """Create a plot of slices through the X and Y axes of the `Pupil`.

        Parameters
        ----------
        lw : `float`, optional
            line width
        zorder : `int`, optional
            zorder
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
        ux, x = self.slice_x
        uy, y = self.slice_y

        fig, ax = share_fig_ax(fig, ax)

        ax.plot(ux, x, lw=lw, zorder=zorder, label='X')
        ax.plot(uy, y, lw=lw, zorder=zorder, label='Y')
        ax.set(xlabel=f'{self.xaxis_label} [{self.spatial_unit}]',
               ylabel=f'{self.zaxis_label} [{self.phase_unit}]')
        ax.legend(title='Slice')
        return fig, ax

    def interferogram(self, visibility=1, passes=2, interp_method='lanczos', fig=None, ax=None):
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
        phase = self.change_phase_unit(to='waves', inplace=False)

        fig, ax = share_fig_ax(fig, ax)
        plotdata = (visibility * e.sin(2 * e.pi * passes * phase))
        im = ax.imshow(plotdata,
                       extent=[-epd / 2, epd / 2, -epd / 2, epd / 2],
                       cmap='Greys_r',
                       interpolation=interp_method,
                       clim=(-1, 1),
                       origin='lower')
        fig.colorbar(im, label=r'Wrapped Phase [$\lambda$]', ax=ax, fraction=0.046)
        ax.set(xlabel=r'Pupil $\xi$ [mm]',
               ylabel=r'Pupil $\eta$ [mm]')
        return fig, ax
