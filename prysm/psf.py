"""A base point spread function interface."""
from scipy import optimize

from matplotlib import colors, patches

from .coordinates import cart_to_polar
from .util import share_fig_ax, sort_xy
from .convolution import Convolvable
from .propagation import (
    prop_pupil_plane_to_psf_plane,
    prop_pupil_plane_to_psf_plane_units,
)

from prysm import mathops as m


FIRST_AIRY_ENCIRCLED = 0.8377850436212378


class PSF(Convolvable):
    """A Point Spread Function.

    Attributes
    ----------
    center_x : `int`
        center sample along x
    center_y : `int`
        center sample along y
    data : `numpy.ndarray`
        PSF normalized intensity data
    sample_spacing : `float`
        center to center spacing of samples
    unit_x : `numpy.ndarray`
        x Cartesian axis locations of samples, 1D ndarray
    unit_y `numpy.ndarray`
        y Cartesian axis locations of samples, 1D ndarray

    """
    def __init__(self, data, unit_x, unit_y):
        """Create a PSF object.

        Parameters
        ----------
        data : `numpy.ndarray`
            intensity data for the PSF
        unit_x : `numpy.ndarray`
            1D ndarray defining x data grid
        unit_y  : `numpy.ndarray`
            1D ndarray defining y data grid
        sample_spacing : `float`
            center-to-center spacing of samples, expressed in microns

        """
        super().__init__(data, unit_x, unit_y, has_analytic_ft=False)
        self.data /= self.data.max()
        self._ee = {}
        self._mtf = None
        self._nu_p = None
        self._dnx = None
        self._dny = None

    def encircled_energy(self, radius):
        """Compute the encircled energy of the PSF.

        Parameters
        ----------
        radius : `float` or iterable
            radius or radii to evaluate encircled energy at

        Returns
        -------
        encircled energy
            if radius is a float, returns a float, else returns a list.

        Notes
        -----
        implementation of "Simplified Method for Calculating Encircled Energy,"
        Baliga, J. V. and Cohn, B. D., doi: 10.1117/12.944334

        """
        from .otf import MTF

        if hasattr(radius, '__iter__'):
            # user wants multiple points
            # um to mm, cy/mm assumed in Fourier plane
            radius_is_array = True
        else:
            radius_is_array = False

        # compute MTF from the PSF
        if self._mtf is None:
            self._mtf = MTF.from_psf(self)
            nx, ny = m.meshgrid(self._mtf.unit_x, self._mtf.unit_y)
            self._nu_p = m.sqrt(nx ** 2 + ny ** 2)
            self._dnx, self._dny = ny[1, 0] - ny[0, 0], nx[0, 1] - nx[0, 0]

        if radius_is_array:
            out = []
            for r in radius:
                if r not in self._ee:
                    self._ee[r] = _encircled_energy_core(self._mtf.data,
                                                         r / 1e3,
                                                         self._nu_p,
                                                         self._dnx,
                                                         self._dny)
                out.append(self._ee[r])
            return m.asarray(out)
        else:
            if radius not in self._ee:
                self._ee[radius] = _encircled_energy_core(self._mtf.data,
                                                          radius / 1e3,
                                                          self._nu_p,
                                                          self._dnx,
                                                          self._dny)
            return self._ee[radius]

    def ee_radius(self, energy=FIRST_AIRY_ENCIRCLED):

        k, v = list(self._ee.keys()), list(self._ee.values())
        if energy in v:
            idx = v.index(energy)
            return k[idx]

        def optfcn(x):
            return abs(self.encircled_energy(x) - energy)

        # golden seems to perform best in presence of shallow local minima as in
        # the encircled energy
        return optimize.golden(optfcn)

    def ee_radius_diffraction(self, energy=FIRST_AIRY_ENCIRCLED):
        return _inverse_analytic_encircled_energy(self.fno, self.wavelength, energy)

    def ee_radius_ratio_to_diffraction(self, energy=FIRST_AIRY_ENCIRCLED):
        self_rad = self.ee_radius(energy)
        diff_rad = _inverse_analytic_encircled_energy(self.fno, self.wavelength, energy)
        return self_rad / diff_rad

    # plotting -----------------------------------------------------------------

    def plot2d(self, axlim=25, power=1, clim=(None, None), interp_method='lanczos',
               pix_grid=None, invert=False, fig=None, ax=None,
               show_axlabels=True, show_colorbar=True,
               circle_ee=None, circle_ee_lw=None):
        """Create a 2D plot of the PSF.

        Parameters
        ----------
        axlim : `float`
            limits of axis, symmetric. xlim=(-axlim,axlim), ylim=(-axlim, axlim)
        power : `float`
            power to stretch the data by for plotting
        clim : iterable
            limits to use for log color scaling.  If power != 1 and
            clim != (None, None), clim (log axes) takes precedence
        interp_method : `string`
            method used to interpolate the image between samples of the PSF
        pix_grid : `float`
            if not None, overlays gridlines with spacing equal to pix_grid.
            Intended to show the collection into camera pixels while still in
            the oversampled domain
        invert : `bool`, optional
            whether to invert the color scale
        fig : `matplotlib.figure.Figure`, optional:
            Figure containing the plot
        ax : `matplotlib.axes.Axis`, optional:
            Axis containing the plot
        show_axlabels : `bool`
            whether or not to show the axis labels
        show_colorbar : `bool`
            whether or not to show the colorbar
        circle_ee : `float`, optional
            relative encircled energy to draw a circle at, in addition to
            diffraction limited airy radius (1.22*λ*F#).  First airy zero occurs
            at circle_ee=0.8377850436212378
        circle_ee_lw : `float`, optional
            linewidth passed to matplotlib for the encircled energy circles

        Returns
        -------
        fig : `matplotlib.figure.Figure`, optional
            Figure containing the plot
        ax : `matplotlib.axes.Axis`, optional
            Axis containing the plot

        """
        label_str = 'Normalized Intensity [a.u.]'

        left, right = self.unit_x[0], self.unit_x[-1]
        bottom, top = self.unit_y[0], self.unit_y[-1]

        if invert:
            cmap = 'Greys'
        else:
            cmap = 'Greys_r'

        fig, ax = share_fig_ax(fig, ax)

        plt_opts = {
            'extent': [left, right, bottom, top],
            'origin': 'lower',
            'cmap': cmap,
            'interpolation': interp_method,
        }
        cb_opts = {}
        if power is not 1:
            plt_opts['norm'] = colors.PowerNorm(1/power)
            plt_opts['clim'] = (0, 1)
        elif clim[1] is not None:
            plt_opts['norm'] = colors.LogNorm(*clim)
            cb_opts = {'extend': 'both'}

        im = ax.imshow(self.data, **plt_opts)
        if show_colorbar:
            cb = fig.colorbar(im, label=label_str, ax=ax, fraction=0.046, **cb_opts)
            cb.outline.set_edgecolor('k')
            cb.outline.set_linewidth(0.5)
        if show_axlabels:
            ax.set(xlabel=r'Image Plane $x$ [$\mu m$]',
                   ylabel=r'Image Plane $y$ [$\mu m$]')

        ax.set(xlim=(-axlim, axlim),
               ylim=(-axlim, axlim))

        if pix_grid is not None:
            # if pixel grid is desired, add it
            mult = m.floor(axlim / pix_grid)
            gmin, gmax = -mult * pix_grid, mult * pix_grid
            pts = m.arange(gmin, gmax, pix_grid)
            ax.set_yticks(pts, minor=True)
            ax.set_xticks(pts, minor=True)
            ax.yaxis.grid(True, which='minor', color='white', alpha=0.25)
            ax.xaxis.grid(True, which='minor', color='white', alpha=0.25)

        if circle_ee is not None:
            if self.fno is None:
                raise ValueError('F/# must be known to compute EE, set self.fno')
            elif self.wavelength is None:
                raise ValueError('wavelength must be known to compute EE, set self.wavelength')

            radius = self.ee_radius(circle_ee)
            analytic = _inverse_analytic_encircled_energy(self.fno, self.wavelength, circle_ee)

            c_diff = patches.Circle((0, 0), analytic, fill=False, color='r', ls='--', lw=circle_ee_lw)
            c_true = patches.Circle((0, 0), radius, fill=False, color='r', lw=circle_ee_lw)
            ax.add_artist(c_diff)
            ax.add_artist(c_true)
            ax.legend([c_diff, c_true], ['Diff. Lim.', 'Actual'], ncol=2)

        return fig, ax

    def plot_encircled_energy(self, axlim=None, npts=50, lw=3, fig=None, ax=None):
        """Make a 1D plot of the encircled energy at the given azimuth.

        Parameters
        ----------
        azimuth : `float`
            azimuth to plot at, in degrees
        axlim : `float`
            limits of axis, will plot [0, axlim]
        npts : `int`, optional
            number of points to use from [0, axlim]
        lw : `float`, optional
            linewidth provided directly to matplotlib
        fig : `matplotlib.figure.Figure`, optional
            Figure containing the plot
        ax : `matplotlib.axes.Axis`, optional:
            Axis containing the plot

        Returns
        -------
        fig : `matplotlib.figure.Figure`, optional
            Figure containing the plot
        ax : `matplotlib.axes.Axis`, optional:
            Axis containing the plot

        """
        if axlim is None:
            if len(self._ee) is not 0:
                xx, yy = sort_xy(self._ee.keys(), self._ee.values())
            else:
                raise ValueError('if no values for encircled energy have been computed, axlim must be provided')
        elif axlim is 0:
            raise ValueError('computing from 0 to 0 is stupid')
        else:
            xx = m.linspace(1e-5, axlim, npts)
            yy = self.encircled_energy(xx)

        fig, ax = share_fig_ax(fig, ax)
        ax.plot(xx, yy, lw=3)
        ax.set(xlabel=r'Image Plane Distance [$\mu m$]',
               ylabel=r'Encircled Energy [Rel 1.0]',
               xlim=(0, axlim))
        return fig, ax

    # plotting -----------------------------------------------------------------

    # helpers ------------------------------------------------------------------

    def _renorm(self, to='peak'):
        """Renormalize the PSF to unit peak intensity.

        Parameters
        ----------
        to : `string`, {'peak', 'total'}
            renormalization target; produces a PSF of unit peak or total intensity

        Returns
        -------
        `PSF`
            a renormalized PSF instance

        """
        if to.lower() == 'peak':
            self.data /= self.data.max()
        elif to.lower() == 'total':
            ttl = self.data.sum()
            self.data /= ttl
        return self

    # helpers ------------------------------------------------------------------

    @staticmethod
    def from_pupil(pupil, efl, Q=2):
        """Use scalar diffraction propogation to generate a PSF from a pupil.

        Parameters
        ----------
        pupil : `Pupil`
            Pupil, with OPD data and wavefunction
        efl : `int` or `float`
            effective focal length of the optical system
        Q : `int` or `float`
            ratio of pupil sample count to PSF sample count; Q > 2 satisfies nyquist

        Returns
        -------
        `PSF`
            A new PSF instance

        """
        # propagate PSF data
        fcn, ss, wvl = pupil.fcn, pupil.sample_spacing, pupil.wavelength
        data = prop_pupil_plane_to_psf_plane(fcn, Q)
        ux, uy = prop_pupil_plane_to_psf_plane_units(fcn, ss, efl, wvl, Q)
        psf = PSF(data, ux, uy)

        # determine the F/#, assumes:
        # - pupil fills x or y width of array
        # - pupil is not elliptical at an odd angle
        s = fcn.shape
        if s[1] > s[0]:
            u = pupil.unit_x
        else:
            u = pupil.unit_y

        epd = u[-1] - u[0]

        psf.fno = efl / epd
        psf.wavelength = wvl
        return psf


class AiryDisk(PSF):
    """An airy disk, the PSF of a circular aperture."""
    def __init__(self, fno, wavelength, extent, samples):
        """Create a new AiryDisk.

        Parameters
        ----------
        fno : `float`
            F/# associated with the PSF
        wavelength : `float`
            wavelength of light, in microns
        extent : `float`
            cartesian window half-width, e.g. 10 will make an RoI 20x20 microns wide
        samples : `int`
            number of samples across full width

        """
        x = m.linspace(-extent, extent, samples)
        y = m.linspace(-extent, extent, samples)
        xx, yy = m.meshgrid(x, y)
        rho, phi = cart_to_polar(xx, yy)
        data = _airydisk(rho, fno, wavelength)
        self.fno = fno
        self.wavelength = wavelength
        super().__init__(data, x, y)
        self.has_analytic_ft = True

    def analytic_ft(self, unit_x, unit_y):
        """Analytic fourier transform of an airy disk.

        Parameters
        ----------
        unit_x : `numpy.ndarray`
            sample points in x axis
        unit_y : `numpy.ndarray`
            sample points in y axis

        Returns
        -------
        `numpy.ndarray`
            2D numpy array containing the analytic fourier transform

        """
        from .otf import diffraction_limited_mtf
        r, p = cart_to_polar(*m.meshgrid(unit_x, unit_y))
        return diffraction_limited_mtf(self.fno, self.wavelength, r*1e3)  # um to mm


def _airydisk(unit_r, fno, wavelength):
    """Compute the airy disk function over a given spatial distance.

    Parameters
    ----------
    unit_r : `numpy.ndarray`
        ndarray with units of um
    fno : `float`
        F/# of the system
    wavelength : `float`
        wavelength of light, um

    Returns
    -------
    `numpy.ndarray`
        ndarray containing the airy pattern

    """
    u_eff = unit_r * m.pi / wavelength / fno
    return abs(2 * m.jinc(u_eff)) ** 2


def _encircled_energy_core(mtf_data, radius, nu_p, dx, dy):
    """Core computation of encircled energy, based on Baliga 1988.

    Parameters
    ----------
    mtf_data : `numpy.ndarray`
        unaliased MTF data
    radius : `float`
        radius of "detector"
    nu_p : `numpy.ndarray`
        radial spatial frequencies
    dx : `float`
        x frequency delta
    dy : `float`
        y frequency delta

    Returns
    -------
    `float`
        encircled energy for given radius

    """
    integration_fourier = m.j1(2 * m.pi * radius * nu_p) / nu_p
    # division by nu_p will cause a NaN at the origin, 0.5 is the
    # analytical value of jinc there
    integration_fourier[m.isnan(integration_fourier)] = 0.5
    dat = mtf_data * integration_fourier
    return radius * dat.sum() * dx * dy


def _analytical_encircled_energy(fno, wavelength, points):
    """Compute the analytical encircled energy for a diffraction limited circular aperture.

    Parameters
    ----------
    fno : `float`
        F/#
    wavelength : `float`
        wavelength of light
    points : `numpy.ndarray`
        radii of "detector"

    Returns
    -------
    `numpy.ndarray`
        encircled energy values

    """
    p = points * m.pi / fno / wavelength
    return 1 - m.j0(p)**2 - m.j1(p)**2


def _inverse_analytic_encircled_energy(fno, wavelength, energy=FIRST_AIRY_ENCIRCLED):
    def optfcn(x):
        return abs(_analytical_encircled_energy(fno, wavelength, x) - energy)

    return optimize.golden(optfcn)
