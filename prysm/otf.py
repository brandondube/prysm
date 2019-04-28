"""A base optical transfer function interface."""
from scipy import interpolate

from .conf import config
from .mathops import engine as e
from ._basicdata import BasicData
from .psf import PSF
from .fttools import forward_ft_unit
from .util import share_fig_ax
from .coordinates import polar_to_cart, uniform_cart_to_polar


class MTF(BasicData):
    """Modulation Transfer Function."""
    _data_attr = 'data'

    def __init__(self, data, x, y=None):
        """Create an MTF object.

        Parameters
        ----------
        data : `numpy.ndarray`
            MTF values on 2D grid
        x : `numpy.ndarray`
            array of x units, 1D
        y : `numpy.ndarray`
            array of y units, 1D

        """
        if y is None:
            y = x
        self.data = data
        self.x = x
        self.y = y

        self.interpf_2d = None
        self.interpf_tan = None
        self.interpf_sag = None

    # quick-access slices ------------------------------------------------------

    @property
    def tan(self):
        """Retrieve the tangential MTF.

        Notes
        -----
        Assumes the object is extended in y.  If the object is extended
        along a different azimuth, this will not return the tangential MTF.

        Returns
        -------
        self.x : `numpy.ndarray`
            ordinate
        self.data : `numpy.ndarray`
            coordiante

        """
        u, s = self.slice_x
        return u[self.center_x:], s[self.center_x:]

    @property
    def sag(self):
        """Retrieve the sagittal MTF.

        Notes
        -----
        Assumes the object is extended in y.  If the object is extended along a different
        azimuth, this will not return the sagittal MTF.

        Returns
        -------
        self.x : `numpy.ndarray`
            ordinate
        self.data : `numpy.ndarray`
            coordiante

        """
        u, s = self.slice_y
        return u[self.center_y:], s[self.center_y:]

    def exact_polar(self, freqs, azimuths=None):
        """Retrieve the MTF at the specified frequency-azimuth pairs.

        Parameters
        ----------
        freqs : iterable
            radial frequencies to retrieve MTF for
        azimuths : iterable
            corresponding azimuths to retrieve MTF for

        Returns
        -------
        `list`
            MTF at the given points

        """
        self._make_interp_function_2d()

        # handle user-unspecified azimuth
        if azimuths is None:
            if type(freqs) in (int, float):
                # single azimuth
                azimuths = 0
            else:
                azimuths = [0] * len(freqs)
        # handle single azimuth, multiple freqs
        elif type(azimuths) in (int, float):
            azimuths = [azimuths] * len(freqs)

        azimuths = e.radians(azimuths)
        # handle single value case
        if type(freqs) in (int, float):
            x, y = polar_to_cart(freqs, azimuths)
            return float(self.interpf_2d((x, y), method='linear'))

        outs = []
        for freq, az in zip(freqs, azimuths):
            x, y = polar_to_cart(freq, az)
            outs.append(float(self.interpf_2d((x, y), method='linear')))
        return e.asarray(outs)

    def exact_xy(self, x, y=None):
        """Retrieve the MTF at the specified X-Y frequency pairs.

        Parameters
        ----------
        x : iterable
            X frequencies to retrieve the MTF at
        y : iterable
            Y frequencies to retrieve the MTF at

        Returns
        -------
        `list`
            MTF at the given points

        """
        self._make_interp_function_2d()

        # handle data along just x
        if y is None:
            if type(x) in (int, float):
                # single azimuth
                y = 0
            else:
                y = [0] * len(x)

        elif type(y) in (int, float):
            y = [y] * len(x)

        # handle data just along y
        if type(x) in (int, float):
            x = [x] * len(y)

        x, y = e.asarray(x), e.asarray(y)
        outs = []
        for x, y in zip(x, y):
            outs.append(float(self.interpf_2d((x, y), method='linear')))
        return e.asarray(outs)

    def exact_tan(self, freq):
        """Return data at an exact x coordinate along the y=0 axis.

        Parameters
        ----------
        freq : `number` or `numpy.ndarray`
            frequency or frequencies to return

        Returns
        -------
        `numpy.ndarray`
            ndarray of MTF values

        """
        self._make_interp_function_tansag()
        return self.interpf_tan(freq)

    def exact_sag(self, freq):
        """Return data at an exact y coordinate along the x=0 axis.

        Parameters
        ----------
        freq : `number` or `numpy.ndarray`
            frequency or frequencies to return

        Returns
        -------
        `numpy.ndarray`
            ndarray of MTF values

        """
        self._make_interp_function_tansag()
        return self.interpf_sag(freq)

    def azimuthal_average(self):
        """Return the azimuthally averaged MTF.

        Returns
        -------
        nu : `numpy.ndarray`
            spatial frequencies
        mtf : `numpy.ndarray`
            mtf values

        """
        nu, theta, mtf = uniform_cart_to_polar(self.x, self.y, self.data)
        l = len(nu) // 2
        return nu[:l], mtf.mean(axis=0)[:l]

    # quick-access slices ------------------------------------------------------

    # plotting -----------------------------------------------------------------

    def plot2d(self, max_freq=200, power=1, cmap=config.image_colormap, fig=None, ax=None):
        """Create a 2D plot of the MTF.

        Parameters
        ----------
        max_freq : `float`
            Maximum frequency to plot to.  Axis limits will be ((-max_freq, max_freq), (-max_freq, max_freq)).
        power : `float`
            inverse of power to stretch the MTF to/by, e.g. power=2 will plot MTF^(1/2)
        fig : `matplotlib.figure.Figure`, optional:
            Figure to draw plot in
        ax : `matplotlib.axes.Axis`, optional:
            Axis to draw plot in

        Returns
        -------
        fig : `matplotlib.figure.Figure`
            Figure to draw plot in
        ax : `matplotlib.axes.Axis`
            Axis to draw plot in

        """
        from matplotlib import colors

        left, right = self.x[0], self.x[-1]
        bottom, top = self.y[0], self.y[-1]

        fig, ax = share_fig_ax(fig, ax)

        im = ax.imshow(self.data,
                       extent=[left, right, bottom, top],
                       origin='lower',
                       cmap='Greys_r',
                       clim=(0, 1),
                       norm=colors.PowerNorm(1/power),
                       interpolation='lanczos')
        cb = fig.colorbar(im, label='MTF [Rel 1.0]', ax=ax, fraction=0.046)
        cb.outline.set_edgecolor('k')
        cb.outline.set_linewidth(0.5)
        ax.set(xlabel=r'$\nu_x$ [cy/mm]',
               ylabel=r'$\nu_y$ [cy/mm]',
               xlim=(-max_freq, max_freq),
               ylim=(-max_freq, max_freq))
        return fig, ax

    def plot_tan_sag(self, max_freq=200, lw=config.lw, zorder=config.zorder, fig=None, ax=None, labels=('Tangential', 'Sagittal')):
        """Create a plot of the tangential and sagittal MTF.

        Parameters
        ----------
        max_freq : `float`
            Maximum frequency to plot to.  Axis limits will be ((-max_freq, max_freq), (-max_freq, max_freq))
        lw : `float`, optional
            line width
        zorder : `int`
            zorder
        fig : `matplotlib.figure.Figure`, optional:
            Figure to draw plot in
        ax : `matplotlib.axes.Axis`, optional:
            Axis to draw plot in
        labels : `iterable`
            set of labels for the two lines that will be plotted

        Returns
        -------
        fig : `matplotlib.figure.Figure`
            Figure to draw plot in
        ax : `matplotlib.axes.Axis`
            Axis to draw plot in

        """
        ut, tan = self.tan
        us, sag = self.sag

        fig, ax = share_fig_ax(fig, ax)
        ax.plot(ut, tan, label=labels[0], linestyle='-', lw=lw, zorder=zorder)
        ax.plot(us, sag, label=labels[1], linestyle='--', lw=lw, zorder=zorder)
        ax.set(xlabel='Spatial Frequency [cy/mm]',
               ylabel='MTF [Rel 1.0]',
               xlim=(0, max_freq),
               ylim=(0, 1))
        ax.legend(loc='lower left')
        return fig, ax

    def plot_azimuthal_average(self, max_freq=200, lw=config.lw, zorder=config.zorder, fig=None, ax=None):
        """Create a plot of the azimuthally averaged MTF.

        Parameters
        ----------
        max_freq : `float`
            Maximum frequency to plot to.  Axis limits will be ((-max_freq, max_freq), (-max_freq, max_freq))
        lw : `float`, optional
            line width
        zorder : `int`, optional
        fig : `matplotlib.figure.Figure`, optional:
            Figure to draw plot in
        ax : `matplotlib.axes.Axis`, optional:
            Axis to draw plot in
        labels : `iterable`
            set of labels for the two lines that will be plotted

        Returns
        -------
        fig : `matplotlib.figure.Figure`
            Figure to draw plot in
        ax : `matplotlib.axes.Axis`
            Axis to draw plot in

        """

        u, azavg = self.azimuthal_average()

        fig, ax = share_fig_ax(fig, ax)
        ax.plot(u, azavg, lw=lw, zorder=zorder)
        ax.set(xlabel='Spatial Frequency [cy/mm]',
               ylabel='MTF [Rel 1.0]',
               xlim=(0, max_freq),
               ylim=(0, 1))

        return fig, ax

    # plotting -----------------------------------------------------------------

    # helpers ------------------------------------------------------------------

    def _make_interp_function_2d(self):
        """Generate a 2D interpolation function for this instance of MTF, used to procure MTF at exact frequencies.

        Returns
        -------
        `scipy.interpolate.RegularGridInterpolator`, this instance of an MTF object.

        """
        if self.interpf_2d is None:
            self.interpf_2d = interpolate.RegularGridInterpolator((self.x, self.y), self.data)

        return self.interpf_2d

    def _make_interp_function_tansag(self):
        """Generate two interpolation functions for tangential and sagittal MTF.

        Returns
        -------
        self.interpf_tan : `scipy.interpolate.interp1d`
            tangential interpolator
        self.interpf_sag : `scipy.interpolate.interp1d`
            sagittal interpolator

        """
        if self.interpf_tan is None or self.interpf_sag is None:
            ut, tan = self.tan
            us, sag = self.sag

            self.interpf_tan = interpolate.interp1d(ut, tan)
            self.interpf_sag = interpolate.interp1d(us, sag)

        return self.interpf_tan, self.interpf_sag

    @staticmethod
    def from_psf(psf):
        """Generate an MTF from a PSF.

        Parameters
        ----------
        psf : `PSF`
            PSF to compute an MTF from

        Returns
        -------
        `MTF`
            A new MTF instance

        """
        if getattr(psf, '_mtf', None) is not None:
            return psf._mtf
        else:
            dat = abs(e.fft.fftshift(e.fft.fft2(psf.data)))  # no need to ifftshift first - phase is unimportant
            x = forward_ft_unit(psf.sample_spacing / 1e3, psf.samples_x)  # 1e3 for microns => mm
            y = forward_ft_unit(psf.sample_spacing / 1e3, psf.samples_y)
            return MTF(dat / dat[psf.center_y, psf.center_x], x, y)

    @staticmethod
    def from_pupil(pupil, efl, Q=2):
        """Generate an MTF from a pupil, given a focal length (propagation distance).

        Parameters
        ----------
        pupil : `Pupil`
            A pupil to propagate to a PSF, and convert to an MTF
        efl : `float`
            Effective focal length or propagation distance of the wavefunction
        Q : `number`
            ratio of pupil sample count to PSF sample count.  Q > 2 satisfies nyquist

        Returns
        -------
        `MTF`
            A new MTF instance

        """
        psf = PSF.from_pupil(pupil, efl=efl, Q=Q)
        return MTF.from_psf(psf)


def diffraction_limited_mtf(fno, wavelength, frequencies=None, samples=128):
    """Give the diffraction limited MTF for a circular pupil and the given parameters.

    Parameters
    ----------
    fno : `float`
        f/# of the lens.
    wavelength : `float`
        wavelength of light, in microns.
    frequencies : `numpy.ndarray`
        spatial frequencies of interest, in cy/mm if frequencies are given, samples is ignored.
    samples : `int`
        number of points in the output array, if frequencies not given.

    Returns
    -------
    if frequencies not given:
        frequencies : `numpy.ndarray`
            array of ordinate data
        mtf : `numpy.ndarray`
            array of coordinate data
    else:
        mtf : `numpy.ndarray`
            array of MTF data

    Notes
    -----
    If frequencies are given, just returns the MTF.  If frequencies are not
    given, returns both the frequencies and the MTF.

    """
    extinction = 1 / (wavelength / 1000 * fno)
    if frequencies is None:
        normalized_frequency = e.linspace(0, 1, samples)
    else:
        normalized_frequency = e.asarray(frequencies) / extinction
        try:
            normalized_frequency[normalized_frequency > 1] = 1  # clamp values
        except TypeError:  # single freq
            if normalized_frequency > 1:
                normalized_frequency = 1

    mtf = _difflim_mtf_core(normalized_frequency)

    if frequencies is None:
        return normalized_frequency * extinction, mtf
    else:
        return mtf


def _difflim_mtf_core(normalized_frequency):
    """Compute the MTF at a given normalized spatial frequency.

    Parameters
    ----------
    normalized_frequency : `numpy.ndarray`
        normalized frequency; function is defined over [0, and takes a value of 0 for [1,

    Returns
    -------
    `numpy.ndarray`
        The diffraction MTF function at a given normalized spatial frequency

    """
    return (2 / e.pi) * \
           (e.arccos(normalized_frequency) - normalized_frequency *
            e.sqrt(1 - normalized_frequency ** 2))


def longexposure_otf(nu, Cn, z, f, lambdabar, h_z_by_r=2.91):
    """Compute the long exposure OTF for given parameters.

    Parameters
    ----------
    nu : `numpy.ndarray`
        spatial frequencies, cy/mm
    Cn: `float`
        atmospheric structure constant of refractive index, ranges ~ 10^-13 - 10^-17
    z : `float`
        propagation distance through atmosphere, m
    f : `float`
        effective focal length of the optical system, mm
    lambdabar : `float`
        mean wavelength, microns
    h_z_by_r : `float`, optional
        constant for h[z/r] -- see Eq. 8.5-37 & 8.5-38 in Statistical Optics, J. Goodman, 2nd ed.

    Returns
    -------
    `numpy.ndarray`
        the OTF

    """
    # homogenize units
    nu = nu / 1e3
    f = f / 1e3
    lambdabar = lambdabar / 1e6

    power = 5/3
    const1 = - e.pi ** 2 * 2 * h_z_by_r * Cn ** 2
    const2 = z * f ** power / (lambdabar ** 3)
    nupow = nu ** power
    const = const1 * const2
    return e.exp(const * nupow)


def estimate_Cn(P=1013, T=273.15, Ct=1e-4):
    """Use Weng et al to estimate Cn from meteorological data.

    Parameters
    ----------
    P : `float`
        atmospheric pressure in hPa
    T : `float`
        temperature in Kelvin
    Ct : `float`
        atmospheric struction constant of temperature, typically 10^-5 - 10^-2 near the surface

    Returns
    -------
    `float`
        Cn

    """
    return (79 * P / (T ** 2)) * Ct ** 2 * 1e-12
