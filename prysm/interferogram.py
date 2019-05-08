"""tools to analyze interferometric data."""
import warnings
import inspect

from scipy import signal, optimize

from .conf import config
from .mathops import engine as e
from ._phase import OpticalPhase
from .zernike import defocus, zernikefit, FringeZernike
from .io import read_zygo_dat, read_zygo_datx, write_zygo_ascii
from .fttools import forward_ft_unit
from .coordinates import cart_to_polar, uniform_cart_to_polar
from .util import share_fig_ax, mean, rms  # NOQA
from .geometry import mcache


def fit_plane(x, y, z):
    """Fit a plane to data.

    Parameters
    ----------
    x : `numpy.ndarray`
        1D array of x (axis 1) values
    y : `numpy.ndarray`
        1D array of y (axis 0) values
    z : `numpy.ndarray`
        2D array of z values

    Returns
    -------
    `numpy.ndarray`
        array representation of plane

    """
    pts = e.isfinite(z)
    if len(z.shape) > 1:
        x, y = e.meshgrid(x, y)
        xx, yy = x[pts].flatten(), y[pts].flatten()
    else:
        xx, yy = x, y

    flat = e.ones(xx.shape)

    coefs = e.linalg.lstsq(e.stack([xx, yy, flat]).T, z[pts].flatten(), rcond=None)[0]
    plane_fit = coefs[0] * x + coefs[1] * y + coefs[2]
    return plane_fit


def fit_sphere(z):
    """Fit a sphere to data.

    Parameters
    ----------
    z : `numpy.ndarray`
        2D array of data

    Returns
    -------
    `numpy.ndarray`
        sphere data

    """
    x, y = e.linspace(-1, 1, z.shape[1]), e.linspace(-1, 1, z.shape[0])
    xx, yy = e.meshgrid(x, y)
    pts = e.isfinite(z)
    xx_, yy_ = xx[pts].flatten(), yy[pts].flatten()
    rho, phi = cart_to_polar(xx_, yy_)
    focus = defocus(rho, phi)

    coefs = e.linalg.lstsq(e.stack([focus, e.ones(focus.shape)]).T, z[pts].flatten(), rcond=None)[0]
    rho, phi = cart_to_polar(xx, yy)
    sphere = defocus(rho, phi) * coefs[0]
    return sphere


def make_window(signal, sample_spacing, which=None, alpha=4):
    """Generate a window function to be used in PSD analysis.

    Parameters
    ----------
    signal : `numpy.ndarray`
        signal or phase data
    sample_spacing : `float`
        spacing of samples in the input data
    which : `str,` {'welch', 'hann', None}, optional
        which window to produce.  If auto, attempts to guess the appropriate
        window based on the input signal
    alpha : `float`, optional
        alpha value for welch window

    Notes
    -----
    For 2D welch, see:
    Power Spectral Density Specification and Analysis of Large Optical Surfaces
    E. Sidick, JPL

    Returns
    -------
    `numpy.ndarray`
        window array

    """
    s = signal.shape

    if which is None:
        # attempt to guess best window
        ysamples = int(round(s[0] * 0.02, 0))
        xsamples = int(round(s[1] * 0.02, 0))
        corner1 = signal[:ysamples, :xsamples] == 0
        corner2 = signal[-ysamples:, :xsamples] == 0
        corner3 = signal[:ysamples, -xsamples:] == 0
        corner4 = signal[-ysamples:, -xsamples:] == 0
        if corner1.all() and corner2.all() and corner3.all() and corner4.all():
            # four corners all "black" -- circular data, Welch window is best
            # looks wrong but 2D welch takes x, y while indices are y, x
            y, x = (e.arange(N) - (N / 2) for N in s)
            which = window_2d_welch(x, y)
        else:
            # if not circular, square data; use Hanning window
            y, x = (e.hanning(N) for N in s)
            which = e.outer(x, y)
    else:
        if type(which) is str:
            # known window type
            wl = which.lower()
            if wl == 'welch':
                y, x = (e.arange(N) - (N / 2) for N in s)
                which = window_2d_welch(x, y, alpha=alpha)
            elif wl in ('hann', 'hanning'):
                y, x = (e.hanning(N) for N in s)
                which = e.outer(y, x)
            else:
                raise ValueError('unknown window type')

    return which  # window provided as ndarray


def psd(height, sample_spacing, window=None):
    """Compute the power spectral density of a signal.

    Parameters
    ----------
    height : `numpy.ndarray`
        height or phase data
    sample_spacing : `float`
        spacing of samples in the input data
    window : {'welch', 'hann'} or ndarray, optional

    Returns
    -------
    x : `numpy.ndarray`
        ordinate x frequency axis
    y : `numpy.ndarray`
        ordinate y frequency axis
    psd : `numpy.ndarray`
        power spectral density

    Notes
    -----
    See GH_FFT for a rigorous treatment of FFT scalings
    https://holometer.fnal.gov/GH_FFT.pdf

    """
    window = make_window(height, sample_spacing, window)
    fft = e.fft.ifftshift(e.fft.fft2(e.fft.fftshift(height * window)))
    psd = abs(fft)**2  # mag squared first as per GH_FFT

    fs = 1 / sample_spacing
    S2 = (window**2).sum()
    coef = S2 * fs * fs
    psd /= coef

    ux = forward_ft_unit(sample_spacing, height.shape[1])
    uy = forward_ft_unit(sample_spacing, height.shape[0])
    return ux, uy, psd


def bandlimited_rms(ux, uy, psd, wllow=None, wlhigh=None, flow=None, fhigh=None):
    """Calculate the bandlimited RMS of a signal from its PSD.

    Parameters
    ----------
    ux : `numpy.ndarray`
        x spatial frequencies
    uy : `numpy.ndarray`
        y spatial frequencies
    psd : `numpy.ndarray`
        power spectral density
    wllow : `float`
        short spatial scale
    wlhigh : `float`
        long spatial scale
    flow : `float`
        low frequency
    fhigh : `float`
        high frequency

    Returns
    -------
    `float`
        band-limited RMS value.

    """
    if wllow is not None or wlhigh is not None:
        # spatial period given
        if wllow is None:
            flow = 0
        else:
            fhigh = 1 / wllow

        if wlhigh is None:
            fhigh = max(ux[-1], uy[-1])
        else:
            flow = 1 / wlhigh
    elif flow is not None or fhigh is not None:
        # spatial frequency given
        if flow is None:
            flow = 0
        if fhigh is None:
            fhigh = max(ux[-1], uy[-1])
    else:
        raise ValueError('must specify either period (wavelength) or frequency')

    ux2, uy2 = e.meshgrid(ux, uy)
    r, p = cart_to_polar(ux2, uy2)

    if flow is None:
        warnings.warn('no lower limit given, using 0 for low frequency')
        flow = 0

    if fhigh is None:
        warnings.warn('no upper limit given, using limit imposed by data.')
        fhigh = r.max()

    work = psd.copy()
    work[r < flow] = 0
    work[r > fhigh] = 0
    first = e.trapz(work, uy, axis=0)
    second = e.trapz(first, ux, axis=0)
    return e.sqrt(second)


def window_2d_welch(x, y, alpha=8):
    """Return a 2D welch window for a given alpha.

    Parameters
    ----------
    x : `numpy.ndarray`
        x values, 1D array
    y : `numpy.ndarray`
        y values, 1D array
    alpha : `float`
        alpha (edge roll) parameter

    Returns
    -------
    `numpy.ndarray`
        window

    """
    xx, yy = e.meshgrid(x, y)
    r, _ = cart_to_polar(xx, yy)

    rmax = max(x.max(), y.max())
    window = 1 - abs(r/rmax)**alpha
    return window


def abc_psd(nu, a, b, c):
    """Lorentzian model of a Power Spectral Density.

    Parameters
    ----------
    nu : `numpy.ndarray` or `float`
        spatial frequency
    a : `float`
        a coefficient
    b : `float`
        b coefficient
    c : `float`
        c coefficient

    Returns
    -------
    `numpy.ndarray`
        value of PSD model

    """
    return a / (1 + (nu/b)**2)**(c/2)


def ab_psd(nu, a, b):
    """Inverse power model of a Power Spectral Density.

    Parameters
    ----------
    nu : `numpy.ndarray` or `float`
        spatial frequency
    a : `float`
        a coefficient
    b : `float`
        b coefficient

    Returns
    -------
    `numpy.ndarray`
        value of PSD model

    """
    return a * nu ** (-b)


def synthesize_surface_from_psd(psd, nu_x, nu_y):
    """Synthesize a surface height map from PSD data.

    Parameters
    ----------
    psd : `numpy.ndarray`
        PSD data, units nm²/(cy/mm)²
    nu_x : `numpy.ndarray`
        x spatial frequency, cy/mm
    nu_y : `numpy.ndarray`
        y spatial frequency, cy_mm

    """
    # generate a random phase to be matched to the PSD
    randnums = e.random.rand(*psd.shape)
    randfft = e.fft.fft2(randnums)
    phase = e.angle(randfft)

    # calculate the output window
    # the 0th element of nu_y has the greatest frequency in magnitude because of
    # the convention to put the nyquist sample at -fs instead of +fs for even-size arrays
    fs = -2 * nu_y[0]
    dx = dy = 1 / fs
    ny, nx = psd.shape
    x, y = e.arange(nx) * dx, e.arange(ny) * dy

    # calculate the area of the output window, "S2" in GH_FFT notation
    A = x[-1] * y[-1]

    # use ifft to compute the PSD
    signal = e.exp(1j * phase) * e.sqrt(A * psd)

    coef = 1 / dx / dy
    out = e.fft.ifftshift(e.fft.ifft2(e.fft.fftshift(signal))) * coef
    out = out.real
    return x, y, out


def render_synthetic_surface(size, samples, rms=None, mask='circle', psd_fcn=abc_psd, **psd_fcn_kwargs):  # NOQA
    """Render a synthetic surface with a given RMS value given a PSD function.

    Parameters
    ----------
    size : `float`
        diameter of the output surface, mm
    samples : `int`
        number of samples across the output surface
    rms : `float`
        desired RMS value of the output, if rms=None, no normalization is done
    mask : `str`, optional
        mask defining the clear aperture
    psd_fcn : `callable`
        function used to generate the PSD
    **psd_fcn_kwargs:
        keyword arguments passed to psd_fcn in addition to nu
        if psd_fcn == abc_psd, kwargs are a, b, c
        elif psd_Fcn == ab_psd kwargs are a, b

        kwargs will be user-defined for user PSD functions

    Returns
    -------
    x : `numpy.ndarray`
        x coordinates, mm
    y: `numpy.ndarray`
        y coordinates, mm
    z : `numpy.ndarray`
        height data, nm

    """
    # compute the grid and PSD
    sample_spacing = size / (samples - 1)
    nu_x = nu_y = forward_ft_unit(sample_spacing, samples)
    center = samples // 2  # some bullshit here to gloss over zeros for ab_psd
    nu_x[center] = nu_x[center+1] / 10
    nu_y[center] = nu_y[center+1] / 10
    nu_xx, nu_yy = e.meshgrid(nu_x, nu_y)

    nu_r, _ = cart_to_polar(nu_xx, nu_yy)
    psd = psd_fcn(nu_r, **psd_fcn_kwargs)

    # synthesize a surface from the PSD
    x, y, z = synthesize_surface_from_psd(psd, nu_x, nu_y)

    # mask
    mask = mcache(mask, samples)
    z[mask == 0] = e.nan

    # possibly scale RMS
    if rms is not None:
        z_rms = globals()['rms'](z)  # rms function is shadowed by rms kwarg
        scale_factor = rms / z_rms
        z *= scale_factor

    return x, y, z


def fit_psd(f, psd, callable=abc_psd, guess=None, return_='coefficients'):
    """Fit parameters to a PSD curve.

    Parameters
    ----------
    f : `numpy.ndarray`
        spatial frequency, cy/length
    psd : `numpy.ndarray`
        1D PSD, units of height^2 / (cy/length)^2
    callable : callable, optional
        a callable object that takes parameters of (frequency, *); all other parameters will be fit
    return_ : `str`, optional, {'coefficients', 'optres'}
        what to return; either return the coefficients (optres.x) or the optimization result (optres)

    Returns
    -------
    optres
        `scipy.optimization.OptimizationResult`
    coefficients
        `numpy.ndarray` of coefficients

    """
    sig = inspect.signature(callable)
    nparams = len(sig.parameters) - 1  # -1; offset for frequency parameter

    if nparams < 3:  # ab-type PSD
        # arbitrarily drop the lowest frequency bins; due to removal of piston/tiptilt/power
        # the PSD will roll off in this region, we want to just fit the flat part
        f = f[5:]
        psd = psd[5:]

    if guess is None:
        initial_args = [1] * nparams
        initial_args[0] = 100
    else:
        initial_args = guess

    D = e.log10(psd)
    N = D.shape[0]

    def optfcn(x):
        M = callable(f, *x)
        M = e.log10(M)
        cost_vec = (D - M) ** 2
        cost = cost_vec.sum() / N
        return cost

    optres = optimize.basinhopping(optfcn, initial_args, minimizer_kwargs=dict(method='L-BFGS-B'))
    if return_.lower() != 'coefficients':
        return optres
    else:
        return optres.x


class Interferogram(OpticalPhase):
    """Class containing logic and data for working with interferometric data."""

    def __init__(self, phase, intensity=None, x=None, y=None, scale='px', phase_unit='nm', meta=None):
        """Create a new Interferogram instance.

        Parameters
        ----------
        phase : `numpy.ndarray`
            phase values, units of phase_unit
        intensity : `numpy.ndarray`, optional
            intensity array from interferometer camera
        x : `numpy.ndarray`, optional
            x (axis 1) values, units of scale
        y : `numpy.ndarray`, optional
            y (axis 0) values, units of scale
        phase_unit : `str`, optional
            unit to use to represent phase
        meta : `dict`
            dictionary of any metadata.  if a wavelength or Wavelength key is
            present, this will also be stored in self.wavelength

        """
        if x is None:  # assume x, y given together
            x = e.arange(phase.shape[1], dtype=config.precision)
            y = e.arange(phase.shape[0], dtype=config.precision)
            scale = 'px'
            self.lateral_res = 1

        if meta:
            wvl = meta.get('wavelength', None)
            if wvl is None:
                wvl = meta.get('Wavelength')

            if wvl is not None:
                wvl *= 1e6  # m to um
        else:
            wvl = 1

        super().__init__(x=x, y=y, phase=phase,
                         wavelength=wvl, phase_unit=phase_unit,
                         spatial_unit=scale)

        self.xaxis_label = 'X'
        self.yaxis_label = 'Y'
        self.zaxis_label = 'Height'
        self.intensity = intensity
        self.meta = meta

        if scale != 'px':
            self.change_spatial_unit(to=scale, inplace=True)

    @property
    def dropout_percentage(self):
        """Percentage of pixels in the data that are invalid (NaN)."""
        return e.count_nonzero(e.isnan(self.phase)) / self.phase.size * 100

    @property
    def pvr(self):
        """Peak-to-Valley residual.

        Notes
        -----
        See:
        C. Evans, "Robust Estimation of PV for Optical Surface Specification and Testing"
        in Optical Fabrication and Testing, OSA Technical Digest (CD)
        (Optical Society of America, 2008), paper OWA4.
        http://www.opticsinfobase.org/abstract.cfm?URI=OFT-2008-OWA4

        """
        coefs, residual = zernikefit(self.phase, terms=36, residual=True, map_='fringe')
        fz = FringeZernike(coefs, samples=self.shape[0])
        return fz.pv + 3 * residual

    def fit_zernikes(self, terms, map_='noll', norm=True, residual=False):
        """Fit Zernikes to the interferometric data.

        Parameters
        ----------
        terms : `int`
            number of terms to fit
        map_ : `str`, {'noll', 'fringe'}, optional
            which set ("map") of Zernikes to fit to
        norm : `bool`, optional
            whether to orthonormalize the terms to unit RMS value
        residual : `bool`
            if true, return two values (coefficients, residual), else return
            only coefficients

        Returns
        -------
        coefs : `numpy.ndarray`
            Zernike coefficients, same units as self.phase_unit
        residual : `float`
            RMS residual of the fit, same units as self.phase_unit

        """
        return zernikefit(self.phase, terms=terms, map_=map_, norm=norm, residual=residual)

    def fill(self, _with=0):
        """Fill invalid (NaN) values.

        Parameters
        ----------
        _with : `float`, optional
            value to fill with

        Returns
        -------
        `Interferogram`
            self

        """
        nans = e.isnan(self.phase)
        self.phase[nans] = _with
        return self

    def crop(self):
        """Crop data to rectangle bounding non-NaN region."""
        nans = e.isfinite(self.phase)
        nancols = e.any(nans, axis=0)
        nanrows = e.any(nans, axis=1)

        left, right = nanrows.argmax(), nanrows[::-1].argmax()
        top, bottom = nancols.argmax(), nancols[::-1].argmax()
        if left == right == top == bottom == 0:
            return self

        if left == 0 and bottom == 0:
            lr = slice(None)
        if left == 0:
            lr = slice(-right)
        elif right == 0:
            lr = slice(left, self.phase.shape[0])
        else:
            lr = slice(left, -right)

        if top == 0 and bottom == 0:
            tb = slice(None)
        elif top == 0:
            tb = slice(-bottom)
        elif bottom == 0:
            tb = slice(top, self.phase.shape[1])
        else:
            tb = slice(top, -bottom)

        self.phase = self.phase[lr, tb]
        self.y, self.x = self.y[lr], self.x[tb]
        self.x -= self.x[0]
        self.y -= self.y[0]
        return self

    def recenter(self):
        """Adjust the x and y coordinates so the data is centered on 0,0."""
        mxx, mnx = self.x[-1], self.x[0]
        mxy, mny = self.y[-1], self.y[0]
        cx = (mxx + mnx) / 2
        cy = (mxy + mny) / 2
        self.x -= cx
        self.y -= cy
        return self

    def remove_piston(self):
        """Remove piston from the data by subtracting the mean value."""
        self.phase -= mean(self.phase)
        return self

    def remove_tiptilt(self):
        """Remove tip/tilt from the data by least squares fitting and subtracting a plane."""
        plane = fit_plane(self.x, self.y, self.phase)
        self.phase -= plane
        return self

    def remove_power(self):
        """Remove power from the data by least squares fitting."""
        sphere = fit_sphere(self.phase)
        self.phase -= sphere
        return self

    def remove_piston_tiptilt(self):
        """Remove piston/tip/tilt from the data, see remove_tiptilt and remove_piston."""
        self.remove_piston()
        self.remove_tiptilt()
        return self

    def remove_piston_tiptilt_power(self):
        """Remove piston/tip/tilt/power from the data."""
        self.remove_piston()
        self.remove_tiptilt()
        self.remove_power()
        return self

    def mask(self, shape_or_mask, diameter=None):
        """Mask the signal.

        The mask will be inscribed in the axis with fewer pixels.  I.e., for
        a interferogram with 1280x1000 pixels, the mask will be 1000x1000 at
        largest.

        Parameters
        ----------
        shape_or_mask : `str` or `numpy.ndarray`
            valid shape from prysm.geometry or array containing mask
        diameter : `float`
            diameter of the mask, in self.spatial_units
        mask : `numpy.ndarray`
            user-provided mask

        Returns
        -------
        self
            modified Interferogram instance.

        """
        if isinstance(shape_or_mask, str):
            if diameter is None:
                diameter = self.diameter
            mask = mcache(shape_or_mask, min(self.shape), radius=diameter / min(self.diameter_x, self.diameter_y))
            base = e.zeros(self.shape, dtype=config.precision)
            difference = abs(self.shape[0] - self.shape[1])
            l, u = int(e.floor(difference / 2)), int(e.ceil(difference / 2))
            if u == 0:  # guard against nocrop scenario
                _slice = slice(None)
            else:
                _slice = slice(l, -u)
            if self.shape[0] < self.shape[1]:
                base[:, _slice] = mask
            else:
                base[_slice, :] = mask

            mask = base
        else:
            mask = shape_or_mask

        hitpts = mask == 0
        self.phase[hitpts] = e.nan
        return self

    def filter(self, critical_frequency=None, critical_period=None,
               kind='bessel', type_=None, order=1, filtkwargs=dict()):
        """Apply a frequency-domain filter to the phase data.

        Parameters
        ----------
        critical_frequency : `float` or length-2 tuple
            critical ("cutoff") frequency/frequencies of the filter.  Units of cy/self.spatial_unit
        critical_period : `float` or length-2 tuple
            critical ("cutoff") period/s of the filter.  Units of self.spatial_unit.
            Will clobber critical_frequency if both given
        kind : `str`, optional
            filter type -- see scipy.signal for filter types and possible extra arguments.  Examples are:
            - bessel
            - butter
            - ellip
            - cheby2
        type_ : `str`, optional, {'lowpass', 'highpass', 'bandpass', 'bandreject'}
            filter type -- lowpass, highpass, bandpass, or bandreject
            defaults to lowpass if single freq/period given or bandpass if two given
        order : `int`, optional
            order of the filter
        filtkwargs : `dict`, optional
            kwargs passed to the filter constructor

        Returns
        -------
        `Interferogram`
            self

        Notes
        -----
        These filters are implemented using scipy.signal and are a rigorous treatment that defaults to use of higher
        order filters with strong out-of-band rejection.  This choices is not in accord with the one in made by
        some software shipping with commercial interferometers.

        """
        fs = 1 / self.sample_spacing
        nyquist = fs / 2

        if critical_frequency is None and critical_period is None:
            raise ValueError('must provide critical frequenc(ies) or critical period(s).')

        if critical_period is not None:
            if hasattr(critical_period, '__iter__'):
                critical_frequency = [1 / x for x in reversed(critical_period)]
            else:
                critical_frequency = 1 / critical_period

        if hasattr(critical_frequency, '__iter__'):
            critical_frequency = [c / nyquist for c in critical_frequency]
            if type_ is None:
                type_ = 'bandpass'
        else:
            critical_frequency = critical_frequency / nyquist
            if type_ is None:
                type_ = 'lowpass'

        if type_ == 'bandreject':
            type_ = 'bandstop'

        filtfunc = getattr(signal, kind)

        b, a = filtfunc(N=order, Wn=critical_frequency, btype=type_, analog=False, output='ba', **filtkwargs)

        filt_y = signal.lfilter(b, a, self.phase, axis=0)
        filt_both = signal.lfilter(b, a, filt_y, axis=1)
        self.phase = filt_both
        return self

    def latcal(self, plate_scale, unit='mm'):
        """Perform lateral calibration.

        This probably won't do what you want if your data already has spatial
        units of anything but pixels (px).

        Parameters
        ----------
        plate_scale : `float`
            center-to-center sample spacing of pixels, in (unit)s.
        unit : `str`, optional
            unit associated with the plate scale.

        Returns
        -------
        self
            modified `Interferogram` instance.

        """
        self.change_spatial_unit(to=unit, inplace=True)  # will be 0..n spatial units
        # sloppy to do this here...
        self.x *= plate_scale
        self.y *= plate_scale
        return self

    def pad(self, value, unit='spatial'):
        """Pad the interferogram.

        Parameters
        ----------
        value : `float`
            how much to pad the interferogram
        unit : `str`, {'spatial', 'px'}, optional
            what unit to use for padding, spatial units (self.spatial_unit), or pixels

        Returns
        -------
        `Interferogram`
            self

        """
        unit = unit.lower()
        if unit in ('px', 'pixel', 'pixels'):
            npx = value
        else:
            npx = int(e.ceil(value / self.sample_spacing))

        if e.isnan(self.phase[0, 0]):
            fill_val = e.nan
        else:
            fill_val = 0

        s = self.shape
        out = e.empty((s[0] + 2 * npx, s[1] + 2 * npx), dtype=self.phase.dtype)
        out[:, :] = fill_val
        out[npx:-npx, npx:-npx] = self.phase
        self.phase = out

        x = e.arange(out.shape[1], dtype=config.precision) * self.sample_spacing
        y = e.arange(out.shape[0], dtype=config.precision) * self.sample_spacing
        self.x = x
        self.y = y
        return self

    def spike_clip(self, nsigma=3):
        """Clip points in the data that exceed a certain multiple of the standard deviation.

        Parameters
        ----------
        nsigma : `float`
            number of standard deviations to keep

        Returns
        -------
        self
            this Interferogram instance.

        """
        pts_over_nsigma = abs(self.phase) > nsigma * self.std
        self.phase[pts_over_nsigma] = e.nan
        return self

    def psd(self):
        """Power spectral density of the data., units (self.phase_unit^2)/((cy/self.spatial_unit)^2).

        Returns
        -------
        x : `numpy.ndarray`
            ordinate x frequency axis
        y : `numpy.ndarray`
            ordinate y frequency axis
        psd : `numpy.ndarray`
            power spectral density

        """
        return psd(self.phase, self.sample_spacing)

    def psd_slices(self, x=True, y=True, azavg=True, azmin=False, azmax=False):
        """Power spectral density of the data., units (self.phase_unit^2)/((cy/self.spatial_unit)^2).

        Returns
        -------
        `dict`
            with keys x, y, avg.  Each containing a tuple of (unit, psd)

        """
        xx, yy, _psd = self.psd()
        lx, ly = len(xx)//2, len(yy)//2

        out = {}
        if x:
            out['x'] = (xx[lx:], _psd[ly, lx:])

        if y:
            out['y'] = (yy[ly:], _psd[ly:, lx])

        if azavg or azmin or azmax:
            rho, phi, _psdrp = uniform_cart_to_polar(xx, yy, _psd)

        if azavg:
            out['azavg'] = (rho, _psdrp.mean(axis=0))

        if azmin:
            out['azmin'] = (rho, _psdrp.min(axis=0))

        if azmax:
            out['azmax'] = (rho, _psdrp.max(axis=0))

        return out

    def bandlimited_rms(self, wllow=None, wlhigh=None, flow=None, fhigh=None):
        """Calculate the bandlimited RMS of a signal from its PSD.

        Parameters
        ----------
        wllow : `float`
            short spatial scale
        wlhigh : `float`
            long spatial scale
        flow : `float`
            low frequency
        fhigh : `float`
            high frequency

        Returns
        -------
        `float`
            band-limited RMS value.

        """
        return bandlimited_rms(*self.psd(),
                               wllow=wllow,
                               wlhigh=wlhigh,
                               flow=flow,
                               fhigh=fhigh)

    def total_integrated_scatter(self, wavelength, incident_angle=0):
        """Calculate the total integrated scatter (TIS) for an angle or angles.

        Parameters
        ----------
        wavelength : `float`
            wavelength of light in microns.
        incident_angle : `float` or `numpy.ndarray`
            incident angle(s) of light.

        Returns
        -------
        `float` or `numpy.ndarray`
            TIS value.

        """
        if self.spatial_unit != 'μm':
            raise ValueError('Use microns for spatial unit when evaluating TIS.')

        upper_limit = 1 / wavelength
        kernel = 4 * e.pi * e.cos(e.radians(incident_angle))
        kernel *= self.bandlimited_rms(upper_limit, None) / wavelength
        return 1 - e.exp(-kernel**2)

    def plot_psd2d(self, axlim=None, clim=(1e-9, 1e2), cmap=config.image_colormap,
                   interp_method='lanczos', fig=None, ax=None):
        """Plot the two dimensional PSD.

        Parameters
        ----------
        axlim : `float`, optional
            symmetrical axis limit
        clim : `tuple`, optional
            lower, upper limits on color scale
        cmap : `str`, optional
            colormap
        interp_method : `str`, optional
            method used to interpolate the image, passed directly to matplotlib imshow
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
        from matplotlib import colors
        x, y, psd = self.psd()

        if axlim is None:
            lims = (None, None)
        else:
            lims = (-axlim, axlim)

        fig, ax = share_fig_ax(fig, ax)
        im = ax.imshow(psd,
                       extent=[x[0], x[-1], y[0], y[-1]],
                       origin='lower',
                       cmap=cmap,
                       norm=colors.LogNorm(*clim),
                       interpolation=interp_method)

        ax.set(xlim=lims, xlabel=r'$\nu_x$' + f' [cy/{self.spatial_unit}]',
               ylim=lims, ylabel=r'$\nu_y$' + f' [cy/{self.spatial_unit}]')

        cb = fig.colorbar(im,
                          label='PSD [' + self.phase_unit + '²' + f'/(cy/{self.spatial_unit})' + '²]',
                          ax=ax, fraction=0.046, extend='both')
        cb.outline.set_edgecolor('k')
        cb.outline.set_linewidth(0.5)

        return fig, ax

    def plot_psd_slices(self, x=True, y=True, azavg=True, azmin=False, azmax=False,
                        a=None, b=None, c=None, mode='freq', alpha=1, legend=True,
                        lw=config.lw, zorder=config.zorder, xlim=None, ylim=None, fig=None, ax=None):
        """Plot the x, y, and average PSD on a linear x axis.

        Parameters
        ----------
        x : `bool`, optional
            whether to plot the "x" PSD
        y : `bool`, optional
            whether to plot the "y" PSD
        azavg: `bool`, optional
            whether to plot the azimuthally averaged PSD
        azmin : `bool`, optional
            whether to plot the azimuthal minimum PSD
        azmax : `bool`, optional
            whether to plot the azimuthal maximum PSD
        a : `float`, optional
            a coefficient of Lorentzian PSD model plotted alongside data
        b : `float`, optional
            b coefficient of Lorentzian PSD model plotted alongside data
        c : `float`, optional
            c coefficient of Lorentzian PSD model plotted alongside data
        mode : `str`, {'freq', 'period'}
            x-axis mode, either frequency or period
        alpha : `float`, optional
            alpha value for the line(s), passed directly to matplotlib
        legend : `bool`, optional
            if True, display the legend
        lw : `float`, optional
            linewidth provided directly to matplotlib
        zorder : `int`, optional
            zorder provided directly to matplotlib
        xlim : `tuple`, optional
            len 2 tuple of low, high x axis limits
        ylim : `tuple`, optional
            len 2 tuple of low, high y axis limits
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

        Notes
        -----
        if a, b given but not c, an AB / inverse power model will be used for the PSD.
        If a, b, c are given the Lorentzian model will be used.

        """
        data = self.psd_slices(x=x, y=y, azavg=azavg, azmin=azmin, azmax=azmax)
        keys = list(data.keys())
        # keys 0 => first item
        # second 0 => first item in tuple of (unit, value)
        # 1: => skip the 0 frequency bin
        r = data[keys[0]][0][1:]

        if mode != 'freq':
            label = 'Period'
            unit = self.spatial_unit
        else:
            label = 'Frequency'
            unit = f'cy/{self.spatial_unit}'

        fig, ax = share_fig_ax(fig, ax)
        for dat in data:
            ax.loglog(*data[dat], lw=lw, label=dat, alpha=alpha)

        if a is not None:
            if c is not None:
                requirement = abc_psd(a=a, b=b, c=c, nu=r)
            else:
                requirement = ab_psd(a=a, b=b, nu=r)
            ax.loglog(r, requirement, c='k', lw=lw*2)

        if mode != 'freq':
            from matplotlib import pyplot as plt
            locs, labs = plt.xticks()
            labs = [str(1/loc) if loc != 0 else str(loc) for loc in locs]
            plt.xticks(locs, labs)
            xlim = [1/x if x != 0 else x for x in xlim]

        if legend:
            ax.legend(title='Slice')

        ax.set(xlim=xlim, xlabel=f'Spatial {label} [{unit}]',
               ylim=ylim, ylabel=r'PSD [' + f'{self.phase_unit}²/(cy/{self.spatial_unit})²]')

        return fig, ax

    def save_zygo_ascii(self, file, high_phase_res=True):
        """Save the interferogram to a Zygo ASCII file.

        Parameters
        ----------
        file : Path_like, `str`, or File_like
            where to save to.

        """
        phase = self.change_phase_unit(to='waves', inplace=False)
        write_zygo_ascii(file, phase=phase,
                         x=self.x, y=self.y,
                         intensity=None, wavelength=self.wavelength,
                         high_phase_res=high_phase_res)

    @staticmethod
    def from_zygo_dat(path, multi_intensity_action='first', scale='mm'):
        """Create a new interferogram from a zygo dat file.

        Parameters
        ----------
        path : path_like
            path to a zygo dat file
        multi_intensity_action : str, optional
            see `io.read_zygo_dat`
        scale : `str`, optional, {'um', 'mm'}
            what xy scale to label the data with, microns or mm

        Returns
        -------
        `Interferogram`
            new Interferogram instance

        """
        if str(path).endswith('datx'):
            # datx file, use datx reader
            zydat = read_zygo_datx(path)
            res = zydat['meta']['Lateral Resolution']
        else:
            # dat file, use dat file reader
            zydat = read_zygo_dat(path, multi_intensity_action=multi_intensity_action)
            res = zydat['meta']['lateral_resolution']  # meters

        phase = zydat['phase']
        if res == 0.0:
            res = 1
            scale = 'px'

        if scale != 'px':
            _scale = 'm'
        else:
            _scale = 'px'

        i = Interferogram(phase=phase, intensity=zydat['intensity'],
                          x=e.arange(phase.shape[1]) * res, y=e.arange(phase.shape[0]) * res,
                          scale=_scale, meta=zydat['meta'])
        return i.change_spatial_unit(to=scale.lower(), inplace=True)

    @staticmethod  # NOQA
    def render_from_psd(size, samples, rms=None,
                        mask='circle', phase_unit='nm', psd_fcn=abc_psd, **psd_fcn_kwargs):
        """Render a synthetic surface with a given RMS value given a PSD function.

        Parameters
        ----------
        size : `float`
            diameter of the output surface, mm
        samples : `int`
            number of samples across the output surface
        rms : `float`
            desired RMS value of the output, if rms=None, no normalization is done
        mask : `str`, optional
            mask defining the clear aperture
        psd_fcn : `callable`
            function used to generate the PSD
        **psd_fcn_kwargs:
            keyword arguments passed to psd_fcn in addition to nu
            if psd_fcn == abc_psd, kwargs are a, b, c
            elif psd_Fcn == ab_psd kwargs are a, b

            kwargs will be user-defined for user PSD functions

        Returns
        -------
        `Interferogram`
            new interferogram instance

        """
        x, y, z = render_synthetic_surface(size=size, samples=samples, rms=rms,
                                           mask=mask, psd_fcn=psd_fcn, **psd_fcn_kwargs)
        return Interferogram(phase=z, x=x, y=y, scale='mm', phase_unit=phase_unit)
