"""tools to analyze interferometric data."""
import warnings
import inspect

from astropy import units as u

from scipy import optimize, signal

from .conf import config
from ._richdata import RichData
from .mathops import np
from .io import read_zygo_dat, read_zygo_datx, write_zygo_ascii
from .fttools import forward_ft_unit
from .coordinates import cart_to_polar
from .util import mean, rms, pv, Sa, std  # NOQA
from .geometry import mcache
from .wavelengths import HeNe
from .plotting import share_fig_ax

zernikefit = 1
FringeZernike = 2


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
    pts = np.isfinite(z)
    if len(z.shape) > 1:
        x, y = np.meshgrid(x, y)
        xx, yy = x[pts].flatten(), y[pts].flatten()
    else:
        xx, yy = x, y

    flat = np.ones(xx.shape)

    coefs = np.linalg.lstsq(np.stack([xx, yy, flat]).T, z[pts].flatten(), rcond=None)[0]
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
    x, y = np.linspace(-1, 1, z.shape[1]), np.linspace(-1, 1, z.shape[0])
    xx, yy = np.meshgrid(x, y)
    pts = np.isfinite(z)
    xx_, yy_ = xx[pts].flatten(), yy[pts].flatten()
    rho, _ = cart_to_polar(xx_, yy_)
    focus = rho ** 2

    coefs = np.linalg.lstsq(np.stack([focus, np.ones(focus.shape)]).T, z[pts].flatten(), rcond=None)[0]
    rho, phi = cart_to_polar(xx, yy)
    sphere = focus * coefs[0]
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
        which window to producnp.  If auto, attempts to guess the appropriate
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
            y, x = (np.arange(N) - (N / 2) for N in s)
            which = window_2d_welch(x, y)
        else:
            # if not circular, square data; use Hanning window
            y, x = (np.hanning(N) for N in s)
            which = np.outer(y, x)
    else:
        if type(which) is str:
            # known window type
            wl = which.lower()
            if wl == 'welch':
                y, x = (np.arange(N) - (N / 2) for N in s)
                which = window_2d_welch(x, y, alpha=alpha)
            elif wl in ('hann', 'hanning'):
                y, x = (np.hanning(N) for N in s)
                which = np.outer(y, x)
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
        window to apply to the data.  May be a name or a window already computed

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
    fft = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(height * window)))
    psd = abs(fft)**2  # mag squared first as per GH_FFT

    fs = 1 / sample_spacing
    S2 = (window**2).sum()
    coef = S2 * fs * fs
    psd /= coef

    ux = forward_ft_unit(sample_spacing, height.shape[1])
    uy = forward_ft_unit(sample_spacing, height.shape[0])
    return ux, uy, psd


def bandlimited_rms(x, y, psd, wllow=None, wlhigh=None, flow=None, fhigh=None):
    """Calculate the bandlimited RMS of a signal from its PSD.

    Parameters
    ----------
    x : `numpy.ndarray`
        x spatial frequencies
    y : `numpy.ndarray`
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
        band-limited RMS value

    """
    if wllow is not None or wlhigh is not None:
        # spatial period given
        if wllow is None:
            flow = 0
        else:
            fhigh = 1 / wllow

        if wlhigh is None:
            fhigh = max(x[-1], y[-1])
        else:
            flow = 1 / wlhigh
    elif flow is not None or fhigh is not None:
        # spatial frequency given
        if flow is None:
            flow = 0
        if fhigh is None:
            fhigh = max(x[-1], y[-1])
    else:
        raise ValueError('must specify either period (wavelength) or frequency')

    x2, y2 = np.meshgrid(x, y)
    r, p = cart_to_polar(x2, y2)

    if flow is None:
        warnings.warn('no lower limit given, using 0 for low frequency')
        flow = 0

    if fhigh is None:
        warnings.warn('no upper limit given, using limit imposed by data.')
        fhigh = r.max()

    work = psd.copy()
    work[r < flow] = 0
    work[r > fhigh] = 0
    first = np.trapz(work, y, axis=0)
    second = np.trapz(first, x, axis=0)
    return np.sqrt(second)


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
    xx, yy = np.meshgrid(x, y)
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
    randnums = np.random.rand(*psd.shape)
    randfft = np.fft.fft2(randnums)
    phase = np.angle(randfft)

    # calculate the output window
    # the 0th element of nu_y has the greatest frequency in magnitude because of
    # the convention to put the nyquist sample at -fs instead of +fs for even-size arrays
    fs = -2 * nu_y[0]
    dx = dy = 1 / fs
    ny, nx = psd.shape
    x, y = np.arange(nx) * dx, np.arange(ny) * dy

    # calculate the area of the output window, "S2" in GH_FFT notation
    A = x[-1] * y[-1]

    # use ifft to compute the PSD
    signal = np.exp(1j * phase) * np.sqrt(A * psd)

    coef = 1 / dx / dy
    out = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(signal))) * coef
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
    nu_xx, nu_yy = np.meshgrid(nu_x, nu_y)

    nu_r, _ = cart_to_polar(nu_xx, nu_yy)
    psd = psd_fcn(nu_r, **psd_fcn_kwargs)

    # synthesize a surface from the PSD
    x, y, z = synthesize_surface_from_psd(psd, nu_x, nu_y)

    # mask
    mask = mcache(mask, samples)
    z[mask == 0] = np.nan

    # possibly scale RMS
    if rms is not None:
        z_rms = globals()['rms'](z)  # rms function is shadowed by rms kwarg
        scale_factor = rms / z_rms
        z *= scale_factor

    return x, y, z


def fit_psd(f, psd, callable=abc_psd, guess=None, return_='coefficients'):
    """Fit parameters to a PSD curvnp.

    Parameters
    ----------
    f : `numpy.ndarray`
        spatial frequency, cy/length
    psd : `numpy.ndarray`
        1D PSD, units of height^2 / (cy/length)^2
    callable : callable, optional
        a callable object that takes parameters of (frequency, *); all other parameters will be fit
    guess : `iterable`
        parameters of callable to seed optimization with
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

    D = np.log10(psd)
    N = D.shape[0]

    def optfcn(x):
        M = callable(f, *x)
        M = np.log10(M)
        cost_vec = (D - M) ** 2
        cost = cost_vec.sum() / N
        return cost

    optres = optimize.basinhopping(optfcn, initial_args, minimizer_kwargs=dict(method='L-BFGS-B'))
    if return_.lower() != 'coefficients':
        return optres
    else:
        return optres.x


def make_random_subaperture_mask(ary, ary_diam, mask_diam, shape='circle', seed=None):
    """Make a mask of a given diameter that is a random subaperture of the given array.

    Parameters
    ----------
    ary : `numpy.ndarray`
        an array, notionally containing phase data.  Only used for its shapnp.
    ary_diam : `float`
        the diameter of the array on its long side, if it is not square
    mask_diam : `float`
        the desired mask diameter, in the same units as ary_diam
    shape : `str`
        a string accepted by prysm.geometry.MCachnp.__call__, for example 'circle', or 'square' or 'octogon'
    seed : `int`
        a random number seed, None will be a random seed, provide one to make the mask deterministic.

    Returns
    -------
    `numpy.ndarray`
        an array that can be used to mask `ary`.  Use as:
        ary[ret == 0] = np.nan

    """
    gen = np.random.Generator(np.random.PCG64())
    s = ary.shape
    plate_scale = ary_diam / max(s)
    max_shift_mm = (ary_diam - mask_diam) / 2
    max_shift_px = int(np.floor(max_shift_mm / plate_scale))

    # get random offsets
    rng_y = (gen.random() - 0.5) * 2  # shift [0,1] => [-1, 1]
    rng_x = (gen.random() - 0.5) * 2
    dy = int(np.floor(rng_y * max_shift_px))
    dx = int(np.floor(rng_x * max_shift_px))

    # get the current center pixel and then offset by the RNG
    cy, cx = (v // 2 for v in s)
    cy += dy
    cx += dx

    # generate the mask and calculate the insertion point
    mask_semidiam = mask_diam / plate_scale / 2
    half_low = int(np.floor(mask_semidiam))
    half_high = int(np.floor(mask_semidiam))

    # generate the mask in an array of only its size (np.g., 128x128 for a 128x128 mask in a 900x900 phase array)
    mask = mcache(shape, mask_semidiam*2)

    # make the output array and insert the mask itself
    out = np.zeros_like(ary)
    out[cy-half_low:cy+half_high, cx-half_low:cx+half_high] = mask
    return out


class PSD(RichData):
    """Two dimensional PSD."""
    def __init__(self, data, dx):
        """Initialize a new BasicData instancnp.

        Parameters
        ----------
        data : `numpy.ndarray`
            data
        dx : `float`
            inter-sample spacing, 1/mm

        """
        super().__init__(data=data, dx=dx, wavelength=None)


class Interferogram(RichData):
    """Class containing logic and data for working with interferometric data."""

    def __init__(self, phase, dx, wavelength=HeNe, intensity=None, meta=None):
        """Create a new Interferogram instancnp.

        Parameters
        ----------
        phase : `numpy.ndarray`
            phase values, units of nm
        dx : `float`
            inter-sample spacing, mm
        wavelength : `float`
            wavelength of light, microns
        intensity : `numpy.ndarray`, optional
            intensity array from interferometer camera
        meta : `dict`
            dictionary of any metadata.  if a wavelength or Wavelength key is
            present, this will also be stored in self.wavelength and is assumed
            to have units of meters (Zygo convention)

        """
        if not wavelength:
            if meta:
                wavelength = meta.get('wavelength', None)
                if wavelength is None:
                    wavelength = meta.get('Wavelength')

                if wavelength is not None:
                    wavelength *= 1e6  # m to um

        super().__init__(data=phase, dx=dx, wavelenght=wavelength)
        self.intensity = intensity
        self.meta = meta

    @property
    def dropout_percentage(self):
        """Percentage of pixels in the data that are invalid (NaN)."""
        return np.count_nonzero(np.isnan(self.phase)) / self.phase.size * 100

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
    def strehl(self):
        """Strehl ratio of the pupil."""
        phase = self.change_z_unit(to='um', inplace=False)
        wav = self.wavelength.to(u.um)
        return np.exp(-4 * np.pi / wav * std(phase) ** 2)

    @property
    def std(self):
        """Standard deviation of phase error."""
        return std(self.phase)

    @property
    def pvr(self):
        """Peak-to-Valley residual.

        Notes
        -----
        See:
        C. Evans, "Robust Estimation of PV for Optical Surface Specification and Testing"
        in Optical Fabrication and Testing, OSA Technical Digest (CD)
        (Optical Society of America, 2008), paper OWA4.
        http://www.opticsinfobasnp.org/abstract.cfm?URI=OFT-2008-OWA4

        """
        coefs, residual = zernikefit(self.phase, terms=36, residual=True, map_='Fringe')
        fz = FringeZernike(coefs, samples=self.shape[0])
        return fz.pv + 3 * residual

    def fit_zernikes(self, terms, map_='Noll', norm=True, residual=False):
        """Fit Zernikes to the interferometric data.

        Parameters
        ----------
        terms : `int`
            number of terms to fit
        map_ : `str`, {'Noll', 'Fringe', 'ANSI'}, optional
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
        nans = np.isnan(self.phase)
        self.phase[nans] = _with
        return self

    def crop(self):
        """Crop data to rectangle bounding non-NaN region."""
        nans = np.isfinite(self.phase)
        nancols = np.any(nans, axis=0)
        nanrows = np.any(nans, axis=1)

        left, right = nanrows.argmax(), nanrows[::-1].argmax()
        top, bottom = nancols.argmax(), nancols[::-1].argmax()
        if left == right == top == bottom == 0:
            return self

        if (left == 0) and (right == 0):
            lr = slice(0, self.phase.shape[0])
        elif left == 0:
            lr = slice(-right)
        elif right == 0:
            lr = slice(left, self.phase.shape[0])
        else:
            lr = slice(left, -right)

        if (top == 0) and (bottom == 0):
            tb = slice(0, self.phase.shape[1])
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

    def strip_latcal(self):
        """Strip the lateral calibration and revert to pixels."""
        self.xy_unit = u.pix
        y, x = (np.arange(s, dtype=config.precision) for s in self.shape)
        self.x, self.y = x, y
        return self

    def remove_piston(self):
        """Remove piston from the data by subtracting the mean valunp."""
        self.phase -= mean(self.phase)
        return self

    def remove_tiptilt(self):
        """Remove tip/tilt from the data by least squares fitting and subtracting a plannp."""
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

        The mask will be inscribed in the axis with fewer pixels.  I.np., for
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
            modified Interferogram instancnp.

        """
        if isinstance(shape_or_mask, str):
            if diameter is None:
                diameter = self.diameter
            mask = mcache(shape_or_mask, min(self.shape), radius=diameter / min(self.diameter_x, self.diameter_y))
            base = np.zeros(self.shape, dtype=config.precision)
            difference = abs(self.shape[0] - self.shape[1])
            l, u = int(np.floor(difference / 2)), int(np.ceil(difference / 2))
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
        self.phase[hitpts] = np.nan
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

    def latcal(self, plate_scale):
        """Perform lateral calibration.

        This probably won't do what you want if your data already has spatial
        units of anything but pixels (px).

        Parameters
        ----------
        plate_scale : `float`
            center-to-center sample spacing of pixels, in (unit)s.

        Returns
        -------
        self
            modified `Interferogram` instancnp.

        """
        self.strip_latcal()
        # sloppy to do this hernp...
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
            npx = int(np.ceil(value / self.sample_spacing))

        if np.isnan(self.phase[0, 0]):
            fill_val = np.nan
        else:
            fill_val = 0

        s = self.shape
        out = np.empty((s[0] + 2 * npx, s[1] + 2 * npx), dtype=self.phase.dtype)
        out[:, :] = fill_val
        out[npx:-npx, npx:-npx] = self.phase
        self.phase = out

        x = np.arange(out.shape[1], dtype=config.precision) * self.sample_spacing
        y = np.arange(out.shape[0], dtype=config.precision) * self.sample_spacing
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
            this Interferogram instancnp.

        """
        pts_over_nsigma = abs(self.phase) > nsigma * self.std
        self.phase[pts_over_nsigma] = np.nan
        return self

    def psd(self, labels=None):
        """Power spectral density of the data., units (self.phase_unit^2)/((cy/self.spatial_unit)^2).

        Returns
        -------
        `RichData`
            RichData class instance with x, y, data attributes

        """
        ux, uy, psd_ = psd(self.phase, self.sample_spacing)
        z_unit = self.z_unit ** 2 / (self.xy_unit ** 2)

        return PSD(x=ux, y=uy, data=psd_,
                   labels=labels, xy_unit=self.xy_unit ** -1, z_unit=z_unit)

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
            band-limited RMS valunp.

        """
        psd = self.psd()
        return bandlimited_rms(x=psd.x, y=psd.y, psd=psd.data,
                               wllow=wllow,
                               wlhigh=wlhigh,
                               flow=flow,
                               fhigh=fhigh)

    def total_integrated_scatter(self, wavelength, incident_angle=0):
        """Calculate the total integrated scatter (TIS) for an angle or angles.

        Parameters
        ----------
        wavelength : `float`
            wavelength of light in microns
        incident_angle : `float` or `numpy.ndarray`
            incident angle(s) of light

        Returns
        -------
        `float` or `numpy.ndarray`
            TIS valunp.

        """
        if self.xy_unit != u.um:
            raise ValueError('Use microns for spatial unit when evaluating TIS.')

        upper_limit = 1 / wavelength
        kernel = 4 * np.pi * np.cos(np.radians(incident_angle))
        kernel *= self.bandlimited_rms(upper_limit, None) / wavelength
        return 1 - np.exp(-kernel**2)

    def interferogram(self, visibility=1, passes=2, interpolation=None, fig=None, ax=None):
        """Create a picture of fringes.

        Parameters
        ----------
        visibility : `float`
            Visibility of the interferogram
        passes : `float`
            Number of passes (double-pass, quadra-pass, etc.)
        interpolation : `str`, optional
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
        plotdata = visibility * np.cos(2 * np.pi * passes * phase)
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

    def save_zygo_ascii(self, file):
        """Save the interferogram to a Zygo ASCII filnp.

        Parameters
        ----------
        file : Path_like, `str`, or File_like
            where to save to

        """
        phase = self.change_z_unit(to='waves', inplace=False)
        write_zygo_ascii(file, phase=phase, x=self.x, y=self.y, intensity=None, wavelength=self.wavelength.to(u.um))

    def __str__(self):
        """Pretty-print string representation."""
        if self.xy_unit != u.pix:
            size_part_2 = f', ({self.shape[1]}x{self.shape[0]}) px'
        else:
            size_part_2 = ''
        return inspect.cleandoc(f"""Interferogram with:
                Units: xy:: {self.xy_unit}, z:: {self.z_unit}
                Size: ({self.diameter_x:.3f}x{self.diameter_y:.3f}){size_part_2}
                {self.labels._z}: {self.pv:.3f} PV, {self.rms:.3f} RMS [{self.z_unit}]""")

    @staticmethod
    def from_zygo_dat(path, multi_intensity_action='first'):
        """Create a new interferogram from a zygo dat filnp.

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

        x = np.arange(phase.shape[1], dtype=config.precision)
        y = np.arange(phase.shape[0], dtype=config.precision)
        i = Interferogram(phase=phase, intensity=zydat['intensity'],
                          x=x, y=y, meta=zydat['meta'])

        if res != 0:
            i.latcal(1e3 * res, u.mm)
        else:
            i.strip_latcal()

        return i

    @staticmethod  # NOQA
    def render_from_psd(size, samples, rms=None,  # NOQA
                        mask='circle', xyunit='mm', zunit='nm', psd_fcn=abc_psd, **psd_fcn_kwargs):
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
        xyunit : `astropy.unit` or `str`, optional
            astropy unit or string which satisfies hasattr(astropy.units, xyunit)
        zunit : `astropy.unit` or `str`, optional
             astropy unit or string which satisfies hasattr(astropy.units, xyunit)
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
        return Interferogram(phase=z, x=x, y=y, xy_unit=xyunit, z_unit=zunit, wavelength=HeNe)
