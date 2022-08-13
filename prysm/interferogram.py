"""tools to analyze interferometric data."""
import warnings
import inspect
import random
import math

from scipy import optimize

from .conf import config
from ._richdata import RichData
from .mathops import np, fft, jinc
from .io import (
    read_zygo_dat,
    read_zygo_datx,
    write_zygo_ascii
)
from .fttools import forward_ft_unit
from .coordinates import (
    cart_to_polar,
    broadcast_1d_to_2d,
    make_xy_grid,
    optimize_xy_separable
)
from prysm.polynomials import lstsq, mode_1d_to_2d
from .util import mean, rms, pv, Sa, std  # NOQA
from .wavelengths import HeNe
from .plotting import share_fig_ax


def _rmax_square_array(r):
    loc = list(r.shape)
    loc[1] = loc[1] // 2
    loc[0] = loc[0] - 1
    loc = tuple(loc)
    rmax = r[loc]
    return rmax


def fit_plane(x, y, z):
    """Fit a plane to data.

    Parameters
    ----------
    x : numpy.ndarray
        2D array of x (axis 1) values
    y : numpy.ndarray
        2D array of y (axis 0) values
    z : numpy.ndarray
        2D array of z values

    Returns
    -------
    numpy.ndarray
        array representation of plane

    """
    xx, yy = optimize_xy_separable(x, y)

    mode1 = xx
    mode2 = yy
    mode1 = mode_1d_to_2d(mode1, x, y, 'x')
    mode2 = mode_1d_to_2d(mode2, x, y, 'y')

    coefs = lstsq([mode1, mode2], z)
    plane_fit = coefs[0] * mode1 + coefs[1] * mode2
    return plane_fit


def fit_sphere(z):
    """Fit a sphere to data.

    Parameters
    ----------
    z : numpy.ndarray
        2D array of data

    Returns
    -------
    numpy.ndarray, numpy.ndarray
        mask, sphere

    """
    x, y = np.linspace(-1, 1, z.shape[1]), np.linspace(-1, 1, z.shape[0])
    xx, yy = np.meshgrid(x, y)
    pts = np.isfinite(z)
    xx_, yy_ = xx[pts].flatten(), yy[pts].flatten()
    rho = np.sqrt(xx_**2 + yy_**2)
    focus = rho ** 2

    coefs = np.linalg.lstsq(np.stack([focus.flatten(), np.ones(focus.shape)]).T, z[pts].flatten(), rcond=None)[0]
    rho, phi = cart_to_polar(xx, yy)
    sphere = focus * coefs[0]
    return pts, sphere


def make_window(signal, dx, which=None, alpha=4):
    """Generate a window function to be used in PSD analysis.

    Parameters
    ----------
    signal : numpy.ndarray
        signal or phase data
    dx : float
        spacing of samples in the input data
    which : str, {'welch', 'hann', None}, optional
        which window to producnp.  If auto, attempts to guess the appropriate
        window based on the input signal
    alpha : float, optional
        alpha value for welch window

    Notes
    -----
    For 2D welch, see:
    Power Spectral Density Specification and Analysis of Large Optical Surfaces
    E. Sidick, JPL

    Returns
    -------
    numpy.ndarray
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
            x, y = make_xy_grid(s, dx=dx)
            r, _ = cart_to_polar(x, y)
            which = window_2d_welch(r, alpha=alpha)
        else:
            # if not circular, square data; use Hanning window
            y, x = (np.hanning(N) for N in s)
            which = np.outer(y, x)
    else:
        if type(which) is str:
            # known window type
            wl = which.lower()
            if wl == 'welch':
                x, y = make_xy_grid(s, dx=dx)
                r, _ = cart_to_polar(x, y)
                which = window_2d_welch(r, alpha=alpha)
            elif wl in ('hann', 'hanning'):
                y, x = (np.hanning(N) for N in s)
                which = np.outer(y, x)
            else:
                raise ValueError('unknown window type')

    return which  # window provided as ndarray


def psd(height, dx, window=None):
    """Compute the power spectral density of a signal.

    Parameters
    ----------
    height : numpy.ndarray
        height or phase data
    dx : float
        spacing of samples in the input data
    window : {'welch', 'hann'} or ndarray, optional
        window to apply to the data.  May be a name or a window already computed

    Returns
    -------
    x : numpy.ndarray
        ordinate x frequency axis
    y : numpy.ndarray
        ordinate y frequency axis
    psd : numpy.ndarray
        power spectral density

    Notes
    -----
    See GH_FFT for a rigorous treatment of FFT scalings
    https://holometer.fnal.gov/GH_FFT.pdf

    """
    window = make_window(height, dx, window)
    ft = fft.ifftshift(fft.fft2(fft.fftshift(height * window)))
    psd = abs(ft)**2  # mag squared first as per GH_FFT

    fs = 1 / dx
    S2 = (window**2).sum()
    coef = S2 * fs * fs
    psd /= coef

    ux = forward_ft_unit(dx, height.shape[1])
    uy = forward_ft_unit(dx, height.shape[0])
    ux, uy = broadcast_1d_to_2d(ux, uy)
    return ux, uy, psd


def bandlimited_rms(r, psd, wllow=None, wlhigh=None, flow=None, fhigh=None):
    """Calculate the bandlimited RMS of a signal from its PSD.

    Parameters
    ----------
    r : numpy.ndarray
        radial spatial frequencies
    psd : numpy.ndarray
        power spectral density
    wllow : float
        short spatial scale
    wlhigh : float
        long spatial scale
    flow : float
        low frequency
    fhigh : float
        high frequency

    Returns
    -------
    float
        band-limited RMS value

    """
    default_max = r.max()
    if wllow is not None or wlhigh is not None:
        # spatial period given
        if wllow is None:
            flow = 0
        else:
            fhigh = 1 / wllow

        if wlhigh is None:
            fhigh = default_max
        else:
            flow = 1 / wlhigh
    elif flow is not None or fhigh is not None:
        # spatial frequency given
        if flow is None:
            flow = 0
        if fhigh is None:
            fhigh = default_max
    else:
        raise ValueError('must specify either period (wavelength) or frequency')

    if flow is None:
        warnings.warn('no lower limit given, using 0 for low frequency')
        flow = 0

    if fhigh is None:
        warnings.warn('no upper limit given, using limit imposed by data.')
        fhigh = r.max()

    work = psd.copy()
    work[r < flow] = 0
    work[r > fhigh] = 0
    if r.ndim == 2:
        c = tuple(s//2 for s in work.shape)
        c2 = list(c)
        c2[0] = c2[0] - 1
        c2 = tuple(c2)
        pt1 = r[c]
        pt2 = r[c2]
    else:
        c = r.shape[0]//2
        pt1 = r[c]
        pt2 = r[c-1]
    # prysm doesn't enforce the user to be "top left" or "lower left" origin,
    # abs makes sure we do things right no matter what
    dx = abs(pt2 - pt1)
    reduced = np.trapz(work, dx=dx, axis=0)

    if r.ndim == 2:
        reduced = np.trapz(reduced, dx=dx, axis=0)

    return np.sqrt(reduced)


def window_2d_welch(r, alpha=8):
    """Return a 2D welch window for a given alpha.

    Parameters
    ----------
    r : numpy.ndarray
        radial coordinate
    alpha : float
        alpha (edge roll) parameter

    Returns
    -------
    numpy.ndarray
        window

    """
    rmax = _rmax_square_array(r)
    window = 1 - abs(r/rmax)**alpha
    return window


def abc_psd(nu, a, b, c):
    """Lorentzian model of a Power Spectral Density.

    Parameters
    ----------
    nu : numpy.ndarray or float
        spatial frequency
    a : float
        a coefficient
    b : float
        b coefficient
    c : float
        c coefficient

    Returns
    -------
    numpy.ndarray
        value of PSD model

    """
    return a / (1 + (nu/b)**c)


def ab_psd(nu, a, b):
    """Inverse power model of a Power Spectral Density.

    Parameters
    ----------
    nu : numpy.ndarray or float
        spatial frequency
    a : float
        a coefficient
    b : float
        b coefficient

    Returns
    -------
    numpy.ndarray
        value of PSD model

    """
    return a * nu ** (-b)


def synthesize_surface_from_psd(psd, nu_x, nu_y):
    """Synthesize a surface height map from PSD data.

    Parameters
    ----------
    psd : numpy.ndarray
        PSD data, units nm²/(cy/mm)²
    nu_x : numpy.ndarray
        x spatial frequency, cy/mm
    nu_y : numpy.ndarray
        y spatial frequency, cy_mm

    """
    # generate a random phase to be matched to the PSD
    randnums = np.random.rand(*psd.shape)
    randfft = fft.fft2(randnums)
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
    out = fft.ifftshift(fft.ifft2(fft.fftshift(signal))) * coef
    out = out.real
    return x, y, out


def render_synthetic_surface(size, samples, rms=None, mask=None, psd_fcn=abc_psd, **psd_fcn_kwargs):  # NOQA
    """Render a synthetic surface with a given RMS value given a PSD function.

    Parameters
    ----------
    size : float
        diameter of the output surface, mm
    samples : int
        number of samples across the output surface
    rms : float, optional
        desired RMS value of the output, if rms=None, no normalization is done
    mask : numpy.ndarray, optional
        mask defining the pupil aperture
    psd_fcn : callable
        function used to generate the PSD
    **psd_fcn_kwargs:
        keyword arguments passed to psd_fcn in addition to nu
        if psd_fcn == abc_psd, kwargs are a, b, c
        elif psd_Fcn == ab_psd kwargs are a, b

        kwargs will be user-defined for user PSD functions

    Returns
    -------
    x : numpy.ndarray
        x coordinates, mm
    y: numpy.ndarray
        y coordinates, mm
    z : numpy.ndarray
        height data, nm

    """
    # compute the grid and PSD
    dxg = size / (samples - 1)
    nu_x = nu_y = forward_ft_unit(dxg, samples)
    center = samples // 2  # some bullshit here to gloss over zeros for ab_psd
    nu_x[center] = nu_x[center+1] / 10
    nu_y[center] = nu_y[center+1] / 10
    nu_xx, nu_yy = np.meshgrid(nu_x, nu_y)

    nu_r, _ = cart_to_polar(nu_xx, nu_yy)
    psd = psd_fcn(nu_r, **psd_fcn_kwargs)

    # synthesize a surface from the PSD
    x, y, z = synthesize_surface_from_psd(psd, nu_x, nu_y)

    # mask
    if mask is not None:
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
    f : numpy.ndarray
        spatial frequency, cy/length
    psd : numpy.ndarray
        1D PSD, units of height^2 / (cy/length)^2
    callable : callable, optional
        a callable object that takes parameters of (frequency, *); all other parameters will be fit
    guess : iterable
        parameters of callable to seed optimization with
    return_ : str, optional, {'coefficients', 'optres'}
        what to return; either return the coefficients (optres.x) or the optimization result (optres)

    Returns
    -------
    optres
        scipy.optimization.OptimizationResult
    coefficients
        numpy.ndarray of coefficients

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


def hann2d(M, N):
    """Hanning window in 2D."""
    # M = num rows
    # N = num cols
    n = np.arange(N)[np.newaxis, :] - (N//2)
    m = np.arange(M)[:, np.newaxis] - (M//2)

    nn = np.hypot(n, m)
    N = min(N, M)
    w = np.cos(np.pi/N * nn) ** 2
    w[nn > N // 2] = 0
    return w


def ideal_lpf_iir2d(r, dx, fc_over_nyq):
    """Ideal impulse response of a 2D lowpass filter."""
    c = np.pi * fc_over_nyq / dx
    # fc/nyq^2 * pi = area of circle; /2 = jinc has peak of 1
    return jinc(r*c) * (fc_over_nyq**2 * np.pi / 2)


def designfilt2d(r, dx, fc, typ='lowpass'):
    """Design a rotationally symmetric filter for 2D data.

    Parameters
    ----------
    r : numpy.ndarray
        radial coordinates of data to be filtered
    dx : float
        sample spacing of r
    fc : float or tuple of 2 floats
        corner frequency of the filter if low or high pass, lower and upper
        frequencies for band pass and reject filters
    typ : str, {'lowpass' , 'lp', 'highpass', 'hp', 'bandpass', 'bp', 'bandreject', 'br'}
        what type of filter.  Can use two-letter shorthands.

    Returns
    -------
    numpy.ndarray
        2D array containing the infinite impulse response, h.
        Convolution of the data with this "PSF" will produce
        the desired spectral filtering

    """
    w = hann2d(*r.shape)
    nyq = 1 / (2*dx)
    tl = typ.lower()
    if tl in ('lp', 'lowpass'):
        fc_over_nyq = fc/nyq
        h = ideal_lpf_iir2d(r, dx, fc_over_nyq)
        hprime = w * h
        H = fft.fft2(hprime)
        H = abs(H)
    elif tl in ('hp', 'highpass'):
        fc_over_nyq = fc/nyq
        h = ideal_lpf_iir2d(r, dx, fc_over_nyq)
        hprime = w * h
        H = fft.fft2(hprime)
        H = abs(H)
        H = 1 - H
    elif tl in ('bp', 'bandpass'):
        # bandpass is made by producing the transfer function of low and high pass,
        # then combining them
        hl = ideal_lpf_iir2d(r, dx, fc[0]/nyq)
        hh = ideal_lpf_iir2d(r, dx, fc[1]/nyq)
        hlp = hl * w  # h_low prime
        hhp = hh * w
        Hl = fft.fft2(hlp)
        Hh = fft.fft2(hhp)
        Hl = abs(Hl)
        Hh = abs(Hh)
        Hh = 1 - Hh
        H = 1 - (Hh + Hl)
    elif tl in ('br', 'bandreject'):
        hl = ideal_lpf_iir2d(r, dx, fc[0]/nyq)
        hh = ideal_lpf_iir2d(r, dx, fc[1]/nyq)
        hlp = hl * w  # h_low prime
        hhp = hh * w
        Hl = fft.fft2(hlp)
        Hh = fft.fft2(hhp)
        Hl = abs(Hl)
        Hh = abs(Hh)
        Hh = 1 - Hh
        H = (Hh + Hl)

    return H


def make_random_subaperture_mask(shape, mask):
    """Make a mask of a given diameter that is a random subaperture of the given array.

    Parameters
    ----------
    shape : tuple
        length two tuple, containing (m, n) of the returned mask
    mask : numpy.ndarray
        mask to apply for sub-apertures

    Returns
    -------
    numpy.ndarray
        an array that can be used to mask ary.  Use as:
        ary[ret == 0] = np.nan

    """
    max_shift = [(s1-s2) for s1, s2 in zip(shape, mask.shape)]

    # get random offsets
    rng_y = random.random()
    rng_x = random.random()
    dy = math.floor(rng_y * max_shift[0])
    dx = math.floor(rng_x * max_shift[1])

    high_y = mask.shape[0] + dy
    high_x = mask.shape[1] + dx
    # make the output array and insert the mask itself
    out = np.zeros(shape, dtype=bool)
    out[dy:high_y, dx:high_x] = mask
    return out


class Interferogram(RichData):
    """Class containing logic and data for working with interferometric data."""

    def __init__(self, phase, dx=0, wavelength=HeNe, intensity=None, meta=None):
        """Create a new Interferogram instance.

        Parameters
        ----------
        phase : numpy.ndarray
            phase values, units of nm
        dx : float
            sample spacing in mm; if zero the data has no lateral calibration
            (xy scale only "px", not mm)
        wavelength : float
            wavelength of light, microns
        intensity : numpy.ndarray, optional
            intensity array from interferometer camera
        meta : dict
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

        super().__init__(data=phase, dx=dx, wavelength=wavelength)
        self.intensity = intensity
        self.meta = meta
        if dx == 0:
            self._latcaled = False
        else:
            self._latcaled = True

    @property
    def dropout_percentage(self):
        """Percentage of pixels in the data that are invalid (NaN)."""
        return np.count_nonzero(np.isnan(self.data)) / self.data.size * 100

    @property
    def pv(self):
        """Peak-to-Valley phase error.  DIN/ISO St."""
        return pv(self.data)

    @property
    def rms(self):
        """RMS phase error.  DIN/ISO Sq."""
        return rms(self.data)

    @property
    def Sa(self):
        """Sa phase error.  DIN/ISO Sa."""
        return Sa(self.data)

    @property
    def strehl(self):
        """Strehl ratio of the data, assuming it represents wavefront error."""
        # Welford, Aberrations of Optical Systems, Eq. (13.2), p.243

        # 1e3 um => nm, all units same
        wvl = self.wavelength * 1e3
        prefix = (4 * np.pi / wvl**2)
        coef = std(self.data) ** 2
        return 1 - prefix * coef

    @property
    def std(self):
        """Standard deviation of phase error."""
        return std(self.data)

    def pvr(self, normalization_radius=None):
        """Peak-to-Valley residual.

        Parameters
        ----------
        normalization_radius : float
            radius used to normalize the radial coordinate during Zernike computation.
            If None, the data array is assumed square and the radius is automatically
            chosen to be the radius of the array.

        Notes
        -----
        See:
        C. Evans, "Robust Estimation of PV for Optical Surface Specification and Testing"
        in Optical Fabrication and Testing, OSA Technical Digest (CD)
        (Optical Society of America, 2008), paper OWA4.
        http://www.opticsinfobase.org/abstract.cfm?URI=OFT-2008-OWA4

        """
        from prysm.polynomials import (
            zernike_nm_sequence,
            fringe_to_nm,
            lstsq,
            sum_of_2d_modes
        )

        r = self.r
        t = self.t
        if normalization_radius is None:
            shp = self.data.shape
            if shp[0] != shp[1]:
                raise ValueError('pvr: if normalization_radius is None, data must be square')

            normalization_radius = _rmax_square_array(r)

        r = r / normalization_radius
        mask = r > 1
        data = self.data.copy()
        data[mask] = np.nan

        nms = [fringe_to_nm(j) for j in range(1, 38)]  # 1 => 37; 36 terms
        basis = list(zernike_nm_sequence(nms, r, t, norm=False))  # slightly faster without norm, no need for pvr
        coefs = lstsq(basis, data)

        projected = sum_of_2d_modes(basis, coefs)
        projected[mask] = np.nan

        fit_err = data - projected
        rms_resid = rms(fit_err)
        pv_fit = pv(projected)

        pvr = pv_fit + 3 * rms_resid
        return pvr

    def fill(self, _with=0):
        """Fill invalid (NaN) values.

        Parameters
        ----------
        _with : float, optional
            value to fill with

        Returns
        -------
        Interferogram
            self

        """
        nans = np.isnan(self.data)
        self.data[nans] = _with
        return self

    def crop(self):
        """Crop data to rectangle bounding non-NaN region."""
        nans = np.isfinite(self.data)
        nancols = np.any(nans, axis=0)
        nanrows = np.any(nans, axis=1)

        left, right = nanrows.argmax(), nanrows[::-1].argmax()
        top, bottom = nancols.argmax(), nancols[::-1].argmax()
        if left == right == top == bottom == 0:
            return self

        if (left == 0) and (right == 0):
            lr = slice(0, self.data.shape[0])
        elif left == 0:
            lr = slice(-right)
        elif right == 0:
            lr = slice(left, self.data.shape[0])
        else:
            lr = slice(left, -right)

        if (top == 0) and (bottom == 0):
            tb = slice(0, self.data.shape[1])
        elif top == 0:
            tb = slice(-bottom)
        elif bottom == 0:
            tb = slice(top, self.data.shape[1])
        else:
            tb = slice(top, -bottom)

        self.data = self.data[lr, tb]
        # now cropped data, need to adjust coords
        # do nothing if they have not been computed
        if self._x is not None:
            self.x = self.x[lr, tb]
            self.y = self.y[lr, tb]
        if self._r is not None:
            self.r = self.r[lr, tb]
            self.t = self.t[lr, tb]

    def recenter(self):
        """Adjust the x and y coordinates so the data is centered on 0,0 in the FFT sense (contains a zero sample)."""
        c = tuple((s//2 for s in self.shape))
        self.x -= self.x[c]
        self.y -= self.y[c]
        self._r = None
        self._t = None
        return self

    def remove_piston(self):
        """Remove piston from the data by subtracting the mean valunp."""
        self.data -= mean(self.data)
        return self

    def remove_tiptilt(self):
        """Remove tip/tilt from the data by least squares fitting and subtracting a plannp."""
        plane = fit_plane(self.x, self.y, self.data)
        self.data -= plane
        return self

    def remove_power(self):
        """Remove power from the data by least squares fitting."""
        mask, sphere = fit_sphere(self.data)
        self.data[mask] -= sphere
        return self

    def mask(self, mask):
        """Mask the signal.

        Parameters
        ----------
        mask : numpy.ndarray
            binary ndarray indicating pixels to keep (True) and discard (False)

        Returns
        -------
        self
            modified Interferogram instance.

        """
        self.data[~mask] = np.nan
        return self

    def strip_latcal(self):
        """Strip the lateral calibration and revert to pixels."""
        self.y, self.x = (np.arange(s, dtype=config.precision) for s in self.shape)
        self._latcaled = False
        return self

    def latcal(self, plate_scale):
        """Perform lateral calibration.

        This probably won't do what you want if your data already has spatial
        units of anything but pixels (px).

        Parameters
        ----------
        plate_scale : float
            center-to-center sample spacing of pixels, in (unit)s.

        Returns
        -------
        self
            modified Interferogram instancnp.

        """
        self.strip_latcal()
        # sloppy to strip, but it is what it is
        self.x *= plate_scale
        self.y *= plate_scale
        self.dx = plate_scale
        self._latcaled = True
        return self

    def pad(self, value):
        """Pad the interferogram with N samples of NaN or zeros.

        NaNs are used if NaNs fill the periphery of the data.  If zeros fill
        the periphery, zero is used.

        Parameters
        ----------
        value : int
            how many samples to pad the data with

        Returns
        -------
        Interferogram
            self

        """
        npx = value

        if np.isnan(self.data[0, 0]):
            fill_val = np.nan
        else:
            fill_val = 0

        s = self.shape
        out = np.empty((s[0] + 2 * npx, s[1] + 2 * npx), dtype=self.data.dtype)
        out[:, :] = fill_val
        out[npx:-npx, npx:-npx] = self.data
        self.data = out

        return self.latcal(self.dx)

    def spike_clip(self, nsigma=3):
        """Clip points in the data that exceed a certain multiple of the standard deviation.

        Parameters
        ----------
        nsigma : float
            number of standard deviations to keep

        Returns
        -------
        self
            this Interferogram instancnp.

        """
        pts_over_nsigma = abs(self.data) > nsigma * self.std
        self.data[pts_over_nsigma] = np.nan
        return self

    def psd(self):
        """Power spectral density of the data., units ~nm^2/mm^2, assuming z axis has units of nm and x/y mm.

        Returns
        -------
        RichData
            RichData class instance with x, y, data attributes

        """
        ux, uy, psd_ = psd(self.data, self.dx)

        p = RichData(psd_, 0, self.wavelength)
        p.x = ux
        p.y = uy
        p.dx = ux[1] - ux[0]
        p._default_twosided = False
        return p

    def filter(self, fc, typ='lowpass'):
        """Apply a frequency domain filter to the data.

        Parameters
        ----------
        fc : float or length 2 tuple
            scalar critical frequency for the filter for either low or highpass
            (lower, upper) critical frequencies for bandpass and bandreject filters
        typ : str, {'lp', 'hp', 'bp', 'br', 'lowpass', 'highpass', 'bandpass', 'bandreject'}
            what type of filter to apply

        """
        H = designfilt2d(self.r, self.dx, fc, typ)
        D = fft.fft2(self.data)
        Dprime = D * H
        dprime = fft.ifft2(Dprime)
        self.data = dprime.real

    def bandlimited_rms(self, wllow=None, wlhigh=None, flow=None, fhigh=None):
        """Calculate the bandlimited RMS of a signal from its PSD.

        Parameters
        ----------
        wllow : float
            short spatial scale
        wlhigh : float
            long spatial scale
        flow : float
            low frequency
        fhigh : float
            high frequency

        Returns
        -------
        float
            band-limited RMS valunp.

        """
        psd = self.psd()
        return bandlimited_rms(r=psd.r, psd=psd.data, wllow=wllow, wlhigh=wlhigh, flow=flow, fhigh=fhigh)

    def total_integrated_scatter(self, wavelength, incident_angle=0):
        """Calculate the total integrated scatter (TIS) for an angle or angles.

        Assumes the spatial units of self are mm.

        Parameters
        ----------
        wavelength : float
            wavelength of light in microns
        incident_angle : float or numpy.ndarray
            incident angle(s) of light

        Returns
        -------
        float or numpy.ndarray
            TIS

        """
        # 1000/L vs 1/L, um to mm
        upper_limit = 1000 / wavelength
        kernel = 4 * np.pi * np.cos(np.radians(incident_angle))
        kernel *= self.bandlimited_rms(upper_limit, None) / wavelength
        return 1 - np.exp(-kernel**2)

    def interferogram(self, visibility=1, passes=2, tilt_waves=(0, 0), interpolation=None, fig=None, ax=None):
        """Create a picture of fringes.

        Parameters
        ----------
        visibility : float
            Visibility of the interferogram
        passes : float
            Number of passes (double-pass, quadra-pass, etc.)
        tilt_waves : tuple
            (x,y) waves of tilt to use for the interferogram
        interpolation : str, optional
            interpolation method, passed directly to matplotlib
        fig : matplotlib.figure.Figure, optional
            Figure to draw plot in
        ax : matplotlib.axes.Axis
            Axis to draw plot in

        Returns
        -------
        fig : matplotlib.figure.Figure, optional
            Figure containing the plot
        ax : matplotlib.axes.Axis, optional:
            Axis containing the plot

        """
        data = self.data
        # divide by two because -1 to 1 is 2 units PV, waves are "1" PV
        yramp = np.linspace(-1, 1, data.shape[0]) * (tilt_waves[1] / 2)
        xramp = np.linspace(-1, 1, data.shape[1]) * (tilt_waves[0] / 2)
        yramp = np.broadcast_to(yramp, reversed(data.shape)).T
        xramp = np.broadcast_to(xramp, data.shape)
        phase = self.data / 1e3 * self.wavelength  # 1e3 = nm to um
        phase = phase + (xramp + yramp)
        fig, ax = share_fig_ax(fig, ax)
        plotdata = visibility * np.cos(2 * np.pi * passes * phase)
        x, y = self.x, self.y
        im = ax.imshow(plotdata,
                       extent=[x.min(), x.max(), y.min(), y.max()],
                       cmap='gray',
                       interpolation=interpolation,
                       clim=(-1, 1),
                       origin='lower')
        fig.colorbar(im, label=r'Wrapped Phase [$\lambda$]', ax=ax, fraction=0.046)
        return fig, ax

    def save_zygo_ascii(self, file):
        """Save the interferogram to a Zygo ASCII filnp.

        Parameters
        ----------
        file : Path_like, str, or File_like
            where to save to

        """
        sf = 1 / (self.wavelength * 1e3)
        phase = self.data * sf
        write_zygo_ascii(file, phase=phase, dx=self.dx, intensity=None, wavelength=self.wavelength)

    def __str__(self):
        """Pretty-print string representation."""
        if self._latcaled:
            z_unit = 'mm'
        else:
            z_unit = 'px'
        diameter_y, diameter_x = self.support_y, self.support_x
        return inspect.cleandoc(f"""Interferogram with:
                Size: ({diameter_x:.3f}x{diameter_y:.3f}){z_unit}
                {self.pv:.3f} PV, {self.rms:.3f} RMS nm""")

    @staticmethod
    def from_zygo_dat(path, multi_intensity_action='first'):
        """Create a new interferogram from a zygo dat filnp.

        Parameters
        ----------
        path : path_like
            path to a zygo dat file
        multi_intensity_action : str, optional
            see io.read_zygo_dat
        scale : str, optional, {'um', 'mm'}
            what xy scale to label the data with, microns or mm

        Returns
        -------
        Interferogram
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

        i = Interferogram(phase=phase, dx=res*1e3, intensity=zydat['intensity'], meta=zydat['meta'], wavelength=None)
        return i

    @staticmethod  # NOQA
    def render_from_psd(size, samples, rms=None,  # NOQA
                        mask='circle', psd_fcn=abc_psd, **psd_fcn_kwargs):
        """Render a synthetic surface with a given RMS value given a PSD function.

        Parameters
        ----------
        size : float
            diameter of the output surface, mm
        samples : int
            number of samples across the output surface
        rms : float
            desired RMS value of the output, if rms=None, no normalization is done
        mask : str, optional
            mask defining the clear aperture
        psd_fcn : callable
            function used to generate the PSD
        **psd_fcn_kwargs:
            keyword arguments passed to psd_fcn in addition to nu
            if psd_fcn == abc_psd, kwargs are a, b, c
            elif psd_Fcn == ab_psd kwargs are a, b

            kwargs will be user-defined for user PSD functions

        Returns
        -------
        Interferogram
            new interferogram instance

        """
        x, y, z = render_synthetic_surface(size=size, samples=samples, rms=rms,
                                           mask=mask, psd_fcn=psd_fcn, **psd_fcn_kwargs)

        dx = x[1] - x[0]
        return Interferogram(phase=z, dx=dx, wavelength=HeNe)


# below this line is commented out, but working code that was written to design the 2D filtering code.
# It is equivalent, but 1D.

# def hann(N):
#     n = np.arange(N)
#     return np.sin(np.pi/N * n)**2

# def ideal_lpf_iir(x, fc_over_nyq):
#     """Ideal PSF of a low-pass filter."""
#     dx = x[1] - x[0]
#     return np.sinc(x*(fc_over_nyq/dx)) * fc_over_nyq


# def ideal_hpf_iir(x, fc_over_nyq):
#     """Ideal PSF of a high-pass filter."""
#     dx = x[1]-x[0]
#     c = 1/dx
#     term1 = np.sinc(x*c)
#     term2 = ideal_lpf_iir(x, fc_over_nyq)
#     return term1 - term2

# def ideal_bpf_iir(x, fl_over_nyq, fh_over_nyq):
#     """Ideal PSF of a band-pass filter."""
#     term1 = ideal_lpf_iir(x, fh_over_nyq)
#     term2 = ideal_lpf_iir(x, fl_over_nyq)
#     return term1 - term2


# def gaussfilt1d(x, fl=None, fh=None, typ='lowpass'):
#     fft = fft
#     dx = x[1] - x[0]
#     nu = fft.fftfreq(len(x), dx)
#     H = abs(nu) <= fh
#     softener = gauss(nu, 0, 0.25*fh)
#     H = conv(H, softener)
#     return H

# def designfilt1d(x, fc, typ='lowpass', N=None):
#     lx = len(x)
#     if N is None:
#         N = lx
#         xprime = x
#     else:
#         offset = (lx - N) // 2
#         xprime = x[offset:offset+N]

#     w = hann(N)
#     dx = x[1] - x[0]
#     nyq = 1 / (2*dx)
#     tl = typ.lower()
#     if tl in ('lp', 'lowpass'):
#         fc_over_nyq = fc/nyq
#         h = ideal_lpf_iir(xprime, fc_over_nyq)
#     elif tl in ('hp', 'highpass'):
#         fc_over_nyq = fc/nyq
#         h = ideal_hpf_iir(xprime, fc_over_nyq)
#     elif tl in ('bp', 'bandpass'):
#         fl_over_nyq = fc[0]/nyq
#         fh_over_nyq = fc[1]/nyq
#         h = ideal_bpf_iir(xprime, fl_over_nyq, fh_over_nyq)
#     elif tl in ('br', 'bandreject'):
#         # NOTE: band-reject we do by making a bandpass
#         # and taking 1 minus the transfer function
#         fl_over_nyq = fc[0]/nyq
#         fh_over_nyq = fc[1]/nyq
#         h = ideal_bpf_iir(xprime, fl_over_nyq, fh_over_nyq)

#     hprime = w * h
#     if N != lx:
#         tmp = np.zeros(x.shape, dtype=hprime.dtype)
#         tmp[:N] = hprime
#         hprime = tmp

#     H = fft.fft(hprime)
#     H = abs(H) # zero phase filter
#     if tl in ('br', 'bandreject'):
#         H = 1 - H
#     return H

# def apply_tf(f, H):
#     F = fft.fft(f)
#     Fprime = F * H
#     fprime = fft.ifft(Fprime)
#     return fprime.real

# def to_db(H):
#     return 10 * np.log10(H)
