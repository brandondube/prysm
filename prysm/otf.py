"""MTF/PTF/OTF calculations."""
import numbers

from scipy.special import j1 as _besselj1

from .mathops import np, fft
from .coordinates import make_xy_grid
from ._richdata import RichData


def _center(shape):
    """Pixel index of the (floor) center of a 2D array of given shape."""
    return tuple(int(np.floor(s / 2)) for s in shape)


def _unwrap_psf(psf, dx):
    """Resolve a PSF container-or-array to a bare array and its sample spacing."""
    if not hasattr(psf, 'ndim'):  # container object, not array
        dx = psf.dx
        psf = psf.data

    if dx is None:
        raise ValueError('dx is None: dx must be provided if psf is an array')

    return psf, dx


def transform_psf(psf, dx=None):
    """Transform a PSF to k-space without further modification."""
    psf, dx = _unwrap_psf(psf, dx)
    data = fft.fftshift(fft.fft2(fft.ifftshift(psf)))
    df = 1000 / (data.shape[0] * dx)  # cy/um to cy/mm
    return data, df


def transform_psf_adjoint(data_bar):
    """Apply the adjoint of the linear FT performed by transform_psf.

    transform_psf maps a real PSF to the complex k-space field
    fftshift(fft2(ifftshift(psf))).  This is its conjugate-transpose: it maps a
    gradient defined on the k-space field back to the PSF plane.  The df scalar
    returned by transform_psf is sampling metadata and carries no gradient.

    Parameters
    ----------
    data_bar : ndarray
        gradient at the k-space (OTF) plane

    Returns
    -------
    ndarray
        gradient at the PSF plane

    """
    # forward uses fft2 with the default ('backward') normalization, whose
    # conjugate transpose is the unnormalized inverse transform, i.e. ifft2 with
    # norm='forward'.  fftshift/ifftshift are permutations; their adjoints are
    # one another.
    return fft.fftshift(fft.ifft2(fft.ifftshift(data_bar), norm='forward'))


def _normalized_transform(psf, dx):
    """Forward-transform a PSF and divide by its central value.

    Returns the center-normalized complex transform (the OTF), the raw complex
    transform (kept for return_more and the adjoints), and the spatial-frequency
    sample spacing.  The MTF and PTF are the modulus and argument of the
    normalized transform, so one call serves mtf_from_psf, ptf_from_psf,
    otf_from_psf, and mtf_ptf_otf_from_psf with a single forward FT.
    """
    data, df = transform_psf(psf, dx)
    cy, cx = _center(data.shape)
    normalized = data / data[cy, cx]
    return normalized, data, df


def mtf_from_psf(psf, dx=None, return_more=False):
    """Compute the MTF from a given PSF.

    Parameters
    ----------
    psf : prysm.RichData or ndarray
        object with data property having 2D data containing the psf,
        or the array itself
    dx : float
        sample spacing of the data
    return_more : bool
        if True, also return the complex k-space transform of the PSF (the same
        array transform_psf produces).  Hand it to mtf_from_psf_adjoint as data
        to skip recomputing the forward FT in the reverse pass.

    Returns
    -------
    RichData
        container holding the MTF, ready for plotting or slicing.
    ndarray
        the complex transform; only returned if return_more is True.

    """
    normalized, data, df = _normalized_transform(psf, dx)
    rd = RichData(data=abs(normalized), dx=df, wavelength=None)
    if return_more:
        return rd, data
    return rd


def ptf_from_psf(psf, dx=None, return_more=False):
    """Compute the PTF from a given PSF.

    Parameters
    ----------
    psf : prysm.RichData or ndarray
        object with data property having 2D data containing the psf,
        or the array itself
    dx : float
        sample spacing of the data
    return_more : bool
        if True, also return the complex k-space transform of the PSF (the same
        array transform_psf produces).  Hand it to ptf_from_psf_adjoint as data
        to skip recomputing the forward FT in the reverse pass.

    Returns
    -------
    RichData
        container holding the MTF, ready for plotting or slicing.
    ndarray
        the complex transform; only returned if return_more is True.

    """
    # normalizing the complex transform before taking the angle references the
    # phase to the central value; doing it after conversion to radians with a -=
    # could be slightly faster but the phase wrapping there would be tricky.
    normalized, data, df = _normalized_transform(psf, dx)
    rd = RichData(data=np.angle(normalized), dx=df, wavelength=None)
    if return_more:
        return rd, data
    return rd


def otf_from_psf(psf, dx=None, return_more=False):
    """Compute the OTF from a given PSF.

    Parameters
    ----------
    psf : ndarray
        2D data containing the psf
    dx : float
        sample spacing of the data
    return_more : bool
        if True, also return the complex k-space transform of the PSF (the same
        array transform_psf produces, prior to the center normalization).  Hand
        it to otf_from_psf_adjoint as data to skip recomputing the forward FT in
        the reverse pass.

    Returns
    -------
    RichData
        container holding the OTF, complex.
    ndarray
        the unnormalized complex transform; only returned if return_more is True.

    """
    normalized, data, df = _normalized_transform(psf, dx)
    rd = RichData(data=normalized, dx=df, wavelength=None)
    if return_more:
        return rd, data
    return rd


def mtf_ptf_otf_from_psf(psf, dx=None, return_more=False):
    """Compute the MTF, PTF, and OTF from a PSF with a single forward transform.

    The three transfer functions are the modulus, argument, and complex value of
    the same center-normalized transform, so computing them together does the
    forward FT once instead of once per quantity.

    Parameters
    ----------
    psf : prysm.RichData or ndarray
        object with data property having 2D data containing the psf,
        or the array itself
    dx : float
        sample spacing of the data
    return_more : bool
        if True, also return the raw complex transform of the PSF, prior to the
        center normalization, as transform_psf produces it.

    Returns
    -------
    mtf, ptf, otf : RichData
        the modulation, phase, and complex optical transfer functions
    ndarray
        the raw complex transform; only returned if return_more is True.

    """
    normalized, data, df = _normalized_transform(psf, dx)
    mtf = RichData(data=abs(normalized), dx=df, wavelength=None)
    ptf = RichData(data=np.angle(normalized), dx=df, wavelength=None)
    otf = RichData(data=normalized, dx=df, wavelength=None)
    if return_more:
        return mtf, ptf, otf, data
    return mtf, ptf, otf


def mtf_from_psf_adjoint(mtf_bar, psf=None, dx=None, data=None):
    """Apply the adjoint of mtf_from_psf.

    Maps a gradient defined on the (center-normalized) MTF back to the real PSF.
    The forward map is mtf = abs(F[psf]) / abs(F[psf])[center]; this differentiates
    through both the modulus and the normalization by the central value.

    Parameters
    ----------
    mtf_bar : ndarray
        gradient at the MTF plane
    psf : prysm.RichData or ndarray
        the PSF the MTF was computed from; used to recompute the forward FT when
        data is not supplied
    dx : float
        sample spacing of the PSF; required if psf is a bare array
    data : ndarray
        the complex transform from mtf_from_psf(..., return_more=True).  When
        given, the forward FT is reused instead of recomputed and psf/dx are
        ignored.

    Returns
    -------
    ndarray
        gradient at the PSF plane

    """
    if data is None:
        data, _ = transform_psf(psf, dx)
    cy, cx = _center(data.shape)
    mag = np.abs(data)
    a = mag[cy, cx]
    # d(|d|)/d(d) gives the unit-phasor d/|d|; divided by the center magnitude
    data_bar = mtf_bar * data / mag / a
    # the central value also enters through the 1/a normalization of every pixel
    S = np.sum(mtf_bar * mag)
    data_bar[cy, cx] -= S * data[cy, cx] / a ** 3
    return transform_psf_adjoint(data_bar).real


def ptf_from_psf_adjoint(ptf_bar, psf=None, dx=None, data=None):
    """Apply the adjoint of ptf_from_psf.

    Maps a gradient defined on the (center-referenced) PTF back to the real PSF.
    The forward map is ptf = angle(F[psf]) - angle(F[psf])[center].

    Parameters
    ----------
    ptf_bar : ndarray
        gradient at the PTF plane
    psf : prysm.RichData or ndarray
        the PSF the PTF was computed from; used to recompute the forward FT when
        data is not supplied
    dx : float
        sample spacing of the PSF; required if psf is a bare array
    data : ndarray
        the complex transform from ptf_from_psf(..., return_more=True).  When
        given, the forward FT is reused instead of recomputed and psf/dx are
        ignored.

    Returns
    -------
    ndarray
        gradient at the PSF plane

    """
    if data is None:
        data, _ = transform_psf(psf, dx)
    cy, cx = _center(data.shape)
    msq = data.real * data.real + data.imag * data.imag
    # d(angle(d))/d(d) gives 1j*d/|d|^2
    data_bar = ptf_bar * 1j * data / msq
    # every pixel is referenced to the central phase, so it gets the summed pull
    data_bar[cy, cx] -= np.sum(ptf_bar) * 1j * data[cy, cx] / msq[cy, cx]
    return transform_psf_adjoint(data_bar).real


def otf_from_psf_adjoint(otf_bar, psf=None, dx=None, data=None):
    """Apply the adjoint of otf_from_psf.

    Maps a gradient defined on the (center-normalized) complex OTF back to the
    real PSF.  The forward map is otf = F[psf] / F[psf][center].

    Parameters
    ----------
    otf_bar : ndarray
        gradient at the OTF plane; complex
    psf : prysm.RichData or ndarray
        the PSF the OTF was computed from; used to recompute the forward FT when
        data is not supplied
    dx : float
        sample spacing of the PSF; required if psf is a bare array
    data : ndarray
        the unnormalized complex transform from otf_from_psf(..., return_more=True).
        When given, the forward FT is reused instead of recomputed and psf/dx are
        ignored.

    Returns
    -------
    ndarray
        gradient at the PSF plane

    """
    if data is None:
        data, _ = transform_psf(psf, dx)
    cy, cx = _center(data.shape)
    c = data[cy, cx]
    cc = np.conj(c)
    data_bar = otf_bar / cc
    # the central value divides every pixel, so it accumulates the whole field
    data_bar[cy, cx] -= np.sum(np.conj(data) * otf_bar) / cc ** 2
    return transform_psf_adjoint(data_bar).real


def _encircled_energy_geometry(shape, df):
    """Radial spatial-frequency grid and frequency cell deltas for encircled energy.

    Parameters
    ----------
    shape : tuple of int
        shape of the MTF array
    df : float
        spatial-frequency sample spacing (the dx of the MTF RichData)

    Returns
    -------
    nu_p : ndarray
        radial spatial frequencies, with the zero-frequency bin nudged off zero
        to avoid division by zero
    dnx, dny : float
        frequency-axis sample spacings

    """
    nx, ny = make_xy_grid(shape, dx=df)
    nu_p = np.hypot(nx, ny)
    # this is meaninglessly small and will avoid division by 0
    nu_p[nu_p == 0] = 1e-16
    dnx, dny = ny[1, 0] - ny[0, 0], nx[0, 1] - nx[0, 0]
    return nu_p, dnx, dny


def encircled_energy(psf, dx, radius, return_more=False):
    """Compute the encircled energy of the PSF.

    Parameters
    ----------
    psf : ndarray
        2D array containing PSF data
    dx : float
        sample spacing of psf
    radius : float or iterable
        radius or radii to evaluate encircled energy at
    return_more : bool
        if True, also return the complex k-space transform of the PSF.  Hand it
        to encircled_energy_adjoint as data to skip recomputing the forward FT in
        the reverse pass.

    Returns
    -------
    encircled energy
        if radius is a float, returns a float, else returns an array.
    ndarray
        the complex transform; only returned if return_more is True.

    Notes
    -----
    implementation of "Simplified Method for Calculating Encircled Energy,"
    Baliga, J. V. and Cohn, B. D., doi: 10.1117/12.944334

    """
    # compute MTF from the PSF
    mtf, data = mtf_from_psf(psf, dx, return_more=True)
    nu_p, dnx, dny = _encircled_energy_geometry(mtf.shape, mtf.dx)

    if not isinstance(radius, numbers.Number):
        out = np.asarray([_encircled_energy_core(mtf.data, r / 1e3, nu_p, dnx, dny)
                          for r in radius])
    else:
        out = _encircled_energy_core(mtf.data, radius / 1e3, nu_p, dnx, dny)

    if return_more:
        return out, data
    return out


def _encircled_energy_core(mtf_data, radius, nu_p, dx, dy):
    """Core computation of encircled energy, based on Baliga 1988.

    Parameters
    ----------
    mtf_data : ndarray
        unaliased MTF data
    radius : float
        radius of "detector"
    nu_p : ndarray
        radial spatial frequencies
    dx : float
        x frequency delta
    dy : float
        y frequency delta

    Returns
    -------
    float
        encircled energy for given radius

    """
    integration_fourier = _besselj1(2 * np.pi * radius * nu_p) / nu_p
    dat = mtf_data * integration_fourier
    return radius * dat.sum() * dx * dy


def encircled_energy_adjoint(ee_bar, psf=None, dx=None, radius=None, data=None):
    """Apply the adjoint of encircled_energy.

    Encircled energy is a linear functional of the MTF (a radius-weighted sum of
    a Hankel kernel against the MTF), so its adjoint folds the per-radius
    gradients into a single MTF-plane gradient and routes that back through
    mtf_from_psf_adjoint to the PSF.

    Parameters
    ----------
    ee_bar : float or ndarray
        gradient of the loss with respect to the encircled energy; a scalar for a
        single radius, otherwise one value per radius matching radius
    psf : prysm.RichData or ndarray
        the PSF the encircled energy was computed from; used to recompute the
        forward FT when data is not supplied
    dx : float
        sample spacing of the PSF; always required, as it sets the frequency grid
    radius : float or iterable
        the radius/radii encircled_energy was evaluated at
    data : ndarray
        the complex transform from encircled_energy(..., return_more=True).  When
        given, the forward FT is reused instead of recomputed.

    Returns
    -------
    ndarray
        gradient at the PSF plane

    """
    if data is not None:
        shape = data.shape
        if dx is None:
            raise ValueError('dx is None: dx must be provided to set the frequency grid')
        dxv = dx
    else:
        arr, dxv = _unwrap_psf(psf, dx)
        shape = arr.shape

    df = 1000 / (shape[0] * dxv)  # cy/um to cy/mm; matches transform_psf
    nu_p, dnx, dny = _encircled_energy_geometry(shape, df)

    if isinstance(radius, numbers.Number):
        radii = (radius,)
        ee_bar = (ee_bar,)
    else:
        radii = radius

    # d(EE_r)/d(mtf) = r * J1(2*pi*r*nu_p)/nu_p * dnx * dny
    mtf_bar = 0.0
    for rb, r in zip(ee_bar, radii):
        ri = r / 1e3
        kernel = _besselj1(2 * np.pi * ri * nu_p) / nu_p
        mtf_bar = mtf_bar + rb * ri * kernel * dnx * dny

    return mtf_from_psf_adjoint(mtf_bar, psf=psf, dx=dx, data=data)


def diffraction_limited_mtf(fno, wavelength, frequencies=None, samples=128):
    """Give the diffraction limited MTF for a circular pupil and the given parameters.

    Parameters
    ----------
    fno : float
        f/# of the lens.
    wavelength : float
        wavelength of light, in microns.
    frequencies : ndarray
        spatial frequencies of interest, in cy/mm if frequencies are given, samples is ignored.
    samples : int
        number of points in the output array, if frequencies not given.

    Returns
    -------
    if frequencies not given:
        frequencies : ndarray
            array of ordinate data
        mtf : ndarray
            array of coordinate data
    else:
        mtf : ndarray
            array of MTF data

    Notes
    -----
    If frequencies are given, just returns the MTF.  If frequencies are not
    given, returns both the frequencies and the MTF.

    """
    extinction = 1 / (wavelength / 1000 * fno)
    if frequencies is None:
        normalized_frequency = np.linspace(0, 1, samples)
    else:
        normalized_frequency = abs(np.asarray(frequencies) / extinction)
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
    normalized_frequency : ndarray
        normalized frequency; function is defined over [0, and takes a value of 0 for [1,

    Returns
    -------
    ndarray
        The diffraction MTF function at a given normalized spatial frequency

    """
    return (2 / np.pi) * \
           (np.arccos(normalized_frequency) - normalized_frequency *
            np.sqrt(1 - normalized_frequency ** 2))


def longexposure_otf(nu, Cn, z, f, lambdabar, h_z_by_r=2.91):
    """Compute the long exposure OTF for given parameters.

    Parameters
    ----------
    nu : ndarray
        spatial frequencies, cy/mm
    Cn: float
        atmospheric structure constant of refractive index, ranges ~ 10^-13 - 10^-17
    z : float
        propagation distance through atmosphere, m
    f : float
        effective focal length of the optical system, mm
    lambdabar : float
        mean wavelength, microns
    h_z_by_r : float, optional
        constant for h[z/r] -- see Eq. 8.5-37 & 8.5-38 in Statistical Optics, J. Goodman, 2nd ed.

    Returns
    -------
    ndarray
        the OTF

    """
    # homogenize units
    nu = nu / 1e3
    f = f / 1e3
    lambdabar = lambdabar / 1e6

    power = 5/3
    const1 = - np.pi ** 2 * 2 * h_z_by_r * Cn ** 2
    const2 = z * f ** power / (lambdabar ** 3)
    nupow = nu ** power
    const = const1 * const2
    return np.exp(const * nupow)


def komogorov(r, r0):
    """Calculate the phase structure function D_phi in the komogorov approximation.

    Parameters
    ----------
    r : ndarray
        r, radial frequency parameter (object space)
    r0 : float
        Fried parameter

    Returns
    -------
    ndarray

    """
    return 6.88 * (r/r0) ** (5/3)


def estimate_Cn(P=1013, T=273.15, Ct=1e-4):
    """Use Weng et al to estimate Cn from meteorological data.

    Parameters
    ----------
    P : float
        atmospheric pressure in hPa
    T : float
        temperature in Kelvin
    Ct : float
        atmospheric struction constant of temperature, typically 10^-5 - 10^-2 near the surface

    Returns
    -------
    float
        Cn

    """
    return (79 * P / (T ** 2)) * Ct ** 2 * 1e-12
