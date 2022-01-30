"""MTF/PTF/OTF calculations."""
from .mathops import np, fft
from ._richdata import RichData


def transform_psf(psf, dx=None):
    """Transform a PSF to k-space without further modification."""
    if not hasattr(psf, 'ndim'):  # container object, not array
        dx = psf.dx
        psf = psf.data

    if dx is None:
        raise ValueError('dx is None: dx must be provided if psf is an array')

    data = fft.fftshift(fft.fft2(fft.ifftshift(psf)))
    df = 1000 / (data.shape[0] * dx)  # cy/um to cy/mm
    return data, df


def mtf_from_psf(psf, dx=None):
    """Compute the MTF from a given PSF.

    Parameters
    ----------
    psf : prysm.RichData or numpy.ndarray
        object with data property having 2D data containing the psf,
        or the array itself
    dx : float
        sample spacing of the data

    Returns
    -------
    RichData
        container holding the MTF, ready for plotting or slicing.

    """
    data, df = transform_psf(psf, dx)
    cy, cx = (int(np.ceil(s / 2)) for s in data.shape)
    dat = abs(data)
    dat /= dat[cy, cx]
    return RichData(data=dat, dx=df, wavelength=None)


def ptf_from_psf(psf, dx=None):
    """Compute the PTF from a given PSF.

    Parameters
    ----------
    psf : prysm.RichData or numpy.ndarray
        object with data property having 2D data containing the psf,
        or the array itself
    dx : float
        sample spacing of the data

    Returns
    -------
    RichData
        container holding the MTF, ready for plotting or slicing.

    """
    data, df = transform_psf(psf, dx)
    cy, cx = (int(np.ceil(s / 2)) for s in data.shape)
    # it might be slightly faster to do this after conversion to rad with a -=
    # op, but the phase wrapping there would be tricky.  Best to do this before
    # for robustness.
    data /= data[cy, cx]
    dat = np.angle(data)
    return RichData(data=dat, dx=df, wavelength=None)


def otf_from_psf(psf, dx=None):
    """Compute the OTF from a given PSF.

    Parameters
    ----------
    psf : numpy.ndarray
        2D data containing the psf
    dx : float
        sample spacing of the data

    Returns
    -------
    RichData
        container holding the OTF, complex.

    """
    data, df = transform_psf(psf, dx)
    cy, cx = (int(np.ceil(s / 2)) for s in data.shape)
    data /= data[cy, cx]
    return RichData(data=data, dx=df, wavelength=None)


# TODO: mtf_and_ptf_from_psf to only do the FT one time

def diffraction_limited_mtf(fno, wavelength, frequencies=None, samples=128):
    """Give the diffraction limited MTF for a circular pupil and the given parameters.

    Parameters
    ----------
    fno : float
        f/# of the lens.
    wavelength : float
        wavelength of light, in microns.
    frequencies : numpy.ndarray
        spatial frequencies of interest, in cy/mm if frequencies are given, samples is ignored.
    samples : int
        number of points in the output array, if frequencies not given.

    Returns
    -------
    if frequencies not given:
        frequencies : numpy.ndarray
            array of ordinate data
        mtf : numpy.ndarray
            array of coordinate data
    else:
        mtf : numpy.ndarray
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
    normalized_frequency : numpy.ndarray
        normalized frequency; function is defined over [0, and takes a value of 0 for [1,

    Returns
    -------
    numpy.ndarray
        The diffraction MTF function at a given normalized spatial frequency

    """
    return (2 / np.pi) * \
           (np.arccos(normalized_frequency) - normalized_frequency *
            np.sqrt(1 - normalized_frequency ** 2))


def longexposure_otf(nu, Cn, z, f, lambdabar, h_z_by_r=2.91):
    """Compute the long exposure OTF for given parameters.

    Parameters
    ----------
    nu : numpy.ndarray
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
    numpy.ndarray
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
    r : numpy.ndarray
        r, radial frequency parameter (object space)
    r0 : float
        Fried parameter

    Returns
    -------
    numpy.ndarray

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
