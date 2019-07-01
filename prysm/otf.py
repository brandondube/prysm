"""A base optical transfer function interface."""

from .conf import config
from .mathops import engine as e
from ._basicdata import BasicData
from .psf import PSF
from .fttools import forward_ft_unit


def transform_psf(psf, sample_spacing):
    data = e.fft.fftshift(e.fft.fft2(e.fft.ifftshift(psf.data)))  # no need to ifftshift first - phase is unimportant
    y, x = [forward_ft_unit(sample_spacing / 1e3, s) for s in psf.shape]  # 1e3 for microns => mm
    return x, y, data


class OTF:
    """Optical Transfer Function."""

    def __init__(self, mtf, ptf):
        """Create a new OTF Instance.

        Will have .mtf and .ptf attributes holding the MTF and PTF.

        Parameters
        ----------
        data : `numpy.ndarray`
            complex ndarray, 2D
        x : `numpy.ndarray`
            x Cartesian spatial frequencies
        y : `numpy.ndarray`
            y Cartesian spatial frequencies
        """
        self.mtf = mtf
        self.ptf = ptf

    @staticmethod
    def from_psf(psf, unwrap=True):
        """Create an OTF instance from a PSF.

        Parameters
        ----------
        psf : `PSF`
            Point Spread Function
        unwrap : `bool`, optional
            if True, unwrap phase

        Returns
        -------
        `OTF`
            new OTF instance with mtf and PSF attributes holding MTF and PSF instances

        """
        x, y, ft = transform_psf(psf, psf.sample_spacing)
        mtf = MTF.from_ftdata(ft=ft, x=x, y=y)
        ptf = PTF.from_ftdata(ft=ft, x=x, y=y, unwrap=unwrap)
        return OTF(mtf=mtf, ptf=ptf)

    @staticmethod
    def from_pupil(pupil, efl, Q=config.Q, unwrap=True):
        psf = PSF.from_pupil(pupil, efl=efl, Q=Q)
        return OTF.from_psf(psf, unwrap=unwrap)


class MTF(BasicData):
    """Modulation Transfer Function."""
    _data_attr = 'data'
    _data_type = 'image'
    _default_twosided = False

    def __init__(self, data, x, y):
        """Create a new `MTF` instance.

        Parameters
        ----------
        data : `numpy.ndarray`
            2D array of MTF data
        x : `numpy.ndarray`
            1D array of x spatial frequencies
        y : `numpy.ndarray`
            1D array of y spatial frequencies
        """
        super().__init__(x=x, y=y, data=data, xyunit='mm', zunit='au',
                         xlabel='X Spatial Frequency', ylabel='Y Spatial Frequency',
                         zlabel='MTF')

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
        # some code duplication here:
        # MTF is a hot code path, and the drop of a shift operation
        # improves performance in exchange for sharing some code with
        # the OTF class definition
        dat = e.fft.fftshift(e.fft.fft2(psf.data))  # no need to ifftshift first - phase is unimportant
        x = forward_ft_unit(psf.sample_spacing / 1e3, psf.samples_x)  # 1e3 for microns => mm
        y = forward_ft_unit(psf.sample_spacing / 1e3, psf.samples_y)
        return MTF.from_ftdata(ft=dat, x=x, y=y)

    @staticmethod
    def from_pupil(pupil, efl, Q=2):
        """Generate an MTF from a pupil, given a focal length (propagation distance).

        Parameters
        ----------
        pupil : `Pupil`
            A pupil to propagate to a PSF, and convert to an MTF
        efl : `float`
            Effective focal length or propagation distance of the wavefunction
        Q : `float`
            ratio of pupil sample count to PSF sample count.  Q > 2 satisfies nyquist

        Returns
        -------
        `MTF`
            A new MTF instance

        """
        psf = PSF.from_pupil(pupil, efl=efl, Q=Q)
        return MTF.from_psf(psf)

    @staticmethod
    def from_ftdata(ft, x, y):
        """Generate an MTF from the Fourier transform of a PSF.

        Parameters
        ----------
        ft : `numpy.ndarray`
            2D ndarray of Fourier transform data
        x : `numpy.ndarray`
            1D ndarray of x (axis 1) coordinates
        y : `numpy.ndarray`
            1D ndarray of y (axis 0) coordinates

        Returns
        -------
        `MTF`
            a new MTF instance

        """
        cy, cx = (int(e.ceil(s / 2)) for s in ft.shape)
        dat = abs(ft)
        dat /= dat[cy, cx]
        return MTF(data=dat, x=x, y=y)


class PTF(BasicData):
    """Phase Transfer Function"""

    def __init__(self, data, x, y):
        """Create a new `PTF` instance.

        Parameters
        ----------
        data : `numpy.ndarray`
            2D array of MTF data
        x : `numpy.ndarray`
            1D array of x spatial frequencies
        y : `numpy.ndarray`
            1D array of y spatial frequencies
        """
        super().__init__(x=x, y=y, data=data, xyunit='mm', zunit='au',
                         xlabel='X Spatial Frequency', ylabel='Y Spatial Frequency',
                         zlabel='PTF')

    @staticmethod
    def from_psf(psf, unwrap=True):
        """Generate a PTF from a PSF.

        Parameters
        ----------
        psf : `PSF`
            PSF to compute an MTF from
        unwrap : `bool,` optional
            whether to unwrap the phase

        Returns
        -------
        `PTF`
            A new PTF instance

        """
        # some code duplication here:
        # MTF is a hot code path, and the drop of a shift operation
        # improves performance in exchange for sharing some code with
        # the OTF class definition

        # repeat this duplication in PTF for symmetry more than performance
        dat = e.fft.fftshift(e.fft.fft2(e.fft.ifftshift(psf.data)))
        x = forward_ft_unit(psf.sample_spacing / 1e3, psf.samples_x)  # 1e3 for microns => mm
        y = forward_ft_unit(psf.sample_spacing / 1e3, psf.samples_y)
        return PTF.from_ftdata(ft=dat, x=x, y=y)

    @staticmethod
    def from_pupil(pupil, efl, Q=2, unwrap=True):
        """Generate a PTF from a pupil, given a focal length (propagation distance).

        Parameters
        ----------
        pupil : `Pupil`
            A pupil to propagate to a PSF, and convert to an MTF
        efl : `float`
            Effective focal length or propagation distance of the wavefunction
        Q : `float`, optional
            ratio of pupil sample count to PSF sample count.  Q > 2 satisfies nyquist
        unwrap : `bool,` optional
            whether to unwrap the phase

        Returns
        -------
        `PTF`
            A new PTF instance

        """
        psf = PSF.from_pupil(pupil, efl=efl, Q=Q)
        return PTF.from_psf(psf, unwrap=unwrap)

    @staticmethod
    def from_ftdata(ft, x, y, unwrap=True):
        """Generate a PTF from the Fourier transform of a PSF.

        Parameters
        ----------
        ft : `numpy.ndarray`
            2D ndarray of Fourier transform data
        x : `numpy.ndarray`
            1D ndarray of x (axis 1) coordinates
        y : `numpy.ndarray`
            1D ndarray of y (axis 0) coordinates
        unwrap : `bool`, optional
            if True, unwrap phase

        Returns
        -------
        `PTF`
            a new PTF instance

        """
        ft = e.angle(ft)
        cy, cx = (int(e.ceil(s / 2)) for s in ft.shape)
        offset = ft[cy, cx]
        if offset != 0:
            ft /= offset

        if unwrap:
            from skimage import restoration
            ft = restoration.unwrap_phase(ft)
        return PTF(ft, x, y)


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
