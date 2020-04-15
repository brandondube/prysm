"""Numerical optical propagation."""
from .conf import config
from .mathops import engine as e
from ._richdata import RichData
from .fttools import pad2d, mdft

from astropy import units as u


def prop_pupil_plane_to_psf_plane(wavefunction, Q, incoherent=True, norm=None):
    """Propagate a pupil plane to a PSF plane and compute the grid along which the PSF exists.

    Parameters
    ----------
    wavefunction : `numpy.ndarray`
        the pupil wavefunction
    Q : `float`
        oversampling / padding factor
    incoherent : `bool`, optional
        whether to return the incoherent (real valued) PSF, or the
        coherent (complex-valued) PSF.  Incoherent = |coherent|^2
    norm : `str`, {None, 'ortho'}
        normalization parameter passed directly to numpy/cupy fft

    Returns
    -------
    psf : `numpy.ndarray`
        incoherent point spread function

    """
    if Q != 1:
        padded_wavefront = pad2d(wavefunction, Q)
    else:
        padded_wavefront = wavefunction

    impulse_response = e.fft.ifftshift(e.fft.fft2(e.fft.fftshift(padded_wavefront), norm=norm))
    if incoherent:
        return abs(impulse_response) ** 2
    else:
        return impulse_response


def prop_pupil_plane_to_psf_plane_fixed_sampling(wavefunction, input_sample_spacing, prop_dist, wavelength, output_sample_spacing, output_samples, coherent=False, norm=False):
    """Propagate a pupil function to the PSF plane with fixed sampling.

    Parameters
    ----------
    wavefunction : `numpy.ndarray`
        the pupil wavefunction
    input_sample_spacing : `float`
        spacing between samples in the pupil plane, millimeters
    prop_dist : `float`
        propagation distance along the z distance
    wavelength : `float`
        wavelength of light
    output_sample_spacing : `float`
        sample spacing in the output plane, microns
    output_samples : `int`
        number of samples in the square output array

    Returns
    -------
    x : `numpy.ndarray`
        x axis unit, 1D ndarray
    y : `numpy.ndarray`
        y axis unit, 1D ndarray
    data : `numpy.ndarray`
        2D array of data

    """
    dia = wavefunction.shape[0] * input_sample_spacing
    Q = Q_for_sampling(input_diameter=dia,
                       prop_dist=prop_dist,
                       wavelength=wavelength,
                       output_sample_spacing=output_sample_spacing)
    field = mdft.dft2(ary=wavefunction, Q=Q, samples=output_samples)
    samples_x, samples_y = output_samples, output_samples
    x = e.arange(-1 * int(e.ceil(samples_x / 2)), int(e.floor(samples_x / 2))) * output_sample_spacing
    y = e.arange(-1 * int(e.ceil(samples_y / 2)), int(e.floor(samples_y / 2))) * output_sample_spacing
    if coherent:
        return x, y, field
    else:
        return x, y, abs(field)**2


def prop_psf_plane_to_pupil_plane_fixed_sampling(wavefunction, input_sample_spacing, prop_dist, wavelength, output_sample_spacing, output_samples, norm=False):
    """Propagate an image plane field to the pupil plane with fixed sampling.

    Parameters
    ----------
    wavefunction : `numpy.ndarray`
        the image plane wavefunction
    input_sample_spacing : `float`
        spacing between samples in the pupil plane, millimeters
    prop_dist : `float`
        propagation distance along the z distance
    wavelength : `float`
        wavelength of light
    output_sample_spacing : `float`
        sample spacing in the output plane, microns
    output_samples : `int`
        number of samples in the square output array

    Returns
    -------
    x : `numpy.ndarray`
        x axis unit, 1D ndarray
    y : `numpy.ndarray`
        y axis unit, 1D ndarray
    data : `numpy.ndarray`
        2D array of data

    """
    # we calculate sampling parameters
    # backwards so we can reuse as much code as possible
    dia = output_sample_spacing * output_samples
    Q = Q_for_sampling(input_diameter=dia,
                       prop_dist=prop_dist,
                       wavelength=wavelength,
                       output_sample_spacing=input_sample_spacing)  # not a typo
    Q /= wavefunction.shape[0] / output_samples
    field = mdft.idft2(ary=wavefunction, Q=Q, samples=output_samples)
    samples_x, samples_y = output_samples, output_samples
    x = e.arange(-1 * int(e.ceil(samples_x / 2)), int(e.floor(samples_x / 2))) * output_sample_spacing
    y = e.arange(-1 * int(e.ceil(samples_y / 2)), int(e.floor(samples_y / 2))) * output_sample_spacing
    return x, y, field


def Q_for_sampling(input_diameter, prop_dist, wavelength, output_sample_spacing):
    """Value of Q for a given output sampling, given input sampling.

    Parameters
    ----------
    input_diameter : `float`
        diameter of the input array in millimeters
    prop_dist : `float`
        propagation distance along the z distance
    wavelength : `float`
        wavelength of light
    output_sample_spacing : `float`
        sampling in the output plane, microns

    Returns
    -------
    `float`
        requesite Q

    """
    resolution_element = (wavelength * prop_dist) / (input_diameter)
    return resolution_element / output_sample_spacing


def prop_pupil_plane_to_psf_plane_units(wavefunction, input_sample_spacing, prop_dist, wavelength, Q):
    """Compute the ordinate axes for a pupil plane to PSF plane propagation.

    Parameters
    ----------
    wavefunction : `numpy.ndarray`
        the pupil wavefunction
    input_sample_spacing : `float`
        spacing between samples in the pupil plane
    prop_dist : `float`
        propagation distance along the z distance
    wavelength : `float`
        wavelength of light
    Q : `float`
        oversampling / padding factor

    Returns
    -------
    x : `numpy.ndarray`
        x axis unit, 1D ndarray
    y : `numpy.ndarray`
        y axis unit, 1D ndarray

    """
    s = wavefunction.shape
    samples_x, samples_y = s[1] * Q, s[0] * Q
    sample_spacing_x = pupil_sample_to_psf_sample(pupil_sample=input_sample_spacing,  # factor of
                                                  samples=samples_x,                  # 1e3 corrects
                                                  wavelength=wavelength,              # for unit
                                                  efl=prop_dist) / 1e3                # translation
    sample_spacing_y = pupil_sample_to_psf_sample(pupil_sample=input_sample_spacing,  # factor of
                                                  samples=samples_y,                  # 1e3 corrects
                                                  wavelength=wavelength,              # for unit
                                                  efl=prop_dist) / 1e3                # translation
    x = e.arange(-1 * int(e.ceil(samples_x / 2)), int(e.floor(samples_x / 2))) * sample_spacing_x
    y = e.arange(-1 * int(e.ceil(samples_y / 2)), int(e.floor(samples_y / 2))) * sample_spacing_y
    return x, y


def pupil_sample_to_psf_sample(pupil_sample, samples, wavelength, efl):
    """Convert pupil sample spacing to PSF sample spacing.

    Parameters
    ----------
    pupil_sample : `float`
        sample spacing in the pupil plane
    samples : `int`
        number of samples present in both planes (must be equal)
    wavelength : `float`
        wavelength of light, in microns
    efl : `float`
        effective focal length of the optical system in mm

    Returns
    -------
    `float`
        the sample spacing in the PSF plane

    """
    return (wavelength * efl * 1e3) / (pupil_sample * samples)


def psf_sample_to_pupil_sample(psf_sample, samples, wavelength, efl):
    """Convert PSF sample spacing to pupil sample spacing.

    Parameters
    ----------
    psf_sample : `float`
        sample spacing in the PSF plane
    samples : `int`
        number of samples present in both planes (must be equal)
    wavelength : `float`
        wavelength of light, in microns
    efl : `float`
        effective focal length of the optical system in mm

    Returns
    -------
    `float`
        the sample spacing in the pupil plane

    """
    return (wavelength * efl * 1e3) / (psf_sample * samples)


def fresnel_number(a, L, lambda_):
    """Compute the Fresnel number.

    Notes
    -----
    if the fresnel number is << 1, paraxial assumptions hold for propagation

    Parameters
    ----------
    a : `float`
        characteristic size ("radius") of an aperture
    L : `float`
        distance of observation
    lambda_ : `float`
        wavelength of light, same units as a

    Returns
    -------
    `float`
        the fresnel number for these parameters

    """
    return a**2 / (L * lambda_)


def talbot_distance(a, lambda_):
    """Compute the talbot distance.

    Parameters
    ----------
    a : `float`
        period of the grating, units of microns
    lambda_ : `float`
        wavleength of light, units of microns

    Returns
    -------
    `float`
        talbot distance, units of microns

    """
    num = lambda_
    den = 1 - e.sqrt(1 - lambda_**2/a**2)
    return num / den


def angular_spectrum(field, wvl, sample_spacing, z, Q=2):
    """Propagate a field via the angular spectrum method.

    Parameters
    ----------
    field : `numpy.ndarray`
        2D array of complex electric field values
    wvl : `float`
        wavelength of light, microns
    z : `float`
        propagation distance, units of millimeters
    sample_spacing : `float`
        cartesian sample spacing, units of millimeters
    Q : `float`
        sampling factor used.  Q>=2 for Nyquist sampling of incoherent fields

    Returns
    -------
    `numpy.ndarray`
        2D ndarray of the output field, complex

    """
    # match all the units
    wvl = wvl / 1e3  # um -> mm
    if Q != 1:
        field = pad2d(field, Q=Q)

    ky, kx = (e.fft.fftfreq(s, sample_spacing) for s in field.shape)
    kyy, kxx = e.meshgrid(ky, kx)
    # don't ifftshift, ky, kx computed in shifted space, going to ifft anyway
    forward = e.fft.fft2(e.fft.fftshift(field))
    # kernel = e.zeros_like(forward)
    # wavenumber = 2 * e.pi / wvl
    # transfer_function = e.exp(1j * e.sqrt(wavenumber**2 - kxx**2 - kyy**2) * z)
    transfer_function = e.exp(-1j * e.pi * wvl * z * (kxx**2 + kyy**2))
    res = e.fft.ifftshift(e.fft.ifft2(forward * transfer_function))
    return res


def angular_spectrum_transfer_function(z, kx, ky, k, x0=0, y0=0):
    """Calculate the angular spectrum transfer function.

    Notes
    -----
    the transfer function is given in Eq. (2) of oe-22-21-26256,
    "Modified shifted angular spectrum method for numerical propagation at reduced spatial sampling rates"
    A. Ritter, Opt. Expr. 2014

    Parameters
    ----------
    z : `float`
        propagation distance
    kx : `numpy.ndarray`
        2D array of X spatial frequencies, meshgrid of an output from fftfreq
    ky : `numpy.ndarray`
        2D array of Y spatial ferquencies, meshgrid of an output from fftfreq
    k : `float`
        wavenumber, 2pi/lambda
    x0 : `float`
        x center
    y0 : `float`
        y center

    Returns
    -------
    `numpy.ndarray`
        2D array containing the (complex) transfer function

    """
    term1 = 1j * e.sqrt(k**2 - kx**2 - ky**2)
    if x0 != 0:
        # assume x0, y0 given together
        term2 = 1j * kx * x0
        term3 = 1j * ky * y0
    else:
        term2 = 1j * kx
        term3 = 1j * ky
    return e.exp(term1 + term2 + term3)


def msas_transfer_function(z, kx, ky, k, x0, y0, qx, qy):
    """Calculate the modified shifted angular spectrum transfer function.

    Parameters
    ----------
    z : `float`
        propagation distance
    kx : `numpy.ndarray`
        2D array of X spatial frequencies, meshgrid of an output from fftfreq
    ky : `numpy.ndarray`
        2D array of Y spatial ferquencies, meshgrid of an output from fftfreq
    k : `float`
        wavenumber, 2pi/lambda
    x0 : `float`
        x center
    y0 : `float`
        y center
    qx : `float`
        x spatial frequency of the modifying plane wave
    qy : `float`
        y spatial frequency of the modifying plane wave

    Returns
    -------
    `numpy.ndarray`
        2D array containing the (complex) transfer function

    """
    return angular_spectrum_transfer_function(z=z, kx=kx+qx, ky=ky+qy, k=k, x0=x0, y0=y0)


def modified_shifted_angular_spectrum(field, sample_spacing, k, z, x0, y0, qx=0, qy=0, Q=2):
    """Compute the modified shifted angular spectrum of a field.

    Notes
    -----
    With default parameters of qx == qy == 0, this is simply the shifted
    angular spectrum method

    Parameters
    ----------
    field : `numpy.ndarray`
        2D array holding the (complex) field or wavefunction
    sample_spacing : `float`
        sample spacing of the field in millimeters
    k : `float`
        wavenumber, 2pi/lambda, with lambda in microns
    z : `float`
        propagation distance in millimeters
    x0 : `float`
        distance of the x shift from the origin, millimeters
    y0 : `float`
        distance of the y shift from the origin, millimeters
    qx : `float`
        x spatial frequency of the modifying plane wave
    qy : `float`
        y spatial frequency of the modifying plane wave
    Q : `float`
        sampling factor to use in the propagation, Q>=2 for Nyquist sampling of incoherent fields

    Returns
    -------
    `numpy.ndarray`
        ndarray holding the propagated field
    `float`
        output sample spacing, mm

    """
    y, x = (sample_spacing * e.arange(s, dtype=config.precision) for s in field.shape)
    forward_plane = e.exp(-1j * qx * x - 1j * qy * y)
    backward_plane = e.exp(1j * qx * x + 1j * qy * y)
    forward = prop_pupil_plane_to_psf_plane(field*forward_plane, Q, incoherent=False)
    ky, kx = (e.fft.fftshift(e.fft.fftfreq(s, d=sample_spacing)) for s in forward.shape)
    mod = forward * msas_transfer_function(z=z, kx=kx, ky=ky, k=k, x0=x0, y0=y0, qx=qx, qy=qy)
    backward = e.fft.ifftshift(e.fft.ifft2(e.fft.fftshift(mod))) * backward_plane

    return backward, sample_spacing


class Wavefront(RichData):
    """(Complex) representation of a wavefront."""

    def __init__(self, x, y, fcn, wavelength, space='pupil'):
        """Create a new Wavefront instance.

        Parameters
        ----------
        x : `numpy.ndarray`
            x coordinates
        y : `numpy.ndarray`
            y coordinates
        fcn : `numpy.ndarray`
            complex-valued wavefront array
        wavelength : `float`
            wavelength of light, microns

        """
        super().__init__(x=x, y=y, data=fcn,
                         wavelength=wavelength,
                         labels=config.pupil_labels,
                         xy_unit=config.phase_xy_unit,
                         z_unit=config.phase_z_unit)
        self.space = space

    @property
    def fcn(self):
        """Complex field / wavefunction."""
        return self.data

    @fcn.setter
    def fcn(self, ary):
        self.data = ary

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

    def plane_to_plane(self, dz, Q=2):
        """Perform a plane-to-plane propagation.

        Uses angular spectrum and the free space kernel.

        Parameters
        ----------
        dz : `float`
            inter-plane distance, millimeters
        Q : `float`
            padding factor.  Q=1 does no padding, Q=2 pads 1024 to 2048.

        Returns
        -------
        `Wavefront`
            the wavefront at the new plane

        """
        out = angular_spectrum(self.fcn, self.wavelength.to(u.um), self.sample_spacing, dz)
        return Wavefront(x=self.x, y=self.y, fcn=out, wavelength=self.wavelength, space=self.space)

    def to_focus(self, efl, Q=2):
        """Perform a "pupil" to "psf" plane propgation.

        Uses an FFT with no quadratic phase.

        Parameters
        ----------
        efl : `float`
            focusing distance, millimeters
        Q : `float`
            padding factor.  Q=1 does no padding, Q=2 pads 1024 to 2048.
            To avoid aliasng, the array must be padded such that Q is at least 2
            this may happen organically if your data does not span the array.

        Returns
        -------
        `Wavefront`
            the wavefront at the focal plane

        """
        if self.space != 'pupil':
            raise ValueError('can only propagate from a pupil to psf plane')

        data = prop_pupil_plane_to_psf_plane(self.fcn, Q=Q, incoherent=False)
        x, y = prop_pupil_plane_to_psf_plane_units(
            wavefunction=self.fcn,
            input_sample_spacing=self.sample_spacing,
            prop_dist=efl,
            wavelength=self.wavelength.to(u.um),
            Q=Q)

        return Wavefront(x=x, y=y, fcn=data, wavelength=self.wavelength, space='psf')

    def from_focus(self, efl, Q=2):
        """Perform a "psf" to "pupil" plane propagation.

        uses an FFT with no quadratic phase.

        Parameters
        ----------
        efl : `float`
            un-focusing distance, millimeters
        Q : `float`
            padding factor.  Q=1 does no padding, Q=2 pads 1024 to 2048.
            To avoid aliasng, the array must be padded such that Q is at least 2
            this may happen organically if your data does not span the array.

        Returns
        -------
        `Wavefront`
            the wavefront at the pupil plane

        """
        pass

    def to_focus_fixed_sampling(self, efl, sample_spacing, samples):
        """Perform a "pupil" to "psf" propagation with fixed output sampling.

        Uses matrix triple product DFTs to specify the grid directly.

        Parameters
        ----------
        efl : `float`
            focusing distance, millimeters
        sample_spacing : `float`
            output sample spacing, microns
        samples : `int`
            number of samples in the output plane

        Returns
        -------
        `Wavefront`
            the wavefront at the psf plane

        """
        if self.space != 'pupil':
            raise ValueError('can only propagate from a pupil to psf plane')

        x, y, data = prop_pupil_plane_to_psf_plane_fixed_sampling(
            wavefunction=self.fcn,
            input_sample_spacing=self.sample_spacing,
            prop_dist=efl,
            wavelength=self.wavelength.to(u.um),
            output_sample_spacing=sample_spacing,
            output_samples=samples,
            coherent=True, norm=False)

        return Wavefront(x=x, y=y, fcn=data, wavelength=self.wavelength, space='psf')

    def from_focus_fixed_sampling(self, efl, sample_spacing, samples):
        """Perform a "psf" to "pupil" propagation with fixed output sampling.

        Uses matrix triple product DFTs to specify the grid directly.

        Parameters
        ----------
        efl : `float`
            un-focusing distance, millimeters
        sample_spacing : `float`
            output sample spacing, millimeters
        samples : `int`
            number of samples in the output plane

        Returns
        -------
        `Wavefront`
            wavefront at the pupil plane

        """
        if self.space != 'psf':
            raise ValueError('can only propagate from a psf to pupil plane')

        x, y, data = prop_psf_plane_to_pupil_plane_fixed_sampling(
            wavefunction=self.fcn,
            input_sample_spacing=self.sample_spacing,
            prop_dist=efl,
            wavelength=self.wavelength.to(u.um),
            output_sample_spacing=sample_spacing,
            output_samples=samples,
            norm=False)

        return Wavefront(x=x, y=y, fcn=data, wavelength=self.wavelength, space='pupil')
