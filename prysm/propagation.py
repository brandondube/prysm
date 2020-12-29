"""Numerical optical propagation."""
import numbers
import warnings
import operator
from collections.abc import Iterable


from .conf import config
from .mathops import engine as e
from ._richdata import RichData
from .fttools import pad2d, mdft

from astropy import units as u


def focus(wavefunction, Q, incoherent=True, norm=None):
    """Propagate a pupil plane to a PSF plane.

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
        point spread function

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


def unfocus(wavefunction, Q, norm=None):
    """Propagate a PSF plane to a pupil plane.

    Parameters
    ----------
    wavefunction : `numpy.ndarray`
        the pupil wavefunction
    Q : `float`
        oversampling / padding factor
    norm : `str`, {None, 'ortho'}
        normalization parameter passed directly to numpy/cupy fft

    Returns
    -------
    pupil : `numpy.ndarray`
        field in the pupil plane

    """
    if Q != 1:
        padded_wavefront = pad2d(wavefunction, Q)
    else:
        padded_wavefront = wavefunction

    return e.fft.ifftshift(e.fft.ifft2(e.fft.fftshift(padded_wavefront), norm=norm))


def focus_fixed_sampling(wavefunction, input_sample_spacing, prop_dist,
                         wavelength, output_sample_spacing, output_samples,
                         coherent=False, norm=True):
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
    coherent : `bool`
        if True, returns the complex array.  Else returns its magnitude squared.

    Returns
    -------
    data : `numpy.ndarray`
        2D array of data

    """
    dia = wavefunction.shape[0] * input_sample_spacing
    Q = Q_for_sampling(input_diameter=dia,
                       prop_dist=prop_dist,
                       wavelength=wavelength,
                       output_sample_spacing=output_sample_spacing)
    field = mdft.dft2(ary=wavefunction, Q=Q, samples=output_samples)
    if coherent:
        return field
    else:
        return abs(field)**2


def unfocus_fixed_sampling(wavefunction, input_sample_spacing, prop_dist,
                           wavelength, output_sample_spacing, output_samples,
                           norm=True):
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
    if not isinstance(output_samples, Iterable):
        output_samples = (output_samples, output_samples)

    dias = [output_sample_spacing * s for s in output_samples]
    dia = max(dias)
    Q = Q_for_sampling(input_diameter=dia,
                       prop_dist=prop_dist,
                       wavelength=wavelength,
                       output_sample_spacing=input_sample_spacing)  # not a typo
    Q /= wavefunction.shape[0] / output_samples[0]
    field = mdft.idft2(ary=wavefunction, Q=Q, samples=output_samples)
    return field


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


def focus_units(wavefunction, input_sample_spacing, efl, wavelength, Q):
    """Compute the ordinate axes for a pupil plane to PSF plane propagation.

    Parameters
    ----------
    wavefunction : `numpy.ndarray`
        the pupil wavefunction
    input_sample_spacing : `float`
        spacing between samples in the pupil plane
    efl : `float`
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
                                                  efl=efl) / 1e3                # translation
    sample_spacing_y = pupil_sample_to_psf_sample(pupil_sample=input_sample_spacing,  # factor of
                                                  samples=samples_y,                  # 1e3 corrects
                                                  wavelength=wavelength,              # for unit
                                                  efl=efl) / 1e3                # translation
    x = e.arange(-1 * int(e.ceil(samples_x / 2)), int(e.floor(samples_x / 2))) * sample_spacing_x
    y = e.arange(-1 * int(e.ceil(samples_y / 2)), int(e.floor(samples_y / 2))) * sample_spacing_y
    return x, y


def unfocus_units(wavefunction, input_sample_spacing, efl, wavelength, Q):
    """Compute the ordinate axes for a PSF plane to pupil plane propagation.

    Parameters
    ----------
    wavefunction : `numpy.ndarray`
        the pupil wavefunction
    input_sample_spacing : `float`
        spacing between samples in the PSF plane
    efl : `float`
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
    sample_spacing_x = psf_sample_to_pupil_sample(psf_sample=input_sample_spacing,  # factor of
                                                  samples=samples_x,                  # 1e3 corrects
                                                  wavelength=wavelength,              # for unit
                                                  efl=efl) / 1e3                # translation
    sample_spacing_y = psf_sample_to_pupil_sample(psf_sample=input_sample_spacing,  # factor of
                                                  samples=samples_y,                  # 1e3 corrects
                                                  wavelength=wavelength,              # for unit
                                                  efl=efl) / 1e3                # translation
    x = e.arange(-1 * int(e.ceil(samples_x / 2)), int(e.floor(samples_x / 2))) * sample_spacing_x
    y = e.arange(-1 * int(e.ceil(samples_y / 2)), int(e.floor(samples_y / 2))) * sample_spacing_y
    return x, y


def pupil_sample_to_psf_sample(pupil_sample, samples, wavelength, efl):
    """Convert pupil sample spacing to PSF sample spacing.  fλ/D or Q.

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

    ky, kx = (e.fft.fftfreq(s, sample_spacing).astype(config.precision_complex) for s in field.shape)
    kyy, kxx = e.meshgrid(ky, kx)
    # don't ifftshift, ky, kx computed in shifted space, going to ifft anyway
    forward = e.fft.fft2(e.fft.fftshift(field))
    transfer_function = e.exp(-1j * e.pi * wvl * z * (kxx**2 + kyy**2))
    res = e.fft.ifftshift(e.fft.ifft2(forward * transfer_function))
    return res


class Wavefront(RichData):
    """(Complex) representation of a wavefront."""

    def __init__(self, cmplx_field, dx, wavelength, space='pupil'):
        """Create a new Wavefront instance.

        Parameters
        ----------
        cmplx_field : `numpy.ndarray`
            complex-valued array with both amplitude and phase error
        dx : `float`
            inter-sample spacing, mm (space=pupil) or um (space=psf)
        wavelength : `float`
            wavelength of light, microns
        space : `str`, {'pupil', 'psf'}
            what sort of space the field occupies

        """
        super().__init__(data=cmplx_field, dx=dx, wavelength=wavelength)
        self.space = space

    @property
    def fcn(self):
        """Complex field / wavefunction."""
        warnings.warn("wavefront.fcn property will be deleted in v1 (v0.20+1 release), use .data instead")
        return self.data

    @fcn.setter
    def fcn(self, ary):
        warnings.warn("wavefront.fcn property will be deleted in v1 (v0.20+1 release), use .data instead")
        self.data = ary

    @property
    def intensity(self):
        """Intensity, abs(w)^2."""
        return Wavefront(x=self.x, y=self.y, fcn=abs(self.data)**2, wavelength=self.wavelength, space=self.space)

    @property
    def phase(self):
        """Phase, angle(w).  Possibly wrapped for large OPD."""
        return Wavefront(x=self.x, y=self.y, fcn=e.angle(self.data), wavelength=self.wavelength, space=self.space)

    def __numerical_operation__(self, other, op):
        """Apply an operation to this wavefront with another piece of data."""
        func = getattr(operator, op)
        if isinstance(other, Wavefront):
            criteria = [
                abs(self.sample_spacing - other.sample_spacing) / self.sample_spacing * 100 < 0.001,  # must match to 1 millipercent
                self.shape == other.shape,
                self.wavelength.represents == other.wavelength.represents
            ]

            if not all(criteria):
                raise ValueError('all physicality criteria not met: sample spacing, shape, or wavelength different.')

            data = func(self.data, other.data)
        elif type(other) == type(self.data) or isinstance(other, numbers.Number):  # NOQA
            data = func(other, self.data)
        else:
            raise TypeError(f'unsupported operand type(s) for {op}: \'Wavefront\' and {type(other)}')

        return Wavefront(x=self.x, y=self.y, wavelength=self.wavelength, fcn=data, space=self.space)

    def __mul__(self, other):
        """Multiply this wavefront by something compatible."""
        return self.__numerical_operation__(other, 'mul')

    def __truediv__(self, other):
        """Divide this wavefront by something compatible."""
        return self.__numerical_operation__(other, 'truediv')

    def free_space(self, dz, Q=2):
        """Perform a plane-to-plane free space propagation.

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
        out = angular_spectrum(
            field=self.fcn,
            wvl=self.wavelength.to(u.um),
            sample_spacing=self.sample_spacing,
            z=dz,
            Q=Q)
        return Wavefront(x=self.x, y=self.y, fcn=out, wavelength=self.wavelength, space=self.space)

    def focus(self, efl, Q=2):
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

        data = focus(self.fcn, Q=Q, incoherent=False)
        x, y = focus_units(
            wavefunction=self.fcn,
            input_sample_spacing=self.sample_spacing,
            efl=efl,
            wavelength=self.wavelength.to(u.um),
            Q=Q)

        return Wavefront(x=x, y=y, fcn=data, wavelength=self.wavelength, space='psf')

    def unfocus(self, efl, Q=2):
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
        if self.space != 'psf':
            raise ValueError('can only propagate from a psf to pupil plane')

        data = unfocus(self.fcn, Q=Q)
        x, y = unfocus_units(
            wavefunction=self.fcn,
            input_sample_spacing=self.sample_spacing,
            efl=efl,
            wavelength=self.wavelength.to(u.um),
            Q=Q)

        return Wavefront(x=x, y=y, fcn=data, wavelength=self.wavelength, space='pupil')

    def focus_fixed_sampling(self, efl, sample_spacing, samples):
        """Perform a "pupil" to "psf" propagation with fixed output sampling.

        Uses matrix triple product DFTs to specify the grid directly.

        Parameters
        ----------
        efl : `float`
            focusing distance, millimeters
        sample_spacing : `float`
            output sample spacing, microns
        samples : `int`
            number of samples in the output plane.  If int, interpreted as square
            else interpreted as (x,y), which is the reverse of numpy's (y, x) row major ordering

        Returns
        -------
        `Wavefront`
            the wavefront at the psf plane

        """
        if self.space != 'pupil':
            raise ValueError('can only propagate from a pupil to psf plane')

        if isinstance(samples, int):
            samples = (samples, samples)

        samples_y, samples_x = samples
        # floor div of negative s, not negative of floor div of s
        # has correct rounding semantics for fft grid alignment
        x, y = (e.arange(-s//2, -s//2+s, dtype=config.precision) * sample_spacing for s in (samples_x, samples_y))
        data = focus_fixed_sampling(
            wavefunction=self.fcn,
            input_sample_spacing=self.sample_spacing,
            prop_dist=efl,
            wavelength=self.wavelength.to(u.um),
            output_sample_spacing=sample_spacing,
            output_samples=samples,
            coherent=True, norm=True)

        return Wavefront(x=x, y=y, fcn=data, wavelength=self.wavelength, space='psf')

    def unfocus_fixed_sampling(self, efl, sample_spacing, samples):
        """Perform a "psf" to "pupil" propagation with fixed output sampling.

        Uses matrix triple product DFTs to specify the grid directly.

        Parameters
        ----------
        efl : `float`
            un-focusing distance, millimeters
        sample_spacing : `float`
            output sample spacing, millimeters
        samples : `int`
            number of samples in the output plane.  If int, interpreted as square
            else interpreted as (x,y), which is the reverse of numpy's (y, x) row major ordering

        Returns
        -------
        `Wavefront`
            wavefront at the pupil plane

        """
        if self.space != 'psf':
            raise ValueError('can only propagate from a psf to pupil plane')

        if isinstance(samples, int):
            samples = (samples, samples)

        samples_y, samples_x = samples
        x = e.arange(-1 * int(e.ceil(samples_x / 2)), int(e.floor(samples_x / 2))) * sample_spacing
        y = e.arange(-1 * int(e.ceil(samples_y / 2)), int(e.floor(samples_y / 2))) * sample_spacing

        data = unfocus_fixed_sampling(
            wavefunction=self.fcn,
            input_sample_spacing=self.sample_spacing,
            prop_dist=efl,
            wavelength=self.wavelength.to(u.um),
            output_sample_spacing=sample_spacing,
            output_samples=samples,
            norm=True)

        return Wavefront(x=x, y=y, fcn=data, wavelength=self.wavelength, space='pupil')
