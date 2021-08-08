"""Numerical optical propagation."""
import numbers
import operator
from collections.abc import Iterable


from .conf import config
from .mathops import np, fft
from ._richdata import RichData
from .fttools import pad2d, mdft


def focus(wavefunction, Q):
    """Propagate a pupil plane to a PSF plane.

    Parameters
    ----------
    wavefunction : `numpy.ndarray`
        the pupil wavefunction
    Q : `float`
        oversampling / padding factor

    Returns
    -------
    psf : `numpy.ndarray`
        point spread function

    """
    if Q != 1:
        padded_wavefront = pad2d(wavefunction, Q)
    else:
        padded_wavefront = wavefunction

    impulse_response = fft.fftshift(fft.fft2(fft.ifftshift(padded_wavefront)))
    return impulse_response


def unfocus(wavefunction, Q):
    """Propagate a PSF plane to a pupil plane.

    Parameters
    ----------
    wavefunction : `numpy.ndarray`
        the pupil wavefunction
    Q : `float`
        oversampling / padding factor

    Returns
    -------
    pupil : `numpy.ndarray`
        field in the pupil plane

    """
    if Q != 1:
        padded_wavefront = pad2d(wavefunction, Q)
    else:
        padded_wavefront = wavefunction

    return fft.fftshift(fft.ifft2(fft.ifftshift(padded_wavefront)))


def focus_fixed_sampling(wavefunction, input_dx, prop_dist,
                         wavelength, output_dx, output_samples):
    """Propagate a pupil function to the PSF plane with fixed sampling.

    Parameters
    ----------
    wavefunction : `numpy.ndarray`
        the pupil wavefunction
    input_dx : `float`
        spacing between samples in the pupil plane, millimeters
    prop_dist : `float`
        propagation distance along the z distance
    wavelength : `float`
        wavelength of light
    output_dx : `float`
        sample spacing in the output plane, microns
    output_samples : `int`
        number of samples in the square output array

    Returns
    -------
    data : `numpy.ndarray`
        2D array of data

    """
    dia = wavefunction.shape[0] * input_dx
    Q = Q_for_sampling(input_diameter=dia,
                       prop_dist=prop_dist,
                       wavelength=wavelength,
                       output_dx=output_dx)

    return mdft.dft2(ary=wavefunction, Q=Q, samples=output_samples)


def unfocus_fixed_sampling(wavefunction, input_dx, prop_dist,
                           wavelength, output_dx, output_samples):
    """Propagate an image plane field to the pupil plane with fixed sampling.

    Parameters
    ----------
    wavefunction : `numpy.ndarray`
        the image plane wavefunction
    input_dx : `float`
        spacing between samples in the pupil plane, millimeters
    prop_dist : `float`
        propagation distance along the z distance
    wavelength : `float`
        wavelength of light
    output_dx : `float`
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

    dias = [output_dx * s for s in output_samples]
    dia = max(dias)
    Q = Q_for_sampling(input_diameter=dia,
                       prop_dist=prop_dist,
                       wavelength=wavelength,
                       output_dx=input_dx)  # not a typo

    Q /= wavefunction.shape[0] / output_samples[0]
    field = mdft.idft2(ary=wavefunction, Q=Q, samples=output_samples)
    return field


def Q_for_sampling(input_diameter, prop_dist, wavelength, output_dx):
    """Value of Q for a given output sampling, given input sampling.

    Parameters
    ----------
    input_diameter : `float`
        diameter of the input array in millimeters
    prop_dist : `float`
        propagation distance along the z distance, millimeters
    wavelength : `float`
        wavelength of light, microns
    output_dx : `float`
        sampling in the output plane, microns

    Returns
    -------
    `float`
        requesite Q

    """
    resolution_element = (wavelength * prop_dist) / (input_diameter)
    return resolution_element / output_dx


def focus_units(wavefunction, input_dx, efl, wavelength, Q):
    """Compute the ordinate axes for a pupil plane to PSF plane propagation.

    Parameters
    ----------
    wavefunction : `numpy.ndarray`
        the pupil wavefunction
    input_dx : `float`
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
    dx_x = pupil_sample_to_psf_sample(pupil_sample=input_dx,  # factor of
                                      samples=samples_x,      # 1e3 corrects
                                      wavelength=wavelength,  # for unit
                                      efl=efl) / 1e3          # translation
    dx_y = pupil_sample_to_psf_sample(pupil_sample=input_dx,
                                      samples=samples_y,
                                      wavelength=wavelength,
                                      efl=efl) / 1e3
    x = np.arange(-1 * int(np.ceil(samples_x / 2)), int(np.floor(samples_x / 2))) * dx_x
    y = np.arange(-1 * int(np.ceil(samples_y / 2)), int(np.floor(samples_y / 2))) * dx_y
    return x, y


def unfocus_units(wavefunction, input_dx, efl, wavelength, Q):
    """Compute the ordinate axes for a PSF plane to pupil plane propagation.

    Parameters
    ----------
    wavefunction : `numpy.ndarray`
        the pupil wavefunction
    input_dx : `float`
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
    dx_x = psf_sample_to_pupil_sample(psf_sample=input_dx,    # factor of
                                      samples=samples_x,      # 1e3 corrects
                                      wavelength=wavelength,  # for unit
                                      efl=efl) / 1e3          # translation
    dx_y = psf_sample_to_pupil_sample(psf_sample=input_dx,
                                      samples=samples_y,
                                      wavelength=wavelength,
                                      efl=efl) / 1e3
    x = np.arange(-1 * int(np.ceil(samples_x / 2)), int(np.floor(samples_x / 2))) * dx_x
    y = np.arange(-1 * int(np.ceil(samples_y / 2)), int(np.floor(samples_y / 2))) * dx_y
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
    return (efl * wavelength) / (pupil_sample * samples)


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
    return (efl * wavelength) / (psf_sample * samples)


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
    den = 1 - np.sqrt(1 - lambda_**2/a**2)
    return num / den


def angular_spectrum(field, wvl, dx, z, Q=2):
    """Propagate a field via the angular spectrum method.

    Parameters
    ----------
    field : `numpy.ndarray`
        2D array of complex electric field values
    wvl : `float`
        wavelength of light, microns
    z : `float`
        propagation distance, units of millimeters
    dx : `float`
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

    ky, kx = (fft.fftfreq(s, dx).astype(config.precision) for s in field.shape)
    ky = np.broadcast_to(ky, field.shape).swapaxes(0, 1)
    kx = np.broadcast_to(kx, field.shape)

    transfer_function = np.exp(-1j * np.pi * wvl * z * (kx**2 + ky**2))
    forward = fft.fft2(field)
    return fft.ifft2(forward*transfer_function)


class Wavefront:
    """(Complex) representation of a wavefront."""

    def __init__(self, cmplx_field, wavelength, dx, space='pupil'):
        """Create a new Wavefront instance.

        Parameters
        ----------
        cmplx_field : `numpy.ndarray`
            complex-valued array with both amplitude and phase error
        wavelength : `float`
            wavelength of light, microns
        dx : `float`
            inter-sample spacing, mm (space=pupil) or um (space=psf)
        space : `str`, {'pupil', 'psf'}
            what sort of space the field occupies

        """
        self.data = cmplx_field
        self.wavelength = wavelength
        self.dx = dx
        self.space = space

    @classmethod
    def from_amp_and_phase(cls, amplitude, phase, wavelength, dx):
        """Create a Wavefront from amplitude and phase.

        Parameters
        ----------
        amplitude : `numpy.ndarray`
            array containing the amplitude
        phase : `numpy.ndarray`, optional
            array containing the optical path error with units of nm
            if None, assumed zero
        wavelength : `float`
            wavelength of light with units of microns
        dx : `float`
            sample spacing with units of mm

        """
        if phase is not None:
            phase_prefix = 1j * 2 * np.pi / wavelength / 1e3  # / 1e3 does nm-to-um for phase on a scalar
            P = amplitude * np.exp(phase_prefix * phase)
        else:
            P = amplitude
        return cls(P, wavelength, dx)

    @property
    def intensity(self):
        """Intensity, abs(w)^2."""
        return RichData(abs(self.data)**2, self.dx, self.wavelength)

    @property
    def phase(self):
        """Phase, angle(w).  Possibly wrapped for large OPD."""
        return RichData(np.angle(self.data), self.dx, self.wavelength)

    def __numerical_operation__(self, other, op):
        """Apply an operation to this wavefront with another piece of data."""
        func = getattr(operator, op)
        if isinstance(other, Wavefront):
            criteria = [
                abs(self.dx - other.dx) / self.dx * 100 < 0.1,  # must match to 0.1% (generous, for fp32 compat)
                self.data.shape == other.data.shape,
                self.wavelength == other.wavelength
            ]
            if not all(criteria):
                raise ValueError('all physicality criteria not met: sample spacing, shape, or wavelength different.')

            data = func(self.data, other.data)
        elif type(other) == type(self.data) or isinstance(other, numbers.Number):  # NOQA
            data = func(other, self.data)
        else:
            raise TypeError(f'unsupported operand type(s) for {op}: \'Wavefront\' and {type(other)}')

        return Wavefront(dx=self.dx, wavelength=self.wavelength, cmplx_field=data, space=self.space)

    def __mul__(self, other):
        """Multiply this wavefront by something compatible."""
        return self.__numerical_operation__(other, 'mul')

    def __truediv__(self, other):
        """Divide this wavefront by something compatible."""
        return self.__numerical_operation__(other, 'truediv')

    def free_space(self, dz, Q=1):
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
            field=self.data,
            wvl=self.wavelength,
            dx=self.dx,
            z=dz,
            Q=Q)
        return Wavefront(out, self.wavelength, self.dx, self.space)

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

        data = focus(self.data, Q=Q)
        dx = pupil_sample_to_psf_sample(self.dx, data.shape[1], self.wavelength, efl)

        return Wavefront(data, self.wavelength, dx, space='psf')

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

        data = unfocus(self.data, Q=Q)
        dx = psf_sample_to_pupil_sample(self.dx, data.shape[1], self.wavelength, efl)

        return Wavefront(data, self.wavelength, dx, space='pupil')

    def focus_fixed_sampling(self, efl, dx, samples):
        """Perform a "pupil" to "psf" propagation with fixed output sampling.

        Uses matrix triple product DFTs to specify the grid directly.

        Parameters
        ----------
        efl : `float`
            focusing distance, millimeters
        dx : `float`
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

        data = focus_fixed_sampling(
            wavefunction=self.data,
            input_dx=self.dx,
            prop_dist=efl,
            wavelength=self.wavelength,
            output_dx=dx,
            output_samples=samples,
            coherent=True)

        return Wavefront(dx=dx, cmplx_field=data, wavelength=self.wavelength, space='psf')

    def unfocus_fixed_sampling(self, efl, dx, samples):
        """Perform a "psf" to "pupil" propagation with fixed output sampling.

        Uses matrix triple product DFTs to specify the grid directly.

        Parameters
        ----------
        efl : `float`
            un-focusing distance, millimeters
        dx : `float`
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

        data = unfocus_fixed_sampling(
            wavefunction=self.data,
            input_dx=self.dx,
            prop_dist=efl,
            wavelength=self.wavelength,
            output_dx=dx,
            output_samples=samples)

        return Wavefront(dx=dx, cmplx_field=data, wavelength=self.wavelength, space='pupil')
