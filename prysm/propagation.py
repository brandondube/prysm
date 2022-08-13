"""Numerical optical propagation."""
import copy
import numbers
import operator
import warnings
from collections.abc import Iterable


from .conf import config
from .mathops import np, fft
from ._richdata import RichData
from .fttools import pad2d, crop_center, mdft, czt


def focus(wavefunction, Q):
    """Propagate a pupil plane to a PSF plane.

    Parameters
    ----------
    wavefunction : numpy.ndarray
        the pupil wavefunction
    Q : float
        oversampling / padding factor

    Returns
    -------
    psf : numpy.ndarray
        point spread function

    """
    if Q != 1:
        padded_wavefront = pad2d(wavefunction, Q)
    else:
        padded_wavefront = wavefunction

    impulse_response = fft.fftshift(fft.fft2(fft.ifftshift(padded_wavefront), norm='ortho'))
    return impulse_response


def unfocus(wavefunction, Q):
    """Propagate a PSF plane to a pupil plane.

    Parameters
    ----------
    wavefunction : numpy.ndarray
        the pupil wavefunction
    Q : float
        oversampling / padding factor

    Returns
    -------
    pupil : numpy.ndarray
        field in the pupil plane

    """
    if Q != 1:
        padded_wavefront = pad2d(wavefunction, Q)
    else:
        padded_wavefront = wavefunction

    return fft.fftshift(fft.ifft2(fft.ifftshift(padded_wavefront), norm='ortho'))


def focus_fixed_sampling(wavefunction, input_dx, prop_dist,
                         wavelength, output_dx, output_samples,
                         shift=(0, 0), method='mdft'):
    """Propagate a pupil function to the PSF plane with fixed sampling.

    Parameters
    ----------
    wavefunction : numpy.ndarray
        the pupil wavefunction
    input_dx : float
        spacing between samples in the pupil plane, millimeters
    prop_dist : float
        propagation distance along the z distance
    wavelength : float
        wavelength of light
    output_dx : float
        sample spacing in the output plane, microns
    output_samples : int
        number of samples in the square output array
    shift : tuple of float
        shift in (X, Y), same units as output_dx
    method : str, {'mdft', 'czt'}
        how to propagate the field, matrix DFT or Chirp Z transform
        CZT is usually faster single-threaded and has less memory consumption
        MDFT is usually faster multi-threaded and has more memory consumption

    Returns
    -------
    data : numpy.ndarray
        2D array of data

    """
    dia = wavefunction.shape[0] * input_dx
    Q = Q_for_sampling(input_diameter=dia,
                       prop_dist=prop_dist,
                       wavelength=wavelength,
                       output_dx=output_dx)

    if shift[0] != 0 or shift[1] != 0:
        shift = (shift[0]/output_dx, shift[1]/output_dx)

    if method == 'mdft':
        out = mdft.dft2(ary=wavefunction, Q=Q, samples=output_samples, shift=shift)
    elif method == 'czt':
        out = czt.czt2(ary=wavefunction, Q=Q, samples=output_samples, shift=shift)

    return out


def unfocus_fixed_sampling(wavefunction, input_dx, prop_dist,
                           wavelength, output_dx, output_samples,
                           shift=(0, 0), method='mdft'):
    """Propagate an image plane field to the pupil plane with fixed sampling.

    Parameters
    ----------
    wavefunction : numpy.ndarray
        the image plane wavefunction
    input_dx : float
        spacing between samples in the pupil plane, millimeters
    prop_dist : float
        propagation distance along the z distance
    wavelength : float
        wavelength of light
    output_dx : float
        sample spacing in the output plane, microns
    output_samples : int
        number of samples in the square output array
    shift : tuple of float
        shift in (X, Y), same units as output_dx
    method : str, {'mdft', 'czt'}
        how to propagate the field, matrix DFT or Chirp Z transform
        CZT is usually faster single-threaded and has less memory consumption
        MDFT is usually faster multi-threaded and has more memory consumption

    Returns
    -------
    x : numpy.ndarray
        x axis unit, 1D ndarray
    y : numpy.ndarray
        y axis unit, 1D ndarray
    data : numpy.ndarray
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

    if shift[0] != 0 or shift[1] != 0:
        shift = (shift[0]/output_dx, shift[1]/output_dx)

    if method == 'mdft':
        out = mdft.idft2(ary=wavefunction, Q=Q, samples=output_samples, shift=shift)
    elif method == 'czt':
        out = czt.iczt2(ary=wavefunction, Q=Q, samples=output_samples, shift=shift)

    out *= Q
    return out


def Q_for_sampling(input_diameter, prop_dist, wavelength, output_dx):
    """Value of Q for a given output sampling, given input sampling.

    Parameters
    ----------
    input_diameter : float
        diameter of the input array in millimeters
    prop_dist : float
        propagation distance along the z distance, millimeters
    wavelength : float
        wavelength of light, microns
    output_dx : float
        sampling in the output plane, microns

    Returns
    -------
    float
        requesite Q

    """
    resolution_element = (wavelength * prop_dist) / (input_diameter)
    return resolution_element / output_dx


def pupil_sample_to_psf_sample(pupil_sample, samples, wavelength, efl):
    """Convert pupil sample spacing to PSF sample spacing.  fλ/D or Q.

    Parameters
    ----------
    pupil_sample : float
        sample spacing in the pupil plane
    samples : int
        number of samples present in both planes (must be equal)
    wavelength : float
        wavelength of light, in microns
    efl : float
        effective focal length of the optical system in mm

    Returns
    -------
    float
        the sample spacing in the PSF plane

    """
    return (efl * wavelength) / (pupil_sample * samples)


def psf_sample_to_pupil_sample(psf_sample, samples, wavelength, efl):
    """Convert PSF sample spacing to pupil sample spacing.

    Parameters
    ----------
    psf_sample : float
        sample spacing in the PSF plane
    samples : int
        number of samples present in both planes (must be equal)
    wavelength : float
        wavelength of light, in microns
    efl : float
        effective focal length of the optical system in mm

    Returns
    -------
    float
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
    a : float
        characteristic size ("radius") of an aperture
    L : float
        distance of observation
    lambda_ : float
        wavelength of light, same units as a

    Returns
    -------
    float
        the fresnel number for these parameters

    """
    return a**2 / (L * lambda_)


def talbot_distance(a, lambda_):
    """Compute the talbot distance.

    Parameters
    ----------
    a : float
        period of the grating, units of microns
    lambda_ : float
        wavelength of light, units of microns

    Returns
    -------
    float
        talbot distance, units of microns

    """
    num = lambda_
    den = 1 - np.sqrt(1 - lambda_**2/a**2)
    return num / den


def angular_spectrum(field, wvl, dx, z, Q=2, tf=None):
    """Propagate a field via the angular spectrum method.

    Parameters
    ----------
    field : numpy.ndarray
        2D array of complex electric field values
    wvl : float
        wavelength of light, microns
    z : float
        propagation distance, units of millimeters
    dx : float
        cartesian sample spacing, units of millimeters
    Q : float
        sampling factor used.  Q>=2 for Nyquist sampling of incoherent fields
    tf : numpy.ndarray
        if not None, clobbers all other arguments
        transfer function for the propagation

    Returns
    -------
    numpy.ndarray
        2D ndarray of the output field, complex

    """
    if tf is not None:
        return fft.ifft2(fft.fft2(field) * tf)

    if Q != 1:
        field = pad2d(field, Q=Q)

    transfer_function = angular_spectrum_transfer_function(field.shape, wvl, dx, z)
    forward = fft.fft2(field)
    return fft.ifft2(forward*transfer_function)


def angular_spectrum_transfer_function(samples, wvl, dx, z):
    """Precompute the transfer function of free space.

    Parameters
    ----------
    samples : int or tuple
        (y,x) or (r,c) samples in the output array
    wvl : float
        wavelength of light, microns
    dx : float
        intersample spacing, mm
    z : float
        propagation distance, mm

    Returns
    -------
    numpy.ndarray
        ndarray of shape samples containing the complex valued transfer function
        such that X = fft2(x); xhat = ifft2(X*tf) is signal x after free space propagation

    """
    if isinstance(samples, int):
        samples = (samples, samples)

    wvl = wvl / 1e3
    ky, kx = (fft.fftfreq(s, dx).astype(config.precision) for s in samples)
    ky = np.broadcast_to(ky, samples).swapaxes(0, 1)
    kx = np.broadcast_to(kx, samples)

    return np.exp(-1j * np.pi * wvl * z * (kx*kx + ky*ky))


class Wavefront:
    """(Complex) representation of a wavefront."""

    def __init__(self, cmplx_field, wavelength, dx, space='pupil'):
        """Create a new Wavefront instance.

        Parameters
        ----------
        cmplx_field : numpy.ndarray
            complex-valued array with both amplitude and phase error
        wavelength : float
            wavelength of light, microns
        dx : float
            inter-sample spacing, mm (space=pupil) or um (space=psf)
        space : str, {'pupil', 'psf'}
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
        amplitude : numpy.ndarray
            array containing the amplitude
        phase : numpy.ndarray, optional
            array containing the optical path error with units of nm
            if None, assumed zero
        wavelength : float
            wavelength of light with units of microns
        dx : float
            sample spacing with units of mm

        """
        if phase is not None:
            phase_prefix = 1j * 2 * np.pi / wavelength / 1e3  # / 1e3 does nm-to-um for phase on a scalar
            P = amplitude * np.exp(phase_prefix * phase)
        else:
            P = amplitude
        return cls(P, wavelength, dx)

    @classmethod
    def thin_lens(cls, f, wavelength, x, y):
        """Create a thin lens, used in focusing beams.

        Users are encouraged to not use thin lens + free space propagation to
        take beams to their focus.  In nearly all cases, a different propagation
        scheme is significantly more computational efficient.  For example,
        just using the wf.focus() method.  If you have access to the (unwrapped)
        phase, it is also cheaper to compute the quadratic phase you want and
        add that before wf.from_amp_and_phase) instead of multiplying by a thin
        lens.

        Parameters
        ----------
        f : float
            focal length of the lens, millimeters
        wavelength : float
            wavelength of light, microns
        x : numpy.ndarray
            x coordinates that define the space of the lens, mm
        y : numpy.ndarray
            y coordinates that define the space of the beam, mm

        Returns
        -------
        Wavefront
            a wavefront object having quadratic phase which, when multiplied
            by another wavefront acts as a thin lens

        """
        # the kernel is simply
        #
        # 2pi i  r^2
        # ----- -----
        #  wvl   2f
        #
        # for dimensional reduction to be unitless, wvl, r, f all need the same
        # units, so scale wvl
        w = wavelength / 1e3  # um -> mm
        term1 = 1j * 2 * np.pi / w

        rsq = x * x + y * y
        term2 = rsq / (2 * f)

        cmplx_screen = np.exp(term1 * term2)
        dx = float(x[0, 1] - x[0, 0])  # float conversion for CuPy support
        return cls(cmplx_field=cmplx_screen, wavelength=wavelength, dx=dx, space='pupil')

    @property
    def intensity(self):
        """Intensity, abs(w)^2."""
        return RichData(abs(self.data)**2, self.dx, self.wavelength)

    @property
    def phase(self):
        """Phase, angle(w).  Possibly wrapped for large OPD."""
        return RichData(np.angle(self.data), self.dx, self.wavelength)

    def copy(self):
        """Return a (deep) copy of this instance."""
        return copy.deepcopy(self)

    def pad2d(self, Q, value=0, mode='constant', out_shape=None, inplace=True):
        """Pad the wavefront.

        Parameters
        ----------
        array : numpy.ndarray
            source array
        Q : float, optional
            oversampling factor; ratio of input to output array widths
        value : float, optioanl
            value with which to pad the array
        mode : str, optional
            mode, passed directly to np.pad
        out_shape : tuple
            output shape for the array.  Overrides Q if given.
            in_shape * Q ~= out_shape (up to integer rounding)
        inplace : bool, optional
            if True, mutate this wf and return it, else
            create a new wf with cropped data

        Returns
        -------
        Wavefront
            wavefront with padded data

        """
        padded = pad2d(self.data, Q=Q, value=value, mode=mode, out_shape=out_shape)
        if inplace:
            self.data = padded
            return self

        out = Wavefront(padded, self.wavelength, self.dx, self.space)
        return out

    def crop(self, out_shape, inplace=True):
        """Crop the wavefront to the centermost (out_shape).

        Parameters
        ----------
        out_shape : int or tuple of (int, int)
            the output shape (aka number of pixels) to crop to.
        inplace : bool, optional
            if True, mutate this wf and return it, else
            create a new wf with cropped data
            if out-of-place, will share memory with self via overlap of data

        Returns
        -------
        Wavefront
            cropped wavefront

        """
        cropped = crop_center(self.data, out_shape)
        if inplace:
            self.data = cropped
            return self

        out = Wavefront(cropped, self.wavelength, self.dx, self.space)
        return out

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

    def free_space(self, dz=np.nan, Q=1, tf=None):
        """Perform a plane-to-plane free space propagation.

        Uses angular spectrum and the free space kernel.

        Parameters
        ----------
        dz : float
            inter-plane distance, millimeters
        Q : float
            padding factor.  Q=1 does no padding, Q=2 pads 1024 to 2048.
        tf : numpy.ndarray
            if not None, clobbers all other arguments
            transfer function for the propagation

        Returns
        -------
        Wavefront
            the wavefront at the new plane

        """
        if np.isnan(dz) and tf is None:
            raise ValueError('dz must be provided if tf is None')
        out = angular_spectrum(
            field=self.data,
            wvl=self.wavelength,
            dx=self.dx,
            z=dz,
            Q=Q,
            tf=tf)
        return Wavefront(out, self.wavelength, self.dx, self.space)

    def focus(self, efl, Q=2):
        """Perform a "pupil" to "psf" plane propgation.

        Uses an FFT with no quadratic phase.

        Parameters
        ----------
        efl : float
            focusing distance, millimeters
        Q : float
            padding factor.  Q=1 does no padding, Q=2 pads 1024 to 2048.
            To avoid aliasng, the array must be padded such that Q is at least 2
            this may happen organically if your data does not span the array.

        Returns
        -------
        Wavefront
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
        efl : float
            un-focusing distance, millimeters
        Q : float
            padding factor.  Q=1 does no padding, Q=2 pads 1024 to 2048.
            To avoid aliasng, the array must be padded such that Q is at least 2
            this may happen organically if your data does not span the array.

        Returns
        -------
        Wavefront
            the wavefront at the pupil plane

        """
        if self.space != 'psf':
            raise ValueError('can only propagate from a psf to pupil plane')

        data = unfocus(self.data, Q=Q)
        dx = psf_sample_to_pupil_sample(self.dx, data.shape[1], self.wavelength, efl)

        return Wavefront(data, self.wavelength, dx, space='pupil')

    def focus_fixed_sampling(self, efl, dx, samples, shift=(0, 0), method='mdft'):
        """Perform a "pupil" to "psf" propagation with fixed output sampling.

        Uses matrix triple product DFTs to specify the grid directly.

        Parameters
        ----------
        efl : float
            focusing distance, millimeters
        dx : float
            output sample spacing, microns
        samples : int
            number of samples in the output plane.  If int, interpreted as square
            else interpreted as (x,y), which is the reverse of numpy's (y, x) row major ordering
        shift : tuple of float
            shift in (X, Y), same units as output_dx
        method : str, {'mdft', 'czt'}
            how to propagate the field, matrix DFT or Chirp Z transform
            CZT is usually faster single-threaded and has less memory consumption
            MDFT is usually faster multi-threaded and has more memory consumption

        Returns
        -------
        Wavefront
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
            shift=shift,
            method=method)

        return Wavefront(dx=dx, cmplx_field=data, wavelength=self.wavelength, space='psf')

    def unfocus_fixed_sampling(self, efl, dx, samples, shift=(0, 0), method='mdft'):
        """Perform a "psf" to "pupil" propagation with fixed output sampling.

        Uses matrix triple product DFTs to specify the grid directly.

        Parameters
        ----------
        efl : float
            un-focusing distance, millimeters
        dx : float
            output sample spacing, millimeters
        samples : int
            number of samples in the output plane.  If int, interpreted as square
            else interpreted as (x,y), which is the reverse of numpy's (y, x) row major ordering
        shift : tuple of float
            shift in (X, Y), same units as output_dx
        method : str, {'mdft', 'czt'}
            how to propagate the field, matrix DFT or Chirp Z transform
            CZT is usually faster single-threaded and has less memory consumption
            MDFT is usually faster multi-threaded and has more memory consumption

        Returns
        -------
        Wavefront
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
            output_samples=samples,
            shift=shift,
            method=method)

        return Wavefront(dx=dx, cmplx_field=data, wavelength=self.wavelength, space='pupil')

    def to_fpm_and_back(self, efl, fpm, fpm_dx=None, method='mdft', return_more=False):
        """Propagate to a focal plane mask, apply it, and return.

        This routine handles normalization properly for the user.

        To invoke babinet's principle, simply use to_fpm_and_back(fpm=1 - fpm).

        Parameters
        ----------
        efl : float
            focal length for the propagation
        fpm : Wavefront or numpy.ndarray
            the focal plane mask
        fpm_dx : float
            sampling increment in the focal plane,  microns;
            do not need to pass if fpm is a Wavefront
        method : str, {'mdft', 'czt'}
            how to propagate the field, matrix DFT or Chirp Z transform
            CZT is usually faster single-threaded and has less memory consumption
            MDFT is usually faster multi-threaded and has more memory consumption
        return_more : bool
            if True, return (new_wavefront, field_at_fpm, field_after_fpm)
            else return new_wavefront

        Returns
        -------
        Wavefront, Wavefront, Wavefront
            new wavefront, [field at fpm, field after fpm]

        """
        if isinstance(fpm, Wavefront):
            fpm_samples = fpm.data.shape
            fpm_dx = fpm.dx
        else:
            if fpm_dx is None:
                raise ValueError('fpm was not a Wavefront and fpm_dx was None')

            fpm_samples = fpm.shape

        input_samples = self.data.shape
        input_diameters = [self.dx * s for s in input_samples]
        Q_forward = [Q_for_sampling(d, efl, self.wavelength, fpm_dx) for d in input_diameters]
        # soummer notation: use m, which would be 0.5 for a 2x zoom
        # BDD notation: Q, would be 2 for a 2x zoom
        m_forward = [1/q for q in Q_forward]
        m_reverse = [b/a*m for a, b, m in zip(input_samples, fpm_samples, m_forward)]
        Q_reverse = [1/m for m in m_reverse]

        # prop forward
        kwargs = dict(ary=self.data, Q=Q_forward, samples=fpm_samples, shift=(0, 0))
        if method == 'mdft':
            field_at_fpm = mdft.idft2(**kwargs)
        elif method == 'czt':
            field_at_fpm = czt.iczt2(**kwargs)

        field_after_fpm = field_at_fpm * fpm

        kwargs = dict(ary=field_after_fpm.data, Q=Q_reverse, samples=input_samples, shift=(0, 0))
        if method == 'mdft':
            field_at_next_pupil = mdft.idft2(**kwargs)
        elif method == 'czt':
            field_at_next_pupil = czt.iczt2(**kwargs)

        # scaling
        # TODO: make this handle anamorphic transforms properly
        if Q_forward[0] != Q_forward[1]:
            warnings.warn(f'Forward propagation had Q {Q_forward} which was not uniform between axes, scaling is off')
        if input_samples[0] != input_samples[1]:
            warnings.warn(f'Forward propagation had input shape {input_samples} which was not uniform between axes, scaling is off')
        if fpm_samples[0] != fpm_samples[1]:
            warnings.warn(f'Forward propagation had fpm shape {fpm_samples} which was not uniform between axes, scaling is off')
        # Q_reverse is calculated from Q_forward; if one is consistent the other is

        out = Wavefront(field_at_next_pupil, self.wavelength, self.dx, self.space)
        if return_more:
            return out, field_at_fpm, Wavefront(field_after_fpm, self.wavelength, fpm_dx, 'psf')

        return out

    def babinet(self, efl, lyot, fpm, fpm_dx=None, method='mdft', return_more=False):
        """Propagate through a Lyot-style coronagraph using Babinet's principle.

        This routine handles normalization properly for the user.

        To invoke babinet's principle, simply use to_fpm_and_back(fpm=1 - fpm).

        Parameters
        ----------
        efl : float
            focal length for the propagation
        lyot : Wavefront or numpy.ndarray
            the Lyot stop; if None, equivalent to ones_like(self.data)
        fpm : Wavefront or numpy.ndarray
            1 - fpm
            one minus the focal plane mask (see Soummer et al 2007)
        fpm_dx : float
            sampling increment in the focal plane,  microns;
            do not need to pass if fpm is a Wavefront
        method : str, {'mdft', 'czt'}
            how to propagate the field, matrix DFT or Chirp Z transform
            CZT is usually faster single-threaded and has less memory consumption
            MDFT is usually faster multi-threaded and has more memory consumption
        return_more : bool
            if True, return each plane in the propagation
            else return new_wavefront

        Returns
        -------
        Wavefront, Wavefront, Wavefront, Wavefront
            field after lyot, [field at fpm, field after fpm, field at lyot]

        """
        if return_more:
            field, field_at_fpm, field_after_fpm = \
                self.to_fpm_and_back(efl=efl, fpm=fpm, fpm_dx=fpm_dx, method=method,
                                     return_more=return_more)
        else:
            field = self.to_fpm_and_back(efl=efl, fpm=fpm, fpm_dx=fpm_dx, method=method,
                                         return_more=return_more)
        # DOI: 10.1117/1.JATIS.7.1.019002
        # Eq. 26 with some minor differences in naming
        field_at_lyot = self.data - field.data
        if lyot is not None:
            field_after_lyot = lyot * field_at_lyot
        else:
            field_after_lyot = field_at_lyot

        field_at_lyot = Wavefront(field_at_lyot, self.wavelength, self.dx, self.space)
        field_after_lyot = Wavefront(field_after_lyot, self.wavelength, self.dx, self.space)

        if return_more:
            return field_after_lyot, field_at_fpm, field_after_fpm, field_at_lyot
        return field_after_lyot
