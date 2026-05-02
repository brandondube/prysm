"""Numerical optical propagation."""
import copy
import numbers
import operator
from collections.abc import Iterable

from .conf import config
from .mathops import np, fft, is_odd
from ._richdata import RichData
from .fttools import pad2d, crop_center, fftrange, MDFT, CZT


def focus(wavefunction, Q):
    """Propagate a pupil plane to a PSF plane.

    Parameters
    ----------
    wavefunction : ndarray
        the pupil wavefunction
    Q : float
        oversampling / padding factor

    Returns
    -------
    psf : ndarray
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
    wavefunction : ndarray
        the pupil wavefunction
    Q : float
        oversampling / padding factor

    Returns
    -------
    pupil : ndarray
        field in the pupil plane

    """
    if Q != 1:
        padded_wavefront = pad2d(wavefunction, Q)
    else:
        padded_wavefront = wavefunction

    return fft.fftshift(fft.ifft2(fft.ifftshift(padded_wavefront), norm='ortho'))


def coordinates_for_focus(pupil_dx, pupil_samples, focal_dx, focal_samples,
                          wavelength, efl, focal_shift=(0, 0)):
    """Coordinate / frequency vectors for an MDFT-based pupil ↔ focal propagation.

    The Fraunhofer kernel is ``exp(-2πi · x_pupil · x_focal / (λ · efl))``. This
    returns the input pupil coordinates ``(x, y)`` and the spatial frequencies
    ``(fx, fy)`` that pair with them, where ``fx = x_focal / (λ · efl)``. The
    same ``(x, y, fx, fy)`` quartet works for both directions:

    - Focus (pupil → focal):    ``MDFT(x, y, fx, fy, sign=-1)(pupil) * norm``
    - Unfocus (focal → pupil):  ``MDFT(x, y, fx, fy, sign=-1).adjoint(focal) * norm``

    where ``norm = pupil_dx * focal_dx / (wavelength * efl)``. The unfocus form
    works because the adjoint of the focus operator is the inverse propagation
    (up to the real scalar normalization).

    Parameters
    ----------
    pupil_dx : float
        pupil-plane sample spacing, mm
    pupil_samples : int or (int, int)
        pupil samples; a single int is treated as square, a tuple as ``(rows, cols)``
    focal_dx : float
        focal-plane sample spacing, microns
    focal_samples : int or (int, int)
        focal samples; a single int is treated as square, a tuple as ``(rows, cols)``
    wavelength : float
        wavelength of light, microns
    efl : float
        effective focal length, mm
    focal_shift : (float, float)
        ``(x, y)`` translation of the focal grid center, microns

    Returns
    -------
    x, y : ndarray
        pupil coordinates along the column and row axes, mm
    fx, fy : ndarray
        spatial frequencies along the column and row axes, 1/mm

    """
    if not isinstance(pupil_samples, Iterable):
        pupil_samples = (pupil_samples, pupil_samples)
    if not isinstance(focal_samples, Iterable):
        focal_samples = (focal_samples, focal_samples)

    pny, pnx = pupil_samples
    fny, fnx = focal_samples
    fsx, fsy = focal_shift

    dtype = config.precision
    x = fftrange(pnx, dtype=dtype) * pupil_dx
    y = fftrange(pny, dtype=dtype) * pupil_dx
    # focal positions in microns, then convert to spatial frequency 1/mm:
    # fx = x_focal_mm / (lambda_mm * efl) = x_focal_um / (wavelength_um * efl_mm)
    inv_lz = 1.0 / (wavelength * efl)
    fx = (fftrange(fnx, dtype=dtype) * focal_dx + fsx) * inv_lz
    fy = (fftrange(fny, dtype=dtype) * focal_dx + fsy) * inv_lz
    return x, y, fx, fy


def _focus_norm(pupil_dx, focal_dx, wavelength, prop_dist):
    """Scaling factor that turns a bare MDFT/CZT sum into a unitary-equivalent
    focal-plane field. ``mm * um / (um * mm) = 1`` (dimensionless)."""
    return (pupil_dx * focal_dx) / (wavelength * prop_dist)


def _coords_focal_to_pupil(focal_dx, focal_samples, pupil_dx, pupil_samples,
                           wavelength, efl, focal_shift=(0, 0)):
    """Coordinate / frequency vectors for a focal → pupil MDFT/CZT, oriented
    so that the operator can be applied directly to focal data without using
    the adjoint. Used by the CZT unfocus path, since CZT has no adjoint.
    """
    if not isinstance(pupil_samples, Iterable):
        pupil_samples = (pupil_samples, pupil_samples)
    if not isinstance(focal_samples, Iterable):
        focal_samples = (focal_samples, focal_samples)

    pny, pnx = pupil_samples
    fny, fnx = focal_samples
    fsx, fsy = focal_shift
    dtype = config.precision

    # input coordinates: focal positions in mm (focal_dx is um)
    x = (fftrange(fnx, dtype=dtype) * focal_dx + fsx) * 1e-3
    y = (fftrange(fny, dtype=dtype) * focal_dx + fsy) * 1e-3
    # output frequencies: pupil_mm / (lambda_mm * efl_mm)
    inv_lz_mm = 1e3 / (wavelength * efl)
    fx = fftrange(pnx, dtype=dtype) * pupil_dx * inv_lz_mm
    fy = fftrange(pny, dtype=dtype) * pupil_dx * inv_lz_mm
    return x, y, fx, fy


def focus_fixed_sampling(wavefunction, input_dx, prop_dist,
                         wavelength, output_dx, output_samples,
                         shift=(0, 0), method='mdft'):
    """Propagate a pupil function to the PSF plane with fixed sampling.

    Parameters
    ----------
    wavefunction : ndarray
        the pupil wavefunction
    input_dx : float
        spacing between samples in the pupil plane, mm
    prop_dist : float
        propagation distance along the z axis, mm
    wavelength : float
        wavelength of light, microns
    output_dx : float
        sample spacing in the output plane, microns
    output_samples : int or (int, int)
        number of samples in the output array
    shift : (float, float)
        shift in (X, Y) of the focal grid center, microns
    method : str, {'mdft', 'czt'}
        how to propagate the field, matrix DFT or Chirp Z transform

    Returns
    -------
    ndarray
        focal-plane field

    """
    x, y, fx, fy = coordinates_for_focus(
        pupil_dx=input_dx, pupil_samples=wavefunction.shape,
        focal_dx=output_dx, focal_samples=output_samples,
        wavelength=wavelength, efl=prop_dist, focal_shift=shift,
    )
    norm = _focus_norm(input_dx, output_dx, wavelength, prop_dist)
    if method == 'mdft':
        op = MDFT(x, y, fx, fy, sign=-1)
    elif method == 'czt':
        op = CZT(x, y, fx, fy, sign=-1)
    else:
        raise ValueError(f"method must be 'mdft' or 'czt', got {method!r}")
    return op(wavefunction) * norm


def focus_fixed_sampling_backprop(wavefunction, input_dx, prop_dist,
                                  wavelength, output_dx, output_samples,
                                  shift=(0, 0), method='mdft'):
    """Backpropagate gradient through focus_fixed_sampling.

    Parameters
    ----------
    wavefunction : ndarray
        gradient at the PSF plane (the forward output)
    input_dx : float
        pupil sample spacing, mm (matches the forward call)
    prop_dist : float
        propagation distance, mm
    wavelength : float
        wavelength of light, microns
    output_dx : float
        focal sample spacing, microns (matches the forward call)
    output_samples : int or (int, int)
        shape of the *pupil* array — the shape of the returned gradient
    shift : (float, float)
        focal-grid shift, microns
    method : str, {'mdft', 'czt'}
        propagation method; CZT backprop is not implemented and will raise

    Returns
    -------
    ndarray
        gradient at the pupil plane, shape ``output_samples``

    """
    x, y, fx, fy = coordinates_for_focus(
        pupil_dx=input_dx, pupil_samples=output_samples,
        focal_dx=output_dx, focal_samples=wavefunction.shape,
        wavelength=wavelength, efl=prop_dist, focal_shift=shift,
    )
    norm = _focus_norm(input_dx, output_dx, wavelength, prop_dist)
    if method == 'mdft':
        op = MDFT(x, y, fx, fy, sign=-1)
    elif method == 'czt':
        raise ValueError('gradient backpropagation not yet implemented for CZT')
    else:
        raise ValueError(f"method must be 'mdft' or 'czt', got {method!r}")
    return op.adjoint(wavefunction) * norm


def unfocus_fixed_sampling(wavefunction, input_dx, prop_dist,
                           wavelength, output_dx, output_samples,
                           shift=(0, 0), method='mdft'):
    """Propagate an image-plane field to the pupil plane with fixed sampling.

    Parameters
    ----------
    wavefunction : ndarray
        the image-plane wavefunction
    input_dx : float
        focal sample spacing, microns
    prop_dist : float
        propagation distance, mm
    wavelength : float
        wavelength of light, microns
    output_dx : float
        pupil sample spacing in the output, mm
    output_samples : int or (int, int)
        number of pupil samples in the output
    shift : (float, float)
        shift in (X, Y) of the focal grid center, microns
    method : str, {'mdft', 'czt'}
        how to propagate the field, matrix DFT or Chirp Z transform

    Returns
    -------
    ndarray
        pupil-plane field

    """
    norm = _focus_norm(output_dx, input_dx, wavelength, prop_dist)
    if method == 'mdft':
        # adjoint of the focus operator is the inverse propagation
        x, y, fx, fy = coordinates_for_focus(
            pupil_dx=output_dx, pupil_samples=output_samples,
            focal_dx=input_dx, focal_samples=wavefunction.shape,
            wavelength=wavelength, efl=prop_dist, focal_shift=shift,
        )
        return MDFT(x, y, fx, fy, sign=-1).adjoint(wavefunction) * norm
    elif method == 'czt':
        # CZT has no adjoint; drive it directly in the focal → pupil orientation
        x, y, fx, fy = _coords_focal_to_pupil(
            focal_dx=input_dx, focal_samples=wavefunction.shape,
            pupil_dx=output_dx, pupil_samples=output_samples,
            wavelength=wavelength, efl=prop_dist, focal_shift=shift,
        )
        return CZT(x, y, fx, fy, sign=+1)(wavefunction) * norm
    else:
        raise ValueError(f"method must be 'mdft' or 'czt', got {method!r}")


def unfocus_fixed_sampling_backprop(wavefunction, input_dx, prop_dist,
                                    wavelength, output_dx, output_samples,
                                    shift=(0, 0), method='mdft'):
    """Backpropagate gradient through unfocus_fixed_sampling.

    Parameters
    ----------
    wavefunction : ndarray
        gradient at the pupil plane (the forward output)
    input_dx : float
        focal sample spacing, microns (matches the forward call)
    prop_dist : float
        propagation distance, mm
    wavelength : float
        wavelength of light, microns
    output_dx : float
        pupil sample spacing, mm (matches the forward call)
    output_samples : int or (int, int)
        shape used as the forward call's ``output_samples`` (pupil shape)
    shift : (float, float)
        focal-grid shift, microns
    method : str, {'mdft', 'czt'}
        propagation method; CZT backprop is not implemented and will raise

    Returns
    -------
    ndarray
        gradient at the focal plane

    """
    # backprop input (wavefunction) is at pupil plane; backprop output is at
    # focal plane. output_samples names the *focal* shape we want to return.
    x, y, fx, fy = coordinates_for_focus(
        pupil_dx=output_dx, pupil_samples=wavefunction.shape,
        focal_dx=input_dx, focal_samples=output_samples,
        wavelength=wavelength, efl=prop_dist, focal_shift=shift,
    )
    norm = _focus_norm(output_dx, input_dx, wavelength, prop_dist)
    if method == 'mdft':
        # adjoint of unfocus is focus
        return MDFT(x, y, fx, fy, sign=-1)(wavefunction) * norm
    elif method == 'czt':
        raise ValueError('gradient backpropagation not yet implemented for CZT')
    else:
        raise ValueError(f"method must be 'mdft' or 'czt', got {method!r}")


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
    field : ndarray
        2D array of complex electric field values
    wvl : float
        wavelength of light, microns
    z : float
        propagation distance, units of millimeters
    dx : float
        cartesian sample spacing, units of millimeters
    Q : float
        sampling factor used.  Q>=2 for Nyquist sampling of incoherent fields
    tf : ndarray
        if not None, clobbers all other arguments
        transfer function for the propagation

    Returns
    -------
    ndarray
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
    ndarray
        ndarray of shape samples containing the complex valued transfer function
        such that X = fft2(x); xhat = ifft2(X*tf) is signal x after free space propagation

    """
    if isinstance(samples, int):
        samples = (samples, samples)

    wvl = wvl / 1e3
    ky, kx = (fft.fftfreq(s, dx).astype(config.precision) for s in samples)
    kxx = kx * kx
    kyy = ky * ky

    prefix = -1j*np.pi*wvl*z
    tfx = np.exp(prefix*kxx)
    tfy = np.exp(prefix*kyy)
    return np.outer(tfy, tfx)


def to_fpm_and_back(wavefunction, dx, efl, wavelength, fpm, fpm_dx, shift=(0, 0), method='mdft', return_more=False):
    """Propagate to a focal plane mask, apply it, and return.

    This routine handles normalization properly for the user.

    To invoke babinet's principle, simply use to_fpm_and_back(fpm=1 - fpm).

    Parameters
    ----------
    wavefunction : ndarray
        complex wave to propagate
    dx : float
        inter-sample spacing of wavefunction, mm
    efl : float
        focal length for the propagation
    wavelength : float
        wavelength of light to propagate at, um
    fpm : Wavefront or ndarray
        the focal plane mask
    fpm_dx : float
        sampling increment in the focal plane,  microns;
        do not need to pass if fpm is a Wavefront
    shift : tuple of float, optional
        shift in the image plane to go to the FPM
        appropriate shift will be computed returning to the pupil
    method : str, {'mdft', 'czt'}, optional
        how to propagate the field, matrix DFT or Chirp Z transform
        CZT is usually faster single-threaded and has less memory consumption
        MDFT is usually faster multi-threaded and has more memory consumption
    return_more : bool, optional
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
        fpm = fpm.data
    else:
        if fpm_dx is None:
            raise ValueError('fpm was not a Wavefront and fpm_dx was None')

        fpm_samples = fpm.shape

    field_at_fpm = focus_fixed_sampling(wavefunction, dx, efl, wavelength, fpm_dx, fpm_samples, shift=shift, method=method)  # NOQA

    field_after_fpm = field_at_fpm * fpm

    field_at_next_pupil = unfocus_fixed_sampling(field_after_fpm, fpm_dx, efl, wavelength, dx, wavefunction.shape, shift=shift, method=method)  # NOQA

    if return_more:
        return field_at_next_pupil, field_at_fpm, field_after_fpm
    return field_at_next_pupil


def to_fpm_and_back_backprop(wavefunction, dx, efl, wavelength, fpm, fpm_dx=None,
                             method='mdft', shift=(0, 0), return_more=False):
    """Backpropagate gradient through to_fpm_and_back.

    Parameters
    ----------
    wavefunction : ndarray
        gradient at the next pupil plane (output of the forward to_fpm_and_back)
    dx : float
        inter-sample spacing of the pupil-plane wavefront, mm
    efl : float
        focal length used in the forward propagation
    wavelength : float
        wavelength of light, um
    fpm : Wavefront or ndarray
        the focal plane mask used in the forward propagation
    fpm_dx : float
        sampling increment in the focal plane, microns;
        do not need to pass if fpm is a Wavefront
    method : str, {'mdft', 'czt'}, optional
        propagation method (must match the forward call)
    shift : tuple of float, optional
        shift used in the forward propagation
    return_more : bool, optional
        if True, return (Eabar, Ebbar, intermediate)
        else return Eabar

    Returns
    -------
    ndarray or tuple of ndarray
        gradient at the input pupil; optionally also the intermediate gradients

    """
    if isinstance(fpm, Wavefront):
        fpm_samples = fpm.data.shape
        fpm_dx = fpm.dx
        fpm = fpm.data
    else:
        if fpm_dx is None:
            raise ValueError('fpm was not a Wavefront and fpm_dx was None')

        fpm_samples = fpm.shape

    # do not take complex conjugate of reals (no-op, but numpy still does it)
    if np.iscomplexobj(fpm):
        fpm = fpm.conj()

    Ebbar = unfocus_fixed_sampling_backprop(wavefunction, fpm_dx, efl, wavelength, dx, fpm_samples)
    intermediate = Ebbar * fpm
    Eabar = focus_fixed_sampling_backprop(intermediate, dx, efl, wavelength, fpm_dx, wavefunction.shape)
    if return_more:
        return Eabar, Ebbar, intermediate
    else:
        return Eabar


class Wavefront:
    """(Complex) representation of a wavefront."""

    def __init__(self, cmplx_field, wavelength, dx, space='pupil'):
        """Create a new Wavefront instance.

        Parameters
        ----------
        cmplx_field : ndarray
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
        amplitude : ndarray
            array containing the amplitude
        phase : ndarray, optional
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
    def phase_screen(cls, phase, wavelength, dx):
        """Create a new complex phase screen.

        Parameters
        ----------
        phase : ndarray
            phase or optical path error, units of nm
        wavelength : float
            wavelength of light with units of microns
        dx : float
            sample spacing with units of mm

            """
        phase_prefix = 1j * 2 * np.pi / wavelength / 1e3  # / 1e3 does nm-to-um for phase on a scalar
        E = np.exp(phase_prefix*phase)
        return cls(E, wavelength, dx)

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
        x : ndarray
            x coordinates that define the space of the lens, mm
        y : ndarray
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
        term1 = -1j * 2 * np.pi / w

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

    @property
    def real(self):
        """re(w)."""
        return RichData(np.real(self.data), self.dx, self.wavelength)

    @property
    def imag(self):
        """im(w)."""
        return RichData(np.imag(self.data), self.dx, self.wavelength)

    def copy(self):
        """Return a (deep) copy of this instance."""
        return copy.deepcopy(self)

    def from_amp_and_phase_backprop_phase(self, wf_bar):
        """Gradient backpropagation through from_amp_and_phase -> phase.

        Parameters
        ----------
        wf_bar : Wavefront
            the gradient backpropagated up to wf

        Returns
        -------
        ndarray
            gradient backpropagated to the phase of wf_in

        """
        k = 2 * np.pi / self.wavelength / 1e3  # um -> nm
        # imag(gbar*g)
        return k * np.imag(wf_bar.data * np.conj(self.data))

    def intensity_backprop(self, intensity_bar):
        """Gradient backpropagation through from_amp_and_phase -> phase.

        Parameters
        ----------
        intensity_bar : Wavefront
            the gradient backpropagated up to the intensity step

        Returns
        -------
        ndarray
            gradient backpropagated to the complex wavefront before
            intensity was calculated

        """
        Gbar = 2 * intensity_bar * self.data
        return Wavefront(Gbar, self.wavelength, self.dx, self.space)

    def pad2d(self, Q, value=0, mode='constant', out_shape=None, inplace=True):
        """Pad the wavefront.

        Parameters
        ----------
        array : ndarray
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
            data = func(self.data, other)
        else:
            raise TypeError(f'unsupported operand type(s) for {op}: \'Wavefront\' and {type(other)}')

        return Wavefront(dx=self.dx, wavelength=self.wavelength, cmplx_field=data, space=self.space)

    def __mul__(self, other):
        """Multiply this wavefront by something compatible."""
        return self.__numerical_operation__(other, 'mul')

    def __truediv__(self, other):
        """Divide this wavefront by something compatible."""
        return self.__numerical_operation__(other, 'truediv')

    def __add__(self, other):
        """Perform elementwise addition with other, e1+e2."""
        return self.__numerical_operation__(other, 'add')

    def __sub__(self, other):
        """Perform elementwise subtraction with other, e1-e2."""
        return self.__numerical_operation__(other, 'sub')

    def free_space(self, dz=np.nan, Q=1, tf=None):
        """Perform a plane-to-plane free space propagation.

        Uses angular spectrum and the free space kernel.

        Parameters
        ----------
        dz : float
            inter-plane distance, millimeters
        Q : float
            padding factor.  Q=1 does no padding, Q=2 pads 1024 to 2048.
        tf : ndarray
            if not None, clobbers all other arguments
            transfer function for the propagation

        Returns
        -------
        Wavefront
            the wavefront at the new plane

        """
        if np.isnan(dz) and tf is None:
            raise ValueError('dz must be provided if tf is None')
        out = angular_spectrum(self.data,
                               wvl=self.wavelength,
                               dx=self.dx,
                               z=dz,
                               Q=Q,
                               tf=tf,
        )
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

        data = focus_fixed_sampling(self.data,
                                    input_dx=self.dx,
                                    prop_dist=efl,
                                    wavelength=self.wavelength,
                                    output_dx=dx,
                                    output_samples=samples,
                                    shift=shift,
                                    method=method
        )

        return Wavefront(dx=dx, cmplx_field=data, wavelength=self.wavelength, space='psf')

    def focus_fixed_sampling_backprop(self, efl, dx, samples, shift=(0, 0), method='mdft'):
        """Backprop a "pupil" to "psf" propagation with fixed output sampling.

        ``self`` carries the gradient at the psf plane; the returned Wavefront
        carries the gradient at the pupil plane.

        Parameters
        ----------
        efl : float
            focusing distance, millimeters
        dx : float
            pupil sampling, millimeters
        samples : int
            number of samples in the pupil plane.  If int, interpreted as square
            else interpreted as (x,y), which is the reverse of numpy's (y, x) row major ordering
        shift : tuple of float
            shift in (X, Y), same units as output_dx
        method : str, {'mdft', 'czt'}
            propagation method (must match the forward call); CZT backprop is
            not implemented and will raise

        Returns
        -------
        Wavefront
            the wavefront at the psf plane

        """
        if self.space != 'psf':
            raise ValueError('can only backpropagate from a psf to pupil plane')

        if isinstance(samples, int):
            samples = (samples, samples)

        data = focus_fixed_sampling_backprop(self.data,
                                             input_dx=dx,
                                             prop_dist=efl,
                                             wavelength=self.wavelength,
                                             output_dx=self.dx,
                                             output_samples=samples,
                                             shift=shift,
                                             method=method
        )

        return Wavefront(dx=dx, cmplx_field=data, wavelength=self.wavelength, space='pupil')

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

        data = unfocus_fixed_sampling(self.data,
                                      input_dx=self.dx,
                                      prop_dist=efl,
                                      wavelength=self.wavelength,
                                      output_dx=dx,
                                      output_samples=samples,
                                      shift=shift,
                                      method=method
        )

        return Wavefront(dx=dx, cmplx_field=data, wavelength=self.wavelength, space='pupil')

    def unfocus_fixed_sampling_backprop(self, efl, dx, samples, shift=(0, 0), method='mdft'):
        """Backprop a "psf" to "pupil" propagation with fixed output sampling.

        Parameters
        ----------
        efl : float
            un-focusing distance, millimeters (matches the forward call)
        dx : float
            focal-plane sample spacing, microns (matches the forward's input_dx)
        samples : int or tuple
            shape used as the forward's ``output_samples`` (pupil shape)
        shift : tuple of float
            shift in (X, Y), same units as dx
        method : str, {'mdft', 'czt'}
            propagation method (must match the forward call); CZT backprop is
            not implemented and will raise

        Returns
        -------
        Wavefront
            gradient at the focal plane

        """
        if self.space != 'pupil':
            raise ValueError('can only backpropagate from a pupil to psf plane')

        if isinstance(samples, int):
            samples = (samples, samples)

        data = unfocus_fixed_sampling_backprop(self.data,
                                               input_dx=dx,
                                               prop_dist=efl,
                                               wavelength=self.wavelength,
                                               output_dx=self.dx,
                                               output_samples=samples,
                                               shift=shift,
                                               method=method
        )

        return Wavefront(dx=dx, cmplx_field=data, wavelength=self.wavelength, space='psf')

    def to_fpm_and_back(self, efl, fpm, fpm_dx, method='mdft', shift=(0, 0), return_more=False):
        """Propagate to a focal plane mask, apply it, and return.

        This routine handles normalization properly for the user.

        To invoke babinet's principle, simply use to_fpm_and_back(fpm=1 - fpm).

        Parameters
        ----------
        efl : float
            focal length for the propagation
        fpm : Wavefront or ndarray
            the focal plane mask
        fpm_dx : float
            sampling increment in the focal plane,  microns;
            do not need to pass if fpm is a Wavefront
        method : str, {'mdft', 'czt'}, optional
            how to propagate the field, matrix DFT or Chirp Z transform
            CZT is usually faster single-threaded and has less memory consumption
            MDFT is usually faster multi-threaded and has more memory consumption
        shift : tuple of float, optional
            shift in the image plane to go to the FPM
            appropriate shift will be computed returning to the pupil
        return_more : bool, optional
            if True, return (new_wavefront, field_at_fpm, field_after_fpm)
            else return new_wavefront

        Returns
        -------
        Wavefront, Wavefront, Wavefront
            new wavefront, [field at fpm, field after fpm]

        """
        pak = to_fpm_and_back(self.data, dx=self.dx, wavelength=self.wavelength,
                              efl=efl, fpm=fpm, fpm_dx=fpm_dx, method=method,
                              shift=shift, return_more=return_more)

        if return_more:
            at_next_pupil, at_fpm, after_fpm = pak
            at_next_pupil = Wavefront(at_next_pupil, self.wavelength, self.dx, self.space)
            at_fpm = Wavefront(at_fpm, self.wavelength, fpm_dx, 'psf')
            after_fpm = Wavefront(after_fpm, self.wavelength, fpm_dx, 'psf')
            return at_next_pupil, at_fpm, after_fpm
        else:
            return Wavefront(pak, self.wavelength, self.dx, self.space)

    def to_fpm_and_back_backprop(self, efl, fpm, fpm_dx=None, method='mdft', shift=(0, 0), return_more=False):
        """Backprop the to_fpm_and_back propagation.

        ``self`` carries the gradient at the next pupil (output of the forward
        to_fpm_and_back); the returned Wavefront carries the gradient at the
        original input pupil.

        Parameters
        ----------
        efl : float
            focal length used in the forward propagation
        fpm : Wavefront or ndarray
            the focal plane mask used in the forward propagation
        fpm_dx : float
            sampling increment in the focal plane, microns;
            do not need to pass if fpm is a Wavefront
        method : str, {'mdft', 'czt'}, optional
            propagation method (must match the forward call)
        shift : tuple of float, optional
            shift used in the forward propagation
        return_more : bool, optional
            if True, return (Eabar, Ebbar, intermediate) as Wavefronts
            else return Eabar

        Returns
        -------
        Wavefront or tuple of Wavefront
            gradient at the input pupil; optionally also the intermediate gradients

        """
        pak = to_fpm_and_back_backprop(self.data, dx=self.dx, efl=efl,
                                       wavelength=self.wavelength,
                                       fpm=fpm, fpm_dx=fpm_dx,
                                       method=method, shift=shift,
                                       return_more=return_more)
        if return_more:
            Eabar, Ebbar, intermediate = pak
            Eabar = Wavefront(Eabar, self.wavelength, self.dx, self.space)
            Ebbar = Wavefront(Ebbar, self.wavelength, fpm_dx, 'psf')
            intermediate = Wavefront(intermediate, self.wavelength, fpm_dx, 'psf')
            return Eabar, Ebbar, intermediate
        else:
            return Wavefront(pak, self.wavelength, self.dx, self.space)

    def babinet(self, efl, lyot, fpm, fpm_dx=None, method='mdft', return_more=False):
        """Propagate through a Lyot-style coronagraph using Babinet's principle.

        This routine handles normalization properly for the user.

        Parameters
        ----------
        efl : float
            focal length for the propagation
        lyot : Wavefront or ndarray
            the Lyot stop; if None, equivalent to ones_like(self.data)
        fpm : Wavefront or ndarray
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

        Notes
        -----
        if the substrate's reflectivity or transmissivity is not unity, and/or
        the mask's density is not infinity, babinet's principle works as follows:

        suppose we're modeling a Lyot focal plane mask;
        rr = radial coordinates of the image plane, in lambda/d units
        mask = rr < 5  # 1 inside FPM, 0 outside (babinet-style)

        now create some scalars for background transmission and mask transmission

        tau = 0.9 # background
        tmask = 0.1 # mask

        mask = tau - tau*mask + rmask*mask

        the mask variable now contains 0.9 outside the spot, and 0.1 inside


        Returns
        -------
        Wavefront, Wavefront, Wavefront, Wavefront
            field after lyot, [field at fpm, field after fpm, field at lyot]

        """
        fpm = 1 - fpm
        if return_more:
            field, field_at_fpm, field_after_fpm = \
                self.to_fpm_and_back(efl=efl, fpm=fpm, fpm_dx=fpm_dx, method=method,
                                     return_more=return_more)
        else:
            field = self.to_fpm_and_back(efl=efl, fpm=fpm, fpm_dx=fpm_dx, method=method,
                                         return_more=return_more)


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

    def babinet_backprop(self, efl, lyot, fpm, fpm_dx=None, method='mdft'):
        """Propagate through a Lyot-style coronagraph using Babinet's principle.

        Parameters
        ----------
        efl : float
            focal length for the propagation
        lyot : Wavefront or ndarray
            the Lyot stop; if None, equivalent to ones_like(self.data)
        fpm : Wavefront or ndarray
            np.conj(1 - fpm)
            one minus the focal plane mask (see Soummer et al 2007)
        fpm_dx : float
            sampling increment in the focal plane,  microns;
            do not need to pass if fpm is a Wavefront
        method : str, {'mdft', 'czt'}
            how to propagate the field, matrix DFT or Chirp Z transform
            CZT is usually faster single-threaded and has less memory consumption
            MDFT is usually faster multi-threaded and has more memory consumption

        Returns
        -------
        Wavefront
            back-propagated gradient

        """
        # babinet's principle is implemented by
        # A = DFT(a)       |
        # C = A*B          |
        # c = iDFT(C)      | Cbar to Abar absorbed in to_fpm_and_back_backprop
        # d = c*L          | cbar = dbar * conj(L)
        # f = a - c        | a contributes +cbar; c contributes -(to_fpm_and_back grad)

        fpm = 1 - fpm

        dbar = self.data
        if lyot is not None:
            if np.iscomplexobj(lyot):
                lyot = np.conj(lyot)

            cbar = dbar * lyot
        else:
            cbar = dbar

        cbarW = Wavefront(cbar, self.wavelength, self.dx, self.space)
        abar = cbarW.to_fpm_and_back_backprop(efl=efl, fpm=fpm, fpm_dx=fpm_dx, method=method)

        abar.data = cbar - abar.data
        return abar
