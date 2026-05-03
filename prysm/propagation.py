"""Numerical optical propagation."""
import copy
import numbers
import operator
from collections.abc import Iterable

from .conf import config
from .mathops import np, fft
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
    ``(fx, fy)`` that pair with them, where ``fx = x_focal / (λ · efl)``.

    For end users, prefer :func:`prepare_executor`, which wraps this and bakes
    the optical normalization into the executor. If you do build the executor
    by hand, multiply its result by ``pupil_dx * focal_dx / (wavelength * efl)``.

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


def prepare_executor(pupil_dx, pupil_samples, focal_dx, focal_samples,
                     wavelength, efl, focal_shift=(0, 0), kind='mdft'):
    """Build a reusable MDFT or CZT operator for a pupil ↔ focal propagation.

    Wraps :func:`coordinates_for_focus` and the executor constructor in one
    call. The optical normalization scalar
    ``pupil_dx * focal_dx / (wavelength * efl)`` is baked into the executor's
    ``norm``, so applying the executor produces a unitary-equivalent
    propagated field. The returned operator is in the focus orientation:

    - Focus:    ``executor(pupil_data)`` produces focal data
    - Unfocus:  ``executor.adjoint(focal_data)`` produces pupil data
      (MDFT only — CZT has no adjoint and would need a separate operator
      built in the focal → pupil orientation).

    The pupil and focal sample spacings are also stashed on the returned
    operator as ``executor.pupil_dx`` and ``executor.focal_dx`` for callers
    that need them (e.g. to label an output ``Wavefront``).

    Parameters
    ----------
    pupil_dx, pupil_samples, focal_dx, focal_samples, wavelength, efl, focal_shift
        See :func:`coordinates_for_focus`.
    kind : {'mdft', 'czt'}, optional
        Executor type to build. Default ``'mdft'``.

    Returns
    -------
    MDFT or CZT
        operator suitable for passing to ``focus_dft``, ``unfocus_dft``, etc.

    """
    x, y, fx, fy = coordinates_for_focus(
        pupil_dx, pupil_samples, focal_dx, focal_samples,
        wavelength, efl, focal_shift,
    )
    norm = (pupil_dx * focal_dx) / (wavelength * efl)
    if kind == 'mdft':
        op = MDFT(x, y, fx, fy, sign=-1, norm=norm)
    elif kind == 'czt':
        op = CZT(x, y, fx, fy, sign=-1, norm=norm)
    else:
        raise ValueError(f"kind must be 'mdft' or 'czt', got {kind!r}")
    op.pupil_dx = pupil_dx
    op.focal_dx = focal_dx
    return op


def focus_dft(wavefunction, executor):
    """Propagate a pupil field to the PSF plane via a precomputed executor.

    Parameters
    ----------
    wavefunction : ndarray
        the pupil-plane field; shape must match what the executor was built for.
    executor : MDFT or CZT
        a focus-orientation operator (e.g. from :func:`prepare_executor`).
        Optical normalization is expected to be baked into ``executor.norm``.

    Returns
    -------
    ndarray
        focal-plane field

    """
    return executor(wavefunction)


def focus_dft_backprop(wavefunction, executor):
    """Backpropagate gradient through :func:`focus_dft`.

    Parameters
    ----------
    wavefunction : ndarray
        gradient at the PSF plane
    executor : MDFT
        the same operator used for the forward call. CZT backprop is not
        implemented and will raise.

    Returns
    -------
    ndarray
        gradient at the pupil plane

    """
    if isinstance(executor, CZT):
        raise NotImplementedError('gradient backpropagation not yet implemented for CZT')
    return executor.adjoint(wavefunction)


def unfocus_dft(wavefunction, executor):
    """Propagate an image-plane field to the pupil via a precomputed executor.

    Parameters
    ----------
    wavefunction : ndarray
        the focal-plane field
    executor : MDFT or CZT
        for MDFT, the focus-orientation operator (same as for ``focus_dft``);
        the inverse is taken via ``executor.adjoint``. For CZT, the operator
        must be built in the focal → pupil orientation since CZT has no
        adjoint.

    Returns
    -------
    ndarray
        pupil-plane field

    """
    if isinstance(executor, MDFT):
        return executor.adjoint(wavefunction)
    elif isinstance(executor, CZT):
        return executor(wavefunction)
    raise TypeError(f"executor must be MDFT or CZT, got {type(executor).__name__}")


def unfocus_dft_backprop(wavefunction, executor):
    """Backpropagate gradient through :func:`unfocus_dft`.

    Parameters
    ----------
    wavefunction : ndarray
        gradient at the pupil plane
    executor : MDFT
        the same operator used for the forward call. CZT backprop is not
        implemented and will raise.

    Returns
    -------
    ndarray
        gradient at the focal plane

    """
    if isinstance(executor, CZT):
        raise NotImplementedError('gradient backpropagation not yet implemented for CZT')
    # adjoint of unfocus (which uses .adjoint) is the forward (which uses __call__)
    return executor(wavefunction)


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


def to_fpm_and_back(wavefunction, fpm, executor, return_more=False):
    """Propagate to a focal plane mask, apply it, and return.

    Composition of :func:`focus_dft`, multiplication by ``fpm``, and
    :func:`unfocus_dft`. The same MDFT executor is used for both legs (its
    adjoint provides the inverse). To invoke Babinet's principle, pass
    ``fpm=1 - fpm``.

    Parameters
    ----------
    wavefunction : ndarray
        complex pupil-plane field to propagate
    fpm : Wavefront or ndarray
        the focal plane mask
    executor : MDFT
        bidirectional transform operator; CZT is not supported here.
    return_more : bool, optional
        if True, return (new_wavefront, field_at_fpm, field_after_fpm)
        else return new_wavefront

    Returns
    -------
    ndarray, [ndarray, ndarray]
        next pupil; optionally also field at fpm and field after fpm

    """
    if isinstance(executor, CZT):
        raise TypeError('to_fpm_and_back requires an MDFT executor (bidirectional); CZT is not supported')
    if isinstance(fpm, Wavefront):
        fpm = fpm.data

    field_at_fpm = focus_dft(wavefunction, executor)
    field_after_fpm = field_at_fpm * fpm
    field_at_next_pupil = unfocus_dft(field_after_fpm, executor)

    if return_more:
        return field_at_next_pupil, field_at_fpm, field_after_fpm
    return field_at_next_pupil


def to_fpm_and_back_backprop(wavefunction, fpm, executor, return_more=False):
    """Backpropagate gradient through :func:`to_fpm_and_back`.

    Parameters
    ----------
    wavefunction : ndarray
        gradient at the next pupil plane (output of the forward call)
    fpm : Wavefront or ndarray
        the focal plane mask used in the forward propagation
    executor : MDFT
        the same MDFT used in the forward call. CZT is not supported.
    return_more : bool, optional
        if True, return (Eabar, Ebbar, intermediate)
        else return Eabar

    Returns
    -------
    ndarray or tuple of ndarray
        gradient at the input pupil; optionally also the intermediate gradients

    """
    if isinstance(executor, CZT):
        raise TypeError('to_fpm_and_back_backprop requires an MDFT executor; CZT is not supported')
    if isinstance(fpm, Wavefront):
        fpm = fpm.data

    # do not take complex conjugate of reals (no-op, but numpy still does it)
    if np.iscomplexobj(fpm):
        fpm = fpm.conj()

    Ebbar = unfocus_dft_backprop(wavefunction, executor)
    intermediate = Ebbar * fpm
    Eabar = focus_dft_backprop(intermediate, executor)
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

    def prepare_executor(self, efl, dx, samples, shift=(0, 0), kind='mdft'):
        """Build a reusable MDFT/CZT focus executor for this wavefront.

        Wraps :func:`prepare_executor` (which itself wraps
        :func:`coordinates_for_focus` and the executor constructor). The
        interpretation of ``(dx, samples)`` depends on the wavefront's space:

        - If ``self.space == 'pupil'``: ``self.dx`` and ``self.data.shape`` are
          the pupil-side parameters; ``dx`` (microns) and ``samples`` describe
          the focal plane.
        - If ``self.space == 'psf'``: ``self.dx`` and ``self.data.shape`` are
          the focal-side parameters; ``dx`` (mm) and ``samples`` describe the
          pupil plane.

        The returned executor is in the focus orientation and works for either
        direction — pass it to ``focus_dft`` or ``unfocus_dft`` (which uses
        ``executor.adjoint`` for MDFT).

        Parameters
        ----------
        efl : float
            focal length, mm
        dx : float
            sample spacing of the *other* plane (focal: microns; pupil: mm)
        samples : int or (int, int)
            sample count of the other plane
        shift : (float, float)
            ``(x, y)`` translation of the focal grid, microns
        kind : {'mdft', 'czt'}, optional
            executor type to build. Default ``'mdft'``.

        Returns
        -------
        MDFT or CZT

        """
        if isinstance(samples, int):
            samples = (samples, samples)
        if self.space == 'pupil':
            return prepare_executor(
                pupil_dx=self.dx, pupil_samples=self.data.shape,
                focal_dx=dx, focal_samples=samples,
                wavelength=self.wavelength, efl=efl, focal_shift=shift, kind=kind,
            )
        elif self.space == 'psf':
            return prepare_executor(
                pupil_dx=dx, pupil_samples=samples,
                focal_dx=self.dx, focal_samples=self.data.shape,
                wavelength=self.wavelength, efl=efl, focal_shift=shift, kind=kind,
            )
        raise ValueError(f"unknown space {self.space!r}")

    def focus_dft(self, executor):
        """Pupil → PSF propagation via a precomputed executor.

        Parameters
        ----------
        executor : MDFT or CZT
            focus-orientation operator (e.g. from :meth:`prepare_executor`).

        Returns
        -------
        Wavefront
            the wavefront at the psf plane (dx from ``executor.focal_dx``)

        """
        if self.space != 'pupil':
            raise ValueError('can only propagate from a pupil to psf plane')
        data = focus_dft(self.data, executor)
        return Wavefront(dx=executor.focal_dx, cmplx_field=data, wavelength=self.wavelength, space='psf')

    def focus_dft_backprop(self, executor):
        """Backpropagate gradient through :meth:`focus_dft`.

        ``self`` carries the gradient at the psf plane; the returned Wavefront
        carries the gradient at the pupil plane.

        Parameters
        ----------
        executor : MDFT
            same operator as the forward call. CZT backprop is not implemented.

        Returns
        -------
        Wavefront
            gradient at the pupil plane (dx from ``executor.pupil_dx``)

        """
        if self.space != 'psf':
            raise ValueError('can only backpropagate from a psf to pupil plane')
        data = focus_dft_backprop(self.data, executor)
        return Wavefront(dx=executor.pupil_dx, cmplx_field=data, wavelength=self.wavelength, space='pupil')

    def unfocus_dft(self, executor):
        """PSF → pupil propagation via a precomputed executor.

        Parameters
        ----------
        executor : MDFT or CZT
            for MDFT, the focus-orientation operator (same one used for
            ``focus_dft``); for CZT, an operator built in the focal → pupil
            orientation.

        Returns
        -------
        Wavefront
            wavefront at the pupil plane (dx from ``executor.pupil_dx``)

        """
        if self.space != 'psf':
            raise ValueError('can only propagate from a psf to pupil plane')
        data = unfocus_dft(self.data, executor)
        return Wavefront(dx=executor.pupil_dx, cmplx_field=data, wavelength=self.wavelength, space='pupil')

    def unfocus_dft_backprop(self, executor):
        """Backpropagate gradient through :meth:`unfocus_dft`.

        Parameters
        ----------
        executor : MDFT
            same operator as the forward call. CZT backprop is not implemented.

        Returns
        -------
        Wavefront
            gradient at the focal plane (dx from ``executor.focal_dx``)

        """
        if self.space != 'pupil':
            raise ValueError('can only backpropagate from a pupil to psf plane')
        data = unfocus_dft_backprop(self.data, executor)
        return Wavefront(dx=executor.focal_dx, cmplx_field=data, wavelength=self.wavelength, space='psf')

    def to_fpm_and_back(self, fpm, executor, return_more=False):
        """Propagate to a focal plane mask, apply it, and return.

        Parameters
        ----------
        fpm : Wavefront or ndarray
            the focal plane mask
        executor : MDFT
            bidirectional transform operator. CZT is not supported.
        return_more : bool, optional
            if True, return (new_wavefront, field_at_fpm, field_after_fpm)
            else return new_wavefront

        Returns
        -------
        Wavefront, Wavefront, Wavefront
            new wavefront, [field at fpm, field after fpm]

        """
        pak = to_fpm_and_back(self.data, fpm=fpm, executor=executor, return_more=return_more)

        if return_more:
            at_next_pupil, at_fpm, after_fpm = pak
            at_next_pupil = Wavefront(at_next_pupil, self.wavelength, self.dx, self.space)
            at_fpm = Wavefront(at_fpm, self.wavelength, executor.focal_dx, 'psf')
            after_fpm = Wavefront(after_fpm, self.wavelength, executor.focal_dx, 'psf')
            return at_next_pupil, at_fpm, after_fpm
        else:
            return Wavefront(pak, self.wavelength, self.dx, self.space)

    def to_fpm_and_back_backprop(self, fpm, executor, return_more=False):
        """Backprop the to_fpm_and_back propagation.

        ``self`` carries the gradient at the next pupil (output of the forward
        to_fpm_and_back); the returned Wavefront carries the gradient at the
        original input pupil.

        Parameters
        ----------
        fpm : Wavefront or ndarray
            the focal plane mask used in the forward propagation
        executor : MDFT
            same operator as the forward call.
        return_more : bool, optional
            if True, return (Eabar, Ebbar, intermediate) as Wavefronts
            else return Eabar

        Returns
        -------
        Wavefront or tuple of Wavefront
            gradient at the input pupil; optionally also the intermediate gradients

        """
        pak = to_fpm_and_back_backprop(self.data, fpm=fpm, executor=executor,
                                       return_more=return_more)
        if return_more:
            Eabar, Ebbar, intermediate = pak
            Eabar = Wavefront(Eabar, self.wavelength, self.dx, self.space)
            Ebbar = Wavefront(Ebbar, self.wavelength, executor.focal_dx, 'psf')
            intermediate = Wavefront(intermediate, self.wavelength, executor.focal_dx, 'psf')
            return Eabar, Ebbar, intermediate
        else:
            return Wavefront(pak, self.wavelength, self.dx, self.space)

    def babinet(self, lyot, fpm, executor, return_more=False):
        """Propagate through a Lyot-style coronagraph using Babinet's principle.

        Parameters
        ----------
        lyot : Wavefront or ndarray
            the Lyot stop; if None, equivalent to ones_like(self.data)
        fpm : Wavefront or ndarray
            1 - fpm
            one minus the focal plane mask (see Soummer et al 2007)
        executor : MDFT
            bidirectional transform operator.
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
        result = self.to_fpm_and_back(fpm=fpm, executor=executor, return_more=return_more)
        if return_more:
            field, field_at_fpm, field_after_fpm = result
        else:
            field = result

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

    def babinet_backprop(self, lyot, fpm, executor):
        """Backpropagate gradient through :meth:`babinet`.

        Parameters
        ----------
        lyot : Wavefront or ndarray
            the Lyot stop; if None, equivalent to ones_like(self.data)
        fpm : Wavefront or ndarray
            np.conj(1 - fpm)
            one minus the focal plane mask (see Soummer et al 2007)
        executor : MDFT
            same operator as the forward call.

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

        abar_data = to_fpm_and_back_backprop(cbar, fpm=fpm, executor=executor)
        return Wavefront(cbar - abar_data, self.wavelength, self.dx, self.space)
