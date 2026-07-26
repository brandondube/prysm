"""Wavefront class and more fluent propagation interface.
"""
import copy
import numbers
import operator

from ..mathops import np
from .._richdata import RichData

from ._kernels import phase_prefix
from .fft import (
    focus, focus_adjoint, unfocus, unfocus_adjoint,
    pupil_sample_to_psf_sample, psf_sample_to_pupil_sample,
)
from .dft import (
    prepare_executor, prepare_multiresolution,
    focus_dft, focus_dft_adjoint, unfocus_dft, unfocus_dft_adjoint,
)
from .angular_spectrum import angular_spectrum, angular_spectrum_adjoint
from .coronagraph import (
    to_fpm_and_back, to_fpm_and_back_adjoint,
    to_fpm_and_back_multiresolution, to_fpm_and_back_multiresolution_adjoint,
    babinet, babinet_adjoint,
)
from ..fttools import pad2d, crop_center


def _field_data(field):
    """Return ndarray data from a Wavefront-like field (pass through otherwise)."""
    if isinstance(field, Wavefront):
        return field.data
    return field


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
            P = amplitude * np.exp(phase_prefix(wavelength) * phase)
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
        E = np.exp(phase_prefix(wavelength) * phase)
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
        w = wavelength / 1e3  # um -> mm, matches thin_lens_adjoint
        term1 = -1j * 2 * np.pi / w

        rsq = x * x + y * y
        term2 = rsq / (2 * f)

        cmplx_screen = np.exp(term1 * term2)
        dx = float(x[0, 1] - x[0, 0])  # float conversion for CuPy support
        return cls(cmplx_field=cmplx_screen, wavelength=wavelength, dx=dx, space='pupil')

    @property
    def intensity(self):
        """Intensity, abs(w)^2."""
        data = self.data
        data = (data.real * data.real) + (data.imag * data.imag)
        return RichData(data, self.dx, self.wavelength)

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

    def from_amp_and_phase_adjoint_phase(self, wf_bar):
        """Adjoint of from_amp_and_phase with respect to phase.

        Parameters
        ----------
        wf_bar : Wavefront
            the gradient propagated to wf

        Returns
        -------
        ndarray
            gradient with respect to the phase of wf_in

        """
        k = phase_prefix(self.wavelength)
        # imag(gbar*g)
        return k * np.imag(wf_bar.data * np.conj(self.data))

    def from_amp_and_phase_adjoint_amp(self, wf_bar, phase=None):
        """Adjoint of from_amp_and_phase with respect to amplitude.

        The forward field is P = amplitude * S, with S the unit-modulus phasor
        exp(prefix * phase).  dP/d(amplitude) = S, so the amplitude gradient is
        real(conj(wf_bar) * S).

        Parameters
        ----------
        wf_bar : Wavefront
            the gradient propagated to wf
        phase : ndarray, optional
            the phase used in the forward from_amp_and_phase, units of nm.
            If given, S is reconstructed exactly.  If None, S is recovered from
            self.data as P / abs(P), which is exact where the amplitude is
            nonzero and yields zero gradient where the amplitude vanishes.

        Returns
        -------
        ndarray
            gradient with respect to the amplitude of wf_in

        """
        if phase is not None:
            S = np.exp(phase_prefix(self.wavelength) * phase)
            # real(conj(gbar) * S)
            return np.real(wf_bar.data * np.conj(S))
        # phase not given: recover the unit phasor from the stored field as
        # P / abs(P); real(gbar * conj(P/abs(P))) = real(gbar * conj(P)) / abs(P),
        # taken as zero where the amplitude vanishes
        absP = np.abs(self.data)
        nonzero = absP > 0
        grad = np.real(wf_bar.data * np.conj(self.data))
        return np.where(nonzero, grad / np.where(nonzero, absP, 1), 0)

    def phase_screen_adjoint_phase(self, wf_bar):
        """Adjoint of phase_screen with respect to phase.

        phase_screen is from_amp_and_phase with unit amplitude, so the gradient
        has the same form as from_amp_and_phase_adjoint_phase.

        Parameters
        ----------
        wf_bar : Wavefront
            the gradient propagated to wf

        Returns
        -------
        ndarray
            gradient with respect to the phase of the screen

        """
        return self.from_amp_and_phase_adjoint_phase(wf_bar)

    @classmethod
    def thin_lens_adjoint(cls, f, wavelength, x, y, wf_bar):
        """Adjoint of thin_lens with respect to the focal length f.

        thin_lens maps the scalar focal length f to a quadratic phase screen.
        This is the transpose of that map: given the gradient flowing back to
        the screen, it returns the (scalar) gradient with respect to f, so the
        focal length can be treated as a differentiable design parameter.

        Parameters
        ----------
        f : float
            focal length of the lens used in the forward thin_lens, millimeters
        wavelength : float
            wavelength of light, microns
        x : ndarray
            x coordinates that define the space of the lens, mm
        y : ndarray
            y coordinates that define the space of the beam, mm
        wf_bar : Wavefront or ndarray
            gradient with respect to the lens screen produced by thin_lens

        Returns
        -------
        scalar
            gradient of the loss with respect to the focal length f

        """
        L_bar = _field_data(wf_bar)
        L = cls.thin_lens(f, wavelength, x, y).data  # the forward lens screen
        w = wavelength / 1e3  # um -> mm, matches thin_lens
        rsq = x * x + y * y

        # dL/df = i (pi rsq / (w f^2)) L, and
        # f_bar = sum real(conj(L_bar) dL/df) = coeff sum(rsq imag(L_bar conj(L)))
        coeff = np.pi / (w * f * f)
        return coeff * np.sum(rsq * np.imag(L_bar * np.conj(L)))

    def intensity_adjoint(self, intensity_bar):
        """Adjoint of intensity.

        Parameters
        ----------
        intensity_bar : Wavefront
            the gradient propagated to the intensity step

        Returns
        -------
        ndarray
            gradient with respect to the complex wavefront before intensity
            was calculated

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

    def __numerical_operation__(self, other, op, reverse=False):
        """Apply an operation to this wavefront with another piece of data."""
        func = getattr(operator, op)
        if isinstance(other, Wavefront):
            criteria = [
                abs(self.dx - other.dx) / self.dx * 100 < 0.1,  # must match to 0.1% (generous, for fp32 compat)
                self.data.shape == other.data.shape,
                self.wavelength == other.wavelength,
                self.space == other.space,
            ]
            if not all(criteria):
                raise ValueError('all physicality criteria not met: sample spacing, shape, wavelength, or space different.')

            data = func(other.data, self.data) if reverse else func(self.data, other.data)
        elif type(other) == type(self.data) or isinstance(other, numbers.Number):  # NOQA
            data = func(other, self.data) if reverse else func(self.data, other)
        else:
            raise TypeError(f'unsupported operand type(s) for {op}: \'Wavefront\' and {type(other)}')

        return Wavefront(dx=self.dx, wavelength=self.wavelength, cmplx_field=data, space=self.space)

    def __mul__(self, other):
        """Multiply this wavefront by something compatible."""
        return self.__numerical_operation__(other, 'mul')

    def __rmul__(self, other):
        """Multiply something compatible by this wavefront."""
        return self.__numerical_operation__(other, 'mul', reverse=True)

    def __truediv__(self, other):
        """Divide this wavefront by something compatible."""
        return self.__numerical_operation__(other, 'truediv')

    def __rtruediv__(self, other):
        """Divide something compatible by this wavefront."""
        return self.__numerical_operation__(other, 'truediv', reverse=True)

    def __add__(self, other):
        """Perform elementwise addition with other, e1+e2."""
        return self.__numerical_operation__(other, 'add')

    def __radd__(self, other):
        """Add something compatible to this wavefront."""
        return self.__numerical_operation__(other, 'add', reverse=True)

    def __sub__(self, other):
        """Perform elementwise subtraction with other, e1-e2."""
        return self.__numerical_operation__(other, 'sub')

    def __rsub__(self, other):
        """Subtract this wavefront from something compatible."""
        return self.__numerical_operation__(other, 'sub', reverse=True)

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

    def free_space_adjoint(self, dz=np.nan, Q=1, tf=None):
        """Apply the adjoint of free_space.

        self carries the gradient at the output plane; the returned Wavefront
        carries the gradient at the input plane.

        Parameters
        ----------
        dz : float
            inter-plane distance used for the forward propagation, millimeters
        Q : float
            padding factor used for the forward propagation
        tf : ndarray
            if not None, clobbers all other arguments
            transfer function used for the forward propagation

        Returns
        -------
        Wavefront
            gradient at the input plane

        """
        if np.isnan(dz) and tf is None:
            raise ValueError('dz must be provided if tf is None')
        out = angular_spectrum_adjoint(self.data,
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

    def focus_adjoint(self, efl, Q=2):
        """Apply the adjoint of focus.

        self carries the gradient at the PSF plane; the returned Wavefront
        carries the gradient at the pupil plane.

        Parameters
        ----------
        efl : float
            focusing distance used for the forward propagation, millimeters
        Q : float
            padding factor used for the forward propagation

        Returns
        -------
        Wavefront
            gradient at the pupil plane

        """
        if self.space != 'psf':
            raise ValueError('can only apply adjoint from a psf to pupil plane')

        samples = self.data.shape[1]
        data = focus_adjoint(self.data, Q=Q)
        dx = psf_sample_to_pupil_sample(self.dx, samples, self.wavelength, efl)

        return Wavefront(data, self.wavelength, dx, space='pupil')

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

    def unfocus_adjoint(self, efl, Q=2):
        """Apply the adjoint of unfocus.

        self carries the gradient at the pupil plane; the returned Wavefront
        carries the gradient at the PSF plane.

        Parameters
        ----------
        efl : float
            un-focusing distance used for the forward propagation, millimeters
        Q : float
            padding factor used for the forward propagation

        Returns
        -------
        Wavefront
            gradient at the PSF plane

        """
        if self.space != 'pupil':
            raise ValueError('can only apply adjoint from a pupil to psf plane')

        samples = self.data.shape[1]
        data = unfocus_adjoint(self.data, Q=Q)
        dx = pupil_sample_to_psf_sample(self.dx, samples, self.wavelength, efl)

        return Wavefront(data, self.wavelength, dx, space='psf')

    def prepare_executor(self, efl, dx, samples, shift=(0, 0), kind='mdft'):
        """Build a reusable MDFT, CZT, or FFTDFT focus executor.

        Wraps prepare_executor (which itself wraps
        coordinates_for_focus and the executor constructor). The
        interpretation of (dx, samples) depends on the wavefront's space:

        - If self.space == 'pupil': self.dx and self.data.shape are
          the pupil-side parameters; dx (microns) and samples describe
          the focal plane.
        - If self.space == 'psf': self.dx and self.data.shape are
          the focal-side parameters; dx (mm) and samples describe the
          pupil plane.

        The returned executor is in the focus orientation and works for either
        direction — pass it to focus_dft or unfocus_dft (which uses
        executor.adjoint for MDFT).

        Parameters
        ----------
        efl : float
            focal length, mm
        dx : float
            sample spacing of the *other* plane (focal: microns; pupil: mm)
        samples : int or (int, int)
            sample count of the other plane
        shift : (float, float)
            (x, y) translation of the focal grid, microns
        kind : {'mdft', 'czt', 'fftdft'}, optional
            executor type to build. ``fftdft`` requires FFT-compatible pupil
            and focal sample spacings. Default 'mdft'.

        Returns
        -------
        MDFT, CZT, or FFTDFT

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

    def prepare_multiresolution(self, efl, focal_dx, focal_samples, num_levels,
                                scaling=4.0, fine_samples=None, window=(0.2, 0.7),
                                kind='mdft'):
        """Build a MultiResolutionExecutor for this wavefront.

        Wraps prepare_multiresolution with this wavefront's pupil-side
        geometry; unit_cell_focal_grid computes a (focal_dx, focal_samples)
        pair that spans the full field of view.

        Parameters
        ----------
        efl : float
            focal length, mm
        focal_dx : float
            coarsest-level focal sample spacing, microns
        focal_samples : int or (int, int)
            coarsest-level focal sample count
        num_levels : int
            number of resolution levels
        scaling, fine_samples, window, kind
            see prepare_multiresolution.

        Returns
        -------
        MultiResolutionExecutor

        """
        if self.space != 'pupil':
            raise ValueError('multiresolution propagation begins at a pupil plane')
        return prepare_multiresolution(
            pupil_dx=self.dx, pupil_samples=self.data.shape,
            focal_dx=focal_dx, focal_samples=focal_samples,
            wavelength=self.wavelength, efl=efl, num_levels=num_levels,
            scaling=scaling, fine_samples=fine_samples, window=window, kind=kind,
        )

    def focus_dft(self, executor):
        """Pupil → PSF propagation via a precomputed executor.

        Parameters
        ----------
        executor : MDFT, CZT, or FFTDFT
            (semi-)arbitrary sampling fourier transform executor

        Returns
        -------
        Wavefront
            the wavefront at the psf plane (dx from executor.focal_dx)

        """
        if self.space != 'pupil':
            raise ValueError('can only propagate from a pupil to psf plane')
        data = focus_dft(self.data, executor)
        return Wavefront(dx=executor.focal_dx, cmplx_field=data, wavelength=self.wavelength, space='psf')

    def focus_dft_adjoint(self, executor):
        """Apply the adjoint of focus_dft.

        self carries the gradient at the psf plane; the returned Wavefront
        carries the gradient at the pupil plane.

        Parameters
        ----------
        executor : MDFT, CZT, or FFTDFT
            (semi-)arbitrary sampling fourier transform executor

        Returns
        -------
        Wavefront
            gradient at the pupil plane (dx from executor.pupil_dx)

        """
        if self.space != 'psf':
            raise ValueError('can only apply adjoint from a psf to pupil plane')
        data = focus_dft_adjoint(self.data, executor)
        return Wavefront(dx=executor.pupil_dx, cmplx_field=data, wavelength=self.wavelength, space='pupil')

    def unfocus_dft(self, executor):
        """PSF → pupil propagation via a precomputed executor.

        Parameters
        ----------
        executor : MDFT, CZT, or FFTDFT
            (semi-)arbitrary sampling fourier transform executor

        Returns
        -------
        Wavefront
            wavefront at the pupil plane (dx from executor.pupil_dx)

        """
        if self.space != 'psf':
            raise ValueError('can only propagate from a psf to pupil plane')
        data = unfocus_dft(self.data, executor)
        return Wavefront(dx=executor.pupil_dx, cmplx_field=data, wavelength=self.wavelength, space='pupil')

    def unfocus_dft_adjoint(self, executor):
        """Apply the adjoint of unfocus_dft.

        Parameters
        ----------
        executor : MDFT, CZT, or FFTDFT
            (semi-)arbitrary sampling fourier transform executor

        Returns
        -------
        Wavefront
            gradient at the focal plane (dx from executor.focal_dx)

        """
        if self.space != 'pupil':
            raise ValueError('can only apply adjoint from a pupil to psf plane')
        data = unfocus_dft_adjoint(self.data, executor)
        return Wavefront(dx=executor.focal_dx, cmplx_field=data, wavelength=self.wavelength, space='psf')

    def to_fpm_and_back(self, fpm, executor, return_more=False):
        """Propagate to a focal plane mask, apply it, and return.

        Parameters
        ----------
        fpm : Wavefront or ndarray
            the focal plane mask
        executor : MDFT, CZT, or FFTDFT
            (semi-)arbitrary sampling fourier transform executor
        return_more : bool, optional
            if True, return (new_wavefront, field_at_fpm, field_after_fpm)
            else return new_wavefront

        Returns
        -------
        Wavefront, Wavefront, Wavefront
            new wavefront, [field at fpm, field after fpm]

        """
        fpm = _field_data(fpm)
        pak = to_fpm_and_back(self.data, fpm=fpm, executor=executor, return_more=return_more)

        if return_more:
            at_next_pupil, at_fpm, after_fpm = pak
            at_next_pupil = Wavefront(at_next_pupil, self.wavelength, self.dx, self.space)
            at_fpm = Wavefront(at_fpm, self.wavelength, executor.focal_dx, 'psf')
            after_fpm = Wavefront(after_fpm, self.wavelength, executor.focal_dx, 'psf')
            return at_next_pupil, at_fpm, after_fpm
        else:
            return Wavefront(pak, self.wavelength, self.dx, self.space)

    def to_fpm_and_back_adjoint(self, fpm, executor, return_more=False,
                                return_fpm_grad=False, field_at_fpm=None):
        """Apply the adjoint of the to_fpm_and_back propagation.

        self carries the gradient at the next pupil (output of the forward
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
        return_fpm_grad : bool, optional
            if True, also return the gradient with respect to fpm.
            Requires field_at_fpm from the matching forward propagation.
        field_at_fpm : Wavefront or ndarray, optional
            focal-plane field before the FPM from the forward propagation

        Returns
        -------
        Wavefront or tuple of Wavefront
            gradient at the input pupil; optionally also the intermediate
            gradients and/or the gradient with respect to fpm

        """
        fpm = _field_data(fpm)
        field_at_fpm = _field_data(field_at_fpm)
        pak = to_fpm_and_back_adjoint(self.data, fpm=fpm, executor=executor,
                                      return_more=return_more,
                                      return_fpm_grad=return_fpm_grad,
                                      field_at_fpm=field_at_fpm)
        if return_more:
            if return_fpm_grad:
                Eabar, Ebbar, intermediate, fpm_bar = pak
            else:
                Eabar, Ebbar, intermediate = pak
            Eabar = Wavefront(Eabar, self.wavelength, self.dx, self.space)
            Ebbar = Wavefront(Ebbar, self.wavelength, executor.focal_dx, 'psf')
            intermediate = Wavefront(intermediate, self.wavelength, executor.focal_dx, 'psf')
            if return_fpm_grad:
                fpm_bar = Wavefront(fpm_bar, self.wavelength, executor.focal_dx, 'psf')
                return Eabar, Ebbar, intermediate, fpm_bar
            return Eabar, Ebbar, intermediate
        elif return_fpm_grad:
            Eabar, fpm_bar = pak
            Eabar = Wavefront(Eabar, self.wavelength, self.dx, self.space)
            fpm_bar = Wavefront(fpm_bar, self.wavelength, executor.focal_dx, 'psf')
            return Eabar, fpm_bar
        else:
            return Wavefront(pak, self.wavelength, self.dx, self.space)

    def to_fpm_and_back_multiresolution(self, fpm, executor, return_more=False):
        """Propagate to a focal plane mask and back at multiple resolutions.

        Parameters
        ----------
        fpm : callable
            fpm(xf, yf) -> ndarray, the focal plane mask evaluated on
            focal-plane coordinate grids (microns). See vortex_phase_mask.
        executor : MultiResolutionExecutor
            stack of executors and hand-off windows from prepare_multiresolution
        return_more : bool, optional
            if True, return (new_wavefront, fields_at_fpm, fields_after_fpm)
            with per-level lists of Wavefronts, else return new_wavefront

        Returns
        -------
        Wavefront, [list of Wavefront, list of Wavefront]
            next pupil; optionally also per-level fields at and after the fpm

        """
        if self.space != 'pupil':
            raise ValueError('can only propagate from a pupil to psf plane')
        pak = to_fpm_and_back_multiresolution(self.data, fpm, executor,
                                              return_more=return_more)
        if not return_more:
            return Wavefront(pak, self.wavelength, self.dx, self.space)

        out, at_fpm, after_fpm = pak
        out = Wavefront(out, self.wavelength, self.dx, self.space)
        at_fpm = [Wavefront(f, self.wavelength, ex.focal_dx, 'psf')
                  for f, ex in zip(at_fpm, executor.executors)]
        after_fpm = [Wavefront(f, self.wavelength, ex.focal_dx, 'psf')
                     for f, ex in zip(after_fpm, executor.executors)]
        return out, at_fpm, after_fpm

    def to_fpm_and_back_multiresolution_adjoint(self, fpm, executor, return_more=False,
                                                return_fpm_grad=False, field_at_fpm=None):
        """Apply the adjoint of to_fpm_and_back_multiresolution.

        self carries the gradient at the next pupil; the returned Wavefront
        carries the gradient at the original input pupil.

        Parameters
        ----------
        fpm : callable
            the focal plane mask callable used in the forward propagation
        executor : MultiResolutionExecutor
            stack of executors and hand-off windows from prepare_multiresolution
        return_more : bool, optional
            if True, return (Eabar, Ebbars, intermediates) with per-level
            lists of Wavefronts, else return Eabar
        return_fpm_grad : bool, optional
            if True, also return the per-level gradients with respect to the
            mask samples. Requires field_at_fpm from the matching forward
            propagation.
        field_at_fpm : list of Wavefront or ndarray, optional
            per-level focal-plane fields before the fpm from the forward
            propagation with return_more=True

        Returns
        -------
        Wavefront or tuple
            gradient at the input pupil; optionally also the per-level
            intermediate gradients and/or mask gradients

        """
        if field_at_fpm is not None:
            field_at_fpm = [_field_data(f) for f in field_at_fpm]
        pak = to_fpm_and_back_multiresolution_adjoint(
            self.data, fpm, executor, return_more=return_more,
            return_fpm_grad=return_fpm_grad, field_at_fpm=field_at_fpm,
        )

        def _psf_wrap(fields):
            return [Wavefront(f, self.wavelength, ex.focal_dx, 'psf')
                    for f, ex in zip(fields, executor.executors)]

        if return_more:
            if return_fpm_grad:
                Eabar, Ebbars, intermediates, fpm_bars = pak
            else:
                Eabar, Ebbars, intermediates = pak
            Eabar = Wavefront(Eabar, self.wavelength, self.dx, self.space)
            Ebbars = _psf_wrap(Ebbars)
            intermediates = _psf_wrap(intermediates)
            if return_fpm_grad:
                return Eabar, Ebbars, intermediates, _psf_wrap(fpm_bars)
            return Eabar, Ebbars, intermediates
        elif return_fpm_grad:
            Eabar, fpm_bars = pak
            Eabar = Wavefront(Eabar, self.wavelength, self.dx, self.space)
            return Eabar, _psf_wrap(fpm_bars)
        return Wavefront(pak, self.wavelength, self.dx, self.space)

    def babinet(self, lyot, fpm, executor, return_more=False):
        """Propagate through a Lyot-style coronagraph using Babinet's principle.

        Parameters
        ----------
        lyot : Wavefront or ndarray
            the Lyot stop; if None, equivalent to ones_like(self.data)
        fpm : Wavefront or ndarray
            focal plane mask transmission; 0 inside an opaque occulter and 1
            far from the axis, so the Babinet complement 1 - fpm formed
            internally is compactly supported (see Soummer et al 2007)
        executor : MDFT
            bidirectional transform operator.
        return_more : bool
            if True, return each plane in the propagation
            else return new_wavefront

        Notes
        -----
        fpm must approach 1 at the edge of the focal window; a background
        folded into it directly (e.g. 0.9 outside the spot) breaks the compact
        support of 1 - fpm and silently corrupts the result.  For a mask of
        transmission tmask on a substrate of transmission tau, factor the
        background out:

        spot = rr < 5  # 1 inside the occulting spot
        fpm = 1 - (1 - tmask/tau)*spot  # tmask/tau inside, 1 outside
        field = wf.babinet(tau*lyot, fpm, executor)

        Returns
        -------
        Wavefront, Wavefront, Wavefront, Wavefront
            field after lyot, [field at fpm, field after fpm, field at lyot]

        """
        fpm = _field_data(fpm)
        lyot = _field_data(lyot)
        pak = babinet(self.data, lyot=lyot, fpm=fpm, executor=executor, return_more=return_more)

        if return_more:
            after_lyot, at_fpm, after_fpm, at_lyot = pak
            after_lyot = Wavefront(after_lyot, self.wavelength, self.dx, self.space)
            at_fpm = Wavefront(at_fpm, self.wavelength, executor.focal_dx, 'psf')
            after_fpm = Wavefront(after_fpm, self.wavelength, executor.focal_dx, 'psf')
            at_lyot = Wavefront(at_lyot, self.wavelength, self.dx, self.space)
            return after_lyot, at_fpm, after_fpm, at_lyot
        return Wavefront(pak, self.wavelength, self.dx, self.space)

    def babinet_adjoint(self, lyot, fpm, executor, field_at_fpm=None,
                        field_at_lyot=None, return_fpm_grad=False,
                        return_lyot_grad=False):
        """Apply the adjoint of babinet.

        Parameters
        ----------
        lyot : Wavefront or ndarray
            the Lyot stop; if None, equivalent to ones_like(self.data)
        fpm : Wavefront or ndarray
            the focal plane mask used in the forward propagation
        executor : MDFT, CZT, or FFTDFT
            (semi-)arbitrary sampling fourier transform executor
        field_at_fpm : Wavefront or ndarray, optional
            focal-plane field before the FPM from the matching forward call.
            Required when return_fpm_grad is True.
        field_at_lyot : Wavefront or ndarray, optional
            pupil-plane field before the Lyot stop from the matching forward
            call. Required when return_lyot_grad is True.
        return_fpm_grad : bool, optional
            if True, also return the gradient with respect to the original
            fpm argument passed to babinet.
        return_lyot_grad : bool, optional
            if True, also return the gradient with respect to lyot.

        Returns
        -------
        Wavefront or tuple of Wavefront
            adjoint-propagated gradient; optionally followed by FPM and/or Lyot
            gradients in the order requested by the keyword names

        """
        fpm = _field_data(fpm)
        lyot = _field_data(lyot)
        field_at_fpm = _field_data(field_at_fpm)
        field_at_lyot = _field_data(field_at_lyot)
        pak = babinet_adjoint(self.data, lyot=lyot, fpm=fpm, executor=executor,
                              field_at_fpm=field_at_fpm, field_at_lyot=field_at_lyot,
                              return_fpm_grad=return_fpm_grad,
                              return_lyot_grad=return_lyot_grad)
        if not (return_fpm_grad or return_lyot_grad):
            return Wavefront(pak, self.wavelength, self.dx, self.space)

        pak = list(pak)
        out = [Wavefront(pak[0], self.wavelength, self.dx, self.space)]
        idx = 1
        if return_fpm_grad:
            out.append(Wavefront(pak[idx], self.wavelength, executor.focal_dx, 'psf'))
            idx += 1
        if return_lyot_grad:
            out.append(Wavefront(pak[idx], self.wavelength, self.dx, self.space))
        return tuple(out)
