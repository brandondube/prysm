"""Self-Referenced Interferometer."""

# Cousin of the point diffraction interferometer
import warnings
from prysm.mathops import np
from prysm.propagation import (
    Wavefront,
    focus_dft,
    unfocus_dft,
    prepare_executor,
)
from prysm.coordinates import make_xy_grid, cart_to_polar
from prysm.geometry import circle

from .pdi import evaluate_test_ref_arm_matching

from scipy.special import j0, j1, k0, k1

WF = Wavefront


def smf_mode_field(V, a, b, r):
    U = V * np.sqrt(1-b)
    W = V * np.sqrt(b)
    # inside core
    rnorm = r*(1/a)  # faster to divide on scalar, mul on vector
    rinterior = rnorm < 1
    num = j0(U*rnorm[rinterior])
    den = j1(U)
    out = np.empty_like(r)
    out[rinterior] = num*(1/den)

    rexterior = ~rinterior
    num = k0(W*rnorm[rexterior])
    den = k1(W)
    out[rexterior] = num*(1/den)
    return out


def overlap_integral(E1, E2, sumI1, sumI2):
    num = (E1.conj()*E2).sum()
    num = abs(num) ** 2
    den = sumI1 * sumI2
    return num/den


def to_photonic_fiber_and_back(self, efl, Efib, fib_dx, Ifibsum, executor=None, shift=(0, 0), phase_shift=0, return_more=False):
    """Propagate to a focal plane mask, apply it, and return.

    This routine handles normalization properly for the user.

    To invoke babinet's principle, simply use to_fpm_and_back(fpm=1 - fpm).

    Parameters
    ----------
    efl : float
        focal length for the propagation
    fpm : Wavefront or ndarray
        the focal plane mask
    fib_dx : float
        sampling increment in the focal plane,  microns;
        do not need to pass if fpm is a Wavefront
    executor : MDFT, optional
        precomputed bidirectional transform operator. If None, defaults are
        built per call.
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
    fib_samples = Efib.shape
    input_samples = self.data.shape

    if executor is None:
        executor = prepare_executor(
            pupil_dx=self.dx, pupil_samples=input_samples,
            focal_dx=fib_dx, focal_samples=fib_samples,
            wavelength=self.wavelength, efl=efl, focal_shift=shift,
        )

    field_at_fpm = focus_dft(self.data, executor)

    at_fpm = self.focus_dft(executor)
    I_at_fpm = at_fpm.intensity
    input_power = I_at_fpm.data.sum()
    coupling_loss = overlap_integral(at_fpm.data, Efib, input_power, Ifibsum)
    # propagation of power
    c = (input_power*coupling_loss) ** 0.5
    Eout = Efib * c
    # phase shift the reference beam
    if phase_shift != 0:
        phase_shift = np.exp(1j*phase_shift)
        Eout = Eout * phase_shift

    field_at_next_pupil = unfocus_dft(Eout, executor)

    if input_samples[0] != input_samples[1]:
        warnings.warn(f'Forward propagation had input shape {input_samples} which was not uniform between axes, scaling is off')
    if fib_samples[0] != fib_samples[1]:
        warnings.warn(f'Forward propagation had fpm shape {fib_samples} which was not uniform between axes, scaling is off')

    out = Wavefront(field_at_next_pupil, self.wavelength, self.dx, self.space)
    if return_more:
        if not isinstance(field_at_fpm, Wavefront):
            field_at_fpm = Wavefront(field_at_fpm, out.wavelength, fib_dx, 'psf')
        return out, field_at_fpm, Wavefront(Eout, self.wavelength, fib_dx, 'psf'), coupling_loss

    return out


class SelfReferencedInterferometer:
    """Self-Referenced Interferometer."""
    def __init__(self, x, y, efl, epd, wavelength,
                 fiber_V=2.3, fiber_b=0.5, fiber_a=1.95/2,
                 fiber_samples=256,
                 beamsplitter_RT=(0.8, 0.2)):
        """Create a new Self-Referenced Interferometer.

        Parameters
        ----------
        x : ndarray
            x coordinates for arrays that will be passed to forward_model
            not normalized
        y : ndarray
            y coordinates for arrays that will be passed to forward_model
            not normalized
        efl : float
            focal length in the focusing space behind the grating
        epd : float
            entrance pupil diameter, mm
        wavelength : float
            wavelength of light, um
        pinhole_diameter : float
            diameter of the pinhole placed at the m=0 focus
        pinhole_samples : int
            number of samples across the pinhole placed at the m=0 focus
        beamsplitter_RT : tuple of float
            [R]eference, [T]est arm beamsplitter transmisivities
            (big R / big T, power).  Needed to balance ref/test beam power.

        """
        self.x = x
        self.y = y
        self.dx = x[0, 1] - x[0, 0]
        self.efl = efl
        self.epd = epd
        self.wavelength = wavelength
        self.fno = efl/epd
        self.flambd = self.fno * self.wavelength

        # a is a radius
        # assume mode field < 1.25 x a
        fiber_fov_radius = 10 * 1.25 * fiber_a
        self.dx_pinhole = (2*fiber_fov_radius) / fiber_samples

        xfib, yfib = make_xy_grid(fiber_samples, diameter=2*fiber_fov_radius)
        rfib, tfib = cart_to_polar(xfib, yfib)
        self.Efib = smf_mode_field(fiber_V, fiber_a, fiber_b, rfib)
        self.Efib = self.Efib / (self.Efib**2).sum()**0.5  # unitary fiber mode
        self.Ifib = abs(self.Efib)**2
        self.Ifibsum = self.Ifib.sum()
        self.dxfib = xfib[0, 1] - xfib[0, 0]

        # big R, big T -> little r, little t
        # (power -> amplitude)
        self.ref_r = beamsplitter_RT[0]**0.5
        self.test_t = beamsplitter_RT[1]**0.5

    def forward_model(self, wave_in, phase_shift=0, debug=False):
        """Perform a forward model, returning the intensity at the detector plane.

        Parameters
        ----------
        wave_in : ndarray
            complex wavefunction present at the input to the interferometer
        phase_shift : float
            phase shift, modulo 2pi, if any
        debug : bool
            if True, returns a dict with the fields in the ref arm, before and
            after interacting with the interferometer components

        Returns
        -------
        prysm._richdata.RichData
            intensity at the camera

        """
        if not isinstance(wave_in, WF):
            wave_in = WF(wave_in, self.wavelength, self.dx)

        test_beam = wave_in
        ref_beam = to_photonic_fiber_and_back(wave_in, self.efl, self.Efib,
                                              self.dxfib, self.Ifibsum, phase_shift=phase_shift)
        ref_beam = ref_beam * self.ref_r
        test_beam = test_beam * self.test_t
        total_field = ref_beam + test_beam
        if debug:
            return {
                'at_camera': {'ref': ref_beam, 'test': test_beam},
            }
        return total_field.intensity
