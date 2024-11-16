"""Point Diffraction Interferometry."""

from functools import partial

from prysm.mathops import np
from prysm.coordinates import make_xy_grid
from prysm.propagation import Wavefront as WF
from prysm.geometry import circle


def rectangle_pulse(x, duty=0.5, amplitude=0.5, offset=0.5, period=2*np.pi):
    """Rectangular pulse; generalized square wave.

    This function differs from scipy.signal.square in that the output
    is in [0,1] instead of [-1,1], as well as control over more parameters.

    Parameters
    ----------
    x : ndarray
        spatial domain, the pulse is notionally equivalent to
        np.sign(np.sin(x/period))
    duty : float
        duty cycle of the pulse; a duty of 0.5 == square wave
    amplitude : float
        amplitude of the wave, half of the peak-to-valley
    offset : float
        offset or mean value of the wave
    period : float
        period of the wave

    Returns
    -------
    ndarray
        rectangular pulse

    """
    x = np.asarray(x)
    y = np.zeros_like(x)

    xwrapped = np.mod(x, period)
    mask = xwrapped < (duty*period)
    mask2 = ~mask
    mask3 = abs(xwrapped) < np.finfo(x.dtype).eps

    hi = offset + amplitude
    lo = offset - amplitude
    mid = offset
    y[mask] = hi
    y[mask2] = lo
    y[mask3] = mid
    return y


class PSPDI:
    """Phase Shifting Point Diffraction Interferometer (Medecki Interferometer)."""

    def __init__(self, x, y, efl, epd, wavelength,
                 test_arm_offset=64,
                 test_arm_fov=64,
                 test_arm_samples=256,
                 test_arm_transmissivity=1,
                 pinhole_diameter=0.25,
                 pinhole_samples=128,
                 grating_rulings=64,
                 grating_type='sin_amp',
                 grating_axis='x'):
        """Create a new PS/PDI or Medecki Interferometer.

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
        test_arm_offset : float
            offset of the window for the test arm, in lambda/D
            this number should only ever be different to grating_rulings
            when you wish to model system misalignments
        test_arm_fov : float
            diameter of the circular hole placed at the m=+1 focus, units of
            lambda/D
        test_arm_samples : int
            samples to use across the clear window at the m=1 focus
        test_arm_transmissivity : float
            transmissivity (small r; amplitude) of the test arm, if the
            substrate has an AR coating with reflectance R,
            then this has value 1-sqrt(R) modulo absorbtion in the coating

            The value of this parameter must be optimized to maximize fringe
            visibility for peak performance
        pinhole_diameter : float
            diameter of the pinhole placed at the m=0 focus
        pinhole_samples : int
            number of samples across the pinhole placed at the m=0 focus
        grating_rulings : float
            number of rulings per EPD in the grating
        grating_type : str, {'ronchi'}
            type of grating used in the interferometer
        grating_axis : str, {'x', 'y'}
            which axis the orientation of the grating is in

        """
        grating_type = grating_type.lower()
        grating_axis = grating_axis.lower()
        # inputs
        self.x = x
        self.y = y
        self.dx = x[0, 1] - x[0, 0]
        self.efl = efl
        self.epd = epd
        self.wavelength = wavelength
        self.fno = efl/epd
        self.flambd = self.fno * self.wavelength

        # grating synthesis
        self.grating_rulings = grating_rulings
        self.grating_period = self.epd/grating_rulings
        self.grating_type = grating_type
        self.grating_axis = grating_axis

        if grating_type == 'ronchi':
            f = partial(rectangle_pulse, duty=0.5, amplitude=0.5, offset=0.5, period=self.grating_period)
        elif grating_type == 'sin_amp':
            def f(x):
                prefix = grating_rulings*np.pi/(epd/2)
                sin = np.sin(prefix*x)

                # this does not work the way you expect/want;
                # can't improve efficiency by weakening a sine amp grating
                # square wave with low duty cycle may be best, but brutal
                # to model
                # to make [0,1] => (sin+1)/2
                # want to make [1-a,1], where a = amp
                shifted_sin = (sin+1)/2
                A = 0.1
                squished = shifted_sin * A
                return 1 - squished
        else:
            raise ValueError('unsupported grating type')

        self.grating_func = f

        self.test_arm_offset = test_arm_offset
        self.test_arm_fov = test_arm_fov
        self.test_arm_samples = test_arm_samples
        self.test_arm_eps = test_arm_fov / test_arm_samples
        self.test_arm_fov_compute = (test_arm_fov + self.test_arm_eps) * self.flambd
        self.test_arm_mask_rsq = (test_arm_fov*self.flambd/2)**2
        self.test_arm_transmissivity = test_arm_transmissivity

        if self.grating_axis == 'x':
            self.test_arm_shift = (grating_rulings*self.flambd, 0)
        else:
            self.test_arm_shift = (0, grating_rulings*self.flambd)

        self.pinhole_diameter = pinhole_diameter * self.flambd
        self.pinhole_samples = pinhole_samples
        # -1 is an epsilon to make sure the circle is wholly inside the array
        self.dx_pinhole = pinhole_diameter / (pinhole_samples-1)
        self.pinhole_fov_radius = pinhole_samples/2*self.dx_pinhole

        xph, yph = make_xy_grid(pinhole_samples, diameter=2*self.pinhole_fov_radius)
        rphsq = xph*xph + yph*yph
        self.pinhole = circle((pinhole_diameter/2)**2, rphsq)

        # t = test
        xt, yt = make_xy_grid(test_arm_samples, diameter=self.test_arm_fov_compute)
        self.dx_test_arm = xt[0, 1] - xt[0, 0]

        rtsq = xt*xt + yt*yt
        self.test_mask = circle(self.test_arm_mask_rsq, rtsq)
        del xph, yph, rphsq, xt, yt, rtsq

    def forward_model(self, wave_in, phase_shift=0, debug=False):
        """Perform a forward model, returning the intensity at the detector plane.

        Parameters
        ----------
        wave_in : ndarray
            complex wavefunction present at the input to the interferometer
        phase_shift : float
            phase shift, modulo 2pi, if any
        debug : bool
            if True, returns a dict with the fields in each arm, before and
            after interacting with the interferometer components

        Returns
        -------
        prysm._richdata.RichData
            intensity at the camera

        """
        # reference wave
        if phase_shift != 0:
            # user gives value in [0,2pi] which maps 2pi => period
            phase_shift = phase_shift / (2*np.pi) * self.grating_period
            x = self.x + phase_shift
        else:
            x = self.x
        grating = self.grating_func(x)
        i = wave_in * grating
        if not isinstance(i, WF):
            i = WF(i, self.wavelength, self.dx)

        efl = self.efl
        if debug:
            ref_beam, ref_at_fpm, ref_after_fpm = \
                i.to_fpm_and_back(efl, self.pinhole, self.dx_pinhole, return_more=True)
            test_beam, test_at_fpm, test_after_fpm = \
                i.to_fpm_and_back(efl, self.test_mask, self.dx_test_arm, shift=self.test_arm_shift, return_more=True)
        else:
            ref_beam = i.to_fpm_and_back(efl, self.pinhole, self.dx_pinhole)
            test_beam = i.to_fpm_and_back(efl, self.test_mask, self.dx_test_arm, shift=self.test_arm_shift)

        if self.test_arm_transmissivity != 1:
            test_beam *= self.test_arm_transmissivity

        self.ref_beam = ref_beam
        self.test_beam = test_beam
        total_field = ref_beam + test_beam
        if debug:
            return {
                'total_field': total_field,
                'at_camera': {
                    'ref': ref_beam,
                    'test': test_beam,
                },
                'at_fpm': {
                    'ref': (ref_at_fpm, ref_after_fpm),
                    'test': (test_at_fpm, test_after_fpm),
                }
            }
        return total_field.intensity


def evaluate_test_ref_arm_matching(debug_dict):
    pak = debug_dict['at_camera']
    I1 = pak['ref'].intensity
    I2 = pak['test'].intensity
    ratio = I1.data.mean()/I2.data.mean()
    return ratio, I1, I2
