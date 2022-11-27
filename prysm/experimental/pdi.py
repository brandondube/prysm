"""Point Diffraction Interferometry."""

from functools import partial

from prysm._richdata import RichData
from prysm.mathops import np
from prysm.coordinates import make_xy_grid, cart_to_polar
from prysm.propagation import Wavefront as WF
from prysm.geometry import truecircle

from skimage.restoration import unwrap_phase as ski_unwrap_phase


FIVE_FRAME_PSI_NOMINAL_SHIFTS = (-np.pi, -np.pi/2, 0, +np.pi/2, +np.pi)
FOUR_FRAME_PSI_NOMINAL_SHIFTS = (0, np.pi/2, np.pi, 3/2*np.pi)


def rectangle_pulse(x, duty=0.5, amplitude=0.5, offset=0.5, period=2*np.pi):
    """Rectangular pulse; generalized square wave.

    This function differs from scipy.signal.square in that the output
    is in [0,1] instead of [-1,1], as well as control over more parameters.

    Parameters
    ----------
    x : numpy.ndarray
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
    numpy.ndarray
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
                 test_arm_offset,
                 test_arm_fov,
                 test_arm_samples=256,
                 test_arm_transmissivity=1,
                 pinhole_diameter=0.25,
                 pinhole_samples=128,
                 grating_rulings=64,
                 grating_type='ronchi',
                 grating_axis='x'):
        """Create a new PS/PDI or Medecki Interferometer.

        Parameters
        ----------
        x : numpy.ndarray
            x coordinates for arrays that will be passed to forward_model
            not normalized
        y : numpy.ndarray
            y coordinates for arrays that will be passed to forward_model
            not normalized
        efl : float
            focal length in the focusing space behind the grating
        epd : float
            entrance pupil diameter, mm
        wavelength : float
            wavelength of light, um
        test_arm_offset : float
            TODO
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
        # munge
        if grating_type not in ('ronchi', 'sin'):
            raise ValueError('only ronchi gratings supported for now')
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
        elif grating_type == 'sin':
            raise ValueError('sin grating PS/PDI geometry not worked out yet')
            def f(x):
                prefix = grating_rulings*np.pi/(epd/2)
                phs = np.pi * np.sin(prefix*x)
                return np.exp(1j*phs)

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
        eps = pinhole_diameter / pinhole_samples
        self.pinhole_fov_radius = (pinhole_diameter + eps) * self.flambd

        # now a bit of computation

        # include a tiny epsilon to avoid any bad rounding
        # ph = pinhle; sq = squared;
        # more optimized to true a circle in squared coordinates
        xph, yph = make_xy_grid(pinhole_samples, diameter=2*self.pinhole_fov_radius)
        self.dx_pinhole = xph[0, 1] - x[0, 0]
        rphsq = xph*xph + yph*yph
        self.pinhole = truecircle((pinhole_diameter/2)**2, rphsq)

        # t = test
        xt, yt = make_xy_grid(test_arm_samples, diameter=2*self.test_arm_fov_compute)
        self.dx_test_arm = xt[0, 1] - xt[0, 0]

        rtsq = xt*xt + yt*yt
        self.test_mask = truecircle(self.test_arm_mask_rsq, rtsq)
        del xph, yph, rphsq, xt, yt, rtsq

    def forward_model(self, wave_in, phase_shift=0, debug=False):
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
        if self.grating_type == 'ronchi':
            if debug:
                ref_beam, ref_at_fpm, ref_after_fpm = \
                    i.to_fpm_and_back(efl, self.pinhole, self.dx_pinhole, return_more=True)
                test_beam, test_at_fpm, test_after_fpm = \
                    i.to_fpm_and_back(efl, self.test_mask, self.dx_test_arm, shift=self.test_arm_shift, return_more=True)
            else:
                ref_beam = i.to_fpm_and_back(efl, self.pinhole, self.dx_pinhole)
                test_beam = i.to_fpm_and_back(efl, self.test_mask, self.dx_test_arm, shift=self.test_arm_shift)
        else:
            raise ValueError("unsupported grating type")

        if self.test_arm_transmissivity != 1:
            test_beam *= self.test_arm_transmissivity

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


def four_frame_psi(g0, g1, g2, g3):
    """Sasaki algorithm.

    Ref.
    """
    was_rd = isinstance(g0, RichData)
    if was_rd:
        g00 = g0
        g0, g1, g2, g3 = g0.data, g1.data, g2.data, g3.data

    # Sasaki from degroot
    num = g0 + g1 - g2 - g3
    den = g0 - g1 + g2 - g3

    # other degroot
    num = g3 - g1
    den = g0 - g2

    out = np.arctan(num/den)
    if was_rd:
        out = RichData(out, g00.dx, g00.wavelength)

    return out


def five_frame_psi(g0, g1, g2, g3, g4):
    """Schwider-Hariharan algorithm.

    Ref.
    Digital phase-shifting interferometry: a simple error-compensating phase calculation algorithm.
    doi.org/10.1364/AO.26.002504

    Expects phase shifts -180, -90, 0, 90, 180 deg
    or -pi, -pi/2, 0, +pi/2, +pi

    Parameters
    ----------
    g0 : numpy.ndarray
        frame corresponding to -pi phase shift
    g1 : numpy.ndarray
        frame corresponding to -pi/2 phase shift
    g2 : numpy.ndarray
        frame corresponding to 0 phase shift
    g3 : numpy.ndarray
        frame corresponding to +pi/2 phase shift
    g4 : numpy.ndarray
        frame corresponding to +pi phase shift

    Returns
    -------
    numpy.ndarray
        wrapped phase estimate


    """
    was_rd = isinstance(g0, RichData)
    if was_rd:
        g00 = g0
        g0, g1, g2, g3, g4 = g0.data, g1.data, g2.data, g3.data, g4.data

    num = 2*(g1-g3)
    den = -(g0+g4) + 2*g2
    out = np.arctan(num/den)
    if was_rd:
        out = RichData(out, g00.dx, g00.wavelength)

    return out
    # return np.arctan2(num, den)


def unwrap_phase(wrapped):
    if isinstance(wrapped, RichData):
        wrapped = wrapped.data
    return ski_unwrap_phase(wrapped)
