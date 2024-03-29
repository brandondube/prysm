"""Self-Referenced Interferometer."""

# Cousin of the point diffraction interferometer

from prysm.mathops import np
from prysm.propagation import Wavefront as WF
from prysm.coordinates import make_xy_grid
from prysm.geometry import circle

from .pdi import evaluate_test_ref_arm_matching


class SelfReferencedInterferometer:
    """Self-Referenced Interferometer."""
    def __init__(self, x, y, efl, epd, wavelength,
                 pinhole_diameter=0.25,
                 pinhole_samples=128,
                 beamsplitter_RT=(0.8, 0.2)):
        """Create a new Self-Referenced Interferometer.

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

        self.pinhole_diameter = pinhole_diameter * self.flambd
        self.pinhole_samples = pinhole_samples
        # -1 is an epsilon to make sure the circle is wholly inside the array
        self.dx_pinhole = pinhole_diameter / (pinhole_samples-2)
        self.pinhole_fov_radius = pinhole_samples/2*self.dx_pinhole

        xph, yph = make_xy_grid(pinhole_samples, diameter=2*self.pinhole_fov_radius)
        rphsq = xph*xph + yph*yph
        self.pinhole = circle((pinhole_diameter/2)**2, rphsq)

        # big R, big T -> little r, little t
        # (power -> amplitude)
        self.ref_r = beamsplitter_RT[0]**0.5
        self.test_t = beamsplitter_RT[1]**0.5

    def forward_model(self, wave_in, phase_shift=0, debug=False):
        """Perform a forward model, returning the intensity at the detector plane.

        Parameters
        ----------
        wave_in : numpy.ndarray
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

        # test wave has a phase shift
        if phase_shift != 0:
            phase_shift = np.exp(1j*phase_shift)
            test_beam = wave_in * phase_shift
        else:
            test_beam = wave_in

        if debug:
            ref_beam, ref_at_fpm, ref_after_fpm = \
                wave_in.to_fpm_and_back(self.efl, self.pinhole, self.dx_pinhole, return_more=True)
        else:
            ref_beam = wave_in.to_fpm_and_back(self.efl, self.pinhole, self.dx_pinhole)

        ref_beam = ref_beam * self.ref_r
        test_beam = test_beam * self.test_t
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
                }
            }
        return total_field.intensity
