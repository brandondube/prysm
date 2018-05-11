"""Shack Hartmann sensor modeling tools."""
from collections import deque

from .detector import bindown
from .units import waves_to_microns
from .util import share_fig_ax


class ShackHartmann(object):
    """Shack Hartmann Wavefront Sensor.

    Attributes
    ----------
    buffer_depth : `int`
        number of frames to store from captures
    captures : `deque`
        fixed size set of captures
    captures_simple : `deque`
        simple representation of captures; dot grid matricies of spot centroids
    captures_wvl : `deque`
        wavelength associated with each capture
    lenslet_array_shape : `str`, {'square', 'rectangular'}
        shape of lenslets; square will use less than the entire long side of the sensor
    lenslet_efl : `float`
        efl of lenslets, microns
    lenslet_fno : `float`
        F/# of the lenslets
    lenslet_pitch : `float`
        center-to-center spacing of the lenslets
    megapixels : `float`
        total number of megapixels
    num_lenslets : `tuple`
        (x,y) resolution in lenslets
    pixel_locations_x : `numpy.ndarray`
        x locations of pixel centers
    pixel_locations_y : `numpy.ndarray`
        y locations of pixel centers
    pixel_pitch : `float`
        center to center spacing of pixels
    resolution : `tuple`
        (x,y) resolution, px
    sensor_size : `tuple`
        (x,y) sensor size, mm
    total_lenslets : `int`
        total number of lenslets (analagous to megapixels)
    wavelength : `float`
        operating wavelength, microns.  Can change with exposure

    """
    def __init__(self, sensor_size=(36, 24), pixel_pitch=3.99999,
                 lenslet_pitch=375, lenslet_efl=2000, lenslet_fillfactor=0.9,
                 lenslet_array_shape='square', framebuffer=24,
                 wavelength=0.5):
        """Create a new SHWFS object.

        Parameters
        ----------
        sensor_size : `iterable`
            (x, y) sensor sizes in mm
        pixel_pitch : `float`
            center-to-center pixel spacing in um
        lenslet_pitch : `float`
            center-to-center spacing of lenslets in um
        lenslet_efl : `float`
            lenslet focal length, in microns
        lenslet_fillfactor : `float`
            portion of the sensor height filled by the lenslet array.
            0.9 reserves 5% of the height on both the top and bottom of the array
        lenslet_array_shape : `str`, {'square', 'rectangular'}
            square will inscribe a square array within the detector area,
            rectangular will fill the detector area up to the fill factor
        framebuffer : `int`
            maximum number of frames of data to store
        wavelength : `float`
            wavelength of light, microns

        """
        # process lenslet array shape and lenslet offset
        if lenslet_array_shape.lower() == 'square':
            self.lenslet_array_shape = 'square'
        elif lenslet_array_shape.lower() in ('rectangular', 'rect'):
            self.lenslet_array_shape = 'rectangular'
        lenslet_shift = lenslet_pitch // 2

        # store data related to the silicon
        self.sensor_size = sensor_size
        self.pixel_pitch = pixel_pitch
        self.resolution = (int(sensor_size[0] // (pixel_pitch / 1e3)),
                           int(sensor_size[1] // (pixel_pitch / 1e3)))
        self.megapixels = self.resolution[0] * self.resolution[1] / 1e6
        self.pixel_locations_x = None
        self.pixel_locations_y = None

        # compute lenslet array shifts
        if self.lenslet_array_shape == 'square':
            xidx = 1
            sensor_extra_x = sensor_size[0] - sensor_size[1]
            if sensor_extra_x < 0:
                sensor_extra_y = sensor_size[1] - sensor_size[0]
                yshift = (sensor_extra_y / sensor_size[1]) / 2
                xshift = 0
            else:
                xshift = (sensor_extra_x / sensor_size[0]) / 2
                yshift = 0
        else:
            xidx = 0
            xshift = 0
            yshift = 0
        yidx = 1

        # store lenslet metadata - TODO: figure out why I need the - 1
        self.num_lenslets = (int(sensor_size[xidx] * lenslet_fillfactor // (lenslet_pitch / 1e3)) - 1,
                             int(sensor_size[yidx] * lenslet_fillfactor // (lenslet_pitch / 1e3)) - 1)
        self.total_lenslets = self.num_lenslets[0] * self.num_lenslets[1]
        self.lenslet_pitch = lenslet_pitch
        self.lenslet_efl = lenslet_efl
        self.lenslet_fno = self.lenslet_efl / self.lenslet_pitch

        # compute lenslet locations
        start_factor = (1 - lenslet_fillfactor) / 2
        end_factor = 1 - start_factor
        start_factor_x, end_factor_x = start_factor + xshift, end_factor - xshift
        start_factor_y, end_factor_y = start_factor + yshift, end_factor - yshift

        # factors of 1e3 convert mm to um, and round to 0.1nm to avoid machine precision errors
        lenslet_start_x = round(start_factor_x * sensor_size[0] * 1e3 + lenslet_shift, 4)
        lenslet_start_y = round(start_factor_y * sensor_size[1] * 1e3 + lenslet_shift, 4)
        lenslet_end_x = round(end_factor_x * sensor_size[0] * 1e3 - lenslet_shift, 4)
        lenslet_end_y = round(end_factor_y * sensor_size[1] * 1e3 - lenslet_shift, 4)

        lenslet_pos_x = m.linspace(lenslet_start_x, lenslet_end_x, self.num_lenslets[0])
        lenslet_pos_y = m.linspace(lenslet_start_y, lenslet_end_y, self.num_lenslets[0])
        self.refx, self.refy = m.meshgrid(lenslet_pos_x, lenslet_pos_y)

        # initiate the frame buffer and store the wavelength
        self.buffer_depth = framebuffer
        self.captures = deque(maxlen=framebuffer)
        self.captures_simple = deque(maxlen=framebuffer)
        self.captures_wvl = deque(maxlen=framebuffer)
        self.wavelength = wavelength

    def _prep_pixel_grid(self):
        '''Prepare the pixel grid.

        This function allows a new SH WFS object to lazily compute its pixel
        coordinates only when necessary, as a high resolution sensor will take
        a bit of time to prepare.

        Returns
        -------
        self.pixel_locations_x : `numpy.ndarray`
            array of pixel x coordinates, microns
        self.pixel_locations_y : `numpy.ndarray`
            aray of pixel y coordinates, microns

        '''
        if self.pixel_locations_x is None:
            pp = self.pixel_pitch
            pxx, pxy = self.resolution
            pixextx, pixexty = pp * pxx, pp * pxy
            x = m.arange(0, pixextx, pp)
            y = m.arange(0, pixexty, pp)
            self.pixel_locations_x, self.pixel_locations_y = m.meshgrid(x, y)

        return self.pixel_locations_x, self.pixel_locations_y

    def __repr__(self):
        """Represent object.

        Returns
        -------
        `str`
            a string description of this sensor

        """
        return ('Shack Hartmann sensor with: \n'
                f'\t({self.resolution[0]:}x{self.resolution[1]})px, {self.megapixels:.1f}MP CMOS\n'
                f'\t({self.num_lenslets[0]}x{self.num_lenslets[1]})lenslets, '
                f'{self.total_lenslets:1.0f} wavefront samples\n'
                f'\t{self.buffer_depth} frame buffer, currently storing {len(self.captures)} frames')

    def plot_reference_spots(self, fig=None, ax=None):
        """Create a plot of the reference positions of lenslets.

        Parameters
        ----------
        fig : `matplotlib.figure.Figure`, optional
            Figure containing the plot
        ax : `matplotlib.axes.Axis`, optional:
            Axis containing the plot

        Returns
        -------
        fig : `matplotlib.figure.Figure`, optional
            Figure containing the plot
        ax : `matplotlib.axes.Axis`, optional:
            Axis containing the plot

        """
        fig, ax = share_fig_ax(fig, ax)
        ax.scatter(self.refx / 1e3, self.refy / 1e3, c='k', s=8)
        ax.set(xlim=(0, self.sensor_size[0]), xlabel='Detector Position X [mm]',
               ylim=(0, self.sensor_size[1]), ylabel='Detector Position Y [mm]',
               aspect='equal')
        return fig, ax

    def sample_wavefront(self, pupil, make_image=False, fig=None, ax=None):
        """Sample a wavefront, producing a Shack-Hartmann spot grid.

        Parameters
        ----------
        pupil : `Pupil`
            a pupil object
        make_image : `bool`
            boolean, whether to simulate the actual detector; must be false or will raise
            NotImplementedError
        fig : `matplotlib.figure.Figure`, optional
            Figure containing the plot
        ax : `matplotlib.axes.Axis`, optional:
            Axis containing the plot

        Returns
        -------
        psf_centers_x : `numpy.ndarray`
            x coordinate centers of the PSFs
        psf_centers_y : `numpy.ndarray`
            y coordiante centers of the PSFs

        Notes
        -----
        Algorithm is as follows:
            1.  Compute the gradient of the wavefront.
            2.  Bindown the wavefront such that each sample in the output
                corresponds to the local wavefront gradient at a lenslet.
            3.  Compute the x and y delta of each PSF in the image plane.
            4.  Shift each spot by the corresponding delta.
                This is the end for make_image=False.
            5.  If make_image=True, then the dots are painted onto the pixel
                grid and that array is convolved with a sinc function and
                deconvolved with a rect function.  The sinc is the PSF of the
                lenslets, which are assumed to all be the same and independent
                of the local wavefront they sampled, and the rect is to remove
                the finite width of the delta imposed by the pixel grid.

        """
        # grab the phase from the pupil and convert to units of length
        pupil._correct_phase_units()  # -> waves -> microns
        data = pupil.phase * waves_to_microns(pupil.wavelength)
        if pupil.wavelength != self.wavelength:
            self.wavelength = pupil.wavelength

        # convert the phase error to radians in the paraxial approximation
        data /= self.lenslet_pitch  # epd

        # compute the gradient - TODO: see why gradient is dy,dx not dx,dy
        normalized_sample_spacing = 2 / pupil.samples
        dy, dx = m.gradient(data, normalized_sample_spacing, normalized_sample_spacing)

        # convert the gradient from waves to radians -- angle alpha is made as:
        '''
          dz
        ______
        \    |
         \   |
          \ a| rho
           \ |
            \|
             \
            a = tan(dz/rho)
            can either compute "one sided" a, using the vertex to radius as rho
            and the vertex to radius phase error, or compute "two sided" a using
            the epd and the full wavefront error (PV) as dz.
        '''

        # bin to the lenslet area
        nlenslets_y = self.num_lenslets[1]
        npupilsamples_y = pupil.samples
        npx_bin = npupilsamples_y // nlenslets_y
        dx_binned = bindown(dx, npx_bin, mode='avg')
        dy_binned = bindown(dy, npx_bin, mode='avg')

        # compute the lenslet PSF shift
        shift_x, shift_y = psf_shift(self.lenslet_efl, dx_binned, dy_binned)
        psf_centers_x, psf_centers_y = self.refx + shift_x, self.refy + shift_y
        self.captures_simple.append({
            'x': psf_centers_x,
            'y': psf_centers_y})
        self.captures_wvl.append(pupil.wavelength)
        if make_image:
            pass  # TODO: write image simulation
        else:
            self.captures.append(None)
            return psf_centers_x, psf_centers_y

    def plot_simple_result(self, result_index=-1, type='quiver', fig=None, ax=None):
        '''Plot the simple version of the most recent result.

        Parameters
        ----------
        result_index : `int`
            which capture to plot, defaults to most recent
        type : `str`, {'quiver', 'dots'}
            what type of plot to make
        fig : `matplotlib.figure.Figure`, optional
            Figure containing the plot
        ax : `matplotlib.axes.Axis`, optional:
            Axis containing the plot

        Returns
        -------
        fig : `matplotlib.figure.Figure`, optional
            Figure containing the plot
        ax : `matplotlib.axes.Axis`, optional:
            Axis containing the plot

        '''
        idx = result_index

        fig, ax = share_fig_ax(fig, ax)

        mx, my = self.captures_simple[idx]['x'], self.captures_simple[idx]['y']
        if type.lower() in ('q', 'quiver'):
            dx, dy = mx - self.refx, my - self.refy
            ax.quiver(self.refx, self.refy, dx, dy, scale=1, units='xy', scale_units='xy')
        elif type.lower() in ('d', 'dots'):
            ax.scatter(self.refx, self.refy, c='k', s=8)
            ax.scatter(mx, my, c='r', s=8)

        ax.set(xlim=(0, self.sensor_size[0] * 1e3), xlabel=r'Detector X [$\mu m$]',
               ylim=(0, self.sensor_size[1] * 1e3), ylabel=r'Detector Y [$\mu m$]',
               aspect='equal')
        return fig, ax


def psf_shift(lenslet_efl, dx, dy, mag=1):
    '''Compute the shift of a PSF, in microns.

    Parameters
    ----------
    lenslet_efl : `float`
        EFL of lenslets, microns
    dx : `m.ndarray`
        dx gradient of wavefront
    dy : `m.ndarray`
        dy gradient of wavefront
    mag : `float`
        magnification of the collimation system

    Returns
    -------
    `numpy.ndarray`
        array of PSF shifts

    Notes
    -----
    see eq. 12 of Chanan, "Principles of Wavefront Sensing and Reconstruction"
    delta = m * fl * grad(z)
    m is magnification of SH system
    fl is lenslet focal length
    grad(z) is the x, y gradient of the opd, z, which is expressed in radians.

    '''
    coef = -mag * lenslet_efl
    return coef * dx, coef * dy
