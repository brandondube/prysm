''' Model of optical systems
'''
import warnings
from functools import partial
from copy import deepcopy

from scipy.optimize import minimize

from .conf import config
from .seidel import Seidel
from .psf import PSF
from .otf import MTF
from .util import share_fig_ax
from .thinlens import image_displacement_to_defocus
from .mtf_utils import MTFvFvF

from prysm import mathops as m


class Lens(object):
    ''' Represents a lens or optical system.
    '''
    def __init__(self, **kwargs):
        ''' Create a new Lens object.

        Args:
            efl (`float`): Effective Focal Length.

            fno (`float`): Focal Ratio.

            pupil_magnification (`float`): Ratio of exit pupil to entrance pupil
                diameter.

            aberrations (`dict`): A dictionary

            fields (`iterable`): A set of relative field points to analyze (symmetric)

            fov_x (`float`): half Field of View in X

            fov_y (`float`): half Field of View in Y

            fov_unit (`string`): unit for field of view.  mm, degrees, etc.

            wavelength (`float`): wavelength of light, in um.

            samples (`float`): samples in the pupil plane used to compute wavefronts.
        '''
        efl = 1
        fno = 1
        pupil_magnification = 1
        ab = dict()
        fields = [0, 1]
        fov_x = 0
        fov_y = 21.64
        fov_unit = 'mm'
        wavelength = 0.55
        samples = 128
        if kwargs is not None:
            for key, value in kwargs.items():
                kl = key.lower()
                if kl == 'efl':
                    efl = value
                elif kl == 'fno':
                    fno = value
                elif kl == 'pupil_magnification':
                    pupil_magnification = value
                elif kl in ('aberrations', 'abers', 'abs'):
                    ab = value
                elif kl == 'fields':
                    fields = value
                elif kl == 'fov_x':
                    fov_x = value
                elif kl == 'fov_y':
                    fov_y = value
                elif kl == 'fov_unit':
                    fov_unit = value
                elif kl == 'wavelength':
                    wavelength = value
                elif kl == 'samples':
                    samples = value

        if efl < 0:
            warnings.warn('''
                Negative focal lengths are treated as positive for fresnel
                diffraction propogation to function correctly.  In the context
                of these simulations a positive and negative focal length are
                functionally equivalent and the provide value has had its sign
                flipped.
                ''')
            efl *= -1
        if fno < 0:
            raise ValueError('f/# must by definition be positive')

        self.efl = efl
        self.fno = fno
        self.pupil_magnification = pupil_magnification
        self.epd = efl / fno
        self.xpd = self.epd * pupil_magnification
        self.aberrations = ab
        self.fields = fields
        self.fov_x = fov_x
        self.fov_y = fov_y
        self.fov_unit = fov_unit
        self.wavelength = wavelength
        self.samples = samples

    ####### analytically setting aberrations -----------------------------------

    def autofocus(self, field_index=0):
        ''' Adjusts the W020 aberration coefficient to maximize the MTF at a
            given field index.

        Args:
            field_index (`int`): index of the field to maximize MTF at.

        Returns:
            `Lens` self.
        '''
        coefs = self.aberrations.copy()
        try:
            # try to access the W020 aberration
            float(coefs['W020'])
        except KeyError:
            # if it is not set, make it 0
            coefs['W020'] = 0.0

        def opt_fcn(self, coefs, w020):
            # shift the defocus term appropriately
            abers = coefs.copy()
            abers['W020'] += w020
            pupil = Seidel(**abers, epd=self.epd, samples=self.samples, h=self.fields[field_index])

            # cost value (to be minimized) is RMS wavefront
            return pupil.rms

        opt_fcn = partial(opt_fcn, self, coefs)

        new_defocus = minimize(opt_fcn, x0=0, method='Powell')
        coefs['W020'] += float(new_defocus['x'])
        self.aberrations = coefs.copy()
        return self

    ####### analytically setting aberrations -----------------------------------

    ####### data generation ----------------------------------------------------

    def psf_vs_field(self, num_pts):
        ''' Generates a list of PSFs as a function of field.

        Args:
            num_pts (`int`): number of points to generate a PSF for.

        Returns:
            `list` containing the PSF objects.

        '''
        self._uniformly_spaced_fields(num_pts)
        psfs = []
        for idx in range(num_pts):
            psfs.append(self._make_psf(idx))
        return psfs

    def mtf_vs_field(self, num_pts, freqs=[10, 20, 30, 40, 50]):
        ''' Generates a 2D array of MTF vs field values for the given spatial
            frequencies.

        Args:
            num_pts (`int`): Number of points to compute the MTF at.

            freqs (`iterable`): set of frequencies to compute at.

        Returns:
            `tuple` containing:

                `numpy.ndarray` (Tan) a 3D ndnarray where the columns
                    correspond to fields and the rows correspond to spatial
                    frequencies.

                `numpy.ndarray` (Sag) a 3D ndnarray where the columns
                    correspond to fields and the rows correspond to spatial
                    frequencies.

        '''
        self._uniformly_spaced_fields(num_pts)
        mtfs_t = m.empty((num_pts, len(freqs)))
        mtfs_s = m.empty((num_pts, len(freqs)))
        for idx in range(num_pts):
            mtf = self._make_mtf(idx)
            vals_t = mtf.exact_polar(freqs, 0)
            vals_s = mtf.exact_polar(freqs, 90)
            mtfs_t[idx, :] = vals_t
            mtfs_s[idx, :] = vals_s

        return mtfs_s, mtfs_t

    ####### data generation ----------------------------------------------------

    ####### plotting -----------------------------------------------------------

    def plot_psf_vs_field(self, num_pts, fig=None, axes=None, axlim=25):
        ''' Creates a figure showing the evolution of the PSF over the field
            of view.

        Args:
            num_pts (`int`): Number of points between (0,1) to create a PSF for

        Returns:
            `tuple` containing:

                `matplotlib.pyplot.figure` figure containing the plots.

                `list` the axes the plots are placed in.

        '''
        psfs = self.psf_vs_field(num_pts)
        fig, axes = share_fig_ax(fig, axes, numax=num_pts, sharex=True, sharey=True)

        for idx, (psf, axis) in enumerate(zip(psfs, axes)):
            show_labels = False
            show_colorbar = False
            if idx == 0:
                show_labels = True
            elif idx == num_pts - 1:
                show_colorbar = True
            psf.plot2d(fig=fig, ax=axis, axlim=axlim,
                       show_axlabels=show_labels, show_colorbar=show_colorbar)

        fig_width = 15
        fig.set_size_inches(fig_width, fig_width / num_pts)
        fig.tight_layout()
        return fig, axes

    def plot_mtf_vs_field(self, num_pts, freqs=[10, 20, 30, 40, 50], title='MTF vs Field', minorgrid=True, fig=None, ax=None):
        ''' Generates a plot of the MTF vs Field for the lens.

        Args:
            num_pts (`int`): number of field points to evaluate.

            freqs (`iterable`): frequencies to evaluate the MTF at.

            fig (`matplotlib.pyplot.figure`): figure to plot inside.

            ax (`matplotlib.pyplot.axis`): axis to plot ini.

        Return:
            `tuple` containing:

                `matplotlib.pyplot.figure` figure containing the plot.

                `matplotlib.pyplot.axis` axis containing the plot.

        '''
        data_s, data_t = self.mtf_vs_field(num_pts, freqs)
        flds_abs = m.linspace(0, self.fov_y, num_pts)
        fig, ax = share_fig_ax(fig, ax)
        for i in range(len(freqs)):
            ln, = ax.plot(flds_abs, data_s[:, i], lw=3, ls='--')
            ax.plot(flds_abs, data_t[:, i], lw=3, color=ln.get_color(), label=f'{freqs[i]}lp/mm')

        ax.plot(0, 0, color='k', ls='--', label='Sag')
        ax.plot(0, 0, color='k', label='Tan')
        # todo: respect units of `self`

        if minorgrid is True:
            ax.set_yticks([0.1, 0.3, 0.5, 0.7, 0.9], minor=True)
            ax.grid(True, which='minor')

        ax.set(xlim=(0, self.fov_y), xlabel='Image Height [mm]',
               ylim=(0, 1), ylabel='MTF [Rel. 1.0]',
               title=title)
        ax.legend()

        return fig, ax

    def plot_mtf_thrufocus(self, field_index, focus_range, numpts, freqs, fig=None, ax=None):
        focus, mtfs = self._make_mtf_thrufocus(field_index, focus_range, numpts)
        t = []
        s = []
        for mtf in mtfs:
            t.append(mtf.exact_polar(freqs, 0))
            s.append(mtf.exact_polar(freqs, 90))

        t, s = m.asarray(t), m.asarray(s)
        fig, ax = share_fig_ax(fig, ax)
        for idx, freq in enumerate(freqs):
            l, = ax.plot(focus, t[:, idx], lw=2, label=freq)
            ax.plot(focus, s[:, idx], lw=2, ls='--', c=l.get_color())
        ax.legend(title=r'$\nu$ [cy/mm]')
        ax.set(xlim=(focus[0], focus[-1]), xlabel=r'Defocus [$\mu m$]',
               ylim=(0, 1), ylabel='MTF [Rel. 1.0]',
               title='Through Focus MTF')

        return fig, ax

    ####### plotting -----------------------------------------------------------

    ####### helpers ------------------------------------------------------------

    def _make_pupil(self, field_index):
        ''' Generates the pupil for a given field

        Args:
            field_index (`int`): index of the desired field in the self.fields
                iterable.

        Returns:
            `Pupil` a pupil object.
        '''
        return Seidel(**self.aberrations,
                      epd=self.epd,
                      h=self.fields[field_index],
                      wavelength=self.wavelength,
                      samples=self.samples)

    def _make_psf(self, field_index):
        ''' Generates the psf for a given field

        Args:
            field_index (`int`): index of the desired field in the self.fields
                iterable.

        Returns:
            `PSF` a psf object.
        '''
        p = self._make_pupil(field_index=field_index)
        return PSF.from_pupil(p, self.efl)

    def _make_mtf(self, field_index):
        ''' Generates the mtf for a given field

        Args:
            field_index (`int`): index of the desired field in the self.fields
                iterable.

        Returns:
            `MTF` an MTF object.
        '''
        pp = self._make_psf(field_index=field_index)
        return MTF.from_psf(pp)

    def _make_mtf_thrufocus(self, field_index, focus_range, num_pts):
        ''' Makes a list of MTF objects corresponding to different focus shifts
            for the lens.  Focusrange will be applied symmetrically.

        Args:
            field_index: (`int`): index of the desired field in the self.fields
                iterable.

            focus_range: (`float`): focus range, in microns.

            num_pts (`int`): number of points to compute MTF at.  Note that for
                and even number of points, the zero defocus point will not be
                sampled.

        Returns:
            list of `MTF` objects.

        '''
        # todo: parallelize
        focus_shifts = m.linspace(-focus_range, focus_range, num_pts)
        defocus_wvs = image_displacement_to_defocus(focus_shifts, self.fno, self.wavelength)

        mtfs = []
        pupil = self._make_pupil(field_index)
        for defocus in defocus_wvs:
            defocus_p = Seidel(W020=defocus, epd=self.epd,
                               samples=self.samples, wavelength=self.wavelength)
            psf = PSF.from_pupil(pupil.merge(defocus_p), self.efl)
            mtfs.append(MTF.from_psf(psf))
        return focus_shifts, mtfs

    def _make_mtf_vs_field_vs_focus(self, num_fields, focus_range, num_focus, freqs):
        ''' TODO: docstring
        '''
        self._uniformly_spaced_fields(num_fields)
        net_mtfs = [None] * num_fields
        for idx in range(num_fields):
            focus, net_mtfs[idx] = self._make_mtf_thrufocus(idx, focus_range, num_focus)

        fields = (self.fields[-1] * self.fov_y) * m.linspace(0, 1, num_fields)
        t_cube = m.empty((num_focus, num_fields, len(freqs)))
        s_cube = m.empty((num_focus, num_fields, len(freqs)))
        for idx, mtfs in enumerate(net_mtfs):
            for idx2, submtf in enumerate(mtfs):
                t = submtf.exact_polar(freqs, 0)
                s = submtf.exact_polar(freqs, 90)
                t_cube[idx2, idx, :] = t
                s_cube[idx2, idx, :] = s

        TCube = MTFvFvF(data=t_cube, focus=focus, field=fields, freq=freqs, azimuth='Tan')
        SCube = MTFvFvF(data=s_cube, focus=focus, field=fields, freq=freqs, azimuth='Sag')
        return TCube, SCube

    def _uniformly_spaced_fields(self, num_pts):
        ''' Changes the `fields` property to n evenly spaced points from 0~1.

        Args:
            num_pts (`int`): number of points.

        Returns:
            self.

        '''
        _ = m.arange(0, num_pts, dtype=config.precision)
        flds = _ / _.max()
        self.fields = flds
        return self

    ####### helpers ------------------------------------------------------------

    def clone(self):
        ''' Makes a deep copy of this Lens instance.

        Returns:
            `Lens` a new Lens instance.
        '''
        ret = Lens()
        ret.__dict__ = deepcopy(self.__dict__)
        return ret

    def __repr__(self):
        return (f'Lens with properties:\n\t'
                f'efl: {self.efl}\n\t'
                f'f/#: {self.fno}\n\t'
                f'pupil mag: {self.pupil_magnification}\n\t'
                'Aberrations:\n\t\t'
                f'{str(self.aberrations)}')


def _spherical_defocus_from_monochromatic_mtf(lens, frequencies, mtf_s, mtf_t):
    ''' Uses nonlinear optimization to set the W020, W040, W060, and W080
        coefficients in a lens model based on MTF measurements taken on the
        optical axis.

    Args:
        lens (`Lens`): a lens object.

        frequencies (`iterable`): A set of frequencies the provided MTF values
            correspond to.

        mtf_s (`iterable`): A set of sagittal MTF measurements of equal length
            to the frequencies argument.

        mtf_t (`iterable`): A set of tangential MTF measurements of equal length
            to the frequencies argument.

    Returns:
        `Lens` A new lens object with its aberrations field modified with new
            spherical coefficients.

    '''
    work_lens = lens.clone()

    fcn = partial(_spherical_cost_fcn_raw, frequencies,
                  mtf_s, mtf_t, work_lens)

    results = minimize(fcn, [0, 0, 0, 0], method='Powell')
    W020, W040, W060, W080 = results['x']
    work_lens.aberrations['W020'] = W020
    work_lens.aberrations['W040'] = W040
    work_lens.aberrations['W060'] = W060
    work_lens.aberrations['W080'] = W080
    return work_lens


def _spherical_cost_fcn_raw(frequencies, truth_s, truth_t, lens, abervalues):
    ''' TODO - document.  partial() should be used on this and scipy.minimize'd

        abervalues - array of [W020, W040, W060, W080]
    '''
    pupil = Seidel(epd=lens.epd, samples=lens.samples,
                   W020=abervalues[0],
                   W040=abervalues[1],
                   W060=abervalues[2],
                   W080=abervalues[3])
    psf = PSF.from_pupil(pupil, efl=lens.efl)
    mtf = MTF.from_psf(psf)
    synth_t = mtf.exact_polar(frequencies, 0)
    synth_s = mtf.exact_polar(frequencies, 90)

    truth = m.stack((truth_s, truth_t))
    synth = m.stack((synth_s, synth_t))

    return ((truth - synth) ** 2).sum()
