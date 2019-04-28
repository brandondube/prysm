"""Defines behavior of convolvable items and a base class to encapsulate that behavior."""

from prysm import mathops as m
from ._basicdata import BasicData
from .coordinates import resample_2d_complex
from .conf import config
from .fttools import forward_ft_unit, pad2d
from .util import share_fig_ax


class Convolvable(BasicData):
    """A base class for convolvable objects to inherit from."""
    _data_attr = 'data'

    def __init__(self, x, y, data, has_analytic_ft=False):
        """Create a new Convolvable object.

        Parameters
        ----------
        x : `numpy.ndarray`
            1D ndarray defining x data grid
        y : `numpy.ndarray`
            1D ndarray defining y data grid
        data : `numpy.ndarray`
            2D ndarray of data
        has_analytic_ft : `bool`, optional
            Whether this convolvable overrides self.analytic_ft, and has a known
            analytical fourier tansform

        """
        super().__init__(x=x, y=y, data=data)
        self.has_analytic_ft = has_analytic_ft

    def __str__(self):
        """Pretty print description."""
        return f'{type(self)} with sample spacing {self.sample_spacing:.3f} and support {self.support:.3f} μm'

    @property
    def support_x(self):
        """Width of the domain in X."""
        return (self.samples_x - 1) * self.sample_spacing

    @property
    def support_y(self):
        """Width of the domain in Y."""
        return (self.samples_y - 1) * self.sample_spacing

    @property
    def support(self):
        """Width of the domain."""
        return max((self.support_x, self.support_y))

    def plot_slice_xy(self, axlim=20, lw=config.lw, zorder=config.zorder, fig=None, ax=None):
        """Create a plot of slices through the X and Y axes of the `PSF`.

        Parameters
        ----------
        axlim : `float` or `int`, optional
            axis limits, in microns
        lw : `float`, optional
            line width
        zorder : `int`, optional
            zorder
        fig : `matplotlib.figure.Figure`, optional
            Figure to draw plot in
        ax : `matplotlib.axes.Axis`
            Axis to draw plot in

        Returns
        -------
        fig : `matplotlib.figure.Figure`, optional
            Figure containing the plot
        ax : `matplotlib.axes.Axis`, optional
            Axis containing the plot

        """
        ux, x = self.slice_x
        uy, y = self.slice_y

        label_str = 'Normalized Intensity [a.u.]'
        lims = (0, 1)

        fig, ax = share_fig_ax(fig, ax)

        ax.plot(ux, x, label='X', lw=lw, zorder=zorder)
        ax.plot(uy, y, label='Y', lw=lw, zorder=zorder)
        ax.set(xlabel=f'Image Plane [μm]',
               ylabel=label_str,
               xlim=(-axlim, axlim),
               ylim=lims)
        ax.legend(title='Slice', loc='upper right')
        return fig, ax

    def conv(self, other):
        """Convolves this convolvable with another.

        Parameters
        ----------
        other : `Convolvable`
            A convolvable object
        postprocess: `iterable` or callable or None, optional
            function or sequence of functions to call to post-process the result.
            None if no work should be done.  Functions should accept a single
            argument, the spatial representation of the result, centered on the origin
        dryrun : `bool`, optional
            if True, return a string describing the process of convolution for this case

        Returns
        -------
        `Convolvable`
            a convolvable that lacks an analytical fourier transform

        Notes
        -----
        The algoithm works according to the following cases:
            1.  Both self and other have analytical fourier transforms:
                - The analytic forms will be used to compute the output directly.
                - The output sample spacing will be the finer of the two inputs.
                - The output window will cover the same extent as the "wider"
                  input.  If this window is not an integer number of samples
                  wide, it will be enlarged symmetrically such that it is.  This
                  may mean the output array is not of the same size as either
                  input.
                - An input which contains a sample at (0,0) may not produce an
                  output with a sample at (0,0) if the input samplings are not
                  favorable.  To ensure this does not happen confirm that the
                  inputs are computed over identical grids containing 0 to
                  begin with.
            2.  One of self and other have analytical fourier transforms:
                - The input which does NOT have an analytical fourier transform
                  will define the output grid.
                - The available analytic FT will be used to do the convolution
                  in Fourier space.
            3.  Neither input has an analytic fourier transform:
                3.1, the two convolvables have the same sample spacing to within
                     a numerical precision of 0.1 nm:
                    - the fourier transform of both will be taken.  If one has
                      fewer samples, it will be upsampled in Fourier space
                3.2, the two convolvables have different sample spacing:
                    - The fourier transform of both inputs will be taken.  It is
                      assumed that the more coarsely sampled signal is Nyquist
                      sampled or better, and thus acts as a low-pass filter; the
                      more finaly sampled input will be interpolated onto the
                      same grid as the more coarsely sampled input.  The higher
                      frequency energy would be eliminated by multiplication with
                      the Fourier spectrum of the more coarsely sampled input
                      anyway.

        The subroutines have the following properties with regard to accuracy:
            1.  Computes a perfect numerical representation of the continuous
                output, provided the output grid is capable of Nyquist sampling
                the result.
            2.  If the input that does not have an analytic FT is unaliased,
                computes a perfect numerical representation of the continuous
                output.  If it does not, the input aliasing limits the output.
            3.  Accuracy of computation is dependent on how much energy is
                present at nyquist in the worse-sampled input; if this input
                is worse than Nyquist sampled, then the result will not be
                correct.

        """
        e = ConvolutionEngine(self, other)
        return e.fire()

    def deconv(self, other, balance=1000, reg=None, is_real=True, clip=False, postnormalize=True):
        """Perform the deconvolution of this convolvable object by another.

        Parameters
        ----------
        other : `Convolvable`
            another convolvable object, used as the PSF in a Wiener deconvolution
        balance : `float`, optional
            regularization parameter; passed through to skimage
        reg : `numpy.ndarray`, optional
            regularization operator, passed through to skimage
        is_real : `bool`, optional
            True if self and other are both real
        clip : `bool`, optional
            clips self and other into (0,1)
        postnormalize : `bool`, optional
            normalize the result such that it falls in [0,1]


        Returns
        -------
        `Convolvable`
            a new Convolable object

        Notes
        -----
        See skimage:
        http://scikit-image.org/docs/dev/api/skimage.restoration.html#skimage.restoration.wiener

        """
        from skimage.restoration import wiener

        result = wiener(self.data, other.data, balance=balance, reg=reg, is_real=is_real, clip=clip)
        if postnormalize:
            result += result.min()
            result /= result.max()
        return Convolvable(result, self.x, self.y, False)

    def renorm(self):
        """Renormalize so that the peak is at a value of unity."""
        self.data /= self.data.max()
        return self

    def show(self, xlim=None, ylim=None, interp_method=None, power=1, show_colorbar=True, fig=None, ax=None):
        """Display the image.

        Parameters
        ----------
        xlim : iterable, optional
            x axis limits
        ylim : iterable,
            y axis limits
        interp_method : `string`
            interpolation technique used in display
        power : `float`
            inverse of power to stretch image by.  E.g. power=2 will plot img ** (1/2)
        show_colorbar : `bool`
            whether to show the colorbar or not.
        fig : `matplotlib.figure.Figure`, optional:
            Figure containing the plot
        ax : `matplotlib.axes.Axis`, optional:
            Axis containing the plot

        Returns
        -------
        fig : `matplotlib.figure.Figure`, optional:
            Figure containing the plot
        ax : `matplotlib.axes.Axis`, optional:
            Axis containing the plot

        """
        from matplotlib import colors

        if self.x is not None:
            extx = [self.x[0], self.x[-1]]
            exty = [self.y[0], self.y[-1]]
            ext = [*extx, *exty]
        else:
            ext = None

        if xlim is not None and ylim is None:
            ylim = xlim

        if xlim and not hasattr(xlim, '__iter__'):
            xlim = (-xlim, xlim)

        if ylim and not hasattr(ylim, '__iter__'):
            ylim = (-ylim, ylim)

        fig, ax = share_fig_ax(fig, ax)
        im = ax.imshow(self.data,
                       extent=ext,
                       origin='lower',
                       aspect='equal',
                       norm=colors.PowerNorm(1/power),
                       clim=(0, 1),
                       cmap='Greys_r',
                       interpolation=interp_method)
        ax.set(xlim=xlim, xlabel=r'$x$ [$\mu m$]',
               ylim=ylim, ylabel=r'$y$ [$\mu m$]')
        if show_colorbar:
            fig.colorbar(im, label=r'Normalized Intensity [a.u.]', ax=ax)
        return fig, ax

    def show_fourier(self, freq_x=None, freq_y=None, interp_method='lanczos', fig=None, ax=None):
        """Display the fourier transform of the image.

        Parameters
        ----------
        interp_method : `string`
            method used to interpolate the data for display.
        freq_x : iterable
            x frequencies to use for convolvable with analytical FT and no data
        freq_y : iterable
            y frequencies to use for convolvable with analytic FT and no data
        fig : `matplotlib.figure.Figure`
            Figure containing the plot
        ax : `matplotlib.axes.Axis`
            Axis containing the plot

        Returns
        -------
        fig : `matplotlib.figure.Figure`
            Figure containing the plot
        ax : `matplotlib.axes.Axis`
            Axis containing the plot

        Notes
        -----
        freq_x and freq_y are unused when the convolvable has a .data field.

        """
        if self.has_analytic_ft:
            if self.data is None:
                if freq_x is None or freq_y is None:
                    raise ValueError('Convolvable has analytic FT and no data, must provide x and y coordinates')
            else:
                lx, ly, ss = len(self.x), len(self.y), self.sample_spacing
                freq_x, freq_y = forward_ft_unit(ss, lx), forward_ft_unit(ss, ly)

            data = self.analytic_ft(freq_x, freq_y)
        else:
            data = abs(m.fftshift(m.fft2(pad2d(self.data, 2))))
            data /= data.max()
            freq_x = forward_ft_unit(self.sample_spacing, self.samples_x)
            freq_y = forward_ft_unit(self.sample_spacing, self.samples_y)

        xmin, xmax = freq_x[0], freq_x[-1]
        ymin, ymax = freq_y[0], freq_y[-1]

        fig, ax = share_fig_ax(fig, ax)
        im = ax.imshow(data,
                       extent=[xmin, xmax, ymin, ymax],
                       cmap='Greys_r',
                       interpolation=interp_method,
                       origin='lower')
        fig.colorbar(im, ax=ax, label='Normalized Spectral Intensity [a.u.]')
        ax.set(xlim=(xmin, xmax), xlabel=r' $\nu_x$ [cy/mm]',
               ylim=(ymin, ymax), ylabel=r' $\nu_y$ [cy/mm]')
        return fig, ax

    def save(self, path, nbits=8):
        """Write the image to a png, jpg, tiff, etc.

        Parameters
        ----------
        path : `string`
            path to write the image to
        nbits : `int`
            number of bits in the output image

        """
        from imageio import imwrite
        if nbits == 8:
            typ = m.uint8
        elif nbits == 16:
            typ = m.uint16
        else:
            raise ValueError('must use either 8 or 16 bpp.')
        dat = (self.data * 2**nbits - 1).astype(typ)
        imwrite(path, dat)

    @staticmethod
    def from_file(path, scale):
        """Read a monochrome 8 bit per pixel file into a new Image instance.

        Parameters
        ----------
        path : `string`
            path to a file
        scale : `float`
            pixel scale, in microns

        Returns
        -------
        `Convolvable`
            a new image object

        """
        from imageio import imread
        imgarr = imread(path)
        s = imgarr.shape
        extx, exty = (s[1] * scale) / 2, (s[0] * scale) / 2
        ux, uy = m.arange(-extx, extx, scale), m.arange(-exty, exty, scale)
        return Convolvable(data=m.flip(imgarr, axis=0).astype(config.precision),
                           x=ux, y=uy, has_analytic_ft=False)

class ConvolutionEngine:
    """An engine to facilitate fine-grained control over convolutions."""
    def __init__(self, c1, c2=None, spatial_finalization=(abs,), Q=2, pad_method='linear_ramp'):
        self.c1 = c1
        self.c2 = c2
        self.spatial_finalization = spatial_finalization
        self.Q = Q
        self.pad_method = pad_method

        self.spatial_x = None
        self.spatial_y = None
        self.spatial_data = None
        self.kspace_x = None
        self.kspace_y = None
        self.kspace_data = None

        self.nsamples_x = None
        self.nsamples_y = None
        self.sample_spacing = None

    def fire(self):
        """Convolve self.c1 and self.c2 with no fuss."""
        try:
            return self.merge_analytics()
        except ValueError:
            self.compute_kspace_units()
            self.compute_kspace_data()
            self.compute_spatial_units()
            self.ifft()
            self.crop_output()
            self.postprocess_spatial()
            return Convolvable(*self.spatial, has_analytic_ft=False)

    def compute_kspace_data(self):
        """Compute the k-space representation of the convolution of c1 and c2."""
        if self.c1.has_analytic_ft:
            # units came directly from c2, pad and FT c1
            c2_pad = pad2d(self.c2.data, self.Q, mode=self.pad_method)
            c2_ft = m.fftshift(m.fft2(m.ifftshift(c2_pad)))
            c1_ft = self.c1.analytic_ft(self.kspace_x, self.kspace_y)
        elif self.c2.has_analytic_ft:
            # units came directly from c1, pad and FT c2
            c1_pad = pad2d(self.c1.data, self.Q, mode=self.pad_method)
            c1_ft = m.fftshift(m.fft2(m.ifftshift(c1_pad)))
            c2_ft = self.c2.analytic_ft(self.kspace_x, self.kspace_y)
        else:
            need_to_interp_c1 = False
            need_to_interp_c2 = False
            # units came from both, need to do some interpolation
            cutoff_c1 = 1 / (2 * self.c1.sample_spacing)
            cutoff_c2 = 1 / (2 * self.c2.sample_spacing)
            cutoff = max(cutoff_c1, cutoff_c2)
            support = max(self.c1.support, self.c2.support)
            if not (self.c1.support == support and cutoff_c1 == cutoff):
                need_to_interp_c1 = True
            if not (self.c2.support == support and cutoff_c2 == cutoff):
                need_to_interp_c2 = True

            def resample_data(self, data, sample_spacing):
                c_freq_x = forward_ft_unit(sample_spacing, data.shape[1])
                c_freq_y = forward_ft_unit(sample_spacing, data.shape[0])
                return resample_2d_complex(data, (c_freq_x, c_freq_y), (self.kspace_x, self.kspace_y), bounds_error=False)

            c1_pad = pad2d(self.c1.data, self.Q, mode=self.pad_method)
            c1_ft = m.fftshift(m.fft2(m.ifftshift(c1_pad)))

            c2_pad = pad2d(self.c2.data, self.Q, mode=self.pad_method)
            c2_ft = m.fftshift(m.fft2(m.ifftshift(c2_pad)))

            if need_to_interp_c1:
                c1_ft = resample_data(self, c1_ft, self.c1.sample_spacing)
            if need_to_interp_c2:
                c2_ft = resample_data(self, c2_ft, self.c2.sample_spacing)

        self.kspace_data = c1_ft * c2_ft
        return self

    def compute_kspace_units(self):
        """Compute the k-space domain of the convolution of c1 and c2."""
        if self.c1.has_analytic_ft:
            support_x, support_y = self.c2.support_x, self.c2.support_y
            sample_spacing = self.c2.sample_spacing
        elif self.c2.has_analytic_ft:
            support_x, support_y = self.c1.support_x, self.c1.support_y
            sample_spacing = self.c1.sample_spacing
        else:
            support_x = max(self.c1.support_x, self.c2.support_x)
            support_y = max(self.c1.support_y, self.c2.support_y)
            sample_spacing = min(self.c1.sample_spacing, self.c2.sample_spacing)

        self.sample_spacing = sample_spacing
        self.nsamples_x = int(m.ceil(((support_x / sample_spacing) + 1) * self.Q))
        self.nsamples_y = int(m.ceil(((support_y / sample_spacing) + 1) * self.Q))
        self.kspace_x = forward_ft_unit(sample_spacing, self.nsamples_x, True)
        self.kspace_y = forward_ft_unit(sample_spacing, self.nsamples_y, True)
        return self

    def compute_spatial_units(self):
        """Compute the spatial domain units of the convolution of c1 and c2."""
        dx = -1 / (2 * self.kspace_x[0])  # [0] is -fs/2, [-1] is slightly below fs/2 for even-length arrays
        dy = -1 / (2 * self.kspace_y[0])
        ny, nx = self.kspace_data.shape
        support_x, support_y = dx * nx, dy * ny
        self.spatial_x = m.linspace(-support_x/2, support_x/2, nx)
        self.spatial_y = m.linspace(-support_y/2, support_y/2, ny)
        return self

    def ifft(self):
        """Take the iFT to compute the spatial representation of the convolution of c1 and c2."""
        self.spatial_data = m.fftshift(m.ifft2(m.ifftshift(self.kspace_data)))
        return self

    def crop_output(self):
        """Crop the output in the spatial domain to remove the padded area."""
        s = self.kspace_data.shape
        npx_x, npx_y = s[1] / self.Q / 2, s[0] / self.Q / 2
        cx_left, cx_right = int(m.ceil(npx_x)), int(m.floor(npx_x))
        cy_top, cy_bottom = int(m.ceil(npx_y)), int(m.floor(npx_y))
        self.spatial_data = self.spatial_data[cy_top:-cy_bottom, cx_left:-cx_right]
        self.spatial_x = self.spatial_x[cx_left:-cx_right]
        self.spatial_y = self.spatial_y[cy_top:-cy_bottom]
        return self

    def postprocess_spatial(self):
        """Post-process the spatial domain."""
        if self.spatial_finalization is not None:
            for func in self.spatial_finalization:
                self.spatial_data = func(self.spatial_data)

    def merge_analytics(self):
        """Merge c1 and c2 if they both have analytic FTs, else raise.

        Raises
        ------
        ValueError
            c1 or c2 does not have an analytic FT.

        """
        if not (self.c1.has_analytic_ft and self.c2.has_analytic_ft):
            raise ValueError('both convolvables must have analytic FTs')
        else:
            c_out = Convolvable(None, None, None, has_analytic_ft=True)
            c_out.s1 = self.c1.copy()
            c_out.s2 = self.c2.copy()

            def aft(self, x, y):
                part1 = self.s1.analytic_ft(x, y)
                part2 = self.s2.analytic_ft(x, y)
                return part1 * part2

            c_out.analytic_ft = aft
            return c_out

    @property
    def spatial(self):
        """Spatial representation, x, y, data."""
        return self.spatial_x, self.spatial_y, self.spatial_data

    @property
    def kspace(self):
        """k-space representation, fx, fy, data."""
        return self.kspace_x, self.kspace_y, self.kspace_data
