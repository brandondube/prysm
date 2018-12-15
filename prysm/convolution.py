"""Defines behavior of convolvable items and a base class to encapsulate that behavior.
"""
from scipy.interpolate import interp2d

import matplotlib as mpl

from prysm import mathops as m
from .conf import config
from .fttools import forward_ft_unit, pad2d
from .util import share_fig_ax


class Convolvable(object):
    """A base class for convolvable objects to inherit from.

    Attributes
    ----------
    data : `numpy.ndarray`
        numerical representation of object
    has_analytic_ft : `bool`
        whether this convolvable has an analytical Fourier transform
    sample_spacing : `float`
        center to center spacing of samples
    unit_x : `numpy.ndarray`
        x-axis unit
    unit_y : `numpy.ndarray`
        y-axis unit

    """
    def __init__(self, data, unit_x, unit_y, has_analytic_ft=False):
        """Create a new Convolvable object.

        Parameters
        ----------
        data : `numpy.ndarray`
            2D ndarray of data
        unit_x : `numpy.ndarray`
            1D ndarray defining x data grid
        unit_y : `numpy.ndarray`
            1D ndarray defining y data grid
        has_analytic_ft : `bool`, optional
            Whether this convolvable overrides self.analytic_ft, and has a known
            analytical fourier tansform

        """
        self.data = data
        self.unit_x = unit_x
        self.unit_y = unit_y
        self.has_analytic_ft = has_analytic_ft
        if data is not None:
            self.samples_y, self.samples_x = data.shape
            self.center_y, self.center_x = int(m.ceil(self.samples_y / 2)), int(m.ceil(self.samples_x / 2))
            self.sample_spacing = unit_x[1] - unit_x[0]
        else:
            self.sample_spacing = 1e99

    @property
    def slice_x(self):
        """Retrieve a slice through the x axis of the PSF.

        Returns
        -------
        self.unit_x : `numpy.ndarray`
            ordinate data
        self.data : `numpy.ndarray`
            coordinate data

        """
        return self.unit_x, self.data[self.center_y, :]

    @property
    def slice_y(self):
        """Retrieve a slice through the y axis of the PSF.

        Returns
        -------
        self.unit_y : `numpy.ndarray`
            ordinate data
        self.data : `numpy.ndarray`
            coordinate data

        """
        return self.unit_y, self.data[:, self.center_x]

    @property
    def shape(self):
        return self.data.shape

    @property
    def support_x(self):
        return self.unit_x[-1] - self.unit_x[0]

    @property
    def support_y(self):
        return self.unit_y[-1] - self.unit_x[0]

    @property
    def support(self):
        return max((self.support_x, self.support_y))

    def plot_slice_xy(self, axlim=20, lw=3, fig=None, ax=None):
        """Create a plot of slices through the X and Y axes of the `PSF`.

        Parameters
        ----------
        axlim : `float` or `int`, optional
            axis limits, in microns
        lw : `float`, optional
            linewidth provided directly to matplotlib
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

        ax.plot(ux, x, label='Slice X', lw=lw)
        ax.plot(uy, y, label='Slice Y', lw=lw)
        ax.set(xlabel=r'Image Plane [$\mu m$]',
               ylabel=label_str,
               xlim=(-axlim, axlim),
               ylim=lims)
        ax.legend(loc='upper right')
        return fig, ax

    def conv(self, other):
        """Convolves this convolvable with another.

        Parameters
        ----------
        other : `Convolvable`
            A convolvable object

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
        return _conv(self, other)

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
        return Convolvable(result, self.unit_x, self.unit_y, False)

    def show(self, xlim=None, ylim=None, interp_method=None, power=1, show_colorbar=True, fig=None, ax=None):
        '''Display the image.

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

        '''
        if self.unit_x is not None:
            extx = [self.unit_x[0], self.unit_x[-1]]
            exty = [self.unit_y[0], self.unit_y[-1]]
            ext = [*extx, *exty]
        else:
            ext = None

        if xlim is not None and ylim is None:
            ylim = xlim

        fig, ax = share_fig_ax(fig, ax)
        im = ax.imshow(self.data,
                       extent=ext,
                       origin='lower',
                       norm=mpl.colors.PowerNorm(1/power),
                       clim=(0, 1),
                       cmap='Greys_r',
                       interpolation=interp_method)
        ax.set(xlim=xlim, xlabel=r'$x$ [$\mu m$]',
               ylim=ylim, ylabel=r'$y$ [$\mu m$]')
        if show_colorbar:
            fig.colorbar(im, label=r'Normalized Intensity [a.u.]', ax=ax)
        return fig, ax

    def show_fourier(self, freq_x=None, freq_y=None, interp_method='lanczos', fig=None, ax=None):
        '''Display the fourier transform of the image.

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

        '''
        if self.has_analytic_ft:
            if self.data is None:
                if freq_x is None or freq_y is None:
                    raise ValueError('Convolvable has analytic FT and no data, must provide x and y coordinates')
            else:
                lx, ly, ss = len(self.unit_x), len(self.unit_y), self.sample_spacing
                freq_x, freq_y = forward_ft_unit(ss, lx), forward_ft_unit(ss, ly)

            data = self.analytic_ft(freq_x, freq_y)
        else:
            data = abs(m.fftshift(m.fft2(pad2d(self.data))))
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
        '''Write the image to a png, jpg, tiff, etc.

        Parameters
        ----------
        path : `string`
            path to write the image to
        nbits : `int`
            number of bits in the output image

        '''
        from imageio import imsave
        if nbits is 8:
            typ = m.unit8
        elif nbits is 16:
            typ = m.uint16
        else:
            raise ValueError('must use either 8 or 16 bpp.')
        dat = (self.data * 2**nbits - 1).astype(typ)
        imsave(path, dat)

    @staticmethod
    def from_file(path, scale):
        '''Read a monochrome 8 bit per pixel file into a new Image instance.

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

        '''
        from imageio import imread
        imgarr = imread(path)
        s = imgarr.shape
        extx, exty = s[0] * scale // 2, s[1] * scale // 2
        ux, uy = m.arange(-extx, exty, scale), m.arange(-exty, exty, scale)
        return Convolvable(data=m.flip(imgarr, axis=0).astype(config.precision),
                           unit_x=ux, unit_y=uy, has_analytic_ft=False)


def _conv(convolvable1, convolvable2):
    if convolvable1.has_analytic_ft and convolvable2.has_analytic_ft:
        return double_analytical_ft_convolution(convolvable1, convolvable2)
    elif convolvable1.has_analytic_ft and not convolvable2.has_analytic_ft:
        return single_analytical_ft_convolution(convolvable2, convolvable1)
    elif not convolvable1.has_analytic_ft and convolvable2.has_analytic_ft:
        return single_analytical_ft_convolution(convolvable1, convolvable2)
    else:
        return pure_numerical_ft_convolution(convolvable1, convolvable2)


def _conv_result_core(data1, data2):
        dat = abs(m.fftshift(m.ifft2(data1 * data2)))
        return dat / dat.max()


def double_analytical_ft_convolution(convolvable1, convolvable2):
    """Convolves two convolvable objects utilizing their analytic fourier transforms.

    Parameters
    ----------
    convolvable1 : `Convolvable`
        A Convolvable object
    convolvable2 : `Convolvable`
        A Convolvable object

    Returns
    -------
    `Convolvable`
        Another convolvable

    """
    spatial_x, spatial_y = _compute_output_grid(convolvable1, convolvable2)
    fourier_x = m.fftfreq(spatial_x.shape[0], spatial_x[1] - spatial_x[0])
    fourier_y = m.fftfreq(spatial_y.shape[0], spatial_y[1] - spatial_y[0])
    c1_part = convolvable1.analytic_ft(fourier_x, fourier_y)
    c2_part = convolvable2.analytic_ft(fourier_x, fourier_y)
    out_data = _conv_result_core(c1_part, c2_part)
    return Convolvable(out_data, spatial_x, spatial_y, has_analytic_ft=False)


def single_analytical_ft_convolution(without_analytic, with_analytic):
    """Convolves two convolvable objects utilizing their analytic fourier transforms.

    Parameters
    ----------
    without_analytic : `Convolvable`
        A Convolvable object which lacks an analytic fourier transform
    with_analytic : `Convolvable`
        A Convolvable object which has an analytic fourier transform

    Returns
    -------
    `ConvolutionResult`
        A convolution result

    """
    fourier_data = m.fftshift(m.fft2(m.fftshift(without_analytic.data)))
    fourier_unit_x = forward_ft_unit(without_analytic.sample_spacing, without_analytic.samples_x)
    fourier_unit_y = forward_ft_unit(without_analytic.sample_spacing, without_analytic.samples_y)
    a_ft = with_analytic.analytic_ft(fourier_unit_x, fourier_unit_y)

    result = _conv_result_core(fourier_data, a_ft)
    return Convolvable(result, without_analytic.unit_x, without_analytic.unit_y, False)


def pure_numerical_ft_convolution(convolvable1, convolvable2):
    """Convolves two convolvable objects utilizing their analytic fourier transforms.

    Parameters
    ----------
    convolvable1 : `Convolvable`
        A Convolvable object which lacks an analytic fourier transform
    convolvable2 : `Convolvable`
        A Convolvable object which lacks an analytic fourier transform

    Returns
    -------
    `ConvolutionResult`
        A convolution result

    """
    # logic tree of convolvable cases with specific implementations
    if (convolvable1.sample_spacing - convolvable2.sample_spacing) < 1e-4:
        s1, s2 = convolvable1.shape, convolvable2.shape
        if s1[0] > s2[0]:
            return _numerical_ft_convolution_core_equalspacing_unequalsamplecount(convolvable1, convolvable2)
        elif s1[0] < s2[0]:
            return _numerical_ft_convolution_core_equalspacing_unequalsamplecount(convolvable2, convolvable1)
        else:
            return _numerical_ft_convolution_core_equalspacing(convolvable1, convolvable2)
    else:
        if convolvable1.sample_spacing < convolvable2.sample_spacing:
            return _numerical_ft_convolution_core_unequalspacing(convolvable1, convolvable2)
        else:
            return _numerical_ft_convolution_core_unequalspacing(convolvable2, convolvable1)


def _numerical_ft_convolution_core_equalspacing(convolvable1, convolvable2):
    # two are identically sampled; just FFT convolve them without modification
    ft1 = m.fftshift(m.fft2(m.fftshift(convolvable1.data)))
    ft2 = m.fftshift(m.fft2(m.fftshift(convolvable2.data)))
    data = _conv_result_core(ft1, ft2)
    return Convolvable(data, convolvable1.unit_x, convolvable1.unit_y, False)


def _numerical_ft_convolution_core_equalspacing_unequalsamplecount(more_samples, less_samples):
    # compute the ordinate axes of the input and output
    in_x = forward_ft_unit(less_samples.sample_spacing, less_samples.shape[0])
    in_y = forward_ft_unit(less_samples.sample_spacing, less_samples.shape[1])
    output_x = forward_ft_unit(more_samples.sample_spacing, more_samples.shape[0])
    output_y = forward_ft_unit(more_samples.sample_spacing, more_samples.shape[1])

    # FFT the less sampled one and map it onto the denser grid
    less_fourier = m.fftshift(m.fft2(m.fftshift(less_samples.data)))
    interpf = interp2d(in_x, in_y, less_fourier, kind='linear')
    resampled_less = interpf(output_x, output_y)

    # FFT convolve the two convolvables
    more_fourier = m.fftshift(m.fft2(m.fftshift(more_samples.data)))
    data = _conv_result_core(resampled_less, more_fourier)
    return Convolvable(data, more_samples.unit_x, more_samples.unit_y, False)


def _numerical_ft_convolution_core_unequalspacing(finer_sampled, coarser_sampled):
    # compute the ordinate axes of the input of each
    in_x_more = forward_ft_unit(finer_sampled.sample_spacing, finer_sampled.shape[0])
    in_y_more = forward_ft_unit(finer_sampled.sample_spacing, finer_sampled.shape[1])

    in_x_less = forward_ft_unit(coarser_sampled.sample_spacing, coarser_sampled.shape[0])
    in_y_less = forward_ft_unit(coarser_sampled.sample_spacing, coarser_sampled.shape[1])

    # fourier-space interpolate the larger bandwidth signal onto the grid defined by the lower
    # bandwidth signal.  This assumes the lower bandwidth signal is Nyquist sampled, which is
    # not necessarily the case.  The accuracy of this method depends on the quality of the input.
    more_fourier = m.fftshift(m.fft2(m.fftshift(finer_sampled.data)))
    interpf = interp2d(in_x_more, in_y_more, more_fourier, kind='linear')
    resampled_more = interpf(in_x_less, in_y_less)

    # FFT the less well sampled input and perform the Fourier based convolution.
    less_fourier = m.fftshift(m.fft2(m.fftshift(coarser_sampled.data)))
    data = _conv_result_core(resampled_more, less_fourier)
    return Convolvable(data, in_x_less, in_y_less, False)


def _compute_output_grid(convolvable1, convolvable2):
    # determine output spacing
    output_spacing = min(convolvable1.sample_spacing, convolvable2.sample_spacing)

    # determine region of output
    output_x_left = min(convolvable1.unit_x[0], convolvable2.unit_x[0])
    output_x_right = max(convolvable1.unit_x[-1], convolvable2.unit_x[-1])
    output_y_left = min(convolvable1.unit_y[0], convolvable2.unit_y[0])
    output_y_right = max(convolvable1.unit_y[-1], convolvable2.unit_y[-1])

    # if region is not an integer multiple of sample spacings, enlarge to make this true
    x_rem = (output_x_right - output_x_left) % output_spacing
    y_rem = (output_y_right - output_y_left) % output_spacing
    if x_rem > 1e-3:
        adj = x_rem / 2
        output_x_left -= adj
        output_x_right += adj
    if y_rem > 1e-3:
        adj = y_rem / 2
        output_y_left -= adj
        output_y_right += adj

    # finally, compute the output window
    samples_x = (output_x_right - output_x_left) // output_spacing
    samples_y = (output_y_right - output_y_left) // output_spacing
    unit_out_x = m.linspace(output_x_left, output_x_right, samples_x)
    unit_out_y = m.linspace(output_y_left, output_y_right, samples_y)
    return unit_out_x, unit_out_y
