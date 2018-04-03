"""Defines behavior of convolvable items and a base class to encapsulate that behavior."""
import numpy as np

from scipy.interpolate import interp2d

from .mathops import fft2, ifft2, fftshift, fftfreq
from .fttools import forward_ft_unit


class Convolvable(object):
    """A base class for convolvable objects to inherit from."""
    def __init__(self, data, unit_x, unit_y, has_analytic_ft=False):
        """Create a new Convolvable object.

        Parameters
        ----------
        data : `numpy.ndarray`
            2D ndarray of data
        unit_x : `numpy.ndarray`
            1D ndarray defining x data grid
        unit_y  : `numpy.ndarray`
            1D ndarray defining y data grid
        has_analytic_ft : `bool`, optional
            Whether this convolvable overrides self.analytic_ft, and has a known
            analytical fourier tansform

        Returns
        -------
        `Convolvable`
            New convolvable object

        """
        self.data = data
        self.unit_x = unit_x
        self.unit_y = unit_y
        self.has_analytic_ft = has_analytic_ft
        if data is not None:
            self.samples_x, self.samples_y = data.shape
            self.center_x, self.center_y = self.samples_x // 2, self.samples_y // 2
            self.sample_spacing = unit_x[1] - unit_x[0]
        else:
            self.sample_spacing = 1e99

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
        if self.has_analytic_ft and other.has_analytic_ft:
            return double_analytical_ft_convolution(self, other)
        elif self.has_analytic_ft and not other.has_analytic_ft:
            return single_analytical_ft_convolution(other, self)
        elif not self.has_analytic_ft and other.has_analytic_ft:
            return single_analytical_ft_convolution(self, other)
        else:
            return pure_numerical_ft_convolution(self, other)


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
    fourier_x = fftfreq(spatial_x.shape[0], spatial_x[1] - spatial_x[0])
    fourier_y = fftfreq(spatial_y.shape[0], spatial_y[1] - spatial_y[0])
    c1_part = convolvable1.analytic_ft(fourier_x, fourier_y)
    c2_part = convolvable2.analytic_ft(fourier_x, fourier_y)
    out_data = abs(fftshift(ifft2(c1_part * c2_part)))
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
    fourier_data = fftshift(fft2(fftshift(without_analytic.data)))
    fourier_unit_x = forward_ft_unit(without_analytic.sample_spacing, without_analytic.samples_x)
    fourier_unit_y = forward_ft_unit(without_analytic.sample_spacing, without_analytic.samples_x)
    a_ft = with_analytic.analytic_ft(fourier_unit_x, fourier_unit_y)
    result = abs(fftshift(ifft2(fourier_data * a_ft)))
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
        s1, s2 = convolvable1.data.shape, convolvable2.data.shape
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
    ft1 = fftshift(fft2(fftshift(convolvable1.data)))
    ft2 = fftshift(fft2(fftshift(convolvable2.data)))
    data = abs(fftshift(ifft2(ft1 * ft2)))
    return Convolvable(data, convolvable1.unit_x, convolvable2.unit_y, False)


def _numerical_ft_convolution_core_equalspacing_unequalsamplecount(more_samples, less_samples):
    # compute the ordinate axes of the input and output
    in_x = forward_ft_unit(less_samples.sample_spacing, less_samples.data.shape[0])
    in_y = forward_ft_unit(less_samples.sample_spacing, less_samples.data.shape[1])
    output_x = forward_ft_unit(more_samples.sample_spacing, more_samples.data.shape[0])
    output_y = forward_ft_unit(more_samples.sample_spacing, more_samples.data.shape[1])

    # FFT the less sampled one and map it onto the denser grid
    less_fourier = fftshift(fft2(fftshift(less_samples.data)))
    interpf = interp2d(in_x, in_y, less_fourier, kind='linear')
    resampled_less = interpf(output_x, output_y)

    # FFT convolve the two convolvables
    more_fourier = fftshift(fft2(fftshift(more_samples.data)))
    data = abs(fftshift(ifft2(resampled_less * more_fourier)))
    return Convolvable(data, more_samples.unit_x, more_samples.unit_y, False)


def _numerical_ft_convolution_core_unequalspacing(finer_sampled, coarser_sampled):
    # compute the ordinate axes of the input of each
    in_x_more = forward_ft_unit(finer_sampled.sample_spacing, finer_sampled.data.shape[0])
    in_y_more = forward_ft_unit(finer_sampled.sample_spacing, finer_sampled.data.shape[1])

    in_x_less = forward_ft_unit(coarser_sampled.sample_spacing, coarser_sampled.data.shape[0])
    in_y_less = forward_ft_unit(coarser_sampled.sample_spacing, coarser_sampled.data.shape[1])

    # fourier-space interpolate the larger bandwidth signal onto the grid defined by the lower
    # bandwidth signal.  This assumes the lower bandwidth signal is Nyquist sampled, which is
    # not necessarily the case.  The accuracy of this method depends on the quality of the input.
    more_fourier = fftshift(fft2(fftshift(finer_sampled.data)))
    interpf = interp2d(in_x_more, in_y_more, more_fourier, kind='linear')
    resampled_more = interpf(in_x_less, in_y_less)

    # FFT the less well sampled input and perform the Fourier based convolution.
    less_fourier = fftshift(fft2(fftshift(coarser_sampled.data)))
    data = abs(fftshift(ifft2(resampled_more * less_fourier)))
    return Convolvable(data, in_x_less, in_y_less, False)


def _compute_output_grid(convolvable1, convolvable2):
    # determine output spacing
    if convolvable1.sample_spacing < convolvable2.sample_spacing:
        output_spacing = convolvable1.sample_spacing
    else:
        output_spacing = convolvable2.sample_spacing

    # determine region of output
    if convolvable1.unit_x[0] < convolvable2.unit_x[0]:
        output_x_left = convolvable1.unit_x[0]
    else:
        output_x_left = convolvable2.unit_x[0]

    if convolvable1.unit_x[-1] > convolvable2.unit_x[-1]:
        output_x_right = convolvable1.unit_x[-1]
    else:
        output_x_right = convolvable2.unit_x[-1]

    if convolvable1.unit_y[0] < convolvable2.unit_y[0]:
        output_y_left = convolvable1.unit_y[0]
    else:
        output_y_left = convolvable2.unit_y[0]

    if convolvable1.unit_y[-1] > convolvable2.unit_y[-1]:
        output_y_right = convolvable1.unit_y[-1]
    else:
        output_y_right = convolvable2.unit_y[-1]

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
    unit_out_x = np.linspace(output_x_left, output_x_right, samples_x)
    unit_out_y = np.linspace(output_y_left, output_y_right, samples_y)
    return unit_out_x, unit_out_y
