"""Defines behavior of convolvable items and a base class to encapsulate that behavior.
"""
import numpy as np

from prysm.mathops import fft2, ifft2, fftshift, ifftshift, fftfreq
from prysm.fttools import forward_ft_unit


class Convolvable(object):
    """A base class for convolvable objects to inherit from.
    """
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
            New convolvable object.

        """
        self.data = data
        self.unit_x = unit_x
        self.unit_y = unit_y
        self.has_analytic_ft = has_analytic_ft
        self.sample_spacing = unit_x[1] - unit_x[0]
        if data is not None:
            self.samples_x, self.samples_y = data.shape
            self.center_x, self.center_y = self.samples_x // 2, self.samples_y // 2

    def conv(self, other):
        """Convolves this convolvable with another.

        Parameters
        ----------
        other : Convolvable
            A convolvable object.

        Returns
        -------
        `ConvolutionResult`
            A prysm image.

        Notes
        -----
        The algoithm works according to the following cases:
            1.  Both self and other have analytical fourier transforms:
                - The analytic forms will be used to compute the output directly.
                - The output sample spacing will be the finer of the two inputs.
                - The output window will cover the same extent as the "wider"
                  input.
                - This may mean the output array is not of the same size as
                  either input.
                - An input which contains a sample at (0,0) may produce an output
                  without a sample at (0,0) if the input samplings are not ideal.
                  To ensure this does not happen if it is undesireable, ensure
                  the inputs are computed over identical grids containing 0 to
                  begin with.
            2.  One of self and other have analytical fourier transforms:
                - The input which does NOT have an analytical fourier transform
                  will define the output grid.
                - The available analytic FT will be used to do the convolution
                  in fourier space.
            3.  Neither input has an analytic fourier transform:
                3.1, the two convolvables have the same sample spacing to within
                     a numerical precision of 0.1 nm:
                    - the fourier transform of both will be taken.  If one has
                      fewer samples, it will be upsampled in Fourier space
                3.2, the two convolvables have different sample spacing:
                    - The fourier transform of both inputs will be taken.  That
                      which has a lower nyquist frequency will have a linear
                      taper applied between its final value and 0 to extent its
                      frequency range to that of the finer grid.  Interpolation
                      will also be used to match the sample points in fourier
                      space.

        The subroutines have the following properties with regard to accuracy:
            1.  Computes a perfect numerical representation of the continuous
                output.
            2.  If the input that does not have an analytic FT is unaliased,
                computes a perfect numerical representation of the continuous
                output.  If it does not, the input aliasing limits the output.
            3.  Accuracy of computation is dependent on how much energy is
                present at nyquist in the worse-sampled input.

        """
        if self.has_analytic_ft and other.has_analytic_ft:
            return double_analytical_ft_convolution(self, other)
        elif self.has_analytic_ft and not other.has_analytic_ft:
            return single_analytical_ft_convolution(other, self)
        elif not self.has_analytic_ft and other.has_analytic_ft:
            return single_analytical_ft_convolution(self, other)
        else:
            raise NotImplementedError('Other convolutional cases not implemented')


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
    fourier_y = fftfreq(spatial_y.shape[0], spatial_y[0] - spatial_y[0])
    gridx, gridy = np.meshgrid(fourier_x, fourier_y)
    c1_part = convolvable1.analytic_ft(gridx, gridy)
    c2_part = convolvable2.analytic_ft(gridx, gridy)
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
    fourier_data = fftshift(fft2(ifftshift(without_analytic.data)))
    fourier_unit_x = forward_ft_unit(without_analytic.sample_spacing, without_analytic.samples_x)
    fourier_unit_y = forward_ft_unit(without_analytic.sample_spacing, without_analytic.samples_x)
    a_ft = with_analytic.analytic_ft(fourier_unit_x, fourier_unit_y)
    result = abs(fftshift(ifft2(fourier_data * a_ft)))
    return Convolvable(result, without_analytic.unit_x, without_analytic.unit_y, False)


def pure_numerical_ft_convolution(convolvable1, convolvable2):
    """Convolves two convolvable objects utilizing their analytic fourier transforms.

    Parameters
    ----------
    convolvable1 : Convolvable
        A Convolvable object which lacks an analytic fourier transform
    convolvable2 : Convolvable
        A Convolvable object which lacks an analytic fourier transform

    Returns
    -------
    `ConvolutionResult`
        A convolution result

    """
    raise NotImplementedError()


def _compute_output_grid(convolvable1, convolvable2):
    if convolvable1.sample_spacing < convolvable2.sample_spacing:
        output_spacing = convolvable1.sample_spacing
    else:
        output_spacing = convolvable2.sample_spacing

    if convolvable1.unit_x[0] < convolvable2.unit_x[0]:
        output_x_left = convolvable1.unit_x[0]
    else:
        output_x_right = convolvable2.unit_x[0]

    if convolvable1.unit_y[0] < convolvable2.unit_y[0]:
        output_y_left = convolvable1.unit_y[0]
    else:
        output_y_right = convolvable2.unit_y[0]

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

    samples_x = (output_x_right - output_x_left) // output_spacing
    samples_y = (output_y_right - output_y_left) // output_spacing
    unit_out_x = np.linspace(output_x_left, output_x_right, samples_x)
    unit_out_y = np.linspace(output_y_left, output_y_right, samples_y)
    return unit_out_x, unit_out_y
