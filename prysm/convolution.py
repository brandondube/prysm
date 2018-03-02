"""Defines behavior of convolvable items and a base class to encapsulate that behavior.
"""
from prysm.mathops import fft2, ifft2, fftshift, ifftshift
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
        if self.has_analytic_ft and not other.has_analytic_ft:
            return single_analytical_ft_convolution(other, self)
        elif not self.has_analytic_ft and other.has_analytic_ft:
            return single_analytical_ft_convolution(self, other)
        else:
            raise NotImplementedError('Other convolutional cases not implemented')


class ConvolutionResult(Convolvable):
    """The result of a convolution.

    Subclasses of Convolvable may choose to override conv and add a cast_to
    argument to cast the result back to their own type.

    This is simply a minimal container that contains the data and the grid it is defined on.

    The data is assumed to be centered about the origin.

    """
    def __init__(self, data, unit_x, unit_y):
        """Create a new ConvolutionResult.

        Parameters
        ----------
        data : `numpy.ndarray`
            array of output data
        unit_x : `numpy.ndarray`
            array of x sample locations
        unnit_y : `numpy.ndarray`
            array of y sample locations

        Returns
        -------
        `ConvolutionResult`
            A convolution result

        """
        super().__init__(data, unit_x, unit_y, has_analytic_ft=False)


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
    Image
        A prysm image.

    """
    raise NotImplementedError()


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
    return ConvolutionResult(result, without_analytic.unit_x, without_analytic.unit_y)


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
