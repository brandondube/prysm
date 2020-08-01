"""Defines behavior of convolvable items and a base class to encapsulate that behavior."""
import types

from .mathops import engine as e
from ._richdata import RichData
from .coordinates import resample_2d_complex
from .conf import config
from .fttools import forward_ft_unit, pad2d


class Convolvable(RichData):
    """A base class for convolvable objects to inherit from."""
    _data_type = 'image'

    def __init__(self, x, y, data, has_analytic_ft=False, labels=None, xy_unit=None, z_unit=None):
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
        labels : `Labels`
            labels to use.  If None, will use config.convolvable_labels

        """
        xy_unit = 'um'
        z_unit = 'adu'
        super().__init__(x=x, y=y, data=data,
                         xy_unit=xy_unit or config.image_xy_unit,
                         z_unit=z_unit or config.image_z_unit,
                         labels=labels or config.convolvable_labels)
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

    def conv(self, other):
        """Convolves this convolvable with another.

        Parameters
        ----------
        other : `Convolvable`
            A convolvable object

        Returns
        -------
        `Convolvable`
            a convolvable object

        Notes
        -----
        If self and other both have analytic Fourier transforms, no math will be done and the aFTs
        are merged directly.

        If only one of self or other has an analytic Fourier transform, the output grid will be
        defined by the object which does not have an analytic Fourier transform.

        If neither has an analytic transform, the output grid will:
        - span max(self.support, other.support)
        - have sample spacing min(self.sample_spacing, other.sample_spacing)

        This ensures the signal remains Nyquist sampled and (probably) doesn't expand beyond
        the extent of the output window.  The latter condition will be violated when two large
        convolvables are convolved.

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
        """Renormalize so that the peak is at a value of unity and the minimum value is zero."""
        self.data -= self.data.min()
        self.data /= self.data.max()
        return self

    def msaa(self, factor=2):
        """Multi-Sample anti-aliasing.

        Perform anti-aliasing by averaging blocks of (factor, factor) pixels
        into a simple value.

        Parameters
        ----------
        factor : `int`, optional
            factor by which to decimate the data

        Returns
        -------
        `Convolvable`
            self

        """
        from .detector import bindown
        x, y, data = self.x, self.y, self.data
        data = bindown(data, factor, factor, 'avg')
        self.data = data
        self.x = x[::factor]
        self.y = y[::factor]
        return self

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
            typ = e.uint8
        elif nbits == 16:
            typ = e.uint16
        else:
            raise ValueError('must use either 8 or 16 bpp.')
        dat = e.flipud((self.data * 2**nbits - 1).astype(typ))
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
        ux, uy = e.arange(-extx, extx, scale), e.arange(-exty, exty, scale)
        return Convolvable(data=e.flip(imgarr, axis=0).astype(config.precision),
                           x=ux, y=uy, has_analytic_ft=False)


class ConvolutionEngine:
    """An engine to facilitate fine-grained control over convolutions."""
    def __init__(self, c1, c2=None, spatial_finalization=(abs,), Q=2, pad_method='linear_ramp'):
        """Create a new ConvolutionEngine.

        This object is used to perform the convolution of two things, the instance should be discarded after doing so.

        Parameters
        ----------
        c1 : `Convolvable`
            the first convolvable
        c2 : `Convolvable, optional`
            the second.  Can be provided later.
        spatial_finalization : `tuple` of `Callable`
            sequence of array friendly functions to call in succession
            on the penultimate result, which is complex
        Q : `float`
            amount of padding applied to the objects before convolving.
            Q=2 is Nyquist, Q=1 is no padding.  Q>2 may improve accuracy.
        pad_method : `str`
            method used to pad the data.  Valid argument to numpy.pad.  Which
            is optimal depends on the data, linear_ramp is rarely bad and often
            among the best.

        """
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
            c2_ft = e.fft.fftshift(e.fft.fft2(e.fft.ifftshift(c2_pad)))
            xx, yy = e.meshgrid(self.kspace_x, self.kspace_y)
            c1_ft = self.c1.analytic_ft(xx, yy)
        elif self.c2.has_analytic_ft:
            # units came directly from c1, pad and FT c2
            c1_pad = pad2d(self.c1.data, self.Q, mode=self.pad_method)
            c1_ft = e.fft.fftshift(e.fft.fft2(e.fft.ifftshift(c1_pad)))
            xx, yy = e.meshgrid(self.kspace_x, self.kspace_y)
            c2_ft = self.c2.analytic_ft(xx, yy)
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
                return resample_2d_complex(data,
                                           (c_freq_x, c_freq_y),
                                           (self.kspace_x, self.kspace_y))

            c1_pad = pad2d(self.c1.data, self.Q, mode=self.pad_method)
            c1_ft = e.fft.fftshift(e.fft.fft2(e.fft.ifftshift(c1_pad)))

            c2_pad = pad2d(self.c2.data, self.Q, mode=self.pad_method)
            c2_ft = e.fft.fftshift(e.fft.fft2(e.fft.ifftshift(c2_pad)))

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
        self.nsamples_x = int(e.floor(round(((support_x / sample_spacing) + 1) * self.Q, 6)))
        self.nsamples_y = int(e.floor(round(((support_y / sample_spacing) + 1) * self.Q, 6)))
        self.kspace_x = forward_ft_unit(sample_spacing, self.nsamples_x, True)
        self.kspace_y = forward_ft_unit(sample_spacing, self.nsamples_y, True)
        return self

    def compute_spatial_units(self):
        """Compute the spatial domain units of the convolution of c1 and c2."""
        dx = -1 / (2 * self.kspace_x[0])  # [0] is -fs/2, [-1] is slightly below fs/2 for even-length arrays
        dy = -1 / (2 * self.kspace_y[0])
        ny, nx = self.kspace_data.shape
        support_x, support_y = dx * nx, dy * ny
        self.spatial_x = e.linspace(-support_x/2, support_x/2, nx)
        self.spatial_y = e.linspace(-support_y/2, support_y/2, ny)
        return self

    def ifft(self):
        """Take the iFT to compute the spatial representation of the convolution of c1 and c2."""
        self.spatial_data = e.fft.fftshift(e.fft.ifft2(e.fft.ifftshift(self.kspace_data)))
        return self

    def crop_output(self):
        """Crop the output in the spatial domain to remove the padded area."""
        s = self.kspace_data.shape
        npx_x, npx_y = s[1] / self.Q / 2, s[0] / self.Q / 2
        cx_left, cx_right = int(e.ceil(npx_x)), int(e.floor(npx_x))
        cy_top, cy_bottom = int(e.ceil(npx_y)), int(e.floor(npx_y))
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

            c_out.analytic_ft = types.MethodType(aft, c_out)
            return c_out

    @property
    def spatial(self):
        """Spatial representation, x, y, data."""
        return self.spatial_x, self.spatial_y, self.spatial_data

    @property
    def kspace(self):
        """k-space representation, fx, fy, data."""
        return self.kspace_x, self.kspace_y, self.kspace_data
