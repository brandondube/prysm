"""tools to analyze interferometric data."""
from ._phase import OpticalPhase
from ._zernike import defocus
from .io import read_zygo_dat, read_zygo_datx
from .fttools import forward_ft_unit
from .coordinates import cart_to_polar


from prysm import mathops as m


class Interferogram(OpticalPhase):
    """Class containing logic and data for working with interferometric data."""
    def __init__(self, phase, intensity=None, x=None, y=None, scale='px', meta=None):
        if x is None:  # assume x, y given together
            x = m.arange(phase.shape[1])
            y = m.arange(phase.shape[0])
            scale = 'px'
            self.lateral_res = 1

        super().__init__(unit_x=x, unit_y=y, phase=phase,
                         wavelength=meta.get('wavelength'), phase_unit='nm',
                         spatial_unit='m' if scale != 'px' else scale)

        self.xaxis_label = 'X'
        self.yaxis_label = 'Y'
        self.zaxis_label = 'Height'
        self.intensity = intensity
        self.meta = meta
        if scale != 'px':
            self.change_spatial_unit(to=scale, inplace=True)

    @property
    def dropout_percentage(self):
        return m.count_nonzero(~m.isfinite(self.phase)) / self.phase.size * 100

    def crop(self):
        nans = m.isfinite(self.phase)
        nancols = m.any(nans, axis=0)
        nanrows = m.any(nans, axis=1)

        left, right = nanrows.argmax(), nanrows[::-1].argmax()
        top, bottom = nancols.argmax(), nancols[::-1].argmax()
        self.phase = self.phase[left:-right, top:-bottom]
        self.unit_y, self.unit_x = self.unit_y[left:-right], self.unit_x[top:-bottom]
        return self

    def remove_piston(self):
        """Remove piston from the data by subtracting the mean value."""
        self.phase -= self.phase[m.isfinite(self.phase)].mean()
        return self

    def remove_tiptilt(self):
        """Remove tip/tilt from the data by least squares fitting and subtracting a plane."""
        plane = fit_plane(self.unit_x, self.unit_y, self.phase)
        self.phase -= plane
        return self

    def remove_power(self):
        """Remove power from the data by least squares fitting."""
        sphere = fit_sphere(self.phase)
        self.phase -= sphere
        return self

    def remove_piston_tiptilt(self):
        """Remove piston/tip/tilt from the data, see remove_tiptilt and remove_piston."""
        self.remove_tiptilt()
        self.remove_piston()
        return self

    def mask(self, mask):
        """Apply a mask to the data, expects an array of zeros (remove) and ones (keep)

        Parameters
        ----------
        mask : TYPE
            Description
        """
        hitpts = mask == 0
        self.phase[hitpts] = m.nan
        self.intensity[hitpts] = m.nan

    def bandreject(self, wllow, wlhigh):
        """Apply a band-rejection filter to the phase (height) data.

        Parameters
        ----------
        wllow : `float`
            low wavelength (spatial period), units of self.scale
        wlhigh : `float`
            high wavelength (spatial period), units of self.scale

        Returns
        -------
        `Interferogram`
            in-place modified instance of self
        """
        new_phase = bandreject_filter(self.phase, self.sample_spacing, wllow, wlhigh)
        new_phase[~m.isfinite(self.phase)] = m.nan
        self.phase = new_phase
        return self

    @staticmethod
    def from_zygo_dat(path, multi_intensity_action='first', scale='um'):
        """Create a new interferogram from a zygo dat file.

        Parameters
        ----------
        path : path_like
            path to a zygo dat file
        multi_intensity_action : str, optional
            see `io.read_zygo_dat`
        scale : `str`, optional, {'um', 'mm'}
            what xy scale to label the data with, microns or mm

        Returns
        -------
        `Interferogram`
            new Interferogram instance

        """
        if str(path).endswith('datx'):
            # datx file, use datx reader
            zydat = read_zygo_datx(path)
            res = zydat['meta']['Lateral Resolution']
        else:
            # dat file, use dat file reader
            zydat = read_zygo_dat(path, multi_intensity_action=multi_intensity_action)
            res = zydat['meta']['lateral_resolution']  # meters

        phase = zydat['phase']
        if res == 0.0:
            res = 1
            scale = 'px'
        return Interferogram(phase=phase, intensity=zydat['intensity'],
                             x=m.arange(phase.shape[1]) * res, y=m.arange(phase.shape[0]) * res,
                             scale=scale.lower(), meta=zydat['meta'])


def fit_plane(x, y, z):
    xx, yy = m.meshgrid(x, y)
    pts = m.isfinite(z)
    xx_, yy_ = xx[pts].flatten(), yy[pts].flatten()
    flat = m.ones(xx_.shape)

    coefs = m.lstsq(m.stack([xx_, yy_, flat]).T, z[pts].flatten(), rcond=None)[0]
    plane_fit = coefs[0] * xx + coefs[1] * yy + coefs[2]
    return plane_fit


def fit_sphere(z):
    x, y = m.linspace(-1, 1, z.shape[1]), m.linspace(-1, 1, z.shape[0])
    xx, yy = m.meshgrid(x, y)
    pts = m.isfinite(z)
    xx_, yy_ = xx[pts].flatten(), yy[pts].flatten()
    rho, phi = cart_to_polar(xx_, yy_)
    focus = defocus(rho, phi)

    coefs = m.lstsq(m.stack([focus, m.ones(focus.shape)]).T, z[pts].flatten(), rcond=None)[0]
    rho, phi = cart_to_polar(xx, yy)
    sphere = defocus(rho, phi) * coefs[0]
    return sphere


def bandreject_filter(array, sample_spacing, wllow, wlhigh):
    sy, sx = array.shape

    # compute the bandpass in sample coordinates
    ux, uy = forward_ft_unit(sample_spacing, sx), forward_ft_unit(sample_spacing, sy)
    fhigh, flow = 1/wllow, 1/wlhigh

    # make an ordinate array in frequency space and use it to make a mask
    uxx, uyy = m.meshgrid(ux, uy)
    highpass = ((uxx < -fhigh) | (uxx > fhigh)) | ((uyy < -fhigh) | (uyy > fhigh))
    lowpass = ((uxx > -flow) & (uxx < flow)) & ((uyy > -flow) & (uyy < flow))
    mask = highpass | lowpass

    # adjust NaNs and FFT
    work = array.copy()
    work[~m.isfinite(work)] = 0
    fourier = m.fftshift(m.fft2(m.ifftshift(work)))
    fourier[mask] = 0
    out = m.fftshift(m.ifft2(m.ifftshift(fourier)))
    return out.real
