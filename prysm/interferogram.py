"""tools to analyze interferometric data."""
from prysm.io import read_zygo_dat
from prysm.util import share_fig_ax, pv, rms, Ra
from prysm.fttools import forward_ft_unit
from prysm import mathops as m


class Interferogram(object):
    labels = {
        True: (r'x [$\mu m$]', r'y [$\mu m$]'),
        False: ('x [px]', 'y [px]')
    }
    """Class containing logic and data for working with interferometric data."""
    def __init__(self, phase, intensity=None, x=None, y=None, meta=None):
        self.phase = phase
        self.intensity = intensity
        self.meta = meta
        if x is None:  # assume x, y given together
            self.x = m.arange(phase.shape[1])
            self.y = m.arange(phase.shape[0])
            self._realxy = False
            self.lateral_res = None
        else:
            self.x, self.y = x, y
            self._realxy = True
            self.sample_spacing = x[1] - x[0]

    @property
    def pv(self):
        return pv(self.phase)

    @property
    def rms(self):
        return rms(self.phase)

    @property
    def Ra(self):
        return Ra(self.phase)

    @property
    def dropout_percentage(self):
        return m.count_nonzero(~m.isfinite(self.phase)) / self.phase.size * 100

    def remove_tiptilt(self):
        """Remove tip/tilt from the data by least squares fitting and subtracting a plane."""
        plane = fit_plane(self.x, self.y, self.phase)
        self.phase -= plane
        return self

    def remove_piston(self):
        """Remove piston from the data by subtracting the mean value."""
        self.phase -= self.phase[m.isfinite(self.phase)].mean()
        return self

    def remove_piston_tiptilt(self):
        """Remove piston/tip/tilt from the data, see remove_tiptilt and remove_piston."""
        self.remove_tiptilt()
        self.remove_piston()
        return self

    def bandreject(self, wllow, wlhigh):
        new_phase = bandreject_filter(self.phase, self.sample_spacing, wllow, wlhigh)
        new_phase[~m.isfinite(self.phase)] = m.nan
        self.phase = new_phase
        return self

    def plot2d(self, cmap='inferno', clim=(None, None), interp_method='lanczos', fig=None, ax=None):
        """Plot the data in 2D.

        Parameters
        ----------
        cmap : `str`
            colormap to use, passed directly to matplotlib
        interp_method : `str`
            interpolation method to use, passed directly to matplotlib
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

        """
        fig, ax = share_fig_ax(fig, ax)

        im = ax.imshow(self.phase,
                       extent=[self.x[0], self.x[-1], self.y[0], self.y[-1]],
                       cmap=cmap,
                       clim=clim,
                       interpolation=interp_method)
        fig.colorbar(im, label='Height [nm]', ax=ax, fraction=0.046)
        xlab, ylab = self.__class__.labels[self._realxy]
        ax.set(xlabel=xlab, ylabel=ylab)
        return fig, ax

    @staticmethod
    def from_zygo_dat(path):
        zydat = read_zygo_dat(path)
        res = zydat['meta']['lateral_resolution'] * 1e6  # m to um
        phase = zydat['phase']
        return Interferogram(phase=phase, intensity=zydat['intensity'], meta=zydat['meta'],
                             x=m.arange(phase.shape[1]) * res, y=m.arange(phase.shape[0]) * res)


def fit_plane(x, y, z):
    xx, yy = m.meshgrid(x, y)
    pts = m.isfinite(z)
    xx_, yy_ = xx[pts].flatten(), yy[pts].flatten()
    flat = m.ones(xx_.shape)

    coefs = m.lstsq(m.stack([xx_, yy_, flat]).T, z[pts].flatten(), rcond=None)[0]
    plane_fit = coefs[0] * xx + coefs[1] * yy + coefs[2]
    return plane_fit


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
