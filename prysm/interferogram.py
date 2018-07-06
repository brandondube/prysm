"""tools to analyze interferometric data."""
from prysm.io import read_zygo_dat
from prysm.util import share_fig_ax, pv, rms, Ra
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
        else:
            self.x, self.y = x, y
            self._realxy = True

    @property
    def pv(self):
        return pv(self.phase)

    @property
    def rms(self):
        return rms(self.phase)

    @property
    def Ra(self):
        return Ra(self.phase)

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

    def plot2d(self, cmap='inferno', interp_method='lanczos', fig=None, ax=None):
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
                       interpolation=interp_method)
        cb = fig.colorbar(im, label='Height [nm]', ax=ax, fraction=0.046)
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
