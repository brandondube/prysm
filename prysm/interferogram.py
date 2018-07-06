"""tools to analyze interferometric data."""
from prysm.io import read_zygo_dat
from prysm.util import share_fig_ax, pv, rms, Ra
from prysm import mathops as m

class Interferogram(object):
    """Class containing logic and data for working with interferometric data."""
    def __init__(self, phase, intensity=None, meta=None):
        self.phase = phase
        self.intensity = intensity
        self.meta = meta
        self.x = m.arange(self.phase.shape[1])
        self.y = m.arange(self.phase.shape[0])

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
        plane = fit_plane(self.x, self.y, self.phase)
        self.phase -= plane
        return self

    def remove_piston(self):
        self.phase -= self.phase[m.isfinite(self.phase)].mean()
        return self

    def remove_piston_tiptilt(self):
        self.remove_tiptilt()
        self.remove_piston()
        return self

    def plot2d(self, cmap='inferno', interp_method='lanczos', fig=None, ax=None):
        fig, ax = share_fig_ax(fig, ax)

        im = ax.imshow(self.phase,
                       extent=[self.x[0], self.x[-1], self.y[0], self.y[-1]],
                       cmap=cmap,
                       interpolation=interp_method)
        cb = fig.colorbar(im, label='Height [nm]', ax=ax, fraction=0.046)
        ax.set(xlabel=r'x, [$\mu m$]', ylabel=r'y, [$\mu m$]')
        return fig, ax

    @staticmethod
    def from_zygo_dat(path):
        return Interferogram(**read_zygo_dat(path))



def fit_plane(x, y, z):
    xx, yy = m.meshgrid(x, y)
    pts = m.isfinite(z)
    xx_, yy_ = xx[pts].flatten(), yy[pts].flatten()
    flat = m.ones(xx_.shape)

    coefs = m.lstsq(m.stack([xx_, yy_, flat]).T, z[pts].flatten(), rcond=None)[0]
    plane_fit = coefs[0] * xx + coefs[1] * yy + coefs[2]
    return plane_fit
