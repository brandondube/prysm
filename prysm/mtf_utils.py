"""Utilities for working with MTF data."""

import numpy as np
import pandas as pd

from scipy.interpolate import CloughTocher2DInterpolator
from prysm.mathops import floor, ceil, sin, cos
from prysm.util import correct_gamma, share_fig_ax
from prysm.io import read_trioptics_mtf_vs_field


class MTFvFvF(object):
    """Abstract object representing a cube of MTF vs Field vs Focus data.

    Attributes
    ----------
    azimuth : `str`
        Azimuth associated with the data
    data : `numpy.ndarray`
        3D array of data in shape (focus, field, freq)
    field : `numpy.ndarray`
        array of fields associated with the field axis of data
    focus : `numpy.ndarray`
        array of focus associated with the focus axis of data
    freq : `numpy.ndarray`
        array of frequencies associated with the frequency axis of data

    """
    def __init__(self, data, focus, field, freq, azimuth):
        """Create a new MTFvFvF object.

        Parameters
        ----------
        data : `numpy.ndarray`
            3D array in the shape (focus,field,freq)
        focus : `iterable`
            1D set of the column units, in microns
        field : `iterable`
            1D set of the row units, in any units
        freq : `iterable`
            1D set of the z axis units, in cy/mm
        azimuth : `string` or `float`
            azimuth this data cube is associated with

        """
        self.data = data
        self.focus = focus
        self.field = field
        self.freq = freq
        self.azimuth = azimuth

    def plot2d(self, freq, symmetric=False, contours=True, interp_method='lanczos', fig=None, ax=None):
        """Create a 2D plot of the cube, an "MTF vs Field vs Focus" plot.

        Parameters
        ----------
        freq : `float`
            frequency to plot, will be rounded to the closest value present in the self.freq iterable
        symmetric : `bool`
            make the plot symmetric by mirroring it about the x-axis origin
        contours : `bool`
            plot contours, yes or no (T/F)
        interp_method : `string`
            interpolation method used for the plot
        fig : `matplotlib.figure.Figure`
            Figure to plot inside
        ax : `matplotlib.axes.Axis`
            Axis to plot inside

        Returns
        -------
        fig : `matplotlib.figure.Figure`
            figure containing the plot
        axis : `matplotlib.axes.Axis`
            axis containing the plot

        """
        ext_x = [self.field[0], self.field[-1]]
        ext_y = [self.focus[0], self.focus[-1]]
        freq_idx = np.searchsorted(self.freq, freq)

        # if the plot is symmetric, mirror the data
        if symmetric is True:
            dat = correct_gamma(
                np.concatenate((
                    self.data[:, ::-1, freq_idx],
                    self.data[:, :, freq_idx]),
                    axis=1))
            ext_x[0] = ext_x[1] * -1
        else:
            dat = correct_gamma(self.data[:, :, freq_idx])

        ext = [ext_x[0], ext_x[1], ext_y[0], ext_y[1]]

        fig, ax = share_fig_ax(fig, ax)
        im = ax.imshow(dat,
                       extent=ext,
                       origin='lower',
                       cmap='inferno',
                       clim=(0, 1),
                       interpolation=interp_method,
                       aspect='auto')

        if contours is True:
            contours = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            cs = ax.contour(dat, contours, colors='0.15', linewidths=0.75, extent=ext)
            ax.clabel(cs, fmt='%1.1f', rightside_up=True)

        fig.colorbar(im, label=f'MTF @ {freq} cy/mm', ax=ax, fraction=0.046)
        ax.set(xlim=(ext_x[0], ext_x[1]), xlabel='Image Height [mm]',
               ylim=(ext_y[0], ext_y[1]), ylabel=r'Focus [$\mu$m]')
        return fig, ax

    def trace_focus(self, algorithm='avg'):
        """Find the focus position in each field.

        This is, in effect, the "field curvature" for this azimuth.

        Parameters
        ----------
        algorithm : `str`
            algorithm to use to trace focus, currently only supports '0.5', see
            notes for a description of this technique

        Returns
        -------
        `numpy.ndarray`
            focal surface sag, in microns, vs field

        Notes
        -----
        Algorithm '0.5' uses the frequency that has its peak closest to 0.5
        on-axis to estimate the focus coresponding to the minimum RMS WFE
        condition.  This is based on the following assumptions:
            - Any combination of third, fifth, and seventh order spherical
                aberration will produce a focus shift that depends on
                frequency, and this dependence can be well fit by an
                equation of the form y(x) = ax^2 + bx + c.  If this is true,
                then the frequency which peaks at 0.5 will be near the
                vertex of the quadratic, which converges to the min RMS WFE
                condition.
            - Coma, while it enhances depth of field, does not shift the
                focus peak.
            - Astigmatism and field curvature are the dominant cause of any
                shift in best focus with field.
            - Chromatic aberrations do not influence the thru-focus MTF peak
                in a way that varies with field.

        Raises
        ------
        ValueError
            if an unsupported algorithm is entered

        """
        if algorithm == '0.5':
            # locate the frequency index on axis
            idx_axis = np.searchsorted(self.field, 0)
            idx_freq = abs(self.data[:, idx_axis, :].max(axis=0) - 0.5).argmin(axis=1)
            focus_idx = self.data[:, np.arange(self.data.shape[1]), idx_freq].argmax(axis=0)
            return self.focus[focus_idx], self.field
        elif algorithm.lower() in ('avg', 'average'):
            if self.freq[0] == 0:
                # if the zero frequency is included, exclude it from our calculations
                avg_idxs = self.data.argmax(axis=0)[:, 1:].mean(axis=1)
            else:
                avg_idxs = self.data.argmax(axis=0).mean(axis=1)

            # account for fractional indexes
            focus_out = avg_idxs.copy()
            for i, idx in enumerate(avg_idxs):
                li, ri = floor(idx), ceil(idx)
                lf, rf = self.focus[li], self.focus[ri]
                diff = rf - lf
                part = idx % 1
                focus_out[i] = lf + diff * part

            return focus_out, self.field
        else:
            raise ValueError('0.5 is only algorithm supported')

    @staticmethod
    def from_dataframe(df):
        """Return a pair of MTFvFvF objects for the tangential and one for the sagittal MTF.

        Parameters
        ----------
        df : `pandas.DataFrame`
            a dataframe

        Returns
        -------
        t_cube : `MTFvFvF`
            tangential MTFvFvF
        s_cube : `MTFvFvF`
            sagittal MTFvFvF

        """
        # copy the dataframe for manipulation
        df = df.copy()
        df.Fields = df.Field.round(4)
        df.Focus = df.Focus.round(6)
        sorted_df = df.sort_values(by=['Focus', 'Field', 'Freq'])
        T = sorted_df[sorted_df.Azimuth == 'Tan']
        S = sorted_df[sorted_df.Azimuth == 'Sag']
        focus = np.unique(df.Focus.as_matrix())
        fields = np.unique(df.Fields.as_matrix())
        freqs = np.unique(df.Freq.as_matrix())
        d1, d2, d3 = len(focus), len(fields), len(freqs)
        t_mat = T.as_matrix.reshape((d1, d2, d3))
        s_mat = S.as_matrix.reshape((d1, d2, d3))
        t_cube = MTFvFvF(data=t_mat, focus=focus, field=fields, freq=freqs, azimuth='Tan')
        s_cube = MTFvFvF(data=s_mat, focus=focus, field=fields, freq=freqs, azimuth='Sag')
        return t_cube, s_cube


def mtf_ts_extractor(mtf, freqs):
    """Extract the T and S MTF from a PSF object.

    Parameters
    ----------
    mtf : `MTF`
        MTF object
    freqs : iterable
        set of frequencies to extract

    Returns
    -------
    tan : `numpy.ndarray`
        array of tangential MTF values
    sag : `numpy.ndarray`
        array of sagittal MTF values

    """
    tan = mtf.exact_tan(freqs)
    sag = mtf.exact_sag(freqs)
    return tan, sag


def mtf_ts_to_dataframe(tan, sag, freqs, field=0, focus=0):
    """Create a Pandas dataframe from tangential and sagittal MTF data.

    Parameters
    ----------
    tan : `numpy.ndarray`
        vector of tangential MTF data
    sag : `numpy.ndarray`
        vector of sagittal MTF data
    freqs : iterable
        vector of spatial frequencies for the data
    field : `float`
        relative field associated with the data
    focus : `float`
        focus offset (um) associated with the data

    Returns
    -------
    pandas dataframe.

    """
    rows = []
    for f, s, t in zip(freqs, tan, sag):
        base_dict = {
            'Field': field,
            'Focus': focus,
            'Freq': f,
        }
        rows.append({**base_dict, **{
            'Azimuth': 'Tan',
            'MTF': t,
        }})
        rows.append({**base_dict, **{
            'Azimuth': 'Sag',
            'MTF': s,
        }})
    return pd.DataFrame(data=rows)


class MTFFFD(object):
    """An MTF Full-Field Display; stores MTF vs Field vs Frequency and supports plotting."""

    def __init__(self, data, field_x, field_y, freq):
        """Create a new MTFFFD object.

        Parameters
        ----------
        data : `numpy.ndarray`
            3D ndarray of data with axes field_x, field_y, freq
        field_x : `numpy.ndarray`
            1D array of x fields
        field_y : `numpy.ndarray`
            1D array of y fields
        freq : `numpy.ndarray`
            1D array of frequencies

        """
        self.data = data
        self.field_x = field_x
        self.field_y = field_y
        self.freq = freq

    def plot2d(self, freq, fig=None, ax=None):
        """Plot the MTF FFD.

        Parameters
        ----------
        freq : `float`
            frequency to plot at
        fig : `matplotlib.figure.Figure`
            figure containing the plot
        axis : `matplotlib.axes.Axis`
            axis containing the plot

        Returns
        -------
        fig : `matplotlib.figure.Figure`
            figure containing the plot
        axis : `matplotlib.axes.Axis`
            axis containing the plot

        """
        idx = np.searchsorted(self.freq, freq)
        extx = (self.field_x[0], self.field_x[-1])
        exty = (self.field_y[0], self.field_y[-1])
        fig, ax = share_fig_ax(fig, ax)
        im = ax.imshow(self.data[:, :, idx],
                       extent=[*extx, *exty],
                       origin='lower',
                       interpolation='lanczos')
        fig.colorbar(im)
        return fig, ax

    @staticmethod
    def from_dataframe(df, azimuth):
        """Create a new MTFFFD from a DataFrame.

        Parameters
        ----------
        df : `pandas.DataFrame`
            a pandas df
        azimuth : `str`
            which azimuth to extract

        Returns
        -------
        `MTFFFD`
            a new MTFFD object

        """
        raise NotImplemented('not yet complete, df schema needs to be designed')
        # return MTFFFD(data=dat, field_x=x, field_y=y, freq=freqs)

    @staticmethod
    def trioptics_files_to_FFDs(*paths, azimuths, ret=('tan', 'sag')):
        """Convert a set of trioptics files to MTF FFD object(s).

        Parameters
        ----------
        *paths : path_like
            paths to trioptics files
        azimuths : iterable of `strs`
            azimuths, one per path
        ret : tuple, optional
            strings representing outputs, {'tan', 'sag'} are the only currently implemented options

        Returns
        -------
        `MTFFFD`
            MTF FFD object

        Raises
        ------
        NotImplemented
            return option is not available

        """
        ret = (r.lower() for r in ret)
        # extract data from files
        azimuths = np.asarray(azimuths, dtype=np.float64)
        freqs, xs, ys, ts, ss = [], [], [], [], []
        for path, angle in zip(paths, azimuths):
            d = read_trioptics_mtf_vs_field(path)
            imght, freq, t, s = d['field'], d['freq'], d['tan'], d['sag']
            x, y = imght * cos(angle), imght * sin(angle)
            freqs.append(freq)
            xs.append(x)
            ys.append(y)
            ts.append(t)
            ss.append(s)

        # convert to arrays and interpolate onto a regular 2D grid via a cubic interpolator
        xarr, yarr, farr = np.asarray(xs), np.asarray(ys), np.asarray(freqs)
        # eq1, eq2, eq3 = np.all(xarr, axis=1), np.all(yarr, axis=1), np.all(farr, axis=1)
        pts = (xarr, yarr)
        val_tan, val_sag = np.asarray(ts), np.asarray(ss)
        interpf_tan = CloughTocher2DInterpolator(pts, val_tan)
        interpf_sag = CloughTocher2DInterpolator(pts, val_sag)
        npts = len(xs[0])
        xmin, xmax, ymin, ymax = xarr.min(), xarr.max(), yarr.min(), yarr.max()
        out_x, out_y = np.linspace(xmin, xmax, npts), np.linspace(ymin, ymax, npts)
        xx, yy = np.meshgrid(out_x, out_y)
        tan = interpf_tan((xx, yy))
        sag = interpf_sag((xx, yy))
        if ret in set([('tan', 'sag'), ('t', 's')]):
            return MTFFFD(tan, out_x, out_y, farr[:, 0]), MTFFFD(sag, out_x, out_y, farr[:, 0])
        else:
            raise NotImplemented('other returns not implemented')
