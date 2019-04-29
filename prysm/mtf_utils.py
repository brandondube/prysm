"""Utilities for working with MTF data."""
import operator

from scipy.interpolate import griddata, RegularGridInterpolator as RGI

from .mathops import engine as e
from .util import share_fig_ax
from .io import read_trioptics_mtf_vs_field, read_trioptics_mtfvfvf


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
            plot contours
        interp_method : `string`
            interpolation method used for the plot
        fig : `matplotlib.figure.Figure`, optional:
            Figure to plot inside
        ax : `matplotlib.axes.Axis`, optional:
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
        freq_idx = e.searchsorted(self.freq, freq)

        # if the plot is symmetric, mirror the data
        if symmetric is True:
            dat = e.concatenate((self.data[:, ::-1, freq_idx], self.data[:, :, freq_idx]), axis=1)
            ext_x[0] = ext_x[1] * -1
        else:
            dat = self.data[:, :, freq_idx]

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

    def plot_thrufocus_singlefield(self, field, freqs=(10, 20, 30, 40, 50), _range=100, fig=None, ax=None):
        """Create a plot of Thru-Focus MTF for a single field point.

        Parameters
        ----------
        field : `float`
            which field point to plot, in same units as self.field
        freqs : `iterable`
            frequencies to plot, will be rounded to the closest values present in the self.freq iterable
        _range : `float`
            +/- focus range to plot, symmetric
        fig : `matplotlib.figure.Figure`, optional
            Figure to plot inside
        ax : `matplotlib.axes.Axis`
            Axis to plot inside

        Returns
        -------
        fig : `matplotlib.figure.Figure`, optional
            figure containing the plot
        axis : `matplotlib.axes.Axis`
            axis containing the plot

        """
        field_idx = e.searchsorted(self.field, field)
        freq_idxs = [e.searchsorted(self.freq, f) for f in freqs]
        range_idxs = [e.searchsorted(self.focus, r) for r in (-_range, _range)]
        xaxis_pts = self.focus[range_idxs[0]:range_idxs[1]]

        mtf_arrays = []
        for idx, freq in zip(freq_idxs, freqs):
            data = self.data[range_idxs[0]:range_idxs[1], field_idx, idx]
            mtf_arrays.append(data)

        fig, ax = share_fig_ax(fig, ax)
        for data, freq in zip(mtf_arrays, freqs):
            ax.plot(xaxis_pts, data, label=freq)

        ax.legend(title=r'$\nu$ [cy/mm]')
        ax.set(xlim=(xaxis_pts[0], xaxis_pts[-1]), xlabel=r'Focus [$\mu m$]',
               ylim=(0, 1), ylabel='MTF [Rel. 1.0]')
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
        field : `numpy.ndarray`
            array of field values, mm
        focus : `numpy.ndarray`
            array of focus values, microns

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
            idx_axis = e.searchsorted(self.field, 0)
            idx_freq = abs(self.data[:, idx_axis, :].max(axis=0) - 0.5).argmin(axis=0)
            focus_idx = self.data[:, e.arange(self.data.shape[1]), idx_freq].argmax(axis=0)
            return self.field, self.focus[focus_idx],
        elif algorithm.lower() in ('avg', 'average'):
            if self.freq[0] == 0:
                # if the zero frequency is included, exclude it from our calculations
                avg_idxs = self.data.argmax(axis=0)[:, 1:].mean(axis=1)
            else:
                avg_idxs = self.data.argmax(axis=0).mean(axis=1)

            # account for fractional indexes
            focus_out = avg_idxs.copy()
            for i, idx in enumerate(avg_idxs):
                li, ri = int(e.floor(idx)), int(e.ceil(idx))
                lf, rf = self.focus[li], self.focus[ri]
                diff = rf - lf
                part = idx % 1
                focus_out[i] = lf + diff * part

            return self.field, focus_out
        else:
            raise ValueError('0.5 is only algorithm supported')

    def __arithmatic_bus__(self, other, op):
        """Core checking and return logic for arithmatic operations."""
        if type(other) == type(self):
            # both MTFvFvFs, check alignment of data
            same_x = e.allclose(self.field, other.field)
            same_y = e.allclose(self.focus, other.focus)
            same_freq = e.allclose(self.freq, other.freq)
            if not same_x and same_y and same_freq:
                raise ValueError('x or y coordinates or frequencies mismatch between MTFvFvFs')
            else:
                target = other.data
        elif type(other) in {int, float}:
            target = other
        else:
            raise ValueError('MTFvFvFs can only be added to each other')

        op = getattr(operator, op)
        data = op(self.data, target)
        return MTFvFvF(data, self.focus, self.field, self.freq, self.azimuth)

    def __add__(self, other):
        """Add something to an MTFvFvF."""
        return self.__arithmatic_bus__(other, 'add')

    def __sub__(self, other):
        """Subtract something from an MTFvFvF."""
        return self.__arithmatic_bus__(other, 'sub')

    def __mul__(self, other):
        """Multiply an MTFvFvF by something."""
        return self.__arithmatic_bus__(other, 'mul')

    def __truediv__(self, other):
        """Divide an MTFvFvF by something."""
        return self.__arithmatic_bus__(other, 'truediv')

    def __imul__(self, other):
        """Multiply an MTFvFvF by something in-place."""
        if type(other) not in {int, float}:
            raise ValueError('can only mul by ints and floats')

        self.data *= other
        return self

    def __itruediv__(self, other):
        """Divide an MTFvFvF by something in-place."""
        if type(other) not in {int, float}:
            raise ValueError('can only div by ints and floats')

        self.data /= other
        return self

    @staticmethod
    def from_dataframe(df):
        """Return a pair of MTFvFvF objects for the tangential and one for the sagittal MTF.

        Parameters
        ----------
        df : `pandas.DataFrame`
            a dataframe with columns Focus, Field, Freq, Azimuth, MTF

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
        focus = e.unique(df.Focus.values)
        fields = e.unique(df.Fields.values)
        freqs = e.unique(df.Freq.values)
        d1, d2, d3 = len(focus), len(fields), len(freqs)
        t_mat = T.MTF.values.reshape((d1, d2, d3))
        s_mat = S.MTF.values.reshape((d1, d2, d3))
        t_cube = MTFvFvF(data=t_mat, focus=focus, field=fields, freq=freqs, azimuth='Tan')
        s_cube = MTFvFvF(data=s_mat, focus=focus, field=fields, freq=freqs, azimuth='Sag')
        return t_cube, s_cube

    @staticmethod
    def from_trioptics_file(file_path):
        """Create a new MTFvFvF object from a trioptics file.

        Parameters
        ----------
        file_path : path_like
            path to a file

        Returns
        -------
        `MTFvFvF`
            new MTFvFvF object

        """
        return MTFvFvF(**read_trioptics_mtfvfvf(file_path))


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
    import pandas as pd
    rows = []
    for f, t, s in zip(freqs, tan, sag):
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

    def plot2d(self, freq, show_contours=True,
               cmap='inferno', clim=(0, 1), show_cb=True,
               fig=None, ax=None):
        """Plot the MTF FFD.

        Parameters
        ----------
        freq : `float`
            frequency to plot at
        show_contours : `bool`
            whether to plot contours
        cmap : `str`
            colormap to pass to `imshow`
        clim : `iterable`
            length 2 iterable with lower, upper bounds of colors
        show_cb : `bool`
            whether to show the colorbar or not
        fig : `matplotlib.figure.Figure`, optional
            figure containing the plot
        ax : `matplotlib.axes.Axis`
            axis containing the plot

        Returns
        -------
        fig : `matplotlib.figure.Figure`, optional
            figure containing the plot
        axis : `matplotlib.axes.Axis`
            axis containing the plot

        """
        idx = e.searchsorted(self.freq, freq)
        extx = (self.field_x[0], self.field_x[-1])
        exty = (self.field_y[0], self.field_y[-1])
        ext = [*extx, *exty]
        fig, ax = share_fig_ax(fig, ax)
        im = ax.imshow(self.data[:, :, idx],
                       extent=ext,
                       origin='lower',
                       interpolation='gaussian',
                       cmap=cmap,
                       clim=clim)

        if show_contours is True:
            if clim[0] < 0:
                contours = list(e.arange(clim[0], clim[1] + 0.1, 0.1))
            else:
                contours = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            cs = ax.contour(self.data[:, :, idx], contours, colors='0.15', linewidths=0.75, extent=ext)
            ax.clabel(cs, fmt='%1.1f', rightside_up=True)

        ax.set(xlabel='Image Plane X [mm]', ylabel='Image Plane Y [mm]')
        if show_cb:
            fig.colorbar(im, label=f'MTF @ {freq} cy/mm', ax=ax, fraction=0.046)
        return fig, ax

    def __arithmatic_bus__(self, other, op):
        """Centralized checking logic for arithmatic operations."""
        if type(other) == type(self):
            # both MTFvFvFs, check alignment of data
            same_x = e.allclose(self.field_x, other.field_x)
            same_y = e.allclose(self.field_y, other.field_y)
            same_freq = e.allclose(self.freq, other.freq)
            if not same_x and same_y and same_freq:
                raise ValueError('x or y coordinates or frequencies mismatch between MTFFFDs')
            else:
                target = other.data
        elif type(other) in {int, float}:
            target = other
        else:
            raise ValueError('MTFFFDs can only be added to each other')

        op = getattr(operator, op)
        data = op(self.data, target)
        return MTFvFvF(data, self.field_x, self.field_y, self.freq)

    def __add__(self, other):
        """Add something to an MTF FFD."""
        return self.__arithmatic_bus__(other, 'add')

    def __sub__(self, other):
        """Subtract something from an MTF FFD."""
        return self.__arithmatic_bus__(other, 'sub')

    def __mul__(self, other):
        """Multiply an MTF FFD by something."""
        return self.__arithmatic_bus__(other, 'mul')

    def __truediv__(self, other):
        """Divide an MTF FFD by something."""
        return self.__arithmatic_bus__(other, 'truediv')

    def __imul__(self, other):
        """Multiply an MTF FFD by something in-place."""
        if type(other) not in {int, float}:
            raise ValueError('can only mul by ints and floats')

        self.data *= other
        return self

    def __itruediv__(self, other):
        """Divide an MTF FFD by something in place."""
        if type(other) not in {int, float}:
            raise ValueError('can only div by ints and floats')

        self.data /= other
        return self

    @staticmethod
    def from_trioptics_files(paths, azimuths, upsample=10, ret=('tan', 'sag')):
        """Convert a set of trioptics files to MTF FFD object(s).

        Parameters
        ----------
        paths : path_like
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
        azimuths = e.radians(e.asarray(azimuths, dtype=e.float64))
        freqs, ts, ss = [], [], []
        for path, angle in zip(paths, azimuths):
            d = read_trioptics_mtf_vs_field(path)
            imght, freq, t, s = d['field'], d['freq'], d['tan'], d['sag']
            freqs.append(freq)
            ts.append(t)
            ss.append(s)

        xx, yy, tan, sag = radial_mtf_to_mtfffd_data(ts, ss, imght, azimuths, upsample=10)
        if ret == ('tan', 'sag'):
            return MTFFFD(tan, xx, yy, freq), MTFFFD(sag, xx, yy, freq)
        else:
            raise NotImplementedError('other returns not implemented')

    @staticmethod
    def from_polar_data(tan, sag, fields, azimuths, freqs, upsample=10):
        x, y, t, s = radial_mtf_to_mtfffd_data(tan, sag, fields, azimuths, upsample)
        return MTFFFD(t, x, y, freqs), MTFFFD(s, x, y, freqs)


def radial_mtf_to_mtfffd_data(tan, sag, imagehts, azimuths, upsample):
    """Take radial MTF data and map it to inputs to the MTFFFD constructor.

    Performs upsampling/interpolation in cartesian coordinates

    Parameters
    ----------
    tan : `np.ndarray`
        tangential data
    sag : `np.ndarray`
        sagittal data
    imagehts : `np.ndarray`
        array of image heights
    azimuths : iterable
        azimuths corresponding to the first dimension of the tan/sag arrays
    upsample : `float`
        upsampling factor

    Returns
    -------
    out_x : `np.ndarray`
        x coordinates of the output data
    out_y : `np.ndarray`
        y coordinates of the output data
    tan : `np.ndarray`
        tangential data
    sag : `np.ndarray`
        sagittal data

    """
    azimuths = e.asarray(azimuths)
    imagehts = e.asarray(imagehts)

    if imagehts[0] > imagehts[-1]:
        # distortion profiled, values "reversed"
        # just flip imagehts, since spacing matters and not exact values
        imagehts = imagehts[::-1]
    amin, amax = min(azimuths), max(azimuths)
    imin, imax = min(imagehts), max(imagehts)
    aq = e.linspace(amin, amax, int(len(azimuths) * upsample))
    iq = e.linspace(imin, imax, int(len(imagehts) * 4))  # hard-code 4x linear upsample, change later
    aa, ii = e.meshgrid(aq, iq, indexing='ij')

    # for each frequency, build an interpolating function and upsample
    up_t = e.empty((len(aq), tan.shape[1], len(iq)))
    up_s = e.empty((len(aq), sag.shape[1], len(iq)))
    for idx in range(tan.shape[1]):
        t, s = tan[:, idx, :], sag[:, idx, :]
        interpft = RGI((azimuths, imagehts), t, method='linear')
        interpfs = RGI((azimuths, imagehts), s, method='linear')
        up_t[:, idx, :] = interpft((aa, ii))
        up_s[:, idx, :] = interpfs((aa, ii))

    # compute the locations of the samples on a cartesian grid
    xd, yd = e.outer(e.cos(e.radians(aq)), iq), e.outer(e.sin(e.radians(aq)), iq)
    samples = e.stack([xd.ravel(), yd.ravel()], axis=1)

    # for the output cartesian grid, figure out the x-y coverage and build a regular grid
    absamin = min(abs(azimuths))
    closest_to_90 = azimuths[e.argmin(azimuths-90)]
    xfctr = e.cos(e.radians(absamin))
    yfctr = e.cos(e.radians(closest_to_90))
    xmin, xmax = imin * xfctr, imax * xfctr
    ymin, ymax = imin * yfctr, imax * yfctr
    xq, yq = e.linspace(xmin, xmax, len(iq)), e.linspace(ymin, ymax, len(iq))
    xx, yy = e.meshgrid(xq, yq)

    outt, outs = [], []
    # for each frequency, interpolate onto the cartesian grid
    for idx in range(up_t.shape[1]):
        datt = griddata(samples, up_t[:, idx, :].ravel(), (xx, yy), method='linear')
        dats = griddata(samples, up_s[:, idx, :].ravel(), (xx, yy), method='linear')
        outt.append(datt.reshape(xx.shape))
        outs.append(dats.reshape(xx.shape))

    outt, outs = e.rollaxis(e.asarray(outt), 0, 3), e.rollaxis(e.asarray(outs), 0, 3)
    return xq, yq, outt, outs


def plot_mtf_vs_field(data_dict, fig=None, ax=None):
    """Plot MTF vs Field.

    Parameters
    ----------
    data_dict : `dict`
        dictionary with keys tan, sag, fields, frequencies
    fig : `matplotlib.figure.Figure`, optional
        figure containing the plot
    axis : `matplotlib.axes.Axis`
        axis containing the plot

    Returns
    -------
    fig : `matplotlib.figure.Figure`, optional
        figure containing the plot
    axis : `matplotlib.axes.Axis`
        axis containing the plot

    """
    tan_mtf_array, sag_mtf_array = data_dict['tan'], data_dict['sag']
    fields, frequencies = data_dict['field'], data_dict['freq']
    freqs = _int_check_frequencies(frequencies)

    fig, ax = share_fig_ax(fig, ax)

    for idx in range(tan_mtf_array.shape[0]):
        l, = ax.plot(fields, tan_mtf_array[idx, :], label=freqs[idx])
        ax.plot(fields, sag_mtf_array[idx, :], c=l.get_color(), ls='--')

    ax.legend(title=r'$\nu$ [cy/mm]')
    ax.set(xlim=(0, 14), xlabel='Image Height [mm]',
           ylim=(0, 1), ylabel='MTF [Rel. 1.0]')
    return fig, ax


def _int_check_frequencies(frequencies):
    freqs = []
    for freq in frequencies:
        if freq % 1 == 0:
            freqs.append(int(freq))
        else:
            freqs.append(freq)
    return freqs
