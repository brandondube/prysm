"""Utilities for working with MTF data."""
import operator

from .mathops import np, interpolate
from .plotting import share_fig_ax
from .io import read_trioptics_mtf_vs_field, read_trioptics_mtfvfvf


class MTFvFvF(object):
    """Abstract object representing a cube of MTF vs Field vs Focus data.

    Attributes
    ----------
    azimuth : str
        Azimuth associated with the data
    data : numpy.ndarray
        3D array of data in shape (focus, field, freq)
    field : numpy.ndarray
        array of fields associated with the field axis of data
    focus : numpy.ndarray
        array of focus associated with the focus axis of data
    freq : numpy.ndarray
        array of frequencies associated with the frequency axis of data

    """
    def __init__(self, data, focus, field, freq, azimuth):
        """Create a new MTFvFvF object.

        Parameters
        ----------
        data : numpy.ndarray
            3D array in the shape (focus,field,freq)
        focus : iterable
            1D set of the column units, in microns
        field : iterable
            1D set of the row units, in any units
        freq : iterable
            1D set of the z axis units, in cy/mm
        azimuth : string or float
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
        freq : float
            frequency to plot, will be rounded to the closest value present in the self.freq iterable
        symmetric : bool
            make the plot symmetric by mirroring it about the x-axis origin
        contours : bool
            plot contours
        interp_method : string
            interpolation method used for the plot
        fig : matplotlib.figure.Figure, optional:
            Figure to plot inside
        ax : matplotlib.axes.Axis, optional:
            Axis to plot inside

        Returns
        -------
        fig : matplotlib.figure.Figure
            figure containing the plot
        axis : matplotlib.axes.Axis
            axis containing the plot

        """
        ext_x = [self.field[0], self.field[-1]]
        ext_y = [self.focus[0], self.focus[-1]]
        freq_idx = np.searchsorted(self.freq, freq)

        # if the plot is symmetric, mirror the data
        if symmetric is True:
            dat = np.concatenate((self.data[:, ::-1, freq_idx], self.data[:, :, freq_idx]), axis=1)
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
        field : float
            which field point to plot, in same units as self.field
        freqs : iterable
            frequencies to plot, will be rounded to the closest values present in the self.freq iterable
        _range : float
            +/- focus range to plot, symmetric
        fig : matplotlib.figure.Figure, optional
            Figure to plot inside
        ax : matplotlib.axes.Axis
            Axis to plot inside

        Returns
        -------
        fig : matplotlib.figure.Figure, optional
            figure containing the plot
        axis : matplotlib.axes.Axis
            axis containing the plot

        """
        field_idx = np.searchsorted(self.field, field)
        freq_idxs = [np.searchsorted(self.freq, f) for f in freqs]
        range_idxs = [np.searchsorted(self.focus, r) for r in (-_range, _range)]
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
        algorithm : str
            algorithm to use to trace focus, currently only supports '0.5', see
            notes for a description of this technique

        Returns
        -------
        field : numpy.ndarray
            array of field values, mm
        focus : numpy.ndarray
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
            idx_axis = np.searchsorted(self.field, 0)
            idx_freq = abs(self.data[:, idx_axis, :].max(axis=0) - 0.5).argmin(axis=0)
            focus_idx = self.data[:, np.arange(self.data.shape[1]), idx_freq].argmax(axis=0)
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
                li, ri = int(np.floor(idx)), int(np.ceil(idx))
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
            same_x = np.allclose(self.field, other.field)
            same_y = np.allclose(self.focus, other.focus)
            same_freq = np.allclose(self.freq, other.freq)
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
        df : pandas.DataFrame
            a dataframe with columns Focus, Field, Freq, Azimuth, MTF

        Returns
        -------
        t_cube : MTFvFvF
            tangential MTFvFvF
        s_cube : MTFvFvF
            sagittal MTFvFvF

        """
        # copy the dataframe for manipulation
        df = df.copy()
        df['Fields'] = df.Field.round(4)
        df['Focus'] = df.Focus.round(6)
        sorted_df = df.sort_values(by=['Focus', 'Field', 'Freq'])
        T = sorted_df[sorted_df.Azimuth == 'Tan']
        S = sorted_df[sorted_df.Azimuth == 'Sag']
        focus = np.unique(df.Focus.values)
        fields = np.unique(df.Fields.values)
        freqs = np.unique(df.Freq.values)
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
        MTFvFvF
            new MTFvFvF object

        """
        return MTFvFvF(**read_trioptics_mtfvfvf(file_path))


def plot_mtf_vs_field(data_dict, fig=None, ax=None, labels=('MTF', 'Freq [lp/mm]', 'Field [mm]', 'Az'), palette=None):
    """Plot MTF vs Field.

    Parameters
    ----------
    data_dict : dict
        dictionary with keys tan, sag, fields, freq
    fig : matplotlib.figure.Figure, optional
        figure containing the plot
    axis : matplotlib.axes.Axis
        axis containing the plot

    Returns
    -------
    fig : matplotlib.figure.Figure, optional
        figure containing the plot
    axis : matplotlib.axes.Axis
        axis containing the plot

    """
    import pandas as pd
    import seaborn as sns

    if palette is None:
        palette = 'tab10'

    tan = data_dict['tan']
    sag = data_dict['sag']
    freqs = _int_check_frequencies(data_dict['freq'])
    fields = data_dict['field']
    # tan, sag have indices of [freq][field]
    proto_df = []
    for i, freq in enumerate(freqs):
        for j, field in enumerate(fields):
            local_t = (tan[i][j], freq, field, 'tan')
            local_s = (sag[i][j], freq, field, 'sag')
            proto_df.append(local_t)
            proto_df.append(local_s)

    df = pd.DataFrame(data=proto_df, columns=labels)

    fig, ax = share_fig_ax(fig, ax)

    ax = sns.lineplot(x=labels[2], y=labels[0], hue=labels[1], style=labels[3], data=df, palette=palette, legend='full')
    ax.set(ylim=(0, 1))
    return fig, ax


def _int_check_frequencies(frequencies):
    freqs = []
    for freq in frequencies:
        if freq % 1 == 0:
            freqs.append(int(freq))
        else:
            freqs.append(freq)
    return freqs
