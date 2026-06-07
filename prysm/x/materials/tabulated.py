"""Tabulated wavelength and temperature material models."""

import warnings

from prysm.mathops import interpolate, np
from prysm.conf import config

from .core import BaseMaterial, MaterialRangeError, MissingKError


class MaterialData:
    """Validated wavelength, n, optional k, and optional uncertainty samples."""

    def __init__(self, wavelengths, n, *, k=None, sigma_n=None, metadata=None):
        self.wavelengths = wavelengths.copy()
        self.n = n.copy()
        self.k = None if k is None else k.copy()
        self.sigma_n = None if sigma_n is None else sigma_n.copy()
        self.metadata = dict(metadata or {})

    @property
    def wavelength_range(self):
        """Return the wavelength sample range in microns."""
        return float(self.wavelengths[0]), float(self.wavelengths[-1])


def _match_query_dtype(value, query):
    dtype = getattr(query, 'dtype', None)
    if dtype is None or not np.issubdtype(dtype, np.floating):
        dtype = np.dtype(config.precision)
    if hasattr(value, 'astype'):
        return value.astype(dtype, copy=False)
    return dtype.type(value)


def _nearest_interp(x, xp, fp):
    idx = np.searchsorted(xp, x, side='left')
    idx = np.clip(idx, 0, len(xp) - 1)
    left = np.clip(idx - 1, 0, len(xp) - 1)
    choose_left = np.abs(x - xp[left]) <= np.abs(x - xp[idx])
    idx = np.where(choose_left, left, idx)
    return fp[idx]


def _linear_interp(x, xp, fp, extrapolate):
    out = _match_query_dtype(np.interp(x, xp, fp), x)
    if not extrapolate or len(xp) < 2:
        return out

    left_slope = (fp[1] - fp[0]) / (xp[1] - xp[0])
    right_slope = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
    left = fp[0] + np.subtract(x, xp[0]) * left_slope
    right = fp[-1] + np.subtract(x, xp[-1]) * right_slope
    out = np.where(np.less(x, xp[0]), left, out)
    out = np.where(np.greater(x, xp[-1]), right, out)
    return out


def _interp1d(x, xp, fp, method, extrapolate):
    if method == 'linear':
        return _linear_interp(x, xp, fp, extrapolate)
    if method == 'nearest':
        return _nearest_interp(x, xp, fp)
    if method == 'log':
        if np.any(fp <= 0):
            raise ValueError('log interpolation requires positive samples')
        return np.exp(_linear_interp(x, xp, np.log(fp), extrapolate))
    if method == 'pchip':
        pchip = interpolate.PchipInterpolator(xp, fp, extrapolate=extrapolate)
        return _match_query_dtype(pchip(x), x)
    raise ValueError("interpolation method must be 'linear', 'nearest', 'pchip', or 'log'")


def _normalize_interp_method(method):
    key = str(method).lower()
    aliases = {
        'interp': 'linear',
        'log_interp': 'log',
    }
    key = aliases.get(key, key)
    if key not in ('linear', 'nearest', 'pchip', 'log'):
        raise ValueError(
            "interpolation method must be 'linear', 'nearest', 'pchip', or 'log'"
        )
    return key


def _validate_axis(values, name):
    """Validate a 1-D coordinate axis: finite, positive, strictly increasing."""
    if not np.all(np.isfinite(values)):
        raise ValueError(f'{name} must contain only finite values')
    if np.any(values <= 0):
        raise ValueError(f'{name} must be positive')
    if values.size > 1 and np.any(np.diff(values) <= 0):
        raise ValueError(f'{name} must be strictly increasing with no duplicates')


def _validate_samples(wavelengths, n, k, sigma_n, sigma_k):
    if wavelengths.ndim != 1:
        raise ValueError('wavelengths must be a 1D array')
    if wavelengths.size == 0:
        raise ValueError('wavelengths must contain at least one value')
    _validate_axis(wavelengths, 'wavelengths')
    if n.shape != wavelengths.shape:
        raise ValueError('n samples must match wavelengths')
    if not np.all(np.isfinite(n)):
        raise ValueError('n samples must contain only finite values')
    if k is not None:
        if k.shape != wavelengths.shape:
            raise ValueError('k samples must match wavelengths')
        if not np.all(np.isfinite(k)):
            raise ValueError('k samples must contain only finite values')
        if np.any(k < 0):
            raise ValueError('k must be nonnegative')
    if sigma_n is not None:
        if sigma_n.shape != wavelengths.shape:
            raise ValueError('sigma_n samples must match wavelengths')
        if not np.all(np.isfinite(sigma_n)):
            raise ValueError('sigma_n samples must contain only finite values')
    if sigma_k is not None:
        if sigma_k.shape != wavelengths.shape:
            raise ValueError('sigma_k samples must match wavelengths')
        if not np.all(np.isfinite(sigma_k)):
            raise ValueError('sigma_k samples must contain only finite values')


class TabulatedMaterial(BaseMaterial):
    """Material with tabulated n(lambda) and optional k(lambda)."""

    def __init__(
        self,
        name,
        wavelengths,
        n,
        *,
        k=None,
        interpolation='linear',
        n_interpolation=None,
        k_interpolation=None,
        sigma_n=None,
        sigma_k=None,
        extrapolate=False,
        method=None,
        k_method=None,
        k_zero_policy='raise',
        **kwargs,
    ):
        missing_k = kwargs.pop('missing_k', 'zero' if k is None else 'raise')
        wavelengths = np.array(wavelengths, dtype=config.precision)
        n = np.array(n, dtype=config.precision)
        k = None if k is None else np.array(k, dtype=config.precision)
        sigma_n = None if sigma_n is None else np.array(sigma_n, dtype=config.precision)
        sigma_k = None if sigma_k is None else np.array(sigma_k, dtype=config.precision)
        _validate_samples(wavelengths, n, k, sigma_n, sigma_k)
        if wavelengths.size < 2 and (method or interpolation) != 'nearest':
            raise ValueError('at least two samples are required for interpolation')
        if method is not None:
            interpolation = method
        if k_method is not None:
            k_interpolation = k_method
        interpolation = _normalize_interp_method(interpolation)
        n_interpolation = _normalize_interp_method(n_interpolation or interpolation)
        k_interpolation = _normalize_interp_method(k_interpolation or interpolation)
        if k_zero_policy not in ('raise', 'linear'):
            raise ValueError("k_zero_policy must be 'raise' or 'linear'")
        if k_interpolation == 'log' and k is not None and np.any(k == 0) \
                and k_zero_policy == 'raise':
            raise ValueError(
                "log interpolation for k requires positive k samples; set "
                "k_zero_policy='linear' to handle zeros explicitly"
            )
        metadata = dict(kwargs.pop('metadata', {}) or {})
        if extrapolate:
            metadata['extrapolate_wavelength'] = True
        metadata.update({
            'method': n_interpolation,
            'k_method': k_interpolation if k is not None else None,
            'extrapolate': bool(extrapolate),
            'missing_k': missing_k,
            'k_zero_policy': k_zero_policy,
        })
        wavelength_range = kwargs.pop(
            'wavelength_range',
            (float(wavelengths[0]), float(wavelengths[-1])),
        )
        super().__init__(
            name,
            wavelength_range=wavelength_range,
            metadata=metadata,
            missing_k=missing_k,
            **kwargs,
        )
        self.wavelengths = wavelengths
        self.n_samples = n
        self.k_samples = k
        self.sigma_n = sigma_n
        self.sigma_k = sigma_k
        self.n_interpolation = n_interpolation
        self.k_interpolation = k_interpolation
        self.method = n_interpolation
        self.k_method = k_interpolation
        self.k_zero_policy = k_zero_policy
        self.extrapolate = extrapolate
        self.data = MaterialData(
            wavelengths,
            n,
            k=k,
            sigma_n=sigma_n,
            metadata=metadata,
        )
        self.fit_report = None

    def n(self, wvl_um, temperature=None):
        """Interpolate real refractive index."""
        self._check_wavelength(wvl_um)
        self._check_temperature(temperature)
        return _interp1d(
            wvl_um,
            self.wavelengths,
            self.n_samples,
            self.n_interpolation,
            self.extrapolate,
        )

    def _check_wavelength(self, wvl):
        if self.metadata.get('extrapolate_wavelength'):
            return
        lo, hi = self.wavelength_range
        outside = (np.less(wvl, lo)) | (np.greater(wvl, hi))
        if np.any(outside):
            raise MaterialRangeError(
                f'wavelength for {self.name} outside material range '
                f'{lo:g} to {hi:g} um'
            )

    def k(self, wvl_um, temperature=None):
        """Interpolate extinction coefficient."""
        self._check_wavelength(wvl_um)
        self._check_temperature(temperature)
        if self.k_samples is None:
            if self.missing_k == 'raise':
                raise MissingKError(f'material {self.name} has no k samples')
            return self._missing_k(wvl_um)
        method = self.k_interpolation
        if method == 'log' and np.any(self.k_samples == 0) \
                and self.k_zero_policy == 'linear':
            method = 'linear'
        return _interp1d(
            wvl_um,
            self.wavelengths,
            self.k_samples,
            method,
            self.extrapolate,
        )


def _coerce_temperature_grid(grid, temperatures, wavelengths, label, layout=None):
    if grid is None:
        return None
    arr = np.array(grid, dtype=config.precision)
    tw = (len(temperatures), len(wavelengths))
    wt = (len(wavelengths), len(temperatures))
    if tw == wt and arr.shape == tw:
        # square grid: the orientation is ambiguous, so honor the layout.
        if layout == ('wavelength', 'temperature'):
            return arr.T
        return arr
    if arr.shape == tw:
        return arr
    if arr.shape == wt:
        return arr.T
    raise ValueError(f'{label} grid must have shape temperature x wavelength')


def _bracket(xp, x, extrapolate):
    """Bracketing indices and interpolation fraction for query x on grid xp.

    The interpolated value is f[lower] + (f[upper] - f[lower]) * fraction.  A
    single-sample axis yields a zero fraction (constant); without extrapolation
    the fraction is clamped to [0, 1] so out-of-range queries hold at the
    endpoint, while extrapolation leaves it unclamped so the end segment extends
    linearly -- matching _linear_interp.
    """
    if xp.shape[0] == 1:
        return 0, 0, x * 0
    idx = np.clip(np.searchsorted(xp, x, side='right'), 1, xp.shape[0] - 1)
    x0 = xp[idx - 1]
    frac = (x - x0) / (xp[idx] - x0)
    if not extrapolate:
        frac = np.clip(frac, 0.0, 1.0)
    return idx - 1, idx, frac


def _interp_grid(wavelengths, temperatures, grid, wvl, temp, extrapolate):
    """Separable bilinear interpolation of a wavelength-temperature grid.

    Fully vectorized over the broadcast (wavelength, temperature) query points:
    no per-point Python loop and no item assignment, so it stays backend-pure
    and avoids per-element host syncs.
    """
    wvl_b, temp_b = np.broadcast_arrays(wvl, temp)
    w = np.reshape(wvl_b, (-1,))
    t = np.reshape(temp_b, (-1,))
    iw0, iw1, fw = _bracket(wavelengths, w, extrapolate)
    it0, it1, ft = _bracket(temperatures, t, extrapolate)
    g0 = grid[it0, iw0] + (grid[it0, iw1] - grid[it0, iw0]) * fw
    g1 = grid[it1, iw0] + (grid[it1, iw1] - grid[it1, iw0]) * fw
    out = g0 + (g1 - g0) * ft
    if hasattr(out, 'astype'):
        out = out.astype(grid.dtype, copy=False)
    return np.reshape(out, wvl_b.shape)


class TemperatureGridMaterial(BaseMaterial):
    """Material with n(lambda, T) samples on a wavelength-temperature grid."""

    def __init__(
        self,
        name,
        wavelengths,
        temperatures,
        n,
        *,
        k=None,
        dn_dlambda=None,
        dn_dT=None,
        sigma_n=None,
        extrapolate=False,
        layout=None,
        **kwargs,
    ):
        missing_k = kwargs.pop('missing_k', 'zero' if k is None else 'raise')
        wavelengths = np.array(wavelengths, dtype=config.precision)
        temperatures = np.array(temperatures, dtype=config.precision)
        if wavelengths.ndim != 1:
            raise ValueError('wavelengths must be a 1D array')
        if temperatures.ndim != 1:
            raise ValueError('temperatures must be a 1D array')
        w_order = np.argsort(wavelengths)
        t_order = np.argsort(temperatures)
        wavelengths = wavelengths[w_order]
        temperatures = temperatures[t_order]
        _validate_axis(wavelengths, 'wavelengths')
        _validate_axis(temperatures, 'temperatures')
        if layout is None and len(wavelengths) == len(temperatures):
            warnings.warn(
                f'{name} grid is square; assuming (temperature, wavelength) '
                "layout. Pass layout=('temperature', 'wavelength') or "
                "('wavelength', 'temperature') to disambiguate.",
                stacklevel=2,
            )
        n_grid = _coerce_temperature_grid(n, temperatures, wavelengths, 'n', layout)
        n_grid = n_grid[t_order][:, w_order]
        k_grid = _coerce_temperature_grid(k, temperatures, wavelengths, 'k', layout)
        if k_grid is not None:
            k_grid = k_grid[t_order][:, w_order]
        dn_dlambda_grid = _coerce_temperature_grid(
            dn_dlambda, temperatures, wavelengths, 'dn_dlambda', layout
        )
        if dn_dlambda_grid is not None:
            dn_dlambda_grid = dn_dlambda_grid[t_order][:, w_order]
        dn_dT_grid = _coerce_temperature_grid(dn_dT, temperatures, wavelengths, 'dn_dT', layout)
        if dn_dT_grid is not None:
            dn_dT_grid = dn_dT_grid[t_order][:, w_order]
        sigma_n_grid = _coerce_temperature_grid(
            sigma_n, temperatures, wavelengths, 'sigma_n', layout
        )
        if sigma_n_grid is not None:
            sigma_n_grid = sigma_n_grid[t_order][:, w_order]

        metadata = dict(kwargs.pop('metadata', {}) or {})
        if extrapolate:
            metadata['extrapolate_wavelength'] = True
            metadata['extrapolate_temperature'] = True
        wavelength_range = kwargs.pop(
            'wavelength_range',
            (float(wavelengths[0]), float(wavelengths[-1])),
        )
        temperature_range = kwargs.pop(
            'temperature_range',
            (float(temperatures[0]), float(temperatures[-1])),
        )
        super().__init__(
            name,
            wavelength_range=wavelength_range,
            temperature_range=temperature_range,
            metadata=metadata,
            missing_k=missing_k,
            **kwargs,
        )
        self.wavelengths = wavelengths
        self.temperatures = temperatures
        self.n_grid = n_grid
        self.k_grid = k_grid
        self.dn_dlambda_grid = dn_dlambda_grid
        self.dn_dT_grid = dn_dT_grid
        self.sigma_n = sigma_n_grid
        self.extrapolate = extrapolate

    def _temperature(self, temperature):
        if temperature is None:
            if len(self.temperatures) == 1:
                return self.temperatures[0]
            raise ValueError(f'temperature is required for {self.name}')
        return temperature

    def n(self, wvl_um, temperature=None):
        """Interpolate n(lambda, T)."""
        temp = self._temperature(temperature)
        self._check_wavelength(wvl_um)
        self._check_temperature(temp)
        return _interp_grid(
            self.wavelengths, self.temperatures, self.n_grid,
            wvl_um, temp, self.extrapolate,
        )

    def k(self, wvl_um, temperature=None):
        """Interpolate k(lambda, T), or apply the missing-k policy."""
        temp = self._temperature(temperature)
        self._check_wavelength(wvl_um)
        self._check_temperature(temp)
        if self.k_grid is None:
            if self.missing_k == 'raise':
                raise MissingKError(f'material {self.name} has no k grid')
            wvl_b, temp_b = np.broadcast_arrays(wvl_um, temp)
            return np.zeros(wvl_b.shape, dtype=self.n_grid.dtype) + temp_b * 0
        return _interp_grid(
            self.wavelengths, self.temperatures, self.k_grid,
            wvl_um, temp, self.extrapolate,
        )

    def dn_dlambda(self, wvl_um, temperature=None):
        """Return measured/interpolated or finite-difference dn/dlambda."""
        if self.dn_dlambda_grid is None:
            return super().dn_dlambda(wvl_um, temperature=temperature)
        temp = self._temperature(temperature)
        self._check_wavelength(wvl_um)
        self._check_temperature(temp)
        return _interp_grid(
            self.wavelengths,
            self.temperatures,
            self.dn_dlambda_grid,
            wvl_um,
            temp,
            self.extrapolate,
        )

    def dn_dT(self, wvl_um, temperature):
        """Return measured/interpolated or finite-difference dn/dT."""
        if self.dn_dT_grid is None:
            return super().dn_dT(wvl_um, temperature)
        self._check_wavelength(wvl_um)
        self._check_temperature(temperature)
        return _interp_grid(
            self.wavelengths,
            self.temperatures,
            self.dn_dT_grid,
            wvl_um,
            temperature,
            self.extrapolate,
        )
