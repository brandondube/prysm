"""Coefficient-backed materials fitted from wavelength samples."""

from dataclasses import dataclass

from prysm.mathops import np, optimize

from .core import BaseMaterial, MaterialRangeError
from .tabulated import MaterialData, TabulatedMaterial


@dataclass(frozen=True)
class FitReport:
    """Diagnostics from fitting a dispersion model to measured samples."""

    model: str
    coefficients: dict
    residuals: np.ndarray
    max_abs_error: float
    rms_error: float
    sample_count: int
    parameter_count: int
    degrees_of_freedom: int
    wavelength_range: tuple
    condition_number: float
    warnings: tuple
    success: bool = True
    message: str = ''


def _normalize_model(model):
    key = str(model).lower()
    if key not in ('constant', 'cauchy', 'sellmeier1', 'schott'):
        raise ValueError(
            f'unknown fit model {model!r}; expected constant, cauchy, '
            'sellmeier1, or schott'
        )
    return key


def _model_terms(model, terms, sample_count):
    if model == 'constant':
        return 1
    if model == 'cauchy':
        if sample_count == 1:
            raise ValueError('one sample supports only the constant fit model')
        return 2 if terms is None else int(terms)
    if model == 'sellmeier1':
        return 1 if terms is None else int(terms)
    if model == 'schott':
        return 6
    raise ValueError(model)


def _parameter_names(model, terms):
    if model == 'constant':
        return ('n0',)
    if model == 'cauchy':
        return tuple(f'A{i}' for i in range(terms))
    if model == 'sellmeier1':
        b = tuple(f'B{i}' for i in range(terms))
        c = tuple(f'C{i}' for i in range(terms))
        return b + c
    if model == 'schott':
        return tuple(f'c{i}' for i in range(6))
    raise ValueError(model)


def _cauchy_design(wvl, terms):
    return np.column_stack([wvl ** (-2 * i) for i in range(terms)])


def _schott_design(wvl):
    w2 = wvl * wvl
    return np.column_stack([
        np.ones_like(wvl),
        w2,
        1 / w2,
        1 / w2 ** 2,
        1 / w2 ** 3,
        1 / w2 ** 4,
    ])


def _sellmeier1_eval(wvl, coeffs, terms):
    b = coeffs[:terms]
    c = coeffs[terms:]
    w2 = wvl * wvl
    n2 = 1.0 + wvl * 0
    for bi, ci in zip(b, c):
        n2 = n2 + bi * w2 / (w2 - ci)
    return np.sqrt(n2)


def _evaluate_fit_model(model, coeffs, wvl, terms):
    """Evaluate a fitted dispersion model, backend-pure and shape-preserving.

    Written as direct sums (not design-matrix @ coeffs) so a scalar input
    yields a scalar and the consumption path stays differentiable on non-numpy
    backends; the design-matrix helpers are reserved for the fit itself.
    """
    if model == 'constant':
        return coeffs[0] + wvl * 0
    if model == 'cauchy':
        out = coeffs[0] + wvl * 0
        for i in range(1, terms):
            out = out + coeffs[i] * wvl ** (-2 * i)
        return out
    if model == 'sellmeier1':
        return _sellmeier1_eval(wvl, coeffs, terms)
    if model == 'schott':
        w2 = wvl * wvl
        n2 = (
            coeffs[0]
            + coeffs[1] * w2
            + coeffs[2] / w2
            + coeffs[3] / w2 ** 2
            + coeffs[4] / w2 ** 3
            + coeffs[5] / w2 ** 4
        )
        return np.sqrt(n2)
    raise ValueError(model)


def _normalize_bounds(bounds, n_params):
    if bounds is None:
        return None
    lo, hi = bounds
    lo = np.broadcast_to(np.asarray(lo, dtype=float), (n_params,)).copy()
    hi = np.broadcast_to(np.asarray(hi, dtype=float), (n_params,)).copy()
    if np.any(lo > hi):
        raise ValueError('lower bounds must not exceed upper bounds')
    return lo, hi


def _weighted_design(A, y, sigma):
    if sigma is None:
        return A, y
    weights = 1 / sigma
    return A * weights[:, None], y * weights


def _linear_fit(A, y, *, sigma=None, bounds=None):
    A_w, y_w = _weighted_design(A, y, sigma)
    if bounds is None:
        coeffs, _, rank, svals = np.linalg.lstsq(A_w, y_w, rcond=None)
        return coeffs, rank, svals, 'linear least squares'
    bounds = _normalize_bounds(bounds, A.shape[1])
    result = optimize.lsq_linear(A_w, y_w, bounds=bounds)
    svals = np.linalg.svd(A_w, compute_uv=False)
    rank = int(np.linalg.matrix_rank(A_w))
    if not result.success:
        raise ValueError(f'bounded linear fit failed: {result.message}')
    return result.x, rank, svals, result.message


def _sellmeier_initial(n, terms):
    strength = max(float(np.mean(n) ** 2 - 1), 0.1)
    b = np.full(terms, strength / terms, dtype=float)
    c = 0.01 * (np.arange(terms, dtype=float) + 1)
    return np.concatenate([b, c])


def _fit_sellmeier1(data, terms, *, bounds=None, initial=None):
    n_params = len(_parameter_names('sellmeier1', terms))
    if initial is None:
        initial = _sellmeier_initial(data.n, terms)
    else:
        initial = np.asarray(initial, dtype=float)
    if initial.shape != (n_params,):
        raise ValueError(f'initial must contain {n_params} parameters')
    bounds = _normalize_bounds(bounds, n_params)
    if bounds is None:
        bounds = (-np.inf * np.ones(n_params), np.inf * np.ones(n_params))

    def residuals(p):
        model_n = _sellmeier1_eval(data.wavelengths, p, terms)
        resid = model_n - data.n
        if not np.all(np.isfinite(resid)):
            resid = np.full(data.n.shape, 1e12, dtype=float)
        if data.sigma_n is not None:
            resid = resid / data.sigma_n
        return resid

    result = optimize.least_squares(residuals, initial, bounds=bounds)
    if not result.success:
        raise ValueError(f'sellmeier1 fit failed: {result.message}')
    svals = np.linalg.svd(result.jac, compute_uv=False)
    rank = int(np.linalg.matrix_rank(result.jac))
    return result.x, rank, svals, result.message


def _fit_coefficients(data, model, terms, *, bounds=None, initial=None):
    if model == 'constant':
        A = np.ones((data.wavelengths.size, 1), dtype=float)
        return _linear_fit(A, data.n, sigma=data.sigma_n, bounds=bounds)
    if model == 'cauchy':
        A = _cauchy_design(data.wavelengths, terms)
        return _linear_fit(A, data.n, sigma=data.sigma_n, bounds=bounds)
    if model == 'schott':
        A = _schott_design(data.wavelengths)
        sigma = None if data.sigma_n is None else 2 * data.n * data.sigma_n
        return _linear_fit(A, data.n * data.n, sigma=sigma, bounds=bounds)
    if model == 'sellmeier1':
        return _fit_sellmeier1(data, terms, bounds=bounds, initial=initial)
    raise ValueError(model)


def _condition_number(svals):
    if svals is None or len(svals) == 0:
        return np.inf
    smax = float(np.max(svals))
    smin = float(np.min(svals))
    if smin == 0:
        return np.inf
    return smax / smin


def _fit_warnings(rank, n_params, dof, cond, allow_exact):
    warnings = []
    if dof < 0:
        warnings.append('fit is underdetermined; coefficients are not unique')
    elif dof == 0:
        warnings.append('fit has zero degrees of freedom')
    if rank < n_params:
        warnings.append('fit Jacobian or design matrix is rank deficient')
    if cond > 1e12:
        warnings.append('fit Jacobian or design matrix is ill conditioned')
    if allow_exact:
        warnings.append('allow_exact=True was used')
    return tuple(warnings)


def _make_fit_report(model, names, coeffs, data, terms, rank, svals, message,
                     allow_exact):
    residuals = _evaluate_fit_model(model, coeffs, data.wavelengths, terms) - data.n
    if not np.all(np.isfinite(residuals)):
        raise ValueError(f'{model} fit produced non-finite residuals')
    max_abs = float(np.max(np.abs(residuals)))
    rms = float(np.sqrt(np.mean(residuals * residuals)))
    dof = int(data.wavelengths.size - len(coeffs))
    cond = float(_condition_number(svals))
    return FitReport(
        model=model,
        coefficients={name: float(value) for name, value in zip(names, coeffs)},
        residuals=residuals.copy(),
        max_abs_error=max_abs,
        rms_error=rms,
        sample_count=int(data.wavelengths.size),
        parameter_count=int(len(coeffs)),
        degrees_of_freedom=dof,
        wavelength_range=data.wavelength_range,
        condition_number=cond,
        warnings=_fit_warnings(rank, len(coeffs), dof, cond, allow_exact),
        success=True,
        message=str(message),
    )


def _check_error_thresholds(report, max_abs_error, rms_error):
    if max_abs_error is not None and report.max_abs_error > max_abs_error:
        raise ValueError(
            f'{report.model} fit max_abs_error {report.max_abs_error:g} '
            f'exceeds requested {float(max_abs_error):g}'
        )
    if rms_error is not None and report.rms_error > rms_error:
        raise ValueError(
            f'{report.model} fit rms_error {report.rms_error:g} '
            f'exceeds requested {float(rms_error):g}'
        )


def _check_sellmeier_poles(name, coeffs, terms, wavelength_range):
    lo, hi = wavelength_range
    c = coeffs[terms:]
    poles = np.sqrt(c[c > 0])
    inside = (poles >= lo) & (poles <= hi)
    if np.any(inside):
        raise ValueError(
            f'sellmeier1 fit for {name} has a pole inside the fitted '
            'wavelength range'
        )


class FittedMaterial(BaseMaterial):
    """Coefficient-backed material fitted from wavelength and n samples."""

    def __init__(
        self,
        name,
        model,
        coefficients,
        *,
        wavelength_range,
        terms=None,
        fit_report=None,
        extrapolate=False,
        **kwargs,
    ):
        model = _normalize_model(model)
        if terms is None:
            try:
                n_coefficients = len(coefficients)
            except TypeError:
                n_coefficients = None
            if model == 'constant':
                terms = 1
            elif model == 'cauchy' and n_coefficients is not None:
                terms = n_coefficients
            elif model == 'sellmeier1' and n_coefficients is not None:
                if n_coefficients % 2:
                    raise ValueError(
                        'sellmeier1 coefficients must contain paired B and C values'
                    )
                terms = n_coefficients // 2
            elif model == 'schott':
                terms = 6
            else:
                terms = 1 if model == 'sellmeier1' else 2
        terms = int(terms)
        if model == 'schott':
            terms = 6
        elif terms < 1:
            raise ValueError(f'{model} terms must be at least one')
        names = _parameter_names(model, terms)
        if isinstance(coefficients, dict):
            coeffs = np.asarray([coefficients[name] for name in names], dtype=float)
        else:
            coeffs = np.asarray(coefficients, dtype=float)
        if coeffs.shape != (len(names),):
            raise ValueError(f'coefficients must contain {len(names)} values')
        if not np.all(np.isfinite(coeffs)):
            raise ValueError('coefficients must be finite')

        lo, hi = wavelength_range
        if lo is None or hi is None or lo <= 0 or hi <= 0 or lo > hi:
            raise ValueError('wavelength_range must be positive and ordered')
        metadata = dict(kwargs.pop('metadata', {}) or {})
        if extrapolate:
            metadata['extrapolate_wavelength'] = True
        metadata.update({
            'model': model,
            'terms': terms,
            'coefficients': {name: float(value) for name, value in zip(names, coeffs)},
            'extrapolate': bool(extrapolate),
        })
        super().__init__(
            name,
            wavelength_range=(float(lo), float(hi)),
            metadata=metadata,
            **kwargs,
        )
        self.model = model
        self.terms = terms
        self.parameter_names = names
        self.coefficients = coeffs.copy()
        self.coefficient_table = metadata['coefficients']
        self.extrapolate = bool(extrapolate)
        self.fit_report = fit_report

    @classmethod
    def from_samples(
        cls,
        name,
        wavelengths,
        n,
        *,
        model='cauchy',
        terms=None,
        sigma_n=None,
        max_abs_error=None,
        rms_error=None,
        extrapolate=False,
        allow_exact=False,
        bounds=None,
        initial=None,
        **kwargs,
    ):
        """Fit a material model from measured wavelength and n samples."""
        data = MaterialData(
            np.asarray(wavelengths, dtype=float),
            np.asarray(n, dtype=float),
            sigma_n=None if sigma_n is None else np.asarray(sigma_n, dtype=float),
            metadata=kwargs.get('metadata'),
        )
        if data.wavelengths.ndim != 1 or data.wavelengths.size == 0:
            raise ValueError('wavelengths must be a non-empty 1D array')
        if data.n.shape != data.wavelengths.shape:
            raise ValueError('wavelengths and n must have the same length')
        if data.sigma_n is not None and data.sigma_n.shape != data.wavelengths.shape:
            raise ValueError('wavelengths and sigma_n must have the same length')
        if not np.all(np.isfinite(data.wavelengths)):
            raise ValueError('wavelengths must contain only finite values')
        if not np.all(np.isfinite(data.n)):
            raise ValueError('n samples must contain only finite values')
        if data.sigma_n is not None and not np.all(np.isfinite(data.sigma_n)):
            raise ValueError('sigma_n must contain only finite values')
        if np.any(data.wavelengths <= 0) or np.any(np.diff(data.wavelengths) <= 0):
            raise ValueError('wavelengths must be strictly increasing with no duplicates')
        model = _normalize_model(model)
        terms = _model_terms(model, terms, data.wavelengths.size)
        n_params = len(_parameter_names(model, terms))
        if data.wavelengths.size < n_params and not bool(allow_exact):
            raise ValueError(
                f'{model} fit is underdetermined: {data.wavelengths.size} '
                f'samples for {n_params} parameters; pass allow_exact=True '
                'to request an exact underdetermined fit'
            )
        coeffs, rank, svals, message = _fit_coefficients(
            data,
            model,
            terms,
            bounds=bounds,
            initial=initial,
        )
        if model == 'sellmeier1':
            _check_sellmeier_poles(name, coeffs, terms, data.wavelength_range)
        names = _parameter_names(model, terms)
        report = _make_fit_report(
            model,
            names,
            coeffs,
            data,
            terms,
            rank,
            svals,
            message,
            bool(allow_exact),
        )
        _check_error_thresholds(report, max_abs_error, rms_error)
        return cls(
            name,
            model,
            coeffs,
            wavelength_range=data.wavelength_range,
            terms=terms,
            fit_report=report,
            extrapolate=extrapolate,
            **kwargs,
        )

    def _check_range(self, wvl):
        if self.extrapolate:
            return
        lo, hi = self.wavelength_range
        outside = (wvl < lo) | (wvl > hi)
        if np.any(outside):
            raise MaterialRangeError(
                f'wavelength for {self.name} outside material range '
                f'{lo:g} to {hi:g} um'
            )

    def __call__(self, wvl_um):
        """Return real refractive index at wavelength wvl_um in microns."""
        return self.n(wvl_um)

    def n(self, wvl_um, temperature=None):
        """Return real refractive index at wavelength wvl_um in microns."""
        self._check_range(wvl_um)
        return _evaluate_fit_model(self.model, self.coefficients, wvl_um, self.terms)

    def k(self, wvl_um, temperature=None):
        """Return zero extinction for real-index fitted materials."""
        self._check_range(wvl_um)
        return self._missing_k(wvl_um)


def from_samples(name, wavelengths, n, *, k=None, model=None, method='linear',
                 **kwargs):
    """Construct a tabulated material or fit a model from samples."""
    if model is None:
        return TabulatedMaterial(
            name,
            wavelengths,
            n,
            k=k,
            method=method,
            **kwargs,
        )
    if k is not None:
        raise ValueError('fitted materials do not support k samples yet')
    return FittedMaterial.from_samples(name, wavelengths, n, model=model, **kwargs)


def fit_material(name, wavelengths, n, **kwargs):
    """Fit a material model from measured wavelength and n samples."""
    return FittedMaterial.from_samples(name, wavelengths, n, **kwargs)
