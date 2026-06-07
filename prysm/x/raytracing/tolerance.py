"""Tolerancing: sensitivity tables and Monte Carlo over the analysis primitives.

Two top-level entry points wrap the Variable factories from design.py
with a perturbation distribution:

- sensitivity_table(prescription, perturbations, merit, *, step=None) ->
  SensitivityTable.  For each perturbation, perturb the prescription by
  +/- step and record the resulting change in merit; report the
  centered-difference dMerit/dParam along with absolute +/- deltas.
- monte_carlo(prescription, perturbations, merit, n_trials, *, seed)
  -> MonteCarloResult.  Draw n_trials independent samples from each
  perturbation's distribution, run the merit on the perturbed
  prescription each time, restore the prescription on exit.  The
  MonteCarloResult exposes .merits (raw array), .summary() (stats
  dict), and .yield_at(threshold) (fraction with merit <= threshold).

Perturbations target a single LensData DOF slot, addressed by a category
('curvature', 'radius', 'conic', 'coefs', 'thickness', 'tilt', 'decenter')
and a surface index.  The .normal / .normal_relative / .uniform / .triangular
classmethods capture the slot's nominal value at construction and store a step
(one-sigma or half-width).  An optional name= kwarg labels the table row.

Limitations to keep in mind:
- Perturbations are independent / orthogonal; correlated perturbations
  (e.g. glass index vs temperature) require a custom sampler.
- The merit callable receives the perturbed LensData and must return a
  scalar.  Use operand_as_merit() to wrap a design.py operand.

Example
-------
    from prysm.x.raytracing.tolerance import (
        Perturbation, sensitivity_table, monte_carlo, operand_as_merit,
    )
    from prysm.x.raytracing.design import RmsSpotRadius
    P, S = launch(ld, Field(0., 0.), 0.55, Sampling.hex(nrings=4), epd=10.0)
    perts = [
        Perturbation.normal_relative(ld, 'curvature', 0, 0.001, name='c1'),
        Perturbation.normal(ld, 'conic', 0, 0.01, name='k1'),
        Perturbation.normal(ld, 'thickness', 0, 0.02, name='t1'),
    ]
    merit = operand_as_merit(RmsSpotRadius(P, S, wavelength=0.55))
    table = sensitivity_table(ld, perts, merit)
    result = monte_carlo(ld, perts, merit, n_trials=1000, seed=42)

"""

from prysm.conf import config
from prysm.mathops import np

from .design import _TraceCache
from .sensitivity import central_difference


def _as_lens(lensdata):
    """Return the LensData spine, unwrapping an OpticalSystem if given one."""
    return getattr(lensdata, 'lens', lensdata)


def _resolve_slot(lensdata, category, surface, component=None):
    """Resolve a (category, surface) pair to a single LensData DOF slot.

    Multi-component categories (tilt, decenter on a coordinate break) resolve
    to three DOFs; pass component to select one by its offset -- decenter
    0/1/2 = dx/dy/dz, tilt 0/1/2 = rz/ry/rx (matching CoordBreak's layout).

    Accepts either a LensData or an OpticalSystem wrapping one.

    """
    lensdata = _as_lens(lensdata)
    slots = lensdata._category_slots(category, surface)
    if component is not None:
        slots = [s for s in slots if s[2] == int(component)]
    if len(slots) != 1:
        extra = '' if component is None else f' component {component}'
        raise ValueError(
            f'perturbation target {category!r} on surface {surface!r}{extra} '
            f'resolved to {len(slots)} DOFs; tolerancing needs exactly one '
            'scalar DOF (for tilt/decenter pass component=0/1/2)'
        )
    return slots[0]


# ---------- Perturbation ----------------------------------------------------

class Perturbation:
    """A single LensData DOF slot plus a sampling distribution.

    Built via the classmethod factories Perturbation.normal /
    .normal_relative / .uniform / .triangular; each captures the slot's
    nominal value at construction and stores a step (one-sigma or
    half-width) used as the default for sensitivity FD.

    """

    __slots__ = ('name', 'lensdata', 'slot', 'sampler', 'nominal', 'step')

    def __init__(self, lensdata, slot, sampler, nominal, step, name=''):
        self.name = str(name)
        self.lensdata = _as_lens(lensdata)
        self.slot = slot
        self.sampler = sampler  # callable(rng) -> sampled float
        self.nominal = float(nominal)
        self.step = float(step)

    def set(self, value):
        """Set the targeted DOF and invalidate the compiled system."""
        self.lensdata.spec.set_value(self.slot, value)
        self.lensdata._invalidate()

    def sample(self, rng):
        """Draw one sample from this perturbation's distribution."""
        return float(self.sampler(rng))

    def reset(self):
        """Restore the targeted DOF to its nominal value."""
        self.set(self.nominal)

    def __repr__(self):
        return (
            f'Perturbation(name={self.name!r}, nominal={self.nominal:g}, '
            f'step={self.step:g})'
        )

    @classmethod
    def normal(cls, lensdata, category, surface, sigma, name='', component=None):
        """Normal(nominal, sigma) distribution.  sigma is absolute.

        component selects one axis of a multi-DOF tilt/decenter slot (see
        _resolve_slot); leave it None for scalar categories.

        """
        lensdata = _as_lens(lensdata)
        slot = _resolve_slot(lensdata, category, surface, component)
        nom = float(lensdata.spec.get_value(slot))
        sigma = float(sigma)

        def sampler(rng):
            return float(rng.normal(nom, sigma))

        return cls(lensdata, slot, sampler, nom, sigma, name)

    @classmethod
    def normal_relative(cls, lensdata, category, surface, sigma_rel, name='',
                        component=None):
        """Normal distribution with sigma = sigma_rel * |nominal DOF|.

        For a DOF whose nominal is 0 this collapses to a delta function --
        use .normal with an explicit absolute sigma in that case.

        """
        lensdata = _as_lens(lensdata)
        slot = _resolve_slot(lensdata, category, surface, component)
        nom = float(lensdata.spec.get_value(slot))
        sigma = abs(nom) * float(sigma_rel)

        def sampler(rng):
            return float(rng.normal(nom, sigma))

        return cls(lensdata, slot, sampler, nom, sigma, name)

    @classmethod
    def uniform(cls, lensdata, category, surface, half_width, name='',
                component=None):
        """Uniform over (nominal - half_width, nominal + half_width)."""
        lensdata = _as_lens(lensdata)
        slot = _resolve_slot(lensdata, category, surface, component)
        nom = float(lensdata.spec.get_value(slot))
        hw = abs(float(half_width))

        def sampler(rng):
            return float(rng.uniform(nom - hw, nom + hw))

        return cls(lensdata, slot, sampler, nom, hw, name)

    @classmethod
    def triangular(cls, lensdata, category, surface, half_width, name='',
                   component=None):
        """Triangular distribution centered on nominal with half-width hw."""
        lensdata = _as_lens(lensdata)
        slot = _resolve_slot(lensdata, category, surface, component)
        nom = float(lensdata.spec.get_value(slot))
        hw = abs(float(half_width))

        def sampler(rng):
            return float(rng.triangular(nom - hw, nom, nom + hw))

        return cls(lensdata, slot, sampler, nom, hw, name)


# ---------- operand_as_merit ------------------------------------------------

def operand_as_merit(operand):
    """Wrap a design.py operand into a one-arg merit(prescription) -> float.

    Creates a fresh _TraceCache for each call so the operand can hit it
    without leaking cache state between merit evaluations.

    """

    def merit(prescription):
        cache = _TraceCache(prescription)
        return float(operand(prescription, cache))

    return merit


# ---------- SensitivityTable ------------------------------------------------

class SensitivityTable:
    """Per-parameter centered-difference sensitivity report.

    .rows is a list of dicts with keys: name, nominal, step, merit_plus,
    merit_minus, delta_plus, delta_minus, sensitivity (= centered dM/dx).
    .merit_nominal is the merit at the unperturbed prescription.

    Pretty repr renders as a column-aligned table.

    """

    __slots__ = ('rows', 'merit_nominal')

    def __init__(self, rows, merit_nominal):
        self.rows = list(rows)
        self.merit_nominal = float(merit_nominal)

    def names(self):
        return [r['name'] for r in self.rows]

    def sensitivities(self):
        return np.array([r['sensitivity'] for r in self.rows])

    def worst_delta_per_row(self):
        """max(|delta_plus|, |delta_minus|) per row, in row order."""
        return np.array([
            max(abs(r['delta_plus']), abs(r['delta_minus']))
            for r in self.rows
        ])

    def __repr__(self):
        lines = [
            f'SensitivityTable(merit_nominal={self.merit_nominal:.6g}):',
            (f'{"name":<20} {"nominal":>14} {"step":>12} '
             f'{"d_plus":>12} {"d_minus":>12} {"dM/dx":>12}'),
        ]
        for r in self.rows:
            lines.append(
                f'{r["name"]:<20} {r["nominal"]:>14.6g} {r["step"]:>12.6g} '
                f'{r["delta_plus"]:>12.6g} {r["delta_minus"]:>12.6g} '
                f'{r["sensitivity"]:>12.6g}'
            )
        return '\n'.join(lines)


def sensitivity_table(prescription, perturbations, merit, *, step=None):
    """Centered-difference sensitivity of merit w.r.t. each perturbation.

    For each Perturbation in `perturbations`, set parameter to
    (nominal + h) and (nominal - h) in turn, evaluate merit, restore.
    The default h is the perturbation's own .step (one sigma for
    normal/normal_relative; half-width for uniform/triangular); pass an
    explicit step= to override globally.

    Parameters
    ----------
    prescription : sequence of Surface
    perturbations : iterable of Perturbation
    merit : callable merit(prescription) -> float
    step : float, optional
        global override for the per-perturbation step.

    Returns
    -------
    SensitivityTable

    Notes
    -----
    This is the general fallback: it treats merit as a black box and costs two
    full bundle traces per perturbation (2P traces for P parameters).  When the
    merit reduces to a head the differential engine supports -- RMS wavefront
    error, spot size, or distortion -- prefer the adjoint multi_objective_sensi-
    tivity or the wavefront-differential sensitivity APIs, which recover the
    entire sensitivity vector from a single traced bundle plus a reverse (or
    tangent) sweep, an order of magnitude cheaper than this finite difference.
    Use this routine only for merits those engines do not cover.

    """
    perturbations = list(perturbations)
    m_nom = float(merit(prescription))
    rows = []
    for p in perturbations:
        h = float(step) if step is not None else p.step
        if h == 0.0:
            # zero-step perturbation contributes no sensitivity
            rows.append({
                'name': p.name,
                'nominal': p.nominal,
                'step': 0.0,
                'merit_nominal': m_nom,
                'merit_plus': m_nom,
                'merit_minus': m_nom,
                'delta_plus': 0.0,
                'delta_minus': 0.0,
                'sensitivity': 0.0,
            })
            continue
        def probe(value, p=p):
            p.set(value)
            return merit(prescription)
        try:
            m_plus, m_minus = central_difference(probe, p.nominal, h)
        finally:
            p.set(p.nominal)
        rows.append({
            'name': p.name,
            'nominal': p.nominal,
            'step': h,
            'merit_nominal': m_nom,
            'merit_plus': m_plus,
            'merit_minus': m_minus,
            'delta_plus': m_plus - m_nom,
            'delta_minus': m_minus - m_nom,
            'sensitivity': (m_plus - m_minus) / (2.0 * h),
        })
    return SensitivityTable(rows, merit_nominal=m_nom)


# ---------- MonteCarloResult ------------------------------------------------

class MonteCarloResult:
    """Outcome of a tolerancing Monte Carlo trial run.

    .merits : ndarray (n_trials,) of merit values per trial.
    .sampled_x : ndarray (n_trials, n_params) or None; sampled parameter
                 values per trial.  Populated only when monte_carlo was
                 called with record_samples=True.
    .nominals : ndarray (n_params,) of nominal parameter values.
    .names : list of str, length n_params.

    """

    __slots__ = ('merits', 'sampled_x', 'nominals', 'names')

    def __init__(self, merits, sampled_x, nominals, names):
        self.merits = np.asarray(merits, dtype=config.precision)
        self.sampled_x = (None if sampled_x is None
                          else np.asarray(sampled_x, dtype=config.precision))
        self.nominals = np.asarray(nominals, dtype=config.precision)
        self.names = list(names)

    @property
    def n_trials(self):
        return int(self.merits.shape[0])

    def summary(self):
        """Common stats over the merit distribution.

        Returns
        -------
        dict with n_trials, min, max, mean, std, median, p95, p99.

        """
        m = self.merits
        return {
            'n_trials': int(m.shape[0]),
            'min': float(m.min()),
            'max': float(m.max()),
            'mean': float(m.mean()),
            'std': float(m.std()),
            'median': float(np.median(m)),
            'p95': float(np.percentile(m, 95)),
            'p99': float(np.percentile(m, 99)),
        }

    def yield_at(self, threshold):
        """Fraction of trials with merit <= threshold."""
        return float((self.merits <= float(threshold)).mean())

    def __repr__(self):
        s = self.summary()
        return (
            f'MonteCarloResult(n={s["n_trials"]}, '
            f'mean={s["mean"]:.6g}, std={s["std"]:.6g}, '
            f'p95={s["p95"]:.6g})'
        )


def monte_carlo(prescription, perturbations, merit, n_trials, *,
                seed=None, record_samples=False):
    """Run a Monte Carlo tolerancing simulation.

    For each of n_trials independent trials:
    1. Sample every perturbation's distribution.
    2. Set each parameter via its setter.
    3. Evaluate merit(prescription) and record the result.
    Once all trials are done (or on any exception), restore each
    parameter to its nominal value.

    Parameters
    ----------
    prescription : sequence of Surface
    perturbations : iterable of Perturbation
    merit : callable merit(prescription) -> float
    n_trials : int
    seed : int, optional
        seed for numpy's default_rng for reproducibility.
    record_samples : bool, optional
        if True, also record the (n_trials, n_params) sampled values
        for diagnostic plots.

    Returns
    -------
    MonteCarloResult

    """
    perturbations = list(perturbations)
    n_p = len(perturbations)
    rng = np.random.default_rng(seed)
    merits = np.empty(int(n_trials), dtype=config.precision)
    sampled_x = (np.empty((int(n_trials), n_p), dtype=config.precision)
                 if record_samples else None)
    try:
        for trial in range(int(n_trials)):
            for i, p in enumerate(perturbations):
                v = p.sample(rng)
                p.set(v)
                if record_samples:
                    sampled_x[trial, i] = v
            merits[trial] = float(merit(prescription))
    finally:
        for p in perturbations:
            p.reset()
    nominals = [p.nominal for p in perturbations]
    names = [p.name for p in perturbations]
    return MonteCarloResult(merits, sampled_x, nominals, names)
