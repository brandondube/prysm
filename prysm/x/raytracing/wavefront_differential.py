"""Wavefront-differential tolerancing front end.

One nominal differential trace (via _diff_raytrace.wavefront_with_tangents)
yields the nominal OPD W0 and the per-tolerance wavefront-derivative maps
dW_p = dOPD/dtau_p.  From those this module builds the Rimmer-style
wavefront-error quadratic and everything derived from it, with no further ray
tracing.

Reference: M. P. Rimmer, "Analysis of Perturbed Lens Systems," Applied Optics
9(3), 533-537 (1970).

The model is the small-degradation linearization of the wavefront,
    W(tau) = W0 + sum_p dW_p tau_p,
so RMS wavefront error squared is the pupil quadratic form
    RMS^2(tau) = mean(W^2) = C + B . tau + tau^T G tau,
    C   = mean(W0^2)                 (nominal RMS^2),
    B_p = 2 mean(W0 dW_p)            (linear / cross term with the nominal),
    G_pq = mean(dW_p dW_q)           (Gram; A_p = G_pp is the self term).
This matches design.WavefrontRMS exactly (RMS about zero, chief anchored at 0),
which is the merit the FD sensitivity_table / slow monte_carlo validate against.

For a single tolerance scaled by T this is the standard quadratic form
    RMS(T) = sqrt(A T^2 + B T + C),
and everything else follows:

- sensitivity      dRMS/dtau at 0 = B / (2 sqrt(C))  (matches the FD slope)
- inverse sens.    solve A T^2 + B T + C = RMS_target^2 for the allowed range
- RSS roll-up      independent zero-mean tau: E[RMS^2] = C + sum_p sigma_p^2 A_p
- fast Monte Carlo draw tau vectors, evaluate the quadratic (no re-trace) ->
  a merit distribution / cumulative-probability curve, validated against the
  slow re-tracing monte_carlo.

Cross-terms (off-diagonal G) are captured automatically -- the advantage over
FD, which only ever moves one tolerance at a time.

Example
-------
    from prysm.x.raytracing.launch import Field, Sampling, launch
    from prysm.x.raytracing.tolerance import Perturbation
    from prysm.x.raytracing.wavefront_differential import wavefront_differential
    P, S = launch(ld, Field(2.5, 2.5), 0.55, Sampling.rect(n=7), epd=10.0)
    perts = [
        Perturbation.normal(ld, 'curvature', 0, 1e-4, name='c1'),
        Perturbation.normal(ld, 'thickness', 0, 0.02, name='t1'),
    ]
    wd = wavefront_differential(ld, perts, P, S, 0.55)
    print(wd.sensitivity_table())
    result = wd.fast_monte_carlo(perts, n_trials=10000, seed=0)

"""

from prysm.conf import config
from prysm.mathops import np

from ._diff_raytrace import seeds_from_perturbations, wavefront_with_tangents
from ._meta import system_wavelength
from .analysis import wavefront_zernike_fit
from .tolerance import MonteCarloResult


def wavefront_differential(lensdata, perturbations, P, S, wavelength, *,
                           extra_seeds=None, extra_names=None, extra_steps=None,
                           compensators=None, comp_rcond=1e-9,
                           chief_index=None,
                           axis_point=None, axis_dir=None, P_xp=None,
                           field=None, pose_step=1e-6):
    """Build the wavefront-differential model from one nominal trace.

    Maps the perturbations to differential seeds, runs the single nominal
    differential trace, and returns a WavefrontDifferential holding the
    quadratic (C, B, Gram) for the launched bundle.

    extra_seeds appends already-built DiffSeeds as further tolerance columns --
    the home for perturbations that are not LensData DOF slots, in particular a
    Zernike surface irregularity (CYN/CYD via _diff_raytrace.seed_irregularity).
    They get the full quadratic / sensitivity / Zernike-coefficient
    treatment; only fast_monte_carlo (which needs Perturbation samplers) does
    not cover them.

    If compensators are given, their derivative maps are traced alongside the
    tolerances (still one trace) and the wavefront W0 and every tolerance map
    are projected onto the orthogonal complement of the compensator subspace
    (SVD least squares).  The returned model is then the compensated one: every
    sensitivity, RMS, and Monte Carlo is reported after the compensators have
    been re-optimized per perturbation -- the linear-least-squares analog of
    lens-design software re-solving the back focus (etc.) for each tolerance.

    Parameters
    ----------
    lensdata : LensData
        the nominal system; the perturbations must target this same LensData.
    perturbations : sequence of tolerance.Perturbation
        the tolerance set; defines the parameter axis order.  Their .step
        values are the default per-tolerance scales used for sensitivity and
        RSS roll-up.
    P, S : ndarray, (N, 3)
        launch bundle (typically from launch()).
    wavelength : float
        microns.
    extra_seeds : sequence of _diff_raytrace.DiffSeed, optional
        extra tolerance columns appended after the perturbations (e.g.
        irregularity); not mapped from LensData DOFs.
    extra_names : sequence of str, optional
        names for the extra_seeds columns; defaults to each seed's .name.
    extra_steps : sequence of float, optional
        per-column default scales for the extra_seeds; defaults to 1.0.
    compensators : sequence of tolerance.Perturbation, optional
        DOFs (e.g. an image-gap despace for back focus) re-optimized per
        perturbation by projecting them out of the wavefront.
    comp_rcond : float, optional
        relative singular-value cutoff for the compensator pseudo-inverse.
    chief_index, axis_point, axis_dir, P_xp, field
        forwarded to wavefront_with_tangents (reference-sphere controls).
    pose_step : float, optional
        layout-FD step for pose tangents (see seeds_from_perturbations).

    Returns
    -------
    WavefrontDifferential
        compensated when compensators is given, else the uncompensated model.

    """
    perturbations = list(perturbations)
    extra_seeds = list(extra_seeds) if extra_seeds else []
    compensators = list(compensators) if compensators else []
    wavelength = system_wavelength(lensdata, wavelength)
    n_tol = len(perturbations) + len(extra_seeds)
    # one trace carries the tolerance maps, extra seeds, and compensator maps
    seeds = (seeds_from_perturbations(perturbations, pose_step=pose_step)
             + extra_seeds
             + seeds_from_perturbations(compensators, pose_step=pose_step))
    opd, x_pupil, y_pupil, dW = wavefront_with_tangents(
        lensdata.to_surfaces(), P, S, wavelength, seeds,
        chief_index=chief_index,
        axis_point=axis_point, axis_dir=axis_dir, P_xp=P_xp,
        field=field, output='length')
    if extra_names is None:
        extra_names = [s.name or f'extra{i}' for i, s in enumerate(extra_seeds)]
    else:
        extra_names = list(extra_names)
    extra_steps = ([1.0] * len(extra_seeds) if extra_steps is None
                   else list(extra_steps))
    names = ([p.name or f'tol{i}' for i, p in enumerate(perturbations)]
             + extra_names)
    steps = [p.step for p in perturbations] + list(extra_steps)

    tol_maps = dW[:, :n_tol]
    if not compensators:
        return WavefrontDifferential(opd, tol_maps, names=names, steps=steps,
                                     x_pupil=x_pupil, y_pupil=y_pupil)

    comp_maps = dW[:, n_tol:]
    comp_names = [c.name or f'comp{i}' for i, c in enumerate(compensators)]
    opd_c, tol_c, _ = compensate(opd, tol_maps, comp_maps, rcond=comp_rcond)
    # compensator motion rates dc/dtau = -M+ D use the UNprojected tol maps
    motions = -(np.linalg.pinv(comp_maps, rcond=comp_rcond) @ tol_maps)
    return WavefrontDifferential(opd_c, tol_c, names=names, steps=steps,
                                 x_pupil=x_pupil, y_pupil=y_pupil,
                                 comp_names=comp_names, comp_maps=comp_maps,
                                 comp_motions=motions)


# ---------- compensator projection (SVD least squares) ----------------------

def _orthonormal_basis(M, rcond):
    """Orthonormal basis of col(M) for singular values above rcond * max."""
    M = np.asarray(M, dtype=config.precision)
    if M.ndim != 2 or M.shape[1] == 0:
        return M.reshape(M.shape[0], 0)
    U, s, _ = np.linalg.svd(M, full_matrices=False)
    if s.shape[0] == 0:
        return U[:, :0]
    rank = int(np.sum(s > rcond * s[0]))
    return U[:, :rank]


def project_out(v, basis):
    """Remove the components of v lying in span(basis) (orthonormal columns).

    v is (N,) or (N, K); basis is (N, r) with orthonormal columns.  Returns the
    residual (I - basis basis^T) v -- the part orthogonal to the subspace.
    """
    basis = np.asarray(basis, dtype=config.precision)
    if basis.shape[1] == 0:
        return np.asarray(v, dtype=config.precision)
    v = np.asarray(v, dtype=config.precision)
    return v - basis @ (basis.T @ v)


def compensate(opd, tol_maps, comp_maps, *, rcond=1e-9):
    """Project the wavefront and tolerance maps off the compensator subspace.

    The compensated wavefront for a perturbation is the residual after the
    best least-squares compensator motion, i.e. the projection of W0 + (tol
    maps) onto the orthogonal complement of span(comp_maps).  Doing it to W0
    and to each tolerance map turns the compensated tolerancing problem back
    into a plain wavefront-error quadratic on the projected maps.

    Returns (opd_proj, tol_maps_proj, basis) where basis is the orthonormal
    compensator basis used for the projection.
    """
    basis = _orthonormal_basis(comp_maps, rcond)
    return project_out(opd, basis), project_out(tol_maps, basis), basis


class WavefrontDifferential:
    """The wavefront-error quadratic for one launch bundle and a fixed tolerance set.

    Holds the nominal wavefront W0 and the per-tolerance derivative maps dW,
    plus the assembled quadratic coefficients
        C   nominal RMS^2 = mean(W0^2)
        B   (P,) linear coefficients 2 mean(W0 dW_p)
        G   (P, P) Gram mean(dW_p dW_q); A = diag(G) is the self term.
    All RMS values are in the OPD length units of the trace (prysm gauge,
    chief == 0), the same units design.WavefrontRMS reports.

    Construct via wavefront_differential(); the bare constructor is handy for
    feeding precomputed (opd, dW) maps.

    """

    __slots__ = ('W0', 'dW', 'names', 'steps', 'x_pupil', 'y_pupil',
                 'n_samples', 'n_params', 'C', 'B', 'G', 'A', 'rms_nominal',
                 'comp_names', 'comp_maps', 'comp_motions')

    def __init__(self, opd, dW, *, names=None, steps=None,
                 x_pupil=None, y_pupil=None, comp_names=None, comp_maps=None,
                 comp_motions=None):
        dt = config.precision
        self.W0 = np.asarray(opd, dtype=dt).ravel()
        self.dW = np.asarray(dW, dtype=dt)
        if self.dW.ndim != 2 or self.dW.shape[0] != self.W0.shape[0]:
            raise ValueError(
                f'dW must be (N, P) parallel to opd (N={self.W0.shape[0]}); '
                f'got {self.dW.shape}')
        self.n_samples, self.n_params = self.dW.shape
        self.names = (list(names) if names is not None
                      else [f'tol{i}' for i in range(self.n_params)])
        self.steps = (np.asarray(steps, dtype=dt) if steps is not None
                      else np.ones(self.n_params, dtype=dt))
        self.x_pupil = None if x_pupil is None else np.asarray(x_pupil)
        self.y_pupil = None if y_pupil is None else np.asarray(y_pupil)
        # compensator metadata (None unless this is a compensated model)
        self.comp_names = None if comp_names is None else list(comp_names)
        self.comp_maps = None if comp_maps is None else np.asarray(comp_maps,
                                                                   dtype=dt)
        self.comp_motions = (None if comp_motions is None
                             else np.asarray(comp_motions, dtype=dt))

        n = self.n_samples
        self.C = float(np.mean(self.W0 * self.W0))
        self.B = 2.0 * np.mean(self.W0[:, None] * self.dW, axis=0)
        self.G = (self.dW.T @ self.dW) / n
        self.A = np.diag(self.G).copy()
        self.rms_nominal = float(np.sqrt(self.C))

    # ---------- per-tolerance quadratic ------------------------------------

    def quadratic_coeffs(self, p):
        """(A, B, C) of RMS^2(T) = A T^2 + B T + C for tolerance p alone."""
        return float(self.A[p]), float(self.B[p]), self.C

    def rms_at(self, p, T):
        """Predicted RMS with tolerance p set to value T, others nominal."""
        A, B, C = self.quadratic_coeffs(p)
        T = np.asarray(T, dtype=config.precision)
        val = A * T * T + B * T + C
        return np.sqrt(np.maximum(val, 0.0))

    def sensitivity(self):
        """dRMS/dtau at nominal for every tolerance: B / (2 sqrt(C)).

        The first-order slope the FD sensitivity_table of WavefrontRMS measures.
        """
        if self.rms_nominal == 0.0:
            # at a perfect wavefront the slope is undefined (RMS ~ |T|);
            # report sqrt(A) as the local rate of |W| growth.
            return np.sqrt(self.A)
        return self.B / (2.0 * self.rms_nominal)

    # ---------- full quadratic form ----------------------------------------

    def predict_rms_sq(self, tau):
        """RMS^2(tau) = C + B . tau + tau^T G tau, vectorized over rows of tau.

        tau is (P,) for a single perturbation vector or (M, P) for a batch;
        returns a scalar or (M,) accordingly.  The form is a mean of squares,
        so it is >= 0 up to roundoff (clipped at 0).
        """
        tau = np.asarray(tau, dtype=config.precision)
        single = tau.ndim == 1
        if single:
            tau = tau[None, :]
        lin = tau @ self.B
        quad = np.sum((tau @ self.G) * tau, axis=1)
        val = np.maximum(self.C + lin + quad, 0.0)
        return float(val[0]) if single else val

    def predict_rms(self, tau):
        """sqrt(predict_rms_sq(tau)) -- the predicted RMS for tau."""
        return np.sqrt(self.predict_rms_sq(tau))

    def gram(self):
        """The (P, P) cross-term Gram matrix mean(dW_p dW_q)."""
        return self.G

    # ---------- Zernike-coefficient sensitivities --------------------------

    def zernike_sensitivity(self, nms, *, normalization_radius=None, norm=True):
        """Sensitivity of fitted wavefront Zernike coefficients to each tolerance.

        Fits the nominal wavefront W0 and every per-tolerance map dW_p onto the
        Zernike basis nms with analysis.wavefront_zernike_fit.  The least-squares
        fit is linear in the OPD, so fitting dW_p yields exactly dc/dtau_p; a
        single shared normalization radius keeps the basis identical across all
        the fits.  Reports per-coefficient sensitivities (for wavefront content
        such as astigmatism or coma growth) alongside the scalar RMS
        sensitivity.

        Needs the pupil coordinates the model was built with (present when it
        came from wavefront_differential).

        Parameters
        ----------
        nms : iterable of (int, int)
            Zernike (n, m) indices to fit.
        normalization_radius : float, optional
            radius for the pupil-coordinate normalization; defaults to the pupil
            extent max(sqrt(x^2 + y^2)) so every map shares one basis.
        norm : bool, optional
            orthonormal (unit-RMS) Zernikes if True (default).

        Returns
        -------
        nominal_coefs : ndarray, (K,)
            Zernike coefficients of the nominal wavefront W0.
        dcoefs : ndarray, (K, P)
            dc_k/dtau_p; column p is the coefficient sensitivity to tolerance p.

        """
        if self.x_pupil is None or self.y_pupil is None:
            raise ValueError(
                'zernike_sensitivity needs the pupil coordinates; build the '
                'model via wavefront_differential (which records them)')
        nms = list(nms)
        x = self.x_pupil
        y = self.y_pupil
        if normalization_radius is None:
            normalization_radius = float(np.sqrt(np.max(x * x + y * y)))
        nominal_coefs, _ = wavefront_zernike_fit(
            self.W0, x, y, nms,
            normalization_radius=normalization_radius, norm=norm)
        dcoefs = np.empty((len(nms), self.n_params), dtype=config.precision)
        for p in range(self.n_params):
            coefs_p, _ = wavefront_zernike_fit(
                self.dW[:, p], x, y, nms,
                normalization_radius=normalization_radius, norm=norm)
            dcoefs[:, p] = coefs_p
        return np.asarray(nominal_coefs, dtype=config.precision), dcoefs

    # ---------- compensators -----------------------------------------------

    @property
    def is_compensated(self):
        """True when this model projects out a compensator subspace."""
        return self.comp_maps is not None

    def compensator_motions(self):
        """Per-tolerance compensator motion rate dc/dtau (K, P).

        The least-squares compensator setting that minimizes the perturbed
        wavefront is c(tau) = -M+ (W0 + D tau); the part that tracks each
        tolerance is dc/dtau = -M+ D (D the unprojected tolerance maps), the
        usual compensator-pickup table next to each tolerance.  Raises if there
        are no compensators.
        """
        if self.comp_motions is None:
            raise ValueError('this model has no compensators')
        return self.comp_motions

    # ---------- RSS roll-up ------------------------------------------------

    def _scales(self, scales):
        if scales is None:
            return self.steps
        scales = np.asarray(scales, dtype=config.precision)
        if scales.ndim == 0:
            scales = np.full(self.n_params, float(scales), dtype=config.precision)
        return scales

    def expected_rms_sq(self, scales=None, *, cross_terms=False):
        """E[RMS^2] for independent zero-mean tolerances of std `scales`.

        With tau_p independent, zero-mean, variance sigma_p^2, the linear term
        averages out and E[RMS^2] = C + sum_p sigma_p^2 A_p (diagonal Gram).
        Set cross_terms=True to instead include the full Gram, which is only
        correct when the perturbations are correlated with covariance G-shaped;
        for the standard independent case leave it False.

        scales defaults to the perturbations' .step (one sigma / half-width).
        """
        s = self._scales(scales)
        if cross_terms:
            extra = float(s @ self.G @ s)
        else:
            extra = float(np.sum(s * s * self.A))
        return self.C + extra

    def expected_rms(self, scales=None, *, cross_terms=False):
        """sqrt(expected_rms_sq) -- the RSS-rolled-up predicted RMS."""
        return float(np.sqrt(max(self.expected_rms_sq(
            scales, cross_terms=cross_terms), 0.0)))

    def rms_change_per_tolerance(self, scales=None):
        """Per-tolerance RMS minus nominal at tau_p = +scale_p (others 0).

        This is the individual-sensitivity column of the Rimmer-style
        quadratic; sqrt sum of squares of these is one common (conservative)
        RSS estimate, while
        expected_rms() is the statistically exact independent roll-up.
        """
        s = self._scales(scales)
        rms_p = np.sqrt(np.maximum(self.A * s * s + self.B * s + self.C, 0.0))
        return rms_p - self.rms_nominal

    # ---------- inverse sensitivity ----------------------------------------

    def inverse_sensitivity(self, target_delta_rms, *, tiny=1e-30):
        """Allowed per-tolerance value range for a target RMS increase.

        Solves A T^2 + B T + C = RMS_target^2 with RMS_target = rms_nominal +
        target_delta_rms, returning (T_lo, T_hi) per tolerance -- the interval
        of values keeping the single-tolerance RMS at or below the target.

        Because RMS_target > rms_nominal the constant term C - RMS_target^2 is
        negative, so a tolerance with any sensitivity (A > 0) has one negative
        and one positive root straddling 0.  A purely linear tolerance (A ~ 0)
        gets one finite bound and +/- inf on the slack side; a tolerance with
        no first-order effect at all (A ~ 0 and B ~ 0) is unbounded both ways.

        Returns
        -------
        t_lo, t_hi : ndarray, (P,)
            lower and upper allowed value of each tolerance.

        """
        target_rms = self.rms_nominal + float(target_delta_rms)
        cc = self.C - target_rms * target_rms  # <= 0 for a positive target
        t_lo = np.empty(self.n_params, dtype=config.precision)
        t_hi = np.empty(self.n_params, dtype=config.precision)
        for p in range(self.n_params):
            A, B = float(self.A[p]), float(self.B[p])
            if abs(A) <= tiny:
                if abs(B) <= tiny:
                    t_lo[p], t_hi[p] = -np.inf, np.inf
                    continue
                root = -cc / B           # A T^2 negligible: B T + cc = 0
                if root >= 0:
                    t_lo[p], t_hi[p] = -np.inf, root
                else:
                    t_lo[p], t_hi[p] = root, np.inf
                continue
            disc = B * B - 4.0 * A * cc   # >= B^2 >= 0 since A>0, cc<=0
            sq = np.sqrt(max(disc, 0.0))
            r1 = (-B - sq) / (2.0 * A)
            r2 = (-B + sq) / (2.0 * A)
            t_lo[p], t_hi[p] = (r1, r2) if r1 <= r2 else (r2, r1)
        return t_lo, t_hi

    # ---------- fast Monte Carlo over the quadratic ------------------------

    def fast_monte_carlo(self, perturbations, n_trials, *, seed=None,
                         record_samples=False):
        """Monte Carlo over the quadratic -- no re-tracing.

        Draws tau exactly as tolerance.monte_carlo does (per trial, per
        perturbation, same RNG order), so with a shared seed the sampled
        deviations match the slow re-tracing run and the merit distributions
        agree to the linearization error.  Returns a tolerance.MonteCarloResult
        so it is directly comparable to monte_carlo's output.

        perturbations must be the same sequence (same order) the model was
        built from; their nominals define tau = sample - nominal.
        """
        perturbations = list(perturbations)
        if len(perturbations) != self.n_params:
            raise ValueError(
                f'expected {self.n_params} perturbations to match the model, '
                f'got {len(perturbations)}')
        rng = np.random.default_rng(seed)
        n_trials = int(n_trials)
        nominals = np.array([p.nominal for p in perturbations],
                            dtype=config.precision)
        tau = np.empty((n_trials, self.n_params), dtype=config.precision)
        sampled = (np.empty((n_trials, self.n_params), dtype=config.precision)
                   if record_samples else None)
        for trial in range(n_trials):
            for i, p in enumerate(perturbations):
                v = p.sample(rng)
                tau[trial, i] = v - nominals[i]
                if record_samples:
                    sampled[trial, i] = v
        merits = self.predict_rms(tau)
        names = [p.name for p in perturbations]
        return MonteCarloResult(merits, sampled, nominals, names)

    # ---------- reporting --------------------------------------------------

    def rows(self, scales=None):
        """Per-tolerance report rows: name, A, B, C, sensitivity, delta_rms."""
        s = self._scales(scales)
        sens = self.sensitivity()
        drms = self.rms_change_per_tolerance(scales)
        out = []
        for p in range(self.n_params):
            out.append({
                'name': self.names[p],
                'A': float(self.A[p]),
                'B': float(self.B[p]),
                'C': self.C,
                'scale': float(s[p]),
                'sensitivity': float(sens[p]),
                'delta_rms': float(drms[p]),
            })
        return out

    def sensitivity_table(self, scales=None):
        """Column-aligned per-tolerance sensitivity report (a string)."""
        lines = [
            f'WavefrontDifferential(rms_nominal={self.rms_nominal:.6g}):',
            (f'{"name":<20} {"scale":>12} {"A":>12} {"B":>12} '
             f'{"dRMS/dtau":>12} {"dRMS@scale":>12}'),
        ]
        for r in self.rows(scales):
            lines.append(
                f'{r["name"]:<20} {r["scale"]:>12.6g} {r["A"]:>12.6g} '
                f'{r["B"]:>12.6g} {r["sensitivity"]:>12.6g} '
                f'{r["delta_rms"]:>12.6g}')
        return '\n'.join(lines)

    def __repr__(self):
        return (f'WavefrontDifferential(n_samples={self.n_samples}, '
                f'n_params={self.n_params}, '
                f'rms_nominal={self.rms_nominal:.6g})')


# ---------- cumulative-probability curve ------------------------------------

def cumulative_probability(merits):
    """Empirical cumulative-probability curve of a merit sample.

    Accepts a MonteCarloResult or a raw merit array.  Returns (thresholds,
    probability) with thresholds the sorted merits and probability[i] the
    fraction of trials with merit <= thresholds[i], i.e. a cumulative
    probability vs degraded-performance curve.
    """
    m = getattr(merits, 'merits', merits)
    m = np.sort(np.asarray(m, dtype=config.precision))
    n = m.shape[0]
    prob = (np.arange(1, n + 1, dtype=config.precision)) / n
    return m, prob


__all__ = [
    'wavefront_differential',
    'WavefrontDifferential',
    'compensate',
    'project_out',
    'cumulative_probability',
]
