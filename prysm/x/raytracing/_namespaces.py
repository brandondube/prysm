"""Inner OpticalSystem verb namespaces."""


class _OptNamespace:
    """Design and optimization verbs under sys.opt."""

    __slots__ = ('_sys',)

    def __init__(self, system):
        self._sys = system

    # -- DOF selection (category x surface-range); chainable --
    def vary(self, category, surfaces='all'):
        """Mark a category of DOFs free over a range of surfaces."""
        self._sys._design.vary(category, surfaces)
        return self

    def vary_all(self):
        """Mark every scalar DOF free (except pickup/solve dependents)."""
        self._sys._design.vary_all()
        return self

    def freeze(self, category, surfaces='all'):
        """Inverse of vary."""
        self._sys._design.freeze(category, surfaces)
        return self

    def freeze_all(self):
        """Mark every scalar DOF fixed."""
        self._sys._design.freeze_all()
        return self

    def constrain(self, category, *, lo=None, hi=None, relative=None,
                  surfaces='all'):
        """Set box bounds on a category of DOFs over a range of surfaces."""
        self._sys._design.constrain(category, lo=lo, hi=hi, relative=relative,
                                    surfaces=surfaces)
        return self

    def pickup(self, category, surface, *, from_surface, from_category=None,
               scale=1.0, offset=0.0):
        """Make a DOF a pickup of another: dependent = scale*source + offset."""
        self._sys._design.pickup(category, surface, from_surface=from_surface,
                                 from_category=from_category, scale=scale,
                                 offset=offset)
        return self

    # -- optimizer free vector --
    def pack(self):
        """Dense contiguous vector of the free DOFs."""
        return self._sys._design.pack()

    def update(self, x):
        """Scatter a free vector into the rows, resolve dependents, invalidate."""
        self._sys._design.update(x)
        return self

    def bounds(self):
        """(lo, hi) arrays parallel to the free vector."""
        return self._sys._design.bounds()

    # -- problem assembly / solve --
    def problem(self, goal='spot', *, sampling=None, fields=None,
                wavelengths=None, constraints=None):
        """Assemble a design.Problem over this system's free vector."""
        from .design import build_problem
        return build_problem(self._sys, goal, sampling=sampling, fields=fields,
                             wavelengths=wavelengths, constraints=constraints)

    def optimize(self, goal='spot', *, sampling=None, fields=None,
                 wavelengths=None, constraints=None, **solve_kwargs):
        """Build and solve an optimization problem in one shot."""
        prob = self.problem(goal, sampling=sampling, fields=fields,
                            wavelengths=wavelengths, constraints=constraints)
        return prob.solve(**solve_kwargs)


class _SolveNamespace:
    """State-writing solves under sys.solve."""

    __slots__ = ('_sys',)

    def __init__(self, system):
        self._sys = system

    def image_distance(self, surface=None, *, wavelength=None):
        """Solve a gap so the image plane sits at the paraxial image."""
        sys = self._sys
        wvl = sys.wavelength(wavelength)
        sys._design.solve_image_distance(surface, wavelength=wvl)
        return sys

    def clear_image_distance(self):
        """Disable the active paraxial image-distance solve, if any."""
        self._sys._design.clear_image_distance_solve()
        return self._sys

    def apertures(self, fields=None, wavelength=None, *, oversize=1.05):
        """Size auto surface apertures' drawn extents from the ray footprint."""
        from .launch import solve_apertures
        sys = self._sys
        wvl = sys.wavelength(wavelength)
        return solve_apertures(sys, fields=fields, wavelength=wvl,
                               oversize=oversize)

    def vignetting(self, fields=None, wavelength=None, *, tol=1e-3):
        """Solve per-field vignetting factors against the real apertures."""
        from .launch import solve_vignetting
        return solve_vignetting(self._sys, fields=fields,
                                wavelength=wavelength, tol=tol)


class _PlotNamespace:
    """Plotting verbs under sys.plot."""

    __slots__ = ('_sys',)

    def __init__(self, system):
        self._sys = system

    def layout_2d(self, **kwargs):
        """Draw a 2D layout of the optics and rays."""
        from . import plotting
        return plotting.layout(self._sys, **kwargs)

    def spots(self, *, fields=None, wavelengths=None, sampling=None,
              epd=None, reference='centroid', **plot_kwargs):
        """Spot diagrams for the system fields and wavelengths."""
        from . import plotting
        from .analysis import spot_diagrams
        grid = self._sys._cached_grid('spots', spot_diagrams, dict(
            fields=fields, wavelengths=wavelengths, sampling=sampling,
            epd=epd, reference=reference))
        return plotting.plot_spot_diagrams(grid, **plot_kwargs)

    def ray_fans(self, *, fields=None, wavelengths=None, nrays=21,
                 epd=None, distribution='uniform', reference='chief',
                 **plot_kwargs):
        """Transverse ray-aberration fans for the system."""
        from . import plotting
        from .analysis import ray_aberration_fans
        grid = self._sys._cached_grid('ray_fans', ray_aberration_fans, dict(
            fields=fields, wavelengths=wavelengths, nrays=nrays, epd=epd,
            distribution=distribution, reference=reference))
        return plotting.plot_ray_fans(grid, **plot_kwargs)

    def opd_fans(self, *, fields=None, wavelengths=None, nrays=21,
                 epd=None, distribution='uniform', stop_index=None,
                 output='waves', **plot_kwargs):
        """Wavefront OPD fans for the system."""
        from . import plotting
        from .analysis import opd_fans
        grid = self._sys._cached_grid('opd_fans', opd_fans, dict(
            fields=fields, wavelengths=wavelengths, nrays=nrays, epd=epd,
            distribution=distribution, stop_index=stop_index, output=output))
        return plotting.plot_opd_fans(grid, **plot_kwargs)

    def field_curvature(self, *, fields=None, wavelength=None,
                        samples=101, **plot_kwargs):
        """Field-curvature curves for the system."""
        from . import plotting
        from .analysis import field_curvature
        grid = self._sys._cached_grid('field_curvature', field_curvature, dict(
            fields=fields, wavelength=wavelength, samples=samples))
        return plotting.plot_field_curvature(
            self._sys, result=grid, **plot_kwargs)

    def distortion(self, *, fields=None, wavelength=None, epd=None,
                   distortion_type='f-tan', pupil_z=None, samples=101,
                   **plot_kwargs):
        """Percent-distortion curve for the system."""
        from . import plotting
        from .analysis import distortion
        grid = self._sys._cached_grid('distortion', distortion, dict(
            fields=fields, wavelength=wavelength, epd=epd,
            distortion_type=distortion_type, pupil_z=pupil_z,
            samples=samples))
        return plotting.plot_distortion(
            self._sys, result=grid, **plot_kwargs)

    def chromatic_focal_shift(self, *, wavelengths=None,
                              reference_wavelength=None, focus='best',
                              epd=None, field=None, sampling=None,
                              samples=101, **plot_kwargs):
        """Chromatic focal-shift curve for the system."""
        from . import plotting
        from .analysis import chromatic_focal_shift
        data = self._sys._cached_grid(
            'chromatic_focal_shift', chromatic_focal_shift, dict(
                wavelengths=wavelengths,
                reference_wavelength=reference_wavelength, focus=focus,
                epd=epd, field=field, sampling=sampling, samples=samples))
        return plotting.plot_chromatic_focal_shift(
            self._sys, result=data, **plot_kwargs)

    def lateral_color(self, *, fields=None, wavelengths=None, epd=None,
                      samples=101, **plot_kwargs):
        """Lateral-color curves for non-reference wavelengths."""
        from . import plotting
        from .analysis import lateral_color
        data = self._sys._cached_grid('lateral_color', lateral_color, dict(
            fields=fields, wavelengths=wavelengths, epd=epd, samples=samples))
        return plotting.plot_lateral_color(
            self._sys, fields, wavelengths, result=data, samples=samples,
            **plot_kwargs)

    def full_field(self, *, metric='rms spot', samples=15,
                   max_field=None, wavelengths=None, sampling=None,
                   epd=None, stop_index=None, **plot_kwargs):
        """2D metric map over the system field disc."""
        from . import plotting
        from .analysis import full_field
        grid = self._sys._cached_grid('full_field', full_field, dict(
            metric=metric, samples=samples, max_field=max_field,
            wavelengths=wavelengths, sampling=sampling, epd=epd,
            stop_index=stop_index))
        return plotting.plot_full_field(grid, **plot_kwargs)


class _AnalysisNamespace:
    """Analysis verbs under sys.analysis."""

    __slots__ = ('_sys',)

    def __init__(self, system):
        self._sys = system

    def wavefront(self, P, S, wavelength=None, **kwargs):
        """Trace and compute OPD on the chief-ray reference sphere."""
        from .analysis import wavefront
        return wavefront(self._sys, P, S, self._sys.wavelength(wavelength),
                         **kwargs)

    def spot_diagrams(self, **kwargs):
        """Spot diagrams over the system fields and wavelengths."""
        from .analysis import spot_diagrams
        return spot_diagrams(self._sys, **kwargs)

    def ray_aberration_fans(self, **kwargs):
        """Transverse ray-aberration fans."""
        from .analysis import ray_aberration_fans
        return ray_aberration_fans(self._sys, **kwargs)

    def opd_fans(self, **kwargs):
        """Wavefront OPD fans."""
        from .analysis import opd_fans
        return opd_fans(self._sys, **kwargs)

    def distortion(self, **kwargs):
        """Percent distortion over the field."""
        from .analysis import distortion
        return distortion(self._sys, **kwargs)

    def field_curvature(self, **kwargs):
        """Tangential / sagittal field curvature."""
        from .analysis import field_curvature
        return field_curvature(self._sys, **kwargs)

    def lateral_color(self, **kwargs):
        """Lateral color over the field for non-reference wavelengths."""
        from .analysis import lateral_color
        return lateral_color(self._sys, **kwargs)

    def chromatic_focal_shift(self, **kwargs):
        """Chromatic focal shift versus wavelength."""
        from .analysis import chromatic_focal_shift
        return chromatic_focal_shift(self._sys, **kwargs)

    def full_field(self, **kwargs):
        """2D image-quality metric map over the field disc."""
        from .analysis import full_field
        return full_field(self._sys, **kwargs)

    def first_order(self, field=0, wavelength=None, **kwargs):
        """Parabasal first-order properties about a chief ray."""
        return self._sys.first_order(field=field, wavelength=wavelength,
                                     **kwargs)

    def exit_pupil(self, wavelength=None, field=None, **kwargs):
        """Resolved exit-pupil reference point (or None if telecentric)."""
        return self._sys.exit_pupil(wavelength, field=field, **kwargs)


class _TolNamespace:
    """Tolerancing verbs under sys.tol."""

    __slots__ = ('_sys',)

    def __init__(self, system):
        self._sys = system

    def sensitivity(self, perturbations, merit, *, step=None):
        """Centered finite-difference scalar-merit sensitivity table."""
        from .tolerance import sensitivity_table
        return sensitivity_table(self._sys, perturbations, merit, step=step)

    def monte_carlo(self, perturbations, merit, n_trials, **kwargs):
        """Monte Carlo sampling of a scalar merit over perturbations."""
        from .tolerance import monte_carlo
        return monte_carlo(self._sys, perturbations, merit, n_trials, **kwargs)

    def inverse_sensitivity(self, J, budget, **kwargs):
        """Per-tolerance steps that fit a sensitivity Jacobian to a budget."""
        from .adjoint.tolerance_analysis import inverse_sensitivity
        return inverse_sensitivity(J, budget, **kwargs)

    def wavefront(self, perturbations, P, S, wavelength=None, **kwargs):
        """Forward-mode wavefront differential (Code V TOR) for one bundle."""
        from .wavefront_differential import wavefront_differential
        return wavefront_differential(
            self._sys, perturbations, P, S, self._sys.wavelength(wavelength),
            **kwargs)
