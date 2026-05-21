"""Ergonomic sequential prescription wrapper."""

from prysm.mathops import np, optimize

from . import analysis
from . import materials as _materials
from .design import (
    Distortion,
    EFL,
    Problem,
    RmsSpotRadius,
    position_of,
    radius_of,
)
from .launch import Field, Sampling, launch
from .opt import rms_spot_radius
from .paraxial import (
    effective_focal_length,
    first_order,
    paraxial_image_distance,
)
from .plotting import (
    plot_optics,
    plot_ray_paths,
    plot_wave_aberration_fan as _plot_wave_aberration_fan,
)
from .spencer_and_murty import STYPE_EVAL, raytrace
from .surfaces import ConicSag, PlaneSag, Surface, circular_aperture


class Prescription:
    """Sequential optical prescription with common analysis defaults.

    This is a convenience layer over the existing list-of-Surface API.
    The underlying surfaces remain available via .surfaces and the
    object implements the basic sequence protocol, so existing raytracing
    functions can consume a Prescription directly.

    Parameters
    ----------
    surfaces : sequence of Surface
        Sequential surfaces.
    epd : float, optional
        Entrance pupil diameter.
    fields : sequence, optional
        Field objects, y-field angles in degrees, or (hx, hy) pairs.
    wavelengths : mapping or sequence, optional
        Wavelengths in microns.  Mappings provide named wavelengths.
    reference_wavelength : str or float, optional
        Default wavelength for paraxial solves and plots.
    n_ambient : float, optional
        Ambient index in object and image space.  Default is 1.

    """

    def __init__(self, surfaces, *, epd=None, fields=None, wavelengths=None,
                 reference_wavelength=None, n_ambient=1.0):
        self.surfaces = list(surfaces)
        self.epd = epd
        self.fields = _coerce_fields(fields)
        self.wavelengths = _coerce_wavelengths(wavelengths)
        if reference_wavelength is None and self.wavelengths:
            reference_wavelength = next(iter(self.wavelengths))
        self.reference_wavelength = reference_wavelength
        self.n_ambient = float(n_ambient)

    def __len__(self):
        return len(self.surfaces)

    def __iter__(self):
        return iter(self.surfaces)

    def __getitem__(self, item):
        return self.surfaces[item]

    def append(self, surface):
        """Append a surface to the prescription."""
        self.surfaces.append(surface)

    @classmethod
    def refractive_sequence(cls, radii, thicknesses, materials, *,
                            semidiameter, epd=None, fields=None,
                            wavelengths=None, reference_wavelength=None,
                            image_z=None, image_semidiameter=None,
                            n_ambient=1.0):
        """Build a spherical refractive sequence from lens-design rows.

        materials is the post-surface medium for each radius.  For a
        cemented N-BK7/N-F2 doublet this is [N_BK7, N_F2, air].

        """
        radii = list(radii)
        thicknesses = list(thicknesses)
        post_materials = list(materials)
        if len(post_materials) != len(radii):
            raise ValueError('materials must have one entry per radius')
        if len(thicknesses) != len(radii) - 1:
            raise ValueError('thicknesses must have len(radii) - 1 entries')

        surfaces = []
        z = 0.0
        for j, (radius, material) in enumerate(zip(radii, post_materials)):
            surfaces.append(
                Surface(
                    shape=ConicSag(1.0 / float(radius), 0.0),
                    typ='refr',
                    P=[0.0, 0.0, z],
                    n=material,
                    bounding={'outer_radius': float(semidiameter)},
                    aperture=circular_aperture(float(semidiameter)),
                )
            )
            if j < len(thicknesses):
                z += float(thicknesses[j])

        out = cls(
            surfaces, epd=epd, fields=fields, wavelengths=wavelengths,
            reference_wavelength=reference_wavelength,
            n_ambient=n_ambient,
        )
        if image_z is not None:
            out.set_image_z(image_z, semidiameter=image_semidiameter)
        return out

    @property
    def lens_surfaces(self):
        """Surfaces before any final evaluation plane."""
        if self.surfaces and self.surfaces[-1].typ == STYPE_EVAL:
            return self.surfaces[:-1]
        return self.surfaces

    @property
    def image_surface(self):
        """Final evaluation surface, or None if no image plane exists."""
        if self.surfaces and self.surfaces[-1].typ == STYPE_EVAL:
            return self.surfaces[-1]
        return None

    def copy(self):
        """Return a shallow copy that shares surface objects."""
        return type(self)(
            list(self.surfaces), epd=self.epd, fields=list(self.fields),
            wavelengths=dict(self.wavelengths),
            reference_wavelength=self.reference_wavelength,
            n_ambient=self.n_ambient,
        )

    def wavelength(self, wavelength=None):
        """Resolve a wavelength name or scalar to microns."""
        if wavelength is None:
            wavelength = self.reference_wavelength
        if isinstance(wavelength, str):
            return float(self.wavelengths[wavelength])
        return float(wavelength)

    def field(self, field=None):
        """Resolve a field index, scalar y angle, tuple, or Field."""
        if field is None:
            if not self.fields:
                return Field(0.0, 0.0)
            return self.fields[0]
        if isinstance(field, int):
            return self.fields[field]
        return _coerce_field(field)

    def first_order(self, wavelength=None):
        """Return paraxial first-order properties."""
        return first_order(
            self.lens_surfaces, wvl=self.wavelength(wavelength),
            n_ambient=self.n_ambient, epd=self.epd,
        )

    def efl(self, wavelength=None):
        """Return effective focal length."""
        return float(effective_focal_length(
            self.lens_surfaces, wvl=self.wavelength(wavelength),
            n_ambient=self.n_ambient,
        ))

    def paraxial_image_z(self, wavelength=None):
        """Return lab-frame paraxial image z for collimated input."""
        lens = self.lens_surfaces
        dz = paraxial_image_distance(
            lens, wvl=self.wavelength(wavelength),
            n_ambient=self.n_ambient,
        )
        return float(lens[-1].P[2]) + float(dz)

    def set_image_z(self, z, *, semidiameter=None):
        """Set or append the final image evaluation plane."""
        if semidiameter is None:
            semidiameter = max(1.0, float(self.epd or 1.0) / 5.0)
        surface = self.image_surface
        if surface is None:
            self.surfaces.append(
                Surface(
                    shape=PlaneSag(),
                    typ='eval',
                    P=[0.0, 0.0, float(z)],
                    n=_materials.air,
                    bounding={'outer_radius': float(semidiameter)},
                )
            )
        else:
            surface.P[2] = float(z)
        return float(z)

    def solve_paraxial_image(self, wavelength=None, *, semidiameter=None):
        """Move the image plane to the paraxial image and return its z."""
        z = self.paraxial_image_z(wavelength)
        return self.set_image_z(z, semidiameter=semidiameter)

    def set_focal_length(self, target, *, wavelength=None, surfaces=None):
        """Scale selected radii until the paraxial EFL equals target.

        This is a first-order convenience for starting designs.  For real
        design work, use an optimization problem with an EFL constraint.

        """
        wavelength = self.wavelength(wavelength)
        if surfaces is None:
            surfaces = range(len(self.lens_surfaces))
        surfaces = list(surfaces)
        base_radii = [_surface_radius(self.lens_surfaces[j])
                      for j in surfaces]

        def set_scale(scale):
            for j, radius in zip(surfaces, base_radii):
                _set_surface_radius(self.lens_surfaces[j], radius * scale)

        def residual(scale):
            set_scale(scale)
            return self.efl(wavelength) - float(target)

        lo = 0.05
        hi = 20.0
        flo = residual(lo)
        fhi = residual(hi)
        if flo * fhi > 0:
            result = optimize.minimize_scalar(
                lambda s: residual(s) ** 2, bounds=(lo, hi),
                method='bounded',
            )
            set_scale(float(result.x))
        else:
            set_scale(float(optimize.brentq(residual, lo, hi)))
        return self.efl(wavelength)

    def launch(self, *, field=None, wavelength=None, sampling=None,
               pupil_z=0.0, **sampling_kwargs):
        """Build launch (P, S) arrays using prescription defaults."""
        if sampling is None:
            sampling = Sampling.hex(nrings=sampling_kwargs.pop('nrings', 5))
        elif isinstance(sampling, str):
            sampling = _sampling_from_string(sampling, sampling_kwargs)
        return launch(
            self.surfaces, self.field(field), self.wavelength(wavelength),
            sampling, epd=self.epd, n_ambient=self.n_ambient,
            pupil_z=pupil_z,
        )

    def trace(self, *, field=None, wavelength=None, sampling=None,
              pupil_z=0.0, **sampling_kwargs):
        """Trace a launch bundle through the prescription."""
        wvl = self.wavelength(wavelength)
        P, S = self.launch(
            field=field, wavelength=wvl, sampling=sampling,
            pupil_z=pupil_z, **sampling_kwargs,
        )
        return raytrace(self.surfaces, P, S, wvl, n_ambient=self.n_ambient)

    def rms_spot(self, *, field=None, wavelength=None, sampling=None,
                 pupil_z=0.0, centroid=None, **sampling_kwargs):
        """Return RMS spot radius at the image plane."""
        trace = self.trace(
            field=field, wavelength=wavelength, sampling=sampling,
            pupil_z=pupil_z, **sampling_kwargs,
        )
        return rms_spot_radius(trace.P[-1], trace.status, centroid=centroid)

    def optimize_focus(self, *, fields=None, wavelength=None, sampling=None,
                       span=5.0, pupil_z=0.0, **sampling_kwargs):
        """Move image plane to the z that minimizes RMS spot size."""
        if fields is None:
            fields = [self.field(0)]
        else:
            fields = [self.field(f) for f in _as_list(fields)]
        if sampling is None:
            sampling = Sampling.hex(nrings=sampling_kwargs.pop('nrings', 6))
        elif isinstance(sampling, str):
            sampling = _sampling_from_string(sampling, sampling_kwargs)
        wvl = self.wavelength(wavelength)
        center = self.image_surface.P[2] if self.image_surface is not None else self.paraxial_image_z(wvl)

        def merit(z):
            self.set_image_z(z)
            spots = []
            for field in fields:
                spots.append(self.rms_spot(
                    field=field, wavelength=wvl, sampling=sampling,
                    pupil_z=pupil_z,
                ))
            spots = np.asarray(spots)
            return float(np.sqrt(np.mean(spots * spots)))

        result = optimize.minimize_scalar(
            merit, bounds=(center - span, center + span),
            method='bounded', options={'xatol': 1e-6},
        )
        self.set_image_z(float(result.x))
        return result

    def transverse_ray_fan(self, *, field=None, wavelength=None, axis='y',
                           n=41, pupil_z=0.0):
        """Return pupil coordinate and transverse ray aberration."""
        trace = self.trace(
            field=field, wavelength=wavelength,
            sampling=Sampling.fan(n=n, axis=axis), pupil_z=pupil_z,
        )
        return analysis.transverse_ray_aberration(
            trace.P, axis=axis, status=trace.status,
        )

    def wave_aberration_fan(self, *, field=None, wavelength=None, axis='y',
                            n=41, pupil_z=0.0, units='waves',
                            detrend=True):
        """Return normalized pupil coordinate and OPD fan values."""
        wvl = self.wavelength(wavelength)
        P, S = self.launch(
            field=field, wavelength=wvl, sampling=Sampling.fan(n=n, axis=axis),
            pupil_z=pupil_z,
        )
        opd, xp, yp = analysis.wavefront(
            self.surfaces, P, S, wvl, n_ambient=self.n_ambient,
        )
        coord = xp if axis == 'x' else yp
        coord = coord / (float(self.epd) / 2.0)
        order = np.argsort(coord)
        coord = coord[order]
        opd = opd[order]
        units = units.lower()
        if units in ('wave', 'waves'):
            opd = opd / wvl
        elif units in ('nm', 'nanometer', 'nanometers'):
            opd = opd * 1e3
        else:
            raise ValueError("units must be 'waves' or 'nm'")
        if detrend:
            valid = np.isfinite(coord) & np.isfinite(opd)
            if np.count_nonzero(valid) >= 2:
                slope, intercept = np.polyfit(coord[valid], opd[valid], 1)
                opd = opd - (slope * coord + intercept)
        return coord, opd

    def plot_layout(self, *, fields=None, wavelength=None, n=11, axis='y',
                    pupil_z=0.0, fig=None, ax=None):
        """Plot optics and ray paths."""
        import matplotlib.pyplot as plt

        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(8.5, 3.8))
        if fields is None:
            fields = [0]
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for idx, field in enumerate(_as_list(fields)):
            trace = self.trace(
                field=field, wavelength=wavelength,
                sampling=Sampling.fan(n=n, axis=axis), pupil_z=pupil_z,
            )
            if idx == 0:
                plot_optics(
                    self.surfaces, trace, wvl=self.wavelength(wavelength),
                    ambient_index=self.n_ambient, fig=fig, ax=ax,
                    c='0.15', lw=1.4,
                )
            color = colors[idx % len(colors)]
            plot_ray_paths(trace, fig=fig, ax=ax, c=color, alpha=0.75, lw=1.0)
            resolved = self.field(field)
            ax.plot([], [], c=color, label=f'{resolved.hy:g} deg')
        ax.set_xlabel('z [mm]')
        ax.set_ylabel(f'{axis} [mm]')
        ax.set_aspect('equal', adjustable='datalim')
        ax.grid(alpha=0.2)
        return fig, ax

    def plot_transverse_ray_fan(self, *, fields=None, wavelengths=None,
                                axis='y', n=41, fig=None, ax=None):
        """Plot transverse ray aberration fans."""
        import matplotlib.pyplot as plt

        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(6.8, 4.2))
        for field in _as_list(fields if fields is not None else [0]):
            for label, wvl in self._wavelength_items(wavelengths):
                pupil, delta = self.transverse_ray_fan(
                    field=field, wavelength=wvl, axis=axis, n=n,
                )
                field_obj = self.field(field)
                ax.plot(pupil, 1e3 * delta,
                        label=f'{field_obj.hy:g} deg, {label}')
        ax.axhline(0, color='0.2', lw=0.8)
        ax.set_xlabel(f'pupil {axis} [mm]')
        ax.set_ylabel(f'image {axis} minus chief [um]')
        ax.grid(alpha=0.25)
        return fig, ax

    def plot_wave_aberration_fan(self, *, fields=None, wavelengths=None,
                                 axis='y', n=41, units='waves',
                                 detrend=True, fig=None, ax=None):
        """Plot OPD fan."""
        import matplotlib.pyplot as plt

        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(6.8, 4.2))
        for field in _as_list(fields if fields is not None else [0]):
            for label, wvl in self._wavelength_items(wavelengths):
                coord, opd_waves = self.wave_aberration_fan(
                    field=field, wavelength=wvl, axis=axis, n=n,
                    units='waves',
                )
                field_obj = self.field(field)
                _plot_wave_aberration_fan(
                    coord, opd_waves * wvl, wavelength=wvl, units=units,
                    detrend=detrend, axis=axis,
                    label=f'{field_obj.hy:g} deg, {label}', fig=fig, ax=ax,
                )
        ax.axhline(0, color='0.2', lw=0.8)
        ax.grid(alpha=0.25)
        return fig, ax

    def optimization_problem(self):
        """Return a small builder around design.Problem."""
        return DesignSession(self)

    def _wavelength_items(self, wavelengths):
        if wavelengths is None:
            wavelengths = [self.reference_wavelength]
        for item in _as_list(wavelengths):
            label = item if isinstance(item, str) else f'{float(item):g} um'
            yield label, self.wavelength(item)


class DesignSession:
    """Small builder for common prescription optimization problems."""

    def __init__(self, prescription):
        self.prescription = prescription
        self.variables = []
        self.operands = []
        self.history = []
        self.problem = None
        self.result = None

    def vary_radius(self, surface_index):
        """Add one surface radius as a variable."""
        self.variables.append(radius_of(self.prescription.lens_surfaces[surface_index]))
        return self

    def vary_radii(self, surface_indices):
        """Add multiple surface radii as variables."""
        for idx in surface_indices:
            self.vary_radius(idx)
        return self

    def vary_image_z(self):
        """Add final image-plane z as a variable."""
        if self.prescription.image_surface is None:
            self.prescription.solve_paraxial_image()
        self.variables.append(position_of(self.prescription.image_surface, 2))
        return self

    def add_rms_spot(self, *, fields='all', wavelengths='all',
                     sampling=None, weight=1.0, focus=None):
        """Add RMS spot operands for field/wavelength combinations."""
        if fields == 'all':
            fields = range(len(self.prescription.fields))
        if wavelengths == 'all':
            wavelengths = list(self.prescription.wavelengths)
        if sampling is None:
            sampling = Sampling.hex(nrings=3)
        for field in _as_list(fields):
            for wavelength in _as_list(wavelengths):
                wvl = self.prescription.wavelength(wavelength)
                P, S = self.prescription.launch(
                    field=field, wavelength=wvl, sampling=sampling,
                )
                cls = _ParaxialFocusRmsSpotRadius if focus == 'paraxial' else RmsSpotRadius
                if focus == 'paraxial':
                    op = cls(
                        P, S, wvl, self.prescription,
                        target=0.0, weight=weight,
                        n_ambient=self.prescription.n_ambient,
                    )
                else:
                    op = cls(
                        P, S, wvl, target=0.0, weight=weight,
                        n_ambient=self.prescription.n_ambient,
                    )
                self.operands.append(op)
        return self

    def constrain_efl(self, target, *, wavelength=None, weight=1.0):
        """Constrain effective focal length."""
        self.operands.append(EFL(
            self.prescription.wavelength(wavelength), target=target,
            weight=weight, n_ambient=self.prescription.n_ambient,
        ))
        return self

    def constrain_distortion(self, *, field, wavelength=None, target=0.0,
                             weight=1.0):
        """Constrain percent distortion at one field."""
        self.operands.append(Distortion(
            self.prescription.field(field),
            self.prescription.wavelength(wavelength),
            epd=self.prescription.epd,
            target=target,
            weight=weight,
            n_ambient=self.prescription.n_ambient,
        ))
        return self

    def x0(self):
        """Return the current variable vector."""
        return self._problem().x0()

    def solve(self, *, bounds=None, max_nfev=80, **kwargs):
        """Run scipy least_squares and update the prescription in place."""
        problem = self._problem()
        x0 = problem.x0()

        def residuals(x):
            r = problem.residuals(x)
            self.history.append(float(np.sum(r * r)))
            return r

        opts = dict(
            x_scale=np.maximum(np.abs(x0), 1.0),
            max_nfev=max_nfev,
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
        )
        opts.update(kwargs)
        if bounds is None:
            result = optimize.least_squares(residuals, x0, **opts)
        else:
            result = optimize.least_squares(residuals, x0, bounds=bounds, **opts)
        problem.residuals(result.x)
        self.result = result
        return result

    def _problem(self):
        if self.problem is None:
            self.problem = Problem(
                self.prescription.surfaces, self.variables, self.operands,
            )
        return self.problem


class _ParaxialFocusRmsSpotRadius(RmsSpotRadius):
    def __init__(self, P, S, wavelength, prescription, **kwargs):
        super().__init__(P, S, wavelength, **kwargs)
        self._wrapped_prescription = prescription

    def __call__(self, prescription, cache):
        self._wrapped_prescription.solve_paraxial_image()
        return super().__call__(prescription, cache)


def _coerce_field(field):
    if isinstance(field, Field):
        return field
    if np.isscalar(field):
        return Field(0.0, float(field))
    return Field(float(field[0]), float(field[1]))


def _coerce_fields(fields):
    if fields is None:
        return []
    return [_coerce_field(field) for field in fields]


def _coerce_wavelengths(wavelengths):
    if wavelengths is None:
        return {}
    if hasattr(wavelengths, 'items'):
        return dict(wavelengths)
    return {str(i): float(w) for i, w in enumerate(wavelengths)}


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return [value]
    try:
        return list(value)
    except TypeError:
        return [value]


def _sampling_from_string(name, kwargs):
    name = name.lower()
    if name == 'chief':
        return Sampling.chief()
    if name == 'fan':
        return Sampling.fan(
            n=kwargs.pop('n', 11), axis=kwargs.pop('axis', 'y'),
        )
    if name == 'cross':
        return Sampling.cross(n=kwargs.pop('n', 11))
    if name == 'rect':
        return Sampling.rect(n=kwargs.pop('n', 21))
    if name == 'hex':
        return Sampling.hex(nrings=kwargs.pop('nrings', 5))
    if name == 'spiral':
        return Sampling.spiral(nrings=kwargs.pop('nrings', 5))
    raise ValueError(f'unknown sampling {name!r}')


def _surface_radius(surface):
    return 1.0 / float(surface.params['c'])


def _set_surface_radius(surface, radius):
    surface.params['c'] = 1.0 / float(radius)


__all__ = ['Prescription', 'DesignSession']
