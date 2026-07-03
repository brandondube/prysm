"""Surface containers, shape objects, and calculus for raytracing."""

import warnings

from prysm.conf import config
from prysm.coordinates import (
    apply_tilt_decenter,
    coerce_3d_rotation,
    promote_3d_point,
)
from prysm.mathops import np
from prysm.polynomials import (
    cheby1_2d_sum,
    cheby1_2d_sum_der_xy,
    jacobi_radial_sum,
    jacobi_radial_sum_der_xy,
    xy_sum,
    xy_sum_der_xy,
    zernike_sum,
    zernike_sum_der_xy,
)

from .spencer_and_murty import (
    STYPE_EVAL,
    STYPE_OBJ,
    STYPE_IMG,
    STYPE_REFLECT,
    STYPE_REFRACT,
    _is_measurement_surf,
    STATUS_OK,
    STATUS_MISS,
    STATUS_NEWTON,
    STATUS_CLIP,
    STATUS_TIR,
    STATUS_EVANESCENT,
    refract,
    reflect,
    transform_to_local_coords,
    transform_to_global_coords,
)
from .intersections import (
    MARCH_RADIUS_MARGIN,
    SURFACE_INTERSECTION_DEFAULT_MAXITER,
    ConicSeedMixin,
    newton_intersect,
    ray_conic_intersect,
    ray_plane_intersect,
    ray_sphere_intersect,
)
from .aperture import (
    annular_aperture,
    as_aperture,
    circular_aperture,
)
from .phase import PhaseFunction
from .sags import (
    Q2d_and_der,
    Q2d_sag,
    _add_conic_base_derivatives,
    _add_conic_base_sag,
    _conic_base_xy_sag,
    conic_sag,
    conic_sag_and_normal,
    conic_sag_der,
    conic_sag_der_xy,
    conic_sag_hessian,
    conic_sag_param_partials,
    der_direction_cosine_conic,
    even_asphere_sag,
    even_asphere_sag_der_xy,
    fd_step,
    gradient_to_unit_normal,
    plane_sag_and_normal,
    phi_conic,
    product_rule,
    sphere_sag,
    sphere_sag_der,
)


# Sample count per axis for the departure-band precompute (Surface
# departure_band); the max-departure estimate is padded to absorb the
# grid resolution.
DEPARTURE_BAND_SAMPLES = 64
# Max |grad(sag - seed conic sag)| above which the acceptance band can admit
# multiple surface crossings: the crossing spacing scale is ~D/G, the band
# width is ~2D, so G >= 0.5 makes the first-root selection ambiguous.
DEPARTURE_GRADIENT_WARN = 0.5


class DepartureBand:
    """Conic-seed departure bounds for the intersection first-root guarantee.

    bounded is False (numeric fields None) for an analytic shape or a
    conic-seed with no characterizable domain.

    Attributes
    ----------
    max_departure : float
        padded max sag departure from the seed conic over the domain
    domain_radius : float
        disk radius the band was characterized on
    gradient_bound : float
        departure-slope bound for the monotonicity certificate
    lipschitz : float
        sag-slope bound for the Lipschitz-march rescue
    """

    __slots__ = ('bounded', 'max_departure', 'domain_radius',
                 'gradient_bound', 'lipschitz')

    def __init__(self, *, bounded, max_departure=None, domain_radius=None,
                 gradient_bound=None, lipschitz=None):
        self.bounded = bounded
        self.max_departure = max_departure
        self.domain_radius = domain_radius
        self.gradient_bound = gradient_bound
        self.lipschitz = lipschitz

    @classmethod
    def unbounded(cls):
        """A band with no finite bound (analytic shape / no conic domain)."""
        return cls(bounded=False)

    def __repr__(self):
        if not self.bounded:
            return 'DepartureBand(bounded=False)'
        return (f'DepartureBand(max_departure={self.max_departure:g}, '
                f'domain_radius={self.domain_radius:g}, '
                f'gradient_bound={self.gradient_bound:g}, '
                f'lipschitz={self.lipschitz:g})')


def _map_stype(typ):
    """Map a surface interaction name or integer to an STYPE constant."""
    if isinstance(typ, int):
        return typ
    typ_lc = typ.lower()
    if typ_lc in ('refl', 'reflect'):
        return STYPE_REFLECT
    if typ_lc in ('refr', 'refract'):
        return STYPE_REFRACT
    if typ_lc == 'eval':
        return STYPE_EVAL
    if typ_lc in ('obj', 'object'):
        return STYPE_OBJ
    if typ_lc in ('img', 'image'):
        return STYPE_IMG
    raise ValueError(
        f'unknown surface type {typ!r}; expected one of '
        "'refl'/'reflect', 'refr'/'refract', 'eval', 'object', 'image', or an "
        'STYPE_* int.'
    )


class Shape:
    """Base class for sag-bearing shape objects."""

    analytic_intersect = False
    finite_difference_step = None

    # Self-describing DOF layout.  Editable shapes override these and add a
    # from_params classmethod; LensData reads them off the class, and the
    # presence of from_params is what "registers" a shape (deepening 02).
    SCALAR_DOFS = ()
    VECTOR_DOFS = ()
    META_KEYS = ()
    CATEGORIES = {}

    def __init__(self, **params):
        """Initialize a shape with parameter storage.

        Parameters
        ----------
        **params
            Shape parameters exposed both through params and attribute
            lookup.

        """
        self.params = params or None

    def __getattr__(self, name):
        """Look up shape parameters as attributes."""
        params = self.__dict__.get('params')
        if params is not None and name in params:
            return params[name]
        raise AttributeError(name)

    def sag(self, x, y):
        """Evaluate surface sag at local coordinates.

        Parameters
        ----------
        x, y : ndarray
            Local surface coordinates.

        Returns
        -------
        ndarray
            Surface sag.

        """
        raise NotImplementedError

    def sag_and_normal(self, x, y):
        """Sag and unit surface normal at x, y.

        The base implementation differentiates sag with central finite
        differences; shapes with closed-form derivatives override this.

        Parameters
        ----------
        x, y : ndarray
            local surface coordinates.

        Returns
        -------
        z : ndarray
            surface sag at x, y.
        n_hat : ndarray
            shape (..., 3) unit surface normals.

        """
        x = np.asarray(x)
        y = np.asarray(y)
        z = self.sag(x, y)
        if self.finite_difference_step is None:
            try:
                eps = np.sqrt(np.finfo(x.dtype).eps)
            except ValueError:
                eps = np.sqrt(np.finfo(config.precision).eps)
                x = x.astype(config.precision)
                y = y.astype(config.precision)
            h = eps * np.maximum(1.0, np.maximum(np.abs(x), np.abs(y)))
        else:
            h = np.asarray(self.finite_difference_step, dtype=x.dtype)
        Fx = (self.sag(x + h, y) - self.sag(x - h, y)) / (2.0 * h)
        Fy = (self.sag(x, y + h) - self.sag(x, y - h)) / (2.0 * h)
        return z, gradient_to_unit_normal(Fx, Fy)

    def _sag_gradient(self, x, y):
        """Sag gradient (dz/dx, dz/dy) at x, y, via sag_and_normal."""
        _, n_hat = self.sag_and_normal(x, y)
        nz = n_hat[..., 2]
        return -n_hat[..., 0] / nz, -n_hat[..., 1] / nz

    def _fd_step(self, *arrs):
        """Central-difference step, scaled to the coordinate magnitude."""
        return fd_step(self.finite_difference_step, *arrs)

    def sag_hessian(self, x, y):
        """Sag Hessian (sag_xx, sag_xy, sag_yy) at x, y.

        Base implementation central-differences the sag gradient; shapes with
        a closed-form Hessian (Plane, Sphere, Conic) override this.

        """
        x = np.asarray(x)
        y = np.asarray(y)
        h = self._fd_step(x, y)
        fxxp, _ = self._sag_gradient(x + h, y)
        fxxm, _ = self._sag_gradient(x - h, y)
        fxyp, fyyp = self._sag_gradient(x, y + h)
        fxym, fyym = self._sag_gradient(x, y - h)
        sag_xx = (fxxp - fxxm) / (2.0 * h)
        sag_yy = (fyyp - fyym) / (2.0 * h)
        sag_xy = (fxyp - fxym) / (2.0 * h)
        return sag_xx, sag_xy, sag_yy

    def sag_param_partials(self, x, y, name):
        """Partials of sag and sag-gradient wrt a shape parameter at fixed x, y.

        Returns (sag_t, gx_t, gy_t).  The base implementation central-
        differences a scalar parameter stored in self.params -- the local-FD
        fallback that lets freeform-coefficient tolerances reuse the
        differential machinery without a re-trace.  Plane/Sphere/Conic give
        closed forms.

        """
        x = np.asarray(x)
        y = np.asarray(y)
        params = self.params
        if params is None or name not in params:
            raise ValueError(
                f'shape has no parameter {name!r} to differentiate against')
        nominal = params[name]
        h = np.sqrt(np.finfo(config.precision).eps) * max(1.0, abs(float(nominal)))
        try:
            params[name] = nominal + h
            sag_p = self.sag(x, y)
            gxp, gyp = self._sag_gradient(x, y)
            params[name] = nominal - h
            sag_m = self.sag(x, y)
            gxm, gym = self._sag_gradient(x, y)
        finally:
            params[name] = nominal
        sag_t = (sag_p - sag_m) / (2.0 * h)
        gx_t = (gxp - gxm) / (2.0 * h)
        gy_t = (gyp - gym) / (2.0 * h)
        return sag_t, gx_t, gy_t


def _shape_from_params(cls, p):
    """Rebuild a descriptor-declared shape from its stored parameter dict.

    The DOF and meta keys (SCALAR_DOFS + VECTOR_DOFS + META_KEYS) name the shape
    constructor's keyword arguments exactly, so a shape reconstructs by handing
    each stored value back by name -- the single builder behind every editable
    shape's from_params, derived from the self-describing descriptors instead of
    written out per class.  Bound per class (from_params = classmethod(
    _shape_from_params)) so it stays the opt-in registration marker: a shape that
    does not bind it is not editable by LensData.
    """
    keys = cls.SCALAR_DOFS + cls.VECTOR_DOFS + cls.META_KEYS
    return cls(**{key: p[key] for key in keys})


class CallableShape(Shape):
    """Shape wrapper for user-supplied sag and optional normal callables."""

    def __init__(self, sag, sag_and_normal=None, params=None):
        """Create a callable-backed shape.

        Parameters
        ----------
        sag : callable
            Function of x, y returning surface sag.
        sag_and_normal : callable, optional
            Function of x, y returning sag and unit surface normal.
        params : dict, optional
            Metadata or shape parameters to expose through params.

        """
        if sag is None:
            raise TypeError('CallableShape requires sag')
        self._sag = sag
        self._sag_and_normal = sag_and_normal
        self.params = params

    def sag(self, x, y):
        """Evaluate the wrapped sag callable."""
        return self._sag(x, y)

    def sag_and_normal(self, x, y):
        """Evaluate sag and normal from the wrapped callable or finite differences."""
        if self._sag_and_normal is None:
            return super().sag_and_normal(x, y)
        return self._sag_and_normal(x, y)


class Plane(Shape):
    """Plane sag shape for the local surface z = 0."""

    analytic_intersect = True

    SCALAR_DOFS = ()
    VECTOR_DOFS = ()
    META_KEYS = ()
    CATEGORIES = {}

    from_params = classmethod(_shape_from_params)

    def __init__(self):
        """Initialize a plane sag shape (local z = 0; no parameters)."""
        super().__init__()

    def sag(self, x, y):
        """Evaluate zero sag for a plane."""
        return np.zeros_like(x)

    def sag_and_normal(self, x, y):
        """Evaluate plane sag and unit normal."""
        return plane_sag_and_normal(x, y)

    def sag_hessian(self, x, y):
        """Plane sag Hessian (all zero)."""
        z = np.zeros_like(np.asarray(x))
        return z, z, z

    def intersect(self, P, S, sag_and_normal=None, tol_sag=None, maxiter=None):
        """Intersect rays with the plane.

        Parameters
        ----------
        P : ndarray
            Ray origins in the surface local frame.
        S : ndarray
            Unit direction cosines.
        sag_and_normal : callable, optional
            Ignored; accepted for the common shape-intersection interface.
        tol_sag : float, optional
            Ignored convergence tolerances.
        maxiter : int, optional
            Ignored iteration limit.

        Returns
        -------
        Q, n, valid : ndarray, ndarray, ndarray
            Intersection points.
            Unit surface normals, and a Boolean validity mask.

        """
        return ray_plane_intersect(P, S)


class Sphere(Shape):
    """Spherical sag shape.

    Parameters
    ----------
    c : float
        Vertex curvature, reciprocal radius of curvature.

    """

    analytic_intersect = True

    SCALAR_DOFS = ('c',)
    VECTOR_DOFS = ()
    META_KEYS = ()
    CATEGORIES = {'curvature': ['c'], 'radius': ['c']}

    from_params = classmethod(_shape_from_params)

    def __init__(self, c):
        """Initialize a spherical sag shape.

        Parameters
        ----------
        c : float
            vertex curvature, the reciprocal radius of curvature (c = 1/R).
            Pass 1/R, not R; c = 0 is a plane.

        """
        super().__init__(c=c)

    def sag(self, x, y):
        """Evaluate spherical sag at local coordinates."""
        return sphere_sag(self.params['c'], x * x + y * y)

    def sag_and_normal(self, x, y):
        """Evaluate spherical sag and unit normal."""
        return conic_sag_and_normal(self.params['c'], 0.0, x, y)

    def sag_hessian(self, x, y):
        """Spherical sag Hessian (conic Hessian with k=0)."""
        return conic_sag_hessian(self.params['c'], 0.0, x, y)

    def sag_param_partials(self, x, y, name):
        """Partials of sphere sag and gradient wrt 'c'."""
        return conic_sag_param_partials(self.params['c'], 0.0, x, y, name)

    def intersect(self, P, S, sag_and_normal=None, tol_sag=None, maxiter=None):
        """Intersect rays with the sphere.

        Parameters
        ----------
        P : ndarray
            Ray origins in the surface local frame.
        S : ndarray
            Unit direction cosines.
        sag_and_normal : callable, optional
            Ignored; accepted for the common shape-intersection interface.
        tol_sag : float, optional
            Ignored convergence tolerances.
        maxiter : int, optional
            Ignored iteration limit.

        Returns
        -------
        Q, n, valid : ndarray, ndarray, ndarray
            Intersection points.
            Unit surface normals, and a Boolean validity mask.

        """
        return ray_sphere_intersect(P, S, self.params['c'])


class Conic(Shape):
    """Conic sag shape.

    Uses the standard raytracing conic convention: k = -1 is a parabola and
    k = 0 is a sphere.

    Parameters
    ----------
    c : float
        Vertex curvature, reciprocal radius of curvature.
    k : float
        Conic constant.

    """

    analytic_intersect = True

    SCALAR_DOFS = ('c', 'k')
    VECTOR_DOFS = ()
    META_KEYS = ()
    CATEGORIES = {'curvature': ['c'], 'radius': ['c'], 'conic': ['k']}

    from_params = classmethod(_shape_from_params)

    def __init__(self, c, k):
        """Initialize a conic sag shape.

        Parameters
        ----------
        c : float
            vertex curvature, the reciprocal radius of curvature (c = 1/R).
        k : float
            conic constant: 0 a sphere, -1 a parabola, k < -1 a hyperbola,
            -1 < k < 0 a prolate ellipse, k > 0 an oblate ellipse.

        """
        super().__init__(c=c, k=k)

    def sag(self, x, y):
        """Evaluate conic sag at local coordinates."""
        return _conic_base_xy_sag(self.params['c'], self.params['k'], x, y)

    def sag_and_normal(self, x, y):
        """Evaluate conic sag and unit normal."""
        return conic_sag_and_normal(self.params['c'], self.params['k'], x, y)

    def sag_hessian(self, x, y):
        """Conic sag Hessian."""
        return conic_sag_hessian(self.params['c'], self.params['k'], x, y)

    def sag_param_partials(self, x, y, name):
        """Partials of conic sag and gradient wrt 'c' or 'k'."""
        return conic_sag_param_partials(self.params['c'], self.params['k'],
                                        x, y, name)

    def intersect(self, P, S, sag_and_normal=None, tol_sag=None, maxiter=None):
        """Intersect rays with the conic.

        Parameters
        ----------
        P : ndarray
            Ray origins in the surface local frame.
        S : ndarray
            Unit direction cosines.
        sag_and_normal : callable, optional
            Ignored; accepted for the common shape-intersection interface.
        tol_sag : float, optional
            Ignored convergence tolerances.
        maxiter : int, optional
            Ignored iteration limit.

        Returns
        -------
        Q, n, valid : ndarray, ndarray, ndarray
            Intersection points.
            Unit surface normals, and a Boolean validity mask.

        """
        p = self.params
        return ray_conic_intersect(P, S, p['c'], p['k'])


class OffAxisConic(Shape):
    """Off-axis conic sag shape.

    Parameters
    ----------
    c : float
        Parent conic vertex curvature.
    k : float
        Parent conic constant.
    dx, dy : float, optional
        Coordinate offsets from the off-axis surface to the parent conic.

    """

    analytic_intersect = True

    SCALAR_DOFS = ('c', 'k')
    VECTOR_DOFS = ()
    META_KEYS = ('dx', 'dy')
    CATEGORIES = {'curvature': ['c'], 'radius': ['c'], 'conic': ['k']}

    from_params = classmethod(_shape_from_params)

    def __init__(self, c, k, dx=0.0, dy=0.0):
        """Initialize an off-axis conic sag shape."""
        super().__init__(c=c, k=k, dx=dx, dy=dy)

    def sag(self, x, y):
        """Evaluate off-axis conic sag at local coordinates."""
        p = self.params
        X = x + p['dx']
        Y = y + p['dy']
        return conic_sag(p['c'], p['k'], X * X + Y * Y)

    def sag_and_normal(self, x, y):
        """Evaluate off-axis conic sag and unit normal."""
        p = self.params
        return conic_sag_and_normal(p['c'], p['k'], x + p['dx'], y + p['dy'])

    def intersect(self, P, S, sag_and_normal=None, tol_sag=None, maxiter=None):
        """Intersect rays with the off-axis conic.

        Parameters
        ----------
        P : ndarray
            Ray origins in the surface local frame.
        S : ndarray
            Unit direction cosines.
        sag_and_normal : callable, optional
            Ignored; accepted for the common shape-intersection interface.
        tol_sag : float, optional
            Ignored convergence tolerances.
        maxiter : int, optional
            Ignored iteration limit.

        Returns
        -------
        Q, n, valid : ndarray, ndarray, ndarray
            Intersection points.
            Unit surface normals, and a Boolean validity mask.

        """
        p = self.params
        return ray_conic_intersect(P, S, p['c'], p['k'],
                                   dx=p['dx'], dy=p['dy'])


class EvenAsphere(ConicSeedMixin, Shape):
    """Even asphere sag shape with a conic base.

    Parameters
    ----------
    c : float
        Vertex curvature.
    k : float
        Conic constant.
    coefs : sequence of float
        Even asphere coefficients.

    """

    SCALAR_DOFS = ('c', 'k')
    VECTOR_DOFS = ('coefs',)
    META_KEYS = ()
    CATEGORIES = {'curvature': ['c'], 'radius': ['c'], 'conic': ['k'],
                  'coefs': ['coefs']}

    from_params = classmethod(_shape_from_params)

    def __init__(self, c, k, coefs):
        """Initialize an even asphere sag shape.

        Parameters
        ----------
        c : float
            base conic vertex curvature (c = 1/R).
        k : float
            base conic constant (see Conic).
        coefs : sequence of float
            even-power radial coefficients a4, a6, a8, ... multiplying
            r^4, r^6, r^8, ... on top of the conic base; empty for a pure conic.

        """
        coefs = tuple(coefs) if coefs is not None else ()
        super().__init__(c=c, k=k, coefs=coefs)

    def sag(self, x, y):
        """Evaluate even asphere sag at local coordinates."""
        p = self.params
        return even_asphere_sag(p['c'], p['k'], p['coefs'], x * x + y * y)

    def sag_and_normal(self, x, y):
        """Evaluate even asphere sag and unit normal."""
        p = self.params
        rsq = x * x + y * y
        phi = phi_conic(p['c'], p['k'], rsq)
        z = even_asphere_sag(p['c'], p['k'], p['coefs'], rsq)
        dx, dy = even_asphere_sag_der_xy(p['c'], p['k'], p['coefs'],
                                         x, y, phi=phi)
        return z, gradient_to_unit_normal(dx, dy)


class Q2D(ConicSeedMixin, Shape):
    """Q2D asphere sag shape with a conic base.

    Parameters
    ----------
    c : float
        Vertex curvature.
    k : float
        Conic constant.
    normalization_radius : float
        Radius used to normalize the polynomial coordinates.
    cm0 : sequence of float
        Rotationally symmetric Q-polynomial coefficients.
    ams, bms : sequence of sequence of float
        Azimuthal Q-polynomial coefficient groups.
    dx, dy : float, optional
        Coordinate offsets applied during Q2D evaluation.

    """

    SCALAR_DOFS = ('c', 'k')
    VECTOR_DOFS = ()
    META_KEYS = ('normalization_radius', 'cm0', 'ams', 'bms', 'dx', 'dy')
    CATEGORIES = {'curvature': ['c'], 'radius': ['c'], 'conic': ['k']}

    from_params = classmethod(_shape_from_params)

    def __init__(self, c, k, normalization_radius, cm0, ams, bms,
                 dx=0.0, dy=0.0):
        """Initialize a Q2D sag shape."""
        cm0 = tuple(cm0) if cm0 is not None else (0.0,)
        ams = tuple(tuple(am) for am in ams)
        bms = tuple(tuple(bm) for bm in bms)
        super().__init__(c=c, k=k,
                         normalization_radius=float(normalization_radius),
                         cm0=cm0, ams=ams, bms=bms, dx=dx, dy=dy)

    def sag(self, x, y):
        """Evaluate Q2D sag at local coordinates."""
        p = self.params
        return Q2d_sag(p['cm0'], p['ams'], p['bms'],
                       x, y, p['normalization_radius'],
                       p['c'], p['k'], dx=p['dx'], dy=p['dy'])

    def sag_and_normal(self, x, y):
        """Evaluate Q2D sag and unit normal."""
        p = self.params
        z, dr, dt = Q2d_and_der(p['cm0'], p['ams'], p['bms'],
                                x, y, p['normalization_radius'],
                                p['c'], p['k'], dx=p['dx'], dy=p['dy'])
        rsq = x * x + y * y
        r = np.sqrt(rsq)
        on_axis = (r == 0)
        safe_r = np.where(on_axis, 1.0, r)
        cost = x / safe_r
        sint = y / safe_r
        ddx = dr * cost - dt * sint / safe_r
        ddy = dr * sint + dt * cost / safe_r
        if np.any(on_axis):
            ddx = np.where(on_axis, 0.0, ddx)
            ddy = np.where(on_axis, 0.0, ddy)
        return z, gradient_to_unit_normal(ddx, ddy)


class Zernike(ConicSeedMixin, Shape):
    """Zernike polynomial sag shape with a conic base.

    Parameters
    ----------
    c : float
        Vertex curvature.
    k : float
        Conic constant.
    normalization_radius : float
        Radius used to normalize x and y before polynomial evaluation.
    nms : sequence of tuple of int
        Zernike (n, m) mode indices.
    coefs : sequence of float
        Zernike coefficients parallel to nms.
    norm : bool, optional
        If True, use normalized Zernike polynomials.

    """

    SCALAR_DOFS = ('c', 'k')
    VECTOR_DOFS = ('coefs',)
    META_KEYS = ('normalization_radius', 'nms', 'norm')
    CATEGORIES = {'curvature': ['c'], 'radius': ['c'], 'conic': ['k'],
                  'coefs': ['coefs']}

    from_params = classmethod(_shape_from_params)

    def __init__(self, c, k, normalization_radius, nms, coefs, norm=True):
        """Initialize a Zernike sag shape."""
        nms = tuple((int(nn), int(mm)) for nn, mm in nms)
        # coefs are numeric DOFs; keep them tensor-clean (no float() coercion)
        # so they survive an autograd graph when rebuilt from a torch theta.
        coefs = tuple(coefs)
        if len(nms) != len(coefs):
            raise ValueError(
                f'nms and coefs must be parallel; got {len(nms)} and {len(coefs)}'
            )
        super().__init__(c=c, k=k,
                         normalization_radius=float(normalization_radius),
                         nms=nms, coefs=coefs, norm=bool(norm))

    def sag(self, x, y):
        """Evaluate Zernike sag at local coordinates."""
        p = self.params
        norm_r = p['normalization_radius']
        z_p = zernike_sum(p['coefs'], p['nms'],
                          x / norm_r, y / norm_r, norm=p['norm'])
        return _add_conic_base_sag(p['c'], p['k'], x, y, z_p)

    def sag_and_normal(self, x, y):
        """Evaluate Zernike sag and unit normal."""
        p = self.params
        norm_r = p['normalization_radius']
        z_p, ddx_p, ddy_p = zernike_sum_der_xy(
            p['coefs'], p['nms'], x / norm_r, y / norm_r, norm=p['norm'])
        z, ddx, ddy = _add_conic_base_derivatives(
            p['c'], p['k'], x, y, z_p, ddx_p / norm_r, ddy_p / norm_r,
        )
        return z, gradient_to_unit_normal(ddx, ddy)


class XY(ConicSeedMixin, Shape):
    """Power-series x-y polynomial sag shape with a conic base.

    Parameters
    ----------
    c : float
        Vertex curvature.
    k : float
        Conic constant.
    normalization_radius : float
        Radius used to normalize x and y before polynomial evaluation.
    mns : sequence of tuple of int
        Polynomial (m, n) powers.
    coefs : sequence of float
        Polynomial coefficients parallel to mns.

    """

    SCALAR_DOFS = ('c', 'k')
    VECTOR_DOFS = ('coefs',)
    META_KEYS = ('normalization_radius', 'mns')
    CATEGORIES = {'curvature': ['c'], 'radius': ['c'], 'conic': ['k'],
                  'coefs': ['coefs']}

    from_params = classmethod(_shape_from_params)

    def __init__(self, c, k, normalization_radius, mns, coefs):
        """Initialize an x-y polynomial sag shape."""
        mns = tuple((int(mm), int(nn)) for mm, nn in mns)
        # coefs are numeric DOFs; keep them tensor-clean (no float() coercion).
        coefs = tuple(coefs)
        if len(mns) != len(coefs):
            raise ValueError(
                f'mns and coefs must be parallel; got {len(mns)} and {len(coefs)}'
            )
        super().__init__(c=c, k=k,
                         normalization_radius=float(normalization_radius),
                         mns=mns, coefs=coefs)

    def sag(self, x, y):
        """Evaluate x-y polynomial sag at local coordinates."""
        p = self.params
        norm_r = p['normalization_radius']
        z_p = xy_sum(p['coefs'], p['mns'], x / norm_r, y / norm_r,
                     cartesian_grid=False)
        return _add_conic_base_sag(p['c'], p['k'], x, y, z_p)

    def sag_and_normal(self, x, y):
        """Evaluate x-y polynomial sag and unit normal."""
        p = self.params
        norm_r = p['normalization_radius']
        z_p, ddx_p, ddy_p = xy_sum_der_xy(
            p['coefs'], p['mns'], x / norm_r, y / norm_r,
            cartesian_grid=False)
        z, ddx, ddy = _add_conic_base_derivatives(
            p['c'], p['k'], x, y, z_p, ddx_p / norm_r, ddy_p / norm_r,
        )
        return z, gradient_to_unit_normal(ddx, ddy)


class Chebyshev(ConicSeedMixin, Shape):
    """Chebyshev polynomial sag shape with a conic base.

    Parameters
    ----------
    c : float
        Vertex curvature.
    k : float
        Conic constant.
    x_norm, y_norm : float
        Normalization scales for x and y.
    mns : sequence of tuple of int
        Chebyshev (m, n) polynomial orders.
    coefs : sequence of float
        Polynomial coefficients parallel to mns.

    """

    SCALAR_DOFS = ('c', 'k')
    VECTOR_DOFS = ('coefs',)
    META_KEYS = ('x_norm', 'y_norm', 'mns')
    CATEGORIES = {'curvature': ['c'], 'radius': ['c'], 'conic': ['k'],
                  'coefs': ['coefs']}

    from_params = classmethod(_shape_from_params)

    def __init__(self, c, k, x_norm, y_norm, mns, coefs):
        """Initialize a Chebyshev sag shape."""
        mns = tuple((int(mm), int(nn)) for mm, nn in mns)
        # coefs are numeric DOFs; keep them tensor-clean (no float() coercion).
        coefs = tuple(coefs)
        if len(mns) != len(coefs):
            raise ValueError(
                f'mns and coefs must be parallel; got {len(mns)} and {len(coefs)}'
            )
        super().__init__(c=c, k=k, x_norm=float(x_norm), y_norm=float(y_norm),
                         mns=mns, coefs=coefs)

    def sag(self, x, y):
        """Evaluate Chebyshev sag at local coordinates."""
        p = self.params
        z_p = cheby1_2d_sum(p['coefs'], p['mns'],
                            x / p['x_norm'], y / p['y_norm'])
        return _add_conic_base_sag(p['c'], p['k'], x, y, z_p)

    def sag_and_normal(self, x, y):
        """Evaluate Chebyshev sag and unit normal."""
        p = self.params
        xn = p['x_norm']
        yn = p['y_norm']
        z_p, ddx_p, ddy_p = cheby1_2d_sum_der_xy(
            p['coefs'], p['mns'], x / xn, y / yn, xn, yn)
        z, ddx, ddy = _add_conic_base_derivatives(
            p['c'], p['k'], x, y, z_p, ddx_p, ddy_p,
        )
        return z, gradient_to_unit_normal(ddx, ddy)


class Jacobi(ConicSeedMixin, Shape):
    """Radial Jacobi polynomial sag shape with a conic base.

    Parameters
    ----------
    c : float
        Vertex curvature.
    k : float
        Conic constant.
    normalization_radius : float
        Radius used to normalize the radial coordinate.
    alpha, beta : float
        Jacobi polynomial parameters.
    ns : sequence of int
        Jacobi polynomial orders.
    coefs : sequence of float
        Polynomial coefficients parallel to ns.

    """

    SCALAR_DOFS = ('c', 'k')
    VECTOR_DOFS = ('coefs',)
    META_KEYS = ('normalization_radius', 'alpha', 'beta', 'ns')
    CATEGORIES = {'curvature': ['c'], 'radius': ['c'], 'conic': ['k'],
                  'coefs': ['coefs']}

    from_params = classmethod(_shape_from_params)

    def __init__(self, c, k, normalization_radius, alpha, beta, ns, coefs):
        """Initialize a radial Jacobi sag shape."""
        ns = tuple(int(nn) for nn in ns)
        # coefs are numeric DOFs; keep them tensor-clean (no float() coercion).
        coefs = tuple(coefs)
        if len(ns) != len(coefs):
            raise ValueError(
                f'ns and coefs must be parallel; got {len(ns)} and {len(coefs)}'
            )
        super().__init__(c=c, k=k,
                         normalization_radius=float(normalization_radius),
                         alpha=float(alpha), beta=float(beta),
                         ns=ns, coefs=coefs)

    def sag(self, x, y):
        """Evaluate radial Jacobi sag at local coordinates."""
        p = self.params
        z_p = jacobi_radial_sum(p['coefs'], p['ns'], p['alpha'], p['beta'],
                                x, y, p['normalization_radius'])
        return _add_conic_base_sag(p['c'], p['k'], x, y, z_p)

    def sag_and_normal(self, x, y):
        """Evaluate radial Jacobi sag and unit normal."""
        p = self.params
        z_p, ddx_p, ddy_p = jacobi_radial_sum_der_xy(
            p['coefs'], p['ns'], p['alpha'], p['beta'],
            x, y, p['normalization_radius'])
        z, ddx, ddy = _add_conic_base_derivatives(
            p['c'], p['k'], x, y, z_p, ddx_p, ddy_p,
        )
        return z, gradient_to_unit_normal(ddx, ddy)


class Toroid(ConicSeedMixin, Shape):
    """Toroidal sag shape.

    Parameters
    ----------
    c_x, c_y : float
        Curvatures in the x and y directions.
    k_y : float
        Conic constant for the y-direction section.
    coefs_y : sequence of float
        Even-asphere coefficients for the y-direction section.

    """

    SCALAR_DOFS = ('c_x', 'c_y', 'k_y')
    VECTOR_DOFS = ('coefs_y',)
    META_KEYS = ()
    CATEGORIES = {'curvature': ['c_x', 'c_y'], 'conic': ['k_y'],
                  'coefs': ['coefs_y']}

    from_params = classmethod(_shape_from_params)

    def __init__(self, c_x, c_y, k_y, coefs_y):
        """Initialize a toroidal sag shape.

        The x section is a circle of curvature c_x; the y section is an even
        asphere of curvature c_y, conic k_y, and coefficients coefs_y.

        Parameters
        ----------
        c_x, c_y : float
            vertex curvatures (1/R) of the x and y sections.
        k_y : float
            conic constant of the y section.
        coefs_y : sequence of float
            even-asphere coefficients of the y section (r^4, r^6, ...).

        """
        coefs_y = tuple(coefs_y) if coefs_y is not None else ()
        super().__init__(c_x=c_x, c_y=c_y, k_y=k_y, coefs_y=coefs_y)

    def seed_conic(self):
        """Return a conic seed for Newton intersection."""
        p = self.params
        return 0.5 * (p['c_x'] + p['c_y']), 0.0, 0.0, 0.0

    def sag(self, x, y):
        """Evaluate toroidal sag at local coordinates."""
        p = self.params
        z_x = sphere_sag(p['c_x'], x * x)
        z_y = even_asphere_sag(p['c_y'], p['k_y'], p['coefs_y'], y * y)
        return z_x + z_y

    def sag_and_normal(self, x, y):
        """Evaluate toroidal sag and unit normal."""
        p = self.params
        xsq = x * x
        ysq = y * y
        phi_x = phi_conic(p['c_x'], 0.0, xsq)
        z_x = sphere_sag(p['c_x'], xsq, phi=phi_x)
        ddx = (p['c_x'] * x) / phi_x
        zero = np.zeros_like(y)
        z_y = even_asphere_sag(p['c_y'], p['k_y'], p['coefs_y'], ysq)
        _, ddy = even_asphere_sag_der_xy(p['c_y'], p['k_y'],
                                         p['coefs_y'], zero, y)
        return z_x + z_y, gradient_to_unit_normal(ddx, ddy)


class Biconic(ConicSeedMixin, Shape):
    """Biconic sag shape.

    Parameters
    ----------
    c_x, c_y : float
        Curvatures in the x and y directions.
    k_x, k_y : float
        Conic constants in the x and y directions.

    """

    SCALAR_DOFS = ('c_x', 'c_y', 'k_x', 'k_y')
    VECTOR_DOFS = ()
    META_KEYS = ()
    CATEGORIES = {'curvature': ['c_x', 'c_y'], 'conic': ['k_x', 'k_y']}

    from_params = classmethod(_shape_from_params)

    def __init__(self, c_x, c_y, k_x, k_y):
        """Initialize a biconic sag shape."""
        super().__init__(c_x=c_x, c_y=c_y, k_x=k_x, k_y=k_y)

    def seed_conic(self):
        """Return a conic seed for Newton intersection."""
        p = self.params
        c_seed = 0.5 * (p['c_x'] + p['c_y'])
        k_seed = 0.5 * (p['k_x'] + p['k_y'])
        return c_seed, k_seed, 0.0, 0.0

    def sag(self, x, y):
        """Evaluate biconic sag at local coordinates."""
        p = self.params
        c_x = p['c_x']
        c_y = p['c_y']
        xsq = x * x
        ysq = y * y
        phi = np.sqrt(1 - (1.0 + p['k_x']) * c_x * c_x * xsq
                      - (1.0 + p['k_y']) * c_y * c_y * ysq)
        return (c_x * xsq + c_y * ysq) / (1 + phi)

    def sag_and_normal(self, x, y):
        """Evaluate biconic sag and unit normal."""
        p = self.params
        c_x = p['c_x']
        c_y = p['c_y']
        kx = p['k_x']
        ky = p['k_y']
        xsq = x * x
        ysq = y * y
        one_plus_kx = 1.0 + kx
        one_plus_ky = 1.0 + ky
        phi = np.sqrt(1 - one_plus_kx * c_x * c_x * xsq
                      - one_plus_ky * c_y * c_y * ysq)
        one_plus_phi = 1 + phi
        num = c_x * xsq + c_y * ysq
        z = num / one_plus_phi
        two_phi_one_plus_phi = 2 * phi * one_plus_phi
        den = phi * one_plus_phi * one_plus_phi
        ddx = c_x * x * (two_phi_one_plus_phi + num * one_plus_kx * c_x) / den
        ddy = c_y * y * (two_phi_one_plus_phi + num * one_plus_ky * c_y) / den
        return z, gradient_to_unit_normal(ddx, ddy)


class Interaction:
    """Result of one Surface.interact, including optional local intermediates."""

    __slots__ = ('P', 'S', 'n_post', 'opl', 'code',
                 'P0', 'S_loc', 'Q_loc', 'n_hat', 'Sprime', 'S_specular',
                 'grating_grad')

    def __init__(self, P, S, n_post, opl, code,
                 P0, S_loc, Q_loc, n_hat, Sprime, S_specular,
                 grating_grad=None):
        self.P = P              # global outgoing position
        self.S = S              # global outgoing direction
        self.n_post = n_post    # index following the surface
        self.opl = opl          # OPL of the incoming segment (+ grating phase)
        self.code = code        # per-ray STATUS_* outcome at this surface
        self.P0 = P0            # local incoming position (post transform)
        self.S_loc = S_loc      # local incoming direction
        self.Q_loc = Q_loc      # local intersection point
        self.n_hat = n_hat      # local surface normal at Q_loc
        self.Sprime = Sprime    # local post-bend direction (pre global xform)
        # Local direction before the diffractive bend.
        self.S_specular = S_specular
        # (gx, gy) in-plane phase gradient at Q_loc on a grating surface, else
        # None; the diffraction bend computes it, so the AD stacks reuse it
        # instead of re-evaluating the phase function.
        self.grating_grad = grating_grad


class Surface:
    """A posed optical surface with a shape and interaction mode."""

    _analytic_intersect = False

    def __init__(self, shape=None, interaction=None, pose=None, material=None,
                 aperture=None, grating=None, *, P=None, R=None,
                 tilt=None, decenter=None, tilt_radians=False,
                 coating=None):
        """Initialize a posed optical surface.

        Parameters
        ----------
        shape : Shape
            Sag-bearing shape object.
        interaction : str or int
            Surface interaction, one of reflect, refract, eval, or an STYPE
            constant.
        pose : tuple or object, optional
            Surface pose as (P, R) or an object with P and R attributes.
        material : MaterialProtocol or None, optional
            Optical material for refractive surfaces; its .n(wavelength) gives
            the real geometric index and .nk(wavelength) the complex index.
            None for reflective / eval surfaces.
        aperture : Aperture, float, callable, or None, optional
            The surface aperture: clip, drawn extent, substrate, and rim
            features.  A float is a circular clip, a callable an opaque clip,
            None an auto aperture.
        grating : PhaseFunction or tuple, optional
            Diffractive phase function on the surface; None for a plain surface.
            Legacy (period, grating_vector, order) tuples are accepted.
        P : array_like, optional
            Surface vertex position.
        R : array_like, optional
            Surface rotation matrix.
        tilt, decenter : array_like, optional
            Pose adjustments applied after P and R are resolved.
        tilt_radians : bool, optional
            If True, tilt values are interpreted as radians.
        coating : coatings.Stack, optional
            Thin-film stack on this surface; None (default) gives the bare
            Fresnel interface (and, on reflection, the lossless ideal mirror).
            Consumed by the field / polarization paths via
            field.interface_coefficients.

        """
        if shape is None:
            raise TypeError('Surface requires a shape')
        if interaction is None:
            raise TypeError('Surface requires an interaction')
        if pose is not None:
            try:
                P, R = pose
            except (TypeError, ValueError):
                P = pose.P
                R = pose.R
        if P is None:
            raise TypeError('Surface requires a pose or P')

        typ = _map_stype(interaction)
        P = promote_3d_point(P, dtype=config.precision)
        R = coerce_3d_rotation(R)
        P, R = apply_tilt_decenter(P, R, tilt=tilt, decenter=decenter,
                                   tilt_radians=tilt_radians,
                                   dtype=config.precision)
        if typ == STYPE_REFRACT and material is None:
            raise ValueError(
                'refractive surfaces must have a material, not None')

        self.shape = shape
        self.typ = typ
        self.P = P
        self.R = R
        self.material = material
        self.params = shape.params
        self.aperture = aperture
        self.grating = grating
        self.coating = coating
        self.sag = shape.sag
        self.sag_and_normal = shape.sag_and_normal
        self._analytic_intersect = bool(getattr(shape, 'analytic_intersect', False))
        # Cached DepartureBand for the first-root acceptance band.
        self._departure_band = None

    @property
    def aperture(self):
        """The surface Aperture (clip, drawn extent, substrate, rim features)."""
        return self._aperture

    @aperture.setter
    def aperture(self, value):
        # Always an Aperture: a float is a circular clip, a callable an opaque
        # clip, None an auto aperture.
        self._aperture = as_aperture(value)

    @property
    def grating(self):
        """Diffractive phase function on this surface (None for a plain surface)."""
        return self._grating

    @grating.setter
    def grating(self, value):
        # First-class objects only (ADR-0011): a PhaseFunction or None, no
        # (period, g_vec, order) tuple coercion.
        if value is not None and not isinstance(value, PhaseFunction):
            raise TypeError(
                'grating must be a PhaseFunction (LinearGrating, CallablePhase) '
                f'or None; got {value!r}')
        self._grating = value

    def departure_band(self):
        """Conic-seed departure bounds for the first-root acceptance band.

        Returns a DepartureBand; an analytic shape or a surface with no
        characterizable conic domain yields DepartureBand.unbounded().
        """
        if self._departure_band is None:
            self._departure_band = self._compute_departure_band()
        return self._departure_band

    def _compute_departure_band(self):
        shape = self.shape
        if not hasattr(shape, 'seed_conic'):
            return DepartureBand.unbounded()
        c, k, dx, dy = shape.seed_conic()
        # the aperture bounds the characterized domain: the drawn extent if set,
        # else the clip radius (rays land only inside it)
        ap = self.aperture
        R = ap.extent.outer_radius if ap.extent is not None \
            else ap.limiting_radius()
        if R is None:
            p = shape.params or {}
            R = p.get('normalization_radius')
            if R is None and 'x_norm' in p:
                R = max(p['x_norm'], p['y_norm'])
        if R is None:
            ckk = (1.0 + k) * c * c
            if ckk > 0.0:
                # Stay just inside the seed conic's finite sag domain.
                R = 0.999 / np.sqrt(ckk)
        if R is None or not np.isfinite(R) or R <= 0:
            return DepartureBand.unbounded()
        R = float(R)
        n = DEPARTURE_BAND_SAMPLES
        xs = np.linspace(-R, R, n, dtype=config.precision)
        X, Y = np.meshgrid(xs, xs)
        outside = X * X + Y * Y > R * R
        with np.errstate(divide='ignore', invalid='ignore'):
            Xs = X + dx
            Ys = Y + dy
            dep = shape.sag(X, Y) - conic_sag(c, k, Xs * Xs + Ys * Ys)
            # Analytic departure gradient = grad(sag) - grad(seed conic), each
            # read straight from the unit normal (grad = (-n_x, -n_y) / n_z).
            # Evaluated to the rim, where the departure slope peaks; an FD
            # stencil would drop the rim ring and bias G low, over-certifying
            # the monotonicity test.
            _, n_sag = shape.sag_and_normal(X, Y)
            _, n_con = conic_sag_and_normal(c, k, Xs, Ys)
            gx = n_con[..., 0] / n_con[..., 2] - n_sag[..., 0] / n_sag[..., 2]
            gy = n_con[..., 1] / n_con[..., 2] - n_sag[..., 1] / n_sag[..., 2]
            gmag_dep = np.hypot(gx, gy)
        dep[outside] = np.nan
        gmag_dep[outside] = np.nan
        if not np.isfinite(dep).any():
            return DepartureBand.unbounded()
        D = float(np.nanmax(np.abs(dep)))
        # Departure slope bound for the monotonicity certificate.
        G = float(np.nanmax(gmag_dep))
        # Sag slope bound for the Lipschitz rescue, over the enlarged disk.
        R_march = MARCH_RADIUS_MARGIN * R
        xm = np.linspace(-R_march, R_march, n, dtype=config.precision)
        Xm, Ym = np.meshgrid(xm, xm)
        with np.errstate(divide='ignore', invalid='ignore'):
            _, nrm = shape.sag_and_normal(Xm, Ym)
            gmag = np.hypot(nrm[..., 0], nrm[..., 1]) / np.abs(nrm[..., 2])
        gmag[Xm * Xm + Ym * Ym > R_march * R_march] = np.nan
        L = float(np.nanmax(gmag))
        if G >= DEPARTURE_GRADIENT_WARN:
            # Static message: surfaces are recompiled every edit (optimization /
            # tolerancing rebuilds them), so a value-templated warning would
            # defeat Python's once-per-location dedup and flood the loop.
            warnings.warn(
                'a surface departs from its conic seed steeply enough that the '
                'intersection acceptance band can admit multiple ray crossings; '
                'the traced intersection on such a surface may be ambiguous.'
            )
        return DepartureBand(bounded=True, max_departure=1.1 * D,
                             domain_radius=R, gradient_bound=1.1 * G,
                             lipschitz=1.1 * L)

    def diffract(self, S_specular, n_hat, n_post, wvl, Q_loc, grad=None):
        """Apply the diffractive bend from the surface phase function.

        Parameters
        ----------
        S_specular : ndarray
            Specular outgoing direction cosines.
        n_hat : ndarray
            Unit surface normals at the intersection.
        n_post : float or ndarray
            Refractive index after the surface.
        wvl : float
            Wavelength in the same units as the phase coordinates.
        Q_loc : ndarray
            Local intersection points, last axis xyz; where the phase gradient
            is evaluated.
        grad : tuple of ndarray, optional
            precomputed (gx, gy) in-plane phase gradient at Q_loc; evaluated
            from the phase function when None.

        Returns
        -------
        S_out : ndarray
            Diffracted direction cosines, or S_specular where diffraction is
            invalid or no phase function is present.
        valid : ndarray
            Boolean mask indicating valid diffracted rays.

        """
        if self.grating is None:
            return S_specular, np.ones(S_specular.shape[:-1], dtype=bool)
        if grad is None:
            _, gx, gy = self.grating.phase_and_gradient(Q_loc[..., 0],
                                                        Q_loc[..., 1])
        else:
            gx, gy = grad
        G = np.stack([gx, gy, np.zeros_like(gx)], axis=-1)
        G_dot_n = (G * n_hat).sum(-1, keepdims=True)
        G_tan = G - G_dot_n * n_hat
        s_dot_n = (S_specular * n_hat).sum(-1, keepdims=True)
        s_specular_tan = S_specular - s_dot_n * n_hat
        s_diff_tan = s_specular_tan + (wvl / n_post) * G_tan
        tan_sq = (s_diff_tan * s_diff_tan).sum(-1)
        valid = tan_sq <= 1.0
        normal_mag = np.sqrt(np.where(valid, 1.0 - tan_sq, 0.0))
        sign = np.sign(s_dot_n[..., 0])
        S_diff = s_diff_tan + (sign * normal_mag)[..., np.newaxis] * n_hat
        S_diff[~valid] = S_specular[~valid]
        return S_diff, valid

    def grating_phase(self, Q_loc, wvl):
        """OPL added by the diffractive phase at local intersection points.

        Parameters
        ----------
        Q_loc : ndarray
            intersection points in the surface local frame, last axis xyz.
        wvl : float
            wavelength, same length units as the phase coordinates.

        Returns
        -------
        ndarray
            per-ray OPL contribution, shape Q_loc.shape[:-1].

        """
        return wvl * self.grating.phase(Q_loc[..., 0], Q_loc[..., 1])

    def interact(self, P_in, S_in, n_pre, wvl, tol_sag=None,
                 first_segment=False):
        """March one bundle through this surface: intersect, clip, bend, diffract.

        Parameters
        ----------
        P_in, S_in : ndarray
            global incoming position and direction, shape (N, 3).
        n_pre : float or ndarray
            index of the medium preceding the surface.
        wvl : float
            wavelength, microns.
        tol_sag : float, optional
            Newton convergence tolerance, resolved at the leaf.
        first_segment : bool, optional
            True when this is the first surface the bundle meets.

        Returns
        -------
        Interaction
            outgoing ray, following index, segment OPL, per-ray status code, and
            the local-frame intermediates.

        """
        P0, S_loc = transform_to_local_coords(P_in, self.P, S_in, self.R)
        # Later reflect/refract surfaces reject roots behind the incoming ray;
        # measurement planes (eval/object/image) are exempt.
        forward_only = not _is_measurement_surf(self.typ) and not first_segment
        Q_loc, n_hat, converged = self.intersect(P0, S_loc, tol_sag=tol_sag,
                                                 forward_only=forward_only)

        miss = STATUS_MISS if self._analytic_intersect else STATUS_NEWTON
        code = np.where(converged, STATUS_OK, miss)

        if self.aperture.clip is not None:
            inside = self.aperture.clips(Q_loc[..., 0], Q_loc[..., 1])
            code[converged & ~inside] = STATUS_CLIP

        if self.typ == STYPE_REFLECT:
            Sprime = reflect(S_loc, n_hat)
            n_post = n_pre
        elif self.typ == STYPE_REFRACT:
            n_post = self.material.n(wvl)
            Sprime = refract(n_pre, n_post, S_loc, n_hat)
            tir = np.isnan(Sprime).any(axis=-1)
            code[(code == STATUS_OK) & tir] = STATUS_TIR
        else:
            Sprime = S_loc
            n_post = n_pre

        # Preserve the pre-diffraction direction for AD.
        S_specular = Sprime
        opl_grating = None
        grating_grad = None
        if self.grating is not None and self.typ in (STYPE_REFLECT, STYPE_REFRACT):
            # One phase evaluation feeds the bend, the OPL term, and the AD
            # capture (phase_and_gradient returns both at once).
            gphase, gx, gy = self.grating.phase_and_gradient(Q_loc[..., 0],
                                                             Q_loc[..., 1])
            grating_grad = (gx, gy)
            Sprime, valid_diff = self.diffract(Sprime, n_hat, n_post, wvl,
                                               Q_loc, grad=grating_grad)
            code[(code == STATUS_OK) & ~valid_diff] = STATUS_EVANESCENT
            opl_grating = wvl * gphase

        Rt = None if self.R is None else self.R.T
        P_out, S_out = transform_to_global_coords(Q_loc, self.P, Sprime, Rt)

        seg = P_out - P_in
        # Signed segment length supports virtual first segments.
        opl = n_pre * np.sign(np.sum(seg * S_in, axis=-1)) \
            * np.sqrt(np.sum(seg * seg, axis=-1))
        if opl_grating is not None:
            opl = opl + opl_grating

        return Interaction(P_out, S_out, n_post, opl, code,
                           P0, S_loc, Q_loc, n_hat, Sprime, S_specular,
                           grating_grad)

    def intersect(self, P, S, tol_sag=None, maxiter=None, forward_only=False):
        """Intersect rays with the surface shape.

        Parameters
        ----------
        P : ndarray
            Ray origins in the surface local frame.
        S : ndarray
            Unit direction cosines in the surface local frame.
        tol_sag : float, optional
            Absolute convergence tolerance on the surface residual.
        maxiter : int, optional
            Maximum Newton iterations for non-analytic shapes.
        forward_only : bool, optional
            Reject roots behind the ray origin for conic-seeded shapes.  Not
            forwarded to analytic shapes: their closed-form root selection
            already picks the vertex-side sheet, and a local behind-origin
            criterion would wrongly reject legitimate mirror folds (the folded
            segment is "backward" in the signed-segment convention).

        Returns
        -------
        Q, n, valid : ndarray, ndarray, ndarray
            Intersection points.
            Unit surface normals, and a Boolean validity mask.

        """
        # tol_sag stays None here so it resolves at newton_raphson_solve_s,
        # where the working dtype (float32/float64) is known.
        if hasattr(self.shape, 'seed_conic'):
            band = self.departure_band()
            return self.shape.intersect(P, S, self.sag_and_normal,
                                        tol_sag=tol_sag,
                                        maxiter=maxiter,
                                        departure=band.max_departure,
                                        domain_radius=band.domain_radius,
                                        departure_gradient=band.gradient_bound,
                                        sag_lipschitz=band.lipschitz,
                                        forward_only=forward_only)
        if hasattr(self.shape, 'intersect'):
            return self.shape.intersect(P, S, self.sag_and_normal,
                                        tol_sag=tol_sag,
                                        maxiter=maxiter)
        if maxiter is None:
            maxiter = SURFACE_INTERSECTION_DEFAULT_MAXITER
        return newton_intersect(P, S, self.sag_and_normal, tol_sag=tol_sag,
                                maxiter=maxiter)


__all__ = [
    'STYPE_REFLECT',
    'STYPE_REFRACT',
    'STYPE_EVAL',
    'STYPE_OBJ',
    'STYPE_IMG',
    'Shape',
    'CallableShape',
    'Plane',
    'Sphere',
    'Conic',
    'OffAxisConic',
    'EvenAsphere',
    'Q2D',
    'Zernike',
    'XY',
    'Chebyshev',
    'Jacobi',
    'Toroid',
    'Biconic',
    'Surface',
    'DepartureBand',
    'circular_aperture',
    'annular_aperture',
    'product_rule',
    'phi_conic',
    'der_direction_cosine_conic',
    'sphere_sag',
    'sphere_sag_der',
    'conic_sag',
    'conic_sag_der',
    'conic_sag_der_xy',
    'even_asphere_sag',
    'even_asphere_sag_der_xy',
    'Q2d_and_der',
    'Q2d_sag',
    'ray_plane_intersect',
    'ray_sphere_intersect',
    'ray_conic_intersect',
]
