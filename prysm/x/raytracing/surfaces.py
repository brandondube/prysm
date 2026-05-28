"""Surface containers, shape objects, and calculus for raytracing."""

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
    STYPE_REFLECT,
    STYPE_REFRACT,
    resolve_tol_sag,
)
from .intersections import (
    SURFACE_INTERSECTION_DEFAULT_MAXITER,
    ConicSeedMixin,
    newton_intersect,
    ray_conic_intersect,
    ray_plane_intersect,
    ray_sphere_intersect,
)
from ._line_math import normalize_vector
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
    gradient_to_unit_normal,
    plane_sag_and_normal,
    phi_conic,
    product_rule,
    sphere_sag,
    sphere_sag_der,
)


def circular_aperture(radius, x0=0.0, y0=0.0):
    """Create a circular surface aperture predicate.

    Parameters
    ----------
    radius : float
        Radius of the clear aperture.
    x0, y0 : float, optional
        Center of the aperture in local surface coordinates.

    Returns
    -------
    callable
        Predicate returning True for points inside or on the aperture.

    """
    radius = float(radius)
    x0 = float(x0)
    y0 = float(y0)
    rsq = radius * radius

    def aperture(x, y):
        """Evaluate whether local coordinates are inside the aperture."""
        dx = x - x0
        dy = y - y0
        return dx * dx + dy * dy <= rsq

    return aperture


def _ensure_P_vec(P):
    """Promote a point-like object to a 3-vector using configured precision."""
    return promote_3d_point(P, dtype=config.precision)


def _none_or_rotmat(R):
    """Return None or a coerced 3D rotation matrix."""
    return coerce_3d_rotation(R)


def _apply_tilt_decenter(P, R, tilt, decenter, tilt_radians):
    """Apply optional tilt and decenter to a surface pose."""
    return apply_tilt_decenter(P, R, tilt=tilt, decenter=decenter,
                               tilt_radians=tilt_radians,
                               dtype=config.precision)


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
    raise ValueError(
        f'unknown surface type {typ!r}; expected one of '
        "'refl'/'reflect', 'refr'/'refract', 'eval', or an STYPE_* int."
    )


def _validate_n_and_typ(n, typ):
    """Validate that refractive surfaces have a refractive-index model."""
    if typ == STYPE_REFRACT and n is None:
        raise ValueError('refractive surfaces must have a refractive index function, not None')


class Shape:
    """Base class for sag-bearing shape objects."""

    analytic_intersect = False
    finite_difference_step = None

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
        if self.finite_difference_step is not None:
            return np.asarray(self.finite_difference_step, dtype=config.precision)
        eps = np.sqrt(np.finfo(config.precision).eps)
        mag = 1.0
        for a in arrs:
            mag = np.maximum(mag, np.abs(a))
        return eps * mag

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

    def __init__(self):
        """Initialize a plane sag shape."""
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

    def intersect(self, P, S, sag_and_normal=None, tol_sag=None, eps=None,
                  maxiter=None,
                  return_valid=False):
        """Intersect rays with the plane.

        Parameters
        ----------
        P : ndarray
            Ray origins in the surface local frame.
        S : ndarray
            Unit direction cosines.
        sag_and_normal : callable, optional
            Ignored; accepted for the common shape-intersection interface.
        tol_sag, eps : float, optional
            Ignored convergence tolerances.
        maxiter : int, optional
            Ignored iteration limit.
        return_valid : bool, optional
            If True, return a validity mask.

        Returns
        -------
        Q : ndarray
            Intersection points.
        n : ndarray
            Unit surface normals.
        valid : ndarray, optional
            Boolean validity mask, only returned when return_valid is True.

        """
        return ray_plane_intersect(P, S, return_valid=return_valid)


class Sphere(Shape):
    """Spherical sag shape.

    Parameters
    ----------
    c : float
        Vertex curvature, reciprocal radius of curvature.

    """

    analytic_intersect = True

    def __init__(self, c):
        """Initialize a spherical sag shape."""
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

    def intersect(self, P, S, sag_and_normal=None, tol_sag=None, eps=None,
                  maxiter=None,
                  return_valid=False):
        """Intersect rays with the sphere.

        Parameters
        ----------
        P : ndarray
            Ray origins in the surface local frame.
        S : ndarray
            Unit direction cosines.
        sag_and_normal : callable, optional
            Ignored; accepted for the common shape-intersection interface.
        tol_sag, eps : float, optional
            Ignored convergence tolerances.
        maxiter : int, optional
            Ignored iteration limit.
        return_valid : bool, optional
            If True, return a validity mask.

        Returns
        -------
        Q : ndarray
            Intersection points.
        n : ndarray
            Unit surface normals.
        valid : ndarray, optional
            Boolean validity mask, only returned when return_valid is True.

        """
        return ray_sphere_intersect(P, S, self.params['c'],
                                    return_valid=return_valid)


class Conic(Shape):
    """Conic sag shape.

    Parameters
    ----------
    c : float
        Vertex curvature, reciprocal radius of curvature.
    k : float
        Conic constant.

    """

    analytic_intersect = True

    def __init__(self, c, k):
        """Initialize a conic sag shape."""
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

    def intersect(self, P, S, sag_and_normal=None, tol_sag=None, eps=None,
                  maxiter=None,
                  return_valid=False):
        """Intersect rays with the conic.

        Parameters
        ----------
        P : ndarray
            Ray origins in the surface local frame.
        S : ndarray
            Unit direction cosines.
        sag_and_normal : callable, optional
            Ignored; accepted for the common shape-intersection interface.
        tol_sag, eps : float, optional
            Ignored convergence tolerances.
        maxiter : int, optional
            Ignored iteration limit.
        return_valid : bool, optional
            If True, return a validity mask.

        Returns
        -------
        Q : ndarray
            Intersection points.
        n : ndarray
            Unit surface normals.
        valid : ndarray, optional
            Boolean validity mask, only returned when return_valid is True.

        """
        p = self.params
        return ray_conic_intersect(P, S, p['c'], p['k'],
                                   return_valid=return_valid)


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

    def intersect(self, P, S, sag_and_normal=None, tol_sag=None, eps=None,
                  maxiter=None,
                  return_valid=False):
        """Intersect rays with the off-axis conic.

        Parameters
        ----------
        P : ndarray
            Ray origins in the surface local frame.
        S : ndarray
            Unit direction cosines.
        sag_and_normal : callable, optional
            Ignored; accepted for the common shape-intersection interface.
        tol_sag, eps : float, optional
            Ignored convergence tolerances.
        maxiter : int, optional
            Ignored iteration limit.
        return_valid : bool, optional
            If True, return a validity mask.

        Returns
        -------
        Q : ndarray
            Intersection points.
        n : ndarray
            Unit surface normals.
        valid : ndarray, optional
            Boolean validity mask, only returned when return_valid is True.

        """
        p = self.params
        return ray_conic_intersect(P, S, p['c'], p['k'],
                                   dx=p['dx'], dy=p['dy'],
                                   return_valid=return_valid)


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

    def __init__(self, c, k, coefs):
        """Initialize an even asphere sag shape."""
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
            ddx = np.where(on_axis, np.zeros_like(ddx), ddx)
            ddy = np.where(on_axis, np.zeros_like(ddy), ddy)
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

    def __init__(self, c_x, c_y, k_y, coefs_y):
        """Initialize a toroidal sag shape."""
        coefs_y = tuple(coefs_y) if coefs_y is not None else ()
        super().__init__(c_x=float(c_x), c_y=float(c_y), k_y=float(k_y),
                         coefs_y=coefs_y)

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

    def __init__(self, c_x, c_y, k_x, k_y):
        """Initialize a biconic sag shape."""
        super().__init__(c_x=float(c_x), c_y=float(c_y),
                         k_x=float(k_x), k_y=float(k_y))

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


class Surface:
    """A posed optical surface with a shape and interaction mode."""

    _analytic_intersect = False

    def __init__(self, shape=None, interaction=None, pose=None, material=None,
                 aperture=None, grating=None, *, typ=None, P=None, n=None,
                 R=None, bounding=None, tilt=None, decenter=None,
                 tilt_radians=False, edge=None):
        """Initialize a posed optical surface.

        Parameters
        ----------
        shape : Shape
            Sag-bearing shape object.
        interaction : str or int, optional
            Surface interaction, one of reflect, refract, eval, or an STYPE
            constant.  If omitted, typ is used.
        pose : tuple or object, optional
            Surface pose as (P, R) or an object with P and R attributes.
        material : callable or float, optional
            Refractive-index model or value.  If omitted, n is used.
        aperture : callable, optional
            Aperture predicate evaluated in local surface coordinates.
        grating : tuple, optional
            Diffraction grating data as (period, grating_vector, order).
        typ : str or int, optional
            Legacy alias for interaction.
        P : array_like, optional
            Surface vertex position.
        n : callable or float, optional
            Legacy alias for material.
        R : array_like, optional
            Surface rotation matrix.
        bounding : object, optional
            Bounding data carried by the surface.
        tilt, decenter : array_like, optional
            Pose adjustments applied after P and R are resolved.
        tilt_radians : bool, optional
            If True, tilt values are interpreted as radians.
        edge : mapping, optional
            Mechanical edge geometry (outer diameter, chamfers, seats, ...)
            carried for layout drawing.  Consumed by plotting.plot_optics; see
            its lens_edges parameter for the schema.

        """
        if shape is None:
            raise TypeError('Surface requires a shape')
        if interaction is None:
            interaction = typ
        if interaction is None:
            raise TypeError('Surface requires an interaction or typ')
        if pose is not None:
            try:
                P, R = pose
            except (TypeError, ValueError):
                P = pose.P
                R = pose.R
        if P is None:
            raise TypeError('Surface requires a pose or P')

        typ = _map_stype(interaction)
        P = _ensure_P_vec(P)
        R = _none_or_rotmat(R)
        P, R = _apply_tilt_decenter(P, R, tilt, decenter, tilt_radians)
        if material is None:
            material = n
        _validate_n_and_typ(material, typ)

        self.shape = shape
        self.typ = typ
        self.P = P
        self.R = R
        self.n = material
        self.params = shape.params
        self.bounding = bounding
        self.aperture = aperture
        self.grating = grating
        self.edge = edge
        self.sag = shape.sag
        self.sag_and_normal = shape.sag_and_normal
        self._analytic_intersect = bool(getattr(shape, 'analytic_intersect', False))

    def diffract(self, S_specular, r, n_post, wvl):
        """Apply diffraction from the surface grating.

        Parameters
        ----------
        S_specular : ndarray
            Specular outgoing direction cosines.
        r : ndarray
            Local surface point or normal-defining vector used to determine
            the grating plane normal.
        n_post : float or ndarray
            Refractive index after the surface.
        wvl : float
            Wavelength in the same units as the grating period.

        Returns
        -------
        S_out : ndarray
            Diffracted direction cosines, or S_specular where diffraction is
            invalid or no grating is present.
        valid : ndarray
            Boolean mask indicating valid diffracted rays.

        """
        if self.grating is None:
            return S_specular, np.ones(S_specular.shape[:-1], dtype=bool)
        period, g_vec, order = self.grating
        g_vec = np.asarray(g_vec, dtype=S_specular.dtype)
        n_hat = normalize_vector(r, axis=-1)
        q = g_vec / period
        q_dot_n = (q * n_hat).sum(-1, keepdims=True)
        q_tan = q - q_dot_n * n_hat
        s_dot_n = (S_specular * n_hat).sum(-1, keepdims=True)
        s_specular_tan = S_specular - s_dot_n * n_hat
        s_diff_tan = s_specular_tan + (order * wvl / n_post) * q_tan
        tan_sq = (s_diff_tan * s_diff_tan).sum(-1)
        valid = tan_sq <= 1.0
        normal_mag = np.sqrt(np.where(valid, 1.0 - tan_sq, np.zeros_like(tan_sq)))
        sign = np.sign(s_dot_n[..., 0])
        S_diff = s_diff_tan + (sign * normal_mag)[..., np.newaxis] * n_hat
        S_diff = np.where(valid[..., np.newaxis], S_diff, S_specular)
        return S_diff, valid

    def intersect(self, P, S, tol_sag=None, eps=None, maxiter=None,
                  return_valid=False):
        """Intersect rays with the surface shape.

        Parameters
        ----------
        P : ndarray
            Ray origins in the surface local frame.
        S : ndarray
            Unit direction cosines in the surface local frame.
        tol_sag : float, optional
            Absolute convergence tolerance on the surface residual.
        eps : float, optional
            Deprecated alias for tol_sag.
        maxiter : int, optional
            Maximum Newton iterations for non-analytic shapes.
        return_valid : bool, optional
            If True, return a validity mask.

        Returns
        -------
        Q : ndarray
            Intersection points.
        n : ndarray
            Unit surface normals.
        valid : ndarray, optional
            Boolean validity mask, only returned when return_valid is True.

        """
        tol_sag = resolve_tol_sag(tol_sag, eps)
        if hasattr(self.shape, 'intersect'):
            return self.shape.intersect(P, S, self.sag_and_normal,
                                        tol_sag=tol_sag,
                                        maxiter=maxiter,
                                        return_valid=return_valid)
        if maxiter is None:
            maxiter = SURFACE_INTERSECTION_DEFAULT_MAXITER
        return newton_intersect(P, S, self.sag_and_normal, tol_sag=tol_sag,
                                maxiter=maxiter,
                                return_valid=return_valid)


__all__ = [
    'STYPE_REFLECT',
    'STYPE_REFRACT',
    'STYPE_EVAL',
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
    'circular_aperture',
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
    '_ensure_P_vec',
    '_none_or_rotmat',
    '_apply_tilt_decenter',
]
