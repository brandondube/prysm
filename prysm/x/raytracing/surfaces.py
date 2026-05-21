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
)
from .intersections import (
    SURFACE_INTERSECTION_DEFAULT_MAXITER,
    ConicSeedMixin,
    newton_intersect,
    ray_conic_intersect,
    ray_plane_intersect,
    ray_sphere_intersect,
)
from .sags import (
    Q2d_and_der,
    Q2d_sag,
    _add_conic_base_F,
    _add_conic_base_FFp,
    _conic_base_xy,
    _conic_base_xy_F,
    conic_sag,
    conic_sag_der,
    conic_sag_der_xy,
    der_direction_cosine_spheroid,
    even_asphere_sag,
    even_asphere_sag_der_xy,
    off_axis_conic_der,
    off_axis_conic_sag,
    off_axis_conic_sag_der_xy,
    off_axis_conic_sigma,
    off_axis_conic_sigma_der,
    phi_spheroid,
    product_rule,
    sphere_sag,
    sphere_sag_der,
)


def circular_aperture(radius, x0=0.0, y0=0.0):
    """Create a circular surface aperture predicate."""
    radius = float(radius)
    x0 = float(x0)
    y0 = float(y0)
    rsq = radius * radius

    def aperture(x, y):
        dx = x - x0
        dy = y - y0
        return dx * dx + dy * dy <= rsq

    return aperture


def _ensure_P_vec(P):
    return promote_3d_point(P, dtype=config.precision)


def _none_or_rotmat(R):
    return coerce_3d_rotation(R)


def _apply_tilt_decenter(P, R, tilt, decenter, tilt_radians):
    return apply_tilt_decenter(P, R, tilt=tilt, decenter=decenter,
                               tilt_radians=tilt_radians,
                               dtype=config.precision)


def _map_stype(typ):
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
    if typ == STYPE_REFRACT and n is None:
        raise ValueError('refractive surfaces must have a refractive index function, not None')


def _common_surface_kwargs(R=None, bounding=None, aperture=None, tilt=None,
                           decenter=None, tilt_radians=False, grating=None):
    return dict(R=R, bounding=bounding, aperture=aperture, tilt=tilt,
                decenter=decenter, tilt_radians=tilt_radians,
                grating=grating)


class Shape:
    """Base class for sag-bearing shape objects."""

    analytic_intersect = False

    def __init__(self, **params):
        self.params = params or None

    def F(self, x, y):
        raise NotImplementedError

    def FFp(self, x, y):
        raise NotImplementedError


class CallableShape(Shape):
    """Shape wrapper for user-supplied F/FFp callables."""

    def __init__(self, F, FFp, params=None):
        self._F = F
        self._FFp = FFp
        self.params = params

    def F(self, x, y):
        return self._F(x, y)

    def FFp(self, x, y):
        return self._FFp(x, y)


class PlaneSag(Shape):
    analytic_intersect = True

    def __init__(self):
        super().__init__()

    def F(self, x, y):
        zero = np.array([0.], dtype=x.dtype)
        return np.broadcast_to(zero, x.shape)

    def FFp(self, x, y):
        z = self.F(x, y)
        return z, z, z

    def intersect(self, P, S, sag_normal, eps=None, maxiter=None,
                  return_valid=False):
        return ray_plane_intersect(P, S, return_valid=return_valid)


class SphereSag(Shape):
    analytic_intersect = True

    def __init__(self, c):
        super().__init__(c=c)

    def F(self, x, y):
        return sphere_sag(self.params['c'], x * x + y * y)

    def FFp(self, x, y):
        c = self.params['c']
        rsq = x * x + y * y
        phi = phi_spheroid(c, 0.0, rsq)
        z = sphere_sag(c, rsq, phi=phi)
        return z, (c * x) / phi, (c * y) / phi

    def intersect(self, P, S, sag_normal, eps=None, maxiter=None,
                  return_valid=False):
        return ray_sphere_intersect(P, S, self.params['c'],
                                    return_valid=return_valid)


class ConicSag(Shape):
    analytic_intersect = True

    def __init__(self, c, k):
        super().__init__(c=c, k=k)

    def F(self, x, y):
        return _conic_base_xy_F(self.params['c'], self.params['k'], x, y)

    def FFp(self, x, y):
        return _conic_base_xy(self.params['c'], self.params['k'], x, y)

    def intersect(self, P, S, sag_normal, eps=None, maxiter=None,
                  return_valid=False):
        p = self.params
        return ray_conic_intersect(P, S, p['c'], p['k'],
                                   return_valid=return_valid)


class OffAxisConicSag(Shape):
    analytic_intersect = True

    def __init__(self, c, k, dx=0.0, dy=0.0):
        super().__init__(c=c, k=k, dx=dx, dy=dy)

    def F(self, x, y):
        p = self.params
        X = x + p['dx']
        Y = y + p['dy']
        return conic_sag(p['c'], p['k'], X * X + Y * Y)

    def FFp(self, x, y):
        p = self.params
        c, k = p['c'], p['k']
        X = x + p['dx']
        Y = y + p['dy']
        aggregate = X * X + Y * Y
        phi = phi_spheroid(c, k, aggregate)
        z = (c * aggregate) / (1 + phi)
        return z, (c * X) / phi, (c * Y) / phi

    def intersect(self, P, S, sag_normal, eps=None, maxiter=None,
                  return_valid=False):
        p = self.params
        return ray_conic_intersect(P, S, p['c'], p['k'],
                                   dx=p['dx'], dy=p['dy'],
                                   return_valid=return_valid)


class EvenAsphereSag(ConicSeedMixin, Shape):
    def __init__(self, c, k, coefs):
        coefs = tuple(coefs) if coefs is not None else ()
        super().__init__(c=c, k=k, coefs=coefs)

    def F(self, x, y):
        p = self.params
        return even_asphere_sag(p['c'], p['k'], p['coefs'], x * x + y * y)

    def FFp(self, x, y):
        p = self.params
        rsq = x * x + y * y
        phi = phi_spheroid(p['c'], p['k'], rsq)
        z = even_asphere_sag(p['c'], p['k'], p['coefs'], rsq)
        dx, dy = even_asphere_sag_der_xy(p['c'], p['k'], p['coefs'],
                                         x, y, phi=phi)
        return z, dx, dy


class Q2DSag(ConicSeedMixin, Shape):
    def __init__(self, c, k, normalization_radius, cm0, ams, bms,
                 dx=0.0, dy=0.0):
        cm0 = tuple(cm0) if cm0 is not None else (0.0,)
        ams = tuple(tuple(am) for am in ams)
        bms = tuple(tuple(bm) for bm in bms)
        super().__init__(c=c, k=k,
                         normalization_radius=float(normalization_radius),
                         cm0=cm0, ams=ams, bms=bms, dx=dx, dy=dy)

    def F(self, x, y):
        p = self.params
        return Q2d_sag(p['cm0'], p['ams'], p['bms'],
                       x, y, p['normalization_radius'],
                       p['c'], p['k'], dx=p['dx'], dy=p['dy'])

    def FFp(self, x, y):
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
        return z, ddx, ddy


class ZernikeSag(ConicSeedMixin, Shape):
    def __init__(self, c, k, normalization_radius, nms, coefs, norm=True):
        nms = tuple((int(nn), int(mm)) for nn, mm in nms)
        coefs = tuple(float(co) for co in coefs)
        if len(nms) != len(coefs):
            raise ValueError(
                f'nms and coefs must be parallel; got {len(nms)} and {len(coefs)}'
            )
        super().__init__(c=c, k=k,
                         normalization_radius=float(normalization_radius),
                         nms=nms, coefs=coefs, norm=bool(norm))

    def F(self, x, y):
        p = self.params
        norm_r = p['normalization_radius']
        z_p = zernike_sum(p['coefs'], p['nms'],
                          x / norm_r, y / norm_r, norm=p['norm'])
        return _add_conic_base_F(p['c'], p['k'], x, y, z_p)

    def FFp(self, x, y):
        p = self.params
        norm_r = p['normalization_radius']
        z_p, ddx_p, ddy_p = zernike_sum_der_xy(
            p['coefs'], p['nms'], x / norm_r, y / norm_r, norm=p['norm'])
        return _add_conic_base_FFp(p['c'], p['k'], x, y,
                                   z_p, ddx_p / norm_r, ddy_p / norm_r)


class XYSag(ConicSeedMixin, Shape):
    def __init__(self, c, k, normalization_radius, mns, coefs):
        mns = tuple((int(mm), int(nn)) for mm, nn in mns)
        coefs = tuple(float(co) for co in coefs)
        if len(mns) != len(coefs):
            raise ValueError(
                f'mns and coefs must be parallel; got {len(mns)} and {len(coefs)}'
            )
        super().__init__(c=c, k=k,
                         normalization_radius=float(normalization_radius),
                         mns=mns, coefs=coefs)

    def F(self, x, y):
        p = self.params
        norm_r = p['normalization_radius']
        z_p = xy_sum(p['coefs'], p['mns'], x / norm_r, y / norm_r,
                     cartesian_grid=False)
        return _add_conic_base_F(p['c'], p['k'], x, y, z_p)

    def FFp(self, x, y):
        p = self.params
        norm_r = p['normalization_radius']
        z_p, ddx_p, ddy_p = xy_sum_der_xy(
            p['coefs'], p['mns'], x / norm_r, y / norm_r,
            cartesian_grid=False)
        return _add_conic_base_FFp(p['c'], p['k'], x, y,
                                   z_p, ddx_p / norm_r, ddy_p / norm_r)


class ChebyshevSag(ConicSeedMixin, Shape):
    def __init__(self, c, k, x_norm, y_norm, mns, coefs):
        mns = tuple((int(mm), int(nn)) for mm, nn in mns)
        coefs = tuple(float(co) for co in coefs)
        if len(mns) != len(coefs):
            raise ValueError(
                f'mns and coefs must be parallel; got {len(mns)} and {len(coefs)}'
            )
        super().__init__(c=c, k=k, x_norm=float(x_norm), y_norm=float(y_norm),
                         mns=mns, coefs=coefs)

    def F(self, x, y):
        p = self.params
        z_p = cheby1_2d_sum(p['coefs'], p['mns'],
                            x / p['x_norm'], y / p['y_norm'])
        return _add_conic_base_F(p['c'], p['k'], x, y, z_p)

    def FFp(self, x, y):
        p = self.params
        xn = p['x_norm']
        yn = p['y_norm']
        z_p, ddx_p, ddy_p = cheby1_2d_sum_der_xy(
            p['coefs'], p['mns'], x / xn, y / yn, xn, yn)
        return _add_conic_base_FFp(p['c'], p['k'], x, y,
                                   z_p, ddx_p, ddy_p)


class JacobiSag(ConicSeedMixin, Shape):
    def __init__(self, c, k, normalization_radius, alpha, beta, ns, coefs):
        ns = tuple(int(nn) for nn in ns)
        coefs = tuple(float(co) for co in coefs)
        if len(ns) != len(coefs):
            raise ValueError(
                f'ns and coefs must be parallel; got {len(ns)} and {len(coefs)}'
            )
        super().__init__(c=c, k=k,
                         normalization_radius=float(normalization_radius),
                         alpha=float(alpha), beta=float(beta),
                         ns=ns, coefs=coefs)

    def F(self, x, y):
        p = self.params
        z_p = jacobi_radial_sum(p['coefs'], p['ns'], p['alpha'], p['beta'],
                                x, y, p['normalization_radius'])
        return _add_conic_base_F(p['c'], p['k'], x, y, z_p)

    def FFp(self, x, y):
        p = self.params
        z_p, ddx_p, ddy_p = jacobi_radial_sum_der_xy(
            p['coefs'], p['ns'], p['alpha'], p['beta'],
            x, y, p['normalization_radius'])
        return _add_conic_base_FFp(p['c'], p['k'], x, y,
                                   z_p, ddx_p, ddy_p)


class ToroidSag(ConicSeedMixin, Shape):
    def __init__(self, c_x, c_y, k_y, coefs_y):
        coefs_y = tuple(coefs_y) if coefs_y is not None else ()
        super().__init__(c_x=float(c_x), c_y=float(c_y), k_y=float(k_y),
                         coefs_y=coefs_y)

    def seed_conic(self):
        p = self.params
        return 0.5 * (p['c_x'] + p['c_y']), 0.0, 0.0, 0.0

    def F(self, x, y):
        p = self.params
        z_x = sphere_sag(p['c_x'], x * x)
        z_y = even_asphere_sag(p['c_y'], p['k_y'], p['coefs_y'], y * y)
        return z_x + z_y

    def FFp(self, x, y):
        p = self.params
        xsq = x * x
        ysq = y * y
        phi_x = phi_spheroid(p['c_x'], 0.0, xsq)
        z_x = sphere_sag(p['c_x'], xsq, phi=phi_x)
        ddx = (p['c_x'] * x) / phi_x
        zero = np.zeros_like(y)
        z_y = even_asphere_sag(p['c_y'], p['k_y'], p['coefs_y'], ysq)
        _, ddy = even_asphere_sag_der_xy(p['c_y'], p['k_y'],
                                         p['coefs_y'], zero, y)
        return z_x + z_y, ddx, ddy


class BiconicSag(ConicSeedMixin, Shape):
    def __init__(self, c_x, c_y, k_x, k_y):
        super().__init__(c_x=float(c_x), c_y=float(c_y),
                         k_x=float(k_x), k_y=float(k_y))

    def seed_conic(self):
        p = self.params
        c_seed = 0.5 * (p['c_x'] + p['c_y'])
        k_seed = 0.5 * (p['k_x'] + p['k_y'])
        return c_seed, k_seed, 0.0, 0.0

    def F(self, x, y):
        p = self.params
        c_x = p['c_x']
        c_y = p['c_y']
        xsq = x * x
        ysq = y * y
        phi = np.sqrt(1 - (1.0 + p['k_x']) * c_x * c_x * xsq
                      - (1.0 + p['k_y']) * c_y * c_y * ysq)
        return (c_x * xsq + c_y * ysq) / (1 + phi)

    def FFp(self, x, y):
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
        return z, ddx, ddy


class Surface:
    """A posed optical surface with a shape and interaction mode."""

    _analytic_intersect = False

    def __init__(self, shape=None, interaction=None, pose=None, material=None,
                 aperture=None, grating=None, *, typ=None, P=None, n=None,
                 R=None, bounding=None, tilt=None, decenter=None,
                 tilt_radians=False):
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
        self.F = shape.F
        self.FFp = shape.FFp
        self._analytic_intersect = bool(getattr(shape, 'analytic_intersect', False))

    def sag_normal(self, x, y):
        z, Fx, Fy = self.FFp(x, y)
        Fz = np.array([1.], dtype=config.precision)
        Fz = np.broadcast_to(Fz, Fx.shape)
        der = np.stack([-Fx, -Fy, Fz], axis=1)
        return z, der

    def diffract(self, S_specular, r, n_post, wvl):
        if self.grating is None:
            return S_specular, np.ones(S_specular.shape[:-1], dtype=bool)
        period, g_vec, order = self.grating
        g_vec = np.asarray(g_vec, dtype=S_specular.dtype)
        n_norm = np.sqrt((r * r).sum(-1, keepdims=True))
        n_hat = r / n_norm
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

    def intersect(self, P, S, eps=None, maxiter=None, return_valid=False):
        if hasattr(self.shape, 'intersect'):
            return self.shape.intersect(P, S, self.sag_normal, eps=eps,
                                        maxiter=maxiter,
                                        return_valid=return_valid)
        if maxiter is None:
            maxiter = SURFACE_INTERSECTION_DEFAULT_MAXITER
        return newton_intersect(P, S, self.sag_normal, eps=eps,
                                maxiter=maxiter,
                                return_valid=return_valid)


__all__ = [
    'STYPE_REFLECT',
    'STYPE_REFRACT',
    'STYPE_EVAL',
    'Shape',
    'CallableShape',
    'PlaneSag',
    'SphereSag',
    'ConicSag',
    'OffAxisConicSag',
    'EvenAsphereSag',
    'Q2DSag',
    'ZernikeSag',
    'XYSag',
    'ChebyshevSag',
    'JacobiSag',
    'ToroidSag',
    'BiconicSag',
    'Surface',
    'circular_aperture',
    'product_rule',
    'phi_spheroid',
    'der_direction_cosine_spheroid',
    'sphere_sag',
    'sphere_sag_der',
    'conic_sag',
    'conic_sag_der',
    'conic_sag_der_xy',
    'even_asphere_sag',
    'even_asphere_sag_der_xy',
    'off_axis_conic_sag',
    'off_axis_conic_der',
    'off_axis_conic_sigma',
    'off_axis_conic_sigma_der',
    'off_axis_conic_sag_der_xy',
    'Q2d_and_der',
    'Q2d_sag',
    'ray_plane_intersect',
    'ray_sphere_intersect',
    'ray_conic_intersect',
    '_ensure_P_vec',
    '_none_or_rotmat',
    '_apply_tilt_decenter',
]
