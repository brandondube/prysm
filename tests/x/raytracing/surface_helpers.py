"""Test-only surface constructors for explicit shape-based raytracing tests."""

from prysm.x.raytracing.surfaces import (
    Surface,
    Biconic,
    Chebyshev,
    Conic,
    EvenAsphere,
    Jacobi,
    OffAxisConic,
    Plane,
    Q2D,
    Sphere,
    Toroid,
    XY,
    Zernike,
)


def plane(typ, P, n=None, **kwargs):
    return Surface(shape=Plane(), typ=typ, P=P, n=n, **kwargs)


def sphere(c, typ, P, n, **kwargs):
    return Surface(shape=Sphere(c), typ=typ, P=P, n=n, **kwargs)


def conic(c, k, typ, P, n=None, **kwargs):
    return Surface(shape=Conic(c, k), typ=typ, P=P, n=n, **kwargs)


def off_axis_conic(c, k, typ, P, dx=0, dy=0, n=None, **kwargs):
    shape = OffAxisConic(c, k, dx=dx, dy=dy)
    return Surface(shape=shape, typ=typ, P=P, n=n, **kwargs)


def even_asphere(c, k, coefs, typ, P, n=None, **kwargs):
    return Surface(shape=EvenAsphere(c, k, coefs), typ=typ, P=P, n=n,
                   **kwargs)


def q2d(c, k, normalization_radius, cm0, ams, bms, typ, P, dx=0, dy=0,
        n=None, **kwargs):
    shape = Q2D(c, k, normalization_radius, cm0, ams, bms, dx=dx, dy=dy)
    return Surface(shape=shape, typ=typ, P=P, n=n, **kwargs)


def zernike(c, k, normalization_radius, nms, coefs, typ, P, n=None,
            norm=True, **kwargs):
    shape = Zernike(c, k, normalization_radius, nms, coefs, norm=norm)
    return Surface(shape=shape, typ=typ, P=P, n=n, **kwargs)


def xy(c, k, normalization_radius, mns, coefs, typ, P, n=None, **kwargs):
    shape = XY(c, k, normalization_radius, mns, coefs)
    return Surface(shape=shape, typ=typ, P=P, n=n, **kwargs)


def chebyshev(c, k, x_norm, y_norm, mns, coefs, typ, P, n=None, **kwargs):
    shape = Chebyshev(c, k, x_norm, y_norm, mns, coefs)
    return Surface(shape=shape, typ=typ, P=P, n=n, **kwargs)


def jacobi(c, k, normalization_radius, alpha, beta, ns, coefs, typ, P,
           n=None, **kwargs):
    shape = Jacobi(c, k, normalization_radius, alpha, beta, ns, coefs)
    return Surface(shape=shape, typ=typ, P=P, n=n, **kwargs)


def toroid(c_x, c_y, k_y, coefs_y, typ, P, n=None, **kwargs):
    return Surface(shape=Toroid(c_x, c_y, k_y, coefs_y), typ=typ, P=P,
                   n=n, **kwargs)


def biconic(c_x, c_y, k_x, k_y, typ, P, n=None, **kwargs):
    return Surface(shape=Biconic(c_x, c_y, k_x, k_y), typ=typ, P=P,
                   n=n, **kwargs)
