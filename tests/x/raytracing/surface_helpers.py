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


def plane(interaction, P, material=None, **kwargs):
    return Surface(shape=Plane(), interaction=interaction, P=P,
                   material=material, **kwargs)


def sphere(c, interaction, P, material=None, **kwargs):
    return Surface(shape=Sphere(c), interaction=interaction, P=P,
                   material=material, **kwargs)


def conic(c, k, interaction, P, material=None, **kwargs):
    return Surface(shape=Conic(c, k), interaction=interaction, P=P,
                   material=material, **kwargs)


def off_axis_conic(c, k, interaction, P, dx=0, dy=0, material=None, **kwargs):
    shape = OffAxisConic(c, k, dx=dx, dy=dy)
    return Surface(shape=shape, interaction=interaction, P=P,
                   material=material, **kwargs)


def even_asphere(c, k, coefs, interaction, P, material=None, **kwargs):
    return Surface(shape=EvenAsphere(c, k, coefs), interaction=interaction, P=P,
                   material=material, **kwargs)


def q2d(c, k, normalization_radius, cm0, ams, bms, interaction, P, dx=0, dy=0,
        material=None, **kwargs):
    shape = Q2D(c, k, normalization_radius, cm0, ams, bms, dx=dx, dy=dy)
    return Surface(shape=shape, interaction=interaction, P=P,
                   material=material, **kwargs)


def zernike(c, k, normalization_radius, nms, coefs, interaction, P, material=None,
            norm=True, **kwargs):
    shape = Zernike(c, k, normalization_radius, nms, coefs, norm=norm)
    return Surface(shape=shape, interaction=interaction, P=P,
                   material=material, **kwargs)


def xy(c, k, normalization_radius, mns, coefs, interaction, P, material=None,
       **kwargs):
    shape = XY(c, k, normalization_radius, mns, coefs)
    return Surface(shape=shape, interaction=interaction, P=P,
                   material=material, **kwargs)


def chebyshev(c, k, x_norm, y_norm, mns, coefs, interaction, P, material=None,
              **kwargs):
    shape = Chebyshev(c, k, x_norm, y_norm, mns, coefs)
    return Surface(shape=shape, interaction=interaction, P=P,
                   material=material, **kwargs)


def jacobi(c, k, normalization_radius, alpha, beta, ns, coefs, interaction, P,
           material=None, **kwargs):
    shape = Jacobi(c, k, normalization_radius, alpha, beta, ns, coefs)
    return Surface(shape=shape, interaction=interaction, P=P,
                   material=material, **kwargs)


def toroid(c_x, c_y, k_y, coefs_y, interaction, P, material=None, **kwargs):
    return Surface(shape=Toroid(c_x, c_y, k_y, coefs_y),
                   interaction=interaction, P=P, material=material, **kwargs)


def biconic(c_x, c_y, k_x, k_y, interaction, P, material=None, **kwargs):
    return Surface(shape=Biconic(c_x, c_y, k_x, k_y), interaction=interaction,
                   P=P, material=material, **kwargs)
