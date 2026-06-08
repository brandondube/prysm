"""Test-only surface constructors for explicit shape-based raytracing tests."""

from prysm.x.raytracing.analysis import wavefront, resolve_exit_pupil
from prysm.x.raytracing.spencer_and_murty import raytrace
from prysm.x.raytracing.opt import _pupil_center_chief_index
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


def wavefront_with_resolved_exit_pupil(
        prescription, P, S, wavelength, *, chief_index=None, stop_index=None,
        epd=None, axis_point=None, axis_dir=None, **kw):
    """Resolve P_xp from the traced chief ray, then evaluate wavefront."""
    tr = raytrace(prescription, P, S, wavelength)
    ci = chief_index if chief_index is not None else _pupil_center_chief_index(P)
    P_xp = resolve_exit_pupil(prescription, wavelength, stop_index=stop_index,
                              epd=epd, chief=(tr.P[-1, ci], tr.S[-1, ci]),
                              axis_point=axis_point, axis_dir=axis_dir)
    return wavefront(prescription, P, S, wavelength, P_xp=P_xp,
                     chief_index=chief_index, **kw)
