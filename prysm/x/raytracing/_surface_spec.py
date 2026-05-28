"""Shared normalized surface specs for prescription readers."""

from dataclasses import dataclass, field

from .surfaces import (
    Surface,
    Biconic,
    Conic,
    EvenAsphere,
    Plane,
    Toroid,
    XY,
    Zernike,
)


@dataclass
class SurfaceSpec:
    """Parser-neutral surface construction record."""

    kind: str
    typ: str
    P: object
    n: object = None
    params: dict = field(default_factory=dict)
    R: object = None
    bounding: object = None
    aperture: object = None
    tilt: object = None
    decenter: object = None
    tilt_radians: bool = False
    grating: object = None


def build_shape(spec):
    """Build the Shape object for a normalized parser spec (no pose)."""
    kind = spec.kind
    p = spec.params
    if kind == 'plane':
        return Plane()
    if kind == 'conic':
        return Conic(p.get('c', 0.0), p.get('k', 0.0))
    if kind == 'even_asphere':
        return EvenAsphere(p.get('c', 0.0), p.get('k', 0.0),
                              p.get('coefs', ()))
    if kind == 'toroid':
        return Toroid(p['c_x'], p['c_y'], p['k_y'], p.get('coefs_y', ()))
    if kind == 'biconic':
        return Biconic(p['c_x'], p['c_y'], p.get('k_x', 0.0),
                          p.get('k_y', 0.0))
    if kind == 'zernike':
        return Zernike(p.get('c', 0.0), p.get('k', 0.0),
                          p['normalization_radius'], p['nms'], p['coefs'],
                          norm=p.get('norm', True))
    if kind == 'xy':
        return XY(p.get('c', 0.0), p.get('k', 0.0),
                     p['normalization_radius'], p['mns'], p['coefs'])
    raise NotImplementedError(f'unknown surface spec kind {kind!r}')


def build_surface(spec):
    """Build a posed Surface from a normalized parser spec."""
    return Surface(
        shape=build_shape(spec),
        typ=spec.typ, P=spec.P, n=spec.n, R=spec.R,
        bounding=spec.bounding, aperture=spec.aperture,
        tilt=spec.tilt, decenter=spec.decenter,
        tilt_radians=spec.tilt_radians, grating=spec.grating,
    )


__all__ = ['SurfaceSpec', 'build_surface', 'build_shape']
