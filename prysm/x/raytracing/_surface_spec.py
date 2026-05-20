"""Shared normalized surface specs for prescription readers."""

from dataclasses import dataclass, field

from .surfaces import (
    Surface,
    BiconicSag,
    ConicSag,
    EvenAsphereSag,
    PlaneSag,
    ToroidSag,
    XYSag,
    ZernikeSag,
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


def build_surface(spec):
    """Build a Surface from a normalized parser spec."""
    common = dict(typ=spec.typ, P=spec.P, n=spec.n, R=spec.R,
                  bounding=spec.bounding, aperture=spec.aperture,
                  tilt=spec.tilt, decenter=spec.decenter,
                  tilt_radians=spec.tilt_radians, grating=spec.grating)
    kind = spec.kind
    p = spec.params
    if kind == 'plane':
        return Surface(shape=PlaneSag(), **common)
    if kind == 'conic':
        return Surface(shape=ConicSag(p.get('c', 0.0), p.get('k', 0.0)),
                       **common)
    if kind == 'even_asphere':
        return Surface(shape=EvenAsphereSag(p.get('c', 0.0),
                                            p.get('k', 0.0),
                                            p.get('coefs', ())),
                       **common)
    if kind == 'toroid':
        return Surface(shape=ToroidSag(p['c_x'], p['c_y'], p['k_y'],
                                       p.get('coefs_y', ())),
                       **common)
    if kind == 'biconic':
        return Surface(shape=BiconicSag(p['c_x'], p['c_y'],
                                        p.get('k_x', 0.0),
                                        p.get('k_y', 0.0)),
                       **common)
    if kind == 'zernike':
        return Surface(shape=ZernikeSag(
            p.get('c', 0.0), p.get('k', 0.0), p['normalization_radius'],
            p['nms'], p['coefs'], norm=p.get('norm', True),
        ), **common)
    if kind == 'xy':
        return Surface(shape=XYSag(
            p.get('c', 0.0), p.get('k', 0.0), p['normalization_radius'],
            p['mns'], p['coefs'],
        ), **common)
    raise NotImplementedError(f'unknown surface spec kind {kind!r}')


__all__ = ['SurfaceSpec', 'build_surface']
