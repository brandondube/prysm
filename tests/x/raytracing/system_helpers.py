"""Shared OpticalSystem builders for raytracing tests."""

from prysm.x import materials
from prysm.x.raytracing import (
    ApertureSpec,
    Conic,
    Field,
    LensData,
    OpticalSystem,
    Plane,
    Sphere,
)


def singlet_system(fields=None, wavelengths=None, ref=1):
    """Sphere/sphere singlet with a stop at the first powered surface."""
    lens = LensData()
    (lens.add(
        Conic(1 / 50.0, 0.0), typ='refr',
        material=materials.ConstantMaterial(1.5168), thickness=5.0,
    ).add(
        Conic(-1 / 50.0, 0.0), typ='refr', material=materials.air,
        thickness=95.0,
    ).add(Plane(), typ='eval'))
    if fields is None:
        fields = [Field(0, 0), Field(0, 3)]
    if wavelengths is None:
        wavelengths = [0.4861, 0.5876, 0.6563]
    return OpticalSystem(
        lens, aperture=ApertureSpec.epd(10.0), fields=fields,
        wavelengths=wavelengths, reference=ref, stop_index=0,
    )


def doublet_system(aperture=None):
    """Return the common three-surface doublet analysis system."""
    lens = (LensData()
            .add(Sphere(1 / 61.47), thickness=6.0,
                 material=materials.ConstantMaterial(1.5168), aperture=12.0)
            .add(Sphere(-1 / 44.64), thickness=2.5,
                 material=materials.ConstantMaterial(1.673), aperture=12.0)
            .add(Sphere(-1 / 129.94), thickness=0.0,
                 material=materials.air, aperture=12.0))
    system = OpticalSystem(
        lens, aperture=aperture or ApertureSpec.epd(22.0),
        fields=[Field(0, 0), Field(0, 0.7), Field(0, 1.0)],
        wavelengths=[0.486, 0.587, 0.656], reference=1, stop_index=1,
    )
    system.solve.image_distance()
    return system
