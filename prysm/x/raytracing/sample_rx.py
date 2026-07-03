"""Sample prescriptions for tests and notebooks."""
from prysm.x.materials import FormulaMaterial, model_glass, air
from prysm.x.materials.formulas import sellmeier
from prysm.x.raytracing import (
    OpticalSystem, ApertureSpec, FieldSet, Field, LensData, Sphere, Conic,
    Plane,
)

# the two glasses of the 101 doublet (Schott Sellmeier coefficients)
N_BK7 = FormulaMaterial('N-BK7', sellmeier,
    [[1.039612120, 0.231792344, 1.010469450],
     [0.006000699, 0.0200179144, 103.56065300]])
N_SF5 = FormulaMaterial('N-SF5', sellmeier,
    [[1.524818890, 0.187085527, 1.427290150],
     [0.011254756, 0.0588995392, 129.14167500]])


def doublet(rear_semidiameter=12.0):
    """75 mm EFL f/3.4 crown-flint pair; stop on a front dummy plane."""
    return (LensData()
        .add(Plane(), typ='eval', thickness=10)            # front padding (cosmetic)
        .add(Plane(), typ='eval', thickness=0)             # the aperture stop
        .add(Sphere(1/46.44),  thickness=7,   material=N_BK7, aperture=12)
        .add(Sphere(-1/33.77), thickness=2.5, material=N_SF5, aperture=12)
        .add(Sphere(-1/95.94), thickness=0,   material=air,  aperture=rear_semidiameter))


def doublet_conic(rear_semidiameter=12.0):
    """The doublet on conic surfaces, so conic constants are perturbable DOFs."""
    return (LensData()
        .add(Plane(), typ='eval', thickness=10)            # front padding (cosmetic)
        .add(Plane(), typ='eval', thickness=0)             # the aperture stop
        .add(Conic(1/46.44, 0.),  thickness=7,   material=N_BK7, aperture=12)
        .add(Conic(-1/33.77, 0.), thickness=2.5, material=N_SF5, aperture=12)
        .add(Conic(-1/95.94, 0.), thickness=0,   material=air,  aperture=rear_semidiameter))


def fold_mirror(tilt=(0.0, 0.0, 45.0)):
    """Flat fold: 20 mm air path, ben break, mirror; image 15 mm up the folded axis."""
    return (LensData()
        .add(Plane(), typ='refr', material=air, thickness=20.0)
        .add_coordbreak(tilt=tilt, kind='ben')
        .add(Plane(), typ='refl', thickness=15.0))


def decentered_singlet(dy=2.0):
    """Biconvex singlet decentered dy in y between rev-coupled breaks; steers the beam."""
    return (LensData()
        .add(Plane(), typ='eval', thickness=5.0)                 # object padding
        .add_coordbreak(decenter=(0.0, dy, 0.0), kind='basic')   # shift frame up
        .add(Sphere(1/40.0),  thickness=5.0, material=N_BK7, aperture=12)
        .add(Sphere(-1/40.0), thickness=0.0, material=air,  aperture=12)
        .add_coordbreak(decenter=(0.0, dy, 0.0), kind='rev')     # undo
        .add(Plane(), typ='eval', thickness=60.0, aperture=20.0))


# the stop is the 10th powered surface (compiled index; OBJECT is 0)
FISHEYE_STOP_INDEX = 10


def fisheye():
    """Smith MLD ch.14 p.411 f/8 170-degree fish-eye.

    A native transcription of the Code V prescription; the manufacturer glasses
    are inlined as model-glass (nd, Vd) stand-ins so the design carries no AGF
    dependency.  Geometry and first-order properties match the source.
    """
    return (LensData()
        .add(Sphere(1/599.38300),   thickness=35.030,  material=model_glass(1.5168, 64.17), aperture=448.40)
        .add(Sphere(1/235.82500),   thickness=190.161, material=air)
        .add(Sphere(1/605.51300),   thickness=30.025,  material=model_glass(1.4875, 70.41))
        .add(Sphere(1/111.09400),   thickness=120.102, material=air)
        .add(Sphere(-1/452.38400),  thickness=10.008,  material=model_glass(1.4875, 70.41))
        .add(Sphere(1/127.73300),   thickness=45.038,  material=model_glass(1.7847, 26.10))
        .add(Sphere(1/462.89200),   thickness=25.021,  material=air)
        .add(Plane(),               thickness=15.013,  material=model_glass(1.5182, 58.98))
        .add(Plane(),               thickness=36.281,  material=air)
        .add(Plane(),               thickness=13.762,  material=air)              # aperture stop
        .add(Sphere(1/38507.64900), thickness=10.008,  material=model_glass(1.7847, 26.10))
        .add(Sphere(1/95.08100),    thickness=110.093, material=model_glass(1.7440, 44.72))
        .add(Sphere(-1/162.63800),  thickness=130.110, material=air)
        .add(Sphere(1/1376.16700),  thickness=20.017,  material=model_glass(1.7847, 26.10))
        .add(Sphere(1/177.27500),   thickness=150.127, material=model_glass(1.7020, 41.00), aperture=139.00)
        .add(Sphere(-1/400.33900),  thickness=18.766,  material=model_glass(1.6676, 41.93), aperture=139.00)
        .add(Sphere(-1/337.53600),  thickness=150.059, material=air,                        aperture=139.00))


def fisheye_system(fields=(0.0, 30.0, 50.0),
                   wavelengths=(0.6562725, 0.5875618, 0.4861327)):
    """The fish-eye as an f/8 OpticalSystem at robust teaching fields."""
    sys = OpticalSystem(
        fisheye(),
        aperture=ApertureSpec.fno(8),
        fields=FieldSet([Field(0, h, unit='deg') for h in fields]),
        wavelengths=list(wavelengths),
        reference=1,
        stop_index=FISHEYE_STOP_INDEX,
    )
    sys.solve.image_distance()
    return sys
