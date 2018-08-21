"""Basic Zernike functions."""

from prysm import mathops as m
from prysm.conf import config


# See JCW - http://wp.optics.arizona.edu/jcwyant/wp-content/uploads/sites/13/2016/08/ZernikePolynomialsForTheWeb.pdf


def piston(rho, phi):
    return m.ones(rho.shape)


def tip(rho, phi):
    return rho * m.cos(phi)


def tilt(rho, phi):
    return rho * m.sin(phi)


def defocus(rho, phi):
    return 2 * rho**2 - 1


def primary_astigmatism_00(rho, phi):
    return rho**2 * m.cos(2 * phi)


def primary_astigmatism_45(rho, phi):
    return rho**2 * m.sin(2 * phi)


def primary_coma_y(rho, phi):
    return (-2 * rho + 3 * rho**3) * m.cos(phi)


def primary_coma_x(rho, phi):
    return (-2 * rho + 3 * rho**3) * m.sin(phi)


def primary_spherical(rho, phi):
    return 6 * rho**4 - 6 * rho**2 + 1


def primary_trefoil_y(rho, phi):
    return rho**3 * m.cos(3 * phi)


def primary_trefoil_x(rho, phi):
    return rho**3 * m.sin(3 * phi)


def secondary_astigmatism_00(rho, phi):
    return (-3 * rho**2 + 4 * rho**4) * m.cos(2 * phi)


def secondary_astigmatism_45(rho, phi):
    return (-3 * rho**2 + 4 * rho**4) * m.sin(2 * phi)


def secondary_coma_y(rho, phi):
    return (3 * rho - 12 * rho**3 + 10 * rho**5) * m.cos(phi)


def secondary_coma_x(rho, phi):
    return (3 * rho - 12 * rho**3 + 10 * rho**5) * m.sin(phi)


def secondary_spherical(rho, phi):
    return 20 * rho**6 + - 30 * rho**4 + 12 * rho**2 - 1


def primary_tetrafoil_y(rho, phi):
    return rho**4 * m.cos(4 * phi)


def primary_tetrafoil_x(rho, phi):
    return rho**4 * m.sin(4 * phi)


def secondary_trefoil_y(rho, phi):
    return (5 * rho**5 - 4 * rho**3) * m.cos(3 * phi)


def secondary_trefoil_x(rho, phi):
    return (5 * rho**5 - 4 * rho**3) * m.sin(3 * phi)


def tertiary_astigmatism_00(rho, phi):
    return (6 * rho**2 - 20 * rho**4 + 15 * rho**6) * m.cos(2 * phi)


def tertiary_astigmatism_45(rho, phi):
    return (6 * rho**2 - 20 * rho**4 + 15 * rho**6) * m.sin(2 * phi)


def tertiary_coma_y(rho, phi):
    return (-4 * rho + 30 * rho**3 - 60 * rho**5 + 35 * rho**7) * m.cos(phi)


def tertiary_coma_x(rho, phi):
    return (-4 * rho + 30 * rho**3 - 60 * rho**5 + 35 * rho**7) * m.sin(phi)


def tertiary_spherical(rho, phi):
    return 70 * rho**8 - 140 * rho**6 + 90 * rho**4 - 20 * rho**2 + 1


def primary_pentafoil_y(rho, phi):
    return rho**5 * m.cos(5 * phi)


def primary_pentafoil_x(rho, phi):
    return rho**5 * m.sin(5 * phi)


def secondary_tetrafoil_y(rho, phi):
    return (6 * rho**6 - 5 * rho**4) * m.cos(4 * phi)


def secondary_tetrafoil_x(rho, phi):
    return (6 * rho**6 - 5 * rho**4) * m.sin(4 * phi)


def tertiary_trefoil_y(rho, phi):
    return (10 * rho**3 - 30 * rho**5 + 21 * rho**7) * m.cos(3 * phi)


def tertiary_trefoil_x(rho, phi):
    return (10 * rho**3 - 30 * rho**5 + 21 * rho**7) * m.cos(3 * phi)


def quaternary_astigmatism_00(rho, phi):
    return (10 * rho**2 - 30 * rho**4 + 21 * rho**6) * m.cos(2 * phi)


def quaternary_astigmatism_45(rho, phi):
    return (10 * rho**2 - 30 * rho**4 + 21 * rho**6) * m.sin(2 * phi)


def quaternary_coma_y(rho, phi):
    return (5 * rho - 60 * rho**3 + 210 * rho**5 - 280 * rho**7 + 126 * rho**9)\
        * m.cos(phi)


def quaternary_coma_x(rho, phi):
    return (5 * rho - 60 * rho**3 + 210 * rho**5 - 280 * rho**7 + 126 * rho**9)\
        * m.sin(phi)


def quaternary_spherical(rho, phi):
    return 252 * rho**10 \
        - 630 * rho**8 \
        + 560 * rho**6 \
        - 210 * rho**4 \
        + 30 * rho**2 \
        - 1


def primary_hexafoil_y(rho, phi):
    return rho**6 * m.cos(6 * phi)


def primary_hexafoil_x(rho, phi):
    return rho**6 * m.sin(6 * phi)


def secondary_pentafoil_y(rho, phi):
    return (7 * rho**7 - 6 * rho**5) * m.cos(5 * phi)


def secondary_pentafoil_x(rho, phi):
    return (7 * rho**7 - 6 * rho**5) * m.sin(5 * phi)


def tertiary_tetrafoil_y(rho, phi):
    return (28 * rho**8 - 42 * rho**6 + 15 * rho**4) * m.cos(4 * phi)


def tertiary_tetrafoil_x(rho, phi):
    return (28 * rho**8 - 42 * rho**6 + 15 * rho**4) * m.sin(4 * phi)


def quaternary_trefoil_y(rho, phi):
    return (84 * rho**9 - 168 * rho**7 + 105 * rho**5 - 20 * rho**3) * m.cos(3 * phi)


def quaternary_trefoil_x(rho, phi):
    return (84 * rho**9 - 168 * rho**7 + 105 * rho**5 - 20 * rho**3) * m.sin(3 * phi)


def quinternary_astigmatism_00(rho, phi):
    return (210 * rho**10 - 504 * rho**8 + 420 * rho**6 - 140 * rho**4 + 15 * rho**2) \
        * m.cos(2 * phi)


def quinternary_astigmatism_45(rho, phi):
    return (210 * rho**10 - 504 * rho**8 + 420 * rho**6 - 140 * rho**4 + 15 * rho**2) \
        * m.sin(2 * phi)


def quinternary_coma_y(rho, phi):
    return (462 * rho**11 - 1260 * rho**9 + 1260 * rho**7 - 560 * rho**5 + 105 * rho**3 - 6 * rho) \
        * m.cos(phi)


def quinternary_coma_x(rho, phi):
    return (462 * rho**11 - 1260 * rho**9 + 1260 * rho**7 - 560 * rho**5 + 105 * rho**3 - 6 * rho) \
        * m.sin(phi)


def quinternary_spherical(rho, phi):
    return 924 * rho**12 \
        - 2772 * rho**10 \
        + 3150 * rho**8 \
        - 1680 * rho**6 \
        + 420 * rho**4 \
        - 42 * rho**2 \
        + 1


# norms
piston.norm = 1
tip.norm = 2
tilt.norm = 2
defocus.norm = m.sqrt(3)
primary_astigmatism_00.norm = m.sqrt(6)
primary_astigmatism_45.norm = m.sqrt(6)
primary_coma_y.norm = 2 * m.sqrt(2)
primary_coma_x.norm = 2 * m.sqrt(2)
primary_spherical.norm = m.sqrt(5)
primary_trefoil_y.norm = 2 * m.sqrt(2)
primary_trefoil_x.norm = 2 * m.sqrt(2)
secondary_astigmatism_00.norm = m.sqrt(10)
secondary_astigmatism_45.norm = m.sqrt(10)
secondary_coma_y.norm = 2 * m.sqrt(3)
secondary_coma_x.norm = 2 * m.sqrt(3)
secondary_spherical.norm = m.sqrt(7)
primary_tetrafoil_y.norm = m.sqrt(10)
primary_tetrafoil_x.norm = m.sqrt(10)
secondary_trefoil_y.norm = 2 * m.sqrt(3)
secondary_trefoil_x.norm = 2 * m.sqrt(3)
tertiary_astigmatism_00.norm = m.sqrt(14)
tertiary_astigmatism_45.norm = m.sqrt(14)
tertiary_coma_y.norm = 4
tertiary_coma_x.norm = 4
tertiary_spherical.norm = 3
primary_pentafoil_y.norm = 2 * m.sqrt(3)
primary_pentafoil_x.norm = 2 * m.sqrt(3)
secondary_tetrafoil_y.norm = m.sqrt(14)
secondary_tetrafoil_x.norm = m.sqrt(14)
tertiary_trefoil_y.norm = 4
tertiary_trefoil_x.norm = 4
quaternary_astigmatism_00.norm = 3 * m.sqrt(2)
quaternary_astigmatism_45.norm = 3 * m.sqrt(2)
quaternary_coma_y.norm = 2 * m.sqrt(5)
quaternary_coma_x.norm = 2 * m.sqrt(5)
quaternary_spherical.norm = m.sqrt(11)
primary_hexafoil_y.norm = m.sqrt(14)
primary_hexafoil_x.norm = m.sqrt(14)
secondary_pentafoil_y.norm = 4
secondary_pentafoil_x.norm = 4
tertiary_tetrafoil_y.norm = 3 * m.sqrt(2)
tertiary_tetrafoil_x.norm = 3 * m.sqrt(2)
quaternary_trefoil_y.norm = 2 * m.sqrt(5)
quaternary_trefoil_x.norm = 2 * m.sqrt(5)
quinternary_astigmatism_00.norm = m.sqrt(22)
quinternary_astigmatism_45.norm = m.sqrt(22)
quinternary_coma_y.norm = 2 * m.sqrt(6)
quinternary_coma_x.norm = 2 * m.sqrt(6)
quinternary_spherical.norm = m.sqrt(13)

# names
piston.name = 'Piston'
tip.name = 'Tip (Y)'
tilt.name = 'Tilt (X)'
defocus.name = 'Defocus'
primary_astigmatism_00.name = 'Primary Astigmatism 0°'
primary_astigmatism_45.name = 'Primary Astigmatism 45°'
primary_coma_y.name = 'Primary Coma Y'
primary_coma_x.name = 'Primary Coma X'
primary_spherical.name = 'Primary Spherical'
primary_trefoil_y.name = 'Primary Trefoil Y'
primary_trefoil_x.name = 'Primary Trefoil X'
secondary_astigmatism_00.name = 'Secondary Astigmatism 0°'
secondary_astigmatism_45.name = 'Secondary Astigmatism 45°'
secondary_coma_y.name = 'Secondary Coma Y'
secondary_coma_x.name = 'Secondary Coma X'
secondary_spherical.name = 'Secondary Spherical'
primary_tetrafoil_y.name = 'Primary Tetrafoil Y'
primary_tetrafoil_x.name = 'Primary Tetrafoil X'
secondary_trefoil_y.name = 'Secondary Trefoil Y'
secondary_trefoil_x.name = 'Secondary Trefoil X'
tertiary_astigmatism_00.name = 'Tertiary Astigmatism 0°'
tertiary_astigmatism_45.name = 'Tertiary Astigmatism 45°'
tertiary_coma_y.name = 'Tertiary Coma Y'
tertiary_coma_x.name = 'Tertiary Coma X'
tertiary_spherical.name = 'Tertiary Spherical'
primary_pentafoil_y.name = 'Primary Pentafoil Y'
primary_pentafoil_x.name = 'Primary Pentafoil X'
secondary_tetrafoil_y.name = 'Secondary Tetrafoil Y'
secondary_tetrafoil_x.name = 'Secondary Tetrafoil X'
tertiary_trefoil_y.name = 'Tertiary Trefoil Y'
tertiary_trefoil_x.name = 'Tertiary Teefoil X'
quaternary_astigmatism_00.name = 'Quaternary Astigmatism 0°'
quaternary_astigmatism_45.name = 'Quaternary Astigmatism 45°'
quaternary_coma_y.name = 'Quaternary Coma Y'
quaternary_coma_x.name = 'Quaternary Coma X'
quaternary_spherical.name = 'Quaternary Spherical'
primary_hexafoil_y.name = 'Primary Hexafoil Y'
primary_hexafoil_x.name = 'Primary Hexafoil X'
secondary_pentafoil_y.name = 'Secondary Pentafoil Y'
secondary_pentafoil_x.name = 'Secondary Pentafoil X'
tertiary_tetrafoil_y.name = 'Tertiary Tetrafoil Y'
tertiary_tetrafoil_x.name = 'Tertiary Tetrafoil X'
quaternary_trefoil_y.name = 'Quaternary Trefoil Y'
quaternary_trefoil_x.name = 'Quaternary Trefoil X'
quinternary_astigmatism_00.name = 'Quinternary Astigmatism 0°'
quinternary_astigmatism_45.name = 'Quinternary Astigmatism 45°'
quinternary_coma_y.name = 'Quinternary Coma Y'
quinternary_coma_x.name = 'Quinternary Coma X'
quinternary_spherical.name = 'Quinternary Spherical'

_zernikes = [
    piston,
    tip,
    tilt,
    defocus,
    primary_astigmatism_00,
    primary_astigmatism_45,
    primary_coma_y,
    primary_coma_x,
    primary_spherical,
    primary_trefoil_y,
    primary_trefoil_x,
    secondary_astigmatism_00,
    secondary_astigmatism_45,
    secondary_coma_y,
    secondary_coma_x,
    secondary_spherical,
    primary_tetrafoil_y,
    primary_tetrafoil_x,
    secondary_trefoil_y,
    secondary_trefoil_x,
    tertiary_astigmatism_00,
    tertiary_astigmatism_45,
    tertiary_coma_y,
    tertiary_coma_x,
    tertiary_spherical,
    primary_pentafoil_y,
    primary_pentafoil_x,
    secondary_tetrafoil_y,
    secondary_tetrafoil_x,
    tertiary_trefoil_y,
    tertiary_trefoil_x,
    quaternary_astigmatism_00,
    quaternary_astigmatism_45,
    quaternary_coma_y,
    quaternary_coma_x,
    quaternary_spherical,
    primary_hexafoil_y,
    primary_hexafoil_x,
    secondary_pentafoil_y,
    secondary_pentafoil_x,
    tertiary_tetrafoil_y,
    tertiary_tetrafoil_x,
    quaternary_trefoil_y,
    quaternary_trefoil_x,
    quinternary_astigmatism_00,
    quinternary_astigmatism_45,
    quinternary_coma_y,
    quinternary_coma_x,
    quinternary_spherical
]


# need to disable this for now, numba doesn't preserve function attrs with vectorize
# zernikes_cpu = [m.vectorize(func) for func in _zernikes[1:]]  # numba compiled zernikes
# zernikes_cpu.insert(0, m.jit(_zernikes[0]))
zernikes_cpu = [m.jit(_zernikes[0])]
for func in _zernikes[1:]:
    compfunc = m.vectorize(func)
    compfunc.name = func.name
    compfunc.norm = func.norm
    zernikes_cpu.append(compfunc)

zernikes_gpu = [m.fuse(func) for func in _zernikes[1:]]  # cupy compiled zernikes
zernikes_gpu.insert(0, _zernikes[0])


def change_backend(to):
    if to == 'cu':
        globals()['zernikes'] = zernikes_gpu
    elif to == 'np':
        globals()['zernikes'] = zernikes_cpu


config.chbackend_observers.append(change_backend)
config.backend = config.backend  # trigger import of math functions
