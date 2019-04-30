"""Zernike functions."""
from collections import defaultdict
from functools import partial

from .conf import config
from .mathops import engine as e, jit, vectorize, fuse
from .pupil import Pupil
from .coordinates import make_rho_phi_grid, cart_to_polar
from .util import rms, share_fig_ax, sort_xy

# See JCW - http://wp.optics.arizona.edu/jcwyant/wp-content/uploads/sites/13/2016/08/ZernikePolynomialsForTheWeb.pdf


def piston(rho, phi):
    """Zernike Piston."""
    return e.ones(rho.shape)


def tip(rho, phi):
    """Zernike Tilt-Y."""
    return rho * e.cos(phi)


def tilt(rho, phi):
    """Zernike Tilt-X."""
    return rho * e.sin(phi)


def defocus(rho, phi):
    """Zernike defocus."""
    return 2 * rho**2 - 1


def primary_astigmatism_00(rho, phi):
    """Zernike primary astigmatism 0°."""
    return rho**2 * e.cos(2 * phi)


def primary_astigmatism_45(rho, phi):
    """Zernike primary astigmatism 45°."""
    return rho**2 * e.sin(2 * phi)


def primary_coma_y(rho, phi):
    """Zernike primary coma Y."""
    return (-2 * rho + 3 * rho**3) * e.cos(phi)


def primary_coma_x(rho, phi):
    """Zernike primary coma X."""
    return (-2 * rho + 3 * rho**3) * e.sin(phi)


def primary_spherical(rho, phi):
    """Zernike primary Spherical."""
    return 6 * rho**4 - 6 * rho**2 + 1


def primary_trefoil_y(rho, phi):
    """Zernike primary trefoil Y."""
    return rho**3 * e.cos(3 * phi)


def primary_trefoil_x(rho, phi):
    """Zernike primary trefoil X."""
    return rho**3 * e.sin(3 * phi)


def secondary_astigmatism_00(rho, phi):
    """Zernike secondary astigmatism 0°."""
    return (-3 * rho**2 + 4 * rho**4) * e.cos(2 * phi)


def secondary_astigmatism_45(rho, phi):
    """Zernike secondary astigmatism 45°."""
    return (-3 * rho**2 + 4 * rho**4) * e.sin(2 * phi)


def secondary_coma_y(rho, phi):
    """Zernike secondary coma Y."""
    return (3 * rho - 12 * rho**3 + 10 * rho**5) * e.cos(phi)


def secondary_coma_x(rho, phi):
    """Zernike secondary coma X."""
    return (3 * rho - 12 * rho**3 + 10 * rho**5) * e.sin(phi)


def secondary_spherical(rho, phi):
    """Zernike secondary spherical."""
    return 20 * rho**6 + - 30 * rho**4 + 12 * rho**2 - 1


def primary_tetrafoil_y(rho, phi):
    """Zernike primary tetrafoil Y."""
    return rho**4 * e.cos(4 * phi)


def primary_tetrafoil_x(rho, phi):
    """Zernike primary tetrafoil X."""
    return rho**4 * e.sin(4 * phi)


def secondary_trefoil_y(rho, phi):
    """Zernike secondary trefoil Y."""
    return (5 * rho**5 - 4 * rho**3) * e.cos(3 * phi)


def secondary_trefoil_x(rho, phi):
    """Zernike secondary trefoil X."""
    return (5 * rho**5 - 4 * rho**3) * e.sin(3 * phi)


def tertiary_astigmatism_00(rho, phi):
    """Zernike tertiary astigmatism 0°."""
    return (6 * rho**2 - 20 * rho**4 + 15 * rho**6) * e.cos(2 * phi)


def tertiary_astigmatism_45(rho, phi):
    """Zernike tertiary astigmatism 45°."""
    return (6 * rho**2 - 20 * rho**4 + 15 * rho**6) * e.sin(2 * phi)


def tertiary_coma_y(rho, phi):
    """Zernike tertiary coma Y."""
    return (-4 * rho + 30 * rho**3 - 60 * rho**5 + 35 * rho**7) * e.cos(phi)


def tertiary_coma_x(rho, phi):
    """Zernike tertiary coma X."""
    return (-4 * rho + 30 * rho**3 - 60 * rho**5 + 35 * rho**7) * e.sin(phi)


def tertiary_spherical(rho, phi):
    """Zernike tertiary spherical."""
    return 70 * rho**8 - 140 * rho**6 + 90 * rho**4 - 20 * rho**2 + 1


def primary_pentafoil_y(rho, phi):
    """Zernike primary pentafoil Y."""
    return rho**5 * e.cos(5 * phi)


def primary_pentafoil_x(rho, phi):
    """Zernike primary pentafoil X."""
    return rho**5 * e.sin(5 * phi)


def secondary_tetrafoil_y(rho, phi):
    """Zernike secondary tetrafoil Y."""
    return (6 * rho**6 - 5 * rho**4) * e.cos(4 * phi)


def secondary_tetrafoil_x(rho, phi):
    """Zernike secondary tetrafoil X."""
    return (6 * rho**6 - 5 * rho**4) * e.sin(4 * phi)


def tertiary_trefoil_y(rho, phi):
    """Zernike tertiary trefoil Y."""
    return (10 * rho**3 - 30 * rho**5 + 21 * rho**7) * e.cos(3 * phi)


def tertiary_trefoil_x(rho, phi):
    """Zernike tertiary trefoil X."""
    return (10 * rho**3 - 30 * rho**5 + 21 * rho**7) * e.cos(3 * phi)


def quaternary_astigmatism_00(rho, phi):
    """Zernike quaternary astigmatism 0°."""
    return (10 * rho**2 - 30 * rho**4 + 21 * rho**6) * e.cos(2 * phi)


def quaternary_astigmatism_45(rho, phi):
    """Zernike quaternary astigmatism 45°."""
    return (10 * rho**2 - 30 * rho**4 + 21 * rho**6) * e.sin(2 * phi)


def quaternary_coma_y(rho, phi):
    """Zernike quaternary coma Y."""
    return (5 * rho - 60 * rho**3 + 210 * rho**5 - 280 * rho**7 + 126 * rho**9)\
        * e.cos(phi)


def quaternary_coma_x(rho, phi):
    """Zernike quaternary coma X."""
    return (5 * rho - 60 * rho**3 + 210 * rho**5 - 280 * rho**7 + 126 * rho**9)\
        * e.sin(phi)


def quaternary_spherical(rho, phi):
    """Zernike quaternary spherical."""
    return 252 * rho**10 \
        - 630 * rho**8 \
        + 560 * rho**6 \
        - 210 * rho**4 \
        + 30 * rho**2 \
        - 1


def primary_hexafoil_y(rho, phi):
    """Zernike primary hexafoil Y."""
    return rho**6 * e.cos(6 * phi)


def primary_hexafoil_x(rho, phi):
    """Zernike primary hexafoil X."""
    return rho**6 * e.sin(6 * phi)


def secondary_pentafoil_y(rho, phi):
    """Zernike secondary pentafoil Y."""
    return (7 * rho**7 - 6 * rho**5) * e.cos(5 * phi)


def secondary_pentafoil_x(rho, phi):
    """Zernike secondary pentafoil X."""
    return (7 * rho**7 - 6 * rho**5) * e.sin(5 * phi)


def tertiary_tetrafoil_y(rho, phi):
    """Zernike tertiary tetrafoil Y."""
    return (28 * rho**8 - 42 * rho**6 + 15 * rho**4) * e.cos(4 * phi)


def tertiary_tetrafoil_x(rho, phi):
    """Zernike tertiary tetrafoil X."""
    return (28 * rho**8 - 42 * rho**6 + 15 * rho**4) * e.sin(4 * phi)


def quaternary_trefoil_y(rho, phi):
    """Zernike quaternary trefoil Y."""
    return (84 * rho**9 - 168 * rho**7 + 105 * rho**5 - 20 * rho**3) * e.cos(3 * phi)


def quaternary_trefoil_x(rho, phi):
    """Zernike quaternary trefoil X."""
    return (84 * rho**9 - 168 * rho**7 + 105 * rho**5 - 20 * rho**3) * e.sin(3 * phi)


def quinternary_astigmatism_00(rho, phi):
    """Zernike quinternary astigmatism 0°."""
    return (210 * rho**10 - 504 * rho**8 + 420 * rho**6 - 140 * rho**4 + 15 * rho**2) \
        * e.cos(2 * phi)


def quinternary_astigmatism_45(rho, phi):
    """Zernike quinternary astigmatism 45°."""
    return (210 * rho**10 - 504 * rho**8 + 420 * rho**6 - 140 * rho**4 + 15 * rho**2) \
        * e.sin(2 * phi)


def quinternary_coma_y(rho, phi):
    """Zernike quinternary coma Y."""
    return (462 * rho**11 - 1260 * rho**9 + 1260 * rho**7 - 560 * rho**5 + 105 * rho**3 - 6 * rho) \
        * e.cos(phi)


def quinternary_coma_x(rho, phi):
    """Zernike quinternary coma X."""
    return (462 * rho**11 - 1260 * rho**9 + 1260 * rho**7 - 560 * rho**5 + 105 * rho**3 - 6 * rho) \
        * e.sin(phi)


def quinternary_spherical(rho, phi):
    """Zernike quinternary spherical."""
    return 924 * rho**12 \
        - 2772 * rho**10 \
        + 3150 * rho**8 \
        - 1680 * rho**6 \
        + 420 * rho**4 \
        - 42 * rho**2 \
        + 1


def primary_septafoil_y(rho, phi):
    """Zernike primary septafoil."""
    return 4 * rho**7 * e.cos(7 * phi)


def primary_septafoil_x(rho, phi):
    """Zernike primary septafoil."""
    return 4 * rho**7 * e.sin(7 * phi)


# norms
piston.norm = 1
tip.norm = 2
tilt.norm = 2
defocus.norm = e.sqrt(3)
primary_astigmatism_00.norm = e.sqrt(6)
primary_astigmatism_45.norm = e.sqrt(6)
primary_coma_y.norm = 2 * e.sqrt(2)
primary_coma_x.norm = 2 * e.sqrt(2)
primary_spherical.norm = e.sqrt(5)
primary_trefoil_y.norm = 2 * e.sqrt(2)
primary_trefoil_x.norm = 2 * e.sqrt(2)
secondary_astigmatism_00.norm = e.sqrt(10)
secondary_astigmatism_45.norm = e.sqrt(10)
secondary_coma_y.norm = 2 * e.sqrt(3)
secondary_coma_x.norm = 2 * e.sqrt(3)
secondary_spherical.norm = e.sqrt(7)
primary_tetrafoil_y.norm = e.sqrt(10)
primary_tetrafoil_x.norm = e.sqrt(10)
secondary_trefoil_y.norm = 2 * e.sqrt(3)
secondary_trefoil_x.norm = 2 * e.sqrt(3)
tertiary_astigmatism_00.norm = e.sqrt(14)
tertiary_astigmatism_45.norm = e.sqrt(14)
tertiary_coma_y.norm = 4
tertiary_coma_x.norm = 4
tertiary_spherical.norm = 3
primary_pentafoil_y.norm = 2 * e.sqrt(3)
primary_pentafoil_x.norm = 2 * e.sqrt(3)
secondary_tetrafoil_y.norm = e.sqrt(14)
secondary_tetrafoil_x.norm = e.sqrt(14)
tertiary_trefoil_y.norm = 4
tertiary_trefoil_x.norm = 4
quaternary_astigmatism_00.norm = 3 * e.sqrt(2)
quaternary_astigmatism_45.norm = 3 * e.sqrt(2)
quaternary_coma_y.norm = 2 * e.sqrt(5)
quaternary_coma_x.norm = 2 * e.sqrt(5)
quaternary_spherical.norm = e.sqrt(11)
primary_hexafoil_y.norm = e.sqrt(14)
primary_hexafoil_x.norm = e.sqrt(14)
secondary_pentafoil_y.norm = 4
secondary_pentafoil_x.norm = 4
tertiary_tetrafoil_y.norm = 3 * e.sqrt(2)
tertiary_tetrafoil_x.norm = 3 * e.sqrt(2)
quaternary_trefoil_y.norm = 2 * e.sqrt(5)
quaternary_trefoil_x.norm = 2 * e.sqrt(5)
quinternary_astigmatism_00.norm = e.sqrt(22)
quinternary_astigmatism_45.norm = e.sqrt(22)
quinternary_coma_y.norm = 2 * e.sqrt(6)
quinternary_coma_x.norm = 2 * e.sqrt(6)
quinternary_spherical.norm = e.sqrt(13)
primary_septafoil_y.norm = e.sqrt(16)
primary_septafoil_x.norm = e.sqrt(16)

# names
piston.name = 'Piston'
tip.name = 'Tilt Y'
tilt.name = 'Tilt X'
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
tertiary_trefoil_x.name = 'Tertiary Trefoil X'
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
primary_septafoil_y.name = 'Primary Septafoil Y'
primary_septafoil_x.name = 'Primary Septafoil X'

zernikes = [
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
    quinternary_spherical,
    primary_septafoil_y,
    primary_septafoil_x,
]

zernikes_cpu = [jit(zernikes[0])]
for func in zernikes[1:]:
    compfunc = vectorize(func)
    compfunc.name = func.name
    compfunc.norm = func.norm
    zernikes_cpu.append(compfunc)

zernikes_gpu = [fuse(func) for func in zernikes[1:]]  # cupy compiled zernikes
zernikes_gpu.insert(0, zernikes[0])


def change_backend(to):
    """Change the backend between cuda/cupy and CPU."""
    if to == 'cu':
        globals()['zernikes'] = zernikes_gpu
    elif to == 'np':
        globals()['zernikes'] = zernikes_cpu


config.chbackend_observers.append(change_backend)


fringemap = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 8,
    9: 9,
    10: 10,
    11: 11,
    12: 12,
    13: 13,
    14: 14,
    15: 15,
    16: 16,
    17: 17,
    18: 18,
    19: 19,
    20: 20,
    21: 21,
    22: 22,
    23: 23,
    24: 24,
    25: 25,
    26: 26,
    27: 27,
    28: 28,
    29: 29,
    30: 30,
    31: 31,
    32: 32,
    33: 33,
    34: 34,
    35: 35,
    36: 36,
    37: 37,
    38: 38,
    39: 39,
    40: 40,
    41: 41,
    42: 42,
    43: 43,
    44: 44,
    45: 45,
    46: 46,
    47: 47,
    48: 48,
}
nollmap = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 9,
    9: 10,
    10: 8,
    11: 11,
    12: 12,
    13: 16,
    14: 17,
    15: 13,
    16: 14,
    17: 18,
    18: 19,
    19: 25,
    20: 26,
    21: 15,
    22: 20,
    23: 21,
    24: 27,
    25: 28,
    26: 36,
    27: 37,
    28: 22,
    29: 23,
    30: 29,
    31: 30,
    32: 38,
    33: 39,
    34: 49,
    35: 50,
    36: 24,
}
maps = {
    'fringe': fringemap,
    'noll': nollmap,
}


def zernikename(idx, base, map_):
    """Return the name of a Fringe Zernike with the given index and base."""
    return zernikes[map_[idx-base]].name


def zernikes_to_magnitude_angle(coefs, namer):
    """Convert Fringe Zernike polynomial set to a magnitude and phase representation."""
    def mkary():  # default for defaultdict
        return e.zeros(2)

    # make a list of names to go with the coefficients
    names = [namer(i, base=0) for i in range(len(coefs))]
    combinations = defaultdict(mkary)

    # for each name and coefficient, make a len 2 array.  Put the Y or 0 degree values in the first slot
    for coef, name in zip(coefs, names):
        if name.endswith(('X', 'Y', '°')):
            newname = ' '.join(name.split(' ')[:-1])
            if name.endswith('Y'):
                combinations[newname][0] = coef
            elif name.endswith('X'):
                combinations[newname][1] = coef
            elif name[-2] == '5':  # 45 degree case
                combinations[newname][1] = coef
            else:
                combinations[newname][0] = coef
        else:
            combinations[name][0] = coef

    # now go over the combinations and compute the L2 norms and angles
    for name in combinations:
        ovals = combinations[name]
        magnitude = e.sqrt((ovals**2).sum())
        if 'Spheric' in name or 'focus' in name or 'iston' in name:
            phase = 0
        else:
            phase = e.degrees(e.arctan2(*ovals))
        values = (magnitude, phase)
        combinations[name] = values

    return dict(combinations)  # cast to regular dict for return


zernikefuncs = {
    'name': {
        'fringe': partial(zernikename, map_=fringemap),
        'noll': partial(zernikename, map_=nollmap),
    }
}
zernikefuncs.update({
    'magnitude_angle': {
        'fringe': partial(zernikes_to_magnitude_angle, namer=zernikefuncs['name']['fringe']),
        'noll': partial(zernikes_to_magnitude_angle, namer=zernikefuncs['name']['noll']),
    }
})


class ZCache(object):
    """Cache of Zernike terms evaluated over the unit circle."""
    def __init__(self):
        """Create a new FZCache instance."""
        self.normed = defaultdict(dict)
        self.regular = defaultdict(dict)

    def get_zernike(self, number, norm, samples):
        """Get an array of phase values for a given index, norm, and number of samples."""
        if norm is True:
            target = self.normed
        else:
            target = self.regular

        try:
            zern = target[samples][number]
        except KeyError:
            rho, phi = make_rho_phi_grid(samples, aligned='y')
            func = zernikes[number]
            zern = func(rho, phi)
            if norm is True:
                zern *= func.norm

            target[samples][number] = zern.copy()

        return zern

    def __call__(self, number, norm, samples):
        """Get an array of phase values for a given index, norm, and number of samples."""
        return self.get_zernike(number, norm, samples)

    def clear(self, *args):
        """Empty the cache."""
        self.normed = defaultdict(dict)
        self.regular = defaultdict(dict)


zcache = ZCache()
config.chbackend_observers.append(zcache.clear)


class BaseZernike(Pupil):
    """Basic class implementing Zernike features."""
    _map = None
    _namer = None
    _magnituder = None
    _cache = None
    _name = None

    def __init__(self, *args, **kwargs):
        """Initialize a new Zernike instance."""
        if args is not None:
            if len(args) == 0:
                self.coefs = e.zeros(len(self._map), dtype=config.precision)
            else:
                self.coefs = e.asarray([*args[0]], dtype=config.precision)

        self.normalize = False
        pass_args = {}

        self.base = config.zernike_base
        try:
            bb = kwargs['base']
            if bb > 1:
                raise ValueError('It violates convention to use a base greater than 1.')
            elif bb < 0:
                raise ValueError('It is nonsensical to use a negative base.')
            self.base = bb
        except KeyError:
            # user did not specify base
            pass

        if kwargs is not None:
            for key, value in kwargs.items():
                if key[0].lower() == 'z':
                    idx = int(key[1:])  # strip 'Z' from index
                    self.coefs[idx - self.base] = value
                elif key.lower() == 'norm':
                    self.normalize = value
                elif key.lower() == 'base':
                    self.base = value
                else:
                    pass_args[key] = value

        super().__init__(**pass_args)

    def build(self):
        """Use the wavefront coefficients stored in this class instance to build a wavefront model.

        Returns
        -------
        self.phase : `numpy.ndarray`
            arrays containing the phase associated with the pupil
        self.fcn : `numpy.ndarray`
            array containing the wavefunction of the pupil plane

        """
        # build a coordinate system over which to evaluate this function
        self.phase = e.zeros((self.samples, self.samples), dtype=config.precision)
        for term, coef in enumerate(self.coefs):
            # short circuit for speed
            if coef == 0:
                continue
            else:
                idx = self._map[term]
                self.phase += coef * self._cache(idx, self.normalize, self.samples)  # NOQA

        return self

    def top_n(self, n=5):
        """Identify the top n terms in the wavefront.

        Parameters
        ----------
        n : `int`, optional
            identify the top n terms.

        Returns
        -------
        `list`
            list of tuples (magnitude, index, term)

        """
        coefs = e.asarray(self.coefs)
        coefs_work = abs(coefs)
        oidxs = e.arange(len(coefs), dtype=int) + self.base  # "original indexes"
        idxs = e.argpartition(coefs_work, -n)[-n:]  # argpartition does some magic to identify the top n (unsorted)
        idxs = idxs[e.argsort(coefs_work[idxs])[::-1]]  # use argsort to sort them in ascending order and reverse
        big_terms = coefs[idxs]  # finally, take the values from the
        big_idxs = oidxs[idxs]
        names = e.asarray(self.names, dtype=str)[big_idxs - self.base]
        return list(zip(big_terms, big_idxs, names))

    @property
    def magnitudes(self):
        """Return the magnitude and angles of the zernike components in this wavefront."""
        # need to call through class variable to avoid insertion of self as arg
        return self.__class__._magnituder(self.coefs)  # NOQA

    @property
    def names(self):
        """Names of the terms in self."""
        # need to call through class variable to avoid insertion of self as arg
        idxs = e.asarray(range(len(self.coefs))) + self.base
        return [self.__class__._namer(i, base=self.base) for i in idxs]  # NOQA

    def barplot(self, orientation='h', buffer=1, zorder=3, number=True, offset=0, width=0.8, fig=None, ax=None):
        """Create a barplot of coefficients and their names.

        Parameters
        ----------
        orientation : `str`, {'h', 'v', 'horizontal', 'vertical'}
            orientation of the plot
        buffer : `float`, optional
            buffer to use around the left and right (or top and bottom) bars
        zorder : `int`, optional
            zorder of the bars.  Use zorder > 3 to put bars in front of gridlines
        number : `bool`, optional
            if True, plot numbers along the y=0 line showing indices
        offset : `float`, optional
            offset to apply to bars, useful for before/after Zernike breakdowns
        width : `float`, optional
            width of bars, useful for before/after Zernike breakdowns
        fig : `matplotlib.figure.Figure`
            Figure containing the plot
        ax : `matplotlib.axes.Axis`
            Axis containing the plot

        Returns
        -------
        fig : `matplotlib.figure.Figure`
            Figure containing the plot
        ax : `matplotlib.axes.Axis`
            Axis containing the plot

        """
        from matplotlib import pyplot as plt
        fig, ax = share_fig_ax(fig, ax)

        coefs = e.asarray(self.coefs)
        idxs = e.asarray(range(len(coefs))) + self.base
        names = self.names
        lab = f'{self.zaxis_label} [{self.phase_unit}]'
        lims = (idxs[0] - buffer, idxs[-1] + buffer)
        if orientation.lower() in ('h', 'horizontal'):
            vmin, vmax = coefs.min(), coefs.max()
            drange = vmax - vmin
            offsetY = drange * 0.01

            ax.bar(idxs + offset, self.coefs, zorder=zorder, width=width)
            plt.xticks(idxs, names, rotation=90)
            if number:
                for i in idxs:
                    ax.text(i, offsetY, str(i), ha='center')
            ax.set(ylabel=lab, xlim=lims)
        else:
            ax.barh(idxs + offset, self.coefs, zorder=zorder, height=width)
            plt.yticks(idxs, names)
            if number:
                for i in idxs:
                    ax.text(0, i, str(i), ha='center')
            ax.set(xlabel=lab, ylim=lims)
        return fig, ax

    def barplot_magnitudes(self, orientation='h', sort=False,
                           buffer=1, zorder=3, offset=0, width=0.8,
                           fig=None, ax=None):
        """Create a barplot of magnitudes of coefficient pairs and their names.

        E.g., astigmatism will get one bar.

        Parameters
        ----------
        orientation : `str`, {'h', 'v', 'horizontal', 'vertical'}
            orientation of the plot
        sort : `bool`, optional
            whether to sort the zernikes in descending order
        buffer : `float`, optional
            buffer to use around the left and right (or top and bottom) bars
        zorder : `int`, optional
            zorder of the bars.  Use zorder > 3 to put bars in front of gridlines
        offset : `float`, optional
            offset to apply to bars, useful for before/after Zernike breakdowns
        width : `float`, optional
            width of bars, useful for before/after Zernike breakdowns
        fig : `matplotlib.figure.Figure`
            Figure containing the plot
        ax : `matplotlib.axes.Axis`
            Axis containing the plot

        Returns
        -------
        fig : `matplotlib.figure.Figure`
            Figure containing the plot
        ax : `matplotlib.axes.Axis`
            Axis containing the plot

        """
        from matplotlib import pyplot as plt

        magang = self.magnitudes
        mags = [m[0] for m in magang.values()]
        names = magang.keys()
        idxs = e.asarray(list(range(len(names))))

        if sort:
            mags, names = sort_xy(mags, names)
            mags = list(reversed(mags))
            names = list(reversed(names))
        lab = f'{self.zaxis_label} [{self.phase_unit}]'
        lims = (idxs[0] - buffer, idxs[-1] + buffer)
        fig, ax = share_fig_ax(fig, ax)
        if orientation.lower() in ('h', 'horizontal'):
            ax.bar(idxs + offset, mags, zorder=zorder, width=width)
            plt.xticks(idxs, names, rotation=90)
            ax.set(ylabel=lab, xlim=lims)
        else:
            ax.barh(idxs + offset, mags, zorder=zorder, height=width)
            plt.yticks(idxs, names)
            ax.set(xlabel=lab, ylim=lims)
        return fig, ax

    def barplot_topn(self, n=5, orientation='h', buffer=1, zorder=3, fig=None, ax=None):
        """Plot the top n terms in the wavefront.

        Parameters
        ----------
        n : `int`, optional
            plot the top n terms.
        orientation : `str`, {'h', 'v', 'horizontal', 'vertical'}
            orientation of the plot
        buffer : `float`, optional
            buffer to use around the left and right (or top and bottom) bars
        zorder : `int`, optional
            zorder of the bars.  Use zorder > 3 to put bars in front of gridlines
        fig : `matplotlib.figure.Figure`
            Figure containing the plot
        ax : `matplotlib.axes.Axis`
            Axis containing the plot

        Returns
        -------
        fig : `matplotlib.figure.Figure`
            Figure containing the plot
        ax : `matplotlib.axes.Axis`
            Axis containing the plot

        """
        from matplotlib import pyplot as plt

        topn = self.top_n(n)
        magnitudes = [n[0] for n in topn]
        names = [n[2] for n in topn]
        idxs = range(len(names))

        fig, ax = share_fig_ax(fig, ax)

        lab = f'{self.zaxis_label} [{self.phase_unit}]'
        lims = (idxs[0] - buffer, idxs[-1] + buffer)
        if orientation.lower() in ('h', 'horizontal'):
            ax.bar(idxs, magnitudes, zorder=zorder)
            plt.xticks(idxs, names, rotation=90)
            ax.set(ylabel=lab, xlim=lims)
        else:
            ax.barh(idxs, magnitudes, zorder=zorder)
            plt.yticks(idxs, names)
            ax.set(xlabel=lab, ylim=lims)
        return fig, ax

    def truncate(self, n):
        """Truncate the wavefront to the first n terms.

        Parameters
        ----------
        n : `int`
            number of terms to keep.

        Returns
        -------
        `self`
            modified FringeZernike instance.

        """
        if n > len(self.coefs):
            return self
        else:
            self.coefs = self.coefs[:n]
            self.build()
            self.mask(self._mask, self.mask_target)
            return self

    def truncate_topn(self, n):
        """Truncate the pupil to only the top n terms.

        Parameters
        ----------
        n : `int`
            number of parameters to keep

        Returns
        -------
        `self`
            modified FringeZernike instance.

        """
        topn = self.top_n(n)
        new_coefs = e.zeros(len(self.coefs), dtype=config.precision)
        for coef in topn:
            mag, index, *_ = coef
            new_coefs[index-self.base] = mag

        self.coefs = new_coefs
        self.build()
        self.mask(self._mask, self.mask_target)
        return self

    def __str__(self):
        """Pretty-print pupil description."""
        if self.normalize is True:
            header = f'rms normalized {self._name} Zernike description with:\n\t'
        else:
            header = f'{self._name} Zernike description with:\n\t'

        strs = []
        for number, coef in enumerate(self.coefs):
            # skip 0 terms
            if coef == 0:
                continue

            # positive coefficient, prepend with +
            if e.sign(coef) == 1:
                _ = '+' + f'{coef:.3f}'
            # negative, sign comes from the value
            else:
                _ = f'{coef:.3f}'

            # create the name
            idx = self._map[number]
            name = f'Z{number+self.base} - {zernikes[idx].name}'

            strs.append(' '.join([_, name]))
        body = '\n\t'.join(strs)

        footer = f'\n\t{self.pv:.3f} PV, {self.rms:.3f} RMS [{self.phase_unit}]'
        return f'{header}{body}{footer}'


class FringeZernike(BaseZernike):
    """Fringe Zernike description of an optical pupil."""
    _map = fringemap
    _cache = zcache
    _magnituder = zernikefuncs['magnitude_angle']['fringe']
    _namer = zernikefuncs['name']['fringe']
    _name = 'Fringe'


class NollZernike(BaseZernike):
    """Noll Zernike deswcription of an optical pupil."""
    _map = nollmap
    _cache = zcache
    _magnituder = zernikefuncs['magnitude_angle']['noll']
    _namer = zernikefuncs['name']['noll']
    _name = 'Noll'


def zernikefit(data, x=None, y=None,
               rho=None, phi=None, terms=16,
               norm=False, residual=False,
               round_at=6, map_='fringe'):
    """Fits a number of Zernike coefficients to provided data.

    Works by minimizing the mean square error  between each coefficient and the
    given data.  The data should be uniformly sampled in an x,y grid.

    Parameters
    ----------
    data : `numpy.ndarray`
        data to fit to.
    x : `numpy.ndarray`, optional
        x coordinates, same shape as data
    y : `numpy.ndarray`, optional
        y coordinates, same shape as data
    rho : `numpy.ndarray`, optional
        radial coordinates, same shape as data
    phi : `numpy.ndarray`, optional
        azimuthal coordinates, same shape as data
    terms : `int`, optional
        number of terms to fit, fits terms 0~terms
    norm : `bool`, optional
        if True, normalize coefficients to unit RMS value
    residual : `bool`, optional
        if True, return a tuple of (coefficients, residual)
    round_at : `int`, optional
        decimal place to round values at.
    map_ : `str`, optional, {'fringe', 'noll'}
        which ordering of Zernikes to use

    Returns
    -------
    coefficients : `numpy.ndarray`
        an array of coefficients matching the input data.
    residual : `float`
        RMS error between the input data and the fit.

    Raises
    ------
    ValueError
        too many terms requested.

    """
    map_ = maps[map_]
    if terms > len(fringemap):
        raise ValueError(f'number of terms must be less than {len(fringemap)}')

    data = data.T  # transpose to mimic transpose of zernikes

    # precompute the valid indexes in the original data
    pts = e.isfinite(data)

    if x is None and rho is None:
        # set up an x/y rho/phi grid to evaluate Zernikes on
        rho, phi = make_rho_phi_grid(*reversed(data.shape))
        rho = rho[pts].flatten()
        phi = phi[pts].flatten()
    elif rho is None:
        rho, phi = cart_to_polar(x, y)
        rho, phi = rho[pts].flatten(), phi[pts].flatten()

    # compute each Zernike term
    zerns_raw = []
    for i in range(terms):
        func = zernikes[map_[i]]
        base_zern = func(rho, phi)
        if norm:
            base_zern *= func.norm
        zerns_raw.append(base_zern)
    zerns = e.asarray(zerns_raw).T

    # use least squares to compute the coefficients
    meas_pts = data[pts].flatten()
    coefs = e.linalg.lstsq(zerns, meas_pts, rcond=None)[0]
    if round_at is not None:
        coefs = coefs.round(round_at)

    if residual is True:
        components = []
        for zern, coef in zip(zerns_raw, coefs):
            components.append(coef * zern)

        _fit = e.asarray(components)
        _fit = _fit.sum(axis=0)
        rmserr = rms(data[pts].flatten() - _fit)
        return coefs, rmserr
    else:
        return coefs
