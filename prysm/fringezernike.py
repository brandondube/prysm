''' A repository of fringe Zernike aberration descriptions used to model pupils of optical systems.
'''
from collections import defaultdict

from .conf import config
from .pupil import Pupil
from .coordinates import make_rho_phi_grid

from prysm import mathops as m

from prysm import _zernike as z


zernmap = {
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


class FZCache(object):
    def __init__(self):
        self.normed = defaultdict(dict)
        self.regular = defaultdict(dict)

    def get_zernike(self, number, norm, samples):
        if norm is True:
            target = self.normed
        else:
            target = self.regular

        try:
            zern = target[samples][number]
        except KeyError:
            rho, phi = make_rho_phi_grid(samples, aligned='y')
            func = z.zernikes[zernmap[number]]
            zern = func(rho, phi)
            if norm is True:
                zern *= func.__wrapped__.norm

            target[samples][number] = zern.copy()

        return zern

    def clear(self, *args):
        self.normed = defaultdict(dict)
        self.regular = defaultdict(dict)


class FringeZernike(Pupil):
    ''' Fringe Zernike pupil description.

    Properties:
        Inherited from :class:`Pupil`, please see that class.

    Instance Methods:
        build: computes the phase and wavefunction for the pupil.  This method
            is automatically called by the constructor, and does not regularly
            need to be changed by the user.

    Private Instance Methods:
        none

    Static Methods:
        none

    '''
    def __init__(self, *args, **kwargs):
        ''' Creates a FringeZernike Pupil object.

        Args:
            samples (`int`): number of samples across pupil diameter.

            wavelength (`float`): wavelength of light, in um.

            epd: (`float`): diameter of the pupil, in mm.

            opd_unit (`string`): unit OPD is expressed in.  One of:
                ($\lambda$, waves, $\mu m$, microns, um, nm , nanometers).

            base (`int`): 0 or 1, adjusts the base index of the polynomial
                expansion.

            Zx (`float`): Zth fringe Zernike coefficient, in range [0,48] or [1,49] if base 1

        Returns:
            FringeZernike.  A new :class:`FringeZernike` pupil instance.

        Notes:
            Supports multiple syntaxes:
                - args: pass coefficients as a list, including terms up to
                        the highest given Z-index.
                        e.g. passing [1,2,3] will be interpreted as Z0=1, Z1=2, Z3=3.

                - kwargs: pass a named set of Zernike terms.
                          e.g. FringeZernike(Z0=1, Z1=2, Z2=3)

            Supports normalization and unit conversion, can pass kwargs:
                - rms_norm=True: coefficients have unit rms value

                - opd_unit='nm': coefficients are expressed in units of nm

            The kwargs syntax overrides the args syntax.

        '''

        if args is not None:
            if len(args) is 0:
                self.coefs = [0] * len(zernmap)
            else:
                self.coefs = [*args[0]]

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
                elif key in ('rms_norm'):
                    self.normalize = True
                elif key.lower() == 'base':
                    self.base = value
                else:
                    pass_args[key] = value

        super().__init__(**pass_args)

    def build(self):
        '''Uses the wavefront coefficients stored in this class instance to
            build a wavefront model.

        Args:
            none

        Returns:
            (numpy.ndarray, numpy.ndarray) arrays containing the phase, and the
                wavefunction for the pupil.

        '''
        # build a coordinate system over which to evaluate this function
        self.phase = m.zeros((self.samples, self.samples), dtype=config.precision)
        for term, coef in enumerate(self.coefs):
            # short circuit for speed
            if coef == 0:
                continue
            self.phase += coef * zcache.get_zernike(term, self.normalize, self.samples)

        self._phase_to_wavefunction()
        return self.phase, self.fcn

    def __repr__(self):
        ''' Pretty-print pupil description.
        '''
        if self.normalize is True:
            header = 'rms normalized Fringe Zernike description with:\n\t'
        else:
            header = 'Fringe Zernike description with:\n\t'

        strs = []
        for number, (coef, func) in enumerate(zip(self.coefs, z.zernikes)):
            print(func)
            # skip 0 terms
            if coef == 0:
                continue

            # positive coefficient, prepend with +
            if m.sign(coef) == 1:
                _ = '+' + f'{coef:.3f}'
            # negative, sign comes from the value
            else:
                _ = f'{coef:.3f}'

            # create the name
            name = f'Z{number+self.base} - {func.name}'

            strs.append(' '.join([_, name]))
        body = '\n\t'.join(strs)

        footer = f'\n\t{self.pv:.3f} PV, {self.rms:.3f} RMS'
        return f'{header}{body}{footer}'


def fit(data, num_terms=16, rms_norm=False, round_at=6):
    ''' Fits a number of Zernike coefficients to provided data by minimizing
        the root sum square between each coefficient and the given data.  The
        data should be uniformly sampled in an x,y grid.

    Args:

        data (`numpy.ndarray`): data to fit to.

        num_terms (`int`): number of terms to fit, fits terms 0~num_terms.

        rms_norm (`bool`): if true, normalize coefficients to unit RMS value.

        round_at (`int`): decimal place to round values at.

    Returns:
        numpy.ndarray: an array of coefficients matching the input data.

    '''
    if num_terms > len(zernmap):
        raise ValueError(f'number of terms must be less than {len(zernmap)}')

    # precompute the valid indexes in the original data
    pts = m.isfinite(data)

    # set up an x/y rho/phi grid to evaluate Zernikes on
    x, y = m.linspace(-1, 1, data.shape[1]), m.linspace(-1, 1, data.shape[0])
    xx, yy = m.meshgrid(x, y)
    rho = m.sqrt(xx**2 + yy**2)[pts].flatten()
    phi = m.arctan2(xx, yy)[pts].flatten()

    # compute each Zernike term
    zernikes = []
    for i in range(num_terms):
        func = z.zernikes[zernmap[i]]
        base_zern = func(rho, phi)
        if rms_norm:
            base_zern *= func.__wrapped__.norm
        zernikes.append(base_zern)
    zerns = m.asarray(zernikes).T

    # use least squares to compute the coefficients
    coefs = m.lstsq(zerns, data[pts].flatten(), rcond=None)[0]
    return coefs.round(round_at)


zcache = FZCache()
config.chbackend_observers.append(zcache.clear)
