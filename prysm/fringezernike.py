'''A repository of fringe Zernike aberration descriptions used to model pupils of optical systems.'''
from collections import defaultdict

from .conf import config
from .pupil import Pupil
from .coordinates import make_rho_phi_grid, cart_to_polar
from .util import rms

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
                zern *= func.norm

            target[samples][number] = zern.copy()

        return zern

    def __call__(self, number, norm, samples):
        return self.get_zernike(number, norm, samples)

    def clear(self, *args):
        self.normed = defaultdict(dict)
        self.regular = defaultdict(dict)


class FringeZernike(Pupil):
    def __init__(self, *args, **kwargs):
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

        Returns
        -------
        self.phase : `numpy.ndarray`
            arrays containing the phase associated with the pupil
        self.fcn : `numpy.ndarray`
            array containing the wavefunction of the pupil plane

        '''
        # build a coordinate system over which to evaluate this function
        self.phase = m.zeros((self.samples, self.samples), dtype=config.precision)
        for term, coef in enumerate(self.coefs):
            # short circuit for speed
            if coef == 0:
                continue
            self.phase += coef * zcache(term, self.normalize, self.samples)

        return self

    def __repr__(self):
        '''Pretty-print pupil description.'''
        if self.normalize is True:
            header = 'rms normalized Fringe Zernike description with:\n\t'
        else:
            header = 'Fringe Zernike description with:\n\t'

        strs = []
        for number, (coef, func) in enumerate(zip(self.coefs, z.zernikes)):
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


def fit(data, x=None, y=None, rho=None, phi=None, num_terms=16, rms_norm=False, residual=False, round_at=6):
    '''Fits a number of Zernike coefficients to provided data by minimizing
        the root sum square between each coefficient and the given data.  The
        data should be uniformly sampled in an x,y grid.

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
        azimuthal
    num_terms : `int`, optional
        number of terms to fit, fits terms 0~num_terms
    rms_norm : `bool`, optional
        if True, normalize coefficients to unit RMS value
    residual : `bool`, optional
        if True, return a tuple of (coefficients, residual)
    round_at : `int`
        decimal place to round values at.

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

    '''
    if num_terms > len(zernmap):
        raise ValueError(f'number of terms must be less than {len(zernmap)}')

    # precompute the valid indexes in the original data
    pts = m.isfinite(data)

    if x is None and rho is None:
        # set up an x/y rho/phi grid to evaluate Zernikes on
        x, y = m.linspace(-1, 1, data.shape[1]), m.linspace(-1, 1, data.shape[0])
        xx, yy = m.meshgrid(x, y)
        rho, phi = cart_to_polar(xx, yy)
        rho = rho.flatten()
        phi = phi.flatten()
    elif rho is None:
        rho, phi = cart_to_polar(x, y)
        rho, phi = rho.flatten(), phi.flatten()

    # compute each Zernike term
    zernikes = []
    for i in range(num_terms):
        func = z.zernikes[zernmap[i]]
        base_zern = func(rho, phi)
        if rms_norm:
            base_zern *= func.norm
        zernikes.append(base_zern)
    zerns = m.asarray(zernikes).T

    # use least squares to compute the coefficients
    meas_pts = data[pts].flatten()
    coefs = m.lstsq(zerns, meas_pts, rcond=None)[0]
    if round_at is not None:
        coefs = coefs.round(round_at)

    if residual is True:
        components = []
        for zern, coef in zip(zernikes, coefs):
            components.append(coef * zern)

        _fit = m.asarray(components)
        _fit = _fit.sum(axis=0)
        rmserr = rms(data - _fit)
        return coefs, rmserr
    else:
        return coefs


zcache = FZCache()
config.chbackend_observers.append(zcache.clear)
