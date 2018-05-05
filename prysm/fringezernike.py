''' A repository of fringe Zernike aberration descriptions used to model pupils of optical systems.
'''
from collections import defaultdict

import numpy as np

from .conf import config
from .pupil import Pupil
from .coordinates import make_rho_phi_grid

from prysm import mathops as m


@m.jit
def Z0(rho, phi):
    return np.ones(rho.shape)


@m.vectorize
def Z1(rho, phi):
    return rho * m.cos(phi)


@m.vectorize
def Z2(rho, phi):
    return rho * m.sin(phi)


@m.vectorize
def Z3(rho, phi):
    return 2 * rho**2 - 1


@m.vectorize
def Z4(rho, phi):
    return rho**2 * m.cos(2 * phi)


@m.vectorize
def Z5(rho, phi):
    return rho**2 * m.sin(2 * phi)


@m.vectorize
def Z6(rho, phi):
    return (-2 * rho + 3 * rho**3) * m.cos(phi)


@m.vectorize
def Z7(rho, phi):
    return (-2 * rho + 3 * rho**3) * m.sin(phi)


@m.vectorize
def Z8(rho, phi):
    return 6 * rho**4 - 6 * rho**2 + 1


@m.vectorize
def Z9(rho, phi):
    return rho**3 * m.cos(3 * phi)


@m.vectorize
def Z10(rho, phi):
    return rho**3 * m.sin(3 * phi)


@m.vectorize
def Z11(rho, phi):
    return (-3 * rho**2 + 4 * rho**4) * m.cos(2 * phi)


@m.vectorize
def Z12(rho, phi):
    return (-3 * rho**2 + 4 * rho**4) * m.sin(2 * phi)


@m.vectorize
def Z13(rho, phi):
    return (3 * rho - 12 * rho**3 + 10 * rho**5) * m.cos(phi)


@m.vectorize
def Z14(rho, phi):
    return (3 * rho - 12 * rho**3 + 10 * rho**5) * m.sin(phi)


@m.vectorize
def Z15(rho, phi):
    return 20 * rho**6 + - 30 * rho**4 + 12 * rho**2 - 1


@m.vectorize
def Z16(rho, phi):
    return rho**4 * m.cos(4 * phi)


@m.vectorize
def Z17(rho, phi):
    return rho**4 * m.sin(4 * phi)


@m.vectorize
def Z18(rho, phi):
    return (5 * rho**5 - 4 * rho**3) * m.cos(3 * phi)


@m.vectorize
def Z19(rho, phi):
    return (5 * rho**5 - 4 * rho**3) * m.sin(3 * phi)


@m.vectorize
def Z20(rho, phi):
    return (6 * rho**2 - 20 * rho**4 + 15 * rho**6) * m.cos(2 * phi)


@m.vectorize
def Z21(rho, phi):
    return (6 * rho**2 - 20 * rho**4 + 15 * rho**6) * m.sin(2 * phi)


@m.vectorize
def Z22(rho, phi):
    return (-4 * rho + 30 * rho**3 - 60 * rho**5 + 35 * rho**7) * m.cos(phi)


@m.vectorize
def Z23(rho, phi):
    return (-4 * rho + 30 * rho**3 - 60 * rho**5 + 35 * rho**7) * m.sin(phi)


@m.vectorize
def Z24(rho, phi):
    return 70 * rho**8 - 140 * rho**6 + 90 * rho**4 - 20 * rho**2 + 1


@m.vectorize
def Z25(rho, phi):
    return rho**5 * m.cos(5 * phi)


@m.vectorize
def Z26(rho, phi):
    return rho**5 * m.sin(5 * phi)


@m.vectorize
def Z27(rho, phi):
    return (6 * rho**6 - 5 * rho**4) * m.cos(4 * phi)


@m.vectorize
def Z28(rho, phi):
    return (6 * rho**6 - 5 * rho**4) * m.sin(4 * phi)


@m.vectorize
def Z29(rho, phi):
    return (10 * rho**3 - 30 * rho**5 + 21 * rho**7) * m.cos(3 * phi)


@m.vectorize
def Z30(rho, phi):
    return (10 * rho**3 - 30 * rho**5 + 21 * rho**7) * m.cos(3 * phi)


@m.vectorize
def Z31(rho, phi):
    return (10 * rho**2 - 30 * rho**4 + 21 * rho**6) * m.cos(2 * phi)


@m.vectorize
def Z32(rho, phi):
    return (10 * rho**2 - 30 * rho**4 + 21 * rho**6) * m.sin(2 * phi)


@m.vectorize
def Z33(rho, phi):
    return (5 * rho - 60 * rho**3 + 210 * rho**5 - 280 * rho**7 + 126 * rho**9)\
        * m.cos(phi)


@m.vectorize
def Z34(rho, phi):
    return (5 * rho - 60 * rho**3 + 210 * rho**5 - 280 * rho**7 + 126 * rho**9)\
        * m.sin(phi)


@m.vectorize
def Z35(rho, phi):
    return 252 * rho**10 \
        - 630 * rho**8 \
        + 560 * rho**6 \
        - 210 * rho**4 \
        + 30 * rho**2 \
        - 1


@m.vectorize
def Z36(rho, phi):
    return rho**6 * m.cos(6 * phi)


@m.vectorize
def Z37(rho, phi):
    return rho**6 * m.sin(6 * phi)


@m.vectorize
def Z38(rho, phi):
    return (7 * rho**7 - 6 * rho**5) * m.cos(5 * phi)


@m.vectorize
def Z39(rho, phi):
    return (7 * rho**7 - 6 * rho**5) * m.sin(5 * phi)


@m.vectorize
def Z40(rho, phi):
    return (28 * rho**8 - 42 * rho**6 + 15 * rho**4) * m.cos(4 * phi)


@m.vectorize
def Z41(rho, phi):
    return (28 * rho**8 - 42 * rho**6 + 15 * rho**4) * m.sin(4 * phi)


@m.vectorize
def Z42(rho, phi):
    return (84 * rho**9 - 168 * rho**7 + 105 * rho**5 - 20 * rho**3) * m.cos(3 * phi)


@m.vectorize
def Z43(rho, phi):
    return (84 * rho**9 - 168 * rho**7 + 105 * rho**5 - 20 * rho**3) * m.sin(3 * phi)


@m.vectorize
def Z44(rho, phi):
    return (210 * rho**10 - 504 * rho**8 + 420 * rho**6 - 140 * rho**4 + 15 * rho**2) \
        * m.cos(2 * phi)


@m.vectorize
def Z45(rho, phi):
    return (210 * rho**10 - 504 * rho**8 + 420 * rho**6 - 140 * rho**4 + 15 * rho**2) \
        * m.sin(2 * phi)


@m.vectorize
def Z46(rho, phi):
    return (462 * rho**11 - 1260 * rho**9 + 1260 * rho**7 - 560 * rho**5 + 105 * rho**3 - 6 * rho) \
        * m.cos(phi)


@m.vectorize
def Z47(rho, phi):
    return (462 * rho**11 - 1260 * rho**9 + 1260 * rho**7 - 560 * rho**5 + 105 * rho**3 - 6 * rho) \
        * m.sin(phi)


@m.vectorize
def Z48(rho, phi):
    return 924 * rho**12 \
        - 2772 * rho**10 \
        + 3150 * rho**8 \
        - 1680 * rho**6 \
        + 420 * rho**4 \
        - 42 * rho**2 \
        + 1


zernfcns = {
    0: Z0,
    1: Z1,
    2: Z2,
    3: Z3,
    4: Z4,
    5: Z5,
    6: Z6,
    7: Z7,
    8: Z8,
    9: Z9,
    10: Z10,
    11: Z11,
    12: Z12,
    13: Z13,
    14: Z14,
    15: Z15,
    16: Z16,
    17: Z17,
    18: Z18,
    19: Z19,
    20: Z20,
    21: Z21,
    22: Z22,
    23: Z23,
    24: Z24,
    25: Z25,
    26: Z26,
    27: Z27,
    28: Z28,
    29: Z29,
    30: Z30,
    31: Z31,
    32: Z32,
    33: Z33,
    34: Z34,
    35: Z35,
    36: Z36,
    37: Z37,
    38: Z38,
    39: Z39,
    40: Z40,
    41: Z41,
    42: Z42,
    43: Z43,
    44: Z44,
    45: Z45,
    46: Z46,
    47: Z47,
    48: Z48,
}


# See JCW - http://wp.optics.arizona.edu/jcwyant/wp-content/uploads/sites/13/2016/08/ZernikePolynomialsForTheWeb.pdf
_names = (
    'Z0  - Piston / Bias',
    'Z1  - Tilt Y',
    'Z2  - Tilt X',
    'Z3  - Defocus / Power',
    'Z4  - Primary Astigmatism 00deg',
    'Z5  - Primary Astigmatism 45deg',
    'Z6  - Primary Coma Y',
    'Z7  - Primary Coma X',
    'Z8  - Primary Spherical',
    'Z9  - Primary Trefoil Y',
    'Z10 - Primary Trefoil X',
    'Z11 - Secondary Astigmatism 00deg',
    'Z12 - Secondary Astigmatism 45deg',
    'Z13 - Secondary Coma Y',
    'Z14 - Secondary Coma X',
    'Z15 - Secondary Spherical',
    'Z16 - Primary Tetrafoil Y',
    'Z17 - Primary Tetrafoil X',
    'Z18 - Secondary Trefoil Y',
    'Z19 - Secondary Trefoil X',
    'Z20 - Tertiary Astigmatism 00deg',
    'Z21 - Tertiary Astigmatism 45deg',
    'Z22 - Tertiary Coma Y',
    'Z23 - Tertiary Coma X',
    'Z24 - Tertiary Spherical',
    'Z25 - Pentafoil Y',
    'Z26 - Pentafoil X',
    'Z27 - Secondary Tetrafoil Y',
    'Z28 - Secondary Tetrafoil X',
    'Z29 - Tertiary Trefoil Y',
    'Z30 - Tertiary Trefoil X',
    'Z31 - Quarternary Astigmatism 00deg',
    'Z32 - Quarternary Astigmatism 45deg',
    'Z33 - Quarternary Coma Y',
    'Z34 - Quarternary Coma X',
    'Z35 - Quarternary Spherical',
    'Z36 - Primary Hexafoil Y',
    'Z37 - Primary Hexafoil X',
    'Z38 - Secondary Pentafoil Y',
    'Z39 - Secondary Pentafoil X',
    'Z40 - Tertiary Tetrafoil Y',
    'Z41 - Tertiary Tetrafoil X',
    'Z42 - Quaternary Trefoil Y',
    'Z43 - Quaternary Trefoil X',
    'Z44 - Quinternary Astigmatism 00deg',
    'Z45 - Quinternary Astigmatism 45deg',
    'Z46 - Quinternary Coma Y',
    'Z47 - Quinternary Coma X',
    'Z48 - Quarternary Spherical',
)

_normalizations = (
    1,            # Z 0
    2,            # Z 1
    2,            # Z 2
    m.sqrt(3),      # Z 3
    m.sqrt(6),      # Z 4
    m.sqrt(6),      # Z 5
    2 * m.sqrt(2),  # Z 6
    2 * m.sqrt(2),  # Z 7
    m.sqrt(5),      # Z 8
    2 * m.sqrt(2),  # Z 9
    2 * m.sqrt(2),  # Z10
    m.sqrt(10),     # Z11
    m.sqrt(10),     # Z12
    2 * m.sqrt(3),  # Z13
    2 * m.sqrt(3),  # Z14
    m.sqrt(7),      # Z15
    m.sqrt(10),     # Z16
    m.sqrt(10),     # Z17
    2 * m.sqrt(3),  # Z18
    2 * m.sqrt(3),  # Z19
    m.sqrt(14),     # Z20
    m.sqrt(14),     # Z21
    4,            # Z22
    4,            # Z23
    3,            # Z24
    2 * m.sqrt(3),  # Z25
    2 * m.sqrt(3),  # Z26
    m.sqrt(14),     # Z27
    m.sqrt(14),     # Z28
    4,            # Z29
    4,            # Z30
    3 * m.sqrt(2),  # Z31
    3 * m.sqrt(2),  # Z32
    2 * m.sqrt(5),  # Z33
    2 * m.sqrt(5),  # Z34
    m.sqrt(11),     # Z35
    m.sqrt(14),     # Z36
    m.sqrt(14),     # Z37
    4,            # Z38
    4,            # Z39
    3 * m.sqrt(2),  # Z40
    3 * m.sqrt(2),  # Z41
    2 * m.sqrt(5),  # Z42
    2 * m.sqrt(5),  # Z43
    m.sqrt(22),     # Z44
    m.sqrt(22),     # Z45
    2 * m.sqrt(6),  # Z46
    2 * m.sqrt(6),  # Z47
    m.sqrt(13),     # Z48
)


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
            zern = zernfcns[number](rho, phi)
            if norm is True:
                zern *= _normalizations[number]

            target[samples][number] = zern.copy()

        return zern


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
                self.coefs = [0] * len(zernfcns)
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
        self.phase = np.zeros((self.samples, self.samples), dtype=config.precision)
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
        for number, (coef, name) in enumerate(zip(self.coefs, _names)):
            # skip 0 terms
            if coef == 0:
                continue

            # positive coefficient, prepend with +
            if np.sign(coef) == 1:
                _ = '+' + f'{coef:.3f}'
            # negative, sign comes from the value
            else:
                _ = f'{coef:.3f}'

            # adjust term numbers
            if self.base is 1:
                if number > 9:  # two-digit term
                    name_lcl = ''.join([name[0],
                                        str(int(name[1:3]) + 1),
                                        name[3:]])
                else:
                    name_lcl = ''.join([name[0],
                                        str(int(name[1]) + 1),
                                        name[2:]])
            else:
                name_lcl = name

            strs.append(' '.join([_, name_lcl]))
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
    if num_terms > len(zernfcns):
        raise ValueError(f'number of terms must be less than {len(zernfcns)}')

    # precompute the valid indexes in the original data
    pts = np.isfinite(data)

    # set up an x/y rho/phi grid to evaluate Zernikes on
    x, y = np.linspace(-1, 1, data.shape[1]), np.linspace(-1, 1, data.shape[0])
    xx, yy = np.meshgrid(x, y)
    rho = m.sqrt(xx**2 + yy**2)[pts].flatten()
    phi = m.atan2(xx, yy)[pts].flatten()

    # compute each Zernike term
    zernikes = []
    for i in range(num_terms):
        base_zern = zernfcns[i](rho, phi)
        if rms_norm:
            base_zern *= _normalizations[i]
        zernikes.append(base_zern)
    zerns = np.asarray(zernikes).T

    # use least squares to compute the coefficients
    coefs = np.linalg.lstsq(zerns, data[pts].flatten(), rcond=None)[0]
    return coefs.round(round_at)


zcache = FZCache()
