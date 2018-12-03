''' A repository of standard zernike aberration descriptions used to model pupils of
optical systems.
'''

from .conf import config
from .pupil import Pupil

from prysm import mathops as m

_names = (
    'Z0  - Piston / Bias',
    'Z1  - Tilt X',
    'Z2  - Tilt Y',
    'Z3  - Primary Astigmatism 00deg',
    'Z4  - Defocus / Power',
    'Z5  - Primary Astigmatism 45deg',
    'Z6  - Primary Trefoil X',
    'Z7  - Primary Coma X',
    'Z8  - Primary Coma Y',
    'Z9  - Primary Trefoil Y',
    'Z10 - Primary Tetrafoil X',
    'Z11 - Secondary Astigmatism 00deg',
    'Z12 - Primary Spherical',
    'Z13 - Secondary Astigmatism 45deg',
    'Z14 - Primary Tetrafoil Y',
    'Z15 - Primary Pentafoil X',
    'Z16 - Secondary Trefoil X',
    'Z17 - Secondary Coma X',
    'Z18 - Secondary Coma Y',
    'Z19 - Secondary Trefoil Y',
    'Z20 - Primary Pentafoil Y',
    'Z21 - Primary Hexafoil X',
    'Z22 - Secondary Tetrafoil X',
    'Z23 - Tertiary Astigmatism 00deg',
    'Z24 - Secondary Spherical',
    'Z25 - Tertariary Astigmatism 45deg',
    'Z26 - Secondary Tetrafoil Y',
    'Z27 - Primary Hexafoil Y',
    'Z28 - Primary Heptafoil X',
    'Z29 - Secondary Pentafoil X',
    'Z30 - Tertiary Trefoil X',
    'Z31 - Tertiary Coma X',
    'Z32 - Tertiary Coma Y',
    'Z33 - Tertiary Trefoil Y',
    'Z34 - Secondary Pentafoil Y',
    'Z35 - Primary Heptafoil Y',
    'Z36 - Primary Octafoil X',
    'Z37 - Secondary Hexafoil X',
    'Z38 - Tertiary Tetrafoil X',
    'Z39 - Quarternary Astigmatism 00deg',
    'Z40 - Tertiary Spherical',
    'Z41 - Quarternary Astigmatism 45deg',
    'Z42 - Tertiary Tetrafoil Y',
    'Z43 - Secondary Hexafoil Y',
    'Z44 - Primary Octafoil Y',
    'Z45 - Primary Nonafoil X',
    'Z46 - Secondary Heptafoil X',
    'Z47 - Tertiary Pentafoil X',
)


@m.jit
def Z0(rho, phi):
    return m.ones(rho.shape)


@m.vectorize
def Z1(rho, phi):
    return rho * m.cos(phi)


@m.vectorize
def Z2(rho, phi):
    return rho * m.sin(phi)


@m.vectorize
def Z3(rho, phi):
    return rho**2 * m.cos(2 * phi)


@m.vectorize
def Z4(rho, phi):
    return 2 * rho ** 2 - 1


@m.vectorize
def Z5(rho, phi):
    return rho**2 * m.sin(2 * phi)


@m.vectorize
def Z6(rho, phi):
    return rho**3 * m.cos(3 * phi)


@m.vectorize
def Z7(rho, phi):
    return (3 * rho**3 - 2 * rho) * m.cos(phi)


@m.vectorize
def Z8(rho, phi):
    return (3 * rho**3 - 2 * rho) * m.sin(phi)


@m.vectorize
def Z9(rho, phi):
    return rho**3 * m.sin(3 * phi)


@m.vectorize
def Z10(rho, phi):
    return rho**4 * m.cos(4 * phi)


@m.vectorize
def Z11(rho, phi):
    return (4 * rho**4 - 3 * rho**2) * m.cos(2 * phi)


@m.vectorize
def Z12(rho, phi):
    return -6 * rho**2 + 6 * rho**4 + 1


@m.vectorize
def Z13(rho, phi):
    return (4 * rho**4 - 3 * rho**2) * m.sin(2 * phi)


@m.vectorize
def Z14(rho, phi):
    return rho**4 * m.sin(4 * phi)


@m.vectorize
def Z15(rho, phi):
    return rho**5 * m.cos(5 * phi)


@m.vectorize
def Z16(rho, phi):
    return (5 * rho**5 - 4 * rho**3) * m.cos(3 * phi)


@m.vectorize
def Z17(rho, phi):
    return (10 * rho**5 - 12 * rho**3 + 3 * rho) * m.cos(phi)


@m.vectorize
def Z18(rho, phi):
    return (10 * rho**5 - 12 * rho**3 + 3 * rho) * m.sin(phi)


@m.vectorize
def Z19(rho, phi):
    return (5 * rho**5 - 4 * rho**3) * m.sin(3 * phi)


@m.vectorize
def Z20(rho, phi):
    return rho**5 * m.cos(5 * phi)


@m.vectorize
def Z21(rho, phi):
    return rho**6 * m.cos(6 * phi)


@m.vectorize
def Z22(rho, phi):
    return (6 * rho**6 - 5 * rho**4) * m.cos(4 * phi)


@m.vectorize
def Z23(rho, phi):
    return (15 * rho**6 - 20 * rho**4 + 6 * rho**2) * m.cos(2 * phi)


@m.vectorize
def Z24(rho, phi):
    return 20 * rho**6 - 30 * rho**4 + 12 * rho**2 - 1


@m.vectorize
def Z25(rho, phi):
    return (15 * rho**6 - 20 * rho**4 + 6 * rho**2) * m.sin(2 * phi)


@m.vectorize
def Z26(rho, phi):
    return (6 * rho**6 - 5 * rho**4) * m.sin(4 * phi)


@m.vectorize
def Z27(rho, phi):
    return rho**6 * m.sin(6 * phi)


@m.vectorize
def Z28(rho, phi):
    return rho**6 * m.cos(7 * phi)


@m.vectorize
def Z29(rho, phi):
    return (7 * rho**7 - 6 * rho**5) * m.cos(5 * phi)


@m.vectorize
def Z30(rho, phi):
    return (21 * rho**7 - 30 * rho**5 + 10 * rho**3) * m.cos(3 * phi)


@m.vectorize
def Z31(rho, phi):
    return (35 * rho**7 - 60 * rho**5 + 30 * rho**3 - 4 * rho) * m.cos(phi)


@m.vectorize
def Z32(rho, phi):
    return (35 * rho**7 - 60 * rho**5 + 30 * rho**3 - 4 * rho) * m.sin(phi)


@m.vectorize
def Z33(rho, phi):
    return (21 * rho**7 - 30 * rho**5 + 10 * rho**3) * m.sin(3 * phi)


@m.vectorize
def Z34(rho, phi):
    return (7 * rho**7 - 6 * rho**5) * m.sin(5 * phi)


@m.vectorize
def Z35(rho, phi):
    return rho**7 * m.sin(7 * phi)


@m.vectorize
def Z36(rho, phi):
    return rho**8 * m.cos(8 * phi)


@m.vectorize
def Z37(rho, phi):
    return (8 * rho**8 - 7 * rho**6) * m.cos(6 * phi)


@m.vectorize
def Z38(rho, phi):
    return (28 * rho**8 - 42 * rho**6 + 15 * rho**4) * m.cos(4 * phi)


@m.vectorize
def Z39(rho, phi):
    return (56 * rho**8 - 105 * rho**6 + 60 * rho**4 - 10 * rho**2) * m.cos(2 * phi)


@m.vectorize
def Z40(rho, phi):
    return 70 * rho**8 - 140 * rho**7 + 90 * rho**4 - 20 * rho**2 + 1


@m.vectorize
def Z41(rho, phi):
    return (56 * rho**8 - 105 * rho**6 + 60 * rho**4 - 10 * rho**2) * m.cos(2 * phi)


@m.vectorize
def Z42(rho, phi):
    return (28 * rho**8 - 42 * rho**6 + 15 * rho**4) * m.sin(4 * phi)


@m.vectorize
def Z43(rho, phi):
    return (8 * rho**8 - 7 * rho**6) * m.sin(6 * phi)


@m.vectorize
def Z44(rho, phi):
    return rho**8 * m.sin(8 * phi)


@m.vectorize
def Z45(rho, phi):
    return rho**9 * m.cos(9 * phi)


@m.vectorize
def Z46(rho, phi):
    return (9 * rho**9 - 8 * rho**7) * m.cos(7 * phi)


@m.vectorize
def Z47(rho, phi):
    return (36 * rho**9 - 56 * rho**7 + 21 * rho**5) * m.cos(5 * phi)


zernfcns = {
    0.0: Z0,
    1.0: Z1,
    2.0: Z2,
    3.0: Z3,
    4.0: Z4,
    5.0: Z5,
    6.0: Z6,
    7.0: Z7,
    8.0: Z8,
    9.0: Z9,
    10.0: Z10,
    11.0: Z11,
    12.0: Z12,
    13.0: Z13,
    14.0: Z14,
    15.0: Z15,
    16.0: Z16,
    17.0: Z17,
    18.0: Z18,
    19.0: Z19,
    20.0: Z20,
    21.0: Z21,
    22.0: Z22,
    23.0: Z23,
    24.0: Z24,
    25.0: Z25,
    26.0: Z26,
    27.0: Z27,
    28.0: Z28,
    29.0: Z29,
    30.0: Z30,
    31.0: Z31,
    32.0: Z32,
    33.0: Z33,
    34.0: Z34,
    35.0: Z35,
    36.0: Z36,
    37.0: Z37,
    38.0: Z38,
    39.0: Z39,
    40.0: Z40,
    41.0: Z41,
    42.0: Z42,
    43.0: Z43,
    44.0: Z44,
    45.0: Z45,
    46.0: Z46,
    47.0: Z47,
}


def zernwrapper(term, rho, phi):
    ''' Wraps the Z0..Z48 functions.
    '''
    return zernfcns[term](rho, phi)


class StandardZernike(Pupil):
    '''Standard Zernike pupil description

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
        ''' Creates a StandardZernike Pupil object.

        Args:
            samples (int): number of samples across pupil diameter.

            wavelength (float): wavelength of light, in um.

            epd: (float): diameter of the pupil, in mm.

            opd_unit (string): unit OPD is expressed in.  One of:
                ($\lambda$, waves, $\mu m$, microns, um, nm , nanometers).

            base (`int`): 0 or 1, adjusts the base index of the polynomial
                expansion.

            Zx (float): xth standard zernike coefficient, in range [0,47], 0-base.

        Returns:
            StandardZernike.  A new :class:`StandardZernike` pupil instance.

        Notes:
            Supports multiple syntaxes:
                - args: pass coefficients as a list, including terms up to the highest given Z-index.
                        e.g. passing [1,2,3] will be interpreted as Z0=1, Z1=2, Z3=3.

                - kwargs: pass a named set of zernike terms.
                          e.g. StandardZernike(Z0=1, Z1=2, Z2=3)

            Supports unit conversion, can pass kwarg:
                - opd_unit='nm': coefficients are expressed in units of nm

            The kwargs syntax overrides the args syntax.

        '''

        if args is not None:
            if len(args) is 0:
                self.coefs = [0] * len(zernfcns)
            else:
                self.coefs = [*args[0]]
        else:
            self.coefs = [0] * len(zernfcns)

        pass_args = {}

        self.base = config.zernike_base
        try:
            bb = kwargs['base']
            if bb > 1:
                raise ValueError('It violates convention to use a base greater than 1.')
            self.base = bb
        except KeyError:
            # user did not specify base
            pass

        if kwargs is not None:
            for key, value in kwargs.items():
                if key[0].lower() == 'z':
                    idx = int(key[1:])  # strip 'Z' from index
                    self.coefs[idx - self.base] = value
                elif key.lower() == 'base':
                    self.base = value
                else:
                    pass_args[key] = value

        super().__init__(**pass_args)

    def build(self):
        # construct an equation for the phase of the pupil
        # build a coordinate system over which to evaluate this function
        self._gengrid()
        self.phase = m.zeros((self.samples, self.samples), dtype=config.precision)
        for term, coef in enumerate(self.coefs):
            # short circuit for speed
            if coef == 0:
                continue
            self.phase += coef * zernwrapper(term=term,
                                             rho=self.rho,
                                             phi=self.phi)

        self._correct_phase_units()
        return self.phase, self.fcn

    def __repr__(self):
        '''Pretty-print pupil description
        '''
        header = 'Standard Zernike description with:\n\t'

        strs = []
        for number, (coef, name) in enumerate(zip(self.coefs, _names)):
            # skip 0 terms
            if coef == 0:
                continue

            # positive coefficient, prepend with +
            if m.sign(coef) == 1:
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
    ''' Fits a number of zernike coefficients to provided data by minimizing
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
    pts = m.isfinite(data)

    # set up an x/y rho/phi grid to evaluate zernikes on
    x, y = m.linspace(-1, 1, data.shape[1]), m.linspace(-1, 1, data.shape[0])
    xv, yv = m.meshgrid(x, y)
    rho = m.sqrt(xv**2 + yv**2)[pts].flatten()
    phi = m.arctan2(xv, yv)[pts].flatten()

    # compute each zernike term
    zernikes = []
    for i in range(num_terms):
        zernikes.append(zernwrapper(i, rms_norm, rho, phi))
    zerns = m.asarray(zernikes).T

    # use least squares to compute the coefficients
    coefs = m.linalg.lstsq(zerns, data[pts].flatten())[0]
    return coefs.round(round_at)
