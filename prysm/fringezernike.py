''' A repository of fringe zernike aberration descriptions used to model pupils of optical systems.
'''
import numpy as np

from prysm.conf import config
from prysm.mathops import (
    jit,
    vectorize,
    sin,
    cos,
    atan2,
    sqrt,
)
from prysm.pupil import Pupil


_names = (
    'Z0  - Piston / Bias',
    'Z1  - Tilt X',
    'Z2  - Tilt Y',
    'Z3  - Defocus / Power',
    'Z4  - Primary Astigmatism 00deg',
    'Z5  - Primary Astigmatism 45deg',
    'Z6  - Primary Coma X',
    'Z7  - Primary Coma Y',
    'Z8  - Primary Spherical',
    'Z9  - Primary Trefoil X',
    'Z10 - Primary Trefoil Y',
    'Z11 - Secondary Astigmatism 00deg',
    'Z12 - Secondary Astigmatism 45deg',
    'Z13 - Secondary Coma X',
    'Z14 - Secondary Coma Y',
    'Z15 - Secondary Spherical',
    'Z16 - Primary Tetrafoil X',
    'Z17 - Primary Tetrafoil Y',
    'Z18 - Secondary Trefoil X',
    'Z19 - Secondary Trefoil Y',
    'Z20 - Tertiary Astigmatism 00deg',
    'Z21 - Tertiary Astigmatism 45deg',
    'Z22 - Tertiary Coma X',
    'Z23 - Tertiary Coma Y',
    'Z24 - Tertiary Spherical',
    'Z25 - Pentafoil X',
    'Z26 - Pentafoil Y',
    'Z27 - Secondary Tetrafoil X',
    'Z28 - Secondary Tetrafoil Y',
    'Z29 - Tertiary Trefoil X',
    'Z30 - Tertiary Trefoil Y',
    'Z31 - Quarternary Astigmatism 00deg',
    'Z32 - Quarternary Astigmatism 45deg',
    'Z33 - Quarternary Coma X',
    'Z34 - Quarternary Coma Y',
    'Z35 - Quarternary Spherical',
    'Z36 - Primary Hexafoil X',
    'Z37 - Primary Hexafoil Y',
    'Z38 - Secondary Pentafoil X',
    'Z39 - Secondary Pentafoil Y',
    'Z40 - Tertiary Tetrafoil X',
    'Z41 - Tertiary Tetrafoil Y',
    'Z42 - Quaternary Trefoil X',
    'Z43 - Quaternary Trefoil Y',
    'Z44 - Quinternary Astigmatism 00deg',
    'Z45 - Quinternary Astigmatism 45deg',
    'Z46 - Quinternary Coma X',
    'Z47 - Quinternary Coma Y',
    'Z48 - Quarternary Spherical',
)


@jit
def Z0(rho, phi):
    return np.ones(rho.shape)


@vectorize
def Z1(rho, phi):
    return rho * cos(phi)


@vectorize
def Z2(rho, phi):
    return rho * sin(phi)


@vectorize
def Z3(rho, phi):
    return 2 * rho**2 - 1


@vectorize
def Z4(rho, phi):
    return rho**2 * cos(2 * phi)


@vectorize
def Z5(rho, phi):
    return rho**2 * sin(2 * phi)


@vectorize
def Z6(rho, phi):
    return (-2 * rho + 3 * rho**3) * cos(phi)


@vectorize
def Z7(rho, phi):
    return (-2 * rho + 3 * rho**3) * sin(phi)


@vectorize
def Z8(rho, phi):
    return 6 * rho**4 - 6 * rho**2 + 1


@vectorize
def Z9(rho, phi):
    return rho**3 * cos(3 * phi)


@vectorize
def Z10(rho, phi):
    return rho**3 * sin(3 * phi)


@vectorize
def Z11(rho, phi):
    return (-3 * rho**2 + 4 * rho**4) * cos(2 * phi)


@vectorize
def Z12(rho, phi):
    return (-3 * rho**2 + 4 * rho**4) * sin(2 * phi)


@vectorize
def Z13(rho, phi):
    return (3 * rho - 12 * rho**3 + 10 * rho**5) * cos(phi)


@vectorize
def Z14(rho, phi):
    return (3 * rho - 12 * rho**3 + 10 * rho**5) * sin(phi)


@vectorize
def Z15(rho, phi):
    return 20 * rho**6 + - 30 * rho**4 + 12 * rho**2 - 1


@vectorize
def Z16(rho, phi):
    return rho**4 * cos(4 * phi)


@vectorize
def Z17(rho, phi):
    return rho**4 * sin(4 * phi)


@vectorize
def Z18(rho, phi):
    return (5 * rho**5 - 4 * rho**3) * cos(3 * phi)


@vectorize
def Z19(rho, phi):
    return (5 * rho**5 - 4 * rho**3) * sin(3 * phi)


@vectorize
def Z20(rho, phi):
    return (6 * rho**2 - 20 * rho**4 + 15 * rho**6) * cos(2 * phi)


@vectorize
def Z21(rho, phi):
    return (6 * rho**2 - 20 * rho**4 + 15 * rho**6) * sin(2 * phi)


@vectorize
def Z22(rho, phi):
    return (-4 * rho + 30 * rho**3 - 60 * rho**5 + 35 * rho**7) * cos(phi)


@vectorize
def Z23(rho, phi):
    return (-4 * rho + 30 * rho**3 - 60 * rho**5 + 35 * rho**7) * sin(phi)


@vectorize
def Z24(rho, phi):
    return 70 * rho**8 - 140 * rho**6 + 90 * rho**4 - 20 * rho**2 + 1


@vectorize
def Z25(rho, phi):
    return rho**5 * cos(5 * phi)


@vectorize
def Z26(rho, phi):
    return rho**5 * sin(5 * phi)


@vectorize
def Z27(rho, phi):
    return (6 * rho**6 - 5 * rho**4) * cos(4 * phi)


@vectorize
def Z28(rho, phi):
    return (6 * rho**6 - 5 * rho**4) * sin(4 * phi)


@vectorize
def Z29(rho, phi):
    return (10 * rho**3 - 30 * rho**5 + 21 * rho**7) * cos(3 * phi)


@vectorize
def Z30(rho, phi):
    return (10 * rho**3 - 30 * rho**5 + 21 * rho**7) * cos(3 * phi)


@vectorize
def Z31(rho, phi):
    return (10 * rho**2 - 30 * rho**4 + 21 * rho**6) * cos(2 * phi)


@vectorize
def Z32(rho, phi):
    return (10 * rho**2 - 30 * rho**4 + 21 * rho**6) * sin(2 * phi)


@vectorize
def Z33(rho, phi):
    return (5 * rho - 60 * rho**3 + 210 * rho**5 - 280 * rho**7 + 126 * rho**9)\
        * cos(phi)


@vectorize
def Z34(rho, phi):
    return (5 * rho - 60 * rho**3 + 210 * rho**5 - 280 * rho**7 + 126 * rho**9)\
        * sin(phi)


@vectorize
def Z35(rho, phi):
    return 252 * rho**10 \
        - 630 * rho**8 \
        + 560 * rho**6 \
        - 210 * rho**4 \
        + 30 * rho**2 \
        - 1


@vectorize
def Z36(rho, phi):
    return rho**6 * cos(6 * phi)


@vectorize
def Z37(rho, phi):
    return rho**6 * sin(6 * phi)


@vectorize
def Z38(rho, phi):
    return (7 * rho**7 - 6 * rho**5) * cos(5 * phi)


@vectorize
def Z39(rho, phi):
    return (7 * rho**7 - 6 * rho**5) * sin(5 * phi)


@vectorize
def Z40(rho, phi):
    return (28 * rho**8 - 42 * rho**6 + 15 * rho**4) * cos(4 * phi)


@vectorize
def Z41(rho, phi):
    return (28 * rho**8 - 42 * rho**6 + 15 * rho**4) * sin(4 * phi)


@vectorize
def Z42(rho, phi):
    return (84 * rho**9 - 168 * rho**7 + 105 * rho**5 - 20 * rho**3) * cos(3 * phi)


@vectorize
def Z43(rho, phi):
    return (84 * rho**9 - 168 * rho**7 + 105 * rho**5 - 20 * rho**3) * sin(3 * phi)


@vectorize
def Z44(rho, phi):
    return (210 * rho**10 - 504 * rho**8 + 420 * rho**6 - 140 * rho**4 + 15 * rho**2) \
        * cos(2 * phi)


@vectorize
def Z45(rho, phi):
    return (210 * rho**10 - 504 * rho**8 + 420 * rho**6 - 140 * rho**4 + 15 * rho**2) \
        * sin(2 * phi)


@vectorize
def Z46(rho, phi):
    return (462 * rho**11 - 1260 * rho**9 + 1260 * rho**7 - 560 * rho**5 + 105 * rho**3 - 6 * rho) \
        * cos(phi)


@vectorize
def Z47(rho, phi):
    return (462 * rho**11 - 1260 * rho**9 + 1260 * rho**7 - 560 * rho**5 + 105 * rho**3 - 6 * rho) \
        * sin(phi)


@vectorize
def Z48(rho, phi):
    return 924 * rho**12 \
        - 2772 * rho**10 \
        + 3150 * rho**8 \
        - 1680 * rho**6 \
        + 420 * rho**4 \
        - 42 * rho**2 \
        + 1


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
    48.0: Z48,
}


def zernwrapper(term, rms_norm, rho, phi):
    ''' Wraps the Z0..Z48 functions.
    '''
    if rms_norm is True:
        return _normalizations[term] * zernfcns[term](rho, phi)
    else:
        return zernfcns[term](rho, phi)


# See JCW - http://wp.optics.arizona.edu/jcwyant/wp-content/uploads/sites/13/2016/08/ZernikePolynomialsForTheWeb.pdf
_normalizations = (
    1,            # Z 0
    2,            # Z 1
    2,            # Z 2
    sqrt(3),      # Z 3
    sqrt(6),      # Z 4
    sqrt(6),      # Z 5
    2 * sqrt(2),  # Z 6
    2 * sqrt(2),  # Z 7
    sqrt(5),      # Z 8
    2 * sqrt(2),  # Z 9
    2 * sqrt(2),  # Z10
    sqrt(10),     # Z11
    sqrt(10),     # Z12
    2 * sqrt(3),  # Z13
    2 * sqrt(3),  # Z14
    sqrt(7),      # Z15
    sqrt(10),     # Z16
    sqrt(10),     # Z17
    2 * sqrt(3),  # Z18
    2 * sqrt(3),  # Z19
    sqrt(14),     # Z20
    sqrt(14),     # Z21
    4,            # Z22
    4,            # Z23
    3,            # Z24
    2 * sqrt(3),  # Z25
    2 * sqrt(3),  # Z26
    sqrt(14),     # Z27
    sqrt(14),     # Z28
    4,            # Z29
    4,            # Z30
    3 * sqrt(2),  # Z31
    3 * sqrt(2),  # Z32
    2 * sqrt(5),  # Z33
    2 * sqrt(5),  # Z34
    sqrt(11),     # Z35
    sqrt(14),     # Z36
    sqrt(14),     # Z37
    4,            # Z38
    4,            # Z39
    3 * sqrt(2),  # Z40
    3 * sqrt(2),  # Z41
    2 * sqrt(5),  # Z42
    2 * sqrt(5),  # Z43
    sqrt(22),     # Z44
    sqrt(22),     # Z45
    2 * sqrt(6),  # Z46
    2 * sqrt(6),  # Z47
    sqrt(13),     # Z48
)


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

            Zx (`float`): xth fringe zernike coefficient, in range [0,48] or [1,49] if base 1

        Returns:
            FringeZernike.  A new :class:`FringeZernike` pupil instance.

        Notes:
            Supports multiple syntaxes:
                - args: pass coefficients as a list, including terms up to
                        the highest given Z-index.
                        e.g. passing [1,2,3] will be interpreted as Z0=1, Z1=2, Z3=3.

                - kwargs: pass a named set of zernike terms.
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
        self._gengrid()
        self.phase = np.zeros((self.samples, self.samples), dtype=config.precision)
        for term, coef in enumerate(self.coefs):
            # short circuit for speed
            if coef == 0:
                continue
            self.phase += coef * zernwrapper(term=term,
                                             rms_norm=self.normalize,
                                             rho=self.rho,
                                             phi=self.phi)

        self._correct_phase_units()
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
    pts = np.isfinite(data)

    # set up an x/y rho/phi grid to evaluate zernikes on
    x, y = np.linspace(-1, 1, data.shape[1]), np.linspace(-1, 1, data.shape[0])
    xv, yv = np.meshgrid(x, y)
    rho = sqrt(xv**2 + yv**2)[pts].flatten()
    phi = atan2(xv, yv)[pts].flatten()

    # compute each zernike term
    zernikes = []
    for i in range(num_terms):
        zernikes.append(zernwrapper(i, rms_norm, rho, phi))
    zerns = np.asarray(zernikes).T

    # use least squares to compute the coefficients
    coefs = np.linalg.lstsq(zerns, data[pts].flatten())[0]
    return coefs.round(round_at)
