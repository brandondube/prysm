"""A repository of seidel aberration descriptions used to model pupils of optical systems.
"""
from functools import lru_cache

from .conf import config
from .pupil import Pupil

from prysm import mathops as m


class Seidel(Pupil):
    """Seidel pupil description.

    Attributes
    ----------
    coefs : `list`
        list of coefficient values
    eqns : `list`
        list of Wxxx expressions
    field : `int`
        field point associated with this pupil
    phase : `numpy.ndarray`
        phase errors of this pupil
    fcn : `numpy.ndarray`
        wavefunction of this pupil

    """

    def __init__(self, *args, **kwargs):
        """Create a new Seidel instance.

        Parameters
        ----------
        **kwargs
            - 'Wxxx' any argument of this form corresponding to an H H Hopkins wavefront polynomial
            - {'field', 'relative_field', 'h'}, field height

        """
        self.eqns = []
        self.coefs = []
        pass_args = {}
        self.field = 1
        if kwargs is not None:
            for key, value in kwargs.items():
                if key[0].lower() == 'w' and len(key) == 4:
                    self.eqns.append(wexpr_to_opd_expr(key))
                    self.coefs.append(value)
                elif key.lower() in ('field', 'relative_field', 'h'):
                    self.field = value
                else:
                    pass_args[key] = value

        super().__init__(**pass_args)

    def build(self):
        """Use the wavefront coefficients stored in this class instance to build a wavefront model.

        Returns
        -------
        self.phase : `numpy.ndarray`
            phase errors over (x,y) coordinates
        self.fcn : `numpy.ndarray`
            wavefunction over (x,y) coordinates

        """
        mathexprs = ['m.zeros((self.samples, self.samples))']
        for term, coef in zip(self.eqns, self.coefs):
            mathexprs.append(str(coef) + '*(' + term + ')')

        # pull the field point into the namespace our expression wants
        self._gengrid()
        H = self.field
        rho, phi = self.rho, self.phi

        # compute the pupil phase and wave function
        self.phase = eval('+'.join(mathexprs)).astype(config.precision)
        return self.phase

    def __repr__(self):
        """Describe object.

        Returns
        -------
        `str`
            object description

        """
        return str(self.__dict__)


@lru_cache()
def wexpr_to_opd_expr(Wxxx):
    """Convert a W notation to a string with numpy code to evaluate for pupil phase.

    Parameters
    ----------
    Wxxx : `string`
        A string of the form "W000," "W131", etc.

    Returns
    -------
    `string`
        Contains typed numpy expressions to be evaluated to return phase

    """
    # pop the W off and separate the characters
    _ = list(Wxxx[1:])
    H, rho, phi = _[0], _[1], _[2]
    # .format converts to bytecode, f-strings do not.  Micro-optimization here
    return 'H**{0} * rho**{1} * m.cos(phi)**{2}'.format(H, rho, phi)
