"""Optimization problem over a coating stack's layer thicknesses.

CoatingProblem adapts a Stack plus a MeritFunction to the optym.Problem protocol:
the design vector x is the (variable) layer thicknesses, _fg(x) returns the merit
and its analytic gradient via the diff.py adjoint, and residuals(x) exposes the
weighted residual vector for the least-squares (Levenberg-Marquardt) path.

Layer thicknesses are physical and must stay non-negative; this is enforced with
box bounds at the optimizer rather than a reparameterization, so the analytic
gradient flows straight through.
"""

from prysm.conf import config
from prysm.mathops import np

from prysm.x.optym.problem import Problem

from .stack import Stack
from .merit import as_merit
from .diff import thickness_gradient, index_gradient


class CoatingProblem(Problem):
    """A thickness-design problem: minimize a MeritFunction over a Stack.

    Parameters
    ----------
    stack : Stack
        the starting stack; its indices, substrate, and ambient are held fixed
        and its thicknesses seed the design vector.
    merit : MeritFunction, term, or sequence of terms
        the objective; normalized with merit.as_merit.
    variable_layers : sequence of int, optional
        indices of the layers whose design variable is free.  Default: all.
    variables : {'thickness', 'index'}, optional
        which per-layer quantity is optimized.  'index' is for graded-index /
        rugate films and requires numeric (non-dispersive) layer indices.

    """

    has_fg = True

    def __init__(self, stack, merit, *, variable_layers=None,
                 variables='thickness'):
        super().__init__()
        if variables not in ('thickness', 'index'):
            raise ValueError("variables must be 'thickness' or 'index'")
        self.stack0 = stack
        self.merit = as_merit(merit)
        self.variables = variables
        n = len(stack)
        if variable_layers is None:
            variable_layers = list(range(n))
        self.variable_layers = list(variable_layers)
        self._mask = np.zeros(n, dtype=bool)
        self._mask[self.variable_layers] = True
        self._grad_fn = (index_gradient if variables == 'index'
                         else thickness_gradient)
        if variables == 'index':
            for i in self.variable_layers:
                if callable(stack.indices[i]):
                    raise TypeError(
                        'index-variable design needs numeric layer indices; '
                        f'layer {i} is a dispersion callable')

    def x0(self):
        """Initial design vector: the variable layers' thickness or index."""
        if self.variables == 'index':
            vals = np.array([np.real(self.stack0.indices[i])
                             for i in self.variable_layers],
                            dtype=config.precision)
            return vals
        th = np.asarray(self.stack0.thicknesses, dtype=config.precision)
        return th[self._mask].copy()

    def stack_from_x(self, x):
        """Build a Stack with the variable thickness/index replaced by x."""
        x = np.asarray(x, dtype=config.precision)
        if self.variables == 'index':
            indices = list(self.stack0.indices)
            for slot, i in enumerate(self.variable_layers):
                indices[i] = float(x[slot])
            return Stack(indices, self.stack0.thicknesses,
                         self.stack0.substrate_index, self.stack0.ambient_index)
        th = np.array(self.stack0.thicknesses, dtype=config.precision)
        th[self._mask] = x
        return Stack(self.stack0.indices, th, self.stack0.substrate_index,
                     self.stack0.ambient_index)

    def _fg(self, x):
        stack = self.stack_from_x(x)
        val, grad = self.merit.value_and_grad(stack, grad_fn=self._grad_fn)
        return val, grad[self._mask]

    def residuals(self, x):
        """Weighted residual vector at x (for the least-squares path)."""
        stack = self.stack_from_x(x)
        return self.merit.residuals(stack)


__all__ = ['CoatingProblem']
