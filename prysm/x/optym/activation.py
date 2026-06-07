"""Activation functions and related nodes."""
from prysm.mathops import np, row_dot
from prysm.conf import config

# resources used in deriving softmax reverse()
# https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
# https://dlsys.cs.washington.edu/pdf/lecture4.pdf
# https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
# https://cs231n.github.io/optimization-2/#sigmoid
# (pytorch/caffe2/operators/softmax_ops.cu::softmax_gradient_kernel)
# https://github.com/pytorch/pytorch/blob/59a01c49ee180c8d332e14bf3d5cbd1e8707bb65/caffe2/operators/softmax_ops.cu#L800-L832C15
# https://github.com/Nayan143/Backpropagation-SoftMax/


class Softmax:
    """Softmax activation function.

    The final axis holds logits; leading axes are independent variables.

    """
    def __init__(self):
        """Create a new Softmax node."""
        self.out = None
        self.in_shape = None
        self.work_shape = None

    def forward(self, x):
        """Perform Softmax activation on logits.

        Parameters
        ----------
        x : ndarray, shape (A,B,C, ... K)
            Logits with K levels along the final axis.

        Returns
        -------
        ndarray
            Same shape as x, with sum(axis=-1) == 1.

        """
        assert x.ndim > 1, "prysm's softmax is meant for use with multiple independent variables at once"

        xx = x.reshape((-1, x.shape[-1]))
        self.in_shape = x.shape
        self.work_shape = xx.shape

        # newaxis trick; get numpy to broadcast over the last dimension
        xnorm = xx - xx.max(axis=1)[:, np.newaxis]
        e_x = np.exp(xnorm)
        norm = e_x.sum(axis=1)
        self.out = e_x / norm[:, np.newaxis]
        return self.out.reshape(self.in_shape)

    def backprop(self, grad):
        """Backpropagate grad through Softmax.

        Parameters
        ----------
        grad : ndarray
            Upstream gradient, same shape as the input to forward().

        Returns
        -------
        ndarray
            dcost/dsoftmax-input

        """
        # NOTE: the (I - out·outᵀ) Jacobian is symmetric and rank-K-1; a
        # specialized kernel could halve work, but the row_dot + broadcast
        # implementation below already vectorizes well on GPU.
        assert self.out is not None, 'must run forward() before running reverse()'

        grad = grad.reshape(self.work_shape)

        # first step is to compute the dot product between the activation levels
        # and the input gradient
        tmp = row_dot(grad, self.out)
        # tmp will be of shape (K,) for an (N, K) work shape
        tmp = np.broadcast_to(tmp[:, np.newaxis], self.work_shape)

        tmp2 = grad - tmp
        gout = self.out*tmp2
        return gout.reshape(self.in_shape)


class GumbelSoftmax:
    """Softmax with stochastic Gumbel noise.

    See:
    https://arxiv.org/pdf/1611.01144.pdf
    https://arxiv.org/pdf/1611.00712.pdf

    """
    def __init__(self, tau=1, eps=None):
        """Create a new GumbelSoftmax estimator.

        tau is the temperature; smaller positive values are more discrete.
        """
        self.tau = tau
        self.eps = eps or np.finfo(config.precision).eps
        self.rng = np.random.default_rng()
        self.smax = Softmax()

    def forward(self, x):
        """Gumbel-softmax process on x."""
        # draw gumbel noise
        shp = x.shape
        eps = self.eps
        # footnote 1 from https://arxiv.org/pdf/1611.01144.pdf,
        # with a guard against log of 0
        # u = np.random.uniform(low=0, high=1, size=shp)
        # NOTE: rng.gumbel() draws from Gumbel(loc, scale). This formulation
        # uses -log(-log(u)) which is Gumbel(0, 1) but with explicit eps
        # guards against log(0). Equivalent when u is well inside (0, 1);
        # diverges only at the numerical boundary, which we want to guard.
        u = self.rng.uniform(low=0, high=1, size=shp)
        g = -np.log(-np.log(u + eps) + eps)
        # Add Gumbel noise to logits and normalize by temperature.
        y = x + g
        yy = y / self.tau
        return self.smax.forward(yy)

    def backprop(self, protograd):
        """Adjoint of forward()."""
        # first step, back out the softmax
        pg = self.smax.backprop(protograd)
        return pg / self.tau  # dy/dx = dy/dyy, nothing from g


class DiscreteEncoder:
    """Continuous proxy for discrete-valued variables."""
    def __init__(self, estimator, levels):
        """Create a new DiscreteEncoder.

        Parameters
        ----------
        estimator : an initialized estimator
            For example GumbelSoftmax().
        levels : int or ndarray
            if int, self-generates arange(levels)
            else, expected to be K discrete, non-overlapping integer states

        """
        if isinstance(levels, int):
            levels = np.arange(levels)

        self.est = estimator
        self.levels = levels
        self.tmpshape = None

    def forward(self, x):
        """Forward pass through the continuous proxy for optimization.

        Use discretize() to view the current discrete realization.
        """
        levels = self.levels
        expanded_levels = levels[None, :]

        samples = self.est.forward(x)
        # Contract over levels.
        # TODO: this can be done with tensordot / sum_of_2d_modes?
        tmp = (samples * expanded_levels)
        self.tmpshape = tmp.shape
        return tmp.sum(axis=-1)

    def backprop(self, grad):
        """Backpropagation through the continuous proxy for optimization."""
        levels = self.levels
        expanded_levels = levels[None, :]
        # Expand the upstream gradient over levels, then backprop the estimator.
        tmpbar = (np.broadcast_to(grad[:, None], self.tmpshape) * expanded_levels)
        return self.est.backprop(tmpbar)

    def discretize(self, x):
        """Perform discrete encoding of x.

        Early in optimization, this may differ from the continuous proxy.
        """
        encoded = self.est.forward(x)
        # encoded will be (A,B,C ... K)
        # take argmax along dim k, and take that from levels
        indices = np.argmax(encoded, axis=-1)
        return np.take(self.levels, indices)


class _AffineActivation:
    """Base for elementwise activations of the form y = a · f(x - x0) + y0.

    Parameters
    ----------
    a : float
        slope scaling.
    x0 : float
        horizontal shift (input is recentered to x - x0 before the
        underlying activation is applied).
    y0 : float
        vertical shift added to the activation output.

    """

    def __init__(self, a=1, x0=0, y0=0):
        self.a = a
        self.x0 = x0
        self.y0 = y0


class Tanh(_AffineActivation):
    """Affine-scaled hyperbolic tangent: y = tanh(a·(x - x0)) + y0."""

    def forward(self, x):
        x = x - self.x0
        return (2 / (1 + np.exp(-2 * self.a * x)) - 1 + self.y0)

    def backprop(self, x):
        fx = self.forward(x) - self.y0  # peel off y0 to recover tanh value
        return self.a * (1 - fx**2)


class Arctan(_AffineActivation):
    """Affine-scaled arctangent: y = arctan(a·(x - x0)) + y0."""

    def forward(self, x):
        x = x - self.x0
        return np.arctan(self.a * x) + self.y0

    def backprop(self, x):
        u = self.a * (x - self.x0)
        return self.a / (u**2 + 1)


class Softplus(_AffineActivation):
    """Affine-scaled softplus: y = log(1 + exp(a·(x - x0))) + y0."""

    def forward(self, x):
        x = x - self.x0
        return np.log(1 + np.exp(self.a * x)) + self.y0

    def backprop(self, x):
        x = x - self.x0
        return self.a / (1 + np.exp(-self.a * x))


class Sigmoid(_AffineActivation):
    """Affine-scaled logistic sigmoid: y = σ(a·(x - x0)) + y0."""

    def forward(self, x):
        x = x - self.x0
        return (1 / (1 + np.exp(-self.a * x))) + self.y0

    def backprop(self, x):
        sig = self.forward(x) - self.y0  # peel off y0 to recover σ value
        return self.a * sig * (1 - sig)
