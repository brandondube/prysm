"""Activation functions and related nodes."""
from prysm.mathops import np
from prysm.conf import config

from prysm.x.raytracing.spencer_and_murty import _multi_dot

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

    Softmax is a soft, differntiable alternative to argmax.  It is used as a
    component of GumbelSoftmaxEncoder to ecourage / softly force variables to
    take on one of K discrete states.

    The arrays passed to forward() and reverse() may take any number of dimensions.
    The understanding of the inputs should be that the final dimension is what
    is being softmaxed over, and all preceeding dimensions are independent variables.

    For example, to softmax a 256x256 array over 4 activation levels, the input
    should be 256x256x4.

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
        x : numpy.ndarray, shape (A,B,C, ... K)
            any number of leading dimensions, required trailing dimension of
            size K, where K is the number of levels to be used with an encoder

        Returns
        -------
        numpy.ndarray
            same shape as x, activated x, where sum(axis=K) == 1

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
        grad : numpy.ndarray
            gradient of scalar cost function w.r.t. following step in forward
            problem,  of same shape as passed to forward()

        Returns
        -------
        numpy.ndarray
            dcost/dsoftmax-input

        """
        # TODO: look into exploiting the symmetry of the result here
        # to speed up the calculation
        assert self.out is not None, 'must run forward() before running reverse()'

        grad = grad.reshape(self.work_shape)

        # first step is to compute the dot product between the activation levels
        # and the input gradient
        tmp = _multi_dot(grad, self.out)
        # tmp will be of shape (K,) for an (N, K) work shape
        tmp = np.broadcast_to(tmp[:, np.newaxis], self.work_shape)

        tmp2 = grad - tmp
        gout = self.out*tmp2
        return gout.reshape(self.in_shape)


class GumbelSoftmax:
    """GumbelSoftmax combines the softmax activation function with stochastic Gumbel noise to encourage variables to fall into discrete categories.

    See:
    https://arxiv.org/pdf/1611.01144.pdf
    https://arxiv.org/pdf/1611.00712.pdf

    You most likely want to use GumbelSoftmaxEncoder, not this class directly.

    """
    def __init__(self, tau=1, eps=None):
        """Create a new GumbelSoftmax estimator.

        tau is the temperature parameter,
        as tau -> 0, output becomes increasingly discrete; tau should be
        annealed towards zero, but never exactly zero, over the course of
        design/optimization.
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
        # TODO: can this be replaced with rng.gumbel()?
        # what is the relatinship between low, high and gumbel parameters?
        u = self.rng.uniform(low=0, high=1, size=shp)
        g = -np.log(-np.log(u + eps) + eps)
        # x are the "logits" from the paper, add gumbel noise and normalize by temperature
        y = x + g
        yy = y / self.tau
        return self.smax.forward(yy)

    def backprop(self, protograd):
        """Adjoint of forward()."""
        # first step, back out the softmax
        pg = self.smax.backprop(protograd)
        return pg / self.tau  # dy/dx = dy/dyy, nothing from g


class DiscreteEncoder:
    """An encoder that embds a continuous encoder, which encourages values to cluster at discrete states."""
    def __init__(self, estimator, levels):
        """Create a new DiscreteEncoder.

        Parameters
        ----------
        estimator : an initialized estimator
            for example GumbelSoftmax()
        levels : int or numpy.ndarray
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

        use discretize() to view the current best fit discrete realization.
        """
        levels = self.levels
        expanded_levels = levels[None, :]

        samples = self.est.forward(x)
        # this product is of shape (N,K) for N variables and K levels
        # it is then contracted over the levels axis
        # TODO: this can be done with tensordot / sum_of_2d_modes?
        tmp = (samples * expanded_levels)
        self.tmpshape = tmp.shape
        return tmp.sum(axis=-1)

    def backprop(self, grad):
        """Backpropagation through the continuous proxy for optimization."""
        levels = self.levels
        expanded_levels = levels[None, :]
        # TODO: this can be done with tensordot?
        # explanation:
        # grad over sum is "1"
        # with chain rule, d/dx * d/dy, so 1x grad over the dim
        # mul by 1 is a no-op, so just use broadcast_to to expand
        # dimensionality
        # then backprop rule for mul is just mul, so do that
        # then go through the estimator backwards; done
        tmpbar = (np.broadcast_to(grad[:, None], self.tmpshape) * expanded_levels)
        return self.est.backprop(tmpbar)

    def discretize(self, x):
        """Perform discrete encoding of x.

        Note that when the estimator weights are not yet converged or non-sparse
        the output of this function will not match closely to the continuous proxy
        that is actually being optimized.
        """
        encoded = self.est.forward(x)
        # encoded will be (A,B,C ... K)
        # take argmax along dim k, and take that from levels
        indices = np.argmax(encoded, axis=-1)
        return np.take(self.levels, indices)
