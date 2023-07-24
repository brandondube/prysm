"""Optimization primitives for prysm."""

from .activation import (  # NOQA
    Softmax,
    GumbelSoftmax,
    DiscreteEncoder,
)

from .cost import (  # NOQA
    BiasAndGainInvariantError,
)

from .optimizers import (  # NOQA
    GradientDescent,
    AdaGrad,
    ADAM,
    RMSProp
)
