"""Optimization primitives for prysm."""

from .problem import (  # NOQA
    Problem,
    as_problem,
)

from .activation import (  # NOQA
    Softmax,
    GumbelSoftmax,
    DiscreteEncoder,
)

from .cost import (  # NOQA
    mean_square_error
)

from .sample_problems import (  # NOQA
    SphereProblem,
    RosenbrockProblem,
    RastriginProblem,
    HimmelblauProblem,
    sphere,
    rosenbrock,
    rastrigin,
    himmelblau,
)

from .least_squares import (  # NOQA
    DampedLeastSquares,
    DampedLeastSquaresResult,
    damped_least_squares,
)

from .governors import (  # NOQA
    StepRecord,
    GovernorDecision,
    OptimizationResult,
    Governor,
    AnyGovernor,
    AllGovernor,
    MaxIterations,
    MaxEvaluations,
    FunctionTolerance,
    GradientTolerance,
    StepTolerance,
    ConstraintTolerance,
)

from .plotting import (  # NOQA
    plot_convergence,
)

from .optimizers import (  # NOQA
    runN,
    run_until,
    GradientDescent,
    AdaGrad,
    RMSProp,
    AdaMomentum,
    Adam,
    RAdam,
    Yogi,
    LBFGSB,
    PrysmLBFGSB,
)
