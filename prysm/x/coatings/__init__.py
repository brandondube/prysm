"""Thin-film coating design and synthesis.

Built on the committed 2x2 transfer-matrix primitives in prysm.thinfilm, this
package moves from coating analysis (R/T/A of a given stack) toward synthesis:
field-aware merit functions, analytic-gradient refinement, needle optimization,
and rugate synthesis.

Phase 1 is the foundation: a Stack representation plus the field /
partial-product engine that exposes the internal electric and magnetic field at
every boundary -- the intermediate state the coefficient-only
multilayer_stack_rt collapses away, and which every later method consumes.

Phase 2 adds the differentiable transfer-matrix engine (analytic O(N) thickness
gradients) and analytic-gradient refinement; Phase 3 adds needle synthesis.
"""

from .stack import (
    Stack,
    stack_characteristic_matrices,
    forward_products,
    backward_products,
    internal_fields,
    field_at_depth,
    RTA,
    stack_rt,
)
from .diff import (
    forward_eval,
    thickness_gradient,
    index_gradient,
)
from .merit import (
    Reflectance,
    Transmittance,
    LayerAbsorptance,
    FieldIntensityAtBoundary,
    PeakFieldAtInterfaces,
    FieldInLayer,
    MeritFunction,
    as_merit,
)
from .problem import CoatingProblem
from .refine import refine, CoatingResult
from .needle import (
    needle_function,
    insert_needle,
    cleanup,
    synthesize,
    NeedleResult,
)
from .monitoring import (
    monitoring_trace,
    turning_points,
    level_cut,
    cutoff_levels,
    simulate_run,
    monitoring_error_sensitivity,
    choose_monitor_wavelength,
)
from .rugate import (
    quintic_taper,
    discretize_profile,
    rugate_period,
    notch_wavelength,
    sinusoidal_rugate,
    apodize,
    rugate_from_target,
)

__all__ = [
    # stack / field engine (Phase 1)
    'Stack',
    'stack_characteristic_matrices',
    'forward_products',
    'backward_products',
    'internal_fields',
    'field_at_depth',
    'RTA',
    'stack_rt',
    # differentiable engine + refinement (Phase 2)
    'forward_eval',
    'thickness_gradient',
    'index_gradient',
    'Reflectance',
    'Transmittance',
    'LayerAbsorptance',
    'FieldIntensityAtBoundary',
    'PeakFieldAtInterfaces',
    'FieldInLayer',
    'MeritFunction',
    'as_merit',
    'CoatingProblem',
    'refine',
    'CoatingResult',
    # needle synthesis (Phase 3)
    'needle_function',
    'insert_needle',
    'cleanup',
    'synthesize',
    'NeedleResult',
    # monitoring-strategy simulation (Phase 5)
    'monitoring_trace',
    'turning_points',
    'level_cut',
    'cutoff_levels',
    'simulate_run',
    'monitoring_error_sensitivity',
    'choose_monitor_wavelength',
    # rugate / inhomogeneous-index synthesis (Phase 6)
    'quintic_taper',
    'discretize_profile',
    'rugate_period',
    'notch_wavelength',
    'sinusoidal_rugate',
    'apodize',
    'rugate_from_target',
]
