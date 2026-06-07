"""Thin-film coating analysis and synthesis."""

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
    # stack / field engine
    'Stack',
    'stack_characteristic_matrices',
    'forward_products',
    'backward_products',
    'internal_fields',
    'field_at_depth',
    'RTA',
    'stack_rt',
    # differentiable engine + refinement
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
    # needle synthesis
    'needle_function',
    'insert_needle',
    'cleanup',
    'synthesize',
    'NeedleResult',
    # monitoring
    'monitoring_trace',
    'turning_points',
    'level_cut',
    'cutoff_levels',
    'simulate_run',
    'monitoring_error_sensitivity',
    'choose_monitor_wavelength',
    # rugate / inhomogeneous-index synthesis
    'quintic_taper',
    'discretize_profile',
    'rugate_period',
    'notch_wavelength',
    'sinusoidal_rugate',
    'apodize',
    'rugate_from_target',
]
