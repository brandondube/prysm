"""Adjoint-mode differential ray tracing built on prysm's sequential tracer.

Reverse-mode companion to prysm.x.raytracing._diff_raytrace: one backward sweep
yields the gradient of a scalar merit with respect to every tolerance parameter
at once, the transpose of the forward tangent bundle.  Assembling several merit
heads gives the full M x P sensitivity Jacobian for TOR-style tolerance analysis
(sensitivity tables, inverse sensitivity, RSS, compensators) at O(M) trace cost.
"""

from .primitives import (
    adj_opl_segment,
    adj_transform_global,
    adj_refract,
    adj_reflect,
    adj_intersect,
    adj_transform_local,
    adj_intersect_reference_sphere,
    adj_intersect_reference_sphere_full,
    adj_closest_point_on_axis,
)
from .backward_sweep import (
    SurfaceIntermediate,
    TraceIntermediates,
    adjoint_gradient,
)
from .merit_heads import (
    RmsSpotSizeSeed,
    DistortionSeed,
    RmsWfeSeed,
)
from .tolerance_analysis import (
    AdjointResult,
    multi_objective_sensitivity,
    ToleranceSensitivityTable,
    inverse_sensitivity,
    rss_prediction,
    compensated_jacobian,
    multi_objective_budget,
)

__all__ = [
    # primitives
    'adj_opl_segment',
    'adj_transform_global',
    'adj_refract',
    'adj_reflect',
    'adj_intersect',
    'adj_transform_local',
    'adj_intersect_reference_sphere',
    'adj_intersect_reference_sphere_full',
    'adj_closest_point_on_axis',
    # sweep
    'SurfaceIntermediate',
    'TraceIntermediates',
    'adjoint_gradient',
    # merit heads
    'RmsSpotSizeSeed',
    'DistortionSeed',
    'RmsWfeSeed',
    # tolerance analysis
    'AdjointResult',
    'multi_objective_sensitivity',
    'ToleranceSensitivityTable',
    'inverse_sensitivity',
    'rss_prediction',
    'compensated_jacobian',
    'multi_objective_budget',
]
