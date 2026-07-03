"""Adjoint-mode differential ray tracing built on prysm's sequential tracer.

Reverse-mode companion to prysm.x.raytracing._diff_raytrace: one backward sweep
yields the gradient of a scalar merit with respect to every tolerance parameter
at once, the transpose of the forward tangent bundle.  The sweep consumes
seedable design.Merit classes (RmsSpotRadius, WavefrontRMS); assembling several
merits gives the full M x P sensitivity Jacobian for TOR-style tolerance
analysis (sensitivity tables, inverse sensitivity, RSS, compensators) at O(M)
trace cost.
"""

from .primitives import (
    adj_opl_segment,
    adj_transform_global,
    adj_refract,
    adj_reflect,
    adj_intersect,
    adj_transform_local,
    adj_eic_closing,
    adj_eic_closing_full,
    adj_closest_point_on_axis,
)
from .backward_sweep import (
    SurfaceIntermediate,
    TraceIntermediates,
    adjoint_gradient,
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
from .._diff_raytrace import seeds_from_perturbations

__all__ = [
    # primitives
    'adj_opl_segment',
    'adj_transform_global',
    'adj_refract',
    'adj_reflect',
    'adj_intersect',
    'adj_transform_local',
    'adj_eic_closing',
    'adj_eic_closing_full',
    'adj_closest_point_on_axis',
    # sweep
    'SurfaceIntermediate',
    'TraceIntermediates',
    'adjoint_gradient',
    # tolerance analysis
    'AdjointResult',
    'multi_objective_sensitivity',
    'ToleranceSensitivityTable',
    'inverse_sensitivity',
    'rss_prediction',
    'compensated_jacobian',
    'multi_objective_budget',
    # perturbation -> seed bridge
    'seeds_from_perturbations',
]
