"""Guard: config.precision must reach allocations inside raytracing flows.

The mathops backend sweep landed config.precision-aware dtype= kwargs across
analysis / tolerance / launch / paraxial / sensitivity / design.  This test
exercises a handful of representative flows at both precisions and asserts
the dtype of the returned arrays matches `config.precision`.

The fixture restores config.precision=64 on teardown so the rest of the test
suite runs at the package default.
"""
import numpy as np
import pytest

from prysm.conf import config
from tests.x.raytracing.surface_helpers import conic, plane
from prysm.x.raytracing.spencer_and_murty import raytrace
from prysm.x.raytracing.launch import Field, Sampling, launch
from prysm.x.raytracing.analysis import (
    distortion,
    field_curvature,
    axial_color,
    lateral_color,
)
from prysm.x.raytracing.tolerance import (
    Perturbation,
    monte_carlo,
)
from prysm.x.raytracing.design import (
    RmsSpotRadius,
    Problem,
)
from prysm.x.raytracing import LensData, raygen
from prysm.x.raytracing.surfaces import ConicSag, PlaneSag


def _parabola():
    c = -1 / 80.0
    f = 1.0 / (2.0 * c)
    s = conic(c=c, k=-1.0, typ='refl', P=[0, 0, 0])
    img = plane(typ='eval', P=[0, 0, f])
    return [s, img]


def _parabola_ld():
    """LensData twin of _parabola — concave mirror, image folded to z=-40."""
    c = -1 / 80.0
    f = abs(1.0 / (2.0 * c))
    return (LensData(epd=4.0, wavelengths=[0.55e-3])
            .add(ConicSag(c, -1.0), typ='refl', thickness=f)
            .add(PlaneSag(), typ='eval'))


@pytest.fixture(params=[32, 64])
def precision(request):
    old = config.precision
    config.precision = request.param
    try:
        yield request.param
    finally:
        config.precision = old


def _expected_dtype(precision):
    return np.float32 if precision == 32 else np.float64


# ---------- analysis --------------------------------------------------------

def test_distortion_dtype_follows_config_precision(precision):
    presc = _parabola()
    fields = [Field(0.0, 0.0), Field(0.1, 0.0)]
    real_xy, paraxial_xy, percent = distortion(
        presc, fields, 0.55e-3, epd=10.0,
    )
    expected = _expected_dtype(precision)
    assert real_xy.dtype == expected
    assert paraxial_xy.dtype == expected
    assert percent.dtype == expected


def test_field_curvature_dtype_follows_config_precision(precision):
    presc = _parabola()
    fields = [Field(0.0, 0.0), Field(0.1, 0.0)]
    sag_z, tan_z = field_curvature(presc, fields, 0.55e-3, epd=10.0)
    expected = _expected_dtype(precision)
    assert sag_z.dtype == expected
    assert tan_z.dtype == expected


def test_axial_color_dtype_follows_config_precision(precision):
    presc = _parabola()
    out = axial_color(presc, wavelengths=[0.486e-3, 0.587e-3, 0.656e-3])
    assert out.dtype == _expected_dtype(precision)


def test_lateral_color_dtype_follows_config_precision(precision):
    presc = _parabola()
    out = lateral_color(presc, [Field(0.1, 0.0)],
                        wavelengths=[0.486e-3, 0.587e-3], epd=10.0)
    assert out.dtype == _expected_dtype(precision)


# ---------- launch / raygen -------------------------------------------------

def test_sampling_chief_dtype_follows_config_precision(precision):
    pupil_xy = Sampling.chief().build(extent=1.0)
    expected = _expected_dtype(precision)
    assert pupil_xy.shape == (1, 2)
    assert pupil_xy.dtype == expected


def test_raygen_rect_S_dtype_follows_config_precision(precision):
    P, S = raygen.generate_collimated_rect_ray_grid(
        nrays=3, maxx=1.0, z=0.0,
    )
    # raygen returns broadcasted S; the underlying buffer carries dtype.
    assert S.dtype == _expected_dtype(precision)


# ---------- tolerance -------------------------------------------------------

def test_monte_carlo_merits_dtype_follows_config_precision(precision):
    """MonteCarloResult.merits / nominals dtype mirrors config.precision."""
    ld = _parabola_ld()

    def merit(p):
        return 0.0

    # one trivial perturbation; sigma=0 so each trial reports nominal.
    pert = Perturbation.normal(ld, 'curvature', 0, sigma=0.0, name='c0')
    res = monte_carlo(ld, [pert], merit, n_trials=3,
                      seed=42, record_samples=True)
    expected = _expected_dtype(precision)
    assert res.merits.dtype == expected
    assert res.nominals.dtype == expected
    assert res.sampled_x.dtype == expected


# ---------- design ----------------------------------------------------------

def test_problem_residuals_dtype_follows_config_precision(precision):
    ld = _parabola_ld()
    P, S = launch(ld, Field(0., 0.), 0.55e-3,
                  Sampling.fan(n=5), epd=4.0, pupil_z=-10.0)
    op = RmsSpotRadius(P=P, S=S, wavelength=0.55e-3, target=0.0, weight=1.0)
    ld.vary('curvature', surfaces=0)
    prob = Problem(ld, [op])
    out = prob.residuals(prob.x0())
    assert out.dtype == _expected_dtype(precision)
