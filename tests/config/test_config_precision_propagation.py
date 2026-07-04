"""config.precision propagation through raytracing allocations."""
import numpy as np
import pytest

from prysm.conf import config
from tests.x.raytracing.surface_helpers import conic, plane
from prysm.x.raytracing.spencer_and_murty import raytrace
from prysm.x.raytracing.launch import Field, Sampling, launch
from prysm.x.raytracing.analysis import (
    distortion,
    field_curvature,
    chromatic_focal_shift,
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
from prysm.x.raytracing import LensData, OpticalSystem, raygen
from prysm.x.raytracing.surfaces import Conic, Plane


def _parabola():
    c = -1 / 80.0
    f = 1.0 / (2.0 * c)
    s = conic(c=c, k=-1.0, interaction='refl', P=[0, 0, 0])
    img = plane(interaction='eval', P=[0, 0, f])
    return [s, img]


def _parabola_ld():
    """LensData twin of _parabola."""
    c = -1 / 80.0
    f = abs(1.0 / (2.0 * c))
    lens = LensData()
    (lens.add(Conic(c, -1.0), typ='refl', thickness=f)
         .add(Plane(), typ='eval'))
    return OpticalSystem(lens, aperture=4.0, wavelengths=[0.55e-3])


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
    result = distortion(
        presc, fields, 0.55e-3, epd=10.0,
    )
    expected = _expected_dtype(precision)
    assert result.real_xy.dtype == expected
    assert result.paraxial_xy.dtype == expected
    assert result.percent.dtype == expected


def test_field_curvature_dtype_follows_config_precision(precision):
    presc = _parabola()
    fields = [Field(0.0, 0.0), Field(0.1, 0.0)]
    result = field_curvature(presc, fields, 0.55e-3)
    expected = _expected_dtype(precision)
    assert result.x_fan_z.dtype == expected
    assert result.y_fan_z.dtype == expected


def test_chromatic_focal_shift_dtype_follows_config_precision(precision):
    presc = _parabola()
    _, shift = chromatic_focal_shift(
        presc, wavelengths=[0.486e-3, 0.587e-3, 0.656e-3],
        reference_wavelength=0.587e-3, focus='paraxial')
    assert shift.dtype == _expected_dtype(precision)


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
    # surface 1 is the mirror; surface 0 is the OBJECT endpoint.
    pert = Perturbation.normal(ld, 'curvature', 1, sigma=0.0, name='c0')
    res = monte_carlo(ld, [pert], merit, n_trials=3,
                      seed=42, record_samples=True)
    expected = _expected_dtype(precision)
    assert res.merits.dtype == expected
    assert res.nominals.dtype == expected
    assert res.sampled_x.dtype == expected


# ---------- design ----------------------------------------------------------

def test_problem_residuals_dtype_follows_config_precision(precision):
    ld = _parabola_ld()
    op = RmsSpotRadius(Field(0., 0.), 0.55e-3, Sampling.fan(n=5),
                       target=0.0, weight=1.0)
    ld.opt.vary('curvature', surfaces=1)  # surface 1 = mirror (0 = OBJECT)
    prob = Problem(ld, [op])
    out = prob.residuals(prob.x0())
    assert out.dtype == _expected_dtype(precision)
