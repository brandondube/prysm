"""Tests for the whole-system grid analyses + grid plotters.

These cover the two-step ergonomics that mirror Code V / Zemax: an analysis
function that traces every field and wavelength with one consistent pupil
sampling and returns a labelled namedtuple of arrays, and a plotting function
that lays the grid out as subplots.  The data and the plot stay separate.
"""
import numpy as np
import pytest

import matplotlib
matplotlib.use('Agg')

from prysm.x import materials
from prysm.x.raytracing import LensData, OpticalSystem, ApertureSpec, Field
from prysm.x.raytracing.surfaces import Conic, Plane
from prysm.x.raytracing.analysis import (
    ray_aberration_fans,
    opd_fans,
    spot_diagrams,
    spot_rms_radius,
    spot_geometric_radius,
    RayFanGrid,
    OPDFanGrid,
    SpotGrid,
    Sampling,
)
from prysm.x.raytracing import plotting


# ---------- system builders -------------------------------------------------

def _singlet_system(fields=None, wavelengths=None, ref=1):
    """Sphere/sphere singlet with system metadata (stop at the first surface)."""
    lens = LensData()
    (lens.add(Conic(1 / 50.0, 0.0), typ='refr', material=materials.ConstantMaterial(1.5168),
              thickness=5.0)
         .add(Conic(-1 / 50.0, 0.0), typ='refr', material=materials.air,
              thickness=95.0)
         .add(Plane(), typ='eval'))
    if fields is None:
        fields = [Field(0, 0), Field(0, 3)]
    if wavelengths is None:
        wavelengths = [0.4861, 0.5876, 0.6563]  # F, d, C
    return OpticalSystem(lens, aperture=ApertureSpec.epd(10.0), fields=fields,
                         wavelengths=wavelengths, reference=ref,
                         stop_index=0)


# ---------- ray-aberration fans ---------------------------------------------

def test_ray_fans_shape_and_indexing():
    sys = _singlet_system()
    grid = ray_aberration_fans(sys, nrays=21)
    assert isinstance(grid, RayFanGrid)
    nf, nw, npup = grid.x.shape
    assert nf == len(grid.fields) == 2
    assert nw == len(grid.wavelengths) == 3
    assert npup == grid.pupil_x.shape[-1] == 21
    assert grid.pupil_x.shape == grid.pupil_y.shape == (nf, npup)
    assert grid.y.shape == grid.x.shape


def test_ray_fans_default_metadata_from_system():
    """Omitting fields/wavelengths pulls them from the OpticalSystem."""
    sys = _singlet_system()
    grid = ray_aberration_fans(sys, nrays=11)
    assert len(grid.fields) == 2
    np.testing.assert_allclose(sorted(grid.wavelengths),
                               sorted([0.4861, 0.5876, 0.6563]))


def test_ray_fans_pupil_is_per_field_and_normalized():
    sys = _singlet_system()
    grid = ray_aberration_fans(sys, nrays=21)
    # without vignetting the fans span the normalized pupil rim to rim
    assert grid.pupil_x.min() == pytest.approx(-1.0)
    assert grid.pupil_x.max() == pytest.approx(1.0)
    assert grid.pupil_y.min() == pytest.approx(-1.0)
    assert grid.pupil_y.max() == pytest.approx(1.0)
    # one pupil axis per field (vignetting factors are per-field)
    assert grid.pupil_x.shape == (len(grid.fields), 21)


def test_ray_fans_vignetted_field_spans_less_than_unit_pupil():
    # vignetting factors compress the launched fan onto the transmitted pupil;
    # the abscissa shows the truncation (it is never stretched back to +/-1)
    fields = [Field(0, 0),
              Field(0, 3, vignetting={'vuy': 0.3, 'vly': 0.1})]
    sys = _singlet_system(fields=fields)
    grid = ray_aberration_fans(sys, nrays=21)
    np.testing.assert_allclose(grid.pupil_y[0].max(), 1.0)
    np.testing.assert_allclose(grid.pupil_y[1].max(), 0.7)
    np.testing.assert_allclose(grid.pupil_y[1].min(), -0.9)
    # x is unvignetted for this field
    np.testing.assert_allclose(grid.pupil_x[1].max(), 1.0)
    # the bundle stays full length: every fan value is finite
    assert np.isfinite(grid.y[1]).all()


def test_ray_fans_chief_reference_is_zero():
    """The pupil-center ray's error is exactly zero under chief referencing."""
    sys = _singlet_system()
    grid = ray_aberration_fans(sys, nrays=21, reference='chief')
    ci = int(np.argmin(np.abs(grid.pupil_x[0])))
    assert np.nanmax(np.abs(grid.x[:, :, ci])) == 0.0
    assert np.nanmax(np.abs(grid.y[:, :, ci])) == 0.0


def test_ray_fans_centroid_reference_runs():
    sys = _singlet_system()
    grid = ray_aberration_fans(sys, nrays=15, reference='centroid')
    assert np.isfinite(grid.x).any()


def test_ray_fans_bare_prescription_needs_epd():
    sys = _singlet_system()
    with pytest.raises(TypeError):
        ray_aberration_fans(list(sys.to_surfaces()), fields=[Field(0, 0)],
                            wavelengths=[0.5876], nrays=11)
    grid = ray_aberration_fans(list(sys.to_surfaces()), fields=[Field(0, 0)],
                               wavelengths=[0.5876], nrays=11, epd=8.0)
    assert grid.x.shape == (1, 1, 11)


# ---------- OPD fans --------------------------------------------------------

def test_opd_fans_shape_and_chief_zero():
    sys = _singlet_system()
    grid = opd_fans(sys, nrays=21)
    assert isinstance(grid, OPDFanGrid)
    assert grid.x.shape == (2, 3, 21)
    # OPD is chief-referenced: the central ray is ~0 in every panel
    ci = int(np.argmin(np.abs(grid.pupil_x[0])))
    assert np.nanmax(np.abs(grid.x[:, :, ci])) < 1e-9
    assert np.nanmax(np.abs(grid.y[:, :, ci])) < 1e-9


# ---------- spot diagrams ---------------------------------------------------

def test_spot_diagrams_shape_and_validity():
    sys = _singlet_system()
    grid = spot_diagrams(sys, sampling=Sampling.hex(nrings=4))
    assert isinstance(grid, SpotGrid)
    nf, nw, n = grid.x.shape
    assert (nf, nw) == (2, 3)
    assert grid.valid.shape == grid.x.shape
    assert grid.reference.shape == (2, 3, 2)
    assert grid.valid.all()


def test_spot_reference_recovers_absolute_landing():
    """x + reference gives back the absolute image coordinate."""
    sys = _singlet_system()
    grid = spot_diagrams(sys, sampling=Sampling.hex(nrings=3),
                         reference='centroid')
    absolute = grid.x[..., :] + grid.reference[..., 0:1]
    assert np.isfinite(absolute).all()


def test_spot_rms_matches_manual():
    sys = _singlet_system()
    grid = spot_diagrams(sys, sampling=Sampling.hex(nrings=5))
    rms = spot_rms_radius(grid)
    assert rms.shape == (2, 3)
    # recompute centroid-referenced RMS by hand for one panel
    x = grid.x[0, 0]
    y = grid.y[0, 0]
    xc = x - np.nanmean(x)
    yc = y - np.nanmean(y)
    manual = np.sqrt(np.nanmean(xc * xc + yc * yc))
    assert rms[0, 0] == pytest.approx(manual)


def test_spot_geometric_radius_ge_rms():
    sys = _singlet_system()
    grid = spot_diagrams(sys, sampling=Sampling.hex(nrings=5))
    assert np.all(spot_geometric_radius(grid) >= spot_rms_radius(grid))


def test_spot_diagrams_default_sampling():
    sys = _singlet_system()
    grid = spot_diagrams(sys)
    assert grid.x.shape[0] == 2 and grid.x.shape[1] == 3


# ---------- plotters --------------------------------------------------------

def test_plot_ray_fans_layout():
    sys = _singlet_system()
    grid = ray_aberration_fans(sys, nrays=15)
    fig, axs = plotting.plot_ray_fans(grid)
    assert axs.shape == (2, 2)  # 2 fields x (tangential, sagittal)
    fig2, axs2 = plotting.plot_ray_fans(grid, axes='y')
    assert axs2.shape == (2, 1)


def test_plot_opd_fans_layout():
    sys = _singlet_system()
    grid = opd_fans(sys, nrays=15)
    fig, axs = plotting.plot_opd_fans(grid)
    assert axs.shape == (2, 2)


def test_plot_spot_diagrams_layout():
    sys = _singlet_system(fields=[Field(0, 0), Field(0, 2), Field(0, 4)])
    grid = spot_diagrams(sys, sampling=Sampling.hex(nrings=4))
    fig, axs = plotting.plot_spot_diagrams(grid, ncols=2)
    # 3 fields in 2 columns -> 2 rows, last cell blank
    assert axs.shape == (2, 2)
