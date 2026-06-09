"""OpticalSystem convenience plot methods (layout_2d, plot_spots,
plot_ray_fans, plot_opd_fans) and the live-fingerprint trace cache."""
import matplotlib
import numpy as np
import pytest

matplotlib.use('Agg')
from matplotlib import pyplot as plt

from prysm.x.raytracing import (
    OpticalSystem, ApertureSpec, LensData, Sphere, Plane, Field, Sampling,
)
from prysm.x import materials
from prysm.x.raytracing.analysis import spot_diagrams, ray_aberration_fans


def _doublet():
    ld = (LensData()
          .add(Plane(), typ='eval', thickness=10.0)
          .add(Sphere(1 / 61.47), thickness=6.0,
               material=materials.ConstantMaterial(1.5168), semidiameter=12.0)
          .add(Sphere(-1 / 44.64), thickness=2.5,
               material=materials.ConstantMaterial(1.673), semidiameter=12.0)
          .add(Sphere(-1 / 129.94), thickness=0.0,
               material=materials.air, semidiameter=12.0)
          .add(Plane(), typ='eval'))
    sys = OpticalSystem(ld, aperture=ApertureSpec.epd(22.0),
                        fields=[Field(0, 0), Field(0, 0.7), Field(0, 1.0)],
                        wavelengths=[0.486, 0.587, 0.656], reference=1,
                        stop_index=1)
    sys.solve_image_distance()
    return sys


# ---------- layout_2d -------------------------------------------------------

def test_layout_2d_returns_fig_ax_and_draws_field_fans():
    sys = _doublet()
    fig, ax = sys.layout_2d()
    try:
        assert fig is not None and ax is not None
        # one optics outline plus a fan per field -> more lines than fields
        assert len(ax.lines) > len(sys.fields)
    finally:
        plt.close(fig)


def test_layout_2d_honors_overrides():
    sys = _doublet()
    fig, ax = sys.layout_2d(fields=[Field(0, 0)], sampling=5, axis='y')
    try:
        # a 5-ray single-field fan: 5 ray lines + 1 optics outline
        assert len(ax.lines) == 6
    finally:
        plt.close(fig)


def test_layout_2d_accepts_explicit_sampling_object():
    sys = _doublet()
    fig, ax = sys.layout_2d(sampling=Sampling.fan(n=3, axis='y'))
    try:
        assert len(ax.lines) > 0
    finally:
        plt.close(fig)


# ---------- convenience plot methods ----------------------------------------

def test_plot_spots_returns_fig_axs():
    sys = _doublet()
    fig, axs = sys.plot_spots()
    try:
        assert np.asarray(axs).size == len(sys.fields)
    finally:
        plt.close(fig)


def test_plot_ray_fans_and_opd_fans_return_fig_axs():
    sys = _doublet()
    for method in (sys.plot_ray_fans, sys.plot_opd_fans):
        fig, axs = method()
        try:
            assert np.asarray(axs).shape == (len(sys.fields), 2)
        finally:
            plt.close(fig)


def test_convenience_grid_equals_explicit_two_step():
    sys = _doublet()
    explicit = spot_diagrams(sys)
    cached = sys._cached_grid('spots', spot_diagrams, dict(
        fields=None, wavelengths=None, sampling=None, epd=None,
        reference='centroid'))
    np.testing.assert_allclose(cached.x, explicit.x, equal_nan=True)
    np.testing.assert_allclose(cached.y, explicit.y, equal_nan=True)


# ---------- the fingerprint trace cache -------------------------------------

def test_trace_cache_hits_then_invalidates_on_every_change():
    sys = _doublet()
    kw = dict(fields=None, wavelengths=None, nrays=11, epd=None,
              distribution='uniform', reference='chief')

    g1 = sys._cached_grid('ray_fans', ray_aberration_fans, kw)
    g2 = sys._cached_grid('ray_fans', ray_aberration_fans, kw)
    assert g1 is g2  # unchanged system -> cache hit (same object)

    # a lens edit bumps lens._version -> miss
    sys.lens.rows[1].thickness = 6.5
    g3 = sys._cached_grid('ray_fans', ray_aberration_fans, kw)
    assert g3 is not g2

    # a stop_index reassignment (no version bump) -> miss via live fingerprint
    sys.stop_index = 0
    g4 = sys._cached_grid('ray_fans', ray_aberration_fans, kw)
    assert g4 is not g3
    sys.stop_index = 1

    # an in-place fields mutation -> miss (a version counter would not catch it)
    g5 = sys._cached_grid('ray_fans', ray_aberration_fans, kw)
    sys.fields.fields.append(Field(0, 1.4))
    g6 = sys._cached_grid('ray_fans', ray_aberration_fans, kw)
    assert g6 is not g5

    # a wavelength-set reassignment -> miss
    sys.wavelengths = np.asarray([0.55])
    g7 = sys._cached_grid('ray_fans', ray_aberration_fans, kw)
    assert g7 is not g6


def test_trace_cache_distinguishes_call_arguments():
    sys = _doublet()
    a = sys._cached_grid('ray_fans', ray_aberration_fans, dict(nrays=11))
    b = sys._cached_grid('ray_fans', ray_aberration_fans, dict(nrays=21))
    assert a is not b
    assert a.x.shape[-1] == 11 and b.x.shape[-1] == 21
