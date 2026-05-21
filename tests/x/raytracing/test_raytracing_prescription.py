import pytest
import numpy as np
import matplotlib

matplotlib.use('Agg')

from matplotlib import pyplot as plt

from prysm.x.raytracing import FRAUNHOFER_LINES_UM, Prescription
from prysm.x.raytracing import materials


def n_bk7(wvl):
    return 1.5168


def make_singlet():
    return Prescription.refractive_sequence(
        radii=[102.0, -102.0],
        thicknesses=[6.0],
        materials=[n_bk7, materials.air],
        semidiameter=10.0,
        epd=20.0,
        fields=[0],
        wavelengths=FRAUNHOFER_LINES_UM,
        reference_wavelength='d',
    )


def test_refractive_sequence_builds_list_like_prescription():
    rx = make_singlet()
    assert len(rx) == 2
    assert list(rx) == rx.surfaces
    assert rx[0].params['c'] == pytest.approx(1 / 102.0)
    assert rx.wavelength('d') == pytest.approx(FRAUNHOFER_LINES_UM['d'])
    assert rx.field(0).hy == pytest.approx(0.0)


def test_paraxial_image_solve_adds_eval_surface():
    rx = make_singlet()
    z = rx.solve_paraxial_image()
    assert len(rx) == 3
    assert rx.image_surface is rx[-1]
    assert rx[-1].P[2] == pytest.approx(z)


def test_trace_and_rms_spot_use_defaults():
    rx = make_singlet()
    rx.solve_paraxial_image()
    trace = rx.trace(sampling='fan', n=5)
    assert trace.P.shape[1] == 5
    assert rx.rms_spot(sampling='fan', n=5) >= 0.0


def test_wave_aberration_fan_can_return_nm():
    rx = make_singlet()
    rx.solve_paraxial_image()

    coord_waves, opd_waves = rx.wave_aberration_fan(n=5, units='waves')
    coord_nm, opd_nm = rx.wave_aberration_fan(n=5, units='nm')

    np.testing.assert_allclose(coord_nm, coord_waves)
    np.testing.assert_allclose(opd_nm, opd_waves * rx.wavelength('d') * 1e3)


def test_plot_wave_aberration_fan_can_use_nm_and_detrend():
    rx = make_singlet()
    rx.solve_paraxial_image()

    fig, ax = rx.plot_wave_aberration_fan(n=5, units='nm', detrend=True)
    try:
        assert ax.get_ylabel() == 'OPD [nm]'
    finally:
        plt.close(fig)


def test_set_focal_length_scales_starting_design():
    rx = make_singlet()
    rx.set_focal_length(100.0)
    assert rx.efl() == pytest.approx(100.0, rel=1e-6)
