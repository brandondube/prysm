import numpy as np
import prysm.x.polarization as pol
from prysm.coordinates import make_xy_grid, cart_to_polar
from prysm.geometry import circle


def test_rotation_matrix():
    # Make a 45 degree rotation
    angle = np.pi/4
    control = 1/np.sqrt(2) * np.array(
        [
            [1, 1],
            [-1, 1]
         ]
    )

    test = pol.jones_rotation_matrix(angle)
    assert np.allclose(control, test)


def test_linear_retarder():
    # Create a quarter-wave plate
    retardance = np.pi / 2  # qwp retardance
    control = np.array(
        [
            [1, 0],
            [0, 1j]
        ]
    )  # oriented at 0 deg

    test = pol.linear_retarder(retardance)
    assert np.allclose(control, test)


def test_linear_diattenuator():
    # Create an imperfect polarizer with a diattenuation of 0.75
    alpha = 0.5
    control = np.array(
        [
            [1, 0],
            [0, 0.5]
        ]
    )

    test = pol.linear_diattenuator(alpha)
    assert np.allclose(control, test)


def test_half_wave_plate():
    hwp = np.array(
        [
            [1, 0],
            [0, -1]
        ]
    )
    test = pol.half_wave_plate(0)
    assert np.allclose(hwp, test)


def test_quarter_wave_plate():
    qwp = np.array(
        [
            [1, 0],
            [0, 1j]
        ]
    )
    test = pol.quarter_wave_plate()
    assert np.allclose(qwp, test)


def test_linear_polarizer():
    lp = np.array(
        [
            [1, 0],
            [0, 0]
        ]
    )
    test = pol.linear_polarizer()
    assert np.allclose(lp, test)


def test_jones_to_mueller():

    # Make a circular polarizer
    circ_pol = pol.quarter_wave_plate(theta=np.pi/4)

    mueller_test = pol.jones_to_mueller(circ_pol)/2
    mueller_circ = np.array(
        [
            [1, 0, 0, 0],
            [0, 0, 0, -1],
            [0, 0, 1, 0],
            [0, 1, 0, 0]
        ]
    )/2

    assert np.allclose(mueller_circ, mueller_test, atol=1e-5)


def test_pauli_spin_matrix():
    p0 = np.array(
        [
            [1, 0],
            [0, 1]
        ]
    )
    p1 = np.array(
        [
            [1, 0],
            [0, -1]
        ]
    )
    p2 = np.array(
        [
            [0, 1],
            [1, 0]
        ]
    )
    p3 = np.array(
        [
            [0, -1j],
            [1j, 0]
        ]
    )
    cmp = [pol.pauli_spin_matrix(j) for j in range(4)]
    assert np.allclose((p0, p1, p2, p3), cmp)


def test_make_propagation_polarized():

    # construct a circular aperture
    xi, eta = make_xy_grid(256, diameter=10)
    r, t = cart_to_polar(xi, eta)
    A = circle(5, r)
    wave = 1
    samples = A.shape[0]
    dx = 5/samples

    # create the Jones matrix equivalent
    J = np.zeros([*A.shape, 2, 2])
    J[..., 0, 0] = A
    J[..., 1, 1] = A

    # apply the decorator
    pol.add_jones_propagation()

    # test focus
    from prysm.propagation import (
        focus,
        unfocus,
        angular_spectrum,
        focus_fixed_sampling,
        unfocus_fixed_sampling
    )

    # focus works
    A_psf = focus(A, Q=2)
    J_psf = focus(J, Q=2)

    # unfocus works
    A_pupil = unfocus(A_psf, Q=1)
    J_pupil = unfocus(J_psf, Q=1)

    # angular spectrum
    A_prop = angular_spectrum(A_pupil, wvl=wave, dx=dx, z=5e1, Q=1)
    J_prop = angular_spectrum(J_pupil, wvl=wave, dx=dx, z=5e1, Q=1)

    # focus fixed sampling
    A_psf_fixed = focus_fixed_sampling(A, dx, 50, wave, 1000e-3, 256)
    J_psf_fixed = focus_fixed_sampling(J, dx, 50, wave, 1000e-3, 256)

    # unfocus fixed sampling
    A_pupil_fixed = unfocus_fixed_sampling(A_psf_fixed, 1000e-3/256, 50, wave, dx, samples)  # NOQA
    J_pupil_fixed = unfocus_fixed_sampling(J_psf_fixed, 1000e-3/256, 50, wave, dx, samples)  # NOQA

    slc = (..., 0, 0)
    assert np.allclose(A_psf, J_psf[slc])
    assert np.allclose(A_pupil, J_pupil[slc])
    assert np.allclose(A_prop, J_prop[slc])
    assert np.allclose(A_psf_fixed, J_psf_fixed[slc])
    assert np.allclose(A_pupil_fixed, J_pupil_fixed[slc])
