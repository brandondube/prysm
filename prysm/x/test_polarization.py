import numpy as np
import prysm.x.polarization as pol

def test_rotation_matrix():

    # Make a 45 degree rotation
    angle = np.pi/4
    control = 1/np.sqrt(2) * np.array([[1,1],[-1,1]])

    test = pol.jones_rotation_matrix(angle)

    np.testing.assert_allclose(control,test)

def test_linear_retarder():

    # Create a quarter-wave plate
    retardance = np.pi/2 # qwp retardance
    control = np.array([[1,0],[0,1j]]) # oriented at 0 deg

    test = pol.linear_retarder(retardance)

    np.testing.assert_allclose(control,test)

def test_linear_diattenuator():

    # Create an imperfect polarizer with a diattenuation of 0.75
    alpha = 0.5
    control = np.array([[1,0],[0,0.5]])

    test = pol.linear_diattenuator(alpha)

    np.testing.assert_allclose(control,test)

def test_half_wave_plate():

    hwp = np.array([[1,0],[0,-1]])
    test = pol.half_wave_plate(0)

    np.testing.assert_allclose(hwp,test)

def test_quarter_wave_plate():

    qwp = np.array([[1,0],[0,1j]])
    test = pol.quarter_wave_plate()

    np.testing.assert_allclose(qwp,test)

def test_linear_polarizer():

    lp = np.array([[1,0],[0,0]])
    test = pol.linear_polarizer()

    np.testing.assert_allclose(lp,test)

def test_jones_to_mueller():

    # Make a circular polarizer
    circ_pol = pol.quarter_wave_plate(theta=np.pi/4)

    mueller_test = pol.jones_to_mueller(circ_pol)/2
    mueller_circ = np.array([[1,0,0,0],
                             [0,0,0,-1],
                             [0,0,1,0],
                             [0,1,0,0]])/2

    np.testing.assert_allclose(mueller_circ,mueller_test,atol=1e-5)

def test_pauli_spin_matrix():

    p0 = np.array([[1,0],[0,1]])
    p1 = np.array([[1,0],[0,-1]])
    p2 = np.array([[0,1],[1,0]])
    p3 = np.array([[0,-1j],[1j,0]])

    np.testing.assert_allclose((p0,p1,p2,p3),
                              (pol.pauli_spin_matrix(0),
                               pol.pauli_spin_matrix(1),
                               pol.pauli_spin_matrix(2),
                               pol.pauli_spin_matrix(3)))
