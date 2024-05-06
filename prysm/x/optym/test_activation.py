import numpy as np
from scipy.optimize import approx_fprime
from prysm.x.optym.activation import (
    Tanh,
    Arctan,
    Softplus,
    Sigmoid
)

def test_Tanh_fwd():

    x = np.linspace(-1,1)
    tanh = Tanh()

    truth = np.tanh(x)
    test = tanh.forward(x)

    return np.testing.assert_allclose(truth, test)


def test_Tanh_rev():

    x = np.linspace(-1,1)
    tanh = Tanh()
    truth = []

    for u in x:
        truth.append(approx_fprime(u, np.tanh, 1e-9))
    
    truth = np.array(truth)[...,0]
    test = tanh.backprop(x)

    return np.testing.assert_allclose(truth, test, 1e-6)


def test_Arctan_fwd():

    x = np.linspace(-1,1)
    atan = Arctan()

    truth = np.arctan(x)
    test = atan.forward(x)

    return np.testing.assert_allclose(truth, test)


def test_Arctan_rev():

    x = np.linspace(-1,1)
    atan = Arctan()
    truth = []

    for u in x:
        truth.append(approx_fprime(u, np.arctan, 1e-9))
    
    truth = np.array(truth)[...,0]
    test = atan.backprop(x)

    return np.testing.assert_allclose(truth, test, 1e-6)


def test_Softplus_rev():

    x = np.linspace(-1,1)
    soft = Softplus()
    truth = []

    for u in x:
        truth.append(approx_fprime(u, soft.forward, 1e-9))
    
    truth = np.array(truth)[...,0]
    test = soft.backprop(x)

    return np.testing.assert_allclose(truth, test, 1e-6)


def test_Sigmoid_rev():

    x = np.linspace(-1,1)
    sigm = Sigmoid()
    truth = []

    for u in x:
        truth.append(approx_fprime(u, sigm.forward, 1e-9))
    
    truth = np.array(truth)[...,0]
    test = sigm.backprop(x)

    return np.testing.assert_allclose(truth, test, 1e-6)







