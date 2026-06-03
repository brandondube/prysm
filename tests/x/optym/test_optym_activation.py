import numpy as np
from scipy.optimize import approx_fprime
from scipy.special import softmax as sps_softmax
from prysm.x.optym.activation import (
    Softmax,
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


def test_Softmax_fwd():
    # softmax over the trailing axis; leading axes are independent variables
    rng = np.random.default_rng(0)
    x = rng.standard_normal((5, 4))

    out = Softmax().forward(x)

    np.testing.assert_allclose(out, sps_softmax(x, axis=-1))
    np.testing.assert_allclose(out.sum(axis=-1), 1)


def test_Softmax_rev():
    # backprop applies the softmax vector-Jacobian product; compare against a
    # finite-difference Jacobian built independently for each row
    rng = np.random.default_rng(0)
    x = rng.standard_normal((5, 4))
    grad = rng.standard_normal((5, 4))

    sm = Softmax()
    sm.forward(x)
    vjp = sm.backprop(grad)

    for r in range(x.shape[0]):
        J = np.array([
            approx_fprime(x[r], lambda v, i=i: Softmax().forward(v[None, :])[0][i], 1e-7)
            for i in range(x.shape[1])
        ])
        np.testing.assert_allclose(J.T @ grad[r], vjp[r], atol=1e-5)







