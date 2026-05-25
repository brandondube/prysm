"""Phase 2 tests: the backend-pure R_rh rotation builder and tensor-clean
shape reconstruction (autograd through the shape constructors)."""

import itertools

import numpy as np
import pytest

from prysm.coordinates import make_rotation_matrix
from prysm.x.raytracing.lensdata import R_rh
from prysm.x.raytracing.surfaces import (
    ConicSag,
    EvenAsphereSag,
    ZernikeSag,
)


# ---------------------------------------------------------------------------
# R_rh
# ---------------------------------------------------------------------------

def test_R_rh_matches_make_rotation_matrix():
    for rz, ry, rx in itertools.product([0, 5, -12, 30, 90], repeat=3):
        got = np.asarray(R_rh(rz, ry, rx))
        ref = make_rotation_matrix((rz, ry, rx))
        np.testing.assert_array_equal(got, ref)


def test_R_rh_identity_at_zero():
    np.testing.assert_allclose(np.asarray(R_rh(0, 0, 0)), np.eye(3))


def test_R_rh_radians_path():
    np.testing.assert_allclose(
        np.asarray(R_rh(np.pi / 4, 0, 0, radians=True)),
        make_rotation_matrix((45, 0, 0)),
    )


def test_R_rh_is_orthonormal():
    R = np.asarray(R_rh(11.0, -23.0, 47.0))
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(R) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# tensor-clean shapes (no host coercion of numeric DOFs)
# ---------------------------------------------------------------------------

def test_shape_ctors_do_not_float_coerce_coefs():
    # the constructors must store coef DOFs verbatim, never calling float() on
    # them -- float() on a torch tensor detaches it from the autograd graph.
    class _NoFloat:
        def __init__(self, v):
            self.v = v

        def __float__(self):
            raise AssertionError('coef was float()-coerced by the constructor')

    markers = [_NoFloat(1.0), _NoFloat(2.0)]
    z = ZernikeSag(0.0, 0.0, 10.0, [(2, 0), (4, 0)], markers)
    assert z.params['coefs'][0] is markers[0]
    assert z.params['coefs'][1] is markers[1]


@pytest.fixture
def torch_backend():
    torch = pytest.importorskip('torch')
    from prysm.mathops import set_backend_to_pytorch, set_backend_to_defaults
    set_backend_to_pytorch()
    try:
        yield torch
    finally:
        set_backend_to_defaults()


def test_conic_sag_differentiable_through_ctor(torch_backend):
    torch = torch_backend
    c = torch.tensor(1 / 50.0, requires_grad=True)
    k = torch.tensor(-0.5, requires_grad=True)
    shape = ConicSag(c, k)
    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([0.5, 1.0, 1.5])
    shape.sag(x, y).sum().backward()
    assert c.grad is not None
    assert k.grad is not None


def test_even_asphere_coefs_differentiable_through_ctor(torch_backend):
    torch = torch_backend
    c = torch.tensor(1 / 80.0, requires_grad=True)
    coefs = torch.tensor([1e-4, -2e-6], requires_grad=True)
    shape = EvenAsphereSag(c, torch.tensor(0.0), coefs)
    x = torch.tensor([1.0, 2.0])
    y = torch.tensor([0.5, 1.0])
    shape.sag(x, y).sum().backward()
    assert coefs.grad is not None


def test_R_rh_differentiable(torch_backend):
    torch = torch_backend
    rz = torch.tensor(15.0, requires_grad=True)
    R = R_rh(rz, torch.tensor(0.0), torch.tensor(0.0))
    R.sum().backward()
    assert rz.grad is not None
