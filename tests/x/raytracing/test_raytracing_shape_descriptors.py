"""Shapes are self-describing: their DOF layout and rebuild live on the class.

Each editable Shape declares SCALAR_DOFS / VECTOR_DOFS / META_KEYS / CATEGORIES
and a from_params classmethod; LensData reads those instead of a private adapter
table.  These tests pin two properties:

- round-trip identity: from_params(shape.params) and a LensData add/build_shape
  reproduce the original parameters, and
- adding a shape edits exactly one place -- a toy Shape that declares the layout
  is accepted by LensData with nothing registered elsewhere, while one that does
  not raises a clear TypeError.
"""
import numpy as np
import pytest

from prysm.x.raytracing.lensdata import LensData
from prysm.x.raytracing.surfaces import (
    Shape,
    Biconic,
    Chebyshev,
    Conic,
    EvenAsphere,
    Jacobi,
    OffAxisConic,
    Plane,
    Q2D,
    Sphere,
    Toroid,
    XY,
    Zernike,
)


# one representative instance of every registered shape
SHAPES = [
    Plane(),
    Sphere(1 / 50.0),
    Conic(1 / 50.0, -0.5),
    OffAxisConic(1 / 50.0, -0.5, dx=10.0, dy=5.0),
    EvenAsphere(1 / 50.0, -0.5, (1e-4, 1e-6)),
    Q2D(1 / 50.0, -0.5, 10.0, (0.0, 1e-3), ((1e-4,),), ((0.0,),),
        dx=0.0, dy=0.0),
    Zernike(1 / 50.0, -0.5, 10.0, [(2, 0), (4, 0), (3, 1)],
            (1e-3, 2e-4, 3e-4), norm=True),
    XY(1 / 50.0, -0.5, 10.0, [(2, 0), (0, 2)], (1e-4, 2e-4)),
    Chebyshev(1 / 50.0, -0.5, 10.0, 10.0, [(2, 0), (0, 2)], (1e-4, 2e-4)),
    Jacobi(1 / 50.0, -0.5, 10.0, 0.0, 0.0, [2, 4], (1e-3, 2e-4)),
    Toroid(1 / 50.0, 1 / 40.0, -0.3, (1e-4,)),
    Biconic(1 / 50.0, 1 / 40.0, -0.2, -0.3),
]


def _params_equal(a, b):
    assert set(a) == set(b)
    for key in a:
        va, vb = a[key], b[key]
        if isinstance(va, (int, float)) or np.ndim(va) > 0:
            np.testing.assert_allclose(np.asarray(va, dtype=float),
                                       np.asarray(vb, dtype=float))
        else:
            assert va == vb


@pytest.mark.parametrize('shape', SHAPES, ids=lambda s: type(s).__name__)
def test_from_params_round_trips(shape):
    """from_params(shape.params) reconstructs an equal shape."""
    rebuilt = type(shape).from_params(shape.params or {})
    assert type(rebuilt) is type(shape)
    _params_equal(shape.params or {}, rebuilt.params or {})


@pytest.mark.parametrize('shape', SHAPES, ids=lambda s: type(s).__name__)
def test_lensdata_row_round_trips(shape):
    """A LensData row flattens the DOFs and rebuilds the same shape."""
    ld = LensData()
    ld.add(shape, thickness=5.0)
    rebuilt = ld.rows[1].build_shape()   # rows[0] is the OBJECT endpoint
    assert type(rebuilt) is type(shape)
    _params_equal(shape.params or {}, rebuilt.params or {})


def test_descriptor_categories_reference_real_dofs():
    """Every category key names a declared scalar or vector DOF."""
    for shape in SHAPES:
        kind = type(shape)
        declared = set(kind.SCALAR_DOFS) | set(kind.VECTOR_DOFS)
        for keys in kind.CATEGORIES.values():
            assert set(keys) <= declared


class _ToyParabola(Shape):
    """A user shape that declares its layout and nothing else is registered."""

    SCALAR_DOFS = ('c',)
    CATEGORIES = {'curvature': ['c'], 'radius': ['c']}

    @classmethod
    def from_params(cls, p):
        return cls(p['c'])

    def __init__(self, c):
        super().__init__(c=c)

    def sag(self, x, y):
        return 0.5 * self.params['c'] * (x * x + y * y)


def test_adding_a_shape_edits_one_place():
    """A self-describing shape is accepted by LensData with no other edits."""
    ld = LensData()
    ld.add(_ToyParabola(1 / 25.0), thickness=2.0)
    row = ld.rows[1]                     # rows[0] is the OBJECT endpoint
    assert row.shape_kind is _ToyParabola
    assert 'curvature' in row.categories
    rebuilt = row.build_shape()
    assert isinstance(rebuilt, _ToyParabola)
    np.testing.assert_allclose(float(rebuilt.params['c']), 1 / 25.0)


class _UndescribedShape(Shape):
    """A shape that never declared a DOF layout / from_params."""

    def __init__(self):
        super().__init__()

    def sag(self, x, y):
        return np.zeros_like(x)


def test_undescribed_shape_raises_clear_error():
    """A shape without from_params cannot be carried as a row."""
    ld = LensData()
    with pytest.raises(TypeError, match='not registered with LensData'):
        ld.add(_UndescribedShape())
