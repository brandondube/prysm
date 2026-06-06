import pytest

from prysm.x.coatings import RTA, Stack
from prysm.x.materials import ConstantMaterial


def test_coating_stack_prefers_material_nk_protocol():
    lossy = ConstantMaterial('lossy', 1.5, k=0.2)
    stack = Stack([lossy], [0.1], substrate_index=1.5)
    R, T, A = RTA(stack, 0.55, 0.0, 's')
    assert R + T + A[0] == pytest.approx(1.0, abs=1e-12)
    assert A[0] > 0
