"""Q polynomial tests."""

import pytest

from prysm import qpoly


@pytest.mark.parametrize('n', [0, 1, 2, 3, 4, 5, 6])
def test_qbfs_functions(n):
    args = {f'A{n}': 1}
    qbfs_sag = qpoly.QBFSSag(**args)
    assert qbfs_sag


@pytest.mark.parametrize('n', [0, 1, 2, 3, 4, 5, 6])
def test_qcon_functions(n):
    args = {f'A{n}': 1}
    qcon_sag = qpoly.QCONSag(**args)
    assert qcon_sag
