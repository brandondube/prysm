"""Q polynomial tests."""
import pytest

import numpy as np

from prysm import qpoly



# def _true_Q00(x):
#     return np.ones_like(x)

# def _true_Q01(x):
#     return (13 - 16 * x) / np.sqrt(19)

# def _true_Q02(x):
#     num = 2 * (29 - 4 *(25 - 19*x)*x)
#     den = np.sqrt(190)
#     return num / den

# def _true_Q03(x):
#     num = 2 * (207 - 4 * x * (315 - (577 - 320 * x)*x))
#     den = np.sqrt(5090)
#     return num / den



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


# here are some truths typed out from Fig. 3 of oe-20-3-2483
# only n=4 is entered, as the expressions become massive
# and n=4 is large enough to guarantee in all cases that the
# recurrence has begun, and thus all elements of the computation
# are performed correctly


def _true_Q04(x):
    num = 7737 - 16 * x * (4653 - 2 * x * (7381 - 8*(1168 - 509*x)*x))
    den = 3 * np.sqrt(131831)
    return num / den


def _true_Q14(x):
    num = 40786 - 64 * x * (9043 - x * (29083 - 4 * (8578 - 3397 * x) * x))
    den = np.sqrt(1078214594)
    return num / den


def _true_Q24(x):
    num = 220853 - 16 * x * (10684 - x * (282609 - 8 * (37233 - 13682 * x)*x))
    den = 7 * np.sqrt(32522114)
    return num / den


def _true_Q34(x):
    num = 691812 - 64 * x * (76131 - x * (180387 - 16 * (11042 - 3849*x)*x))
    den = 3 * np.sqrt(378538886)
    return num / den


def _true_Q44(x):
    num = 8 * (57981 - 4*x * (58806 - 7 * (10791 - 4456*x)*x))
    den = np.sqrt(1436009498)
    return num / den


def _true_Q55(x):
    num = 32 * (16160001 - 35*x * (1778777 - 32 * (68848 - 27669*x)*x))
    den = 5 * np.sqrt(32527771277001)
    return num / den


# mfn = azimuthal order m and function
@pytest.mark.parametrize('mfn', [
    (0, _true_Q04),
    (1, _true_Q14),
    (2, _true_Q24),
    (3, _true_Q34),
    (4, _true_Q44),
    (5, _true_Q55),
])
def test_2d_Q(mfn):
    # u is the radial variable, "rho"
    u = np.linspace(0, 1, 32)

    # x is the variable under the prescribed transformation
    x = u ** 2
    m, fn = mfn
    leading_term = u ** abs(m) * np.cos(m*0)  # theta = 0
    truth = fn(x) * leading_term
    test = qpoly.Q2d(4, m, u, 0)
    assert np.all_close(truth, test, atol=1e-6)
