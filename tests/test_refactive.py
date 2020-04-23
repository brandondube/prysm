"""Tests for refractive index computations."""

import pytest

from prysm import refractive

WVL = .587725
C7980_ND = 1.458461


def test_cauchy_accuracy_C7980():
    # from corning datasheet
    coefs = [
        2.104025406E+00,
        -1.456000330E-04,
        -9.049135390E-03,
        8.801830992E-03,
        8.435237228E-05,
        1.681656789E-06,
        -1.675425449E-08,
        8.326602461E-10
    ]
    estimated = refractive.cauchy(WVL, coefs[0], *coefs[1:])
    assert estimated == pytest.approx(C7980_ND, abs=0.05)


def test_sellmeier_accuracy_C7980():
    # from corning datasheet
    As = [
        0.68374049400,
        0.42032361300,
        0.58502748000,
    ]
    Bs = [
        0.00460352869,
        0.01339688560,
        64.49327320000
    ]
    estimated = refractive.sellmeier(WVL, As, Bs)
    assert estimated == pytest.approx(C7980_ND, abs=0.001)
