"""Tests for thin film calculations."""
import pytest

from prysm import thinfilm
from prysm.mathops import np

wvl = .587725
n_C7980 = 1.458461
n_MgF2 = 1.3698
n_CeF3 = 1.6290 + 1j * 0.0034836
n_ZrO2 = 2.1588


def test_accuracy_of_monolayer_reflectivity_MgF2_on_C7980():
    stack = [
        (n_MgF2, .150),
        (n_C7980, 10_000),
    ]
    r, _ = thinfilm.multilayer_stack_rt(stack, wvl, 'p')
    R = abs(r)**2
    assert R == pytest.approx(0.022, abs=0.001)  # 98% transmission


def test_accuracy_of_multilayer_reflectivity_on_C7980():
    stack = [
        (n_MgF2, wvl/4),
        (n_ZrO2, wvl/2),
        (n_CeF3, wvl/4),
        (n_C7980, 10_000),
    ]
    r, _ = thinfilm.multilayer_stack_rt(stack, wvl, 's')
    R = abs(r)**2
    assert R == pytest.approx(0.0024, abs=0.0005)  # 99.7% transmission


@pytest.mark.parametrize('pol', ['s', 'p'])
def test_deepstack_loop_same_as_batch(pol):
    thicknesses_mgf2 = np.array([wvl/4, wvl/3, wvl/2])
    looped_Rs = []
    for thick in thicknesses_mgf2:
        stack = [
            (n_MgF2, thick),
            (n_ZrO2, wvl/2),
            (n_CeF3, wvl/4),
            (n_C7980, 10_000),
        ]
        r, _ = thinfilm.multilayer_stack_rt(stack, wvl, pol)
        R = abs(r)**2
        looped_Rs.append(R)

    tm = thicknesses_mgf2
    nmgf2 = np.full(tm.shape, n_MgF2)
    nzro2 = np.full(tm.shape, n_ZrO2)
    n_cef3 = np.full(tm.shape, n_CeF3)
    n_c7980 = np.full(tm.shape, n_C7980)
    t_zro2 = np.full(tm.shape, wvl/2)
    t_cef3 = np.full(tm.shape, wvl/4)
    t_c7980 = np.full(tm.shape, 10_000)
    stack = [
        [nmgf2, tm],
        [nzro2, t_zro2],
        [n_cef3, t_cef3],
        [n_c7980, t_c7980],
    ]
    r, _ = thinfilm.multilayer_stack_rt(stack, wvl, pol)
    R_vectorized = abs(r)**2
    assert np.allclose(R_vectorized, looped_Rs)


@pytest.mark.parametrize('pol', ['s', 'p'])
def test_deepstack_matches_2D_thickness(pol):
    thicknesses_mgf2 = np.array([wvl/4, wvl/3, wvl/2, wvl/1, wvl/0.5, wvl/0.25]).reshape(2, 3)
    thicknesses_mgf2
    looped_Rs = []
    for thick in thicknesses_mgf2.ravel():
        stack = [
            (n_MgF2, thick),
            (n_ZrO2, wvl/2),
            (n_CeF3, wvl/4),
            (n_C7980, 10_000),
        ]
        r, _ = thinfilm.multilayer_stack_rt(stack, wvl, pol)
        R = abs(r)**2
        looped_Rs.append(R)

    looped_Rs = np.array(looped_Rs).reshape(2, 3)

    tm = thicknesses_mgf2
    nmgf2 = np.full(tm.shape, n_MgF2)
    nzro2 = np.full(tm.shape, n_ZrO2)
    n_cef3 = np.full(tm.shape, n_CeF3)
    n_c7980 = np.full(tm.shape, n_C7980)
    t_zro2 = np.full(tm.shape, wvl/2)
    t_cef3 = np.full(tm.shape, wvl/4)
    t_c7980 = np.full(tm.shape, 10_000)
    stack = [
        [nmgf2, tm],
        [nzro2, t_zro2],
        [n_cef3, t_cef3],
        [n_c7980, t_c7980],
    ]
    r, _ = thinfilm.multilayer_stack_rt(stack, wvl, pol)
    R_vectorized = abs(r)**2
    assert np.allclose(R_vectorized, looped_Rs)


def test_brewsters_accuracy():
    ang = thinfilm.brewsters_angle(1, 1.5)
    assert ang == pytest.approx(56.3, abs=1e-2)


def test_critical_accuracy():
    ang = thinfilm.critical_angle(1, 1.5, deg=True)
    assert ang == pytest.approx(41.8, abs=0.02)
