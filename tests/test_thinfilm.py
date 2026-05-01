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
    indices = [n_MgF2]
    thicknesses = [.150]
    r, _ = thinfilm.multilayer_stack_rt(indices, thicknesses, wvl, 'p', n_C7980)
    R = abs(r)**2
    assert R == pytest.approx(0.022, abs=0.001)  # 98% transmission


def test_accuracy_of_multilayer_reflectivity_on_C7980():
    indices = [n_MgF2, n_ZrO2, n_CeF3]
    thicknesses = [wvl/4, wvl/2, wvl/4]
    r, _ = thinfilm.multilayer_stack_rt(indices, thicknesses, wvl, 's', n_C7980)
    R = abs(r)**2
    assert R == pytest.approx(0.0024, abs=0.0005)  # 99.7% transmission


@pytest.mark.parametrize('pol', ['s', 'p'])
def test_deepstack_loop_same_as_batch(pol):
    thicknesses_mgf2 = np.array([wvl/4, wvl/3, wvl/2])
    looped_Rs = []
    for thick in thicknesses_mgf2:
        indices = [n_MgF2, n_ZrO2, n_CeF3]
        thicknesses = [thick, wvl/2, wvl/4]
        r, _ = thinfilm.multilayer_stack_rt(indices, thicknesses, wvl, pol, n_C7980)
        R = abs(r)**2
        looped_Rs.append(R)

    tm = thicknesses_mgf2
    nmgf2 = np.full(tm.shape, n_MgF2)
    nzro2 = np.full(tm.shape, n_ZrO2)
    n_cef3 = np.full(tm.shape, n_CeF3)
    t_zro2 = np.full(tm.shape, wvl/2)
    t_cef3 = np.full(tm.shape, wvl/4)
    indices = [nmgf2, nzro2, n_cef3]
    thicknesses = [tm, t_zro2, t_cef3]
    r, _ = thinfilm.multilayer_stack_rt(indices, thicknesses, wvl, pol, n_C7980)
    R_vectorized = abs(r)**2
    assert np.allclose(R_vectorized, looped_Rs)


@pytest.mark.parametrize('pol', ['s', 'p'])
def test_deepstack_matches_2D_thickness(pol):
    thicknesses_mgf2 = np.array([wvl/4, wvl/3, wvl/2, wvl/1, wvl/0.5, wvl/0.25]).reshape(2, 3)
    thicknesses_mgf2
    looped_Rs = []
    for thick in thicknesses_mgf2.ravel():
        indices = [n_MgF2, n_ZrO2, n_CeF3]
        thicknesses = [thick, wvl/2, wvl/4]
        r, _ = thinfilm.multilayer_stack_rt(indices, thicknesses, wvl, pol, n_C7980)
        R = abs(r)**2
        looped_Rs.append(R)

    looped_Rs = np.array(looped_Rs).reshape(2, 3)

    tm = thicknesses_mgf2
    nmgf2 = np.full(tm.shape, n_MgF2)
    nzro2 = np.full(tm.shape, n_ZrO2)
    n_cef3 = np.full(tm.shape, n_CeF3)
    t_zro2 = np.full(tm.shape, wvl/2)
    t_cef3 = np.full(tm.shape, wvl/4)
    indices = [nmgf2, nzro2, n_cef3]
    thicknesses = [tm, t_zro2, t_cef3]
    r, _ = thinfilm.multilayer_stack_rt(indices, thicknesses, wvl, pol, n_C7980)
    R_vectorized = abs(r)**2
    assert np.allclose(R_vectorized, looped_Rs)


def test_substrate_index_matches_vectorized_shape():
    thicknesses_mgf2 = np.array([wvl/4, wvl/3, wvl/2])
    nmgf2 = np.full(thicknesses_mgf2.shape, n_MgF2)
    substrate = np.full(thicknesses_mgf2.shape, n_C7980)

    r, _ = thinfilm.multilayer_stack_rt(
        [nmgf2],
        [thicknesses_mgf2],
        wvl,
        's',
        substrate,
    )
    assert r.shape == thicknesses_mgf2.shape


def test_indices_and_thicknesses_must_broadcast():
    with pytest.raises(ValueError, match='indices and thicknesses'):
        thinfilm.multilayer_stack_rt(
            [n_MgF2, n_ZrO2],
            [wvl/4, wvl/2, wvl/4],
            wvl,
            's',
            n_C7980,
        )


def test_brewsters_accuracy():
    ang = thinfilm.brewsters_angle(1, 1.5)
    assert ang == pytest.approx(56.3, abs=1e-2)


def test_critical_accuracy():
    ang = thinfilm.critical_angle(1, 1.5, deg=True)
    assert ang == pytest.approx(41.8, abs=0.02)
