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


@pytest.mark.parametrize('pol', ['s', 'p'])
def test_deepstack_matches_2D_indices_thickness_and_substrate(pol):
    x = np.linspace(0, 1, 12).reshape(3, 4)
    indices = []
    thicknesses = []
    for layer in range(5):
        indices.append(1.35 + 0.12 * layer + 0.01j * layer + 1e-3 * x)
        thicknesses.append(wvl / (4 + layer) * (1 + 0.05 * x))
    substrate = n_C7980 + 0.02 * x

    looped_rs = []
    looped_ts = []
    for idx in np.ndindex(x.shape):
        indices_at_point = [n[idx] for n in indices]
        thicknesses_at_point = [t[idx] for t in thicknesses]
        r, t = thinfilm.multilayer_stack_rt(
            indices_at_point,
            thicknesses_at_point,
            wvl,
            pol,
            substrate[idx],
            aoi=23,
        )
        looped_rs.append(r)
        looped_ts.append(t)

    r, t = thinfilm.multilayer_stack_rt(indices, thicknesses, wvl, pol, substrate, aoi=23)

    assert np.allclose(r, np.asarray(looped_rs).reshape(x.shape))
    assert np.allclose(t, np.asarray(looped_ts).reshape(x.shape))


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


def test_cos_snell_matches_snell_aor_branch_for_tir():
    theta = np.radians(np.array([-60, 0, 60]))
    from_snell = np.cos(thinfilm.snell_aor(1.5, 1, theta, deg=False))
    direct = thinfilm._cos_snell(1.5, 1, theta)

    assert np.allclose(direct, from_snell)


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


# --- explicit 2x2 transfer-matrix path (characteristic_matrix_* /
# multilayer_matrix_* / rtot / ttot).  The optimized multilayer_stack_rt only
# computes the A00/A10 coefficients; these functions build the full matrices and
# must agree with it.

def _full_matrix_rt(indices, thicknesses, wavelength, pol, substrate_index, aoi=0, ambient=1.0):
    """r, t computed via the full 2x2 transfer-matrix functions."""
    char = thinfilm.characteristic_matrix_p if pol == 'p' else thinfilm.characteristic_matrix_s
    multi = thinfilm.multilayer_matrix_p if pol == 'p' else thinfilm.multilayer_matrix_s
    mats = [char(wavelength, d, n, thinfilm.snell_aor(ambient, n, aoi, deg=True))
            for n, d in zip(indices, thicknesses)]
    theta_sub = thinfilm.snell_aor(ambient, substrate_index, aoi, deg=True)
    A = multi(ambient, np.radians(aoi), mats, substrate_index, theta_sub)
    return thinfilm.rtot(A), thinfilm.ttot(A)


@pytest.mark.parametrize('pol', ['s', 'p'])
@pytest.mark.parametrize('aoi', [0, 15, 45])
@pytest.mark.parametrize('layers', [
    ([n_MgF2], [.150]),
    ([n_MgF2, n_ZrO2, n_CeF3], [wvl/4, wvl/2, wvl/4]),
])
def test_full_matrix_path_matches_stack_rt(pol, aoi, layers):
    indices, thicknesses = layers
    r_full, t_full = _full_matrix_rt(indices, thicknesses, wvl, pol, n_C7980, aoi=aoi)
    r_ref, t_ref = thinfilm.multilayer_stack_rt(indices, thicknesses, wvl, pol, n_C7980, aoi=aoi)
    assert np.allclose(r_full, r_ref)
    assert np.allclose(t_full, t_ref)


@pytest.mark.parametrize('pol', ['s', 'p'])
@pytest.mark.parametrize('m', [1, 3])
def test_full_matrix_array_substrate_matches_stack_rt(pol, m):
    # guards the multilayer_matrix_{p,s} batched-term4 branch: an array-like
    # theta_np1 must take the array branch even at length 1.  Gating on
    # len(theta_np1) > 1 (rather than "is array-like") regresses m=1 to a
    # ValueError on inhomogeneous shape -- this pins the working behavior.
    substrate = n_C7980 + 0.05 * np.arange(m)
    r_full, t_full = _full_matrix_rt([n_MgF2], [.150], wvl, pol, substrate)
    r_ref, t_ref = thinfilm.multilayer_stack_rt([n_MgF2], [.150], wvl, pol, substrate)
    assert r_full.shape == (m,)
    assert np.allclose(r_full, r_ref)
    assert np.allclose(t_full, t_ref)
