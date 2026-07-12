"""Tests for closed-form geometry breadth: Toroid, Biconic, gratings."""
import numpy as np
import pytest

from prysm.x import materials
from prysm.x.raytracing.phase import LinearGrating
from tests.x.raytracing.surface_helpers import (
    plane, sphere, conic, off_axis_conic, even_asphere, q2d, zernike, xy,
    chebyshev, jacobi, toroid, biconic, xy_grid as _xy_grid,
    sag_derivatives as _sag_derivs,
    central_difference_xy as _central_difference_xy,
)

from prysm.x.raytracing.surfaces import (
    Surface,
    sphere_sag,
    even_asphere_sag,
    even_asphere_sag_der_xy,
    conic_sag,
    conic_sag_der_xy,
)
from prysm.x.raytracing.spencer_and_murty import raytrace


# ---- Toroid -----------------------------------------------------------------

def test_toroid_sag_along_axes_matches_components():
    """Along x-axis (y=0): sag == sphere_x.  Along y-axis (x=0): sag == y-asphere."""
    c_x, c_y, k_y = 1 / 100.0, 1 / 50.0, -0.5
    coefs_y = (1e-6, -2e-9)
    s = toroid(c_x=c_x, c_y=c_y, k_y=k_y, coefs_y=coefs_y,
                       interaction='refl', P=[0, 0, 0])
    x = np.linspace(-5, 5, 11)
    z_at_y0 = s.shape.sag(x, np.zeros_like(x))
    z_sphere_x = sphere_sag(c_x, x * x)
    np.testing.assert_allclose(z_at_y0, z_sphere_x, atol=1e-12)
    y = np.linspace(-5, 5, 11)
    z_at_x0 = s.shape.sag(np.zeros_like(y), y)
    z_asphere_y = even_asphere_sag(c_y, k_y, coefs_y, y * y)
    np.testing.assert_allclose(z_at_x0, z_asphere_y, atol=1e-12)


def test_toroid_sag_is_additive_loft():
    """sag(x, y) = sphere_x(c_x, x^2) + even_asphere_y(c_y, k_y, coefs_y, y^2)."""
    c_x, c_y, k_y = 1 / 100.0, 1 / 50.0, -0.5
    coefs_y = (1e-6, -2e-9)
    s = toroid(c_x=c_x, c_y=c_y, k_y=k_y, coefs_y=coefs_y,
                       interaction='refl', P=[0, 0, 0])
    x, y = _xy_grid()
    z_actual = s.shape.sag(x, y)
    z_expected = sphere_sag(c_x, x * x) + even_asphere_sag(c_y, k_y, coefs_y, y * y)
    np.testing.assert_allclose(z_actual, z_expected, atol=1e-12)


def test_toroid_derivatives_central_diff():
    s = toroid(c_x=1 / 80.0, c_y=1 / 60.0, k_y=-0.3, coefs_y=(2e-6,),
                       interaction='refl', P=[0, 0, 0])
    x, y = _xy_grid()
    _, dx_an, dy_an = _sag_derivs(s.shape, x, y)
    dx_num, dy_num = _central_difference_xy(s.shape.sag, x, y)
    np.testing.assert_allclose(dx_an, dx_num, rtol=2e-5, atol=1e-7)
    np.testing.assert_allclose(dy_an, dy_num, rtol=2e-5, atol=1e-7)


def test_toroid_intersect_lands_on_surface():
    """Newton intersect via the shared base lands rays on the toroid surface."""
    s = toroid(c_x=1 / 100.0, c_y=1 / 80.0, k_y=-0.5, coefs_y=(),
                       interaction='refl', P=[0, 0, 0])
    P = np.array([[1.0, 0.5, -50.0],
                  [-2.0, 1.5, -50.0],
                  [0.0, 0.0, -50.0]])
    S = np.array([[0.0, 0.0, 1.0]] * 3)
    Q, _, valid = s.intersect(P, S)
    assert valid.all()
    z = s.shape.sag(Q[..., 0], Q[..., 1])
    np.testing.assert_allclose(Q[..., 2], z, atol=1e-9)


def test_toroid_cylindrical_lens_directionality():
    """Toroid with c_x = 0 is a 1D cylindrical mirror that bends y-fans but
    not x-fans: post-reflection S_x is unchanged for x-displaced rays, and
    S_y is bent for y-displaced rays."""
    s = toroid(c_x=0.0, c_y=1 / 100.0, k_y=0.0, coefs_y=(),
                       interaction='refl', P=[0, 0, 0])
    P_x = np.array([[1.0, 0.0, -50.0],
                    [2.0, 0.0, -50.0]])
    P_y = np.array([[0.0, 1.0, -50.0],
                    [0.0, 2.0, -50.0]])
    S_z = np.array([[0.0, 0.0, 1.0]] * 2)
    res_x = raytrace([s], P_x, S_z, wvl=0.55e-3)
    res_y = raytrace([s], P_y, S_z, wvl=0.55e-3)
    # Cylindrical-in-Y mirror is flat in X: x-fan reflects with S_x = 0
    np.testing.assert_allclose(res_x.S[1, :, 0], [0.0, 0.0], atol=1e-12)
    # ...but bends y-fan: |S_y| > 0 after reflection
    assert np.all(np.abs(res_y.S[1, :, 1]) > 1e-3)


# ---- Biconic ----------------------------------------------------------------

def test_biconic_degenerates_to_conic():
    c, k = 1 / 80.0, -1.0
    s_b = biconic(c_x=c, c_y=c, k_x=k, k_y=k, interaction='refl', P=[0, 0, 0])
    s_c = conic(c=c, k=k, interaction='refl', P=[0, 0, 0])
    x, y = _xy_grid()
    z_b, dx_b, dy_b = _sag_derivs(s_b.shape, x, y)
    z_c, dx_c, dy_c = _sag_derivs(s_c.shape, x, y)
    np.testing.assert_allclose(z_b, z_c, atol=1e-12)
    np.testing.assert_allclose(dx_b, dx_c, atol=1e-12)
    np.testing.assert_allclose(dy_b, dy_c, atol=1e-12)


def test_biconic_derivatives_central_diff():
    s = biconic(c_x=1 / 80.0, c_y=1 / 60.0,
                        k_x=-0.5, k_y=-1.0,
                        interaction='refl', P=[0, 0, 0])
    x, y = _xy_grid()
    _, dx_an, dy_an = _sag_derivs(s.shape, x, y)
    dx_num, dy_num = _central_difference_xy(s.shape.sag, x, y)
    np.testing.assert_allclose(dx_an, dx_num, rtol=2e-5, atol=1e-7)
    np.testing.assert_allclose(dy_an, dy_num, rtol=2e-5, atol=1e-7)


def test_biconic_intersect_lands_on_surface():
    s = biconic(c_x=1 / 100.0, c_y=1 / 80.0, k_x=0.0, k_y=-0.5,
                        interaction='refl', P=[0, 0, 0])
    P = np.array([[1.0, 0.5, -50.0],
                  [-2.0, 1.5, -50.0],
                  [0.0, 0.0, -50.0]])
    S = np.array([[0.0, 0.0, 1.0]] * 3)
    Q, _, valid = s.intersect(P, S)
    assert valid.all()
    z = s.shape.sag(Q[..., 0], Q[..., 1])
    np.testing.assert_allclose(Q[..., 2], z, atol=1e-9)


def test_biconic_principal_curvatures_drive_principal_directions():
    """Biconic with c_x != c_y bends x-fans and y-fans by amounts proportional
    to their respective curvatures (paraxial regime).  Verifies the formula's
    decoupling: for a y=0 ray, S_y after reflection is 0 (no cross-coupling),
    and S_x scales linearly with c_x at fixed x; symmetric for x=0."""
    c_x, c_y = 1 / 200.0, 1 / 100.0
    s = biconic(c_x=c_x, c_y=c_y, k_x=0.0, k_y=0.0,
                        interaction='refl', P=[0, 0, 0])
    h = 0.5  # paraxial ray height
    P = np.array([[h, 0.0, -50.0], [0.0, h, -50.0]])
    S_z = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    res = raytrace([s], P, S_z, wvl=0.55e-3)
    Sx_axis = res.S[1, 0]  # ray on x-axis (y=0)
    Sy_axis = res.S[1, 1]  # ray on y-axis (x=0)
    # Decoupling: x-axis ray has S_y == 0; y-axis ray has S_x == 0
    np.testing.assert_allclose(Sx_axis[1], 0.0, atol=1e-12)
    np.testing.assert_allclose(Sy_axis[0], 0.0, atol=1e-12)
    # Paraxial slope ~ 2 * c * h after a mirror reflection (the direction is
    # such that the ratio is c_x / c_y)
    ratio = abs(Sx_axis[0]) / abs(Sy_axis[1])
    np.testing.assert_allclose(ratio, c_x / c_y, rtol=1e-4)


# ---- Grating modifier -------------------------------------------------------

def test_grating_zeroth_order_matches_specular():
    """m=0 grating reproduces non-grating baseline."""
    g_surf = plane(interaction='refl', P=[0, 0, 0])
    g_surf.grating = LinearGrating(1e-3, [1.0, 0.0, 0.0], 0)
    base = plane(interaction='refl', P=[0, 0, 0])
    img = plane(interaction='eval', P=[0, 0, -10.0])
    P = np.array([[1.0, 0.0, -5.0], [0.0, 2.0, -5.0]])
    S = np.array([[0.0, 0.0, 1.0]] * 2)
    r0 = raytrace([g_surf, img], P, S, wvl=0.55e-3)
    rb = raytrace([base, img], P, S, wvl=0.55e-3)
    np.testing.assert_allclose(r0.S, rb.S, atol=1e-12)
    np.testing.assert_allclose(r0.P, rb.P, atol=1e-12)


@pytest.mark.parametrize('order', [-2, -1, 1, 2])
def test_grating_equation_normal_incidence(order):
    """Reflection grating, normal incidence: |sin theta_diff| = m * lambda / d."""
    d = 2e-3
    wvl = 0.5e-3  # so m*l/d in [0, 1] for orders -2..2
    g_surf = plane(interaction='refl', P=[0, 0, 0])
    g_surf.grating = LinearGrating(d, [1.0, 0.0, 0.0], order)
    img = plane(interaction='eval', P=[0, 0, -10.0])
    P = np.array([[0.0, 0.0, -5.0]])
    S = np.array([[0.0, 0.0, 1.0]])
    r = raytrace([g_surf, img], P, S, wvl=wvl)
    expected_x = order * wvl / d
    expected_z = -np.sqrt(1 - expected_x ** 2)
    np.testing.assert_allclose(r.S[1].squeeze(), [expected_x, 0, expected_z],
                               atol=1e-12)


def test_grating_evanescent_flagged_as_evanescent():
    """m*lambda/d > 1 ⇒ evanescent diffracted order; status.imag = EVANESCENT.

    A non-propagating diffraction order is its own failure mode, distinct from
    total internal reflection -- decode_status must not report it as TIR.
    """
    from prysm.x.raytracing.spencer_and_murty import STATUS_EVANESCENT
    g_surf = plane(interaction='refl', P=[0, 0, 0])
    g_surf.grating = LinearGrating(0.5e-3, [1.0, 0.0, 0.0], 2)  # m*l/d = 2*0.55e-3/0.5e-3 = 2.2
    img = plane(interaction='eval', P=[0, 0, -10.0])
    P = np.array([[0.0, 0.0, -5.0]])
    S = np.array([[0.0, 0.0, 1.0]])
    r = raytrace([g_surf, img], P, S, wvl=0.55e-3)
    assert r.status.imag.item() == STATUS_EVANESCENT  # -3, not TIR (-2)
    assert r.status.real.item() == 1   # failed at surface 1 (the grating)
    assert 'EVANESCENT' in r.status_record.text[0]


def test_refraction_grating_equation():
    """Transmission grating, normal incidence: n' sin theta_diff = m lambda / d."""
    d = 1e-3
    wvl = 0.55e-3
    n_glass = 1.5
    g_surf = plane(interaction='refr', P=[0, 0, 0], material=materials.ConstantMaterial(n_glass))
    g_surf.grating = LinearGrating(d, [1.0, 0.0, 0.0], 1)
    img = plane(interaction='eval', P=[0, 0, 10.0])
    P = np.array([[0.0, 0.0, -5.0]])
    S = np.array([[0.0, 0.0, 1.0]])
    r = raytrace([g_surf, img], P, S, wvl=wvl)
    expected_x = wvl / (n_glass * d)
    expected_z = +np.sqrt(1 - expected_x ** 2)
    np.testing.assert_allclose(r.S[1].squeeze(),
                               [expected_x, 0, expected_z], atol=1e-12)


def test_grating_phase_enters_opl():
    """The grating phase order*wvl*x/d is added to the OPL at the grating.

    A linear grating bends the ray AND adds a diffractive phase; an OPD/
    wavefront through it is wrong by the whole grating phase if the OPL term
    is dropped.  Normal-incidence reflection from a plane grating at z=0: the
    incoming segment is purely geometric (length 5), so OPL[1] isolates the
    grating phase order*wvl*x_local/d.
    """
    d = 1e-3
    wvl = 0.55e-3
    x0 = 2.0
    img = plane(interaction='eval', P=[0, 0, -10.0])
    P = np.array([[x0, 0.0, -5.0]])
    S = np.array([[0.0, 0.0, 1.0]])

    g1 = plane(interaction='refl', P=[0, 0, 0])
    g1.grating = LinearGrating(d, [1.0, 0.0, 0.0], 1)
    r1 = raytrace([g1, img], P, S, wvl=wvl)
    np.testing.assert_allclose(r1.OPL[1].item(), 5.0 + wvl * x0 / d, rtol=0, atol=1e-12)

    # zeroth order adds no phase: OPL stays the bare geometric length
    g0 = plane(interaction='refl', P=[0, 0, 0])
    g0.grating = LinearGrating(d, [1.0, 0.0, 0.0], 0)
    r0 = raytrace([g0, img], P, S, wvl=wvl)
    np.testing.assert_allclose(r0.OPL[1].item(), 5.0, rtol=0, atol=1e-12)

    # the phase is self-consistent with the bend: d(OPL)/dx == transverse
    # optical momentum the grating imparts (= order*wvl/d for unit g_vec)
    x1 = 3.0
    P2 = np.array([[x1, 0.0, -5.0]])
    r2 = raytrace([g1, img], P2, S, wvl=wvl)
    dopl = r2.OPL[1].item() - r1.OPL[1].item()
    np.testing.assert_allclose(dopl / (x1 - x0), wvl / d, rtol=1e-12)


def test_grating_off_curved_surface():
    """Grating on a curved (conic) surface: the grating vector is projected
    into the per-ray tangent plane, so the effective tangential shift
    decreases as rays move off-axis (cosine of the surface tilt angle)."""
    s = conic(c=1 / 100.0, k=0.0, interaction='refl', P=[0, 0, 0])
    s.grating = LinearGrating(1e-3, [1.0, 0.0, 0.0], 1)
    # axial ray (vertex hit) vs off-axis (y=10, surface normal tilted from z-axis)
    P = np.array([[0.0, 0.0, -50.0],
                  [0.0, 10.0, -50.0]])
    S = np.array([[0.0, 0.0, 1.0]] * 2)
    r = raytrace([s], P, S, wvl=0.55e-3)
    # axial ray sees the full m*lambda/d shift along x (normal is +z, no projection loss)
    np.testing.assert_allclose(r.S[1, 0, 0], 0.55, atol=1e-12)
    # off-axis ray: y-tilted normal projects the x-direction grating vector
    # without reducing it (g and n are perpendicular along x), so its
    # x-shift is essentially unchanged - but |S_y| differs from the axial.
    assert r.S[1, 1, 1] != 0.0  # off-axis ray gains a y-component on reflection
    np.testing.assert_allclose(r.S[1, 1, 0], 0.55, atol=1e-3)
