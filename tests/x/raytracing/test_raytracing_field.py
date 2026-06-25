"""Tests for the complex pupil field bridge."""
import numpy as np
import pytest

from scipy.special import j1

from prysm import thinfilm
from prysm.x import materials
from tests.x.raytracing.surface_helpers import plane, conic
from prysm.x.raytracing.spencer_and_murty import raytrace
from prysm.x.raytracing.launch import Field, Sampling, launch
from prysm.x.raytracing.surfaces import annular_aperture
from prysm.x.raytracing import opt, field


# ---------- shared fixtures -------------------------------------------------

def _slow_parabola(epd_focal_ratio=50.0):
    """A slow, on-axis-stigmatic parabola (apodization ~1, Airy PSF)."""
    c = -1 / 400.0
    f = 1.0 / (2.0 * c)            # -200
    presc = [conic(c=c, k=-1.0, interaction='refl', P=[0, 0, 0]),
             plane(interaction='eval', P=[0, 0, f])]
    return presc, abs(f)


def _fast_singlet():
    """A fast equiconvex singlet, heavy spherical aberration."""
    ng = materials.ConstantMaterial(1.5)
    s1 = conic(c=1 / 20.0, k=0.0, interaction='refr', P=[0, 0, 0],
               material=ng)
    s2 = conic(c=-1 / 20.0, k=0.0, interaction='refr', P=[0, 0, 4.0],
               material=materials.air)
    img = plane(interaction='eval', P=[0, 0, 23.0])
    return [s1, s2, img]


# ---------- the fresnel_rp latent-bug fix ----------------------------------

def test_fresnel_rp_equals_rs_at_normal_incidence():
    """At normal incidence the s/p basis coincide; |r_s| must equal |r_p|."""
    n0, n1 = 1.0, 1.5
    rs = thinfilm.fresnel_rs(n0, n1, 0.0, 0.0)
    rp = thinfilm.fresnel_rp(n0, n1, 0.0, 0.0)
    assert abs(abs(rs) - abs(rp)) < 1e-12
    assert abs(abs(rp) - 0.2) < 1e-12


def test_fresnel_energy_conservation_p_pol():
    """R_p + T_p == 1 for a lossless dielectric interface at oblique angle."""
    n0, n1 = 1.0, 1.5
    th0 = np.radians(40.0)
    th1 = np.arcsin(n0 / n1 * np.sin(th0))
    rp = thinfilm.fresnel_rp(n0, n1, th0, th1)
    tp = thinfilm.fresnel_tp(n0, n1, th0, th1)
    oblique = (n1 * np.cos(th1)) / (n0 * np.cos(th0))
    R = abs(rp) ** 2
    T = oblique * abs(tp) ** 2
    assert abs(R + T - 1.0) < 1e-12


# ---------- surface normals / incidence cosine -----------------------------

def _flat_refractor(angle_deg=0.0):
    """Flat refracting interface n=1->1.5 at z=0, eval plane downstream."""
    s1 = plane(interaction='refr', P=[0, 0, 0], material=materials.ConstantMaterial(1.5))
    img = plane(interaction='eval', P=[0, 0, 10.0])
    return [s1, img]


def test_surface_normals_incidence_matches_field_angle():
    """A collimated bundle at angle a onto a flat plane lands at cos(a)."""
    presc = _flat_refractor()
    wvl = 0.55e-3
    angle = 15.0
    P, S = launch(presc, Field(0.0, angle, kind='angle'), wvl,
                  Sampling.rect(n=5), epd=4.0, pupil_z=-5.0)
    tr = raytrace(presc, P, S, wvl)
    cosI, n0, n1, typ = field.surface_normals_from_trace(presc, tr, wvl)
    # surface 0 is the refractor; incidence cosine magnitude == cos(15 deg)
    assert np.allclose(np.abs(cosI[0]), np.cos(np.radians(angle)), atol=1e-9)
    assert n0[0] == pytest.approx(1.0)
    assert n1[0] == pytest.approx(1.5)


# ---------- unpolarized scalar amplitude -----------------------------------

def test_unpolarized_amplitude_mirror_is_lossless():
    """A bare mirror does not attenuate the scalar amplitude."""
    c = -1 / 80.0
    f = 1.0 / (2.0 * c)
    presc = [conic(c=c, k=-1.0, interaction='refl', P=[0, 0, 0]),
             plane(interaction='eval', P=[0, 0, f])]
    wvl = 0.55e-3
    P, S = launch(presc, Field(0., 0.), wvl, Sampling.rect(n=7),
                  epd=10.0, pupil_z=-50.0)
    tr = raytrace(presc, P, S, wvl)
    amp = field.unpolarized_amplitude(presc, tr, wvl)
    assert np.allclose(amp, 1.0, atol=1e-12)


def test_unpolarized_amplitude_normal_incidence_fresnel():
    """On-axis chief through one flat n=1->1.5 face: amp == sqrt(T_normal)."""
    presc = _flat_refractor()
    wvl = 0.55e-3
    P, S = launch(presc, Field(0., 0.), wvl, Sampling.chief(),
                  epd=4.0, pupil_z=-5.0)
    tr = raytrace(presc, P, S, wvl)
    amp = field.unpolarized_amplitude(presc, tr, wvl)
    R = ((1.0 - 1.5) / (1.0 + 1.5)) ** 2
    assert amp[0] == pytest.approx(np.sqrt(1.0 - R), abs=1e-9)


# ---------- geometric apodization (sine space) -----------------------------

def test_apodization_identity_mapping_is_uniform():
    """If the sphere footprint equals the entrance grid, apodization is flat."""
    x = np.linspace(-1, 1, 11)
    a, b = np.meshgrid(x, x)
    entrance = np.stack([a, b], axis=-1)
    amp = field.amplitude_apodization(entrance, entrance.copy())
    assert np.allclose(amp, amp[5, 5])


def test_apodization_uniform_magnification_scales_inverse():
    """A bundle magnified by m in area dims drops amplitude to 1/m."""
    x = np.linspace(-1, 1, 11)
    a, b = np.meshgrid(x, x)
    entrance = np.stack([a, b], axis=-1)
    sphere = entrance * 2.0   # 2x in each axis -> area x4
    amp = field.amplitude_apodization(entrance, sphere)
    # detJ = 4 -> amp = 1/2, uniform
    assert np.allclose(amp, 0.5, atol=1e-12)


def test_apodization_masks_invalid_rays():
    x = np.linspace(-1, 1, 11)
    a, b = np.meshgrid(x, x)
    entrance = np.stack([a, b], axis=-1)
    valid = np.ones((11, 11), dtype=bool)
    valid[0, 0] = False
    amp = field.amplitude_apodization(entrance, entrance.copy(), valid=valid)
    assert amp[0, 0] == 0.0
    assert amp[5, 5] > 0.0


def test_apodization_nan_neighbor_does_not_zero_valid_rays():
    """A NaN sphere sample must not zero its valid neighbors."""
    x = np.linspace(-1, 1, 11)
    a, b = np.meshgrid(x, x)
    entrance = np.stack([a, b], axis=-1)
    sphere = entrance.copy()
    sphere[3, 7, :] = np.nan          # one interior ray missed
    amp = field.amplitude_apodization(entrance, sphere)
    for r, c in [(3, 6), (3, 8), (2, 7), (4, 7)]:   # the hole's neighbors
        assert np.isfinite(amp[r, c]) and amp[r, c] > 0.0


# ---------- raytrace_field entry point -------------------------------------

def test_raytrace_field_carries_trace_and_amplitude():
    """raytrace_field returns the geometric trace plus matching amplitude."""
    c = -1 / 80.0
    f = 1.0 / (2.0 * c)
    presc = [conic(c=c, k=-1.0, interaction='refl', P=[0, 0, 0]),
             plane(interaction='eval', P=[0, 0, f])]
    wvl = 0.55e-3
    P, S = launch(presc, Field(0., 0.), wvl, Sampling.rect(n=7),
                  epd=10.0, pupil_z=-50.0)
    ft = field.raytrace_field(presc, P, S, wvl)
    tr = raytrace(presc, P, S, wvl)
    np.testing.assert_allclose(ft.P, tr.P)
    np.testing.assert_allclose(ft.status.imag, tr.status.imag)
    # mirror system is lossless
    assert np.allclose(ft.amplitude, 1.0, atol=1e-12)


def test_raytrace_field_tir_gives_zero_amplitude():
    """Rays beyond the critical angle carry zero transmitted amplitude."""
    # leading eval surface sets the immersed launch medium
    presc = [plane(interaction='eval', P=[0, 0, -5.0], material=materials.ConstantMaterial(1.5)),
             plane(interaction='refr', P=[0, 0, 0], material=materials.air),
             plane(interaction='eval', P=[0, 0, 10.0])]
    wvl = 0.55e-3
    P, S = launch(presc, Field(0.0, 50.0, kind='angle'), wvl,
                  Sampling.rect(n=3), epd=2.0, pupil_z=-5.0)
    ft = field.raytrace_field(presc, P, S, wvl)
    assert np.all(ft.amplitude == 0.0)


# ---------- sine-space pupil coordinates -----------------------------------

def test_sine_space_coords_scale_with_sin_theta():
    """The direction-cosine pupil coordinate scales with sin(theta)."""
    scale = 50.0
    thetas = np.radians(np.array([0.0, 10.0, 20.0, 30.0]))
    S_chief = np.array([0.0, 0.0, 1.0])
    # rays at polar angle theta in the y-z plane
    S_last = np.stack([np.zeros_like(thetas),
                       np.sin(thetas),
                       np.cos(thetas)], axis=-1)
    X, Y = field.sine_space_coords(S_last, S_chief, scale)
    assert np.allclose(X, 0.0, atol=1e-9)
    # |Y| = scale * sin(theta): sine space, not tangent space
    assert np.allclose(np.abs(Y), scale * np.sin(thetas), atol=1e-9)


# ---------- Phase 2: orchestration + propagation bridge --------------------

def test_pupil_field_low_na_matches_airy():
    """A slow, aberration-free pupil gives a near-perfect Airy PSF."""
    presc, f = _slow_parabola()
    wvl = 0.5
    pf = field.pupil_field(presc, Field(0., 0.), wvl, epd=4.0, npupil=96,
                           stop_index=0, pupil_z=-100.0)
    assert pf.efl == pytest.approx(f, rel=1e-6)
    wf = field.pupil_field_to_wavefront(pf, npix=128)
    psf = wf.focus(efl=pf.efl, Q=6)
    I = np.abs(psf.data) ** 2
    I /= I.max()
    cy, cx = np.unravel_index(I.argmax(), I.shape)
    yy, xx = np.indices(I.shape)
    r = np.hypot(xx - cx, yy - cy) * psf.dx
    F = abs(pf.efl) / 4.0
    x = np.pi * r / (wvl * F)
    x = np.where(x == 0, 1e-9, x)
    airy = (2 * j1(x) / x) ** 2
    core = r < 2 * 1.22 * wvl * F
    corr = np.corrcoef(I[core].ravel(), airy[core].ravel())[0, 1]
    assert corr > 0.999


def _telecentric_slow(epd=3.0):
    """Image-space-telecentric slow singlet."""
    from prysm.x.raytracing import OpticalSystem, LensData
    from prysm.x.raytracing.surfaces import Conic, Plane
    from prysm.x.raytracing.paraxial import ynu_first_order, paraxial_image_distance
    mat = materials.ConstantMaterial(1.5168)
    c = 1.0 / 120.0
    probe = LensData()
    (probe.add(Conic(c, 0.0), thickness=2.0, material=mat, semidiameter=8.0)
          .add(Conic(-c, 0.0), thickness=120.0, material=materials.air,
               semidiameter=8.0))
    sp = OpticalSystem(probe, aperture=epd, fields=[Field(0, 0, kind='angle')],
                       wavelengths=[0.5875618], reference=0,
                       stop_index=1)   # first powered surface (index 0 is OBJECT)
    ffl = ynu_first_order(sp.to_surfaces(), wvl=sp.wavelength(),
                          stop_index=1).ffl
    # rows: OBJECT(0), front stop plane(1), conic1(2), conic2(3), IMAGE(4)
    lens = LensData()
    (lens.add(Plane(), typ='eval', material=materials.air, semidiameter=epd / 2)
         .add(Conic(c, 0.0), thickness=2.0, material=mat, semidiameter=10.0)
         .add(Conic(-c, 0.0), thickness=120.0, material=materials.air, semidiameter=10.0))
    lens.rows[1].thickness = abs(ffl)
    sysT = OpticalSystem(lens, aperture=epd, fields=[Field(0, 0, kind='angle')],
                         wavelengths=[0.5875618], reference=0,
                         stop_index=1)
    wvl = sysT.wavelength()
    # focus distance from the last lens surface (exclude the image plane row)
    lens.rows[3].thickness = paraxial_image_distance(
        sysT.to_surfaces()[:-1], wvl)
    return sysT, wvl


def test_pupil_field_telecentric_exit_pupil_at_infinity_is_airy():
    """Telecentric image space keeps finite sine-space coordinates."""
    from prysm.x.raytracing.paraxial import ynu_first_order
    sysT, wvl = _telecentric_slow(epd=3.0)
    assert ynu_first_order(sysT.to_surfaces(), wvl, stop_index=1).xp_z is None
    pf = field.pupil_field(sysT, Field(0., 0.), wvl, npupil=96, stop_index=1)
    assert np.all(np.isfinite(pf.X)) and np.all(np.isfinite(pf.Y))
    assert np.isfinite(pf.efl) and pf.efl > 0
    assert pf.P_xp is None  # exit pupil at infinity, recorded as such
    psf, dx = field.pupil_field_psf(pf, npix=128, Q=6)
    assert np.all(np.isfinite(psf))
    cy, cx = np.unravel_index(psf.argmax(), psf.shape)
    assert abs(cy - psf.shape[0] // 2) <= 1 and abs(cx - psf.shape[1] // 2) <= 1
    I = psf / psf.max()
    yy, xx = np.indices(I.shape)
    r = np.hypot(xx - cx, yy - cy) * dx
    F = abs(pf.efl) / 3.0
    x = np.pi * r / (wvl * F)
    x = np.where(x == 0, 1e-9, x)
    airy = (2 * j1(x) / x) ** 2
    core = r < 2 * 1.22 * wvl * F
    assert np.corrcoef(I[core].ravel(), airy[core].ravel())[0, 1] > 0.999


def test_pupil_field_to_wavefront_is_pupil_space():
    presc, f = _slow_parabola()
    pf = field.pupil_field(presc, Field(0., 0.), 0.5, epd=4.0, npupil=64,
                           stop_index=0, pupil_z=-100.0)
    wf = field.pupil_field_to_wavefront(pf, npix=128)
    assert wf.space == 'pupil'
    assert wf.data.shape == (128, 128)
    assert np.iscomplexobj(wf.data)
    assert wf.dx > 0


def test_pupil_field_coating_is_amplitude_only():
    """Fresnel loss attenuates amplitude but leaves OPD unchanged."""
    from prysm.x.raytracing.analysis import wavefront
    presc = _fast_singlet()
    wvl = 0.5
    P, S = launch(presc, Field(0., 0.), wvl, Sampling.rect(n=64),
                  epd=8.0, pupil_z=-20.0)
    # phase-only wavefront over the same inscribed circular pupil
    opd_ref, xr, yr = wavefront(presc, P, S, wvl, P_xp=(0, 0, 0))
    circ = np.hypot(xr, yr) <= 4.0 * (1.0 + 1e-9)
    pf = field.pupil_field(presc, Field(0., 0.), wvl, epd=8.0, npupil=64,
                           P_xp=(0, 0, 0), pupil_z=-20.0)
    # amplitude carries Fresnel loss
    assert float(np.max(pf.amplitude)) < 1.0
    assert float(np.ptp(pf.amplitude)) > 0.0
    # OPD agrees with the phase-only wavefront
    assert np.nanmax(np.abs(opd_ref[circ])) == pytest.approx(
        np.nanmax(np.abs(pf.opd)), rel=1e-6)


def test_fast_singlet_is_spherical_aberration_not_airy():
    """A fast singlet pupil is dominated by spherical aberration."""
    presc = _fast_singlet()
    wvl = 0.5
    pf = field.pupil_field(presc, Field(0., 0.), wvl, epd=8.0, npupil=64,
                           P_xp=(0, 0, 0), pupil_z=-20.0)
    ptv_waves = float(np.ptp(pf.opd)) * 1e3 / wvl   # length(mm)->um->waves
    assert ptv_waves > 1.0   # many waves of aberration


def test_pupil_field_on_axis_requires_pupil_anchor():
    """An on-axis field with neither stop_index nor P_xp raises clearly."""
    presc, f = _slow_parabola()
    with pytest.raises(ValueError, match='exit pupil'):
        field.pupil_field(presc, Field(0., 0.), 0.5, epd=4.0, npupil=16,
                          pupil_z=-100.0)


def test_pupil_field_obscured_chief_needs_centroid_reference():
    """A clipped chief needs reference='centroid'."""
    presc, f = _slow_parabola()
    presc[0].aperture = annular_aperture(0.5, 4.0)    # block the pupil center
    wvl = 0.5
    with pytest.raises(ValueError, match='centroid'):
        field.pupil_field(presc, Field(0., 0.), wvl, epd=4.0, npupil=32,
                          stop_index=0, pupil_z=-100.0)
    pf = field.pupil_field(presc, Field(0., 0.), wvl, epd=4.0, npupil=32,
                           stop_index=0, pupil_z=-100.0, reference='centroid')
    assert pf.opd.shape[0] > 0
    assert np.all(np.isfinite(np.asarray(pf.opd, dtype=float)))


def test_pupil_field_finite_conjugate_apodization_does_not_collapse():
    """A finite-conjugate field keeps finite apodization."""
    ng = materials.ConstantMaterial(1.5)
    presc = [conic(c=1 / 30., k=0, interaction='refr', P=[0, 0, 0],
                   material=ng),
             conic(c=-1 / 30., k=0, interaction='refr', P=[0, 0, 3.],
                   material=materials.air),
             plane(interaction='eval', P=[0, 0, 51.])]
    fld = Field(0.0, 0.0, kind='height', object_z=-80.0)
    pf = field.pupil_field(presc, fld, 0.5, epd=6.0, npupil=48,
                           P_xp=(0, 0, 3.0), pupil_z=0.0)
    amp = np.asarray(pf.amplitude, dtype=float)
    assert np.all(np.isfinite(amp))
    assert float(np.max(amp)) > 0.0


def test_pupil_field_uses_vignetted_pupil_coordinates_for_opd_tilt():
    from prysm.x.raytracing.analysis import wavefront
    from prysm.x.raytracing.opt import _pupil_center_chief_index
    presc = _flat_refractor()
    wvl = 0.5
    epd = 4.0
    npupil = 21
    fld = Field(0.0, 8.0, kind='angle', vignetting={'vuy': 0.5})
    sampling = Sampling.rect(n=npupil)
    P, S = launch(presc, fld, wvl, sampling, epd=epd, pupil_z=-5.0)
    opd_ref, _, _ = wavefront(presc, P, S, wvl, P_xp=(0, 0, 0),
                              field=fld)
    nominal = sampling.build(0.5 * epd)
    chief = _pupil_center_chief_index(P)
    circ = (np.hypot(nominal[:, 0] - nominal[chief, 0],
                     nominal[:, 1] - nominal[chief, 1])
            <= 0.5 * epd * (1 + 1e-9))

    pf = field.pupil_field(presc, fld, wvl, epd=epd, npupil=npupil,
                           P_xp=(0, 0, 0), pupil_z=-5.0)
    np.testing.assert_allclose(pf.opd, opd_ref[circ], atol=1e-10)


# ---------- Phase 3: polarization ray tracing ------------------------------

def test_prt_matrix_matches_fresnel_diattenuation():
    """A single dielectric interface: |P.s|=sqrt(Ts), |P.p|=sqrt(Tp)."""
    presc = [plane(interaction='refr', P=[0, 0, 0], material=materials.ConstantMaterial(1.5)),
             plane(interaction='eval', P=[0, 0, 10.0])]
    wvl = 0.5
    # 40 deg collimated, tilt about x -> plane of incidence is y-z, s = x
    P, S = launch(presc, Field(0., 40., kind='angle'), wvl, Sampling.chief(),
                  epd=1.0, pupil_z=-5.0)
    pr = field.raytrace_prt(presc, P, S, wvl)
    Pmat = pr.P_matrix[0]
    k_in = S[0] / np.linalg.norm(S[0])
    s_hat = np.array([1.0, 0.0, 0.0])
    p_in = np.cross(k_in, s_hat)
    th0 = np.radians(40.0)
    th1 = np.arcsin(1 / 1.5 * np.sin(th0))
    ts = thinfilm.fresnel_ts(1, 1.5, th0, th1)
    tp = thinfilm.fresnel_tp(1, 1.5, th0, th1)
    ob = (1.5 * np.cos(th1)) / (1.0 * np.cos(th0))
    assert np.linalg.norm(Pmat @ s_hat) == pytest.approx(np.sqrt(ob) * abs(ts),
                                                         rel=1e-9)
    assert np.linalg.norm(Pmat @ p_in) == pytest.approx(np.sqrt(ob) * abs(tp),
                                                        rel=1e-9)


def test_prt_unpolarized_degenerates_to_scalar_mirror():
    """An ideal-mirror system's unpolarized PRT PSF equals the scalar PSF."""
    presc, f = _slow_parabola()
    wvl = 0.5
    pf_s = field.pupil_field(presc, Field(0., 0.), wvl, epd=4.0, npupil=96,
                             stop_index=0, pupil_z=-100.0)
    pf_p = field.pupil_field(presc, Field(0., 0.), wvl, epd=4.0, npupil=96,
                             stop_index=0, pupil_z=-100.0, polarized=True)
    ps, _ = field.pupil_field_psf(pf_s, npix=128, Q=4)
    pp, _ = field.pupil_field_psf(pf_p, npix=128, Q=4,
                                  input_polarization='unpolarized')
    ps = ps / ps.max()
    pp = pp / pp.max()
    assert float(np.abs(ps - pp).max()) < 1e-4


def test_prt_unpolarized_degenerates_to_scalar_dielectric():
    """A low-AOI refractive system's unpolarized PRT PSF equals the scalar."""
    ng = materials.ConstantMaterial(1.5)
    presc = [conic(c=1 / 120., k=0, interaction='refr', P=[0, 0, 0],
                   material=ng),
             conic(c=-1 / 120., k=0, interaction='refr', P=[0, 0, 3.],
                   material=materials.air),
             plane(interaction='eval', P=[0, 0, 120.])]
    wvl = 0.5
    pf_s = field.pupil_field(presc, Field(0., 0.), wvl, epd=6.0, npupil=96,
                             stop_index=0, pupil_z=-10.0)
    pf_p = field.pupil_field(presc, Field(0., 0.), wvl, epd=6.0, npupil=96,
                             stop_index=0, pupil_z=-10.0, polarized=True)
    ps, _ = field.pupil_field_psf(pf_s, npix=128, Q=4)
    pp, _ = field.pupil_field_psf(pf_p, npix=128, Q=4,
                                  input_polarization='unpolarized')
    ps = ps / ps.max()
    pp = pp / pp.max()
    assert float(np.abs(ps - pp).max()) < 1e-3


def test_prt_has_cross_polarization_leakage():
    """A fast refractive system rotates polarization (Ey leaks for x input)."""
    presc = _fast_singlet()
    wvl = 0.5
    pf = field.pupil_field(presc, Field(0., 0.), wvl, epd=8.0, npupil=64,
                           P_xp=(0, 0, 0), pupil_z=-20.0, polarized=True)
    wfx, wfy = field.pupil_field_to_wavefront(
        pf, npix=128, input_polarization=(1.0, 0.0, 0.0))
    ex = np.sum(np.abs(wfx.data) ** 2)
    ey = np.sum(np.abs(wfy.data) ** 2)
    # cross-polarization is small but nonzero -- the signature of polarization
    # aberration that a scalar model cannot represent
    assert 0.0 < ey / ex < 0.1


def test_pupil_field_to_wavefront_polarized_needs_input():
    presc, f = _slow_parabola()
    pf = field.pupil_field(presc, Field(0., 0.), 0.5, epd=4.0, npupil=32,
                           stop_index=0, pupil_z=-100.0, polarized=True)
    with pytest.raises(TypeError, match='input_polarization'):
        field.pupil_field_to_wavefront(pf, npix=64)
    comps = field.pupil_field_to_wavefront(pf, npix=64,
                                           input_polarization=(1, 0, 0))
    assert isinstance(comps, list) and len(comps) == 2


# ---------- Phase 4: coatings + polarization unification -------------------

from prysm.x.raytracing.spencer_and_murty import STYPE_REFRACT, STYPE_REFLECT
from prysm.x.coatings.stack import Stack


def test_interface_coefficients_zero_layer_matches_bare_fresnel():
    """A zero-layer TMM stack reduces to the bare Fresnel interface."""
    cosI = np.cos(np.radians(np.array([0.0, 15.0, 35.0, 55.0, 75.0])))
    wvl = 0.55
    bare_s, bare_p = field.interface_coefficients(1.0, 1.5, cosI, STYPE_REFRACT)
    stack = Stack([], [], substrate_index=1.5, ambient_index=1.0)
    cs, cp = field.interface_coefficients(1.0, 1.5, cosI, STYPE_REFRACT,
                                          coating=stack, wavelength=wvl)
    np.testing.assert_allclose(cs, bare_s, atol=1e-12)
    np.testing.assert_allclose(cp, bare_p, atol=1e-12)


def test_interface_coefficients_power_is_unit_for_bare_dielectric():
    """Bare-interface transmittance + reflectance conserve energy (R + T = 1)."""
    cosI = np.cos(np.radians(np.array([0.0, 30.0, 60.0])))
    a_s, a_p = field.interface_coefficients(1.0, 1.5, cosI, STYPE_REFRACT)
    stack = Stack([], [], substrate_index=1.5, ambient_index=1.0)
    r_s, r_p = field.interface_coefficients(1.0, 1.5, cosI, STYPE_REFLECT,
                                            coating=stack, wavelength=0.55)
    # s and p separately: |t|^2 (energy-normalized) + |r|^2 == 1
    np.testing.assert_allclose(np.abs(a_s) ** 2 + np.abs(r_s) ** 2, 1.0,
                               atol=1e-12)
    np.testing.assert_allclose(np.abs(a_p) ** 2 + np.abs(r_p) ** 2, 1.0,
                               atol=1e-12)


def test_quarter_wave_ar_coating_reduces_reflection():
    """A single MgF2 quarter-wave AR drops normal-incidence R from 4% to ~1.4%."""
    wvl = 0.55
    nl = 1.38
    ar = Stack([nl], [wvl / (4 * nl)], substrate_index=1.5, ambient_index=1.0)
    cosI = np.array([1.0])
    a_s, a_p = field.interface_coefficients(1.0, 1.5, cosI, STYPE_REFRACT,
                                            coating=ar, wavelength=wvl)
    T = 0.5 * (np.abs(a_s) ** 2 + np.abs(a_p) ** 2)
    # textbook single-layer AR: R = ((n0*ns - nl^2)/(n0*ns + nl^2))^2
    R_expected = ((1.0 * 1.5 - nl ** 2) / (1.0 * 1.5 + nl ** 2)) ** 2
    assert float(1.0 - T[0]) == pytest.approx(R_expected, abs=1e-9)
    assert float(1.0 - T[0]) < 0.04  # better than the bare interface


def test_metal_mirror_reduces_to_ideal_mirror():
    """A perfect-conductor coating reproduces the bare ideal mirror diag(1, -1)."""
    cosI = np.cos(np.radians(np.array([0.0, 20.0, 45.0, 70.0])))
    pec = Stack([], [], substrate_index=1.0 + 1e7j, ambient_index=1.0)
    a_s, a_p = field.interface_coefficients(1.0, 1.0, cosI, STYPE_REFLECT,
                                            coating=pec, wavelength=0.55)
    np.testing.assert_allclose(a_s, 1.0, atol=1e-5)
    np.testing.assert_allclose(a_p, -1.0, atol=1e-5)


def test_metal_mirror_has_diattenuation_and_retardance():
    """A real metal (complex nk) gives Rs > Rp obliquely and oblique retardance."""
    cosI = np.cos(np.radians(np.array([0.0, 45.0, 70.0])))
    al = Stack([], [], substrate_index=0.96 + 6.7j, ambient_index=1.0)
    a_s, a_p = field.interface_coefficients(1.0, 1.0, cosI, STYPE_REFLECT,
                                            coating=al, wavelength=0.55)
    Rs = np.abs(a_s) ** 2
    Rp = np.abs(a_p) ** 2
    # near-unit reflectance, diattenuating off-axis
    assert np.all(Rs > 0.8) and np.all(Rp > 0.8)
    assert Rs[0] == pytest.approx(Rp[0], rel=1e-9)        # equal at normal
    assert Rs[2] > Rp[2]                                   # Rs > Rp obliquely
    # s-p retardance departs from the ideal-mirror sign obliquely
    retardance = np.degrees(np.angle(a_s) - np.angle(a_p)) % 360.0
    assert retardance[0] == pytest.approx(180.0, abs=1e-6)
    assert abs(retardance[2] - 180.0) > 5.0


def test_surface_coating_raises_text_for_obsolete_kwarg():
    """The coatings= parallel-list kwarg is gone; coatings live on surfaces."""
    presc = _flat_refractor()
    P, S = launch(presc, Field(0., 0.), 0.55, Sampling.chief(),
                  epd=4.0, pupil_z=-5.0)
    with pytest.raises(TypeError):
        field.raytrace_field(presc, P, S, 0.55, coatings=[None, None])


def test_surface_coating_unpolarized_amplitude_beats_bare():
    """An AR coating on a Surface raises its transmitted amplitude vs bare."""
    wvl = 0.55
    nl = 1.38
    ar = Stack([nl], [wvl / (4 * nl)], substrate_index=1.5, ambient_index=1.0)
    bare = _flat_refractor()
    coated = [plane(interaction='refr', P=[0, 0, 0],
                    material=materials.ConstantMaterial(1.5), coating=ar),
              plane(interaction='eval', P=[0, 0, 10.0])]
    P, S = launch(bare, Field(0., 0.), wvl, Sampling.chief(),
                  epd=4.0, pupil_z=-5.0)
    amp_bare = field.raytrace_field(bare, P, S, wvl).amplitude[0]
    amp_coat = field.raytrace_field(coated, P, S, wvl).amplitude[0]
    assert amp_coat > amp_bare
    assert amp_coat == pytest.approx(np.sqrt(1.0 - 0.0141), abs=1e-3)


def test_prt_metal_mirror_matches_provider_reflectance():
    """A metal-coated fold mirror's PRT matrix carries the s/p reflectances."""
    al = Stack([], [], substrate_index=0.96 + 6.7j, ambient_index=1.0)
    # flat mirror at the origin; reflected bundle travels back toward -z.
    presc = [plane(interaction='refl', P=[0, 0, 0], coating=al),
             plane(interaction='eval', P=[0, 0, -10.0])]
    wvl = 0.55
    # 40 deg collimated, tilt about x -> plane of incidence is y-z, s = x
    P, S = launch(presc, Field(0., 40., kind='angle'), wvl, Sampling.chief(),
                  epd=1.0, pupil_z=-5.0)
    pr = field.raytrace_prt(presc, P, S, wvl)
    Pmat = pr.P_matrix[0]
    cosI = np.cos(np.radians(np.array([40.0])))
    a_s, a_p = field.interface_coefficients(1.0, 1.0, cosI, STYPE_REFLECT,
                                            coating=al, wavelength=wvl)
    s_hat = np.array([1.0, 0.0, 0.0])
    k_in = S[0] / np.linalg.norm(S[0])
    p_in = np.cross(k_in, s_hat)
    assert np.linalg.norm(Pmat @ s_hat) == pytest.approx(abs(a_s[0]), rel=1e-9)
    assert np.linalg.norm(Pmat @ p_in) == pytest.approx(abs(a_p[0]), rel=1e-9)
    # the metal mirror is diattenuating: the s and p responses differ
    assert abs(np.linalg.norm(Pmat @ s_hat)
               - np.linalg.norm(Pmat @ p_in)) > 1e-3
