"""Tests for the complex pupil field bridge (field.py), Phase 0.

Covers the shared Fresnel seam, per-surface normal/incidence recovery, the
unpolarized scalar amplitude, and the geometric (sine-space) apodization that
is Hopkins' a(X', Y').
"""
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
    """A NaN (missed/clipped) sphere sample must not zero its valid neighbors.

    np.gradient's central difference spreads a single NaN onto adjacent grid
    points; without the inpaint that zeroes legitimately-transmitted rays just
    inside a vignetting boundary.
    """
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
    # bundle inside glass onto a flat glass->air interface at 50 deg; critical
    # angle is ~41.8 deg, so all rays totally internally reflect and transmit
    # no power.  The launch medium (n=1.5) is carried by a leading eval object
    # surface -- the convention for an immersed launch.
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
    """The direction-cosine pupil coordinate scales with sin(theta) (sine
    space), not tan(theta) -- the parameterization that makes the PSF a plain
    Fourier transform.  Determinate for every conjugate (no exit pupil needed)."""
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
    """Image-space-telecentric slow singlet: stop one front-focal-distance
    ahead of the lens, so the exit pupil is at infinity (xp_z is None)."""
    from prysm.x.raytracing import OpticalSystem, LensData
    from prysm.x.raytracing.surfaces import Conic, Plane
    from prysm.x.raytracing.paraxial import first_order, paraxial_image_distance
    mat = materials.ConstantMaterial(1.5168)
    c = 1.0 / 120.0
    probe = LensData()
    (probe.add(Conic(c, 0.0), thickness=2.0, material=mat, semidiameter=8.0)
          .add(Conic(-c, 0.0), thickness=120.0, material=materials.air,
               semidiameter=8.0))
    sp = OpticalSystem(probe, aperture=epd, fields=[Field(0, 0, kind='angle')],
                       wavelengths=[0.5875618], reference=0,
                       stop_index=0)
    ffl = first_order(sp, stop_index=0).ffl
    lens = LensData()
    (lens.add(Plane(), typ='eval', material=materials.air, semidiameter=epd / 2)
         .add(Conic(c, 0.0), thickness=2.0, material=mat, semidiameter=10.0)
         .add(Conic(-c, 0.0), thickness=120.0, material=materials.air, semidiameter=10.0)
         .add(Plane(), typ='eval', material=materials.air, semidiameter=15.0))
    lens.rows[0].thickness = abs(ffl)
    sysT = OpticalSystem(lens, aperture=epd, fields=[Field(0, 0, kind='angle')],
                         wavelengths=[0.5875618], reference=0,
                         stop_index=0)
    wvl = sysT.wavelength()
    # focus distance from the last lens surface (exclude the image plane row)
    lens.rows[2].thickness = paraxial_image_distance(
        sysT.to_surfaces()[:-1], wvl)
    return sysT, wvl


def test_pupil_field_telecentric_exit_pupil_at_infinity_is_airy():
    """An image-space-telecentric system has its exit pupil at infinity, where a
    position-on-the-sphere coordinate would diverge; the sine-space (direction
    cosine) coordinate scaled by |EFL| stays finite, so the F-number EFL/EPD is
    bounded and the well-corrected on-axis PSF is a clean, centered Airy."""
    from prysm.x.raytracing.paraxial import first_order
    sysT, wvl = _telecentric_slow(epd=3.0)
    assert first_order(sysT, wvl, stop_index=0).xp_z is None
    pf = field.pupil_field(sysT, Field(0., 0.), wvl, npupil=96, stop_index=0)
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
    """Fresnel loss attenuates amplitude but leaves the wavefront unchanged.

    The bare-interface Fresnel factor is part of the scalar amplitude; the OPD
    must be identical to the phase-only wavefront() result on the same rays.
    """
    from prysm.x.raytracing.analysis import wavefront
    presc = _fast_singlet()
    wvl = 0.5
    P, S = launch(presc, Field(0., 0.), wvl, Sampling.rect(n=64),
                  epd=8.0, pupil_z=-20.0)
    # phase-only wavefront on the same launch bundle, masked to the inscribed
    # circular entrance pupil so it covers the same rays pupil_field keeps
    opd_ref, xr, yr = wavefront(presc, P, S, wvl, P_xp=(0, 0, 0))
    circ = np.hypot(xr, yr) <= 4.0 * (1.0 + 1e-9)
    pf = field.pupil_field(presc, Field(0., 0.), wvl, epd=8.0, npupil=64,
                           P_xp=(0, 0, 0), pupil_z=-20.0)
    # amplitude carries Fresnel loss: strictly below the lossless geometric
    # value (< 1 after two glass interfaces), and not all equal
    assert float(np.max(pf.amplitude)) < 1.0
    assert float(np.ptp(pf.amplitude)) > 0.0
    # OPD agrees with the phase-only wavefront (amplitude-only coating), in
    # length units, over the matched circular pupil
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
    """A central obstruction clips the chief; reference='centroid' recovers it.

    Mirrors analysis.wavefront: reference='chief' raises a clear error (not a
    bare IndexError out of the chief-index lookup), and reference='centroid'
    anchors on the surviving ray nearest the pupil center.
    """
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
    """A finite-conjugate (object-height) field must keep a real apodization.

    Regression: the apodization used the launch positions, which coincide at
    the object point for a finite conjugate, so np.gradient divided by zero
    spacing and the amplitude collapsed to zero.  Using the pupil-sample grid
    fixes it -- the amplitude must be finite and nonzero.
    """
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
