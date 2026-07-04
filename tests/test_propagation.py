"""Tests for numerical propagation routines."""
import functools

import pytest

import numpy as np

from prysm import propagation, coordinates, geometry, polynomials
from prysm.wavelengths import HeNe


SAMPLES = 32


@pytest.mark.parametrize('dzeta', [1 / 128.0, 1 / 256.0, 11.123 / 128.0, 1e10 / 2048.0])
def test_psf_to_pupil_sample_inverts_pupil_to_psf_sample(dzeta):
    samples, wvl, efl = 128, 0.55, 10
    psf_sample = propagation.pupil_sample_to_psf_sample(dzeta, samples, wvl, efl)
    dzeta2 = propagation.psf_sample_to_pupil_sample(psf_sample, samples, wvl, efl)
    assert dzeta2 == dzeta


def test_obj_oriented_wavefront_focusing_reverses():
    z = np.random.rand(128, 128)
    dx = 1
    wf = propagation.Wavefront(dx=dx, cmplx_field=z, wavelength=HeNe)
    wf2 = wf.focus(1, 1).unfocus(1, 1)  # first is efl, meaningless.  second is Q, we neglect padding here
    assert np.allclose(wf.data, wf2.data)


@pytest.mark.parametrize('Q', [1, 1.5, 2])
def test_focus_adjoint_is_adjoint(Q):
    rng = np.random.default_rng(789)
    x = rng.normal(size=(9, 12)) + 1j * rng.normal(size=(9, 12))
    y = rng.normal(size=propagation.focus(x, Q=Q).shape)
    y = y + 1j * rng.normal(size=y.shape)

    lhs = np.vdot(propagation.focus(x, Q=Q), y)
    rhs = np.vdot(x, propagation.focus_adjoint(y, Q=Q))

    np.testing.assert_allclose(lhs, rhs, atol=1e-12)


@pytest.mark.parametrize('Q', [1, 1.5, 2])
def test_unfocus_adjoint_is_adjoint(Q):
    rng = np.random.default_rng(987)
    x = rng.normal(size=(9, 12)) + 1j * rng.normal(size=(9, 12))
    y = rng.normal(size=propagation.unfocus(x, Q=Q).shape)
    y = y + 1j * rng.normal(size=y.shape)

    lhs = np.vdot(propagation.unfocus(x, Q=Q), y)
    rhs = np.vdot(x, propagation.unfocus_adjoint(y, Q=Q))

    np.testing.assert_allclose(lhs, rhs, atol=1e-12)


def test_wavefront_focus_adjoint_metadata_and_data():
    rng = np.random.default_rng(135)
    dx = 0.25
    efl = 10
    Q = 2
    data = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
    wf = propagation.Wavefront(dx=dx, cmplx_field=data, wavelength=HeNe, space='pupil')
    psf = wf.focus(efl=efl, Q=Q)
    grad_data = rng.normal(size=psf.data.shape) + 1j * rng.normal(size=psf.data.shape)
    grad = propagation.Wavefront(dx=psf.dx, cmplx_field=grad_data,
                                 wavelength=HeNe, space='psf')

    back = grad.focus_adjoint(efl=efl, Q=Q)

    np.testing.assert_allclose(back.data, propagation.focus_adjoint(grad_data, Q=Q))
    assert back.data.shape == wf.data.shape
    assert back.dx == pytest.approx(wf.dx)
    assert back.space == 'pupil'


def test_wavefront_unfocus_adjoint_metadata_and_data():
    rng = np.random.default_rng(246)
    dx = 0.1
    efl = 10
    Q = 2
    data = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
    wf = propagation.Wavefront(dx=dx, cmplx_field=data, wavelength=HeNe, space='psf')
    pupil = wf.unfocus(efl=efl, Q=Q)
    grad_data = rng.normal(size=pupil.data.shape) + 1j * rng.normal(size=pupil.data.shape)
    grad = propagation.Wavefront(dx=pupil.dx, cmplx_field=grad_data,
                                 wavelength=HeNe, space='pupil')

    back = grad.unfocus_adjoint(efl=efl, Q=Q)

    np.testing.assert_allclose(back.data, propagation.unfocus_adjoint(grad_data, Q=Q))
    assert back.data.shape == wf.data.shape
    assert back.dx == pytest.approx(wf.dx)
    assert back.space == 'psf'


def test_unfocus_fft_mdft_equivalent_Wavefront():
    z = np.random.rand(128, 128)
    dx = 1
    wf = propagation.Wavefront(dx=dx, cmplx_field=z, wavelength=HeNe, space='psf')
    unfocus_fft = wf.unfocus(Q=2, efl=1)
    mdft = wf.prepare_executor(efl=1, dx=unfocus_fft.dx, samples=unfocus_fft.data.shape)
    unfocus_mdft = wf.unfocus_dft(mdft)

    assert np.allclose(unfocus_fft.data, unfocus_mdft.data)


def test_focus_fft_mdft_equivalent_Wavefront():
    dx = 1
    z = np.random.rand(SAMPLES, SAMPLES)
    wf = propagation.Wavefront(dx=dx, cmplx_field=z, wavelength=HeNe, space='pupil')
    focus_fft = wf.focus(Q=2, efl=1)
    mdft = wf.prepare_executor(efl=1, dx=focus_fft.dx, samples=focus_fft.data.shape)
    focus_mdft = wf.focus_dft(mdft)

    assert np.allclose(focus_fft.data, focus_mdft.data)


def test_focus_dft_adjoint_is_adjoint():
    rng = np.random.default_rng(159)
    x = rng.normal(size=(7, 9)) + 1j * rng.normal(size=(7, 9))
    mdft = propagation.prepare_executor(
        pupil_dx=0.25, pupil_samples=x.shape,
        focal_dx=0.1, focal_samples=(8, 11),
        wavelength=HeNe, efl=10.0,
    )
    y = rng.normal(size=(8, 11)) + 1j * rng.normal(size=(8, 11))

    lhs = np.vdot(propagation.focus_dft(x, mdft), y)
    rhs = np.vdot(x, propagation.focus_dft_adjoint(y, mdft))

    np.testing.assert_allclose(lhs, rhs, atol=1e-12)


def test_unfocus_dft_adjoint_is_adjoint():
    rng = np.random.default_rng(7531)
    x = rng.normal(size=(8, 11)) + 1j * rng.normal(size=(8, 11))
    mdft = propagation.prepare_executor(
        pupil_dx=0.25, pupil_samples=(7, 9),
        focal_dx=0.1, focal_samples=x.shape,
        wavelength=HeNe, efl=10.0,
    )
    y = rng.normal(size=(7, 9)) + 1j * rng.normal(size=(7, 9))

    lhs = np.vdot(propagation.unfocus_dft(x, mdft), y)
    rhs = np.vdot(x, propagation.unfocus_dft_adjoint(y, mdft))

    np.testing.assert_allclose(lhs, rhs, atol=1e-12)


def test_free_space_zero_distance_is_identity():
    z = np.random.rand(SAMPLES, SAMPLES)
    wf = propagation.Wavefront(dx=1, cmplx_field=z, wavelength=HeNe, space='pupil')

    out = wf.free_space(0)

    np.testing.assert_allclose(out.data, wf.data, atol=1e-12)
    assert out.dx == wf.dx
    assert out.wavelength == wf.wavelength


@pytest.mark.parametrize('Q', [1, 1.5, 2])
def test_angular_spectrum_adjoint_is_adjoint(Q):
    rng = np.random.default_rng(321)
    x = rng.normal(size=(9, 12)) + 1j * rng.normal(size=(9, 12))
    y = rng.normal(size=propagation.angular_spectrum(x, wvl=HeNe, dx=0.25, z=1.2, Q=Q).shape)
    y = y + 1j * rng.normal(size=y.shape)

    lhs = np.vdot(propagation.angular_spectrum(x, wvl=HeNe, dx=0.25, z=1.2, Q=Q), y)
    rhs = np.vdot(x, propagation.angular_spectrum_adjoint(y, wvl=HeNe, dx=0.25, z=1.2, Q=Q))

    np.testing.assert_allclose(lhs, rhs, atol=1e-12)


def test_angular_spectrum_adjoint_with_tf_is_adjoint():
    rng = np.random.default_rng(654)
    x = rng.normal(size=(9, 12)) + 1j * rng.normal(size=(9, 12))
    y = rng.normal(size=x.shape) + 1j * rng.normal(size=x.shape)
    tf = propagation.angular_spectrum_transfer_function(x.shape, HeNe, 0.25, z=1.2)

    lhs = np.vdot(propagation.angular_spectrum(x, wvl=HeNe, dx=0.25, z=np.nan, tf=tf), y)
    rhs = np.vdot(x, propagation.angular_spectrum_adjoint(y, wvl=HeNe, dx=0.25, z=np.nan, tf=tf))

    np.testing.assert_allclose(lhs, rhs, atol=1e-12)


def test_wavefront_free_space_adjoint_metadata_and_data():
    rng = np.random.default_rng(753)
    dx = 0.25
    dz = 1.2
    Q = 2
    data = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
    wf = propagation.Wavefront(dx=dx, cmplx_field=data, wavelength=HeNe, space='pupil')
    out = wf.free_space(dz=dz, Q=Q)
    grad_data = rng.normal(size=out.data.shape) + 1j * rng.normal(size=out.data.shape)
    grad = propagation.Wavefront(dx=out.dx, cmplx_field=grad_data,
                                 wavelength=HeNe, space=out.space)

    back = grad.free_space_adjoint(dz=dz, Q=Q)

    np.testing.assert_allclose(
        back.data,
        propagation.angular_spectrum_adjoint(grad_data, wvl=HeNe, dx=dx, z=dz, Q=Q),
    )
    assert back.data.shape == wf.data.shape
    assert back.dx == pytest.approx(wf.dx)
    assert back.space == wf.space


def test_talbot_distance_correct():
    wvl = 123.456
    a = 987.654321
    truth = wvl / (1 - np.sqrt(1 - wvl**2/a**2))
    tal = propagation.talbot_distance(a, wvl)
    assert truth == pytest.approx(tal, abs=.1)


def test_fresnel_number_correct():
    wvl = 123.456
    a = 987.654321
    z = 5
    fres = propagation.fresnel_number(a, z, wvl)
    assert fres == (a**2 / (z * wvl))


def test_wavefront_multiply_and_divide_apply_to_data():
    data = np.arange(4, dtype=float).reshape(2, 2).astype(np.complex128)
    wf = propagation.Wavefront(cmplx_field=data, dx=1, wavelength=.6328)

    np.testing.assert_allclose((wf * 2).data, data * 2)
    np.testing.assert_allclose((wf / 2).data, data / 2)


def test_wavefront_scalar_arithmetic_operand_order():
    # non-commutative ops (sub, truediv) must compute self OP other, not other OP self
    data = (np.random.rand(2, 2) + 1).astype(np.complex128)
    wf = propagation.Wavefront(cmplx_field=data, dx=1, wavelength=.6328)
    assert np.allclose((wf - 1.0).data, data - 1.0)
    assert np.allclose((wf / 2.0).data, data / 2.0)


def test_to_fpm_and_back_adjoint_accepts_wavefront_fpm():
    # regression: previously raised AttributeError on fpm.dtype / fpm.conj()
    dx = 1.0
    z = (np.random.rand(SAMPLES, SAMPLES) + 1j * np.random.rand(SAMPLES, SAMPLES))
    wf = propagation.Wavefront(cmplx_field=z, dx=dx, wavelength=HeNe, space='pupil')
    fpm_data = (np.random.rand(SAMPLES, SAMPLES)
                + 1j * np.random.rand(SAMPLES, SAMPLES)).astype(np.complex128)
    fpm = propagation.Wavefront(cmplx_field=fpm_data, dx=0.1, wavelength=HeNe, space='psf')
    mdft = wf.prepare_executor(efl=10.0, dx=fpm.dx, samples=fpm.data.shape)
    out = wf.to_fpm_and_back(fpm=fpm, executor=mdft)
    grad = out.to_fpm_and_back_adjoint(fpm=fpm, executor=mdft)
    assert grad.data.shape == wf.data.shape


def _real_vdot(a, b):
    return np.real(np.vdot(a, b))


def test_to_fpm_and_back_adjoint_is_adjoint_for_input_field():
    rng = np.random.default_rng(2468)
    x = rng.normal(size=(7, 9)) + 1j * rng.normal(size=(7, 9))
    fpm = rng.normal(size=(8, 11)) + 1j * rng.normal(size=(8, 11))
    y = rng.normal(size=x.shape) + 1j * rng.normal(size=x.shape)
    mdft = propagation.prepare_executor(
        pupil_dx=0.25, pupil_samples=x.shape,
        focal_dx=0.1, focal_samples=fpm.shape,
        wavelength=HeNe, efl=10.0,
    )

    lhs = np.vdot(propagation.to_fpm_and_back(x, fpm=fpm, executor=mdft), y)
    rhs = np.vdot(x, propagation.to_fpm_and_back_adjoint(y, fpm=fpm, executor=mdft))

    np.testing.assert_allclose(lhs, rhs, atol=1e-12)


def test_to_fpm_and_back_adjoint_returns_fpm_gradient():
    rng = np.random.default_rng(123)
    dx = 1.0
    z = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
    wf = propagation.Wavefront(cmplx_field=z, dx=dx, wavelength=HeNe, space='pupil')
    fpm_data = rng.normal(size=(8, 8))
    fpm = propagation.Wavefront(cmplx_field=fpm_data, dx=0.1, wavelength=HeNe, space='psf')
    mdft = wf.prepare_executor(efl=10.0, dx=fpm.dx, samples=fpm.data.shape)
    out, at_fpm, _ = wf.to_fpm_and_back(fpm=fpm, executor=mdft, return_more=True)
    outbar_data = rng.normal(size=out.data.shape) + 1j * rng.normal(size=out.data.shape)
    outbar = propagation.Wavefront(cmplx_field=outbar_data, dx=out.dx,
                                   wavelength=HeNe, space=out.space)

    _, fpm_bar = outbar.to_fpm_and_back_adjoint(
        fpm=fpm, executor=mdft, return_fpm_grad=True, field_at_fpm=at_fpm,
    )

    y, x = 3, 4
    eps = 1e-6
    fpm_plus = fpm_data.copy()
    fpm_minus = fpm_data.copy()
    fpm_plus[y, x] += eps
    fpm_minus[y, x] -= eps
    j_plus = _real_vdot(outbar_data, wf.to_fpm_and_back(fpm=fpm_plus, executor=mdft).data)
    j_minus = _real_vdot(outbar_data, wf.to_fpm_and_back(fpm=fpm_minus, executor=mdft).data)
    fd = (j_plus - j_minus) / (2 * eps)
    assert fpm_bar.data[y, x] == pytest.approx(fd, rel=1e-6, abs=1e-8)


def test_babinet_adjoint_returns_fpm_and_lyot_gradients():
    rng = np.random.default_rng(456)
    dx = 1.0
    z = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
    wf = propagation.Wavefront(cmplx_field=z, dx=dx, wavelength=HeNe, space='pupil')
    fpm_data = rng.normal(size=(8, 8))
    lyot_data = rng.normal(size=(8, 8))
    fpm = propagation.Wavefront(cmplx_field=fpm_data, dx=0.1, wavelength=HeNe, space='psf')
    lyot = propagation.Wavefront(cmplx_field=lyot_data, dx=dx, wavelength=HeNe, space='pupil')
    mdft = wf.prepare_executor(efl=10.0, dx=fpm.dx, samples=fpm.data.shape)
    out, at_fpm, _, at_lyot = wf.babinet(lyot=lyot, fpm=fpm, executor=mdft, return_more=True)
    outbar_data = rng.normal(size=out.data.shape) + 1j * rng.normal(size=out.data.shape)
    outbar = propagation.Wavefront(cmplx_field=outbar_data, dx=out.dx,
                                   wavelength=HeNe, space=out.space)

    _, fpm_bar, lyot_bar = outbar.babinet_adjoint(
        lyot=lyot, fpm=fpm, executor=mdft,
        field_at_fpm=at_fpm, field_at_lyot=at_lyot,
        return_fpm_grad=True, return_lyot_grad=True,
    )

    eps = 1e-6
    fy, fx = 2, 5
    fpm_plus = fpm_data.copy()
    fpm_minus = fpm_data.copy()
    fpm_plus[fy, fx] += eps
    fpm_minus[fy, fx] -= eps
    fpm_plus = propagation.Wavefront(fpm_plus, HeNe, fpm.dx, 'psf')
    fpm_minus = propagation.Wavefront(fpm_minus, HeNe, fpm.dx, 'psf')
    j_plus = _real_vdot(outbar_data, wf.babinet(lyot=lyot, fpm=fpm_plus, executor=mdft).data)
    j_minus = _real_vdot(outbar_data, wf.babinet(lyot=lyot, fpm=fpm_minus, executor=mdft).data)
    fd_fpm = (j_plus - j_minus) / (2 * eps)

    ly, lx = 6, 1
    lyot_plus = lyot_data.copy()
    lyot_minus = lyot_data.copy()
    lyot_plus[ly, lx] += eps
    lyot_minus[ly, lx] -= eps
    lyot_plus = propagation.Wavefront(lyot_plus, HeNe, lyot.dx, 'pupil')
    lyot_minus = propagation.Wavefront(lyot_minus, HeNe, lyot.dx, 'pupil')
    j_plus = _real_vdot(outbar_data, wf.babinet(lyot=lyot_plus, fpm=fpm, executor=mdft).data)
    j_minus = _real_vdot(outbar_data, wf.babinet(lyot=lyot_minus, fpm=fpm, executor=mdft).data)
    fd_lyot = (j_plus - j_minus) / (2 * eps)

    assert fpm_bar.data[fy, fx] == pytest.approx(fd_fpm, rel=1e-6, abs=1e-8)
    assert lyot_bar.data[ly, lx] == pytest.approx(fd_lyot, rel=1e-6, abs=1e-8)


def test_precomputed_angular_spectrum_matches_direct_zero_distance():
    data = np.random.rand(4, 4)
    wf = propagation.Wavefront(cmplx_field=data, dx=1, wavelength=.6328)
    tf = propagation.angular_spectrum_transfer_function(wf.data.shape, wf.wavelength, wf.dx, z=0)

    out = wf.free_space(tf=tf)

    np.testing.assert_allclose(out.data, wf.data, atol=1e-12)


def test_thinlens_hopkins_agree():
    # F/10 beam
    x, y = coordinates.make_xy_grid(128, diameter=11)
    dx = x[0, 1] - x[0, 0]
    r = np.hypot(x, y)
    amp = geometry.circle(5, r)
    phs = polynomials.hopkins(0, 2, 0, r/5, 0, 1) * (1.975347661 * HeNe * 1000)  # 1000, nm to um
    wf = propagation.Wavefront.from_amp_and_phase(amp, phs, HeNe, dx)

    # easy case is to choose thin lens efl = 10,000
    # which will result in an overall focal length of 99.0 mm
    # solve defocus delta z relation, then 1000 = 8 * .6328 * 100 * x
    #                                  x = 1000 / 8 / .6328 / 100
    #                                    = 1.975347661
    psf = wf.focus(efl=100, Q=2).intensity

    no_phs_wf = propagation.Wavefront.from_amp_and_phase(amp, None, HeNe, dx)
    tl = propagation.Wavefront.thin_lens(10_000, HeNe, x, y)
    wf = no_phs_wf * tl
    psf2 = wf.focus(efl=100, Q=2).intensity
    assert np.allclose(psf.data, psf2.data, rtol=1e-5)


# --- multi-resolution vortex coronagraph -----------------------------------
def _grey_circle(radius, npup, dx, ss=16):
    """Supersampled (anti-aliased) circular aperture of the given radius."""
    xx, yy = coordinates.make_xy_grid(npup * ss, dx=dx / ss)
    rr = np.hypot(xx, yy)
    fine = (rr < radius).astype(np.float32)
    return fine.reshape(npup, ss, npup, ss).mean(axis=(1, 3))


@functools.lru_cache(maxsize=2)
def _vortex_rig(kind):
    """Build the reusable pieces of a charge-2 vortex coronagraph.

    Returns the aperture, undersized Lyot stop, multi-resolution executor, and
    final-focus machinery so several tests (ideal mask, measured mask) can
    drive the same optical system with different focal plane masks. Cached so
    the heavy aperture and executor construction is shared.
    """
    wvl = HeNe
    efl = 100.0          # mm
    pupil_dx = 0.05      # mm
    npup = 384
    nd = 320             # aperture samples across; Dap = nd * pupil_dx
    Dap = nd * pupil_dx
    x, y = coordinates.make_xy_grid(npup, dx=pupil_dx)
    r = np.hypot(x, y)
    fno = efl / Dap
    lamD = fno * wvl              # microns per lambda/D
    period = wvl * efl / pupil_dx  # full focal field of view, microns

    pupil = _grey_circle(Dap / 2, npup, pupil_dx).astype(complex)
    # undersized Lyot stop at 0.8 of the aperture radius
    lyot = _grey_circle(0.8 * Dap / 2, npup, pupil_dx)

    # coarsest level spans the full field of view at Nyquist (q0 = 2); finer
    # levels zoom into the vortex phase singularity at the focal origin.
    nf0 = 2 * nd
    executor = propagation.prepare_multiresolution(
        pupil_dx, npup, period / nf0, nf0, wvl, efl,
        num_levels=6, fine_samples=256, kind=kind,
    )

    nf = 256
    fdx = lamD / 4
    final = propagation.prepare_executor(pupil_dx, npup, fdx, nf, wvl, efl, kind=kind)
    # normalize by the peak of the non-coronagraphic PSF (no FPM, no Lyot stop)
    ref_peak = (np.abs(propagation.focus_dft(pupil, final)) ** 2).max()

    fx = np.arange(-(nf // 2), nf // 2) * fdx
    XF, YF = np.meshgrid(fx, fx)
    rad_lamD = np.hypot(XF, YF) / lamD

    return {
        'pupil': pupil, 'lyot': lyot, 'executor': executor, 'final': final,
        'ref_peak': ref_peak, 'rad_lamD': rad_lamD, 'lamD': lamD,
        'r': r, 'Dap': Dap, 'pupil_peak': np.abs(pupil).max() ** 2,
    }


def _lyot_field(rig, fpm):
    return propagation.to_fpm_and_back_multiresolution(
        rig['pupil'], fpm, rig['executor'])


def _dark_hole_max(rig, fpm):
    lyot_field = _lyot_field(rig, fpm)
    psf = np.abs(propagation.focus_dft(lyot_field * rig['lyot'], rig['final'])) ** 2
    norm_intensity = psf / rig['ref_peak']
    dark_hole = (rig['rad_lamD'] > 3) & (rig['rad_lamD'] < 10)
    return norm_intensity[dark_hole].max()


@pytest.mark.parametrize('kind', ['mdft', 'czt'])
def test_multiresolution_vortex_dark_hole_below_1e12(kind):
    # after the undersized (0.8 R) Lyot stop, the next focus has a deep dark
    # hole: normalized intensity (PSF / non-coronagraphic peak) below 1e-12
    rig = _vortex_rig(kind)
    assert _dark_hole_max(rig, propagation.vortex_phase_mask(2)) < 1e-12


@pytest.mark.parametrize('kind', ['mdft', 'czt'])
def test_multiresolution_to_fpm_and_back_adjoint_is_adjoint(kind):
    rng = np.random.default_rng(20240530)
    npup = 64
    executor = propagation.prepare_multiresolution(
        pupil_dx=0.1, pupil_samples=npup, focal_dx=2.0, focal_samples=32,
        wavelength=HeNe, efl=10.0, num_levels=3, fine_samples=32, kind=kind,
    )
    fpm = propagation.vortex_phase_mask(2)
    x = rng.standard_normal((npup, npup)) + 1j * rng.standard_normal((npup, npup))
    y = rng.standard_normal((npup, npup)) + 1j * rng.standard_normal((npup, npup))

    lhs = np.vdot(propagation.to_fpm_and_back_multiresolution(x, fpm, executor), y)
    rhs = np.vdot(x, propagation.to_fpm_and_back_multiresolution_adjoint(y, fpm, executor))

    np.testing.assert_allclose(lhs, rhs, rtol=1e-10)


def test_prepare_measured_fpm_interpolates_and_fills():
    # exact recovery on the measurement's own grid; ideal vortex continuation
    # for coordinates beyond the measured extent
    n = 129
    dx = 0.4
    x, y = coordinates.make_xy_grid(n, dx=dx)
    measurement = np.exp(1j * 2 * np.arctan2(y, x))
    fpm = propagation.prepare_measured_fpm(measurement, dx, charge=2)

    np.testing.assert_allclose(fpm(x, y), measurement, atol=1e-12)

    far = np.full((1, 1), 1e5)
    ideal = np.exp(1j * 2 * np.arctan2(far, far))
    np.testing.assert_allclose(fpm(far, far), ideal, atol=1e-12)


def test_prepare_measured_fpm_scalar_fill():
    n = 65
    dx = 1.0
    x, y = coordinates.make_xy_grid(n, dx=dx)
    measurement = np.ones((n, n), dtype=complex)
    fpm = propagation.prepare_measured_fpm(measurement, dx, fill=0.0)
    far = np.full((1, 1), 1e3)
    assert fpm(far, far)[0, 0] == 0.0


def _measured_vortex(lamD, charge=2, extent_lamD=40, samples_per_lamD=8, error=None):
    """A measured-style complex vortex map on its own fine uniform grid."""
    mdx = lamD / samples_per_lamD
    n = int(extent_lamD * samples_per_lamD) // 2 * 2 + 1
    mx, my = coordinates.make_xy_grid(n, dx=mdx)
    phase = charge * np.arctan2(my, mx)
    if error is not None:
        phase = phase + error(np.hypot(mx, my) / lamD)
    return np.exp(1j * phase), mdx


def test_measured_fpm_captures_manufacturing_error():
    # drive the coronagraph with a *measured* mask: an ideal vortex map yields
    # deep suppression, and an injected fabrication phase ripple makes the dark
    # hole measurably brighter
    rig = _vortex_rig('mdft')

    ideal_map, mdx = _measured_vortex(rig['lamD'])
    dh_ideal = _dark_hole_max(rig, propagation.prepare_measured_fpm(ideal_map, mdx, charge=2))

    ripple = lambda r_lamD: 0.05 * np.sin(2 * np.pi * r_lamD / 3.0)  # 50 mrad
    err_map, mdx = _measured_vortex(rig['lamD'], error=ripple)
    dh_error = _dark_hole_max(rig, propagation.prepare_measured_fpm(err_map, mdx, charge=2))

    # the measured ideal map still suppresses starlight strongly...
    assert dh_ideal < 1e-5
    # ...and the manufacturing error degrades the contrast
    assert dh_error > 3 * dh_ideal
