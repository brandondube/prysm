"""dOPD/dtau maps vs central FD of analysis.wavefront."""
import numpy as np
import pytest

from prysm.x.raytracing.launch import Field
from prysm.x.raytracing._diff_raytrace import (
    wavefront_with_tangents,
    seed_curvature,
    seed_conic,
    seed_decenter,
    seed_despace,
    seed_tilt,
    seed_index,
)
from prysm.x.raytracing.analysis import wavefront
from tests.x.raytracing.differential_helpers import (
    BASE, NG, WVL, make_system, ray_bundle,
)


def fd_opd(over_plus, over_minus, P, S, h, output='length'):
    opd_p, _, _ = wavefront(
        make_system(**over_plus), P, S, WVL, output=output)
    opd_m, _, _ = wavefront(
        make_system(**over_minus), P, S, WVL, output=output)
    return (opd_p - opd_m) / (2 * h)


# FD of the composed trace->OPD pipeline (with a moving reference sphere) is
# noisier than FD of the raw kernel; the analytic tangent is the more accurate
# of the two, so atol reflects FD truncation, not derivation error.
def check(seed, over_plus, over_minus, h, rtol=1e-5, atol=1e-7,
          output='length'):
    P, S = ray_bundle()
    opd, x, y, dW = wavefront_with_tangents(make_system(), P, S, WVL, [seed],
                                            output=output)
    dW_fd = fd_opd(over_plus, over_minus, P, S, h, output=output)
    np.testing.assert_allclose(dW[:, 0], dW_fd, rtol=rtol, atol=atol)


# curvature0, conic1, despace1, decenter1_y and index0 are validated in bulk by
# test_all_seeds_one_trace below; these cases cover the remaining seed types.
_H6, _H5 = 1e-6, 1e-5
_SEED_CASES = [
    ('curvature1', seed_curvature(1), dict(c1=BASE['c1'] + _H6), dict(c1=BASE['c1'] - _H6), _H6, {}),
    ('conic0', seed_conic(0), dict(k0=BASE['k0'] + _H5), dict(k0=BASE['k0'] - _H5), _H5, {}),
    ('thickness_fanout', seed_despace([(1, +1), (2, +1)]),
     dict(z1=BASE['z1'] + _H6, zimg=BASE['zimg'] + _H6),
     dict(z1=BASE['z1'] - _H6, zimg=BASE['zimg'] - _H6), _H6, {}),
    ('decenter1_x', seed_decenter(1, 'x'), dict(x1=_H6), dict(x1=-_H6), _H6, {}),
    ('tilt1_x', seed_tilt(1, 'x'), dict(tiltx1=_H6), dict(tiltx1=-_H6), _H6,
     dict(rtol=1e-4, atol=1e-7)),
]


@pytest.mark.parametrize('seed, over_plus, over_minus, h, tols',
                         [c[1:] for c in _SEED_CASES],
                         ids=[c[0] for c in _SEED_CASES])
def test_seed_dW_matches_fd(seed, over_plus, over_minus, h, tols):
    check(seed, over_plus, over_minus, h, **tols)


def test_chief_opd_tangent_is_zero():
    """The chief ray's OPD is identically 0, so dW at the chief must vanish."""
    P, S = ray_bundle()
    center = np.mean(P[:, :2], axis=0)
    chief = int(np.argmin(np.sum((P[:, :2] - center) ** 2, axis=1)))
    _, _, _, dW = wavefront_with_tangents(
        make_system(), P, S, WVL,
        [seed_curvature(0), seed_conic(1), seed_decenter(1, 'y')])
    np.testing.assert_allclose(dW[chief], 0.0, atol=1e-12)


def test_waves_output_scales():
    h = 1e-6
    check(seed_curvature(0), dict(c0=BASE['c0'] + h), dict(c0=BASE['c0'] - h),
          h, output='waves', rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize('output', ['length', 'waves'])
@pytest.mark.parametrize('field', [None, Field(2.0, 3.0)])
def test_nominal_opd_matches_analysis_wavefront(output, field):
    """The nominal opd of wavefront_with_tangents must equal analysis.wavefront.

    The closing kernel is shared, but the differential path keeps its own
    exit-pupil route, ramp, and scaling; this pins them so a future change to
    analysis.wavefront can't silently desync the differential model's W0.
    """
    P, S = ray_bundle()
    sys = make_system()
    opd_ref, x_ref, y_ref = wavefront(
        sys, P, S, WVL, field=field, output=output)
    opd, x, y, _ = wavefront_with_tangents(sys, P, S, WVL, [seed_curvature(0)],
                                           field=field, output=output)
    np.testing.assert_allclose(opd, opd_ref, rtol=0, atol=1e-12)
    np.testing.assert_allclose(x, x_ref, rtol=0, atol=1e-12)
    np.testing.assert_allclose(y, y_ref, rtol=0, atol=1e-12)


def test_all_seeds_one_trace():
    P, S = ray_bundle()
    seeds = [seed_curvature(0), seed_conic(1), seed_despace([(1, +1)]),
             seed_decenter(1, 'y'), seed_index(0)]
    _, _, _, dW = wavefront_with_tangents(make_system(), P, S, WVL, seeds)
    assert dW.shape[1] == 5
    h = 1e-6
    overs = [
        (dict(c0=BASE['c0'] + h), dict(c0=BASE['c0'] - h)),
        (dict(k1=BASE['k1'] + h), dict(k1=BASE['k1'] - h)),
        (dict(z1=BASE['z1'] + h), dict(z1=BASE['z1'] - h)),
        (dict(y1=h), dict(y1=-h)),
        (dict(ng=NG + h), dict(ng=NG - h)),
    ]
    for p, (op, om) in enumerate(overs):
        dW_fd = fd_opd(op, om, P, S, h)
        np.testing.assert_allclose(dW[:, p], dW_fd, rtol=1e-5, atol=1e-7)
