"""Tests for object (target) synthesis routines."""
import pytest

import numpy as np

from prysm import coordinates, fttools, objects


@pytest.fixture
def xy():
    return coordinates.make_xy_grid(33, diameter=2)


@pytest.fixture
def rt(xy):
    return coordinates.cart_to_polar(*xy)


def test_slit_widths_select_expected_axes(xy):
    x, y = xy

    vertical = objects.slit(x, y, width_x=0.25, width_y=None)
    horizontal = objects.slit(x, y, width_x=None, width_y=0.25)
    crossed = objects.slit(x, y, width_x=0.25, width_y=0.25)

    assert vertical[:, 16].all()
    assert not vertical[16, :].all()
    assert horizontal[16, :].all()
    np.testing.assert_array_equal(crossed, vertical | horizontal)


def test_slit_ft_matches_rasterization():
    N = 256
    dx = 1 / 64
    x, y = coordinates.make_xy_grid(N, dx=dx)
    fx = fttools.forward_ft_unit(dx, N)
    fy = fttools.forward_ft_unit(dx, N)

    for wx, wy in ((0.5, None), (None, 0.5), (0.5, 1.0)):
        mask = objects.slit(x, y, wx, wy)
        # effective rasterized widths; edge rows/cols avoid the crossing band
        wx_eff = mask[0, :].sum() * dx if wx is not None else None
        wy_eff = mask[:, 0].sum() * dx if wy is not None else None

        F = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(mask)))
        F = (F / F[N//2, N//2]).real
        an = objects.slit_ft(wx_eff, wy_eff, fx, fy)
        # analytic sinc vs discrete Dirichlet kernel; compare the central half band
        sl = slice(N//2 - N//4, N//2 + N//4)
        np.testing.assert_allclose(an[sl, sl], F[sl, sl], atol=5e-3)


def test_slit_ft_zero_and_none_equivalent():
    fx = fttools.forward_ft_unit(1/32, 32)
    fy = fttools.forward_ft_unit(1/32, 32)

    np.testing.assert_array_equal(objects.slit_ft(2, 0, fx, fy),
                                  objects.slit_ft(2, None, fx, fy))


def test_pinhole_masks_by_radius(rt):
    r, _ = rt

    mask = objects.pinhole(0.25, r)

    assert mask[16, 16]
    assert not mask[0, 0]


def test_pinhole_ft_has_unit_dc(rt):
    r, _ = rt

    ft = objects.pinhole_ft(1, r)

    assert ft[16, 16] == pytest.approx(0.5)


def test_siemensstar_background_and_invalid_background(rt):
    r, t = rt

    white = objects.siemensstar(r, t, 16, oradius=0.4, background='white')
    black = objects.siemensstar(r, t, 16, oradius=0.4, background='black')

    assert white[0, 0] == pytest.approx(0.95)
    assert black[0, 0] == pytest.approx(0.05)
    with pytest.raises(ValueError, match='invalid background'):
        objects.siemensstar(r, t, 16, background='gray')


def test_tiltedsquare_background_sets_inside_and_outside(xy):
    x, y = xy

    white_bg = objects.tiltedsquare(x, y, radius=0.25, contrast=0.8, background='white')
    black_bg = objects.tiltedsquare(x, y, radius=0.25, contrast=0.8, background='black')

    assert white_bg[16, 16] == pytest.approx(0.1)
    assert white_bg[0, 0] == pytest.approx(0.9)
    assert black_bg[16, 16] == pytest.approx(0.9)
    assert black_bg[0, 0] == pytest.approx(0.1)


def test_slantededge_crossed_changes_quadrants(xy):
    x, y = xy

    single = objects.slantededge(x, y, angle=0, crossed=False)
    crossed = objects.slantededge(x, y, angle=0, crossed=True)

    assert single[16, 24] < single[16, 8]
    assert crossed[8, 24] < crossed[8, 8]
    assert crossed[24, 8] < crossed[8, 8]
