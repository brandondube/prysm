"""Tests for the various objects prysm knows how to synthesize."""
import pytest

from prysm import objects


@pytest.mark.parametrize('orientation', ['h', 'v', 'crossed', 'horizontal', 'vertical'])
def test_slit_renders_correctly_for_all_orientations(orientation):
    slit = objects.Slit(1, orientation, 0.05, 19)
    assert (slit.data == 1).all()


@pytest.mark.parametrize('orientation', ['h', 'v', 'crossed'])
def test_slit_analytic_ft_correct_at_origin(orientation):
    s = objects.Slit(1, orientation=orientation)
    aft = s.analytic_ft(0, 0)[0, 0]
    assert aft == 1 or aft == 2


def test_pinhole_renders_properly_undersized_support():
    p = objects.Pinhole(1, 0.01, 10)
    assert (p.data == 1).all()


def test_pinhole_analytic_ft_correct_at_origin():
    p = objects.Pinhole(1, 0, 0)
    assert p.analytic_ft(0, 0)[0, 0] == 0.5


@pytest.mark.parametrize('sinusoid, background', [[True, 'w'], [True, 'b'], [False, 'w'], [False, 'b']])
def test_siemens_star_renders(sinusoid, background):
    ss = objects.SiemensStar(32, sinusoidal=sinusoid,
                             background=background,
                             sample_spacing=1,
                             samples=32)
    assert ss


def test_tiltedsquare_renders():
    ts = objects.TiltedSquare(4)
    assert ts


def test_slantededge_renders():
    se = objects.SlantedEdge()
    assert se


def test_grating_renders():
    g = objects.Grating(1)
    assert g


def test_grating_array_renders():
    ga = objects.GratingArray([1, 2], [1, 2])
    assert ga
