"""tests for basic richdata functions."""

from prysm import sample_files, Interferogram


import pytest

@pytest.mark.parametrize('invert_x', [True, False])
def test_psd_slice_plot_does_not_blow_up(invert_x):
    i = Interferogram.from_zygo_dat(sample_files('dat'))
    fig, ax = i.psd().slices().plot('x', invert_x=invert_x)
    assert fig
    assert ax
