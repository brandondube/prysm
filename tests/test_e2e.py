"""end to end tests."""

from prysm import sample_files, Pupil, Interferogram


def test_pupil_from_interferogram_does_not_error():
    i = Interferogram.from_zygo_dat(sample_files('dat'))
    pu = Pupil.from_interferogram(i)
    assert pu
