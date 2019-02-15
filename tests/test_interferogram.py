"""Interferogram tests."""

import numpy as np

from prysm import sample_files
from prysm.interferogram import Interferogram


def e2e_test_interferogram():
    # only test no errors, not function
    i = Interferogram.from_zygo_dat(sample_files('dat'))
    i = i.remove_piston_tiptilt()
    assert i
