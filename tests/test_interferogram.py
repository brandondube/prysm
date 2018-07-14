"""Interferogram tests."""
from pathlib import Path

import numpy as np

from prysm.interferogram import Interferogram


data_root = Path(__file__).parent / 'io_files'


def e2e_test_interferogram():
    # only test no errors, not function
    i = Interferogram.from_zygo_dat(data_root / 'valid_zygo_dat_file.dat')
    i = i.remove_piston_tiptilt()
    assert i
